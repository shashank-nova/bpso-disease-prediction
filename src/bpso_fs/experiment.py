"""Experiment orchestration: repeated nested cross-validation for every selector.

Protocol per dataset
--------------------
* Outer loop: stratified ``outer_folds``-fold CV, repeated for ``n_seeds`` seeds
  (seed s uses ``StratifiedKFold(shuffle=True, random_state=s)``). Every method sees the
  *same* outer splits, so results are paired by (seed, fold).
* Inside each outer training fold only:
  1. the selection-view preprocessor is fitted (imputation / encoding / scaling);
  2. the selector is fitted (BPSO/GA use an inner stratified CV in the fitness function);
  3. each final classifier is fitted on the selected original columns.
* The outer test fold is used once, for scoring.

Phase 1 runs the wrapper methods (BPSO, GA, alpha sweep). The median BPSO subset size
then fixes ``k`` for the fixed-k baselines (MI, ANOVA, RFE, random) in phase 2.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm.auto import tqdm

from .data.loaders import DatasetBundle
from .data.preprocess import build_model_pipeline, make_selection_view, mask_to_names
from .evaluation import compute_metrics
from .models import PARAM_GRIDS, get_classifier, param_grid_for, resolve_classifiers
from .selectors import (
    AllFeatures,
    BinaryGA,
    BinaryPSO,
    FeatureSelector,
    KBestSelector,
    L1Selector,
    RandomSubset,
    RFESelector,
)

logger = logging.getLogger(__name__)

WRAPPER_METHODS = ("bpso", "ga")
K_METHODS = ("mi_k", "anova_k", "rfe_k", "random_k")
ALL_METHODS = (
    "all",
    "bpso",
    "ga",
    "mi_k",
    "mi_cv",
    "anova_k",
    "anova_cv",
    "rfe_k",
    "l1",
    "random_k",
)

METHOD_LABELS = {
    "all": "All features",
    "bpso": "BPSO (proposed)",
    "ga": "Binary GA",
    "mi_k": "MI (k = BPSO)",
    "mi_cv": "MI (k by CV)",
    "anova_k": "ANOVA-F (k = BPSO)",
    "anova_cv": "ANOVA-F (k by CV)",
    "rfe_k": "RFE (k = BPSO)",
    "l1": "L1-LogReg",
    "random_k": "Random (k = BPSO)",
}


# --------------------------------------------------------------------------- config


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


QUICK_OVERRIDES = {
    "experiment": {"n_seeds": 2, "outer_folds": 3},
    "bpso": {"n_particles": 10, "n_iterations": 10},
    "fitness": {"inner_folds": 3},
    "alpha_sweep": {"n_seeds": 1},
}


def load_config(path: str | Path, quick: bool = False, overrides: dict | None = None) -> dict:
    """Load a YAML config, optionally applying the ``--quick`` smoke-test overrides."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if quick:
        cfg = deep_update(cfg, QUICK_OVERRIDES)
        cfg["experiment"]["quick"] = True
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).parent
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def library_versions() -> dict[str, str]:
    import joblib
    import matplotlib
    import scipy
    import sklearn

    v = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
    }
    try:
        import xgboost

        v["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return v


# --------------------------------------------------------------------------- tasks


@dataclass(frozen=True)
class Task:
    method: str  # base method name, e.g. 'bpso'
    run_name: str  # name used for storage, e.g. 'bpso' or 'bpso_alpha0.80'
    seed: int
    fold: int
    alpha: float
    k: int | None
    classifiers: tuple[str, ...]


def split_seed(seed: int, fold: int) -> int:
    """Deterministic per-(seed, fold) RNG seed for selectors and inner CV."""
    return int(seed) * 1000 + int(fold)


def outer_splits(
    y: np.ndarray, seeds: list[int], n_folds: int
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    out = {}
    for s in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            out[(s, f)] = (tr, te)
    return out


def make_selector(task: Task, cfg: dict, n_features: int) -> FeatureSelector:
    """Instantiate the selector for a task."""
    rs = split_seed(task.seed, task.fold)
    fcfg = cfg["fitness"]
    fitness_kwargs = {
        "alpha": task.alpha,
        "estimator": fcfg.get("estimator", "logreg"),
        "cv": fcfg.get("inner_folds", 5),
        "metric": fcfg.get("metric", "accuracy"),
        "random_state": rs,
    }
    scoring = fcfg.get("metric", "accuracy")
    m = task.method
    if m == "all":
        return AllFeatures()
    if m == "bpso":
        b = cfg["bpso"]
        return BinaryPSO(
            n_particles=b["n_particles"],
            n_iterations=b["n_iterations"],
            w=b.get("w", 0.7298),
            inertia=b.get("inertia", "linear"),
            w_max=b.get("w_max", 0.9),
            w_min=b.get("w_min", 0.4),
            c1=b.get("c1", 2.0),
            c2=b.get("c2", 2.0),
            v_max=b.get("v_max", 6.0),
            transfer_function=b.get("transfer_function", "sigmoid"),
            init_prob=b.get("init_prob", 0.5),
            empty_subset=b.get("empty_subset", "repair"),
            early_stopping_patience=b.get("early_stopping_patience"),
            fitness_kwargs=fitness_kwargs,
            n_jobs=b.get("n_jobs", 1),
            random_state=rs,
        )
    if m == "ga":
        b, g = cfg["bpso"], cfg.get("ga", {})
        return BinaryGA(
            pop_size=b["n_particles"],
            n_generations=b["n_iterations"],
            crossover_rate=g.get("crossover_rate", 0.9),
            mutation_rate=g.get("mutation_rate"),
            tournament_size=g.get("tournament_size", 3),
            n_elite=g.get("n_elite", 1),
            init_prob=b.get("init_prob", 0.5),
            fitness_kwargs=fitness_kwargs,
            random_state=rs,
        )
    inner = fcfg.get("inner_folds", 5)
    if m == "mi_k":
        return KBestSelector("mutual_info", k=task.k, random_state=rs)
    if m == "mi_cv":
        return KBestSelector("mutual_info", k="cv", cv=inner, scoring=scoring, random_state=rs)
    if m == "anova_k":
        return KBestSelector("anova", k=task.k, random_state=rs)
    if m == "anova_cv":
        return KBestSelector("anova", k="cv", cv=inner, scoring=scoring, random_state=rs)
    if m == "rfe_k":
        return RFESelector(k=task.k, random_state=rs)
    if m == "l1":
        return L1Selector(cv=inner, scoring=scoring, random_state=rs)
    if m == "random_k":
        return RandomSubset(k=task.k, random_state=rs)
    raise ValueError(f"Unknown method '{m}'")


def _score_vector(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_task(
    bundle: DatasetBundle, task: Task, train_idx: np.ndarray, test_idx: np.ndarray, cfg: dict
) -> dict:
    """Run one (method, seed, fold): fit view + selector + final classifiers on train, score on test."""
    X_tr = bundle.X.iloc[train_idx]
    X_te = bundle.X.iloc[test_idx]
    y_tr, y_te = bundle.y[train_idx], bundle.y[test_idx]

    t0 = time.perf_counter()
    view = make_selection_view(bundle.numeric_cols, bundle.categorical_cols).fit(X_tr)
    V_tr = view.transform(X_tr)
    selector = make_selector(task, cfg, bundle.n_features)
    selector.fit(V_tr, y_tr)
    mask = selector.get_support()
    selected = mask_to_names(mask, bundle.feature_names)
    selection_time = time.perf_counter() - t0

    rec: dict[str, Any] = {
        "dataset": bundle.name,
        "method": task.run_name,
        "base_method": task.method,
        "seed": task.seed,
        "fold": task.fold,
        "alpha": task.alpha if task.method in WRAPPER_METHODS else None,
        "k_param": task.k,
        "n_features_total": bundle.n_features,
        "n_selected": int(mask.sum()),
        "mask": mask.astype(int).tolist(),
        "selected_features": selected,
        "selection_time": selection_time,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_idx": test_idx.tolist(),
        "y_test": y_te.astype(int).tolist(),
        "classifiers": {},
    }
    for attr in (
        "best_fitness_",
        "best_score_",
        "n_evaluations_",
        "n_fitness_calls_",
        "k_",
        "C_",
        "n_iterations_run_",
    ):
        if hasattr(selector, attr):
            val = getattr(selector, attr)
            rec[attr.rstrip("_")] = None if val is None else float(val)
    if hasattr(selector, "history_"):
        rec["history"] = selector.history_

    tune = cfg.get("tune_classifiers", False)
    for clf_name in task.classifiers:
        t1 = time.perf_counter()
        clf = get_classifier(clf_name, random_state=task.seed)
        pipe = build_model_pipeline(selected, bundle.numeric_cols, bundle.categorical_cols, clf)
        if tune and PARAM_GRIDS.get(clf_name):
            grid = param_grid_for(clf_name, clf)
            pipe = GridSearchCV(
                pipe,
                grid,
                cv=StratifiedKFold(cfg["fitness"].get("inner_folds", 5), shuffle=True,
                                   random_state=split_seed(task.seed, task.fold)),
                scoring=cfg.get("reporting", {}).get("tuning_scoring", "roc_auc"),
            )  # fmt: skip
        model = clone(pipe).fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_score = _score_vector(model, X_te)
        fit_time = time.perf_counter() - t1
        m = compute_metrics(y_te, y_pred, y_score)
        m["fit_time"] = fit_time
        m["total_time"] = selection_time + fit_time
        m["y_score"] = [float(v) for v in y_score]
        m["y_pred"] = [int(v) for v in y_pred]
        if tune and hasattr(model, "best_params_"):
            m["best_params"] = model.best_params_
        rec["classifiers"][clf_name] = m
    return rec


# --------------------------------------------------------------------------- driver


class ExperimentRunner:
    """Runs the full protocol for one dataset and writes results to ``results/<dataset>/``."""

    def __init__(self, cfg: dict, results_dir: str | Path | None = None, n_jobs: int | None = None):
        self.cfg = cfg
        exp = cfg["experiment"]
        self.results_root = Path(results_dir or exp.get("results_dir", "results"))
        self.n_jobs = n_jobs if n_jobs is not None else exp.get("n_jobs", -1)
        self.seeds = [exp.get("seed_offset", 0) + i for i in range(exp["n_seeds"])]
        self.n_folds = exp["outer_folds"]
        self.classifiers = resolve_classifiers(cfg["classifiers"], cfg.get("use_xgboost", True))
        self.methods = [m for m in cfg["methods"] if m in ALL_METHODS]
        unknown = set(cfg["methods"]) - set(ALL_METHODS)
        if unknown:
            raise ValueError(f"Unknown methods in config: {unknown}")

    # -------------------------------------------------------------- persistence
    def _task_hash(self, task: Task) -> str:
        relevant = {
            "method": task.method,
            "alpha": task.alpha,
            "k": task.k,
            "classifiers": list(task.classifiers),
            "outer_folds": self.n_folds,
            "fitness": self.cfg["fitness"],
            "bpso": self.cfg["bpso"] if task.method in WRAPPER_METHODS else None,
            "ga": self.cfg.get("ga") if task.method == "ga" else None,
            "tune": self.cfg.get("tune_classifiers", False),
        }
        return _hash(relevant)

    def _run_path(self, ds_dir: Path, task: Task) -> Path:
        return ds_dir / "runs" / task.run_name / f"s{task.seed}_f{task.fold}.json"

    def _load_cached(self, ds_dir: Path, task: Task) -> dict | None:
        p = self._run_path(ds_dir, task)
        if not p.exists():
            return None
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        return rec if rec.get("task_hash") == self._task_hash(task) else None

    def _execute(
        self, bundle: DatasetBundle, tasks: list[Task], splits: dict, ds_dir: Path, desc: str
    ) -> list[dict]:
        results, todo = [], []
        for t in tasks:
            cached = self._load_cached(ds_dir, t)
            if cached:
                results.append(cached)
            else:
                todo.append(t)
        if results:
            logger.info(
                "%s: %d runs loaded from cache (resume), %d to run", desc, len(results), len(todo)
            )
        if not todo:
            return results

        def job(t: Task) -> dict:
            tr, te = splits[(t.seed, t.fold)]
            rec = run_task(bundle, t, tr, te, self.cfg)
            rec["task_hash"] = self._task_hash(t)
            p = self._run_path(ds_dir, t)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(p)
            return rec

        gen = Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            delayed(job)(t) for t in todo
        )
        for rec in tqdm(gen, total=len(todo), desc=desc, leave=False):
            results.append(rec)
        return results

    # -------------------------------------------------------------- main
    def run_dataset(self, bundle: DatasetBundle) -> dict:
        t_start = time.time()
        ds_dir = self.results_root / bundle.name
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "data_summary.json").write_text(json.dumps(bundle.summary(), indent=2))
        splits = outer_splits(bundle.y, self.seeds, self.n_folds)
        alpha = float(self.cfg["fitness"]["alpha"])
        clfs = tuple(self.classifiers)
        logger.info("[%s] n=%d d=%d seeds=%d folds=%d methods=%s classifiers=%s", bundle.name,
                    len(bundle.y), bundle.n_features, len(self.seeds), self.n_folds, self.methods, clfs)  # fmt: skip

        # ---- phase 1: wrapper methods
        phase1 = [Task(m, m, s, f, alpha, None, clfs) for m in self.methods if m in WRAPPER_METHODS
                  for (s, f) in splits]  # fmt: skip
        records = self._execute(bundle, phase1, splits, ds_dir, f"{bundle.name}: BPSO/GA")

        # ---- k for fixed-k baselines = median BPSO subset size
        bpso_sizes = [r["n_selected"] for r in records if r["method"] == "bpso"]
        if bpso_sizes:
            k_med = int(np.round(np.median(bpso_sizes)))
        else:
            k_med = max(1, bundle.n_features // 2)
            logger.warning("BPSO not run; fixed-k baselines use k = d/2 = %d", k_med)

        # ---- phase 2: other baselines
        phase2 = [Task(m, m, s, f, alpha, k_med if m in K_METHODS else None, clfs)
                  for m in self.methods if m not in WRAPPER_METHODS for (s, f) in splits]  # fmt: skip
        records += self._execute(bundle, phase2, splits, ds_dir, f"{bundle.name}: baselines")

        # ---- alpha sensitivity sweep (BPSO only)
        sweep = self.cfg.get("alpha_sweep", {})
        sweep_records: list[dict] = []
        if sweep.get("enabled", False):
            sw_seeds = self.seeds[: sweep.get("n_seeds", len(self.seeds))]
            sw_clfs = tuple(
                resolve_classifiers(
                    sweep.get("classifiers", list(clfs)), self.cfg.get("use_xgboost", True)
                )
            )
            tasks = []
            for a in sweep["alphas"]:
                for s, f in splits:
                    if s in sw_seeds:
                        tasks.append(
                            Task("bpso", f"bpso_alpha{float(a):.2f}", s, f, float(a), None, sw_clfs)
                        )
            sweep_records = self._execute(
                bundle, tasks, splits, ds_dir, f"{bundle.name}: alpha sweep"
            )

        self._write_outputs(bundle, ds_dir, records, sweep_records, k_med, time.time() - t_start)
        return {
            "dataset": bundle.name,
            "k_median": k_med,
            "n_runs": len(records) + len(sweep_records),
        }

    # -------------------------------------------------------------- outputs
    @staticmethod
    def _flatten(
        records: list[dict],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        rows, sel_rows, hist_rows, pred_rows = [], [], [], []
        for r in records:
            base = {k: r.get(k) for k in ("dataset", "method", "base_method", "seed", "fold", "alpha",
                                          "k_param", "n_features_total", "n_selected", "selection_time",
                                          "best_fitness", "best_score", "n_evaluations", "n_fitness_calls")}  # fmt: skip
            sel_rows.append({**base, "selected_features": ";".join(r["selected_features"]),
                             "mask": "".join(str(int(v)) for v in r["mask"])})  # fmt: skip
            for clf, m in r["classifiers"].items():
                row = {**base, "classifier": clf}
                row.update(
                    {k: v for k, v in m.items() if k not in ("y_score", "y_pred", "best_params")}
                )
                row["selected_features"] = ";".join(r["selected_features"])
                rows.append(row)
                for idx, yt, ys, yp in zip(
                    r["test_idx"], r["y_test"], m["y_score"], m["y_pred"], strict=True
                ):
                    pred_rows.append((r["method"], clf, r["seed"], r["fold"], idx, yt, ys, yp))
            if "history" in r:
                h = r["history"]
                for it in range(len(h["best_fitness"])):
                    hist_rows.append({"method": r["method"], "seed": r["seed"], "fold": r["fold"],
                                      "iteration": it, **{k: h[k][it] for k in h}})  # fmt: skip
        preds = pd.DataFrame(
            pred_rows,
            columns=[
                "method",
                "classifier",
                "seed",
                "fold",
                "index",
                "y_true",
                "y_score",
                "y_pred",
            ],
        )
        return pd.DataFrame(rows), pd.DataFrame(sel_rows), pd.DataFrame(hist_rows), preds

    def _write_outputs(self, bundle, ds_dir, records, sweep_records, k_med, runtime) -> None:
        raw, sel, hist, preds = self._flatten(records)
        sort = ["method", "classifier", "seed", "fold"]
        raw.sort_values(sort).to_csv(ds_dir / "raw_results.csv", index=False)
        sel.sort_values(["method", "seed", "fold"]).to_csv(ds_dir / "selections.csv", index=False)
        if not hist.empty:
            hist.sort_values(["method", "seed", "fold", "iteration"]).to_csv(
                ds_dir / "history.csv", index=False
            )
        preds.to_csv(ds_dir / "predictions.csv.gz", index=False)
        if sweep_records:
            sraw, ssel, _, _ = self._flatten(sweep_records)
            sraw.sort_values(sort).to_csv(ds_dir / "alpha_sweep_results.csv", index=False)
            ssel.to_csv(ds_dir / "alpha_sweep_selections.csv", index=False)
        meta = {
            "dataset": bundle.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "library_versions": library_versions(),
            "config": self.cfg,
            "seeds": self.seeds,
            "outer_folds": self.n_folds,
            "classifiers": self.classifiers,
            "methods": self.methods,
            "k_median_bpso": k_med,
            "wall_time_seconds_this_invocation": runtime,
            "data_source": bundle.source,
        }
        (ds_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    """Log to console and (optionally) a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)  # fmt: skip

#!/usr/bin/env python
"""Run the BPSO feature-selection experiments.

Examples:
    python scripts/run_experiments.py --config configs/default.yaml --dataset all --seeds 30 --n-jobs -1
    python scripts/run_experiments.py --config configs/default.yaml --dataset breast_cancer --quick
    python scripts/run_experiments.py --config configs/default.yaml --csv my.csv --target Outcome
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpso_fs.data.loaders import (  # noqa: E402
    DatasetUnavailableError,
    load_csv_dataset,
    load_dataset,
)
from bpso_fs.experiment import ExperimentRunner, load_config, setup_logging  # noqa: E402

logger = logging.getLogger("run_experiments")


def dataset_configs(folder: Path) -> dict[str, dict]:
    """Map dataset name -> its YAML entry in configs/datasets/."""
    import yaml

    out = {}
    for f in sorted(folder.glob("*.yaml")):
        d = yaml.safe_load(f.read_text()) or {}
        if "name" in d:
            out[d["name"]] = d
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--dataset", default="all", help="dataset name, comma list, or 'all' (config list)"
    )
    p.add_argument("--seeds", type=int, default=None, help="override experiment.n_seeds")
    p.add_argument("--folds", type=int, default=None, help="override experiment.outer_folds")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: 2 seeds, 3 folds, 10 particles, 10 iterations",
    )
    p.add_argument("--results-dir", default=None)
    p.add_argument("--csv", default=None, help="path to a custom CSV dataset")
    p.add_argument("--target", default=None, help="target column for --csv")
    p.add_argument("--positive-label", default=None, help="value of --target meaning 'disease'")
    p.add_argument("--name", default=None, help="dataset name for --csv")
    p.add_argument("--no-alpha-sweep", action="store_true")
    p.add_argument(
        "--require-all-datasets",
        action="store_true",
        help="fail if any requested dataset could not be loaded",
    )
    args = p.parse_args()

    overrides: dict = {"experiment": {}}
    if args.seeds is not None:
        overrides["experiment"]["n_seeds"] = args.seeds
    if args.folds is not None:
        overrides["experiment"]["outer_folds"] = args.folds
    if args.no_alpha_sweep:
        overrides["alpha_sweep"] = {"enabled": False}
    cfg = load_config(args.config, quick=args.quick, overrides=overrides)
    default_dir = "results_quick" if args.quick else cfg["experiment"].get("results_dir", "results")
    results_dir = Path(args.results_dir or default_dir)  # --quick never overwrites real results
    setup_logging(results_dir / "run.log")
    runner = ExperimentRunner(cfg, results_dir=results_dir, n_jobs=args.n_jobs)

    if args.csv:
        if not args.target:
            p.error("--csv requires --target")
        bundles = [load_csv_dataset(args.csv, args.target, args.positive_label, name=args.name)]
    else:
        names = cfg["experiment"]["datasets"] if args.dataset == "all" else args.dataset.split(",")
        ds_cfgs = dataset_configs(Path(args.config).resolve().parent / "datasets")
        bundles = []
        unavailable = []
        for n in (x.strip() for x in names):
            try:
                dc = ds_cfgs.get(n, {})
                if dc.get("type") == "csv":
                    bundles.append(
                        load_csv_dataset(dc["path"], dc["target"], dc.get("positive_label"), name=n,
                                         categorical=dc.get("categorical"), drop=dc.get("drop"))
                    )  # fmt: skip
                else:
                    bundles.append(load_dataset(n))
            except (DatasetUnavailableError, FileNotFoundError, KeyError) as exc:
                logger.error("Skipping dataset '%s':\n%s", n, exc)
                unavailable.append(n)
        if unavailable and args.require_all_datasets:
            logger.error("Required datasets unavailable: %s", ", ".join(unavailable))
            return 1
    if not bundles:
        logger.error("No dataset could be loaded.")
        return 1

    failed = []
    for b in bundles:
        try:
            info = runner.run_dataset(b)
            logger.info("Finished %s: %s", b.name, info)
        except Exception:  # noqa: BLE001 - continue with remaining datasets
            logger.exception("Experiment failed for %s", b.name)
            failed.append(b.name)
    if failed:
        logger.error("Failed datasets: %s", failed)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

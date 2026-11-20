"""Train and load a reusable BPSO-selected binary disease classifier."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .data.preprocess import build_model_pipeline, make_selection_view, mask_to_names
from .fitness import SubsetFitness
from .models import get_classifier
from .selectors.bpso import BinaryPSO


def train_artifact(bundle, config: dict, output: str | Path, classifier: str = "logreg") -> dict:
    """Fit feature selection and a final model on the supplied training dataset."""
    exp = config.get("experiment", {})
    fit_cfg = config.get("fitness", {})
    bpso_cfg = config.get("bpso", {})
    view = make_selection_view(bundle.numeric_cols, bundle.categorical_cols)
    x_view = view.fit_transform(bundle.X)
    seed = int(exp.get("seed_offset", 0))
    fitness = SubsetFitness(
        x_view,
        bundle.y,
        alpha=float(fit_cfg.get("alpha", 0.99)),
        estimator=fit_cfg.get("estimator", "logreg"),
        cv=int(fit_cfg.get("inner_folds", 5)),
        metric=fit_cfg.get("metric", "accuracy"),
        random_state=seed,
    )
    selector = BinaryPSO(
        n_particles=int(bpso_cfg.get("n_particles", 20)),
        n_iterations=int(bpso_cfg.get("n_iterations", 50)),
        w=float(bpso_cfg.get("w", 0.7298)),
        inertia=bpso_cfg.get("inertia", "linear"),
        w_max=float(bpso_cfg.get("w_max", 0.9)),
        w_min=float(bpso_cfg.get("w_min", 0.4)),
        c1=float(bpso_cfg.get("c1", 2.0)),
        c2=float(bpso_cfg.get("c2", 2.0)),
        v_max=float(bpso_cfg.get("v_max", 6.0)),
        transfer_function=bpso_cfg.get("transfer_function", "sigmoid"),
        init_prob=float(bpso_cfg.get("init_prob", 0.5)),
        empty_subset=bpso_cfg.get("empty_subset", "repair"),
        fitness=fitness,
        random_state=seed,
    ).fit(x_view, bundle.y)
    selected = mask_to_names(selector.get_support(), bundle.feature_names)
    pipeline = build_model_pipeline(
        selected,
        bundle.numeric_cols,
        bundle.categorical_cols,
        get_classifier(classifier, random_state=seed),
    )
    pipeline.fit(bundle.X, bundle.y)
    metadata = {
        "dataset": bundle.name,
        "classifier": classifier,
        "features": bundle.feature_names,
        "selected_features": selected,
        "positive_label": 1,
        "negative_label": 0,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(bundle.y)),
        "selection_score_cv": float(selector.best_score_),
        "selection_fitness": float(selector.best_fitness_),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, target)
    return metadata


def load_artifact(path: str | Path) -> dict:
    """Load and validate a model artifact."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"pipeline", "metadata"} <= artifact.keys():
        raise ValueError(f"Invalid model artifact: {path}")
    if not artifact["metadata"].get("features"):
        raise ValueError(f"Model artifact has no feature schema: {path}")
    return artifact


def prediction_frame(records: list[dict], features: list[str]) -> pd.DataFrame:
    """Build a schema-ordered frame and reject missing or empty input."""
    if not records:
        raise ValueError("At least one record is required")
    frame = pd.DataFrame.from_records(records)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
    return frame[features]

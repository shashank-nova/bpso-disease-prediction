"""Final classifiers trained on each selected feature subset."""

from __future__ import annotations

import importlib.util

import sklearn
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

CLASSIFIER_LABELS = {
    "logreg": "Logistic Regression",
    "svm": "SVM (RBF)",
    "rf": "Random Forest",
    "knn": "k-NN",
    "nb": "Gaussian NB",
    "xgb": "XGBoost",
}

#: Small hyper-parameter grids used when ``tune_classifiers: true`` (inner CV).
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "svm": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
    "rf": {"n_estimators": [200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
    "knn": {"n_neighbors": [3, 5, 7, 11]},
    "nb": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    "xgb": {"max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1]},
}


_SVC_PROBABILITY_DEPRECATED = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 9)


def param_grid_for(name: str, estimator: BaseEstimator, prefix: str = "clf__") -> dict[str, list]:
    """Grid keys for a classifier inside a Pipeline step named ``clf``."""
    inner = "estimator__" if isinstance(estimator, CalibratedClassifierCV) else ""
    return {f"{prefix}{inner}{k}": v for k, v in PARAM_GRIDS.get(name, {}).items()}


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def get_classifier(name: str, random_state: int | None = None) -> BaseEstimator:
    """Classifier with documented default hyper-parameters."""
    if name == "logreg":
        return LogisticRegression(max_iter=5000, random_state=random_state)
    if name == "svm":
        if _SVC_PROBABILITY_DEPRECATED:
            # scikit-learn >= 1.9 deprecates SVC(probability=True); this is its documented
            # replacement (Platt scaling with internal 5-fold CV, single final SVC).
            return CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state),
                method="sigmoid",
                cv=5,
                ensemble=False,
            )
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=random_state)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "nb":
        return GaussianNB()
    if name == "xgb":
        if not xgboost_available():
            raise ImportError("xgboost is not installed")
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
            verbosity=0,
        )
    raise ValueError(f"Unknown classifier '{name}'")


def resolve_classifiers(names: list[str], use_xgboost: bool = True) -> list[str]:
    """Drop 'xgb' if it is disabled or not installed."""
    out = []
    for n in names:
        if n == "xgb" and (not use_xgboost or not xgboost_available()):
            continue
        out.append(n)
    return out

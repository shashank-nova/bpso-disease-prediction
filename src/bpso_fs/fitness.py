r"""Objective (fitness) function for wrapper feature selection.

For a binary selection vector :math:`z \in \{0,1\}^d` the fitness to **minimise** is

.. math::

    F(z) = \alpha \,\bigl(1 - P(z)\bigr) + (1 - \alpha)\,\frac{|z|}{d}

where

* :math:`P(z) \in [0, 1]` is the mean inner *k*-fold stratified cross-validation score
  (accuracy, balanced accuracy, F1 or ROC-AUC) of the wrapper classifier trained only on
  the selected features, computed on the outer training fold only;
* :math:`|z| = \sum_j z_j` is the number of selected features and :math:`d` the total
  number of features;
* :math:`\alpha \in [0, 1]` weights predictive performance against subset compactness
  (default :math:`\alpha = 0.99`).

LaTeX for the paper::

    F(z) = \alpha\,\bigl(1 - P(z)\bigr) + (1-\alpha)\,\frac{|z|}{d},
    \qquad |z| = \sum_{j=1}^{d} z_j

An empty subset (:math:`|z| = 0`) is assigned the worst possible fitness, 1.0.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

SCORERS = ("accuracy", "balanced_accuracy", "f1", "roc_auc")


def make_wrapper_estimator(name: str, random_state: int | None = None) -> BaseEstimator:
    """Classifier used *inside* the fitness function."""
    if name == "logreg":
        return LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    raise ValueError(f"Unknown wrapper estimator '{name}' (use 'logreg' or 'knn')")


def fitness_value(score: float, n_selected: int, n_features: int, alpha: float) -> float:
    """F = alpha * (1 - score) + (1 - alpha) * n_selected / n_features."""
    if n_selected == 0:
        return 1.0
    return float(alpha * (1.0 - score) + (1.0 - alpha) * (n_selected / n_features))


class SubsetFitness:
    """Callable fitness for binary masks, with inner CV and memoisation.

    Args:
        X: training matrix (selection view; one column per original feature).
        y: binary labels.
        alpha: performance/size trade-off weight.
        estimator: wrapper classifier name ('logreg' | 'knn') or an sklearn estimator.
        cv: number of inner stratified folds.
        metric: one of :data:`SCORERS`.
        random_state: seed for the inner CV split.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.99,
        estimator: str | BaseEstimator = "logreg",
        cv: int = 5,
        metric: str = "accuracy",
        random_state: int | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if metric not in SCORERS:
            raise ValueError(f"metric must be one of {SCORERS}")
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y).astype(int)
        self.n_features = self.X.shape[1]
        self.alpha = float(alpha)
        self.metric = metric
        self.estimator = (
            make_wrapper_estimator(estimator, random_state)
            if isinstance(estimator, str)
            else estimator
        )
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.folds = list(skf.split(self.X, self.y))
        self.cache: dict[bytes, tuple[float, float]] = {}
        self.n_calls = 0  # every request, including cache hits
        self.n_evaluations = 0  # unique subsets actually scored
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ scoring
    def _score(self, cols: np.ndarray) -> float:
        scores = []
        for tr, va in self.folds:
            est = clone(self.estimator)
            Xtr, Xva = self.X[np.ix_(tr, cols)], self.X[np.ix_(va, cols)]
            est.fit(Xtr, self.y[tr])
            if self.metric == "roc_auc":
                if hasattr(est, "predict_proba"):
                    s = est.predict_proba(Xva)[:, 1]
                else:
                    s = est.decision_function(Xva)
                scores.append(roc_auc_score(self.y[va], s))
            else:
                pred = est.predict(Xva)
                if self.metric == "accuracy":
                    scores.append(accuracy_score(self.y[va], pred))
                elif self.metric == "balanced_accuracy":
                    scores.append(balanced_accuracy_score(self.y[va], pred))
                else:
                    scores.append(f1_score(self.y[va], pred, zero_division=0))
        return float(np.mean(scores))

    def evaluate(self, mask: np.ndarray) -> tuple[float, float]:
        """Return (fitness, inner-CV score) for a binary mask."""
        mask = np.asarray(mask, dtype=bool)
        key = np.packbits(mask).tobytes() + bytes([self.n_features % 256])
        with self._lock:
            self.n_calls += 1
            if key in self.cache:
                return self.cache[key]
        n_sel = int(mask.sum())
        if n_sel == 0:
            result = (1.0, float("nan"))
        else:
            score = self._score(np.flatnonzero(mask))
            result = (fitness_value(score, n_sel, self.n_features, self.alpha), score)
        with self._lock:
            if key not in self.cache:
                self.cache[key] = result
                self.n_evaluations += 1
        return result

    def __call__(self, mask: np.ndarray) -> float:
        return self.evaluate(mask)[0]


FitnessFactory = Callable[[np.ndarray, np.ndarray], SubsetFitness]

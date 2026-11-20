"""Model-based baselines: RFE and L1-regularised logistic regression."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from .base import FeatureSelector


class RFESelector(FeatureSelector):
    """Recursive Feature Elimination with logistic regression down to ``k`` features."""

    name = "rfe"

    def __init__(self, k: int = 5, random_state: int | None = None) -> None:
        self.k = k
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        k = int(np.clip(self.k, 1, X.shape[1]))
        est = LogisticRegression(max_iter=2000, solver="liblinear", random_state=self.random_state)
        rfe = RFE(est, n_features_to_select=k, step=1).fit(X, y)
        self.ranking_ = rfe.ranking_
        self.support_ = rfe.support_


class L1Selector(FeatureSelector):
    """L1 (LASSO) logistic regression; C chosen by inner CV; keeps non-zero coefficients.

    The C grid is ``Cs`` log-spaced values in [1e-3, 10] on standardised features. (Weaker
    regularisation keeps almost every feature and makes liblinear very slow on separable data.)

    If every coefficient is zero, the single feature with the largest |coef| along the
    regularisation path's least-regularised fit is kept, so the subset is never empty.
    """

    name = "l1"

    def __init__(
        self, cv: int = 5, Cs: int = 15, scoring: str = "accuracy", tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:  # fmt: skip
        self.cv = cv
        self.Cs = Cs
        self.scoring = scoring
        self.tol = tol
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # scikit-learn >= 1.8 deprecates the `penalty` argument (FutureWarning only); the
        # model fitted here is the same L1 logistic regression in every supported version.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._fit_l1(X, y)

    def _fit_l1(self, X: np.ndarray, y: np.ndarray) -> None:
        model = LogisticRegressionCV(
            Cs=np.logspace(-3, 1, self.Cs),
            cv=StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state),
            penalty="l1",
            solver="liblinear",
            scoring=self.scoring,
            max_iter=2000,
            random_state=self.random_state,
        ).fit(X, y)
        coef = np.abs(model.coef_.ravel())
        mask = coef > self.tol
        if not mask.any():
            fallback = LogisticRegression(penalty="l1", solver="liblinear", C=1e3, max_iter=5000)
            coef_fb = np.abs(fallback.fit(X, y).coef_.ravel())
            mask = np.zeros_like(mask)
            mask[int(np.argmax(coef_fb))] = True
        self.C_ = float(model.C_[0])
        self.coef_ = model.coef_.ravel()
        self.support_ = mask

"""Filter baselines: mutual information and ANOVA F-test with SelectKBest.

Two k policies:
* fixed ``k`` (in the experiments: the median BPSO subset size for that dataset), and
* ``k='cv'``: k chosen by inner stratified CV of a logistic-regression pipeline.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

from .base import FeatureSelector


class KBestSelector(FeatureSelector):
    """SelectKBest with a fixed k or a k tuned by inner cross-validation.

    Args:
        score: 'mutual_info' or 'anova'.
        k: integer, or 'cv' to tune k over 1..d by inner CV.
        cv: inner folds when k='cv'.
        scoring: sklearn scoring string for the k search.
        random_state: seed (MI estimator and CV shuffling).
    """

    def __init__(
        self,
        score: str = "mutual_info",
        k: int | str = 5,
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: int | None = None,
    ) -> None:
        if score not in ("mutual_info", "anova"):
            raise ValueError("score must be 'mutual_info' or 'anova'")
        self.score = score
        self.k = k
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.name = f"{'mi' if score == 'mutual_info' else 'anova'}_{'cv' if k == 'cv' else 'k'}"

    def _score_func(self):
        if self.score == "mutual_info":
            return partial(mutual_info_classif, random_state=self.random_state)
        return f_classif

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        d = X.shape[1]
        if self.k == "cv":
            # Equivalent to GridSearchCV over Pipeline(SelectKBest(k), LogisticRegression) with
            # k = 1..d, but the univariate scores are computed once per inner fold.
            skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
            scorer = get_scorer(self.scoring)
            cv_scores = np.zeros((self.cv, d))
            for i, (tr, va) in enumerate(skf.split(X, y)):
                s = self._score_func()(X[tr], y[tr])
                s = s[0] if isinstance(s, tuple) else s  # f_classif returns (F, p)
                s = np.nan_to_num(np.asarray(s, dtype=float), nan=-np.inf)
                order = np.argsort(-s, kind="stable")
                for k in range(1, d + 1):
                    cols = order[:k]
                    clf = LogisticRegression(max_iter=2000, solver="liblinear").fit(
                        X[tr][:, cols], y[tr]
                    )
                    cv_scores[i, k - 1] = scorer(clf, X[va][:, cols], y[va])
            mean = cv_scores.mean(axis=0)
            k = int(np.argmax(mean) + 1)  # smallest k among ties
            self.cv_scores_ = mean
        else:
            k = int(np.clip(int(self.k), 1, d))
        kb = SelectKBest(self._score_func(), k=k).fit(X, y)
        self.scores_ = kb.scores_
        self.k_ = k
        self.support_ = kb.get_support()

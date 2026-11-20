"""Common interface for feature selectors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np


class FeatureSelector(ABC):
    """Base class: ``fit(X, y)`` learns a boolean mask over the columns of ``X``.

    Subclasses implement :meth:`_fit` and set ``self.support_``.
    After fitting, ``selection_time_`` holds the wall-clock seconds spent in ``fit``.
    """

    name: str = "selector"
    support_: np.ndarray
    selection_time_: float

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        t0 = time.perf_counter()
        self._fit(X, y)
        self.selection_time_ = time.perf_counter() - t0
        self.support_ = np.asarray(self.support_, dtype=bool)
        if self.support_.shape != (X.shape[1],):
            raise RuntimeError(f"{self.name}: support_ has wrong shape {self.support_.shape}")
        if not self.support_.any():
            raise RuntimeError(f"{self.name}: selected zero features")
        return self

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Boolean mask (or column indices) of the selected features."""
        if not hasattr(self, "support_"):
            raise RuntimeError("Selector is not fitted")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Keep only the selected columns."""
        return np.asarray(X)[:, self.get_support()]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def n_selected_(self) -> int:
        return int(self.get_support().sum())


class AllFeatures(FeatureSelector):
    """No selection: keep every feature (the 'all features' baseline)."""

    name = "all"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.support_ = np.ones(X.shape[1], dtype=bool)


class RandomSubset(FeatureSelector):
    """Uniformly random subset of size ``k`` (sanity baseline)."""

    name = "random"

    def __init__(self, k: int, random_state: int | None = None):
        self.k = k
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        k = int(np.clip(self.k, 1, X.shape[1]))
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[rng.choice(X.shape[1], size=k, replace=False)] = True
        self.support_ = mask

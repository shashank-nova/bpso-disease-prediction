"""Preprocessing pipelines.

Two views of the data are used, both fitted on training data only:

* **Selection view** (:func:`make_selection_view`): one output column per original feature
  (median / most-frequent imputation, ordinal codes for nominal features, standardisation).
  Feature selectors operate on this matrix, so a selection mask maps 1:1 onto the original
  features.
* **Model preprocessor** (:func:`make_model_preprocessor`): used by the final classifiers on
  the *selected* original columns; nominal features are one-hot encoded here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def make_selection_view(
    numeric_cols: Sequence[str], categorical_cols: Sequence[str]
) -> ColumnTransformer:
    """1:1 numeric representation of the original features (output order = numeric + categorical)."""
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                list(numeric_cols),
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(categorical_cols),
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_model_preprocessor(
    numeric_cols: Sequence[str], categorical_cols: Sequence[str]
) -> ColumnTransformer:
    """Preprocessor for the final classifier on a set of (selected) original columns."""
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                list(numeric_cols),
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(categorical_cols),
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed list of DataFrame columns (first step of the final model pipeline)."""

    def __init__(self, columns: Sequence[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> ColumnSelector:  # noqa: D401
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.columns)]


def build_model_pipeline(
    selected: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    classifier: BaseEstimator,
) -> Pipeline:
    """ColumnSelector -> preprocessing -> classifier for the selected original features."""
    sel = list(selected)
    num = [c for c in numeric_cols if c in sel]
    cat = [c for c in categorical_cols if c in sel]
    return Pipeline(
        [
            ("select", ColumnSelector(num + cat)),
            ("prep", make_model_preprocessor(num, cat)),
            ("clf", classifier),
        ]
    )


def mask_to_names(mask: np.ndarray, feature_names: Sequence[str]) -> list[str]:
    """Convert a boolean mask over the original features to feature names."""
    return [f for f, m in zip(feature_names, np.asarray(mask, dtype=bool), strict=True) if m]

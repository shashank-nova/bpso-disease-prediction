"""Dataset loaders.

Every loader returns a :class:`DatasetBundle` whose feature frame is ordered as
``numeric_cols + categorical_cols``. Feature-selection masks throughout the project
refer to this column order.

Downloaded files are cached in ``data/raw/``. If every source for a dataset fails, a
:class:`DatasetUnavailableError` is raised with manual-download instructions; no data
is ever synthesised.

Positive class (label 1) is always "disease present".
"""

from __future__ import annotations

import io
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


class DatasetUnavailableError(RuntimeError):
    """Raised when a dataset cannot be loaded from any source."""


@dataclass
class DatasetBundle:
    """A loaded tabular binary-classification dataset."""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    numeric_cols: list[str]
    categorical_cols: list[str]
    description: str = ""
    source: str = ""
    citation: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        order = list(self.numeric_cols) + list(self.categorical_cols)
        if set(order) != set(self.X.columns):
            raise ValueError("numeric_cols + categorical_cols must cover exactly the X columns")
        self.X = self.X[order].reset_index(drop=True)
        self.y = np.asarray(self.y).astype(int)
        if set(np.unique(self.y)) - {0, 1}:
            raise ValueError("Target must be binary 0/1")

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def summary(self) -> dict:
        """Shape, class balance and missing-value counts."""
        n_pos = int(self.y.sum())
        missing = {c: int(v) for c, v in self.X.isna().sum().items() if v > 0}
        return {
            "name": self.name,
            "n_samples": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "n_numeric": len(self.numeric_cols),
            "n_categorical": len(self.categorical_cols),
            "n_positive": n_pos,
            "n_negative": int(len(self.y) - n_pos),
            "positive_rate": float(n_pos / len(self.y)),
            "n_missing_cells": int(self.X.isna().sum().sum()),
            "n_rows_with_missing": int(self.X.isna().any(axis=1).sum()),
            "missing_per_feature": missing,
            "feature_names": self.feature_names,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "source": self.source,
            "description": self.description,
        }


# --------------------------------------------------------------------------- helpers


def _download(urls: list[str], cache_name: str, timeout: int = 30) -> tuple[bytes, str]:
    """Return file bytes from cache or the first URL that works."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / cache_name
    if cache.exists() and cache.stat().st_size > 0:
        src_file = cache.with_suffix(cache.suffix + ".source")
        src = src_file.read_text().strip() if src_file.exists() else f"cache:{cache}"
        return cache.read_bytes(), src
    errors = []
    for url in urls:
        try:
            logger.info("Downloading %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "bpso-fs/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("empty response")
            cache.write_bytes(data)
            cache.with_suffix(cache.suffix + ".source").write_text(url)
            return data, url
        except Exception as exc:  # noqa: BLE001 - we report all failures together
            errors.append(f"  - {url}: {exc}")
    raise DatasetUnavailableError(
        f"Could not download '{cache_name}'. Tried:\n"
        + "\n".join(errors)
        + f"\nManual fix: download the file from one of the URLs above and save it as\n  {cache}"
    )


# --------------------------------------------------------------------------- datasets


def load_breast_cancer() -> DatasetBundle:
    """Wisconsin Diagnostic Breast Cancer (bundled with scikit-learn).

    scikit-learn encodes 0 = malignant, 1 = benign; we flip it so 1 = malignant (disease).
    """
    from sklearn.datasets import load_breast_cancer as _lbc

    raw = _lbc(as_frame=True)
    X = raw.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    y = 1 - raw.target.to_numpy()
    return DatasetBundle(
        name="breast_cancer",
        X=X,
        y=y,
        numeric_cols=list(X.columns),
        categorical_cols=[],
        description="Wisconsin Diagnostic Breast Cancer; 1 = malignant",
        source="sklearn.datasets.load_breast_cancer",
        citation=(
            "W. Wolberg, O. Mangasarian, N. Street, and W. Street, 'Breast Cancer Wisconsin "
            "(Diagnostic),' UCI Machine Learning Repository, 1995, doi:10.24432/C5DW2B."
        ),
    )


PIMA_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
PIMA_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_pima() -> DatasetBundle:
    """Pima Indians Diabetes (768 x 8). Impossible zeros are treated as missing."""
    data, src = _download(
        [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        ],
        "pima-indians-diabetes.data.csv",
    )
    df = pd.read_csv(io.BytesIO(data), header=None, names=PIMA_COLS + ["Outcome"])
    if df.shape != (768, 9):
        raise DatasetUnavailableError(f"Unexpected Pima shape {df.shape}; expected (768, 9)")
    X = df[PIMA_COLS].astype(float).copy()
    for c in PIMA_ZERO_AS_MISSING:
        X.loc[X[c] == 0, c] = np.nan
    return DatasetBundle(
        name="pima_diabetes",
        X=X,
        y=df["Outcome"].to_numpy(),
        numeric_cols=PIMA_COLS,
        categorical_cols=[],
        description="Pima Indians Diabetes; zeros in Glucose/BloodPressure/SkinThickness/"
        "Insulin/BMI treated as missing; 1 = diabetes",
        source=src,
        citation=(
            "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, "
            "'Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,' "
            "in Proc. Annu. Symp. Comput. Appl. Med. Care, 1988, pp. 261-265."
        ),
    )


HEART_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]  # fmt: skip
HEART_CATEGORICAL = ["cp", "restecg", "slope", "thal"]


def _heart_from_uci(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), header=None, names=HEART_COLS + ["num"], na_values="?")
    return df


def _heart_from_islr(data: bytes) -> pd.DataFrame:
    """Convert the ISLR 'Heart.csv' copy of the Cleveland data to UCI coding."""
    df = pd.read_csv(io.BytesIO(data), na_values=["NA", ""])
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])
    rename = {
        "Age": "age", "Sex": "sex", "ChestPain": "cp", "RestBP": "trestbps", "Chol": "chol",
        "Fbs": "fbs", "RestECG": "restecg", "MaxHR": "thalach", "ExAng": "exang",
        "Oldpeak": "oldpeak", "Slope": "slope", "Ca": "ca", "Thal": "thal", "AHD": "num",
    }  # fmt: skip
    df = df.rename(columns=rename)
    df["cp"] = df["cp"].map({"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4})
    df["thal"] = df["thal"].map({"normal": 3, "fixed": 6, "reversable": 7})
    df["num"] = df["num"].map({"No": 0, "Yes": 1})
    return df[HEART_COLS + ["num"]]


def load_heart() -> DatasetBundle:
    """UCI Heart Disease, Cleveland subset (303 x 13). Target: 0 = no disease, >0 = disease."""
    try:
        data, src = _download(
            [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            ],
            "processed.cleveland.data",
        )
        df = _heart_from_uci(data)
    except DatasetUnavailableError as uci_err:
        logger.warning("UCI source unavailable; trying the ISLR mirror of the same 303 records")
        try:
            data, src = _download(
                ["https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"],
                "Heart_ISLR.csv",
            )
        except DatasetUnavailableError as mirror_err:
            raise DatasetUnavailableError(f"{uci_err}\n{mirror_err}") from mirror_err
        df = _heart_from_islr(data)
    if df.shape[0] != 303:
        raise DatasetUnavailableError(f"Unexpected Cleveland row count {df.shape[0]}; expected 303")
    numeric = [c for c in HEART_COLS if c not in HEART_CATEGORICAL]
    return DatasetBundle(
        name="heart_cleveland",
        X=df[HEART_COLS].astype(float),
        y=(df["num"].astype(float) > 0).astype(int).to_numpy(),
        numeric_cols=numeric,
        categorical_cols=HEART_CATEGORICAL,
        description="UCI Heart Disease (Cleveland); nominal cp/restecg/slope/thal one-hot "
        "encoded in the final models; 1 = disease (num > 0)",
        source=src,
        citation=(
            "A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, 'Heart Disease,' "
            "UCI Machine Learning Repository, 1988, doi:10.24432/C52P4X."
        ),
    )


ILPD_COLS = [
    "Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "AG_Ratio",
]  # fmt: skip


def load_ilpd() -> DatasetBundle:
    """Indian Liver Patient Dataset (583 x 10). Original label 1 = liver patient."""
    data, src = _download(
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/"
            "Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
        ],
        "ILPD.csv",
    )
    df = pd.read_csv(io.BytesIO(data), header=None, names=ILPD_COLS + ["Selector"])
    df["Gender"] = (df["Gender"].astype(str).str.strip().str.lower() == "male").astype(float)
    return DatasetBundle(
        name="ilpd",
        X=df[ILPD_COLS].astype(float),
        y=(df["Selector"] == 1).astype(int).to_numpy(),
        numeric_cols=ILPD_COLS,
        categorical_cols=[],
        description="Indian Liver Patient Dataset; 1 = liver patient",
        source=src,
        citation=(
            "B. Ramana and N. Venkateswarlu, 'ILPD (Indian Liver Patient Dataset),' "
            "UCI Machine Learning Repository, 2012, doi:10.24432/C5D02C."
        ),
    )


def load_ckd() -> DatasetBundle:
    """Chronic Kidney Disease (400 x 24), UCI ARFF file. 1 = ckd."""
    data, src = _download(
        [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/"
            "Chronic_Kidney_Disease.rar",
        ],
        "chronic_kidney_disease.arff",
    )
    text = data.decode("utf-8", errors="ignore")
    if "@data" not in text.lower():
        raise DatasetUnavailableError(
            "The CKD source is a .rar archive. Extract 'chronic_kidney_disease_full.arff' and "
            f"save it as {RAW_DIR / 'chronic_kidney_disease.arff'}"
        )
    attrs, rows, in_data = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if s.lower().startswith("@attribute"):
            attrs.append(s.split()[1].strip("'"))
        elif s.lower().startswith("@data"):
            in_data = True
        elif in_data:
            vals = [v.strip().strip("'") for v in s.split(",")]
            if len(vals) >= len(attrs):
                rows.append(vals[: len(attrs)])
    df = pd.DataFrame(rows, columns=attrs).replace({"?": np.nan, "": np.nan})
    # The UCI ARFF spells these counts ``wbcc`` and ``rbcc``; the paper/code schema
    # uses the common abbreviations ``wc`` and ``rc``.
    aliases = {"wbcc": "wc", "rbcc": "rc"}
    df = df.rename(columns=aliases)
    attrs = [aliases.get(name, name) for name in attrs]
    target = "class"
    numeric = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]  # fmt: skip
    nominal = [c for c in attrs if c not in numeric + [target]]
    X = df[numeric + nominal].copy()
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce")
    for c in nominal:  # encode nominal strings as integer codes (missing stays NaN)
        codes = {v: i for i, v in enumerate(sorted(X[c].dropna().str.strip().unique()))}
        X[c] = X[c].str.strip().map(codes).astype(float)
    return DatasetBundle(
        name="ckd",
        X=X,
        y=(df[target].str.strip() == "ckd").astype(int).to_numpy(),
        numeric_cols=numeric,
        categorical_cols=nominal,
        description="Chronic Kidney Disease; 1 = ckd",
        source=src,
        citation=(
            "L. Rubini, P. Soundarapandian, and P. Eswaran, 'Chronic Kidney Disease,' "
            "UCI Machine Learning Repository, 2015, doi:10.24432/C5G020."
        ),
    )


def load_csv_dataset(
    path: str | Path,
    target: str,
    positive_label: str | int | None = None,
    name: str | None = None,
    categorical: list[str] | None = None,
    drop: list[str] | None = None,
) -> DatasetBundle:
    """Load any CSV with a binary target column.

    Args:
        path: CSV file path.
        target: name of the target column.
        positive_label: value of the target that means "disease". If omitted, the target
            must already be 0/1.
        name: dataset name used for the results folder (defaults to the file stem).
        categorical: columns to treat as nominal categorical. Defaults to object/category dtypes.
        drop: columns to drop (e.g. IDs).
    """
    path = Path(path)
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in {list(df.columns)}")
    df = df.drop(columns=drop or [])
    y_raw = df.pop(target)
    if positive_label is not None:
        y = (y_raw.astype(str) == str(positive_label)).astype(int).to_numpy()
    else:
        y = pd.to_numeric(y_raw).astype(int).to_numpy()
    if categorical is None:
        categorical = [
            c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"
        ]
    for c in categorical:
        codes = {v: i for i, v in enumerate(sorted(df[c].dropna().astype(str).unique()))}
        df[c] = df[c].astype(str).where(df[c].notna()).map(codes).astype(float)
    numeric = [c for c in df.columns if c not in categorical]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return DatasetBundle(
        name=name or path.stem,
        X=df,
        y=y,
        numeric_cols=numeric,
        categorical_cols=list(categorical),
        description=f"Custom CSV dataset from {path.name}",
        source=str(path),
    )


_LOADERS: dict[str, Callable[[], DatasetBundle]] = {
    "breast_cancer": load_breast_cancer,
    "pima_diabetes": load_pima,
    "heart_cleveland": load_heart,
    "ilpd": load_ilpd,
    "ckd": load_ckd,
}


def available_datasets() -> list[str]:
    """Names accepted by :func:`load_dataset`."""
    return list(_LOADERS)


def load_dataset(name: str) -> DatasetBundle:
    """Load a built-in dataset by name."""
    if name not in _LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {available_datasets()}")
    return _LOADERS[name]()

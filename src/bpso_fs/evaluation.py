"""Metrics, statistical tests and feature-subset stability."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """Binary classification metrics (positive class = 1 = disease)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# --------------------------------------------------------------------------- tests


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs are passed through)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    if m == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out.tolist()


def wilcoxon_vs_reference(
    df: pd.DataFrame,
    reference: str,
    metric: str,
    method_col: str = "method",
    pair_cols: tuple[str, ...] = ("seed", "fold"),
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test of ``reference`` vs every other method.

    ``df`` must hold one row per (method, *pair_cols). Returns one row per comparison with
    the median difference (reference - other), raw p-value and Holm-adjusted p-value.
    """
    wide = df.pivot_table(index=list(pair_cols), columns=method_col, values=metric)
    if reference not in wide.columns:
        return pd.DataFrame()
    rows = []
    for other in [c for c in wide.columns if c != reference]:
        pair = wide[[reference, other]].dropna()
        diff = pair[reference] - pair[other]
        if len(pair) < 2 or np.allclose(diff, 0):
            p = 1.0 if len(pair) >= 2 else float("nan")
            stat = float("nan")
        else:
            res = stats.wilcoxon(pair[reference], pair[other], zero_method="wilcox")
            stat, p = float(res.statistic), float(res.pvalue)
        wins = int((diff > 1e-12).sum())
        losses = int((diff < -1e-12).sum())
        rows.append(
            {
                "reference": reference,
                "other": other,
                "metric": metric,
                "n_pairs": int(len(pair)),
                "mean_ref": float(pair[reference].mean()),
                "mean_other": float(pair[other].mean()),
                "median_diff": float(diff.median()),
                "wins": wins,
                "ties": int(len(pair) - wins - losses),
                "losses": losses,
                "statistic": stat,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_correction(out["p_value"].tolist())
    return out


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test over blocks (rows) x methods (columns); higher score is better.

    Returns the statistic, p-value, average ranks (rank 1 = best) and the Nemenyi critical
    difference CD = q_alpha * sqrt(k(k+1) / (6N)).
    """
    scores = scores.dropna()
    n_blocks, k = scores.shape
    ranks = scores.rank(axis=1, ascending=False, method="average")
    avg = ranks.mean().sort_values()
    if n_blocks < 2 or k < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "avg_ranks": avg.to_dict(),
                "cd": float("nan"), "n_blocks": n_blocks, "k": k}  # fmt: skip
    stat, p = stats.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    q = stats.studentized_range.ppf(1 - alpha, k, 1e6) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "avg_ranks": avg.to_dict(),
        "cd": float(cd),
        "q_alpha": float(q),
        "n_blocks": int(n_blocks),
        "k": int(k),
    }


# --------------------------------------------------------------------------- stability


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def mean_pairwise_jaccard(masks: list[np.ndarray]) -> float:
    """Average Jaccard similarity over all pairs of selected subsets (1 = identical)."""
    if len(masks) < 2:
        return float("nan")
    return float(np.mean([jaccard(a, b) for a, b in combinations(masks, 2)]))


def kuncheva_index(masks: list[np.ndarray]) -> float:
    """Kuncheva consistency index; defined only when all subsets have the same size k < d."""
    if len(masks) < 2:
        return float("nan")
    sizes = {int(np.sum(m)) for m in masks}
    d = len(masks[0])
    if len(sizes) != 1:
        return float("nan")
    k = sizes.pop()
    if k == 0 or k == d:
        return float("nan")
    vals = []
    for a, b in combinations(masks, 2):
        r = int((np.asarray(a, bool) & np.asarray(b, bool)).sum())
        vals.append((r * d - k * k) / (k * (d - k)))
    return float(np.mean(vals))


def selection_frequency(masks: list[np.ndarray], feature_names: list[str]) -> pd.Series:
    """Fraction of runs in which each feature was selected."""
    M = np.asarray(masks, dtype=float)
    return pd.Series(M.mean(axis=0), index=feature_names).sort_values(ascending=False)

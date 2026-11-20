"""Paper figures (PDF + PNG, 300 dpi, IEEE column widths, no in-figure titles).

Colour use: BPSO is always slot-1 blue and the GA / all-features comparator slot-2 orange;
other baselines are neutral grey so the proposed method stands out. Sequential data
(confusion matrix) uses a single blue ramp. No dual y-axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .evaluation import METRIC_LABELS  # noqa: E402
from .experiment import METHOD_LABELS  # noqa: E402

SINGLE_COL = 3.5  # inches (IEEE single column)
DOUBLE_COL = 7.16  # inches (IEEE two-column figure*)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
GREY = "#8c8c88"
INK = "#2b2b29"
INK_MUTED = "#6b6b66"
GRID = "#e4e3de"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

METHOD_COLORS = {"bpso": BLUE, "ga": ORANGE, "all": INK_MUTED}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out: Path) -> list[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out.with_suffix(".pdf"), out.with_suffix(".png")]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def _label(method: str) -> str:
    if method.startswith("bpso_alpha"):
        return f"BPSO, α={method.removeprefix('bpso_alpha')}"
    return METHOD_LABELS.get(method, method)


# --------------------------------------------------------------------------- figures


def plot_convergence(history: pd.DataFrame, out: Path) -> list[Path]:
    """Best fitness vs iteration (mean ± std over runs) for BPSO and GA; BPSO swarm mean dotted."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.3))
    for m in ("bpso", "ga"):
        h = history[history["method"] == m]
        if h.empty:
            continue
        g = h.groupby("iteration")["best_fitness"].agg(["mean", "std"]).fillna(0)
        c = METHOD_COLORS[m]
        ax.plot(g.index, g["mean"], color=c, label=f"{_label(m)}: best")
        ax.fill_between(
            g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.15, lw=0
        )
        if m == "bpso":
            gm = h.groupby("iteration")["mean_fitness"].mean()
            ax.plot(gm.index, gm.values, color=c, ls=":", lw=1.2, label=f"{_label(m)}: swarm mean")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness F(z) (lower is better)")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0, handlelength=1.8
    )
    return _save(fig, out)


def plot_subset_size(history: pd.DataFrame, n_features: int, out: Path) -> list[Path]:
    """Size of the global-best subset vs iteration."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.1))
    for m in ("bpso", "ga"):
        h = history[history["method"] == m]
        if h.empty:
            continue
        g = h.groupby("iteration")["best_size"].agg(["mean", "std"]).fillna(0)
        c = METHOD_COLORS[m]
        ax.plot(g.index, g["mean"], color=c, label=_label(m))
        ax.fill_between(
            g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.15, lw=0
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Selected features (best subset)")
    ax.set_ylim(0, n_features + 0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return _save(fig, out)


def plot_feature_frequency(freq: pd.Series, out: Path, top: int = 15) -> list[Path]:
    """Horizontal bars: fraction of BPSO runs selecting each feature."""
    f = freq.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 0.17 * len(f) + 0.6))
    ax.barh(range(len(f)), f.values, color=BLUE, height=0.7)
    ax.set_yticks(range(len(f)))
    ax.set_yticklabels([s.replace("_", " ") for s in f.index])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Selection frequency across BPSO runs")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(f.values):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=6, color=INK_MUTED)
    return _save(fig, out)


def plot_roc(
    preds: pd.DataFrame, classifier: str, out: Path, methods=("bpso", "all")
) -> list[Path]:
    """Mean ROC (pooled out-of-fold predictions per seed, ± std across seeds)."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    grid = np.linspace(0, 1, 201)
    colors = {"bpso": BLUE, "all": ORANGE}
    for m in methods:
        p = preds[(preds["method"] == m) & (preds["classifier"] == classifier)]
        if p.empty:
            continue
        tprs, aucs = [], []
        for _, ps in p.groupby("seed"):
            fpr, tpr, _ = roc_curve(ps["y_true"], ps["y_score"])
            t = np.interp(grid, fpr, tpr)
            t[0] = 0.0
            tprs.append(t)
            from sklearn.metrics import roc_auc_score

            aucs.append(roc_auc_score(ps["y_true"], ps["y_score"]))
        mt, st = np.mean(tprs, axis=0), np.std(tprs, axis=0)
        c = colors.get(m, GREY)
        ax.plot(
            grid, mt, color=c, label=f"{_label(m)} (AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})"
        )
        ax.fill_between(
            grid, np.clip(mt - st, 0, 1), np.clip(mt + st, 0, 1), color=c, alpha=0.15, lw=0
        )
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return _save(fig, out)


def plot_method_comparison(raw: pd.DataFrame, classifier: str, out: Path, methods: list[str],
                           metrics=("f1", "roc_auc")) -> list[Path]:  # fmt: skip
    """Box plots (one per method) of per-fold scores, one panel per metric."""
    r = raw[raw["classifier"] == classifier]
    methods = [m for m in methods if m in set(r["method"])]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(DOUBLE_COL, 0.22 * len(methods) + 0.7), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, met in zip(axes, metrics, strict=True):
        data = [r.loc[r["method"] == m, met].dropna().to_numpy() for m in methods][::-1]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": GREY, "markeredgecolor": "none"},
                        medianprops={"color": INK, "lw": 1.0}, whiskerprops={"color": INK_MUTED, "lw": 0.7},
                        capprops={"color": INK_MUTED, "lw": 0.7})  # fmt: skip
        for patch, m in zip(bp["boxes"], methods[::-1], strict=True):
            patch.set_facecolor(BLUE if m == "bpso" else "#d9d8d2")
            patch.set_edgecolor(INK_MUTED)
            patch.set_linewidth(0.6)
        means = [d.mean() if len(d) else np.nan for d in data]
        ax.scatter(
            means, range(1, len(data) + 1), marker="D", s=10, color=INK, zorder=3, label="Mean"
        )
        ax.set_xlabel(METRIC_LABELS[met])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(1, len(methods) + 1))
    axes[0].set_yticklabels([_label(m) for m in methods[::-1]])
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.94))
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0)
    )
    return _save(fig, out)


def plot_alpha_sensitivity(sweep: pd.DataFrame, classifier: str, n_features: int, out: Path,
                           metrics=("f1", "roc_auc")) -> list[Path]:  # fmt: skip
    """Two stacked panels sharing α: performance (top) and subset size (bottom)."""
    s = sweep[sweep["classifier"] == classifier]
    g = s.groupby("alpha")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.2), sharex=True)
    colors = {"f1": BLUE, "roc_auc": ORANGE, "accuracy": AQUA}
    markers = {"f1": "o", "roc_auc": "s", "accuracy": "^"}
    for met in metrics:
        st = g[met].agg(["mean", "std"])
        a1.errorbar(st.index, st["mean"], yerr=st["std"], color=colors[met], marker=markers[met],
                    ms=4, capsize=2, lw=1.2, label=METRIC_LABELS[met])  # fmt: skip
    a1.set_ylabel("Test score")
    a1.legend(loc="lower right")
    sz = g["n_selected"].agg(["mean", "std"])
    a2.errorbar(
        sz.index, sz["mean"], yerr=sz["std"], color=INK_MUTED, marker="o", ms=4, capsize=2, lw=1.2
    )
    a2.set_ylabel("Selected features")
    a2.set_ylim(0, n_features + 0.5)
    a2.set_xlabel("α (weight on predictive performance)")
    a2.set_xticks(sorted(s["alpha"].unique()))
    fig.tight_layout(h_pad=0.6)
    return _save(fig, out)


def plot_confusion(
    raw: pd.DataFrame, method: str, classifier: str, n_seeds: int, out: Path
) -> list[Path]:
    """Confusion matrix summed over folds, averaged over repetitions; cells show count and row %."""
    r = raw[(raw["method"] == method) & (raw["classifier"] == classifier)]
    cm = np.array(
        [[r["tn"].sum(), r["fp"].sum()], [r["fn"].sum(), r["tp"].sum()]], dtype=float
    ) / max(n_seeds, 1)
    row = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.3, 2.0))
    ax.imshow(row, cmap=BLUES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.1f}\n({row[i, j]:.1%})", ha="center", va="center",
                    fontsize=7, color="white" if row[i, j] > 0.55 else INK)  # fmt: skip
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return _save(fig, out)


def plot_cd_diagram(avg_ranks: dict[str, float], cd: float, out: Path) -> list[Path]:
    """Critical-difference diagram (Demšar, 2006). Rank 1 (best) is on the right; methods
    joined by a thick grey bar are not significantly different (Nemenyi, α = 0.05)."""
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    k = len(items)
    half = (k + 1) // 2
    x = lambda r: -float(r)  # noqa: E731 - rank 1 drawn at the right
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.24 * half + 1.1))
    ax.axis("off")
    y0 = 0.0
    ax.plot([x(k), x(1)], [y0, y0], color=INK, lw=0.8)
    for t in range(1, k + 1):
        ax.plot([x(t), x(t)], [y0, y0 + 0.08], color=INK, lw=0.8)
        ax.text(x(t), y0 + 0.13, str(t), ha="center", va="bottom", fontsize=7)
    ax.plot([x(k), x(k) + cd], [y0 + 0.55, y0 + 0.55], color=INK, lw=1.0)
    for e in (x(k), x(k) + cd):
        ax.plot([e, e], [y0 + 0.5, y0 + 0.6], color=INK, lw=1.0)
    ax.text(x(k) + cd / 2, y0 + 0.62, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    ranks = [r for _, r in items]
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    clique_depth = 0.12 + 0.08 * len(cliques)
    for n, (a, b) in enumerate(cliques):
        yy = y0 - 0.12 - 0.08 * n
        ax.plot([x(ranks[a]) + 0.04, x(ranks[b]) - 0.04], [yy, yy], color=INK_MUTED, lw=2.4,
                solid_capstyle="round")  # fmt: skip
    for i, (m, r) in enumerate(items):
        best_side = i < half  # best half labelled on the right
        depth = -clique_depth - 0.12 - 0.24 * (i if best_side else k - 1 - i)
        xt = x(1) + 0.35 if best_side else x(k) - 0.35
        c = BLUE if m == "bpso" else INK
        ax.plot([x(r), x(r), xt], [y0, depth, depth], color=c, lw=0.8)
        ax.text(xt + (0.08 if best_side else -0.08), depth, f"{_label(m)} ({r:.2f})",
                ha="left" if best_side else "right", va="center", fontsize=7, color=c)  # fmt: skip
    ax.set_xlim(x(k) - 3.2, x(1) + 3.2)
    ax.set_ylim(-clique_depth - 0.3 - 0.24 * half, 0.9)
    return _save(fig, out)

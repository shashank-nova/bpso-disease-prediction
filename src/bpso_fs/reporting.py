"""Result aggregation and export: LaTeX (booktabs) tables for IEEEtran, plus CSV and Markdown.

Every number written here is computed from the files produced by
:mod:`bpso_fs.experiment`; nothing is typed in by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    METRIC_LABELS,
    friedman_nemenyi,
    kuncheva_index,
    mean_pairwise_jaccard,
    selection_frequency,
    wilcoxon_vs_reference,
)
from .experiment import METHOD_LABELS
from .models import CLASSIFIER_LABELS

MAIN_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
DATASET_LABELS = {
    "breast_cancer": "Breast Cancer (WDBC)",
    "pima_diabetes": "Pima Diabetes",
    "heart_cleveland": "Heart Disease (Cleveland)",
    "ilpd": "Indian Liver Patient",
    "ckd": "Chronic Kidney Disease",
}


# --------------------------------------------------------------------------- loading


def load_results(ds_dir: Path) -> dict:
    """Load every output file for one dataset folder."""
    ds_dir = Path(ds_dir)
    out: dict = {"name": ds_dir.name}
    out["raw"] = pd.read_csv(ds_dir / "raw_results.csv")
    out["selections"] = pd.read_csv(ds_dir / "selections.csv", dtype={"mask": str})
    out["summary"] = json.loads((ds_dir / "data_summary.json").read_text())
    out["meta"] = json.loads((ds_dir / "run_metadata.json").read_text())
    for key, fname in (("history", "history.csv"), ("sweep", "alpha_sweep_results.csv")):
        p = ds_dir / fname
        out[key] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    p = ds_dir / "predictions.csv.gz"
    out["preds"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return out


def discover(results_root: Path) -> list[dict]:
    """All dataset folders under results_root that have finished results."""
    return [load_results(d) for d in sorted(Path(results_root).iterdir())
            if d.is_dir() and (d / "raw_results.csv").exists()]  # fmt: skip


def report_classifier(res: dict, setting: str = "auto", reference: str = "bpso") -> str:
    """Classifier used for the headline tables: fixed name, or (auto) the one with the best
    mean ROC-AUC for the reference method on this dataset."""
    raw = res["raw"]
    if setting != "auto":
        return setting
    r = raw[raw["method"] == reference]
    if r.empty:
        r = raw
    return str(r.groupby("classifier")["roc_auc"].mean().idxmax())


def method_order(methods) -> list[str]:
    order = list(METHOD_LABELS)
    return [m for m in order if m in set(methods)] + sorted(set(methods) - set(order))


# --------------------------------------------------------------------------- latex helpers


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}  # fmt: skip
    return "".join(rep.get(c, c) for c in str(s))


def pm(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def latex_table(df: pd.DataFrame, caption: str, label: str, colfmt: str | None = None,
                notes: str | None = None, wide: bool = False, escape_cells: bool = False) -> str:  # fmt: skip
    """booktabs table; cells may already contain LaTeX (escape_cells=False)."""
    env = "table*" if wide else "table"
    colfmt = colfmt or ("l" + "c" * (df.shape[1] - 1))
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{colfmt}}}",
             "\\toprule", " & ".join(f"\\textbf{{{tex_escape(c)}}}" for c in df.columns) + " \\\\",
             "\\midrule"]  # fmt: skip
    for _, row in df.iterrows():
        cells = [tex_escape(v) if escape_cells else str(v) for v in row.tolist()]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def _strip_tex(s: str) -> str:
    return (str(s).replace("$\\pm$", "±").replace("\\textbf{", "").replace("}", "")
            .replace("\\_", "_").replace("\\%", "%").replace("$", ""))  # fmt: skip


def write_table(df_tex: pd.DataFrame, df_plain: pd.DataFrame, stem: Path, **kw) -> None:
    """Write <stem>.tex, <stem>.csv and <stem>.md."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".tex").write_text(latex_table(df_tex, **kw))
    df_plain.to_csv(stem.with_suffix(".csv"), index=False)
    md = df_plain.copy().astype(str)
    header = (
        "| " + " | ".join(md.columns) + " |\n| " + " | ".join("---" for _ in md.columns) + " |\n"
    )
    body = "".join("| " + " | ".join(r) + " |\n" for r in md.values.tolist())
    stem.with_suffix(".md").write_text(header + body)


# --------------------------------------------------------------------------- aggregates


def summarize(raw: pd.DataFrame, classifier: str, metrics=MAIN_METRICS) -> pd.DataFrame:
    """Mean and std of each metric per method (over seeds x folds)."""
    r = raw[raw["classifier"] == classifier]
    agg = r.groupby("method")[list(metrics) + ["n_selected"]].agg(["mean", "std"])
    return agg.loc[method_order(agg.index)]


def stats_table(raw: pd.DataFrame, classifier: str, reference: str = "bpso",
                metrics=("f1", "roc_auc")) -> pd.DataFrame:  # fmt: skip
    r = raw[raw["classifier"] == classifier]
    frames = [wilcoxon_vs_reference(r, reference, m) for m in metrics]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stability(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in selections.groupby("method"):
        masks = [np.array([int(c) for c in s], dtype=bool) for s in g["mask"].astype(str)]
        rows.append(
            {
                "method": m,
                "jaccard": mean_pairwise_jaccard(masks),
                "kuncheva": kuncheva_index(masks),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def bpso_frequency(res: dict) -> pd.Series:
    sel = res["selections"]
    s = sel[sel["method"] == "bpso"]
    masks = [np.array([int(c) for c in m], dtype=bool) for m in s["mask"].astype(str)]
    return selection_frequency(masks, res["summary"]["feature_names"])


# --------------------------------------------------------------------------- all tables


def make_all_tables(
    results: list[dict], tables_dir: Path, cfg_reporting: dict | None = None
) -> list[Path]:
    """Generate every table; returns the .tex paths written."""
    cfg_reporting = cfg_reporting or {}
    tables_dir = Path(tables_dir)
    ref = cfg_reporting.get("reference_method", "bpso")
    written: list[Path] = []

    def out(stem: str) -> Path:
        p = tables_dir / stem
        written.append(p.with_suffix(".tex"))
        return p

    # 1. dataset summary ----------------------------------------------------
    rows = []
    for res in results:
        s = res["summary"]
        rows.append({
            "Dataset": DATASET_LABELS.get(s["name"], s["name"]),
            "Samples": s["n_samples"],
            "Features": s["n_features"],
            "Positive (disease)": f"{s['n_positive']} ({100 * s['positive_rate']:.1f}%)",
            "Missing cells": s["n_missing_cells"],
            "Rows w/ missing": s["n_rows_with_missing"],
        })  # fmt: skip
    df = pd.DataFrame(rows)
    write_table(df.map(tex_escape), df, out("dataset_summary"), caption="Summary of the datasets used.",
                label="tab:datasets", colfmt="lrrrrr")  # fmt: skip

    # 2. BPSO hyper-parameters actually used -------------------------------
    cfg = results[0]["meta"]["config"]
    b, f, e = cfg["bpso"], cfg["fitness"], cfg["experiment"]
    inertia = (f"linear {b['w_max']} $\\rightarrow$ {b['w_min']}" if b.get("inertia", "linear") == "linear"
               else f"constant {b['w']}")  # fmt: skip
    hp = [
        ("Swarm size (particles)", b["n_particles"]),
        ("Iterations $T$", b["n_iterations"]),
        ("Inertia weight $w$", inertia),
        ("Acceleration $c_1$, $c_2$", f"{b['c1']}, {b['c2']}"),
        ("Velocity clamp $v_{\\max}$", b["v_max"]),
        (
            "Transfer function",
            (
                "S-shaped (sigmoid)"
                if b["transfer_function"] == "sigmoid"
                else "V-shaped $|\\tanh(v)|$"
            ),
        ),
        ("Initial bit probability", b["init_prob"]),
        ("Empty-subset handling", b["empty_subset"]),
        ("Fitness weight $\\alpha$", f["alpha"]),
        (
            "Wrapper classifier in fitness",
            {"logreg": "Logistic regression", "knn": "5-NN"}[f["estimator"]],
        ),
        ("Inner CV folds / score $P(z)$", f"{f['inner_folds']} / {tex_escape(f['metric'])}"),
        ("Outer CV", f"stratified {e['outer_folds']}-fold $\\times$ {e['n_seeds']} seeds"),
    ]
    df_tex = pd.DataFrame(hp, columns=["Parameter", "Value"])
    df_plain = df_tex.map(_strip_tex)
    write_table(df_tex, df_plain, out("bpso_hyperparameters"), caption="BPSO and evaluation settings used in the experiments.",
                label="tab:bpso_params", colfmt="ll")  # fmt: skip

    # 3. main results per dataset ------------------------------------------
    all_cls_rows = []
    for res in results:
        name = res["name"]
        clf = report_classifier(res, cfg_reporting.get("report_classifier", "auto"), ref)
        res["report_classifier"] = clf
        summ = summarize(res["raw"], clf)
        tex_rows, plain_rows = [], []
        best = {m: summ[(m, "mean")].max() for m in MAIN_METRICS}
        for method in summ.index:
            tr = {"Method": tex_escape(METHOD_LABELS.get(method, method))}
            pr = {"Method": METHOD_LABELS.get(method, method)}
            ns = summ.loc[method, ("n_selected", "mean")], summ.loc[method, ("n_selected", "std")]
            tr["\\#Feat."] = f"{ns[0]:.1f}"
            pr["#Features"] = f"{ns[0]:.1f} ± {ns[1]:.1f}"
            for met in MAIN_METRICS:
                mu, sd = summ.loc[method, (met, "mean")], summ.loc[method, (met, "std")]
                cell = pm(mu, sd)
                if np.isclose(mu, best[met]):
                    cell = f"\\textbf{{{mu:.3f}}} $\\pm$ {sd:.3f}"
                tr[METRIC_LABELS[met]] = cell
                pr[METRIC_LABELS[met]] = f"{mu:.4f} ± {sd:.4f}"
            tex_rows.append(tr)
            plain_rows.append(pr)
        df_tex = pd.DataFrame(tex_rows).rename(columns={"\\#Feat.": "#Feat."})
        lbl = DATASET_LABELS.get(name, name)
        e2 = res["meta"]
        write_table(
            df_tex, pd.DataFrame(plain_rows), out(f"main_results_{name}"),
            caption=(f"Test performance on {tex_escape(lbl)} with {tex_escape(CLASSIFIER_LABELS.get(clf, clf))} "
                     f"(mean $\\pm$ std over {len(e2['seeds'])} seeds $\\times$ {e2['outer_folds']} outer folds). Best mean in bold."),
            label=f"tab:results_{name}", colfmt="lr" + "c" * len(MAIN_METRICS), wide=True,
        )  # fmt: skip
        # every classifier, primary metrics (CSV/MD only + compact tex)
        for c in sorted(res["raw"]["classifier"].unique()):
            s2 = summarize(res["raw"], c, metrics=["f1", "roc_auc", "accuracy"])
            for method in s2.index:
                all_cls_rows.append({
                    "dataset": name, "classifier": c, "method": method,
                    "n_selected_mean": s2.loc[method, ("n_selected", "mean")],
                    **{f"{m}_{k}": s2.loc[method, (m, k)] for m in ("accuracy", "f1", "roc_auc") for k in ("mean", "std")},
                })  # fmt: skip
    allc = pd.DataFrame(all_cls_rows)
    allc.to_csv(tables_dir / "results_all_classifiers.csv", index=False)
    # compact: F1 of BPSO vs All per classifier per dataset
    comp_rows_tex, comp_rows_plain = [], []
    for res in results:
        a = allc[allc["dataset"] == res["name"]]
        for c in sorted(a["classifier"].unique()):
            row_t = {"Dataset": tex_escape(DATASET_LABELS.get(res["name"], res["name"])),
                     "Classifier": tex_escape(CLASSIFIER_LABELS.get(c, c))}  # fmt: skip
            row_p = dict(
                row_t,
                Dataset=DATASET_LABELS.get(res["name"], res["name"]),
                Classifier=CLASSIFIER_LABELS.get(c, c),
            )
            for m in ("all", "bpso", "ga"):
                q = a[(a["classifier"] == c) & (a["method"] == m)]
                if q.empty:
                    continue
                q = q.iloc[0]
                row_t[f"{METHOD_LABELS[m]} F1"] = pm(q["f1_mean"], q["f1_std"])
                row_t[f"{METHOD_LABELS[m]} AUC"] = pm(q["roc_auc_mean"], q["roc_auc_std"])
                row_p[f"{METHOD_LABELS[m]} F1"] = f"{q['f1_mean']:.4f} ± {q['f1_std']:.4f}"
                row_p[f"{METHOD_LABELS[m]} AUC"] = (
                    f"{q['roc_auc_mean']:.4f} ± {q['roc_auc_std']:.4f}"
                )
            comp_rows_tex.append(row_t)
            comp_rows_plain.append(row_p)
    dft = pd.DataFrame(comp_rows_tex)
    write_table(dft, pd.DataFrame(comp_rows_plain), out("classifier_comparison"),
                caption="F1 and ROC-AUC per classifier: all features vs.\\ BPSO vs.\\ GA subsets (mean $\\pm$ std).",
                label="tab:classifiers", colfmt="ll" + "c" * (dft.shape[1] - 2), wide=True)  # fmt: skip

    # 4. subset sizes, reduction, stability ---------------------------------
    rows_t, rows_p = [], []
    for res in results:
        d = res["summary"]["n_features"]
        sel = res["selections"]
        st = stability(sel)
        for m in method_order(sel["method"].unique()):
            g = sel[sel["method"] == m]["n_selected"]
            red = 100 * (1 - g.mean() / d)
            jac = st.loc[m, "jaccard"] if m in st.index else np.nan
            base = {
                "Dataset": DATASET_LABELS.get(res["name"], res["name"]),
                "Method": METHOD_LABELS.get(m, m),
            }
            rows_p.append({**base, "d": d, "Selected (mean ± std)": f"{g.mean():.2f} ± {g.std():.2f}",
                           "Reduction %": f"{red:.1f}", "Jaccard stability": f"{jac:.3f}"})  # fmt: skip
            rows_t.append({"Dataset": tex_escape(base["Dataset"]), "Method": tex_escape(base["Method"]), "$d$": d,
                           "Selected": pm(g.mean(), g.std(), 2), "Reduction (\\%)": f"{red:.1f}",
                           "Jaccard": "--" if pd.isna(jac) else f"{jac:.3f}"})  # fmt: skip
    write_table(pd.DataFrame(rows_t), pd.DataFrame(rows_p), out("subset_sizes"),
                caption="Selected subset size, dimensionality reduction and selection stability "
                        "(mean pairwise Jaccard similarity of the subsets across runs; 1 = identical).",
                label="tab:subset_sizes", colfmt="llrccc", wide=True)  # fmt: skip

    # 5. most frequently selected features (BPSO) ---------------------------
    rows_t, rows_p = [], []
    for res in results:
        if not (res["selections"]["method"] == "bpso").any():
            continue
        fq = bpso_frequency(res).head(10)
        for rank, (feat, v) in enumerate(fq.items(), 1):
            ds = DATASET_LABELS.get(res["name"], res["name"])
            rows_p.append(
                {"Dataset": ds, "Rank": rank, "Feature": feat, "Selection frequency": f"{v:.3f}"}
            )
            rows_t.append({"Dataset": tex_escape(ds) if rank == 1 else "", "Rank": rank,
                           "Feature": f"\\texttt{{{tex_escape(feat)}}}", "Frequency": f"{v:.2f}"})  # fmt: skip
    write_table(pd.DataFrame(rows_t), pd.DataFrame(rows_p), out("top_features"),
                caption="Features most frequently selected by BPSO across all outer folds and seeds (top 10).",
                label="tab:top_features", colfmt="lrlr")  # fmt: skip

    # 6. statistical tests ---------------------------------------------------
    rows_t, rows_p, all_stats = [], [], []
    for res in results:
        clf = res["report_classifier"]
        st = stats_table(res["raw"], clf, ref)
        for c in sorted(res["raw"]["classifier"].unique()):
            s_all = stats_table(res["raw"], c, ref)
            if not s_all.empty:
                s_all.insert(0, "classifier", c)
                s_all.insert(0, "dataset", res["name"])
                all_stats.append(s_all)
        if st.empty:
            continue
        for other in method_order(st["other"].unique()):
            row_p = {"Dataset": DATASET_LABELS.get(res["name"], res["name"]), "Classifier": clf,
                     "BPSO vs.": METHOD_LABELS.get(other, other)}  # fmt: skip
            row_t = {
                "Dataset": tex_escape(row_p["Dataset"]),
                "BPSO vs.": tex_escape(row_p["BPSO vs."]),
            }
            for met in ("f1", "roc_auc"):
                q = st[(st["other"] == other) & (st["metric"] == met)]
                if q.empty:
                    continue
                q = q.iloc[0]
                wtl = f"{q['wins']}/{q['ties']}/{q['losses']}"
                ml = METRIC_LABELS[met]
                p = q["p_holm"]
                p_s = "--" if pd.isna(p) else (f"{p:.3f}" if p >= 0.001 else "$<$0.001")
                if not pd.isna(p) and p < 0.05:
                    p_s = f"\\textbf{{{p_s}}}"
                row_t[f"$\\Delta${ml}"] = f"{q['mean_ref'] - q['mean_other']:+.3f}"
                row_t[f"{ml} W/T/L"] = wtl
                row_t[f"{ml} $p_\\mathrm{{Holm}}$"] = p_s
                row_p[f"Δ{ml} (BPSO − other)"] = f"{q['mean_ref'] - q['mean_other']:+.4f}"
                row_p[f"{ml} W/T/L"] = wtl
                row_p[f"{ml} p (Holm)"] = "nan" if pd.isna(p) else f"{p:.4g}"
            rows_t.append(row_t)
            rows_p.append(row_p)
    if rows_t:
        dft = pd.DataFrame(rows_t)
        write_table(dft, pd.DataFrame(rows_p), out("statistical_tests"),
                    caption="Paired Wilcoxon signed-rank tests, BPSO vs.\\ each baseline (pairs = seed $\\times$ fold, "
                            "headline classifier per dataset). $\\Delta$ = mean difference (BPSO $-$ baseline); "
                            "W/T/L = BPSO wins/ties/losses; $p$ values Holm-adjusted per dataset and metric; bold: $p<0.05$.",
                    label="tab:stats", colfmt="ll" + "c" * (dft.shape[1] - 2), wide=True)  # fmt: skip
    if all_stats:
        pd.concat(all_stats, ignore_index=True).to_csv(
            tables_dir / "statistical_tests_all_classifiers.csv", index=False
        )

    # Friedman across datasets x classifiers --------------------------------
    pm_metric = cfg_reporting.get("primary_metric", "f1")
    blocks = []
    for res in results:
        g = res["raw"].groupby(["classifier", "method"])[pm_metric].mean().unstack("method")
        g.index = [f"{res['name']}|{c}" for c in g.index]
        blocks.append(g)
    fr = None
    if blocks:
        B = pd.concat(blocks)
        B = B[[m for m in method_order(B.columns)]]
        fr = friedman_nemenyi(B)
        fr["metric"] = pm_metric
        fr["blocks"] = list(B.index)
        (tables_dir / "friedman.json").write_text(json.dumps(fr, indent=2, default=float))
        rows_t = [
            {"Method": tex_escape(METHOD_LABELS.get(m, m)), "Average rank": f"{r:.2f}"}
            for m, r in fr["avg_ranks"].items()
        ]
        rows_p = [
            {"Method": METHOD_LABELS.get(m, m), "Average rank": f"{r:.3f}"}
            for m, r in fr["avg_ranks"].items()
        ]
        write_table(pd.DataFrame(rows_t), pd.DataFrame(rows_p), out("friedman_ranks"),
                    caption=(f"Average Friedman ranks by {METRIC_LABELS[pm_metric]} over {fr['n_blocks']} "
                             f"dataset--classifier blocks (1 = best). Friedman $\\chi^2$ = {fr['statistic']:.2f}, "
                             f"$p$ = {fr['p_value']:.3g}; Nemenyi CD = {fr['cd']:.2f}."),
                    label="tab:friedman", colfmt="lr")  # fmt: skip

    # 7. runtime -----------------------------------------------------------
    rows_t, rows_p = [], []
    for res in results:
        sel = res["selections"]
        for m in method_order(sel["method"].unique()):
            g = sel[sel["method"] == m]
            ev = (
                g["n_evaluations"].mean()
                if "n_evaluations" in g and g["n_evaluations"].notna().any()
                else np.nan
            )
            ds = DATASET_LABELS.get(res["name"], res["name"])
            rows_p.append({"Dataset": ds, "Method": METHOD_LABELS.get(m, m),
                           "Selection time s (mean ± std)": f"{g['selection_time'].mean():.3f} ± {g['selection_time'].std():.3f}",
                           "Unique subsets evaluated": "" if pd.isna(ev) else f"{ev:.0f}"})  # fmt: skip
            rows_t.append({"Dataset": tex_escape(ds), "Method": tex_escape(METHOD_LABELS.get(m, m)),
                           "Selection time (s)": pm(g["selection_time"].mean(), g["selection_time"].std(), 2),
                           "Subsets evaluated": "--" if pd.isna(ev) else f"{ev:.0f}"})  # fmt: skip
    hw = results[0]["meta"]["library_versions"]
    write_table(pd.DataFrame(rows_t), pd.DataFrame(rows_p), out("runtime"),
                caption="Feature-selection wall-clock time per outer training fold (includes fitting the "
                        f"preprocessing). Measured on {tex_escape(hw.get('cpu_count', '?'))} CPU cores, one job per core.",
                label="tab:runtime", colfmt="llrr", wide=True)  # fmt: skip

    # alpha sweep (CSV/MD + tex) ---------------------------------------------
    rows_t, rows_p = [], []
    for res in results:
        sw = res["sweep"]
        if sw.empty:
            continue
        clf = res["report_classifier"]
        s = sw[sw["classifier"] == clf].groupby("alpha")
        for a, g in s:
            ds = DATASET_LABELS.get(res["name"], res["name"])
            rows_p.append({"Dataset": ds, "alpha": a, "Classifier": clf,
                           "Selected": f"{g['n_selected'].mean():.2f} ± {g['n_selected'].std():.2f}",
                           "F1": f"{g['f1'].mean():.4f} ± {g['f1'].std():.4f}",
                           "ROC-AUC": f"{g['roc_auc'].mean():.4f} ± {g['roc_auc'].std():.4f}"})  # fmt: skip
            rows_t.append({"Dataset": tex_escape(ds), "$\\alpha$": f"{a:.2f}",
                           "Selected": pm(g["n_selected"].mean(), g["n_selected"].std(), 2),
                           "F1": pm(g["f1"].mean(), g["f1"].std()), "ROC-AUC": pm(g["roc_auc"].mean(), g["roc_auc"].std())})  # fmt: skip
    if rows_t:
        n_sw = results[0]["meta"]["config"]["alpha_sweep"].get("n_seeds")
        write_table(pd.DataFrame(rows_t), pd.DataFrame(rows_p), out("alpha_sensitivity"),
                    caption=f"Sensitivity of BPSO to the fitness weight $\\alpha$ (headline classifier per dataset; "
                            f"{n_sw} seed(s) $\\times$ all outer folds).",
                    label="tab:alpha", colfmt="lcccc")  # fmt: skip

    # headline classifier record
    (tables_dir / "report_classifiers.json").write_text(
        json.dumps({r["name"]: r["report_classifier"] for r in results}, indent=2)
    )
    return written

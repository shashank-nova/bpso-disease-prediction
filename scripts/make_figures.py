#!/usr/bin/env python
"""Build every paper figure (PDF + PNG) from the files in results/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from bpso_fs import plotting as P  # noqa: E402
from bpso_fs.evaluation import friedman_nemenyi  # noqa: E402
from bpso_fs.reporting import (  # noqa: E402
    bpso_frequency,
    discover,
    method_order,
    report_classifier,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="paper/figures")
    p.add_argument("--report-classifier", default="auto")
    p.add_argument("--primary-metric", default="f1")
    args = p.parse_args()
    results = discover(Path(args.results_dir))
    if not results:
        print(f"No finished results found under {args.results_dir}.")
        return 1
    P.set_style()
    fig_dir = Path(args.figures_dir)
    made = []
    blocks = []
    for res in results:
        name, raw = res["name"], res["raw"]
        d = res["summary"]["n_features"]
        clf = report_classifier(res, args.report_classifier)
        n_seeds = len(res["meta"]["seeds"])
        if not res["history"].empty:
            made += P.plot_convergence(res["history"], fig_dir / f"convergence_{name}")
            made += P.plot_subset_size(res["history"], d, fig_dir / f"subset_size_{name}")
        if (res["selections"]["method"] == "bpso").any():
            made += P.plot_feature_frequency(
                bpso_frequency(res), fig_dir / f"feature_frequency_{name}"
            )
            made += P.plot_confusion(raw, "bpso", clf, n_seeds, fig_dir / f"confusion_bpso_{name}")
        if not res["preds"].empty:
            made += P.plot_roc(res["preds"], clf, fig_dir / f"roc_{name}")
        made += P.plot_method_comparison(
            raw, clf, fig_dir / f"method_comparison_{name}", method_order(raw["method"].unique())
        )
        if not res["sweep"].empty:
            made += P.plot_alpha_sensitivity(
                res["sweep"], clf, d, fig_dir / f"alpha_sensitivity_{name}"
            )
        g = raw.groupby(["classifier", "method"])[args.primary_metric].mean().unstack("method")
        g.index = [f"{name}|{c}" for c in g.index]
        blocks.append(g)
        print(f"{name}: headline classifier = {clf}")
    B = pd.concat(blocks)
    B = B[method_order(B.columns)]
    fr = friedman_nemenyi(B)
    if fr["cd"] == fr["cd"]:  # not NaN
        made += P.plot_cd_diagram(fr["avg_ranks"], fr["cd"], fig_dir / "cd_diagram")
        print(
            "Friedman:",
            json.dumps({k: fr[k] for k in ("statistic", "p_value", "cd", "n_blocks", "k")}),
        )
    for m in made:
        if m.suffix == ".pdf":
            print("wrote", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())

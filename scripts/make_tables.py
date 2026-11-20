#!/usr/bin/env python
"""Build every paper table (.tex + .csv + .md) from the files in results/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpso_fs.reporting import discover, make_all_tables  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--tables-dir", default="paper/tables")
    p.add_argument("--report-classifier", default="auto", help="'auto' or a classifier name")
    p.add_argument("--primary-metric", default="f1")
    args = p.parse_args()
    results = discover(Path(args.results_dir))
    if not results:
        print(
            f"No finished results found under {args.results_dir}. Run scripts/run_experiments.py first."
        )
        return 1
    written = make_all_tables(results, Path(args.tables_dir), {
        "report_classifier": args.report_classifier, "primary_metric": args.primary_metric})  # fmt: skip
    for w in written:
        print("wrote", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

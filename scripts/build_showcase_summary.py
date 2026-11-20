#!/usr/bin/env python
"""Build a compact, versionable metric summary from experiment CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "n_selected",
    "selection_time",
    "fit_time",
    "total_time",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_experiment"))
    parser.add_argument("--output", type=Path, default=Path("assets/experiment_summary.csv"))
    parser.add_argument(
        "--provenance-output", type=Path, default=Path("assets/experiment_provenance.json")
    )
    args = parser.parse_args()

    # key = (dataset, study, method, classifier, metric)
    values: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for dataset_dir in sorted(args.results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for filename, study in (
            ("raw_results.csv", "cross_validation"),
            ("alpha_sweep_results.csv", "alpha_sweep"),
        ):
            source = dataset_dir / filename
            if not source.is_file():
                continue
            with source.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for metric in METRICS:
                        value = row.get(metric, "")
                        if not value:
                            continue
                        try:
                            number = float(value)
                        except ValueError:
                            continue
                        key = (
                            row.get("dataset", dataset_dir.name),
                            study,
                            row["method"],
                            row["classifier"],
                            metric,
                        )
                        values[key].append(number)

    if not values:
        parser.error(f"No experiment metric rows found under {args.results_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = ("dataset", "study", "method", "classifier", "metric", "mean", "std", "n_runs")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (dataset, study, method, classifier, metric), scores in sorted(values.items()):
            writer.writerow(
                {
                    "dataset": dataset,
                    "study": study,
                    "method": method,
                    "classifier": classifier,
                    "metric": metric,
                    "mean": f"{statistics.mean(scores):.8f}",
                    "std": f"{statistics.stdev(scores):.8f}" if len(scores) > 1 else "",
                    "n_runs": len(scores),
                }
            )
    run_metadata = []
    shared_config = None
    for dataset_dir in sorted(args.results_dir.iterdir()):
        metadata_path = dataset_dir / "run_metadata.json"
        if not dataset_dir.is_dir() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shared_config = shared_config or metadata.get("config")
        run_metadata.append(
            {
                "dataset": metadata.get("dataset", dataset_dir.name),
                "timestamp_utc": metadata.get("timestamp_utc"),
                "data_source": metadata.get("data_source"),
                "library_versions": metadata.get("library_versions", {}),
                "wall_time_seconds": metadata.get("wall_time_seconds_this_invocation"),
                "classifiers": metadata.get("classifiers", []),
            }
        )
    if run_metadata:
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_output.write_text(
            json.dumps(
                {"config": shared_config, "datasets": run_metadata}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
    print(f"Wrote {len(values)} aggregated metric rows to {args.output}")
    if run_metadata:
        print(f"Wrote run provenance for {len(run_metadata)} datasets to {args.provenance_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

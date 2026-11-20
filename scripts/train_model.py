#!/usr/bin/env python
"""Train a deployable BPSO-selected model from a built-in dataset or CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from bpso_fs.data.loaders import load_csv_dataset, load_dataset
from bpso_fs.deployment import train_artifact
from bpso_fs.experiment import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reduced.yaml")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--csv", help="custom CSV file")
    parser.add_argument("--target", help="target column for --csv")
    parser.add_argument("--positive-label", help="positive target value")
    parser.add_argument("--name", help="custom dataset name")
    parser.add_argument(
        "--classifier", choices=["logreg", "svm", "rf", "knn", "nb", "xgb"], default="logreg"
    )
    parser.add_argument("--output", default="models/disease_model.joblib")
    parser.add_argument("--force", action="store_true", help="retrain even if the artifact exists")
    args = parser.parse_args()
    if args.csv and not args.target:
        parser.error("--csv requires --target")
    config = load_config(args.config)
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Using existing model artifact: {output.resolve()}")
        return 0
    if args.csv:
        bundle = load_csv_dataset(args.csv, args.target, args.positive_label, name=args.name)
    else:
        bundle = load_dataset(args.dataset)
    metadata = train_artifact(bundle, config, output, args.classifier)
    print(yaml.safe_dump(metadata, sort_keys=False))
    print(f"Saved model artifact: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

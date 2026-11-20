#!/usr/bin/env python
"""Download and cache every dataset; write data summaries to results/<dataset>/data_summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpso_fs.data.loaders import (  # noqa: E402
    DatasetUnavailableError,
    available_datasets,
    load_dataset,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="breast_cancer,pima_diabetes,heart_cleveland",
                   help=f"comma list or 'all' ({', '.join(available_datasets())})")  # fmt: skip
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    names = available_datasets() if args.datasets == "all" else args.datasets.split(",")
    failed = []
    for n in names:
        try:
            b = load_dataset(n.strip())
        except DatasetUnavailableError as exc:
            print(f"[FAILED] {n}\n{exc}\n")
            failed.append(n)
            continue
        s = b.summary()
        out = Path(args.results_dir) / b.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "data_summary.json").write_text(json.dumps(s, indent=2))
        print(f"[ok] {b.name}: {s['n_samples']} x {s['n_features']}, positive rate {s['positive_rate']:.3f}, "
              f"missing cells {s['n_missing_cells']}  (source: {b.source})")  # fmt: skip
    if failed:
        print(f"Datasets that could not be downloaded: {failed} (see the instructions above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end --quick run on breast cancer, then tables and figures."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, cwd=ROOT):
    r = subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=900
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def test_quick_pipeline(tmp_path):
    res, tab, fig = tmp_path / "results", tmp_path / "tables", tmp_path / "figures"
    _run(["scripts/run_experiments.py", "--config", "configs/default.yaml", "--dataset", "breast_cancer",
          "--quick", "--results-dir", str(res), "--n-jobs", "2"])  # fmt: skip
    ds = res / "breast_cancer"
    for f in ("raw_results.csv", "selections.csv", "history.csv", "data_summary.json",
              "run_metadata.json", "predictions.csv.gz", "alpha_sweep_results.csv"):  # fmt: skip
        assert (ds / f).exists(), f

    # resume: a second run loads every task from cache
    r = _run(["scripts/run_experiments.py", "--config", "configs/default.yaml", "--dataset", "breast_cancer",
              "--quick", "--results-dir", str(res), "--n-jobs", "1"])  # fmt: skip
    assert "loaded from cache" in (r.stdout + r.stderr)

    _run(["scripts/make_tables.py", "--results-dir", str(res), "--tables-dir", str(tab)])
    _run(["scripts/make_figures.py", "--results-dir", str(res), "--figures-dir", str(fig)])
    for t in ("dataset_summary", "bpso_hyperparameters", "main_results_breast_cancer", "subset_sizes",
              "top_features", "statistical_tests", "runtime"):  # fmt: skip
        assert (tab / f"{t}.tex").exists(), t
        assert (tab / f"{t}.csv").exists(), t
    for f in ("convergence_breast_cancer", "roc_breast_cancer", "feature_frequency_breast_cancer",
              "method_comparison_breast_cancer", "alpha_sensitivity_breast_cancer", "confusion_bpso_breast_cancer"):  # fmt: skip
        assert (fig / f"{f}.pdf").exists(), f

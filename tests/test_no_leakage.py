"""The preprocessing view, the selector and the final model must only ever see training rows."""

from pathlib import Path

import numpy as np
import pandas as pd

import bpso_fs.experiment as E
from bpso_fs.data.loaders import DatasetBundle
from bpso_fs.experiment import Task, load_config, outer_splits, run_task
from bpso_fs.selectors.base import FeatureSelector

SENTINEL = 1e6


def _bundle():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + rng.normal(scale=0.5, size=n) > 0).astype(int).to_numpy()
    return DatasetBundle("toy", X, y, list(X.columns), [])


class RecordingSelector(FeatureSelector):
    name = "recording"
    seen_rows: list = []

    def _fit(self, X, y):
        RecordingSelector.seen_rows.append((X.shape[0], X.copy()))
        self.support_ = np.ones(X.shape[1], dtype=bool)


def test_selector_and_scaler_fit_on_train_only(monkeypatch):
    b = _bundle()
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", quick=True)
    splits = outer_splits(b.y, [0], 4)
    tr, te = splits[(0, 0)]
    # mark the test rows: if any of them reached the scaler or selector we would see it
    b.X.loc[te, "f1"] = SENTINEL

    recorded_view_index = []
    real_view = E.make_selection_view

    def recording_view(num, cat):
        ct = real_view(num, cat)
        orig_fit = ct.fit

        def fit(X, y=None):
            recorded_view_index.append(np.asarray(X.index))
            return orig_fit(X, y)

        ct.fit = fit
        return ct

    RecordingSelector.seen_rows = []
    monkeypatch.setattr(E, "make_selection_view", recording_view)
    monkeypatch.setattr(E, "make_selector", lambda task, cfg, d: RecordingSelector())
    rec = run_task(b, Task("all", "all", 0, 0, 0.99, None, ("logreg",)), tr, te, cfg)

    # the view (imputer + scaler) was fitted on exactly the training indices
    assert len(recorded_view_index) == 1
    np.testing.assert_array_equal(np.sort(recorded_view_index[0]), np.sort(tr))
    assert set(recorded_view_index[0]).isdisjoint(set(te))
    # the selector saw only training rows, none carrying the sentinel
    n_rows, X_seen = RecordingSelector.seen_rows[0]
    assert n_rows == len(tr)
    assert np.abs(X_seen).max() < 100  # standardised train data, no 1e6 sentinel
    assert rec["n_train"] == len(tr) and rec["n_test"] == len(te)
    assert set(rec["test_idx"]) == set(te.tolist())


def test_outer_splits_are_disjoint_and_paired():
    y = np.r_[np.zeros(50), np.ones(30)].astype(int)
    s1 = outer_splits(y, [0, 1], 5)
    s2 = outer_splits(y, [0, 1], 5)
    for key, (tr, te) in s1.items():
        assert set(tr).isdisjoint(te)
        assert len(tr) + len(te) == len(y)
        np.testing.assert_array_equal(te, s2[key][1])  # same split for every method

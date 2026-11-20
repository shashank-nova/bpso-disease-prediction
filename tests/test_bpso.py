import numpy as np
import pytest
from sklearn.datasets import make_classification

from bpso_fs.fitness import SubsetFitness
from bpso_fs.selectors import BinaryGA, BinaryPSO


@pytest.fixture(scope="module")
def synthetic():
    # 5 informative features first (shuffle=False), then 25 pure-noise features
    X, y = make_classification(
        n_samples=400, n_features=30, n_informative=5, n_redundant=0, n_repeated=0,
        n_clusters_per_class=2, class_sep=1.5, shuffle=False, random_state=0,
    )  # fmt: skip
    return X, y


class RecordingFitness:
    """Fake fitness that records every position it receives."""

    def __init__(self, d):
        self.d = d
        self.seen = []
        self.rng = np.random.default_rng(0)

    def __call__(self, mask):
        self.seen.append(np.asarray(mask).copy())
        return float(mask.sum()) / self.d + self.rng.random() * 1e-3, 0.5


def test_positions_binary_and_velocity_clamped(synthetic):
    X, y = synthetic
    fit = RecordingFitness(X.shape[1])
    s = BinaryPSO(n_particles=8, n_iterations=15, v_max=4.0, fitness=fit, random_state=1).fit(X, y)
    allpos = np.array(fit.seen)
    assert set(np.unique(allpos)) <= {0, 1}
    assert np.all(np.abs(s.final_velocities_) <= 4.0 + 1e-12)
    assert s.support_.dtype == bool and s.support_.shape == (X.shape[1],)


def test_v_shaped_transfer_runs(synthetic):
    X, y = synthetic
    s = BinaryPSO(n_particles=6, n_iterations=5, transfer_function="v_shaped",
                  fitness=RecordingFitness(X.shape[1]), random_state=0).fit(X, y)  # fmt: skip
    assert s.n_selected_ >= 1


def test_empty_subset_repair(synthetic):
    X, y = synthetic
    fit = RecordingFitness(X.shape[1])
    # init_prob=0 makes every initial particle empty -> repair must switch one bit on
    BinaryPSO(n_particles=10, n_iterations=3, init_prob=0.0, fitness=fit, random_state=0).fit(X, y)
    assert all(m.sum() >= 1 for m in fit.seen)


def test_empty_subset_penalize_gets_worst_fitness(synthetic):
    X, y = synthetic
    f = SubsetFitness(X, y, alpha=0.9, cv=3, random_state=0)
    fit, score = f.evaluate(np.zeros(X.shape[1], dtype=bool))
    assert fit == 1.0 and np.isnan(score)


def test_same_seed_same_result(synthetic):
    X, y = synthetic
    kw = dict(n_particles=8, n_iterations=8, fitness_kwargs={"cv": 3})
    a = BinaryPSO(random_state=42, **kw).fit(X, y)
    b = BinaryPSO(random_state=42, **kw).fit(X, y)
    c = BinaryPSO(random_state=7, **kw).fit(X, y)
    np.testing.assert_array_equal(a.get_support(), b.get_support())
    for key in ("best_fitness", "mean_fitness", "best_size", "mean_size"):
        assert a.history_[key] == b.history_[key]
    assert a.best_fitness_ == b.best_fitness_
    assert a.history_["mean_fitness"] != c.history_["mean_fitness"]


def test_global_best_fitness_monotone(synthetic):
    X, y = synthetic
    s = BinaryPSO(n_particles=10, n_iterations=15, fitness_kwargs={"cv": 3}, random_state=3).fit(
        X, y
    )
    bf = np.array(s.history_["best_fitness"])
    assert np.all(np.diff(bf) <= 1e-12)
    assert len(bf) == 16  # initial swarm + 15 iterations


def test_bpso_finds_informative_features(synthetic):
    X, y = synthetic
    s = BinaryPSO(n_particles=20, n_iterations=40, fitness_kwargs={"alpha": 0.9, "cv": 3},
                  random_state=0).fit(X, y)  # fmt: skip
    sel = s.get_support(indices=True)
    informative = set(range(5))
    hits = len(informative & set(sel))
    assert hits >= 4, f"only {hits} of 5 informative features selected: {sel}"
    assert len(sel) <= 10, f"too many noise features kept: {sel}"


def test_ga_same_budget_and_runs(synthetic):
    X, y = synthetic
    g = BinaryGA(pop_size=10, n_generations=10, fitness_kwargs={"cv": 3}, random_state=0).fit(X, y)
    assert g.n_fitness_calls_ == 10 * 11
    assert g.n_selected_ >= 1


def test_inertia_schedule():
    s = BinaryPSO(n_iterations=11, w_max=0.9, w_min=0.4)
    assert s.inertia_at(0) == pytest.approx(0.9)
    assert s.inertia_at(10) == pytest.approx(0.4)
    assert BinaryPSO(inertia="constant", w=0.5).inertia_at(7) == 0.5

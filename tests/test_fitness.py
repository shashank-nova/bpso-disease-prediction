import numpy as np
import pytest
from sklearn.datasets import make_classification

from bpso_fs.fitness import SubsetFitness, fitness_value


@pytest.mark.parametrize(
    "score,n_sel,d,alpha,expected",
    [
        (0.90, 5, 10, 0.99, 0.99 * 0.10 + 0.01 * 0.5),
        (1.00, 10, 10, 0.80, 0.0 + 0.2 * 1.0),
        (0.75, 1, 4, 0.50, 0.5 * 0.25 + 0.5 * 0.25),
        (0.00, 3, 3, 0.99, 0.99 + 0.01),
    ],
)
def test_formula_hand_computed(score, n_sel, d, alpha, expected):
    assert fitness_value(score, n_sel, d, alpha) == pytest.approx(expected)


def test_alpha_one_ignores_subset_size():
    assert fitness_value(0.8, 1, 30, 1.0) == pytest.approx(fitness_value(0.8, 30, 30, 1.0))


def test_alpha_zero_only_size():
    assert fitness_value(0.1, 3, 6, 0.0) == pytest.approx(0.5)


def test_subset_fitness_matches_formula_and_caches():
    X, y = make_classification(n_samples=200, n_features=8, random_state=0)
    f = SubsetFitness(X, y, alpha=0.9, cv=4, metric="accuracy", random_state=0)
    mask = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    fit, score = f.evaluate(mask)
    assert 0 <= score <= 1
    assert fit == pytest.approx(0.9 * (1 - score) + 0.1 * 3 / 8)
    f.evaluate(mask)
    assert f.n_calls == 2 and f.n_evaluations == 1


@pytest.mark.parametrize("metric", ["accuracy", "balanced_accuracy", "f1", "roc_auc"])
def test_metrics_and_knn(metric):
    X, y = make_classification(n_samples=150, n_features=5, random_state=1)
    for est in ("logreg", "knn"):
        _, s = SubsetFitness(X, y, metric=metric, estimator=est, cv=3, random_state=0).evaluate(
            np.ones(5, bool)
        )
        assert 0.0 <= s <= 1.0


def test_invalid_alpha():
    with pytest.raises(ValueError):
        SubsetFitness(np.zeros((10, 2)), np.r_[np.zeros(5), np.ones(5)], alpha=1.5)

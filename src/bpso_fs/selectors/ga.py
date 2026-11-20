"""Binary Genetic Algorithm baseline (same fitness function and budget as BPSO).

Population size = BPSO swarm size and generations = BPSO iterations, so both methods make
the same number of fitness requests: ``pop_size * (n_generations + 1)``.

Operators: tournament selection (size 3), uniform crossover, bit-flip mutation with
probability ``mutation_rate`` (default 1/d), elitism.
"""

from __future__ import annotations

import numpy as np

from ..fitness import SubsetFitness
from .base import FeatureSelector


class BinaryGA(FeatureSelector):
    """Binary genetic algorithm wrapper feature selector."""

    name = "ga"

    def __init__(
        self,
        pop_size: int = 30,
        n_generations: int = 100,
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,
        tournament_size: int = 3,
        n_elite: int = 1,
        init_prob: float = 0.5,
        fitness_kwargs: dict | None = None,
        fitness: SubsetFitness | None = None,
        random_state: int | None = None,
    ) -> None:
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.n_elite = n_elite
        self.init_prob = init_prob
        self.fitness_kwargs = fitness_kwargs or {}
        self.fitness = fitness
        self.random_state = random_state

    def _tournament(self, rng: np.random.Generator, fit: np.ndarray, size: np.ndarray) -> int:
        idx = rng.choice(len(fit), size=self.tournament_size, replace=False)
        return int(idx[np.lexsort((size[idx], fit[idx]))[0]])

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        fobj = self.fitness or SubsetFitness(
            X, y, **{"random_state": self.random_state, **self.fitness_kwargs}
        )
        self.fitness_ = fobj
        pm = self.mutation_rate if self.mutation_rate is not None else 1.0 / d
        n = self.pop_size

        def repair(P: np.ndarray) -> np.ndarray:
            for i in np.flatnonzero(P.sum(axis=1) == 0):
                P[i, rng.integers(d)] = 1
            return P

        def evaluate(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            res = [fobj.evaluate(p.astype(bool)) for p in P]
            return np.array([r[0] for r in res]), np.array([r[1] for r in res])

        pop = repair((rng.random((n, d)) < self.init_prob).astype(np.int8))
        fit, score = evaluate(pop)
        size = pop.sum(axis=1)
        hist: dict[str, list[float]] = {
            k: [] for k in ("best_fitness", "mean_fitness", "best_score", "best_size", "mean_size")
        }

        def best_idx() -> int:
            return int(np.lexsort((size, fit))[0])

        def record() -> None:
            b = best_idx()
            hist["best_fitness"].append(float(fit[b]))
            hist["mean_fitness"].append(float(np.mean(fit)))
            hist["best_score"].append(float(score[b]))
            hist["best_size"].append(int(size[b]))
            hist["mean_size"].append(float(np.mean(size)))

        record()
        for _ in range(self.n_generations):
            order = np.lexsort((size, fit))
            elite = pop[order[: self.n_elite]].copy()
            children = []
            while len(children) < n - self.n_elite:
                a = pop[self._tournament(rng, fit, size)]
                b = pop[self._tournament(rng, fit, size)]
                if rng.random() < self.crossover_rate:
                    m = rng.random(d) < 0.5
                    c1, c2 = np.where(m, a, b), np.where(m, b, a)
                else:
                    c1, c2 = a.copy(), b.copy()
                for c in (c1, c2):
                    flip = rng.random(d) < pm
                    children.append(np.where(flip, 1 - c, c).astype(np.int8))
            pop = repair(np.vstack([elite, np.array(children[: n - self.n_elite])]))
            fit, score = evaluate(pop)
            size = pop.sum(axis=1)
            record()

        b = best_idx()
        self.support_ = pop[b].astype(bool)
        self.best_fitness_ = float(fit[b])
        self.best_score_ = float(score[b])
        self.history_ = hist
        self.n_evaluations_ = fobj.n_evaluations
        self.n_fitness_calls_ = fobj.n_calls

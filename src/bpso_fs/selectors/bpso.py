r"""Binary Particle Swarm Optimization (BPSO) for feature selection.

Each particle :math:`i` has a binary position :math:`x_i \in \{0,1\}^d` (a candidate feature
subset) and a real-valued velocity :math:`v_i \in \mathbb{R}^d`. At iteration :math:`t`:

.. math::

    v_{ij}^{t+1} = w^t v_{ij}^t + c_1 r_{1j} (p_{ij} - x_{ij}^t) + c_2 r_{2j} (g_j - x_{ij}^t),
    \quad v_{ij}^{t+1} \leftarrow \mathrm{clip}(v_{ij}^{t+1}, -v_{\max}, v_{\max})

with :math:`r_{1j}, r_{2j} \sim U(0,1)`, personal best :math:`p_i` and global best :math:`g`.

*S-shaped (sigmoid) transfer* (Kennedy & Eberhart, 1997):

.. math::

    S(v) = \frac{1}{1 + e^{-v}}, \qquad
    x_{ij}^{t+1} = \begin{cases} 1 & \text{if } \rho_{ij} < S(v_{ij}^{t+1}) \\ 0 & \text{otherwise}\end{cases}

*V-shaped transfer* (Mirjalili & Lewis, 2013):

.. math::

    V(v) = |\tanh(v)|, \qquad
    x_{ij}^{t+1} = \begin{cases} 1 - x_{ij}^{t} & \text{if } \rho_{ij} < V(v_{ij}^{t+1}) \\ x_{ij}^{t} & \text{otherwise}\end{cases}

Inertia is constant or decreases linearly:
:math:`w^t = w_{\max} - (w_{\max} - w_{\min})\, t / (T - 1)`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from joblib import Parallel, delayed

from ..fitness import SubsetFitness
from .base import FeatureSelector

FitnessFn = Callable[[np.ndarray], tuple[float, float]]


def sigmoid(v: np.ndarray) -> np.ndarray:
    """S-shaped transfer function 1 / (1 + exp(-v))."""
    return 1.0 / (1.0 + np.exp(-v))


def v_shaped(v: np.ndarray) -> np.ndarray:
    """V-shaped transfer function |tanh(v)|."""
    return np.abs(np.tanh(v))


def _better(f_new: float, n_new: int, f_old: float, n_old: int) -> bool:
    """Lower fitness wins; ties are broken in favour of fewer features."""
    return f_new < f_old or (f_new == f_old and n_new < n_old)


class BinaryPSO(FeatureSelector):
    """Binary PSO wrapper feature selector.

    Args:
        n_particles: swarm size.
        n_iterations: maximum number of iterations T.
        w: constant inertia weight; used when ``inertia='constant'``.
        inertia: 'linear' (w_max -> w_min) or 'constant'.
        w_max, w_min: inertia bounds for the linear schedule.
        c1, c2: cognitive and social acceleration coefficients.
        v_max: velocity clamp.
        transfer_function: 'sigmoid' (S-shaped) or 'v_shaped'.
        init_prob: probability that each bit is 1 at initialisation.
        empty_subset: 'repair' (switch on one random feature) or 'penalize' (worst fitness).
        early_stopping_patience: stop if the global best does not improve for this many
            iterations (None = never).
        fitness_kwargs: passed to :class:`~bpso_fs.fitness.SubsetFitness` when ``fitness`` is None.
        fitness: optional pre-built fitness object/callable returning (fitness, score).
        n_jobs: parallel fitness evaluations per iteration (threads).
        random_state: seed.

    Attributes after ``fit``: ``support_``, ``best_fitness_``, ``best_score_``,
    ``history_`` (dict of per-iteration lists; index 0 = initial swarm),
    ``n_evaluations_`` (unique subsets scored), ``n_fitness_calls_`` (incl. cache hits),
    ``n_iterations_run_``.
    """

    name = "bpso"

    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.7298,
        inertia: str = "linear",
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        v_max: float = 6.0,
        transfer_function: str = "sigmoid",
        init_prob: float = 0.5,
        empty_subset: str = "repair",
        early_stopping_patience: int | None = None,
        fitness_kwargs: dict | None = None,
        fitness: FitnessFn | SubsetFitness | None = None,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        if transfer_function not in ("sigmoid", "v_shaped"):
            raise ValueError("transfer_function must be 'sigmoid' or 'v_shaped'")
        if inertia not in ("linear", "constant"):
            raise ValueError("inertia must be 'linear' or 'constant'")
        if empty_subset not in ("repair", "penalize"):
            raise ValueError("empty_subset must be 'repair' or 'penalize'")
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.inertia = inertia
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.transfer_function = transfer_function
        self.init_prob = init_prob
        self.empty_subset = empty_subset
        self.early_stopping_patience = early_stopping_patience
        self.fitness_kwargs = fitness_kwargs or {}
        self.fitness = fitness
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ helpers
    def inertia_at(self, t: int) -> float:
        """Inertia weight used to compute the velocity at iteration t (0-based)."""
        if self.inertia == "constant":
            return self.w
        if self.n_iterations <= 1:
            return self.w_max
        return self.w_max - (self.w_max - self.w_min) * t / (self.n_iterations - 1)

    def _repair(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.empty_subset == "repair":
            empty = np.flatnonzero(X.sum(axis=1) == 0)
            for i in empty:
                X[i, rng.integers(X.shape[1])] = 1
        return X

    def _evaluate(self, fit_fn: FitnessFn, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        masks = [X[i].astype(bool) for i in range(X.shape[0])]
        if self.n_jobs == 1:
            res = [fit_fn(m) for m in masks]
        else:
            res = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(fit_fn)(m) for m in masks)
        f = np.array([r[0] for r in res], dtype=float)
        s = np.array([r[1] for r in res], dtype=float)
        return f, s

    # ------------------------------------------------------------------ main loop
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        if self.fitness is None:
            kw = {"random_state": self.random_state, **self.fitness_kwargs}
            fitness_obj = SubsetFitness(X, y, **kw)
        else:
            fitness_obj = self.fitness
        fit_fn: FitnessFn = (
            fitness_obj.evaluate if isinstance(fitness_obj, SubsetFitness) else fitness_obj
        )
        self.fitness_ = fitness_obj

        n = self.n_particles
        pos = (rng.random((n, d)) < self.init_prob).astype(np.int8)
        pos = self._repair(pos, rng)
        vel = rng.uniform(-self.v_max, self.v_max, size=(n, d))

        fit, score = self._evaluate(fit_fn, pos)
        sizes = pos.sum(axis=1)
        pbest, pbest_fit, pbest_score = pos.copy(), fit.copy(), score.copy()
        pbest_size = sizes.copy()
        g = int(np.lexsort((pbest_size, pbest_fit))[0])
        gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        gbest_score, gbest_size = float(pbest_score[g]), int(pbest_size[g])

        hist: dict[str, list[float]] = {
            k: []
            for k in (
                "best_fitness",
                "mean_fitness",
                "best_score",
                "best_size",
                "mean_size",
                "inertia",
            )
        }

        def record(w_t: float) -> None:
            hist["best_fitness"].append(gbest_fit)
            hist["mean_fitness"].append(float(np.mean(fit)))
            hist["best_score"].append(gbest_score)
            hist["best_size"].append(gbest_size)
            hist["mean_size"].append(float(np.mean(sizes)))
            hist["inertia"].append(w_t)

        record(float("nan"))
        no_improve = 0
        t_run = 0
        for t in range(self.n_iterations):
            w_t = self.inertia_at(t)
            r1 = rng.random((n, d))
            r2 = rng.random((n, d))
            vel = w_t * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest[None, :] - pos)
            np.clip(vel, -self.v_max, self.v_max, out=vel)

            rho = rng.random((n, d))
            if self.transfer_function == "sigmoid":
                pos = (rho < sigmoid(vel)).astype(np.int8)
            else:
                flip = rho < v_shaped(vel)
                pos = np.where(flip, 1 - pos, pos).astype(np.int8)
            pos = self._repair(pos, rng)

            fit, score = self._evaluate(fit_fn, pos)
            sizes = pos.sum(axis=1)
            for i in range(n):
                if _better(fit[i], sizes[i], pbest_fit[i], pbest_size[i]):
                    pbest[i] = pos[i]
                    pbest_fit[i], pbest_score[i], pbest_size[i] = fit[i], score[i], sizes[i]
            g = int(np.lexsort((pbest_size, pbest_fit))[0])
            if _better(pbest_fit[g], pbest_size[g], gbest_fit, gbest_size):
                gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
                gbest_score, gbest_size = float(pbest_score[g]), int(pbest_size[g])
                no_improve = 0
            else:
                no_improve += 1
            record(w_t)
            t_run = t + 1
            if self.early_stopping_patience and no_improve >= self.early_stopping_patience:
                break

        self.support_ = gbest.astype(bool)
        self.best_fitness_ = gbest_fit
        self.best_score_ = gbest_score
        self.history_ = hist
        self.n_iterations_run_ = t_run
        self.n_evaluations_ = getattr(fitness_obj, "n_evaluations", None)
        self.n_fitness_calls_ = getattr(fitness_obj, "n_calls", None)
        self.final_positions_ = pos.astype(bool)
        self.final_velocities_ = vel

"""Traveling Salesman Problem (TSP) benchmark problems.

This module provides real-world combinatorial optimization problems formulated
as continuous single-task optimization via **random keys encoding**. Each
decision variable x_i in [0, 1] represents the priority of city i; the
visiting order (permutation) is obtained by sorting these priorities (argsort).

This encoding allows standard continuous evolutionary algorithms (GA, DE, PSO,
CMA-ES, etc.) to solve TSP without any problem-specific operator.

The objective is the **total Euclidean tour length** (round trip), to be
minimized.

Problems are ordered from easy to hard (by number of cities):

+-----+---------------------+--------+------------------+
| P   | Instance            | Cities | Description      |
+=====+=====================+========+==================+
| P1  | Random-20           | 20     | Random layout    |
| P2  | Circle-30           | 30     | Circular layout  |
| P3  | Clustered-50        | 50     | 5 clusters       |
| P4  | Random-50           | 50     | Random layout    |
| P5  | Random-100          | 100    | Random layout    |
| P6  | Random-200          | 200    | Random layout    |
+-----+---------------------+--------+------------------+

References
----------
    [1] Reinelt, G. (1991). "TSPLIB -- A Traveling Salesman Problem Library." ORSA Journal on Computing, 3(4), 376-384.
    [2] Bean, J.C. (1994). "Genetic Algorithms and Random Keys for Sequencing and Optimization." ORSA Journal on Computing, 6(2), 154-160.
    [3] Applegate, D.L., Bixby, R.E., Chvatal, V., and Cook, W.J. (2006). "The Traveling Salesman Problem: A Computational Study." Princeton University Press.
"""

import numpy as np
import matplotlib.pyplot as plt
from ddmtolab.Methods.mtop import MTOP


def _tour_length(coords, perm):
    """Compute round-trip Euclidean tour length for a given permutation."""
    ordered = coords[perm]
    diffs = np.diff(ordered, axis=0, append=ordered[:1])
    return np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))


def _make_tsp_objective(coords):
    """Create a vectorized TSP objective using random keys encoding."""
    n_cities = coords.shape[0]

    def objective(x):
        x = np.atleast_2d(x)
        results = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            perm = np.argsort(x[i, :n_cities])
            results[i, 0] = _tour_length(coords, perm)
        return results

    return objective, n_cities


def _generate_random_cities(n_cities, seed=42):
    """Generate random city coordinates in [0, 100] x [0, 100]."""
    rng = np.random.RandomState(seed)
    return rng.rand(n_cities, 2) * 100.0


def _generate_circle_cities(n_cities, radius=50.0):
    """Generate cities equally spaced on a circle."""
    angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
    coords = np.column_stack([
        radius + radius * np.cos(angles),
        radius + radius * np.sin(angles)
    ])
    return coords


def _generate_clustered_cities(n_cities, n_clusters=5, seed=42):
    """Generate cities in clusters."""
    rng = np.random.RandomState(seed)
    cities_per_cluster = n_cities // n_clusters
    remainder = n_cities - cities_per_cluster * n_clusters

    centers = rng.rand(n_clusters, 2) * 80.0 + 10.0
    coords_list = []

    for k in range(n_clusters):
        nc = cities_per_cluster + (1 if k < remainder else 0)
        cluster = centers[k] + rng.randn(nc, 2) * 5.0
        coords_list.append(cluster)

    return np.vstack(coords_list)


class TSP:
    """
    Traveling Salesman Problem (TSP) benchmark suite for single-task
    optimization.

    Decision variables are **random keys** in [0, 1]: continuous values whose
    argsort defines the visiting permutation. This allows any continuous
    optimizer (GA, DE, PSO, CMA-ES, etc.) to solve TSP directly.

    Objective (minimize): total Euclidean round-trip tour length.

    Parameters
    ----------
    seed : int, optional
        Random seed for city coordinate generation (default 42).

    References
    ----------
    .. [1] Reinelt, G. (1991). "TSPLIB -- A Traveling Salesman Problem
           Library." ORSA Journal on Computing, 3(4), 376-384.
    .. [2] Bean, J.C. (1994). "Genetic Algorithms and Random Keys for
           Sequencing and Optimization." ORSA Journal on Computing, 6(2),
           154-160.
    .. [3] Applegate, D.L., Bixby, R.E., Chvatal, V., and Cook, W.J.
           (2006). "The Traveling Salesman Problem: A Computational Study."
           Princeton University Press.
    """

    problem_information = {
        'n_cases': 6,
        'n_tasks': '1',
        'n_dims': '[20, 200]',
        'n_objs': '1',
        'n_cons': '0',
        'type': 'real_world',
    }

    _PROBLEM_NAMES = {
        1: 'Random-20', 2: 'Circle-30', 3: 'Clustered-50',
        4: 'Random-50', 5: 'Random-100', 6: 'Random-200',
    }

    def __init__(self, seed=42):
        self.seed = seed

    def _get_coords(self, problem_id):
        """Return city coordinates for a given problem ID (1-6)."""
        generators = {
            1: lambda: _generate_random_cities(20, seed=self.seed),
            2: lambda: _generate_circle_cities(30),
            3: lambda: _generate_clustered_cities(50, n_clusters=5, seed=self.seed),
            4: lambda: _generate_random_cities(50, seed=self.seed),
            5: lambda: _generate_random_cities(100, seed=self.seed),
            6: lambda: _generate_random_cities(200, seed=self.seed),
        }
        if problem_id not in generators:
            raise ValueError(f"Invalid problem_id={problem_id}, must be 1-6.")
        return generators[problem_id]()

    def _build_problem(self, coords):
        """Build a single-task MTOP from city coordinates."""
        objective, n_cities = _make_tsp_objective(coords)
        problem = MTOP()
        problem.add_task(objective, dim=n_cities, lower_bound=0.0, upper_bound=1.0)
        return problem

    def plot_tour(self, problem_id, decision_vars, title=None,
                  save_path=None, figsize=(8, 4), show=True) -> float:
        """
        Plot the TSP tour defined by decision variables (random keys).

        This is a visualisation helper, not a benchmark problem: unlike the
        ``P1``-``P6`` methods it does not return an :class:`MTOP`, so problem
        discovery skips it.

        Parameters
        ----------
        problem_id : int
            Problem index (1-6).
        decision_vars : np.ndarray, shape (n_cities,)
            Decision variables in [0, 1]. The tour is obtained via argsort.
        title : str, optional
            Figure title. If None, auto-generated from problem name and tour
            length.
        save_path : str, optional
            If provided, save the figure to this path.
        figsize : tuple, optional
            Figure size (default (8, 8)).
        show : bool, optional
            Whether to call plt.show() (default True).

        Returns
        -------
        float
            The total tour length.
        """
        coords = self._get_coords(problem_id)
        decision_vars = np.asarray(decision_vars).flatten()
        perm = np.argsort(decision_vars[:coords.shape[0]])
        length = _tour_length(coords, perm)

        ordered = coords[perm]
        # Close the loop
        loop = np.vstack([ordered, ordered[:1]])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # Tour edges
        ax.plot(loop[:, 0], loop[:, 1], 'o-', color='#1f77b4',
                linewidth=0.6, markersize=2, markerfacecolor='#d62728',
                markeredgecolor='white', markeredgewidth=0.5)
        # Highlight start city
        ax.plot(ordered[0, 0], ordered[0, 1], 's', color='#2ca02c',
                markersize=5, zorder=5, label='Start')

        # Visit order labels on every city
        for idx, (cx, cy) in enumerate(coords[perm]):
            ax.annotate(str(idx), (cx, cy), fontsize=4,
                        ha='center', va='bottom',
                        xytext=(0, 1), textcoords='offset points',
                        color='#333333')

        if title is None:
            name = self._PROBLEM_NAMES.get(problem_id, f'P{problem_id}')
            title = f'{name}  |  Tour Length = {length:.2f}'
        else:
            title = f'{title}  |  Tour Length = {length:.2f}'
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_aspect('equal')
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

        return length

    def P1(self) -> MTOP:
        """
        Problem 1: **Random-20** -- 20 random cities, 20-D.

        Objective: total tour length, minimize.

        References: [1]_ [2]_
        """
        coords = _generate_random_cities(20, seed=self.seed)
        return self._build_problem(coords)

    def P2(self) -> MTOP:
        """
        Problem 2: **Circle-30** -- 30 cities on a circle, 30-D.

        The optimal tour visits cities in order around the circle.
        Objective: total tour length, minimize.

        References: [1]_ [2]_
        """
        coords = _generate_circle_cities(30)
        return self._build_problem(coords)

    def P3(self) -> MTOP:
        """
        Problem 3: **Clustered-50** -- 50 cities in 5 clusters, 50-D.

        Cities are grouped in 5 clusters; intra-cluster distances are small.
        Objective: total tour length, minimize.

        References: [1]_ [2]_ [3]_
        """
        coords = _generate_clustered_cities(50, n_clusters=5, seed=self.seed)
        return self._build_problem(coords)

    def P4(self) -> MTOP:
        """
        Problem 4: **Random-50** -- 50 random cities, 50-D.

        Objective: total tour length, minimize.

        References: [1]_ [2]_ [3]_
        """
        coords = _generate_random_cities(50, seed=self.seed)
        return self._build_problem(coords)

    def P5(self) -> MTOP:
        """
        Problem 5: **Random-100** -- 100 random cities, 100-D.

        Objective: total tour length, minimize.

        References: [1]_ [2]_ [3]_
        """
        coords = _generate_random_cities(100, seed=self.seed)
        return self._build_problem(coords)

    def P6(self) -> MTOP:
        """
        Problem 6: **Random-200** -- 200 random cities, 200-D.

        Objective: total tour length, minimize.

        References: [1]_ [2]_ [3]_
        """
        coords = _generate_random_cities(200, seed=self.seed)
        return self._build_problem(coords)

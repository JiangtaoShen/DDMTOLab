import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.algo_utils import nd_sort


class ZDT:
    """
    Implementation of the ZDT test suite for multi-objective optimization.

    The ZDT test problems (ZDT1 to ZDT6) are standard bi-objective optimization
    benchmarks proposed by Zitzler, Deb, and Thiele (2000).

    Each method in this class generates a Multi-Task Optimization Problem (MTOP)
    instance containing a single ZDT task.

    References
    ----------
    E. Zitzler, K. Deb, and L. Thiele. Comparison of multiobjective
    evolutionary algorithms: Empirical results. Evolutionary Computation,
    2000, 8(2): 173-195.

    Notes
    -----
    All ZDT problems have exactly M=2 objectives. The decision space dimension
    can be adjusted via the `dim` parameter.
    """

    def ZDT1(self, dim=30) -> MTOP:
        """
        Generates the **ZDT1** problem.

        ZDT1 features a convex Pareto front and tests the ability to converge
        to the optimal front uniformly.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the ZDT1 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            obj = np.zeros((n_samples, 2))
            # First objective: f1 = x1
            obj[:, 0] = x[:, 0]
            # g function: g = 1 + 9 * mean(x2, ..., xn)
            g = 1 + 9 * np.mean(x[:, 1:], axis=1)
            # h function: h = 1 - sqrt(f1 / g)
            h = 1 - np.sqrt(obj[:, 0] / g)
            # Second objective: f2 = g * h
            obj[:, 1] = g * h
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def ZDT2(self, dim=30) -> MTOP:
        """
        Generates the **ZDT2** problem.

        ZDT2 features a non-convex Pareto front and tests the ability to
        maintain diversity in non-convex regions.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the ZDT2 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            obj = np.zeros((n_samples, 2))
            # First objective: f1 = x1
            obj[:, 0] = x[:, 0]
            # g function: g = 1 + 9 * mean(x2, ..., xn)
            g = 1 + 9 * np.mean(x[:, 1:], axis=1)
            # h function: h = 1 - (f1 / g)^2
            h = 1 - (obj[:, 0] / g) ** 2
            # Second objective: f2 = g * h
            obj[:, 1] = g * h
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def ZDT3(self, dim=30) -> MTOP:
        """
        Generates the **ZDT3** problem.

        ZDT3 features a disconnected Pareto front and tests the ability to
        maintain subpopulations in different regions.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the ZDT3 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            obj = np.zeros((n_samples, 2))
            # First objective: f1 = x1
            obj[:, 0] = x[:, 0]
            # g function: g = 1 + 9 * mean(x2, ..., xn)
            g = 1 + 9 * np.mean(x[:, 1:], axis=1)
            # h function: h = 1 - sqrt(f1 / g) - (f1 / g) * sin(10 * pi * f1)
            h = 1 - np.sqrt(obj[:, 0] / g) - (obj[:, 0] / g) * np.sin(10 * np.pi * obj[:, 0])
            # Second objective: f2 = g * h
            obj[:, 1] = g * h
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def ZDT4(self, dim=10) -> MTOP:
        """
        Generates the **ZDT4** problem.

        ZDT4 features multiple local Pareto fronts and tests the ability to
        avoid local optima. It has 21^9 local Pareto fronts.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 10).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the ZDT4 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            obj = np.zeros((n_samples, 2))
            # First objective: f1 = x1
            obj[:, 0] = x[:, 0]
            # g function: g = 1 + 10 * (n - 1) + sum((xi^2 - 10 * cos(4 * pi * xi)))
            g = 1 + 10 * (x.shape[1] - 1) + np.sum(
                x[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * x[:, 1:]), axis=1
            )
            # h function: h = 1 - sqrt(f1 / g)
            h = 1 - np.sqrt(obj[:, 0] / g)
            # Second objective: f2 = g * h
            obj[:, 1] = g * h
            return obj

        lb = np.hstack([0, -5 * np.ones(dim - 1)])
        ub = np.hstack([1, 5 * np.ones(dim - 1)])
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def ZDT5(self, dim=80) -> MTOP:
        """
        Generates the **ZDT5** problem.

        ZDT5 is a binary-encoded problem with a deceptive fitness landscape.
        The dimension is adjusted to be 30 + 5k where k is an integer.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 80). The actual dimension
            will be adjusted to 30 + 5k format.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the ZDT5 task.

        Notes
        -----
        Decision variables are binary-encoded. Continuous inputs are converted
        to binary by thresholding at 0.5: x > 0.5 → 1, x ≤ 0.5 → 0.
        """
        # Adjust dimension to 30 + 5k format
        dim = int(np.ceil(max(dim - 30, 1) / 5)) * 5 + 30

        def T1(x):
            x = np.atleast_2d(x)

            # Convert continuous values to binary: x > 0.5 → 1, x ≤ 0.5 → 0
            x_binary = (x > 0.5).astype(int)

            n_samples = x_binary.shape[0]
            # Number of groups: 1 group of 30 + (dim - 30) / 5 groups of 5
            n_groups = 1 + (x_binary.shape[1] - 30) // 5
            u = np.zeros((n_samples, n_groups))

            # First group: sum of first 30 variables
            u[:, 0] = np.sum(x_binary[:, :30], axis=1)

            # Remaining groups: sum of each 5-variable block
            for i in range(1, n_groups):
                start_idx = (i - 1) * 5 + 30
                end_idx = start_idx + 5
                u[:, i] = np.sum(x_binary[:, start_idx:end_idx], axis=1)

            # v function: deceptive fitness
            v = np.zeros_like(u)
            v[u < 5] = 2 + u[u < 5]
            v[u == 5] = 1

            # First objective: f1 = 1 + u1
            obj = np.zeros((n_samples, 2))
            obj[:, 0] = 1 + u[:, 0]

            # g function: sum of v values (excluding first group)
            g = np.sum(v[:, 1:], axis=1)

            # h function: h = 1 / f1
            h = 1.0 / obj[:, 0]

            # Second objective: f2 = g * h
            obj[:, 1] = g * h
            return obj

        # Binary variables: lower bound 0, upper bound 1
        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def ZDT6(self, dim=10) -> MTOP:
        """
        Generates the **ZDT6** problem.

        ZDT6 features a non-uniform search space and non-convex Pareto front.
        It has low density of solutions near the Pareto front.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 10).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the ZDT6 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            obj = np.zeros((n_samples, 2))
            # First objective: f1 = 1 - exp(-4 * x1) * sin^6(6 * pi * x1)
            obj[:, 0] = 1 - np.exp(-4 * x[:, 0]) * (np.sin(6 * np.pi * x[:, 0]) ** 6)
            # g function: g = 1 + 9 * (mean(x2, ..., xn))^0.25
            g = 1 + 9 * (np.mean(x[:, 1:], axis=1) ** 0.25)
            # h function: h = 1 - (f1 / g)^2
            h = 1 - (obj[:, 0] / g) ** 2
            # Second objective: f2 = g * h
            obj[:, 1] = g * h
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem


# --- Pareto Front (PF) Functions ---

def ZDT1_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for ZDT1.

    The PF is convex: f2 = 1 - sqrt(f1).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 2 for ZDT1, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) representing the PF points.
    """
    assert M == 2, "ZDT1 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - np.sqrt(f1)
    return np.hstack([f1, f2])


def ZDT2_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for ZDT2.

    The PF is non-convex: f2 = 1 - f1^2.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 2 for ZDT2, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) representing the PF points.
    """
    assert M == 2, "ZDT2 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - f1 ** 2
    return np.hstack([f1, f2])


def ZDT3_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for ZDT3.

    The PF is disconnected: f2 = 1 - sqrt(f1) - f1 * sin(10 * pi * f1).
    Uses efficient region sampling (no non-dominated sorting needed).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 2 for ZDT3, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) representing the PF points.
    """
    assert M == 2, "ZDT3 only supports M=2 objectives"

    # ZDT3 has 5 disconnected Pareto-optimal regions
    # These boundaries ensure all points are non-dominated
    regions = [
        (0.0000, 0.0830),
        (0.1822, 0.2577),
        (0.4093, 0.4538),
        (0.6183, 0.6525),
        (0.8233, 0.8518)
    ]

    # Calculate the relative length of each region
    region_lengths = [end - start for start, end in regions]
    total_length = sum(region_lengths)

    # Distribute N points proportionally across regions
    points_per_region = [int(N * length / total_length) for length in region_lengths]

    # Ensure exactly N points (adjust last region for rounding errors)
    points_per_region[-1] += N - sum(points_per_region)

    # Generate uniformly spaced points within each region
    pf_segments = []
    for (start, end), n_points in zip(regions, points_per_region):
        if n_points > 0:
            f1 = np.linspace(start, end, n_points).reshape(-1, 1)
            f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
            pf_segments.append(np.hstack([f1, f2]))

    return np.vstack(pf_segments)


def ZDT4_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for ZDT4.

    The PF is identical to ZDT1: f2 = 1 - sqrt(f1).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 2 for ZDT4, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) representing the PF points.
    """
    assert M == 2, "ZDT4 only supports M=2 objectives"
    return ZDT1_PF(N, M)


def ZDT5_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for ZDT5.

    The PF is discrete with 31 specific points.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF (not used for ZDT5,
        which has a fixed discrete PF with 31 points).
    M : int, optional
        Number of objectives (must be 2 for ZDT5, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (31, 2) representing the PF points.

    Notes
    -----
    ZDT5 has a discrete PF with exactly 31 points regardless of N.
    The actual dimension used should be passed via problem configuration.
    """
    assert M == 2, "ZDT5 only supports M=2 objectives"
    # For ZDT5, we assume dim=80 as default
    # In practice, this should be consistent with the problem instance
    dim = 80
    dim = int(np.ceil(max(dim - 30, 1) / 5)) * 5 + 30

    # f1 ranges from 1 to 31
    f1 = np.arange(1, 32).reshape(-1, 1)
    # f2 = (dim - 30) / 5 / f1
    f2 = ((dim - 30) / 5) / f1
    return np.hstack([f1, f2])


def ZDT6_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for ZDT6.

    The PF is non-convex: f2 = 1 - f1^2, with f1 starting from ~0.280775.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 2 for ZDT6, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) representing the PF points.
    """
    assert M == 2, "ZDT6 only supports M=2 objectives"
    minf1 = 0.280775
    f1 = np.linspace(minf1, 1, N).reshape(-1, 1)
    f2 = 1 - f1 ** 2
    return np.hstack([f1, f2])


SETTINGS = {
    'metric': 'IGD',
    'n_ref': 10000,
    'ZDT1': {'T1': ZDT1_PF},
    'ZDT2': {'T1': ZDT2_PF},
    'ZDT3': {'T1': ZDT3_PF},
    'ZDT4': {'T1': ZDT4_PF},
    'ZDT5': {'T1': ZDT5_PF},
    'ZDT6': {'T1': ZDT6_PF},
}
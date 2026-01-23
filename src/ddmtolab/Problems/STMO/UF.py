import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point


class UF:
    """
    Implementation of the UF test suite for multi-objective optimization.

    The UF test problems (UF1 to UF10) are unconstrained benchmark MOPs
    proposed by Zhang et al. (2009) for the CEC 2009 special session and
    competition.

    Each method in this class generates a Multi-Task Optimization Problem (MTOP)
    instance containing a single UF task.

    References
    ----------
    Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, and S. Tiwari.
    Multiobjective optimization test instances for the CEC 2009 special
    session and competition. School of CS & EE, University of Essex, Working
    Report CES-487, 2009.

    Notes
    -----
    UF1-UF7 have M=2 objectives, UF8-UF10 have M=3 objectives.
    The decision space dimension can be adjusted via the `dim` parameter.
    """

    def UF1(self, dim=30) -> MTOP:
        """
        Generates the **UF1** problem.

        UF1 is a bi-objective problem with a convex Pareto front.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF1 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 2))

            # J1: 3:2:D (MATLAB) -> indices 2,4,6,... (Python 0-indexed)
            # J2: 2:2:D (MATLAB) -> indices 1,3,5,... (Python 0-indexed)
            J1 = np.arange(2, n_vars, 2)
            J2 = np.arange(1, n_vars, 2)

            # Y transformation
            Y = x - np.sin(
                6 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # Objectives
            obj[:, 0] = x[:, 0] + 2 * np.mean(Y[:, J1] ** 2, axis=1)
            obj[:, 1] = 1 - np.sqrt(x[:, 0]) + 2 * np.mean(Y[:, J2] ** 2, axis=1)

            return obj

        lb = np.hstack([0, -np.ones(dim - 1)])
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF2(self, dim=30) -> MTOP:
        """
        Generates the **UF2** problem.

        UF2 is a bi-objective problem with more complex variable linkage.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF2 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 2))

            J1 = np.arange(2, n_vars, 2)
            J2 = np.arange(1, n_vars, 2)

            Y = np.zeros_like(x)

            # For J1 indices (even in MATLAB 1-indexing)
            if len(J1) > 0:
                X1 = x[:, 0:1]
                term = (0.3 * X1 ** 2 * np.cos(24 * np.pi * X1 + 4 * (J1 + 1) * np.pi / n_vars) +
                        0.6 * X1)
                Y[:, J1] = x[:, J1] - term * np.cos(6 * np.pi * X1 + (J1 + 1) * np.pi / n_vars)

            # For J2 indices (odd in MATLAB 1-indexing)
            if len(J2) > 0:
                X1 = x[:, 0:1]
                term = (0.3 * X1 ** 2 * np.cos(24 * np.pi * X1 + 4 * (J2 + 1) * np.pi / n_vars) +
                        0.6 * X1)
                Y[:, J2] = x[:, J2] - term * np.sin(6 * np.pi * X1 + (J2 + 1) * np.pi / n_vars)

            # Objectives
            obj[:, 0] = x[:, 0] + 2 * np.mean(Y[:, J1] ** 2, axis=1)
            obj[:, 1] = 1 - np.sqrt(x[:, 0]) + 2 * np.mean(Y[:, J2] ** 2, axis=1)

            return obj

        lb = np.hstack([0, -np.ones(dim - 1)])
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF3(self, dim=30) -> MTOP:
        """
        Generates the **UF3** problem.

        UF3 features a complex landscape with product terms.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF3 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 2))

            J1 = np.arange(2, n_vars, 2)
            J2 = np.arange(1, n_vars, 2)

            # Y transformation
            power_exp = 0.5 * (1 + 3 * (np.arange(1, n_vars + 1) - 2) / (n_vars - 2))
            Y = x - x[:, 0:1] ** power_exp

            # Product terms
            if len(J1) > 0:
                prod_J1 = np.prod(np.cos(20 * Y[:, J1] * np.pi / np.sqrt(J1 + 1)), axis=1)
                sum_J1 = np.sum(Y[:, J1] ** 2, axis=1)
                obj[:, 0] = x[:, 0] + 2 / len(J1) * (4 * sum_J1 - 2 * prod_J1 + 2)
            else:
                obj[:, 0] = x[:, 0]

            if len(J2) > 0:
                prod_J2 = np.prod(np.cos(20 * Y[:, J2] * np.pi / np.sqrt(J2 + 1)), axis=1)
                sum_J2 = np.sum(Y[:, J2] ** 2, axis=1)
                obj[:, 1] = 1 - np.sqrt(x[:, 0]) + 2 / len(J2) * (4 * sum_J2 - 2 * prod_J2 + 2)
            else:
                obj[:, 1] = 1 - np.sqrt(x[:, 0])

            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF4(self, dim=30) -> MTOP:
        """
        Generates the **UF4** problem.

        UF4 features a complex sine-based transformation with h function.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF4 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 2))

            J1 = np.arange(2, n_vars, 2)
            J2 = np.arange(1, n_vars, 2)

            # Y transformation
            Y = x - np.sin(
                6 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # h transformation
            hY = np.abs(Y) / (1 + np.exp(2 * np.abs(Y)))

            # Objectives
            obj[:, 0] = x[:, 0] + 2 * np.mean(hY[:, J1], axis=1)
            obj[:, 1] = 1 - x[:, 0] ** 2 + 2 * np.mean(hY[:, J2], axis=1)

            return obj

        lb = np.hstack([0, -2 * np.ones(dim - 1)])
        ub = np.hstack([1, 2 * np.ones(dim - 1)])
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF5(self, dim=30) -> MTOP:
        """
        Generates the **UF5** problem.

        UF5 features multimodal landscape with oscillatory Pareto front.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF5 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 2))

            J1 = np.arange(2, n_vars, 2)
            J2 = np.arange(1, n_vars, 2)

            # Y transformation
            Y = x - np.sin(
                6 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # h transformation
            hY = 2 * Y ** 2 - np.cos(4 * np.pi * Y) + 1

            # Oscillatory term
            epsilon = 0.1
            N = 10
            osc_term = (1 / (2 * N) + epsilon) * np.abs(np.sin(2 * N * np.pi * x[:, 0]))

            # Objectives
            obj[:, 0] = x[:, 0] + osc_term + 2 * np.mean(hY[:, J1], axis=1)
            obj[:, 1] = 1 - x[:, 0] + osc_term + 2 * np.mean(hY[:, J2], axis=1)

            return obj

        lb = np.hstack([0, -np.ones(dim - 1)])
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF6(self, dim=30) -> MTOP:
        """
        Generates the **UF6** problem.

        UF6 features disconnected Pareto front with product terms.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF6 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 2))

            J1 = np.arange(2, n_vars, 2)
            J2 = np.arange(1, n_vars, 2)

            # Y transformation
            Y = x - np.sin(
                6 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # Oscillatory term with max(0, ...)
            epsilon = 0.1
            N = 2
            osc_term = np.maximum(0, 2 * (1 / (2 * N) + epsilon) * np.sin(2 * N * np.pi * x[:, 0]))

            # Product and sum terms
            if len(J1) > 0:
                prod_J1 = np.prod(np.cos(20 * Y[:, J1] * np.pi / np.sqrt(J1 + 1)), axis=1)
                sum_J1 = np.sum(Y[:, J1] ** 2, axis=1)
                term_J1 = 2 / len(J1) * (4 * sum_J1 - 2 * prod_J1 + 2)
            else:
                term_J1 = 0

            if len(J2) > 0:
                prod_J2 = np.prod(np.cos(20 * Y[:, J2] * np.pi / np.sqrt(J2 + 1)), axis=1)
                sum_J2 = np.sum(Y[:, J2] ** 2, axis=1)
                term_J2 = 2 / len(J2) * (4 * sum_J2 - 2 * prod_J2 + 2)
            else:
                term_J2 = 0

            # Objectives
            obj[:, 0] = x[:, 0] + osc_term + term_J1
            obj[:, 1] = 1 - x[:, 0] + osc_term + term_J2

            return obj

        lb = np.hstack([0, -np.ones(dim - 1)])
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF7(self, dim=30) -> MTOP:
        """
        Generates the **UF7** problem.

        UF7 features a power transformation in the objectives.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF7 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 2))

            J1 = np.arange(2, n_vars, 2)
            J2 = np.arange(1, n_vars, 2)

            # Y transformation
            Y = x - np.sin(
                6 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # Objectives with power 0.2
            obj[:, 0] = x[:, 0] ** 0.2 + 2 * np.mean(Y[:, J1] ** 2, axis=1)
            obj[:, 1] = 1 - x[:, 0] ** 0.2 + 2 * np.mean(Y[:, J2] ** 2, axis=1)

            return obj

        lb = np.hstack([0, -np.ones(dim - 1)])
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF8(self, dim=30) -> MTOP:
        """
        Generates the **UF8** problem.

        UF8 is a three-objective problem with spherical Pareto front.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF8 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 3))

            # J1: 4:3:D -> indices 3,6,9,... (Python 0-indexed)
            # J2: 5:3:D -> indices 4,7,10,... (Python 0-indexed)
            # J3: 3:3:D -> indices 2,5,8,... (Python 0-indexed)
            J1 = np.arange(3, n_vars, 3)
            J2 = np.arange(4, n_vars, 3)
            J3 = np.arange(2, n_vars, 3)

            # Y transformation
            Y = x - 2 * x[:, 1:2] * np.sin(
                2 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # Objectives
            obj[:, 0] = (np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) +
                         2 * np.mean(Y[:, J1] ** 2, axis=1))
            obj[:, 1] = (np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) +
                         2 * np.mean(Y[:, J2] ** 2, axis=1))
            obj[:, 2] = np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(Y[:, J3] ** 2, axis=1)

            return obj

        lb = np.hstack([0, 0, -2 * np.ones(dim - 2)])
        ub = np.hstack([1, 1, 2 * np.ones(dim - 2)])
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF9(self, dim=30) -> MTOP:
        """
        Generates the **UF9** problem.

        UF9 is a three-objective problem with disconnected Pareto front.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF9 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 3))

            J1 = np.arange(3, n_vars, 3)
            J2 = np.arange(4, n_vars, 3)
            J3 = np.arange(2, n_vars, 3)

            # Y transformation
            Y = x - 2 * x[:, 1:2] * np.sin(
                2 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # Environment selection
            epsilon = 0.5 * (np.maximum(0, 1.1 * (1 - 4 * (2 * x[:, 0] - 1) ** 2)) +
                             2 * x[:, 0])

            # Objectives
            obj[:, 0] = epsilon * x[:, 1] + 2 * np.mean(Y[:, J1] ** 2, axis=1)
            obj[:, 1] = (0.5 * (np.maximum(0, 1.1 * (1 - 4 * (2 * x[:, 0] - 1) ** 2)) -
                                2 * x[:, 0] + 2) * x[:, 1] +
                         2 * np.mean(Y[:, J2] ** 2, axis=1))
            obj[:, 2] = 1 - x[:, 1] + 2 * np.mean(Y[:, J3] ** 2, axis=1)

            return obj

        lb = np.hstack([0, 0, -2 * np.ones(dim - 2)])
        ub = np.hstack([1, 1, 2 * np.ones(dim - 2)])
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def UF10(self, dim=30) -> MTOP:
        """
        Generates the **UF10** problem.

        UF10 is a three-objective problem with spherical Pareto front and
        complex landscape.

        Parameters
        ----------
        dim : int, optional
            Number of decision variables (default is 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the UF10 task.
        """

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            n_vars = x.shape[1]
            obj = np.zeros((n_samples, 3))

            J1 = np.arange(3, n_vars, 3)
            J2 = np.arange(4, n_vars, 3)
            J3 = np.arange(2, n_vars, 3)

            # Y transformation
            Y = x - 2 * x[:, 1:2] * np.sin(
                2 * np.pi * x[:, 0:1] +
                np.pi * (np.arange(1, n_vars + 1)) / n_vars
            )

            # Apply h function
            Y = 4 * Y ** 2 - np.cos(8 * np.pi * Y) + 1

            # Objectives
            obj[:, 0] = (np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) +
                         2 * np.mean(Y[:, J1], axis=1))
            obj[:, 1] = (np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) +
                         2 * np.mean(Y[:, J2], axis=1))
            obj[:, 2] = np.sin(0.5 * x[:, 0] * np.pi) + 2 * np.mean(Y[:, J3], axis=1)

            return obj

        lb = np.hstack([0, 0, -2 * np.ones(dim - 2)])
        ub = np.hstack([1, 1, 2 * np.ones(dim - 2)])
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem


# --- Pareto Front (PF) Functions ---

def UF1_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for UF1.

    The PF is: f2 = 1 - sqrt(f1).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 2 for UF1, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) representing the PF points.
    """
    assert M == 2, "UF1 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - f1 ** 0.5
    return np.hstack([f1, f2])


def UF2_PF(N: int, M: int = 2) -> np.ndarray:
    """Computes the Pareto Front for UF2."""
    assert M == 2, "UF2 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - f1 ** 0.5
    return np.hstack([f1, f2])


def UF3_PF(N: int, M: int = 2) -> np.ndarray:
    """Computes the Pareto Front for UF3."""
    assert M == 2, "UF3 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - f1 ** 0.5
    return np.hstack([f1, f2])


def UF4_PF(N: int, M: int = 2) -> np.ndarray:
    """Computes the Pareto Front for UF4."""
    assert M == 2, "UF4 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - f1 ** 2
    return np.hstack([f1, f2])


def UF5_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front for UF5 (discrete with 21 points).

    Parameters
    ----------
    N : int
        Number of points to generate (not used, returns fixed 21 points).
    M : int, optional
        Number of objectives (must be 2 for UF5, default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (21, 2) representing the PF points.
    """
    assert M == 2, "UF5 only supports M=2 objectives"
    # UF5 has discrete PF with specific points
    f1 = (np.arange(0, 21) / 20).reshape(-1, 1)
    f2 = 1 - f1
    return np.hstack([f1, f2])


def UF6_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front for UF6 (disconnected).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 2 for UF6, default is 2).

    Returns
    -------
    np.ndarray
        Array representing the disconnected PF points.
    """
    assert M == 2, "UF6 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - f1

    # Remove disconnected regions: (0, 1/4) and (1/2, 3/4)
    mask = ~((f1 > 0) & (f1 < 1 / 4) | (f1 > 1 / 2) & (f1 < 3 / 4))
    pf = np.hstack([f1, f2])
    return pf[mask.flatten()]


def UF7_PF(N: int, M: int = 2) -> np.ndarray:
    """Computes the Pareto Front for UF7."""
    assert M == 2, "UF7 only supports M=2 objectives"
    f1 = np.linspace(0, 1, N).reshape(-1, 1)
    f2 = 1 - f1
    return np.hstack([f1, f2])


def UF8_PF(N: int, M: int = 3) -> np.ndarray:
    """
    Computes the Pareto Front for UF8 (unit sphere).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 3 for UF8, default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N_actual, 3) representing points on unit sphere.
    """
    assert M == 3, "UF8 only supports M=3 objectives"
    R, _ = uniform_point(N, M)
    # Normalize to unit sphere
    R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))
    return R


def UF9_PF(N: int, M: int = 3) -> np.ndarray:
    """
    Computes the Pareto Front for UF9.

    Parameters
    ----------
    N : int
        Approximate number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 3 for UF9, default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) representing the disconnected PF points,
        where n_points ≤ N after filtering.
    """
    assert M == 3, "UF9 only supports M=3 objectives"

    # Generate uniformly distributed points on the unit simplex
    R = uniform_point(N, 3)[0]

    # Define the disconnected region mask
    mask = (R[:, 0] > (1 - R[:, 2]) / 4) & (R[:, 0] < (1 - R[:, 2]) * 3 / 4)

    # Filter out points in the disconnected region
    R = R[~mask]

    return R


def UF10_PF(N: int, M: int = 3) -> np.ndarray:
    """
    Computes the Pareto Front for UF10 (unit sphere).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (must be 3 for UF10, default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N_actual, 3) representing points on unit sphere.
    """
    assert M == 3, "UF10 only supports M=3 objectives"
    R, _ = uniform_point(N, M)
    # Normalize to unit sphere
    R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))
    return R


SETTINGS = {
    'metric': 'IGD',
    'n_ref': 10000,
    'UF1': {'T1': UF1_PF},
    'UF2': {'T1': UF2_PF},
    'UF3': {'T1': UF3_PF},
    'UF4': {'T1': UF4_PF},
    'UF5': {'T1': UF5_PF},
    'UF6': {'T1': UF6_PF},
    'UF7': {'T1': UF7_PF},
    'UF8': {'T1': UF8_PF},
    'UF9': {'T1': UF9_PF},
    'UF10': {'T1': UF10_PF},
}
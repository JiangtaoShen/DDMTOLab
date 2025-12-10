import numpy as np
from Methods.mtop import MTOP
from Methods.Algo_Methods.uniform_point import uniform_point


class DTLZ:
    """
    Implementation of the DTLZ test suite for multi-objective optimization.

    The DTLZ test problems (DTLZ1 to DTLZ7) are standard unconstrained
    multi-objective optimization benchmarks. DTLZ8 and DTLZ9 are constrained.

    Each method in this class generates a Multi-Task Optimization Problem (MTOP)
    instance containing a single DTLZ task.

    Notes
    -----
    The decision variables (x) are typically split into M objectives (x[0:M-1])
    and k complexity variables (x[M-1:]).
    """

    def DTLZ1(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ1** problem.

        DTLZ1 features a simple linear Pareto-optimal front (PF) and a complex
        multi-modal search space due to the g-function.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M + k - 1,
            where k=5 for DTLZ1 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ1 task.
        """
        k = 5
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 100 * (dim - M + 1 + np.sum((xM - 0.5) ** 2 - np.cos(20 * np.pi * (xM - 0.5)), axis=1))
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = 0.5 * (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= x[:, j]
                if i > 0:
                    obj[:, i] *= (1 - x[:, M - i - 1])
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ2(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ2** problem.

        DTLZ2 features a simple convex spherical PF and a simple uni-modal
        g-function.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M + k - 1,
            where k=10 for DTLZ2 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ2 task.
        """
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = np.sum((xM - 0.5) ** 2, axis=1)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ3(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ3** problem.

        DTLZ3 features a convex spherical PF (similar to DTLZ2) but has a
        multi-modal g-function, making it difficult to converge.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M + k - 1,
            where k=10 for DTLZ3 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ3 task.
        """
        k = 10
        if dim is None:
            dim = M + k - 1
        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 100 * (k + np.sum((xM - 0.5) ** 2 - np.cos(20 * np.pi * (xM - 0.5)), axis=1))
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ4(self, M=3, dim=None, alpha=100) -> MTOP:
        """
        Generates the **DTLZ4** problem.

        DTLZ4 features a convex spherical PF (similar to DTLZ2) but introduces
        a bias towards certain objective regions due to the exponent :math:`\\alpha`,
        making diversity maintenance challenging.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M + k - 1,
            where k=10 for DTLZ4 (default is None).
        alpha : int, optional
            Exponent used to bias the solution (default is 100).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ4 task.
        """
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            x_modified = x.copy()
            x_modified[:, :M - 1] = x[:, :M - 1] ** alpha
            xM = x[:, M - 1:]
            g = np.sum((xM - 0.5) ** 2, axis=1)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x_modified[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x_modified[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ5(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ5** problem.

        DTLZ5 features a degenerated (curve-like) PF, lying on a :math:`(M-1)`-dimensional
        manifold of the M-dimensional objective space.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M + k - 1,
            where k=10 for DTLZ5 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ5 task.
        """
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = np.sum((xM - 0.5) ** 2, axis=1)
            x_modified = x.copy()
            if M > 2:
                Temp = np.tile(g.reshape(-1, 1), (1, M - 2))
                x_modified[:, 1:M - 1] = (1 + 2 * Temp * x[:, 1:M - 1]) / (2 + 2 * Temp)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x_modified[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x_modified[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ6(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ6** problem.

        DTLZ6 also features a degenerated (curve-like) PF (similar to DTLZ5),
        but introduces a connectivity difficulty with its g-function (:math:`g(x_M) = \\sum x_M^{0.1}`).

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M + k - 1,
            where k=10 for DTLZ6 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ6 task.
        """
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = np.sum(xM ** 0.1, axis=1)
            x_modified = x.copy()
            if M > 2:
                Temp = np.tile(g.reshape(-1, 1), (1, M - 2))
                x_modified[:, 1:M - 1] = (1 + 2 * Temp * x[:, 1:M - 1]) / (2 + 2 * Temp)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x_modified[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x_modified[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ7(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ7** problem.

        DTLZ7 features a disconnected PF and is used to test an algorithm's
        ability to converge to multiple disconnected regions.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M + k - 1,
            where k=20 for DTLZ7 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ7 task.
        """
        k = 20
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 1 + 9 * np.mean(xM, axis=1)
            obj = np.zeros((n_samples, M))
            obj[:, :M - 1] = x[:, :M - 1]
            h = M - np.sum(
                obj[:, :M - 1] / (1 + g.reshape(-1, 1)) *
                (1 + np.sin(3 * np.pi * obj[:, :M - 1])),
                axis=1
            )
            obj[:, M - 1] = (1 + g) * h
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ8(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ8** problem (Constrained).

        DTLZ8 has simple objective functions but complex constraints, typically
        resulting in a PF that is a linear or piecewise linear manifold.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M * k,
            where k=10 for DTLZ8 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ8 task.
        """
        k = 10
        if dim is None:
            dim = M * k

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            obj = np.zeros((n_samples, M))
            # Calculate objective f_m as the mean of the m-th block of decision variables
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.mean(x[:, start_idx:end_idx], axis=1)
            return obj

        def C1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            obj = np.zeros((n_samples, M))
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.mean(x[:, start_idx:end_idx], axis=1)

            cons = np.zeros((n_samples, M))
            # Constraints c_i (i=1 to M-1)
            cons[:, :M - 1] = 1 - np.tile(obj[:, M - 1:M], (1, M - 1)) - 4 * obj[:, :M - 1]

            # Last constraint c_M (handled differently based on M)
            if M == 2:
                # If M=2, the last constraint is often defined to be non-binding or 0
                cons[:, M - 1] = 0
            else:
                # For M > 2, a sum of the two smallest objectives is used
                sorted_obj = np.sort(obj[:, :M - 1], axis=1)
                cons[:, M - 1] = 1 - 2 * obj[:, M - 1] - np.sum(sorted_obj[:, :2], axis=1)
            return cons

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        # DTLZ8 is a constrained problem (C1 is the constraint function)
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ9(self, M=3, dim=None) -> MTOP:
        """
        Generates the **DTLZ9** problem (Constrained).

        DTLZ9 has simple objective functions but constraints that define a
        parabolic shape for the PF.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to M * k,
            where k=10 for DTLZ9 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the DTLZ9 task.
        """
        k = 10
        if dim is None:
            dim = M * k

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            # Decision variables are transformed
            x_transformed = x ** 0.1
            obj = np.zeros((n_samples, M))
            # Calculate objective f_m as the sum of the transformed m-th block
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.sum(x_transformed[:, start_idx:end_idx], axis=1)
            return obj

        def C1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            x_transformed = x ** 0.1
            obj = np.zeros((n_samples, M))
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.sum(x_transformed[:, start_idx:end_idx], axis=1)
            # Constraints c_i (i=1 to M-1) define the parabolic PF shape
            cons = 1 - np.tile(obj[:, M - 1:M] ** 2, (1, M - 1)) - obj[:, :M - 1] ** 2
            return cons

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        # DTLZ9 is a constrained problem (C1 is the constraint function)
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem


# --- Pareto Front (PF) Functions ---

def DTLZ1_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for DTLZ1.

    The PF is a linear hyperplane in the objective space: :math:`\\sum_{i=1}^{M} f_i = 0.5`.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    W, _ = uniform_point(N, M)
    return W / 2

def DTLZ2_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for DTLZ2.

    The PF is the unit sphere in the positive orthant.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    W, _ = uniform_point(N, M)
    norms = np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))
    return W / norms

# DTLZ3, DTLZ4 share the same PF shape (unit sphere) as DTLZ2
DTLZ3_PF = DTLZ2_PF
DTLZ4_PF = DTLZ2_PF

def DTLZ5_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for DTLZ5.

    The PF is a degenerate curve (arc on the unit sphere).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    t = np.linspace(0, 1, N)
    R = np.column_stack([t, 1 - t])
    norms = np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))
    R = R / norms
    if M > 2:
        first_col_repeated = np.tile(R[:, 0:1], (1, M - 2))
        R = np.hstack([first_col_repeated, R])
        powers = np.concatenate([[M - 2], np.arange(M - 2, -1, -1)])
        scale_factors = np.sqrt(2) ** powers
        R = R / scale_factors.reshape(1, -1)
    return R

# DTLZ6 shares the same PF shape (degenerate curve) as DTLZ5
DTLZ6_PF = DTLZ5_PF

def DTLZ7_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for DTLZ7.

    The PF is composed of :math:`2^{M-1}` disconnected segments.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    interval = np.array([0, 0.251412, 0.631627, 0.859401])
    median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
    X, _ = uniform_point(N, M - 1, 'grid')
    mask_low = X <= median
    X[mask_low] = X[mask_low] * (interval[1] - interval[0]) / median + interval[0]
    mask_high = X > median
    X[mask_high] = (X[mask_high] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
    h = M - np.sum(X / 2 * (1 + np.sin(3 * np.pi * X)), axis=1, keepdims=True)
    optimum = np.hstack([X, 2 * h])
    return optimum

def DTLZ8_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for DTLZ8 (Constrained).

    The PF is complex and constrained, generally a piecewise linear manifold.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the PF points (N' <= N).
    """
    if M == 2:
        temp = np.linspace(0, 1, N).reshape(-1, 1)
        optimum = np.hstack([(1 - temp) / 4, temp])
    else:
        # Complex calculation involving uniform points and filtering based on constraints
        temp, _ = uniform_point(N // (M - 1), 3)
        temp[:, 2] = temp[:, 2] / 2
        mask = (temp[:, 0] >= (1 - temp[:, 2]) / 4) & \
               (temp[:, 0] <= temp[:, 1]) & \
               (temp[:, 2] <= 1 / 3)
        temp = temp[mask, :]
        n_temp = temp.shape[0]
        optimum = np.zeros((n_temp * (M - 1), M))
        for i in range(M - 1):
            start_idx = i * n_temp
            end_idx = (i + 1) * n_temp
            optimum[start_idx:end_idx, :M - 1] = np.tile(temp[:, 1], (M - 1, 1)).T
            optimum[start_idx:end_idx, M - 1] = temp[:, 2]
            optimum[start_idx:end_idx, i] = temp[:, 0]
        gap_values = np.unique(optimum[:, M - 1])
        if len(gap_values) > 1:
            gap = np.sort(gap_values)[1] - np.sort(gap_values)[0]
            temp_extra = np.arange(1 / 3, 1 + gap, gap).reshape(-1, 1)
            extra_points = np.hstack([
                np.tile((1 - temp_extra) / 4, (1, M - 1)),
                temp_extra
            ])
            optimum = np.vstack([optimum, extra_points])
        optimum = np.unique(optimum, axis=0)
    return optimum

def DTLZ9_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for DTLZ9 (Constrained).

    The PF is a curve on a unit sphere segment (parabolic/quadric relationship).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    Temp = np.linspace(0, 1, N).reshape(-1, 1)
    optimum = np.hstack([
        np.tile(np.cos(0.5 * np.pi * Temp), (1, M - 1)),
        np.sin(0.5 * np.pi * Temp)
    ])
    return optimum


SETTINGS = {
    """
    Configuration settings for DTLZ problem evaluation.

    Attributes
    ----------
    metric : str
        The performance metric used for evaluation (e.g., 'IGD').
    n_ref : int
        The number of reference points used to approximate the true Pareto Front.
    DTLZ1 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ1_PF).
    DTLZ2 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ2_PF).
    DTLZ3 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ3_PF).
    DTLZ4 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ4_PF).
    DTLZ5 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ5_PF).
    DTLZ6 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ6_PF).
    DTLZ7 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ7_PF).
    DTLZ8 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ8_PF).
    DTLZ9 : dict
        Mapping of task names ('T1') to their respective PF functions (DTLZ9_PF).
    """
    'metric': 'IGD',
    'n_ref': 10000,
    'DTLZ1': {'T1': DTLZ1_PF},
    'DTLZ2': {'T1': DTLZ2_PF},
    'DTLZ3': {'T1': DTLZ3_PF},
    'DTLZ4': {'T1': DTLZ4_PF},
    'DTLZ5': {'T1': DTLZ5_PF},
    'DTLZ6': {'T1': DTLZ6_PF},
    'DTLZ7': {'T1': DTLZ7_PF},
    'DTLZ8': {'T1': DTLZ8_PF},
    'DTLZ9': {'T1': DTLZ9_PF},
}
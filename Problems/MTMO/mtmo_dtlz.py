import numpy as np
from Methods.mtop import MTOP
from Methods.Algo_Methods.uniform_point import uniform_point


class MTMO_DTLZ:
    """
    Implementation of Multi-Task Multi-Objective (MTMO) test problems based on DTLZ suite.

    This class provides combinations of DTLZ problems as multi-task optimization benchmarks,
    where each problem consists of two related DTLZ tasks with potentially different
    characteristics (e.g., multi-modal vs. uni-modal, different complexity).
    """

    def P1(self, M=3, dim=None) -> MTOP:
        """
        Generates **P1**: T1 (DTLZ2) vs T2 (DTLZ1).

        This problem combines:
        - T1 (DTLZ2): Simple convex spherical PF with uni-modal g-function
        - T2 (DTLZ1): Linear PF with complex multi-modal g-function

        The combination tests knowledge transfer between problems with different
        landscape difficulties (uni-modal vs. multi-modal).

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set based on task requirements.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing DTLZ2 and DTLZ1.
        """
        # Task 1: DTLZ2
        k1 = 10
        if dim is None:
            dim1 = M + k1 - 1
        else:
            dim1 = dim

        def T1(x):
            """DTLZ2 objective function"""
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

        # Task 2: DTLZ1
        k2 = 5
        if dim is None:
            dim2 = M + k2 - 1
        else:
            dim2 = dim

        def T2(x):
            """DTLZ1 objective function"""
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 100 * (dim2 - M + 1 + np.sum((xM - 0.5) ** 2 - np.cos(20 * np.pi * (xM - 0.5)), axis=1))
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = 0.5 * (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= x[:, j]
                if i > 0:
                    obj[:, i] *= (1 - x[:, M - i - 1])
            return obj

        lb = np.zeros(max(dim1, dim2))
        ub = np.ones(max(dim1, dim2))

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim1, lower_bound=lb[:dim1], upper_bound=ub[:dim1])
        problem.add_task(objective_func=T2, dim=dim2, lower_bound=lb[:dim2], upper_bound=ub[:dim2])
        return problem

    def P2(self, M=3, dim=None) -> MTOP:
        """
        Generates **P2**: T1 (DTLZ2) vs T2 (DTLZ3).

        This problem combines:
        - T1 (DTLZ2): Simple convex spherical PF with uni-modal g-function
        - T2 (DTLZ3): Convex spherical PF (same as DTLZ2) with multi-modal g-function

        The combination tests knowledge transfer between problems with the same PF shape
        but different convergence difficulties (uni-modal vs. multi-modal landscape).

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set based on task requirements.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing DTLZ2 and DTLZ3.
        """
        # Task 1: DTLZ2
        k1 = 10
        if dim is None:
            dim1 = M + k1 - 1
        else:
            dim1 = dim

        def T1(x):
            """DTLZ2 objective function"""
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

        # Task 2: DTLZ3
        k2 = 10
        if dim is None:
            dim2 = M + k2 - 1
        else:
            dim2 = dim

        def T2(x):
            """DTLZ3 objective function"""
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 100 * (k2 + np.sum((xM - 0.5) ** 2 - np.cos(20 * np.pi * (xM - 0.5)), axis=1))
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(max(dim1, dim2))
        ub = np.ones(max(dim1, dim2))

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim1, lower_bound=lb[:dim1], upper_bound=ub[:dim1])
        problem.add_task(objective_func=T2, dim=dim2, lower_bound=lb[:dim2], upper_bound=ub[:dim2])
        return problem

    def P3(self, M=3, dim=None, alpha=100) -> MTOP:
        """
        Generates **P3**: T1 (DTLZ2) vs T2 (DTLZ4).

        This problem combines:
        - T1 (DTLZ2): Simple convex spherical PF with uni-modal g-function
        - T2 (DTLZ4): Convex spherical PF with biased objective space (alpha parameter)

        The combination tests knowledge transfer between problems with the same PF shape
        but different diversity maintenance challenges (uniform vs. biased distribution).

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set based on task requirements.
        alpha : int, optional
            Exponent used in DTLZ4 to bias the solution distribution (default is 100).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing DTLZ2 and DTLZ4.
        """
        # Task 1: DTLZ2
        k1 = 10
        if dim is None:
            dim1 = M + k1 - 1
        else:
            dim1 = dim

        def T1(x):
            """DTLZ2 objective function"""
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

        # Task 2: DTLZ4
        k2 = 10
        if dim is None:
            dim2 = M + k2 - 1
        else:
            dim2 = dim

        def T2(x):
            """DTLZ4 objective function"""
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

        lb = np.zeros(max(dim1, dim2))
        ub = np.ones(max(dim1, dim2))

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim1, lower_bound=lb[:dim1], upper_bound=ub[:dim1])
        problem.add_task(objective_func=T2, dim=dim2, lower_bound=lb[:dim2], upper_bound=ub[:dim2])
        return problem


# --- Pareto Front (PF) Functions for MTMO Problems ---

def P1_T1_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for P1, Task 1 (DTLZ2).

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


def P1_T2_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for P1, Task 2 (DTLZ1).

    The PF is a linear hyperplane: sum(f_i) = 0.5.

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


def P2_T1_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for P2, Task 1 (DTLZ2).

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


def P2_T2_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for P2, Task 2 (DTLZ3).

    The PF is the unit sphere in the positive orthant (same as DTLZ2).

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


def P3_T1_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for P3, Task 1 (DTLZ2).

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


def P3_T2_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for P3, Task 2 (DTLZ4).

    The PF is the unit sphere in the positive orthant (same as DTLZ2).

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


# Configuration settings for evaluation
SETTINGS = {
    'metric': 'IGD',
    'n_ref': 10000,
    'P1': {'T1': P1_T1_PF, 'T2': P1_T2_PF},
    'P2': {'T1': P2_T1_PF, 'T2': P2_T2_PF},
    'P3': {'T1': P3_T1_PF, 'T2': P3_T2_PF},
}
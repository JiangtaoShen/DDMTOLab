import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point


class MTMO_DTLZ:
    """
    Multi-Task Multi-Objective DTLZ benchmark problems.

    These problems combine different DTLZ test functions as separate tasks
    within a multi-task optimization framework.
    """

    problem_information = {
        'n_cases': 1,
        'n_tasks': '2',
        'n_dims': 'D',
        'n_objs': 'M',
        'n_cons': '0',
        'type': 'synthetic',
    }

    def P1(self, M=3, D=10) -> MTOP:
        """
        Generates Problem 1: **T1 (DTLZ2) vs T2 (DTLZ3)**.

        - T1: DTLZ2 with a simple uni-modal g-function. PF is the unit sphere.
        - T2: DTLZ3 with a multi-modal g-function. PF is the unit sphere (same shape
          as DTLZ2) but much harder to converge due to many local fronts.
        - Relationship: Both tasks share the same PF shape but differ in landscape
          difficulty, enabling potential knowledge transfer of convergence information.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        D : int, optional
            Number of decision variables (default is 10). Must satisfy D >= M.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """

        def T1(x):
            """Task 1: DTLZ2 (3-objective, 10D)"""
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

        def T2(x):
            """Task 2: DTLZ3 (3-objective, 10D)"""
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            k = D - M + 1
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

        lb = np.zeros(D)
        ub = np.ones(D)

        problem = MTOP()
        problem.add_task(T1, dim=D, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=D, lower_bound=lb, upper_bound=ub)
        return problem


# --- True Pareto Front (PF) Functions ---

def P1_T1_PF(N, M=3) -> np.ndarray:
    """
    Computes the True Pareto Front for P1, Task 1 (DTLZ2).

    The PF is the unit sphere in the positive orthant.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    W, _ = uniform_point(N, M)
    norms = np.sqrt(np.sum(W ** 2, axis=1, keepDs=True))
    return W / norms


# DTLZ3 shares the same PF shape as DTLZ2
P1_T2_PF = P1_T1_PF

SETTINGS = {
    'metric': 'IGD',
    'n_pf': 1000,
    'P1': {'T1': P1_T1_PF, 'T2': P1_T2_PF},
}

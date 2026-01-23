import pkgutil
import io
import scipy.io
import numpy as np
from ddmtolab.Methods.mtop import MTOP


class MTMOInstances:
    """
    Additional Multi-Task Multi-Objective Optimization (MTMO) benchmark problems.

    These problems consist of two multi-objective optimization tasks with different
    characteristics to test knowledge transfer capabilities.

    Attributes
    ----------
    data_dir : str
        The directory path for problem data files.
    """

    def __init__(self):
        self.data_dir = 'data_mtmo_instance'

    def P1(self) -> MTOP:
        """
        Generates MTMO Instance 1: **T1 (ZDT4_R, Rastrigin) vs T2 (ZDT4_G, Griewank)**.

        Both tasks are 2-objective, 10-dimensional.

        - T1: Modified ZDT1-like with Rastrigin component in g-function. PF is continuous, convex.
        - T2: Modified ZDT2-like with Griewank component in g-function. PF is continuous, non-convex.
        - Relationship: Different g-functions create different landscape difficulties.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 10

        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTMO',
                                       f'{self.data_dir}/M_MTMO_Instance.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        M = mat_data['M']

        def T1(x):
            """Task 1: ZDT4_R - Rastrigin component"""
            x = np.atleast_2d(x)
            D = x.shape[1]
            # M = 1 (identity), y = x[:, 1:]
            y = x[:, 1:]
            gx = 1.0 + 10.0 * (D - 1) + np.sum(y ** 2 - 10.0 * np.cos(4.0 * np.pi * y), axis=1)
            f1 = x[:, 0]
            f2 = gx * (1.0 - np.sqrt(x[:, 0] / gx))
            return np.column_stack([f1, f2])

        def T2(x):
            """Task 2: ZDT4_G - Griewank component"""
            x = np.atleast_2d(x)
            D = x.shape[1]

            y = (M @ x[:, 1:].T).T

            gx = 2.0 + np.sum(y ** 2, axis=1) / 4000.0

            gx_2 = np.ones(x.shape[0])
            for i in range(2, D + 1):
                gx_2 *= np.cos(x[:, i - 1] / np.sqrt(i))

            gx = gx - gx_2

            f1 = x[:, 0]
            f2 = gx * (1.0 - np.sqrt(f1 / gx))

            return np.column_stack([f1, f2])

        # Task 1 bounds: Rastrigin uses [-5, 5]
        lb1 = np.array([0.0] + [-5.0] * (dim - 1))
        ub1 = np.array([1.0] + [5.0] * (dim - 1))

        # Task 2 bounds: Griewank uses [-512, 512]
        lb2 = np.array([0.0] + [-512.0] * (dim - 1))
        ub2 = np.array([1.0] + [512.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P2(self) -> MTOP:
        """
        Generates MTMO Instance 2: **T1 (ZDT4_RC, Rastrigin + Constraint) vs T2 (ZDT4_A, Ackley)**.

        T1 is 2-objective with 1 constraint (10-dimensional).
        T2 is 2-objective without constraints (10-dimensional).

        - T1: Modified ZDT1-like with Rastrigin component and a sinusoidal constraint.
              PF is continuous, convex, but partially infeasible.
        - T2: Modified ZDT2-like with Ackley component in g-function. PF is continuous, non-convex.
        - Relationship: One task has constraints while the other doesn't, testing constraint handling transfer.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 10

        # Load rotation matrix M
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTMO',
                                       f'{self.data_dir}/M_MTMO_Instance.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        M = mat_data['M']

        def T1(x):
            """Task 1: ZDT4_RC - Rastrigin component (constrained)"""
            x = np.atleast_2d(x)
            D = x.shape[1]
            # M = 1 (identity), y = x[:, 1:]
            y = x[:, 1:]

            gx = 1.0 + 10.0 * (D - 1) + np.sum(y ** 2 - 10.0 * np.cos(4.0 * np.pi * y), axis=1)
            f1 = x[:, 0]
            f2 = gx * (1.0 - np.sqrt(x[:, 0] / gx))
            return np.column_stack([f1, f2])

        def T1_constraint(x):
            """Constraint for Task 1: Sinusoidal constraint (ZDT4_RC)"""
            x = np.atleast_2d(x)
            D = x.shape[1]
            y = x[:, 1:]
            gx = 1.0 + 10.0 * (D - 1) + np.sum(y ** 2 - 10.0 * np.cos(4.0 * np.pi * y), axis=1)
            f1 = x[:, 0]
            f2 = gx * (1.0 - np.sqrt(x[:, 0] / gx))

            theta = -0.05 * np.pi
            a, b, c, d, e = 40.0, 5.0, 1.0, 6.0, 0.0

            constraint = (a * np.abs(np.sin(b * np.pi *
                                            (np.sin(theta) * (f2 - e) + np.cos(theta) * f1) ** c)) ** d) - \
                         np.cos(theta) * (f2 - e) + np.sin(theta) * f1

            # Constraint should be <= 0 (violation when > 0)
            constraint = np.where(constraint < 0, 0, constraint)
            return constraint.reshape(-1, 1)

        def T2(x):
            """Task 2: ZDT4_A - Ackley component (no constraint)"""
            x = np.atleast_2d(x)
            D = x.shape[1]

            y = (M @ x[:, 1:].T).T

            # Ackley-like g function
            sum_sq = np.sum(y ** 2, axis=1)
            sum_cos = np.sum(np.cos(2.0 * np.pi * y), axis=1)
            gx = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / (D - 1))) - \
                 np.exp(sum_cos / (D - 1)) + 21.0 + np.e

            f1 = x[:, 0]
            f2 = gx * (1.0 - np.sqrt(f1 / gx))
            return np.column_stack([f1, f2])

        # Task 1 bounds: Rastrigin uses [-5, 5]
        lb1 = np.array([0.0] + [-5.0] * (dim - 1))
        ub1 = np.array([1.0] + [5.0] * (dim - 1))

        # Task 2 bounds: Ackley uses [-32, 32]
        lb2 = np.array([0.0] + [-32.0] * (dim - 1))
        ub2 = np.array([1.0] + [32.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, constraint_func=T1_constraint,
                         lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem


# --- True Pareto Front (PF) Functions ---

def P1_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for Instance 1, Task 1.

    The PF is inverse square root (convex): f_2 = 1 - sqrt(f_1).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) representing the PF points.
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


P1_T2_PF = P1_T1_PF
P2_T1_PF = P1_T1_PF
P2_T2_PF = P1_T1_PF

# Settings dictionary for the new instances
SETTINGS = {
    'metric': 'IGD',
    'n_pf': 1000,
    'pf_path': './MOReference',
    'P1': {'T1': P1_T1_PF, 'T2': P1_T2_PF},
    'P2': {'T1': P2_T1_PF, 'T2': P2_T2_PF},
}
import scipy.io
import numpy as np
from Methods.mtop import MTOP

class CEC17MTMO:
    """
    Implementation of the CEC 2017 Competition on Evolutionary Multi-Task Multi-Objective
    Optimization (MTMO) benchmark problems P1 to P9.

    These problems consist of two multi-objective optimization tasks (MO-tasks)
    with shared variables, designed to test knowledge transfer in the presence of
    multiple conflicting objectives. All tasks are minimization problems.

    Parameters
    ----------
    mat_dir : str, optional
        Directory path to the MAT files containing rotation matrices (M), shift
        vectors (S), and other parameters for the tasks (default: '../Problems/MTMO/data_cec17mtmo').

    Attributes
    ----------
    mat_dir : str
        The directory path for problem data files.
    """

    def __init__(self, mat_dir='../Problems/MTMO/data_cec17mtmo'):
        self.mat_dir = mat_dir

    def P1(self) -> MTOP:
        """
        Generates Problem 1: **T1 (ZDT3-like) vs T2 (ZDT2-like)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: Modified ZDT3-like, PF is discontinuous, non-convex (Curved, Piecewise).
        - T2: Modified ZDT2-like, PF is continuous, non-convex.
        - Relationship: Decision space overlap exists only in :math:`x_1` dimension (:math:`[0, 1]`).

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        def T1(x):
            x = np.atleast_2d(x)
            q = 1.0 + np.sum(x[:, 1:] ** 2, axis=1)
            x1 = x[:, 0]
            f1 = q * np.cos(np.pi * x1 / 2)
            f2 = q * np.sin(np.pi * x1 / 2)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            q = 1.0 + 9.0 / (dim - 1) * np.sum(np.abs(x[:, 1:]), axis=1)
            x1 = x[:, 0]
            f1 = x1
            f2 = q * (1.0 - (x1 / q) ** 2)
            return np.vstack([f1, f2]).T

        lb = np.array([0.0] + [-100.0] * (dim - 1))
        ub = np.array([1.0] + [100.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self) -> MTOP:
        """
        Generates Problem 2: **T1 (ZDT2-like, Rosenbrock) vs T2 (ZDT3-like, Rotated)**.

        Both tasks are 2-objective, 10-dimensional.

        - T1: Modified ZDT2-like with Rosenbrock-like component.
        - T2: Modified ZDT3-like with rotated non-linear component (Mcm2).
        - Relationship: Decision variables are coupled and search spaces are different.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 10
        Mcm2 = scipy.io.loadmat(f'{self.mat_dir}/Mcm2.mat')['Mcm2']
        Scm2 = scipy.io.loadmat(f'{self.mat_dir}/Scm2.mat')['Scm2'].flatten()

        def T1(x):
            x = np.atleast_2d(x)
            # Rosenbrock-like component in g-function
            q = 1 + np.sum(100 * ((x[:, 1:-1] ** 2 - x[:, 2:]) ** 2 + (1 - x[:, 1:-1]) ** 2), axis=1)
            f1 = x[:, 0]
            f2 = q * (1 - (x[:, 0] / q) ** 2)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            # Shifted and rotated component
            z = (x[:, 1:] - Scm2) @ Mcm2.T
            q = 1 + 9 / (dim - 1) * np.sum(np.abs(z), axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2)
            f2 = q * np.sin(np.pi * x[:, 0] / 2)
            return np.vstack([f1, f2]).T

        lb = np.array([0.0] + [-5.0] * (dim - 1))
        ub = np.array([1.0] + [5.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self) -> MTOP:
        """
        Generates Problem 3: **T1 (ZDT3-like, Rastrigin) vs T2 (ZDT1-like, Ackley)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: Modified ZDT3-like with Rastrigin-like component in :math:`g`. PF is discontinuous, non-convex.
        - T2: Modified ZDT1-like with Ackley-like component in :math:`g`. PF is continuous, convex.
        - Relationship: Tasks have different search spaces and different :math:`g`-functions.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 1:]
            # Rastrigin-like component in g
            q = 1.0 + np.sum(part ** 2 - 10.0 * np.cos(2.0 * np.pi * part) + 10.0, axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            r = x.shape[1]
            part = x[:, 1:]
            # Ackley-like component in g
            term1 = 21.0 + np.e
            sum_sq = np.sum(part ** 2, axis=1)
            sum_cos = np.sum(np.cos(2.0 * np.pi * part), axis=1)
            q = term1 - 20.0 * np.exp(-0.2 * np.sqrt((1.0 / (r - 1)) * sum_sq)) - np.exp((1.0 / (r - 1)) * sum_cos)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        lb1 = np.concatenate(([0.0], -2.0 * np.ones(dim - 1)))
        ub1 = np.concatenate(([1.0], 2.0 * np.ones(dim - 1)))

        lb2 = np.concatenate(([0.0], -1.0 * np.ones(dim - 1)))
        ub2 = np.concatenate(([1.0], 1.0 * np.ones(dim - 1)))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P4(self) -> MTOP:
        """
        Generates Problem 4: **T1 (ZDT1-like, Sphere) vs T2 (ZDT1-like, Rastrigin)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: Modified ZDT1-like with Sphere component in :math:`g`. PF is continuous, convex.
        - T2: Modified ZDT1-like with shifted Rastrigin component (Sph2) in :math:`g`. PF is continuous, convex.
        - Relationship: The :math:`g`-functions and search spaces are different.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50
        Sph2 = scipy.io.loadmat(f'{self.mat_dir}/Sph2.mat')['Sph2'].flatten()

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 1:]
            # Sphere component in g
            q = 1.0 + np.sum(part ** 2, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            z = x[:, 1:] - Sph2
            # Shifted Rastrigin component in g
            q = 1.0 + np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        lb = np.array([0.0] + [-100.0] * (dim - 1))
        ub = np.array([1.0] + [100.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self) -> MTOP:
        """
        Generates Problem 5: **T1 (ZDT3-like, Rotated Sphere) vs T2 (ZDT2-like, Rotated Rastrigin)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: Modified ZDT3-like with rotated and shifted Sphere component in :math:`g` (Mpm1, Spm1). PF is discontinuous, non-convex.
        - T2: Modified ZDT2-like with rotated Rastrigin component in :math:`g` (Mpm2). PF is continuous, non-convex.
        - Relationship: Different problem landscapes and global optimum locations in the search space.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50
        Mpm1 = scipy.io.loadmat(f'{self.mat_dir}/Mpm1.mat')['Mpm1']
        Spm1 = scipy.io.loadmat(f'{self.mat_dir}/Spm1.mat')['Spm1'].flatten()
        Mpm2 = scipy.io.loadmat(f'{self.mat_dir}/Mpm2.mat')['Mpm2']

        def T1(x):
            x = np.atleast_2d(x)
            # Rotated and shifted Sphere in g
            z = (x[:, 1:] - Spm1) @ Mpm1.T
            q = 1.0 + np.sum(z ** 2, axis=1)
            a = np.cos(np.pi * x[:, 0] / 2.0)
            b = np.sin(np.pi * x[:, 0] / 2.0)
            f1 = q * a
            f2 = q * b
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            # Rotated Rastrigin in g
            z = x[:, 1:] @ Mpm2.T
            term = z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0
            q = 1.0 + np.sum(term, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - (x[:, 0] / q) ** 2)
            return np.vstack([f1, f2]).T

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P6(self) -> MTOP:
        """
        Generates Problem 6: **T1 (ZDT3-like, Griewank) vs T2 (ZDT3-like, Ackley)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: Modified ZDT3-like with Griewank component in :math:`g`. PF is discontinuous, non-convex.
        - T2: Modified ZDT3-like with shifted Ackley component (Spl2) in :math:`g`. PF is discontinuous, non-convex.
        - Relationship: PF shapes are similar, but :math:`g`-functions and search spaces are different.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50
        Spl2 = scipy.io.loadmat(f'{self.mat_dir}/Spl2.mat')['Spl2'].flatten()

        def T1(x):
            x = np.atleast_2d(x)
            r = x.shape[1]
            # Griewank component in g
            sum_sq = np.sum(x[:, 1:] ** 2, axis=1)
            denom = np.sqrt(np.arange(1, r))
            cos_terms = np.cos(x[:, 1:] / denom)
            prod_cos = np.prod(cos_terms, axis=1)
            q = 2.0 + (1.0 / 4000.0) * sum_sq - prod_cos
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            r = x.shape[1]
            z = x[:, 1:] - Spl2
            # Shifted Ackley component in g
            sum_sq = np.sum(z ** 2, axis=1)
            sum_cos = np.sum(np.cos(2.0 * np.pi * z), axis=1)
            q = 21.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt((1.0 / (r - 1)) * sum_sq)) - np.exp(
                (1.0 / (r - 1)) * sum_cos)
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        lb1 = np.concatenate(([0.0], -50.0 * np.ones(dim - 1)))
        ub1 = np.concatenate(([1.0], 50.0 * np.ones(dim - 1)))

        lb2 = np.concatenate(([0.0], -100.0 * np.ones(dim - 1)))
        ub2 = np.concatenate(([1.0], 100.0 * np.ones(dim - 1)))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P7(self) -> MTOP:
        """
        Generates Problem 7: **T1 (ZDT3-like, Rosenbrock) vs T2 (ZDT1-like, Sphere)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: Modified ZDT3-like with Rosenbrock component in :math:`g`. PF is discontinuous, non-convex.
        - T2: Modified ZDT1-like with Sphere component in :math:`g`. PF is continuous, convex.
        - Relationship: Highly multi-modal :math:`g`-function in T1 and different PF shapes.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 1:-1]
            next_part = x[:, 2:]
            # Rosenbrock component in g
            q = 1.0 + np.sum(100.0 * (part ** 2 - next_part) ** 2 + (1.0 - part) ** 2, axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            # Sphere component in g
            q = 1.0 + np.sum(x[:, 1:] ** 2, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        lb = np.full(dim, -80.0)
        ub = np.full(dim, 80.0)
        lb[0] = 0.0
        ub[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self) -> MTOP:
        """
        Generates Problem 8: **T1 (DTLZ1-like, 3-obj) vs T2 (ZDT2-like, 2-obj)**.

        T1 is 3-objective, T2 is 2-objective. Both are 20-dimensional.

        - T1: Modified DTLZ1-like with Rosenbrock component in :math:`g`. PF is a **plane** (linear).
        - T2: Modified ZDT2-like with rotated Sphere component in :math:`g` (Mnm2). PF is continuous, non-convex.
        - Relationship: Tasks have different number of objectives and different PF shapes/geometries.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 20
        Mnm2 = scipy.io.loadmat(f'{self.mat_dir}/Mnm2.mat')['Mnm2']

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 2:-1]
            next_part = x[:, 3:]
            # Rosenbrock component in g (DTLZ1-like)
            q = 1.0 + np.sum(100 * (part ** 2 - next_part) ** 2 + (1.0 - part) ** 2, axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2) * np.cos(np.pi * x[:, 1] / 2)
            f2 = q * np.cos(np.pi * x[:, 0] / 2) * np.sin(np.pi * x[:, 1] / 2)
            f3 = q * np.sin(np.pi * x[:, 0] / 2)
            return np.vstack([f1, f2, f3]).T

        def T2(x):
            x = np.atleast_2d(x)
            # Rotated Sphere component in g (ZDT2-like)
            z = x[:, 2:] @ Mnm2.T
            q = 1.0 + np.sum(z ** 2, axis=1)
            s = 0.5 * np.sum(x[:, :2], axis=1)
            f1 = s
            f2 = q * (1.0 - (s / q) ** 2)
            return np.vstack([f1, f2]).T

        lb = np.full(dim, -20.0)
        ub = np.full(dim, 20.0)
        lb[:2] = 0.0
        ub[:2] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P9(self) -> MTOP:
        """
        Generates Problem 9: **T1 (DTLZ1-like, 3-obj) vs T2 (ZDT2-like, 2-obj)**.

        T1 is 3-objective (25-dimensional), T2 is 2-objective (50-dimensional).

        - T1: Modified DTLZ1-like with shifted Griewank component (Snl1) in :math:`g`. PF is a **plane** (linear).
        - T2: Modified ZDT2-like with Ackley component in :math:`g`. PF is continuous, non-convex.
        - Relationship: Different objectives, different dimensions, and different PF shapes.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        Snl1 = scipy.io.loadmat(f'{self.mat_dir}/Snl1.mat')['Snl1'].flatten()
        dim1 = 25
        dim2 = 50

        def T1(x):
            x = np.atleast_2d(x)
            z = x[:, 2:25] - Snl1
            a = np.arange(1, 24)
            # Shifted Griewank component in g (DTLZ1-like)
            q = 2.0 + (1.0 / 4000.0) * np.sum(z ** 2, axis=1) - np.prod(np.cos(z / np.sqrt(a)), axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2) * np.cos(np.pi * x[:, 1] / 2)
            f2 = q * np.cos(np.pi * x[:, 0] / 2) * np.sin(np.pi * x[:, 1] / 2)
            f3 = q * np.sin(np.pi * x[:, 0] / 2)
            return np.vstack([f1, f2, f3]).T

        def T2(x):
            x = np.atleast_2d(x)
            r = x.shape[1]
            # Ackley component in g (ZDT2-like)
            q = 21.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(x[:, 2:] ** 2, axis=1) / (r - 2))) \
                - np.exp(np.sum(np.cos(2.0 * np.pi * x[:, 2:]), axis=1) / (r - 2))
            s = 0.5 * np.sum(x[:, :2], axis=1)
            f1 = s
            f2 = q * (1.0 - (s / q) ** 2)
            return np.vstack([f1, f2]).T

        lb1 = np.full(dim1, -50.0)
        ub1 = np.full(dim1, 50.0)
        lb1[:2] = 0.0
        ub1[:2] = 1.0

        lb2 = np.full(dim2, -100.0)
        ub2 = np.full(dim2, 100.0)
        lb2[:2] = 0.0
        ub2[:2] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim1, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim2, lower_bound=lb2, upper_bound=ub2)
        return problem


# --- True Pareto Front (PF) Functions ---

def P1_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P1, Task 1.

    The PF is a quarter circle arc (non-convex).

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
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def P1_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P1, Task 2.

    The PF is parabolic-like (non-convex): :math:`f_2 = 1 - f_1^2`.

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
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P2_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P2, Task 1.

    The PF is parabolic-like (non-convex): :math:`f_2 = 1 - f_1^2`.

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
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P2_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P2, Task 2.

    The PF is a quarter circle arc (non-convex).

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
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def P3_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P3, Task 1.

    The PF is a quarter circle arc (non-convex).

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
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def P3_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P3, Task 2.

    The PF is inverse square root (convex): :math:`f_2 = 1 - \\sqrt{f_1}`.

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


def P4_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P4, Task 1.

    The PF is inverse square root (convex): :math:`f_2 = 1 - \\sqrt{f_1}`.

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


def P4_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P4, Task 2.

    The PF is inverse square root (convex): :math:`f_2 = 1 - \\sqrt{f_1}`.

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


def P5_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P5, Task 1.

    The PF is a quarter circle arc (non-convex).

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
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def P5_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P5, Task 2.

    The PF is parabolic-like (non-convex): :math:`f_2 = 1 - f_1^2`.

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
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P6_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P6, Task 1.

    The PF is a quarter circle arc (non-convex).

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
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def P6_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P6, Task 2.

    The PF is a quarter circle arc (non-convex).

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
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def P7_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P7, Task 1.

    The PF is a quarter circle arc (non-convex).

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
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def P7_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P7, Task 2.

    The PF is inverse square root (convex): :math:`f_2 = 1 - \\sqrt{f_1}`.

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


def P8_T1_PF(N, M=3) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P8, Task 1 (3-objective).

    The PF is a portion of a **sphere** (linear for DTLZ1-like).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N', 3) representing the PF points.
    """
    n_sqrt = int(np.sqrt(N))
    theta = np.linspace(0, np.pi / 2, n_sqrt)
    phi = np.linspace(0, np.pi / 2, n_sqrt)

    points = []
    for t in theta:
        for p in phi:
            # DTLZ1-like PF: f1 + f2 + f3 = 1
            f1 = np.sin(t) * np.cos(p)
            f2 = np.sin(t) * np.sin(p)
            f3 = np.cos(t)
            points.append([f1, f2, f3])

    return np.array(points)


def P8_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P8, Task 2 (2-objective).

    The PF is parabolic-like (non-convex): :math:`f_2 = 1 - f_1^2`.

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
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P9_T1_PF(N, M=3) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P9, Task 1 (3-objective).

    The PF is a portion of a **sphere** (linear for DTLZ1-like).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N', 3) representing the PF points.
    """
    n_sqrt = int(np.sqrt(N))
    theta = np.linspace(0, np.pi / 2, n_sqrt)
    phi = np.linspace(0, np.pi / 2, n_sqrt)

    points = []
    for t in theta:
        for p in phi:
            # DTLZ1-like PF: f1 + f2 + f3 = 1
            f1 = np.sin(t) * np.cos(p)
            f2 = np.sin(t) * np.sin(p)
            f3 = np.cos(t)
            points.append([f1, f2, f3])

    return np.array(points)


def P9_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P9, Task 2 (2-objective).

    The PF is parabolic-like (non-convex): :math:`f_2 = 1 - f_1^2`.

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
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


SETTINGS = {
    'metric': 'IGD',
    'n_pf': 1000,
    'pf_path': './MOReference',
    'P1': {'T1': P1_T1_PF, 'T2': P1_T2_PF},
    'P2': {'T1': P2_T1_PF, 'T2': P2_T2_PF},
    'P3': {'T1': P3_T1_PF, 'T2': P3_T2_PF},
    'P4': {'T1': P4_T1_PF, 'T2': P4_T2_PF},
    'P5': {'T1': P5_T1_PF, 'T2': P5_T2_PF},
    'P6': {'T1': P6_T1_PF, 'T2': P6_T2_PF},
    'P7': {'T1': P7_T1_PF, 'T2': P7_T2_PF},
    'P8': {'T1': P8_T1_PF, 'T2': P8_T2_PF},
    'P9': {'T1': P9_T1_PF, 'T2': P9_T2_PF},
}
"""Rotated and shifted single-objective benchmark landscapes.

The same 9 textbook functions as
:mod:`~ddmtolab.Problems.STSO.classical_so`, but with a non-identity rotation
matrix and a non-zero offset drawn from a fixed seed, so the global optimum is
no longer at the origin.
"""

from ddmtolab.Problems.BasicFunctions.basic_functions import *
from ddmtolab.Methods.mtop import MTOP
import numpy as np


def _generate_rotation_and_offset(D, seed=1234):
    """
    Generate rotation matrix and offset vector for STSOtest problems.

    Parameters
    ----------
    D : int
        Dimensionality of the search space.
    seed : int, optional
        Random seed for reproducibility (default is 1234).

    Returns
    -------
    M : numpy.ndarray
        Orthogonal rotation matrix of shape (D, D).
    o : numpy.ndarray
        Offset vector of shape (1, D).
    """
    np.random.seed(seed)
    random_matrix = np.random.randn(D, D)
    M, _ = np.linalg.qr(random_matrix)
    o = np.random.uniform(0., 10., (1, D))
    return M, o


class STSOtest:
    """
    Modified Single-Task Optimization (STSOtest) benchmark problems.

    This class provides a set of standard single-objective optimization
    benchmark functions (e.g., Ackley, Rastrigin, Sphere) configured as
    Multi-Task Optimization Problems (MTOPs) with only one task.
    Unlike CLASSICALSO, this class uses non-identity rotation matrices (M)
    and non-zero offset vectors (o) to create different but corresponding
    problem instances.
    """

    problem_information = {
        'n_cases': 9,
        'n_tasks': '1',
        'n_dims': 'D',
        'n_objs': '1',
        'n_cons': '0',
        'type': 'synthetic',
    }

    def __init__(self):
        pass

    def P1(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Ackley** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Ackley task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Ackley(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -100.0)
        ub = np.full(D, 100.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Elliptic** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Elliptic task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Elliptic(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -100.0)
        ub = np.full(D, 100.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Griewank** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Griewank task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Griewank(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -100.0)
        ub = np.full(D, 100.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P4(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Rastrigin** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Rastrigin task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Rastrigin(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -100.0)
        ub = np.full(D, 100.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Rosenbrock** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Rosenbrock task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Rosenbrock(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -100.0)
        ub = np.full(D, 100.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P6(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Schwefel** function (F6).

        The search space is set to [-500.0, 500.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Schwefel task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -500.0)
        ub = np.full(D, 500.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P7(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Schwefel 2.22** function (F7).

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Schwefel 2.22 task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel2(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -100.0)
        ub = np.full(D, 100.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Sphere** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Sphere task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Sphere(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -100.0)
        ub = np.full(D, 100.0)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

    def P9(self, D=50) -> MTOP:
        """
        Generates a single-task MTOP based on the **Weierstrass** function.

        The search space is set to [-0.5, 0.5] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Weierstrass task.
        """
        M, o = _generate_rotation_and_offset(D)

        def Task(x):
            x = np.atleast_2d(x)
            return Weierstrass(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -0.5)
        ub = np.full(D, 0.5)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

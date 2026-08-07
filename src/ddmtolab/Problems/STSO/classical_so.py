"""Classical single-objective benchmark landscapes.

9 unconstrained textbook functions (Ackley, Rastrigin, Sphere, ...) exposed as
single-task MTOPs with an identity rotation and a zero shift, so the global
optimum sits at the origin. Use :mod:`~ddmtolab.Problems.STSO.stsotest` for the
rotated and shifted variants.
"""

from ddmtolab.Problems.BasicFunctions.basic_functions import *
from ddmtolab.Methods.mtop import MTOP
import numpy as np

class CLASSICALSO:
    """
    Classical Single-Task Optimization (CLASSICALSO) benchmark problems.

    This class provides a set of standard single-objective optimization
    benchmark functions (e.g., Ackley, Rastrigin, Sphere) configured as
    Multi-Task Optimization Problems (MTOPs) with only one task.
    This serves as a baseline for comparing single-task solvers or as
    individual tasks in a multi-task setting.
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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Ackley task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Elliptic task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Griewank task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Rastrigin task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Rosenbrock task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Schwefel task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Schwefel 2.22 task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Sphere task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

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

        Parameters
        ----------
        D : int, optional
            Number of decision variables (default is 50).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Weierstrass task.
        """
        M = np.eye(D, dtype=float)
        o = np.zeros((1, D), dtype=float)

        def Task(x):
            x = np.atleast_2d(x)
            return Weierstrass(x, M, o, 0.0)

        problem = MTOP()
        lb = np.full(D, -0.5)
        ub = np.full(D, 0.5)
        problem.add_task(Task, dim=D, lower_bound=lb, upper_bound=ub)
        return problem

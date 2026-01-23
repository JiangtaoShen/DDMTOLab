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

    Parameters
    ----------
    dim : int, optional
        The dimensionality of the search space for all tasks (default is 50).

    Attributes
    ----------
    dim : int
        The dimensionality of the problem.
    M : numpy.ndarray
        Identity matrix used for rotation/transformation (defaults to identity).
    o : numpy.ndarray
        Offset vector used for shifting the optimum (defaults to zero vector).
    """

    def __init__(self, dim=50):
        self.dim = dim
        self.M = np.eye(self.dim, dtype=float)
        self.o = np.zeros((1, self.dim), dtype=float)

    def P1(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Ackley** function.

        The search space is set to [-100.0, 100.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Ackley task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Ackley(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Elliptic** function.

        The search space is set to [-100.0, 100.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Elliptic task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Elliptic(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Griewank** function.

        The search space is set to [-100.0, 100.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Griewank task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Griewank(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P4(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Rastrigin** function.

        The search space is set to [-100.0, 100.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Rastrigin task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Rastrigin(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Rosenbrock** function.

        The search space is set to [-100.0, 100.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Rosenbrock task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Rosenbrock(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P6(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Schwefel** function (F6).

        The search space is set to [-500.0, 500.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Schwefel task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -500.0)
        ub = np.full(self.dim, 500.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P7(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Schwefel 2.22** function (F7).

        The search space is set to [-100.0, 100.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Schwefel 2.22 task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel2(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Sphere** function.

        The search space is set to [-100.0, 100.0] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Sphere task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Sphere(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P9(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Weierstrass** function.

        The search space is set to [-0.5, 0.5] in all dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the Weierstrass task.
        """
        def Task(x):
            x = np.atleast_2d(x)
            return Weierstrass(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -0.5)
        ub = np.full(self.dim, 0.5)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem
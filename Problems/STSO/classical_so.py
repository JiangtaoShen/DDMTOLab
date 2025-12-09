from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP

class CLASSICALSO:

    def __init__(self, dim=50):
        self.dim = dim
        self.M = np.eye(self.dim, dtype=float)
        self.o = np.zeros((1, self.dim), dtype=float)

    def P1(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Ackley(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Elliptic(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Griewank(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P4(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Rastrigin(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Rosenbrock(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P6(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -500.0)
        ub = np.full(self.dim, 500.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P7(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel2(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Sphere(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P9(self):
        def Task(x):
            x = np.atleast_2d(x)
            return Weierstrass(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -0.5)
        ub = np.full(self.dim, 0.5)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem
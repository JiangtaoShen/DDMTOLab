import scipy.io
from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP
import numpy as np
import os


class STOP:
    """
    Implementation of STOP (Scalable Test Problem Generator for Sequential Transfer Optimization)
    benchmark problems.

    These problems are designed to challenge algorithms with many tasks derived from
    various benchmark functions with different characteristics, testing transfer learning
    across many similar but distinct tasks.

    Reference:
    X. Xue et al., "A Scalable Test Problem Generator for Sequential Transfer Optimization,"
    IEEE Trans. Cybern., vol. 55, no. 5, pp. 2110-2123, 2025.

    Parameters
    ----------
    mat_dir : str, optional
        Directory path to the MAT files containing task data
        (default: '../Problems/MTSO/data_stop').

    Attributes
    ----------
    mat_dir : str
        The directory path for problem data files.
    """

    def __init__(self, mat_dir='../Problems/MTSO/data_stop'):
        self.mat_dir = mat_dir

    @staticmethod
    def _get_function_and_bounds(func_name, dim):
        """
        Get the objective function and bounds based on function name.

        Parameters
        ----------
        func_name : str
            Name of the function (e.g., 'Sphere', 'Ellipsoid', etc.)
        dim : int
            Dimension of the problem

        Returns
        -------
        tuple
            (function, lower_bound, upper_bound)
        """
        func_name = func_name.strip()

        if func_name == 'Sphere':
            func = lambda x: Sphere(x, np.eye(dim), np.zeros(dim), 0)
            lb = np.full(dim, -100.0)
            ub = np.full(dim, 100.0)
        elif func_name == 'Ellipsoid':
            func = lambda x: STOP.S_Ellipsoid(x, np.zeros(dim))
            lb = np.full(dim, -50.0)
            ub = np.full(dim, 50.0)
        elif func_name == 'Schwefel':
            func = lambda x: STOP.S_Schwefel(x, np.zeros(dim))
            lb = np.full(dim, -30.0)
            ub = np.full(dim, 30.0)
        elif func_name == 'Quartic':
            func = lambda x: STOP.S_Quartic(x, np.zeros(dim))
            lb = np.full(dim, -5.0)
            ub = np.full(dim, 5.0)
        elif func_name == 'Ackley':
            func = lambda x: Ackley(x, np.eye(dim), np.zeros(dim), 0)
            lb = np.full(dim, -32.0)
            ub = np.full(dim, 32.0)
        elif func_name == 'Rastrigin':
            func = lambda x: Rastrigin(x, np.eye(dim), np.zeros(dim), 0)
            lb = np.full(dim, -10.0)
            ub = np.full(dim, 10.0)
        elif func_name == 'Griewank':
            func = lambda x: Griewank(x, np.eye(dim), np.zeros(dim), 0)
            lb = np.full(dim, -200.0)
            ub = np.full(dim, 200.0)
        elif func_name == 'Levy':
            func = lambda x: STOP.S_Levy(x, np.zeros(dim))
            lb = np.full(dim, -20.0)
            ub = np.full(dim, 20.0)
        else:
            raise ValueError(f"Unknown function name: {func_name}")

        return func, lb, ub

    def _create_task_function(self, func_name, dim, opt):
        """
        Create task function based on function name.

        Parameters
        ----------
        func_name : str
            Name of the function
        dim : int
            Dimension
        opt : np.ndarray
            Optimal point (shift vector)

        Returns
        -------
        callable
            Task function
        """
        func_name = func_name.strip()

        if func_name == 'Levy':
            return lambda x, o=opt: self.S_Levy(x, o)
        elif func_name == 'Ackley':
            return lambda x, o=opt: Ackley(x, np.eye(dim), o, 0)
        elif func_name == 'Schwefel':
            return lambda x, o=opt: self.S_Schwefel(x, o)
        elif func_name == 'Quartic':
            return lambda x, o=opt: self.S_Quartic(x, o)
        elif func_name == 'Rastrigin':
            return lambda x, o=opt: Rastrigin(x, np.eye(dim), o, 0)
        elif func_name == 'Griewank':
            return lambda x, o=opt: Griewank(x, np.eye(dim), o, 0)
        elif func_name == 'Ellipsoid':
            return lambda x, o=opt: self.S_Ellipsoid(x, o)
        elif func_name == 'Sphere':
            return lambda x, o=opt: Sphere(x, np.eye(dim), o, 0)
        else:
            raise ValueError(f"Unknown function: {func_name}")

    def _load_and_create_problem(self, mat_filename, dim, task_num):
        """
        Generic method to load MAT file and create MTOP problem.

        Parameters
        ----------
        mat_filename : str
            MAT file name
        dim : int
            Problem dimension
        task_num : int
            Number of tasks

        Returns
        -------
        MTOP
            Multi-Task Optimization Problem
        """
        mat_file = os.path.join(self.mat_dir, mat_filename)
        mat_data = scipy.io.loadmat(mat_file)

        target = mat_data['target']
        sources = mat_data['sources'] if task_num > 1 else None

        problem = MTOP()

        # Add target task
        target_opt = target[0, 0]['x_best'].flatten()
        target_name = target[0, 0]['name'][0]
        _, lb, ub = self._get_function_and_bounds(target_name, dim)
        task_function = self._create_task_function(target_name, dim, target_opt)
        problem.add_task(task_function, dim=dim, lower_bound=lb, upper_bound=ub)

        # Add source tasks
        if task_num > 1:
            for i in range(task_num - 1):
                source_opt = sources[0, i]['x_best'].flatten()
                source_name = sources[0, i]['name'][0]
                _, lb, ub = self._get_function_and_bounds(source_name, dim)
                task_function = self._create_task_function(source_name, dim, source_opt)
                problem.add_task(task_function, dim=dim, lower_bound=lb, upper_bound=ub)

        return problem

    def STOP1(self, task_num=10) -> MTOP:
        """STOP Problem 1: Sphere-Ta-hh2-d50-k49"""
        return self._load_and_create_problem('Sphere-Ta-hh2-d50-k49.mat', 50, task_num)

    def STOP2(self, task_num=10) -> MTOP:
        """STOP Problem 2: Ellipsoid-Te-hh2-d25-k49"""
        return self._load_and_create_problem('Ellipsoid-Te-hh2-d25-k49.mat', 25, task_num)

    def STOP3(self, task_num=10) -> MTOP:
        """STOP Problem 3: Schwefel-Ta-hh2-d30-k49"""
        return self._load_and_create_problem('Schwefel-Ta-hh2-d30-k49.mat', 30, task_num)

    def STOP4(self, task_num=10) -> MTOP:
        """STOP Problem 4: Quartic-Te-hh2-d50-k49"""
        return self._load_and_create_problem('Quartic-Te-hh2-d50-k49.mat', 50, task_num)

    def STOP5(self, task_num=10) -> MTOP:
        """STOP Problem 5: Ackley-Ta-hm1-d25-k49"""
        return self._load_and_create_problem('Ackley-Ta-hm1-d25-k49.mat', 25, task_num)

    def STOP6(self, task_num=10) -> MTOP:
        """STOP Problem 6: Rastrigin-Te-hm2-d50-k49"""
        return self._load_and_create_problem('Rastrigin-Te-hm2-d50-k49.mat', 50, task_num)

    def STOP7(self, task_num=10) -> MTOP:
        """STOP Problem 7: Griewank-Ta-hm3-d25-k49"""
        return self._load_and_create_problem('Griewank-Ta-hm3-d25-k49.mat', 25, task_num)

    def STOP8(self, task_num=10) -> MTOP:
        """STOP Problem 8: Levy-Te-hm4-d30-k49"""
        return self._load_and_create_problem('Levy-Te-hm4-d30-k49.mat', 30, task_num)

    def STOP9(self, task_num=10) -> MTOP:
        """STOP Problem 9: Sphere-Ta-hl1-d25-k49"""
        return self._load_and_create_problem('Sphere-Ta-hl1-d25-k49.mat', 25, task_num)

    def STOP10(self, task_num=10) -> MTOP:
        """STOP Problem 10: Rastrigin-Te-hl2-d30-k49"""
        return self._load_and_create_problem('Rastrigin-Te-hl2-d30-k49.mat', 30, task_num)

    def STOP11(self, task_num=10) -> MTOP:
        """STOP Problem 11: Ackley-Ta-hl2-d50-k49"""
        return self._load_and_create_problem('Ackley-Ta-hl2-d50-k49.mat', 50, task_num)

    def STOP12(self, task_num=10) -> MTOP:
        """STOP Problem 12: Ellipsoid-Te-hl1-d50-k49"""
        return self._load_and_create_problem('Ellipsoid-Te-hl1-d50-k49.mat', 50, task_num)

    @staticmethod
    def S_Ellipsoid(var, opt):
        """Ellipsoid function (STOP variant, shifted only)."""
        if var.ndim != 2:
            raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
        ps, D = var.shape
        w = np.arange(D, 0, -1)
        diff = var - opt
        Obj = np.sum((diff ** 2) * w, axis=1)
        return Obj.reshape(-1, 1)

    @staticmethod
    def S_Levy(var, opt):
        """Levy function (STOP variant, shifted only)."""
        if var.ndim != 2:
            raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
        ps, D = var.shape

        w = 1 + (var - opt) / 4
        term1 = np.sin(np.pi * w[:, 0]) ** 2
        wd = w[:, -1]
        term3 = (wd - 1) ** 2 * (1 + np.sin(2 * np.pi * wd) ** 2)

        if D > 1:
            wi = w[:, :-1]
            term2 = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2), axis=1)
        else:
            term2 = 0

        Obj = term1 + term2 + term3
        return Obj.reshape(-1, 1)

    @staticmethod
    def S_Quartic(var, opt):
        """Quartic function with noise (STOP variant, shifted only)."""
        if var.ndim != 2:
            raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
        ps, D = var.shape

        w = np.arange(1, D + 1)
        diff = var - opt
        Obj = np.sum((diff ** 4) * w, axis=1) + np.random.rand(ps)
        return Obj.reshape(-1, 1)

    @staticmethod
    def S_Schwefel(var, opt):
        """Schwefel 2.2 function (STOP variant, shifted only)."""
        if var.ndim != 2:
            raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")

        diff = var - opt
        abs_diff = np.abs(diff)
        Obj = np.sum(abs_diff, axis=1) + np.prod(abs_diff, axis=1)
        return Obj.reshape(-1, 1)
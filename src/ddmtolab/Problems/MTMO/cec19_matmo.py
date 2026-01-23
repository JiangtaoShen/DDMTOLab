import numpy as np
import pkgutil
import io
from ddmtolab.Methods.mtop import MTOP


class CEC19_MaTMO:
    """
    Implementation of CEC19 MaTMO benchmark problems P1-P6.

    CRITICAL: P1, P4, P6 use DTLZ formulation
              P2, P3, P5 use ZDT formulation

    Attributes
    ----------
    data_dir : str
        The directory path for problem data files.
    """

    def __init__(self):
        self.data_dir = 'data_cec19matmo'

    def _load_shift_rotation(self, problem_id, task_id):
        """Load shift vector and rotation matrix."""
        shift_path = f'{self.data_dir}/SVector/S{problem_id}/S{problem_id}_{task_id}.txt'
        rotation_path = f'{self.data_dir}/M/M{problem_id}/M{problem_id}_{task_id}.txt'

        shift_bytes = pkgutil.get_data('ddmtolab.Problems.MTMO', shift_path)
        shift_vector = np.loadtxt(io.BytesIO(shift_bytes))

        rotation_bytes = pkgutil.get_data('ddmtolab.Problems.MTMO', rotation_path)
        rotation_matrix = np.loadtxt(io.BytesIO(rotation_bytes))

        return shift_vector, rotation_matrix

    def _eval_g_function(self, x, shift_vector, rotation_matrix, g_type, lb, ub):
        """Evaluate g-function with shift, rotation, and boundary handling."""
        x = np.atleast_2d(x)

        # Apply shift and rotation - MUST use .T
        z = (x - shift_vector) @ rotation_matrix.T

        # Apply boundary constraints
        z = np.clip(z, lb, ub)

        if g_type == 'Sphere':
            g = np.sum(z ** 2, axis=1)

        elif g_type == 'Rosenbrock':
            g = np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (1 - z[:, :-1]) ** 2, axis=1)

        elif g_type == 'Ackley':
            n = z.shape[1]
            sum_sq = np.sum(z ** 2, axis=1) / n
            sum_cos = np.sum(np.cos(2 * np.pi * z), axis=1) / n
            g = -20 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20 + np.e

        elif g_type == 'Griewank':
            n = z.shape[1]
            indices = np.sqrt(np.arange(1, n + 1))
            sum_sq = np.sum(z ** 2, axis=1)
            prod_cos = np.prod(np.cos(z / indices), axis=1)
            g = 1 + sum_sq / 4000 - prod_cos

        elif g_type == 'Rastrigin':
            n = z.shape[1]
            a = 10 * n
            g = np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z), axis=1) + a

        elif g_type == 'Mean':
            n = z.shape[1]
            g = 9 * np.sum(np.abs(z), axis=1) / n

        else:
            raise ValueError(f"Unknown g_type: {g_type}")

        return g

    def _eval_f1(self, x, f1_type):
        """Evaluate f1 function (for ZDT-type problems)."""
        x = np.atleast_2d(x)

        if f1_type == 'linear':
            # Mean of convergence variables
            f1 = np.mean(x, axis=1)
        else:  # 'nonlinear'
            sum_sq = np.sum(x ** 2, axis=1)
            r = np.sqrt(sum_sq)
            f1 = 1 - np.exp(-4 * r) * (np.sin(5 * np.pi * r) ** 4)

        return f1

    def _eval_h(self, f1, g, h_type):
        """Evaluate h function for ZDT-type problems."""
        if h_type == 'convex':
            h = 1 - np.sqrt(f1 / g)
        elif h_type == 'concave':
            h = 1 - (f1 / g) ** 2
        else:
            raise ValueError(f"Unknown h_type: {h_type}")

        return h

    def P1(self, task_num=10) -> MTOP:
        """
        P1: Sphere + Circular PF (DTLZ formulation)

        - Dimension: 50
        - g-function: Sphere
        - PF: Circular (DTLZ-style)
        """
        problem_id = 1
        dim = 50

        lb = np.full(dim, -100.0)
        ub = np.full(dim, 100.0)
        lb[0] = 0.0
        ub[0] = 1.0

        g_type = 'Sphere'

        problem = MTOP()

        for task_id in range(1, task_num + 1):
            shift_vector, rotation_matrix = self._load_shift_rotation(problem_id, task_id)

            def create_task_func(sv, rm):
                def task_func(x):
                    x = np.atleast_2d(x)

                    # DTLZ formulation
                    g = self._eval_g_function(x[:, 1:], sv, rm, g_type, lb[1:], ub[1:])

                    f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
                    f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)

                    return np.column_stack([f1, f2])

                return task_func

            task_func = create_task_func(shift_vector, rotation_matrix)
            problem.add_task(task_func, dim=dim, lower_bound=lb, upper_bound=ub)

        return problem

    def P2(self, task_num=10) -> MTOP:
        """
        P2: Mean + Concave PF (ZDT formulation)

        - Dimension: 50
        - g-function: Mean
        - PF: Concave
        """
        problem_id = 2
        dim = 50

        lb = np.full(dim, -100.0)
        ub = np.full(dim, 100.0)
        lb[0] = 0.0
        ub[0] = 1.0

        g_type = 'Mean'
        f1_type = 'linear'
        h_type = 'concave'

        problem = MTOP()

        for task_id in range(1, task_num + 1):
            shift_vector, rotation_matrix = self._load_shift_rotation(problem_id, task_id)

            def create_task_func(sv, rm):
                def task_func(x):
                    x = np.atleast_2d(x)

                    # ZDT formulation
                    f1 = self._eval_f1(x[:, 0:1], f1_type)
                    g = self._eval_g_function(x[:, 1:], sv, rm, g_type, lb[1:], ub[1:])
                    g = g + 1
                    f2 = g * self._eval_h(f1, g, h_type)

                    return np.column_stack([f1, f2])

                return task_func

            task_func = create_task_func(shift_vector, rotation_matrix)
            problem.add_task(task_func, dim=dim, lower_bound=lb, upper_bound=ub)

        return problem

    def P3(self, task_num=10) -> MTOP:
        """
        P3: Rosenbrock + Concave PF (ZDT formulation)

        - Dimension: 10
        - g-function: Rosenbrock
        - PF: Concave
        """
        problem_id = 3
        dim = 10

        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        lb[0] = 0.0
        ub[0] = 1.0

        g_type = 'Rosenbrock'
        f1_type = 'linear'
        h_type = 'concave'

        problem = MTOP()

        for task_id in range(1, task_num + 1):
            shift_vector, rotation_matrix = self._load_shift_rotation(problem_id, task_id)

            def create_task_func(sv, rm):
                def task_func(x):
                    x = np.atleast_2d(x)

                    # ZDT formulation
                    f1 = self._eval_f1(x[:, 0:1], f1_type)
                    g = self._eval_g_function(x[:, 1:], sv, rm, g_type, lb[1:], ub[1:])
                    g = g + 1
                    f2 = g * self._eval_h(f1, g, h_type)

                    return np.column_stack([f1, f2])

                return task_func

            task_func = create_task_func(shift_vector, rotation_matrix)
            problem.add_task(task_func, dim=dim, lower_bound=lb, upper_bound=ub)

        return problem

    def P4(self, task_num=10) -> MTOP:
        """
        P4: Rastrigin + Circular PF (DTLZ formulation)

        - Dimension: 50
        - g-function: Rastrigin
        - PF: Circular (DTLZ-style)
        """
        problem_id = 4
        dim = 50

        lb = np.full(dim, -2.0)
        ub = np.full(dim, 2.0)
        lb[0] = 0.0
        ub[0] = 1.0

        g_type = 'Rastrigin'

        problem = MTOP()

        for task_id in range(1, task_num + 1):
            shift_vector, rotation_matrix = self._load_shift_rotation(problem_id, task_id)

            def create_task_func(sv, rm):
                def task_func(x):
                    x = np.atleast_2d(x)

                    # DTLZ formulation
                    g = self._eval_g_function(x[:, 1:], sv, rm, g_type, lb[1:], ub[1:])

                    f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
                    f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)

                    return np.column_stack([f1, f2])

                return task_func

            task_func = create_task_func(shift_vector, rotation_matrix)
            problem.add_task(task_func, dim=dim, lower_bound=lb, upper_bound=ub)

        return problem

    def P5(self, task_num=10) -> MTOP:
        """
        P5: Ackley + Convex PF (ZDT formulation)

        - Dimension: 50
        - g-function: Ackley
        - PF: Convex
        """
        problem_id = 5
        dim = 50

        lb = np.full(dim, -1.0)
        ub = np.full(dim, 1.0)
        lb[0] = 0.0
        ub[0] = 1.0

        g_type = 'Ackley'
        f1_type = 'linear'
        h_type = 'convex'

        problem = MTOP()

        for task_id in range(1, task_num + 1):
            shift_vector, rotation_matrix = self._load_shift_rotation(problem_id, task_id)

            def create_task_func(sv, rm):
                def task_func(x):
                    x = np.atleast_2d(x)

                    # ZDT formulation
                    f1 = self._eval_f1(x[:, 0:1], f1_type)
                    g = self._eval_g_function(x[:, 1:], sv, rm, g_type, lb[1:], ub[1:])
                    g = g + 1
                    f2 = g * self._eval_h(f1, g, h_type)

                    return np.column_stack([f1, f2])

                return task_func

            task_func = create_task_func(shift_vector, rotation_matrix)
            problem.add_task(task_func, dim=dim, lower_bound=lb, upper_bound=ub)

        return problem

    def P6(self, task_num=10) -> MTOP:
        """
        P6: Griewank + Circular PF (DTLZ formulation)

        - Dimension: 50
        - g-function: Griewank
        - PF: Circular (DTLZ-style)
        """
        problem_id = 6
        dim = 50

        lb = np.full(dim, -50.0)
        ub = np.full(dim, 50.0)
        lb[0] = 0.0
        ub[0] = 1.0

        g_type = 'Griewank'

        problem = MTOP()

        for task_id in range(1, task_num + 1):
            shift_vector, rotation_matrix = self._load_shift_rotation(problem_id, task_id)

            def create_task_func(sv, rm):
                def task_func(x):
                    x = np.atleast_2d(x)

                    # DTLZ formulation
                    g = self._eval_g_function(x[:, 1:], sv, rm, g_type, lb[1:], ub[1:])

                    f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
                    f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)

                    return np.column_stack([f1, f2])

                return task_func

            task_func = create_task_func(shift_vector, rotation_matrix)
            problem.add_task(task_func, dim=dim, lower_bound=lb, upper_bound=ub)

        return problem


# Pareto Front functions
def P1_PF(N, M=2):
    """Circular PF for P1"""
    f1 = np.linspace(0, 1, N)
    f2 = np.sqrt(1 - f1 ** 2)
    return np.column_stack([f1, f2])


def P2_PF(N, M=2):
    """Concave PF for P2"""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P3_PF(N, M=2):
    """Concave PF for P3"""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P4_PF(N, M=2):
    """Circular PF for P4"""
    f1 = np.linspace(0, 1, N)
    f2 = np.sqrt(1 - f1 ** 2)
    return np.column_stack([f1, f2])


def P5_PF(N, M=2):
    """Convex PF for P5"""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P6_PF(N, M=2):
    """Circular PF for P6"""
    f1 = np.linspace(0, 1, N)
    f2 = np.sqrt(1 - f1 ** 2)
    return np.column_stack([f1, f2])


SETTINGS = {
    'metric': 'IGD',
    'n_pf': 10000,
    'pf_path': './MOReference',
    'P1': {'all_tasks': P1_PF},
    'P2': {'all_tasks': P2_PF},
    'P3': {'all_tasks': P3_PF},
    'P4': {'all_tasks': P4_PF},
    'P5': {'all_tasks': P5_PF},
    'P6': {'all_tasks': P6_PF},
}
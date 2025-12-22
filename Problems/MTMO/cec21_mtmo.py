import numpy as np
from Methods.mtop import MTOP


class CEC21MTMOFunctions:
    """
    Helper class containing all g-function implementations for CEC21 MTMO benchmark.

    This class provides implementations of various benchmark functions used in the
    CEC 2021 Multi-Task Multi-Objective Optimization competition, including:
    - Basic functions: Rosenbrock (F4), Rastrigin (F8, F9)
    - Complex functions: Modified Schwefel (F11), ExGriewRosen (F15)
    - Hybrid functions: F17, F18, F19, F20, F22
    """

    @staticmethod
    def eval_f1_linear(x):
        """Linear f1 function for MMZDT problems."""
        # MATLAB: A = sum(x, 2); F1Function = A ./ size(x, 2);
        return np.sum(x, axis=1) / x.shape[1]

    @staticmethod
    def eval_F4(x, shift, rotation):
        """Rosenbrock Function F4."""
        x = x.copy()
        x = x - shift
        x = x * (2.048 / 100)
        x = (rotation @ x.T).T

        # Rosenbrock - 使用副本
        x_copy = x.copy()
        x_copy[:, 0] = x_copy[:, 0] + 1
        t = np.zeros(x.shape[0])
        for i in range(x.shape[1] - 1):
            x_copy[:, i + 1] = x_copy[:, i + 1] + 1
            t += 100 * (x_copy[:, i] ** 2 - x_copy[:, i + 1]) ** 2 + (1 - x_copy[:, i]) ** 2

        return t

    @staticmethod
    def eval_F8(x, shift, rotation):
        """
        Rastrigin Function F8 (non-rotated).

        A highly multimodal function with many local optima.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_dims).
        shift : np.ndarray
            Shift vector.
        rotation : np.ndarray
            Rotation matrix (not used in this variant).

        Returns
        -------
        np.ndarray
            Function values of shape (n_samples,).
        """
        x = x - shift
        x = x * (5.12 / 100)

        # Rastrigin
        a = 10 * x.shape[1]
        result = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1) + a

        return result

    @staticmethod
    def eval_F9(x, shift, rotation):
        """
        Rastrigin Function F9 (rotated).

        Rotated version of the Rastrigin function.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_dims).
        shift : np.ndarray
            Shift vector.
        rotation : np.ndarray
            Rotation matrix.

        Returns
        -------
        np.ndarray
            Function values of shape (n_samples,).
        """
        x = x - shift
        x = x * (5.12 / 100)
        x = (rotation @ x.T).T

        # Rastrigin
        a = 10 * x.shape[1]
        result = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1) + a

        return result

    @staticmethod
    def eval_F11(x, shift, rotation):
        """
        Modified Schwefel Function F11.

        A deceptive function with a second-best minimum far from the global optimum.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_dims).
        shift : np.ndarray
            Shift vector.
        rotation : np.ndarray
            Rotation matrix.

        Returns
        -------
        np.ndarray
            Function values of shape (n_samples,).
        """
        x = x - shift
        x = x * (1000.0 / 100)
        x = (rotation @ x.T).T

        # Modified Schwefel
        n = x.shape[1]
        result = CEC21MTMOFunctions._modified_schwefel(x, n)

        return result

    @staticmethod
    def eval_F15(x, shift, rotation):
        """Expanded Griewank plus Rosenbrock Function F15."""
        x = x.copy()
        x = x - shift
        x = x * (5.0 / 100)
        x = (rotation @ x.T).T

        # ExGriewRosen - 使用副本
        x_copy = x.copy()
        x_copy[:, 0] = x_copy[:, 0] + 1
        result = np.zeros(x.shape[0])

        for i in range(x.shape[1] - 1):
            x_copy[:, i + 1] = x_copy[:, i + 1] + 1
            t = 100 * ((x_copy[:, i] ** 2 - x_copy[:, i + 1]) ** 2) + (x_copy[:, i] - 1) ** 2
            result += (t ** 2) / 4000 - np.cos(t) + 1

        # Wrap around: last to first
        index = x.shape[1] - 1
        t = 100 * ((x_copy[:, index] ** 2 - x_copy[:, 0]) ** 2) + (x_copy[:, index] - 1) ** 2
        result += (t ** 2) / 4000 - np.cos(t) + 1

        return result

    @staticmethod
    def eval_F17(x, shift, rotation):
        """
        Hybrid Composition Function F17.
        """
        # 确保shift的广播正确
        x = x - shift.reshape(1, -1)  # 明确reshape
        x = (rotation @ x.T).T

        D = x.shape[1]
        n1 = int(np.ceil(0.3 * D))
        n2 = int(np.ceil(0.3 * D))
        n3 = D - n1 - n2

        x1 = x[:, :n1] * (1000.0 / 100)
        x2 = x[:, n1:n1 + n2] * (5.12 / 100)
        x3 = x[:, n1 + n2:]

        # Modified Schwefel
        result = CEC21MTMOFunctions._modified_schwefel(x1, n1)

        # Rastrigin
        a = 10 * n2
        result += np.sum(x2 ** 2 - 10 * np.cos(2 * np.pi * x2), axis=1) + a

        # Elliptic
        a = 1e6
        for i in range(n3):
            result += (a ** (i / max(n3 - 1, 1))) * (x3[:, i] ** 2)

        return result

    @staticmethod
    def eval_F18(x, shift, rotation):
        """
        Hybrid Composition Function F18.

        Combines Cigar (30%) + HGBat (30%) + Rastrigin (40%).

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_dims).
        shift : np.ndarray
            Shift vector.
        rotation : np.ndarray
            Rotation matrix.

        Returns
        -------
        np.ndarray
            Function values of shape (n_samples,).
        """
        x = x - shift
        x = (rotation @ x.T).T

        D = x.shape[1]
        n1 = int(np.ceil(0.3 * D))
        n2 = int(np.ceil(0.3 * D))
        n3 = D - n1 - n2

        x1 = x[:, :n1]
        x2 = x[:, n1:n1 + n2] * (5.0 / 100)
        x3 = x[:, n1 + n2:] * (5.12 / 100)

        # Cigar
        a = np.sum(x1[:, 1:] ** 2, axis=1)
        result = x1[:, 0] ** 2 + a * 1e6

        # HGBat
        x2 = x2 - 1
        sum1 = np.sum(x2 ** 2, axis=1)
        sum2 = np.sum(x2, axis=1)
        result += (np.abs(sum1 ** 2 - sum2 ** 2)) ** 0.5 + (0.5 * sum1 + sum2) / n2 + 0.5

        # Rastrigin
        a = 10 * n3
        result += np.sum(x3 ** 2 - 10 * np.cos(2 * np.pi * x3), axis=1) + a

        return result

    @staticmethod
    def eval_F19(x, shift, rotation):
        """
        Hybrid Composition Function F19.

        Combines Griewank (20%) + Weierstrass (20%) + Rosenbrock (30%) + Schaffer F6 (30%).
        """
        x = x.copy()
        x = x - shift
        x = (rotation @ x.T).T

        D = x.shape[1]
        n1 = int(np.ceil(0.2 * D))
        n2 = int(np.ceil(0.2 * D))
        n3 = int(np.ceil(0.3 * D))
        n4 = D - n1 - n2 - n3

        x1 = x[:, :n1] * (600.0 / 100)
        x2 = x[:, n1:n1 + n2] * (0.5 / 100)
        x3 = x[:, n1 + n2:n1 + n2 + n3] * (2.048 / 100)
        x4 = x[:, n1 + n2 + n3:]

        # Griewank
        t = np.arange(1, n1 + 1)
        sum1 = np.sum(x1 ** 2, axis=1)
        prod1 = np.prod(np.cos(x1 / np.sqrt(t)), axis=1)
        result = 1 + sum1 / 4000 - prod1

        # Weierstrass
        a = 0.5
        b = 3
        kmax = 20
        part1 = np.zeros(x.shape[0])
        for i in range(n2):
            for k in range(kmax + 1):
                part1 += a ** k * np.cos(2 * np.pi * (b ** k) * (x2[:, i] + 0.5))

        part2 = 0
        for k in range(kmax + 1):
            part2 += a ** k * np.cos(2 * np.pi * (b ** k) * 0.5)

        result += part1 - n2 * part2

        # Rosenbrock - 修正：使用副本
        t = np.zeros(x.shape[0])
        x3_copy = x3.copy()
        x3_copy[:, 0] = x3_copy[:, 0] + 1
        for i in range(n3 - 1):
            x3_copy[:, i + 1] = x3_copy[:, i + 1] + 1
            t += 100 * (x3_copy[:, i] ** 2 - x3_copy[:, i + 1]) ** 2 + (1 - x3_copy[:, i]) ** 2

        result += t

        # Schaffer F6
        t = np.zeros(x.shape[0])
        for i in range(n4):
            next_idx = (i + 1) % n4
            pSum = x4[:, i] ** 2 + x4[:, next_idx] ** 2
            t += 0.5 + ((np.sin(pSum ** 0.5)) ** 2 - 0.5) / ((1 + 0.001 * pSum) ** 2)

        result += t

        return result

    @staticmethod
    def eval_F20(x, shift, rotation):
        """Hybrid Composition Function F20."""
        x = x.copy()
        x = x - shift
        x = (rotation @ x.T).T

        D = x.shape[1]
        n1 = int(np.ceil(0.2 * D))
        n2 = int(np.ceil(0.2 * D))
        n3 = int(np.ceil(0.3 * D))
        n4 = D - n1 - n2 - n3

        x1 = x[:, :n1] * (5.0 / 100)
        x2 = x[:, n1:n1 + n2]
        x3 = x[:, n1 + n2:n1 + n2 + n3] * (5.0 / 100)
        x4 = x[:, n1 + n2 + n3:] * (5.12 / 100)

        # HGBat - 使用副本
        x1_copy = x1.copy()
        x1_copy = x1_copy - 1
        sum1 = np.sum(x1_copy ** 2, axis=1)
        sum2 = np.sum(x1_copy, axis=1)
        result = (np.abs(sum1 ** 2 - sum2 ** 2)) ** 0.5 + (0.5 * sum1 + sum2) / n1 + 0.5

        # Discus
        result += 1e6 * (x2[:, 0] ** 2) + np.sum(x2[:, 1:] ** 2, axis=1)

        # ExGriewRosen - 使用副本
        x3_copy = x3.copy()
        x3_copy[:, 0] = x3_copy[:, 0] + 1
        for i in range(n3 - 1):
            x3_copy[:, i + 1] = x3_copy[:, i + 1] + 1
            t = 100 * (x3_copy[:, i] ** 2 - x3_copy[:, i + 1]) ** 2 + (x3_copy[:, i] - 1) ** 2
            result += (t ** 2) / 4000 - np.cos(t) + 1

        t = 100 * (x3_copy[:, n3 - 1] ** 2 - x3_copy[:, 0]) ** 2 + (x3_copy[:, n3 - 1] - 1) ** 2
        result += (t ** 2) / 4000 - np.cos(t) + 1

        # Rastrigin
        a = 10 * n4
        result += np.sum(x4 ** 2 - 10 * np.cos(2 * np.pi * x4), axis=1) + a

        return result

    @staticmethod
    def eval_F22(x, shift, rotation):
        """
        Hybrid Composition Function F22.

        Combines Katsuura (10%) + HappyCat (20%) + ExGriewRosen (20%) +
        Modified Schwefel (20%) + Ackley (30%).
        """
        x = x.copy()
        x = x - shift
        x = (rotation @ x.T).T

        D = x.shape[1]
        n1 = int(np.ceil(0.1 * D))
        n2 = int(np.ceil(0.2 * D))
        n3 = int(np.ceil(0.2 * D))
        n4 = int(np.ceil(0.2 * D))
        n5 = D - n1 - n2 - n3 - n4

        x1 = x[:, :n1].copy() * (5.0 / 100)
        x2 = x[:, n1:n1 + n2].copy() * (5.0 / 100)
        x3 = x[:, n1 + n2:n1 + n2 + n3].copy() * (5.0 / 100)
        x4 = x[:, n1 + n2 + n3:n1 + n2 + n3 + n4].copy() * (1000.0 / 100)
        x5 = x[:, n1 + n2 + n3 + n4:].copy()

        # Katsuura - 修正
        index = 32
        prod1 = np.zeros(x.shape[0])  # ✓ 在外层循环外初始化
        result = np.ones(x.shape[0])
        for i in range(n1):
            for j in range(1, index + 1):
                prod1 += np.abs(2 ** j * x1[:, i] - np.round(2 ** j * x1[:, i])) / (2 ** j)
            result *= (1 + i * prod1) ** (10 / (n1 ** 1.2))  # ✓ 使用 i 而不是 (i+1)

        result = (10 / (n1 ** 1.2)) * (result - 1)

        # HappyCat
        x2 = x2 - 1
        sum1 = np.sum(x2 ** 2, axis=1)
        sum2 = np.sum(x2, axis=1)
        result += np.abs(sum1 - n2) ** 0.25 + (0.5 * sum1 + sum2) / n2 + 0.5

        # ExGriewRosen
        x3[:, 0] = x3[:, 0] + 1
        for i in range(n3 - 1):
            x3[:, i + 1] = x3[:, i + 1] + 1
            t = 100 * ((x3[:, i] ** 2 - x3[:, i + 1]) ** 2) + (x3[:, i] - 1) ** 2
            result += (t ** 2) / 4000 - np.cos(t) + 1

        t = 100 * ((x3[:, n3 - 1] ** 2 - x3[:, 0]) ** 2) + (x3[:, n3 - 1] - 1) ** 2
        result += (t ** 2) / 4000 - np.cos(t) + 1

        # Modified Schwefel
        result += CEC21MTMOFunctions._modified_schwefel(x4, n4)

        # Ackley
        sum1 = np.sum(x5 ** 2, axis=1) / n5
        sum2 = np.sum(np.cos(2 * np.pi * x5), axis=1) / n5
        result += (-20) * np.exp(-0.2 * np.sqrt(sum1)) - np.exp(sum2) + 20 + np.e

        return result

    @staticmethod
    def _modified_schwefel(x, n):
        """
        Modified Schwefel function helper.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n_samples, n_dims).
        n : int
            Number of dimensions.

        Returns
        -------
        np.ndarray
            Function values of shape (n_samples,).
        """
        prod1 = np.zeros(x.shape[0])
        for i in range(x.shape[1]):
            z = x[:, i] + 4.209687462275036e+002

            gz = np.zeros_like(z)
            mask1 = np.abs(z) <= 500
            mask2 = z > 500
            mask3 = ~(mask1 | mask2)

            # Case 1: |z| <= 500
            gz[mask1] = z[mask1] * np.sin(np.abs(z[mask1]) ** 0.5)

            # Case 2: z > 500
            temp = 500 - np.mod(z[mask2], 500)
            gz[mask2] = temp * np.sin(np.abs(temp) ** 0.5) - ((z[mask2] - 500) ** 2) / (10000 * n)

            # Case 3: z < -500
            temp = np.mod(np.abs(z[mask3]), 500) - 500
            gz[mask3] = temp * np.sin(np.abs(500 - np.mod(np.abs(z[mask3]), 500)) ** 0.5) - ((z[mask3] + 500) ** 2) / (
                        10000 * n)

            prod1 += gz

        return 418.9829 * n - prod1



class CEC21MTMO:
    """
    Implementation of the CEC 2021 Competition on Evolutionary Multi-Task Multi-Objective
    Optimization (MTMO) benchmark problems.

    These problems consist of two multi-objective optimization tasks (MO-tasks)
    with shared variables, designed to test knowledge transfer in the presence of
    multiple conflicting objectives. All tasks are minimization problems.

    Parameters
    ----------
    mat_dir : str, optional
        Directory path to the data files containing rotation matrices and shift
        vectors for the tasks (default: '../Problems/MTMO/data_cec21mtmo').

    Attributes
    ----------
    mat_dir : str
        The directory path for problem data files.
    funcs : CEC21MTMOFunctions
        Instance of the functions helper class.
    """

    def __init__(self, mat_dir='../Problems/MTMO/data_cec21mtmo'):
        self.mat_dir = mat_dir
        self.funcs = CEC21MTMOFunctions()

    def P1(self) -> MTOP:
        """
        Generates Problem 1: **T1 (MMDTLZ, F17) vs T2 (MMZDT, F17)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F17 (Modified Schwefel + Rastrigin + Elliptic).
        - T2: MMZDT-type with linear f1, hybrid function F17, and concave h.
        - Relationship: Different task types but same g-function complexity.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_1/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_1/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_1/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_1/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F17 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F17)
            g = self.funcs.eval_F17(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F17 g-function, and concave h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F17)
            g = self.funcs.eval_F17(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Concave h-function
            f2 = g * (1 - (f1 / g) ** 2)
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self) -> MTOP:
        """
        Generates Problem 2: **T1 (MMDTLZ, F19) vs T2 (MMDTLZ, F19)**.
        """
        dim = 50

        # 加载rotation matrices and shift vectors
        # Task 0 对应 matrix_1, bias_1
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_2/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_2/bias_1', dtype=float)
        # Task 1 对应 matrix_2, bias_2
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_2/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_2/bias_2', dtype=float)

        def T1(x):  # 这是任务0，应该用 rotation1, shift1
            """Task 0: MMDTLZ with F19 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F19)
            g = self.funcs.eval_F19(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):  # 这是任务1，应该用 rotation2, shift2
            """Task 1: MMDTLZ with F19 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F19)
            g = self.funcs.eval_F19(x[:, 1:], shift2, rotation2)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self) -> MTOP:
        """
        Generates Problem 3: **T1 (MMDTLZ, F22) vs T2 (MMZDT, F22)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F22 (Katsuura + HappyCat + ExGriewRosen + Modified Schwefel + Ackley).
        - T2: MMZDT-type with linear f1, hybrid function F22, and convex h.
        - Relationship: Different task types but same g-function complexity.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_3/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_3/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_3/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_3/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F22 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F22)
            g = self.funcs.eval_F22(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F22 g-function, and convex h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F22)
            g = self.funcs.eval_F22(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Convex h-function
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P4(self) -> MTOP:
        """
        Generates Problem 4: **T1 (MMZDT, F15) vs T2 (MMZDT, F15)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMZDT-type with linear f1, hybrid function F15 (ExGriewRosen), and convex h.
        - T2: MMZDT-type with linear f1, hybrid function F15 (ExGriewRosen), and convex h.
        - Relationship: Same task type and g-function, both convex.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_4/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_4/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_4/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_4/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMZDT with linear f1, F15 g-function, and convex h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F15)
            g = self.funcs.eval_F15(x[:, 1:], shift1, rotation1)
            g = g + 1
            # Convex h-function
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F15 g-function, and convex h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F15)
            g = self.funcs.eval_F15(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Convex h-function
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self) -> MTOP:
        """
        Generates Problem 5: **T1 (MMDTLZ, F4) vs T2 (MMZDT, F4)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F4 (Rosenbrock).
        - T2: MMZDT-type with linear f1, hybrid function F4 (Rosenbrock), and concave h.
        - Relationship: Different task types but same g-function (Rosenbrock).

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_5/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_5/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_5/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_5/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F4 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F4 - Rosenbrock)
            g = self.funcs.eval_F4(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F4 g-function, and concave h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F4 - Rosenbrock)
            g = self.funcs.eval_F4(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Concave h-function
            f2 = g * (1 - (f1 / g) ** 2)
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P6(self) -> MTOP:
        """
        Generates Problem 6: **T1 (MMDTLZ, F9) vs T2 (MMDTLZ, F9)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F9 (Rastrigin, rotated).
        - T2: MMDTLZ-type with hybrid function F9 (Rastrigin, rotated).
        - Relationship: Same task type and g-function.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_6/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_6/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_6/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_6/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F9 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F9 - Rastrigin rotated)
            g = self.funcs.eval_F9(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMDTLZ with F9 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F9 - Rastrigin rotated)
            g = self.funcs.eval_F9(x[:, 1:], shift2, rotation2)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P7(self) -> MTOP:
        """
        Generates Problem 7: **T1 (MMDTLZ, F8) vs T2 (MMZDT, F8)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F8 (Rastrigin, non-rotated).
        - T2: MMZDT-type with linear f1, hybrid function F8 (Rastrigin, non-rotated), and convex h.
        - Relationship: Different task types but same g-function.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_7/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_7/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_7/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_7/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F8 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F8 - Rastrigin non-rotated)
            g = self.funcs.eval_F8(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F8 g-function, and convex h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F8 - Rastrigin non-rotated)
            g = self.funcs.eval_F8(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Convex h-function
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self) -> MTOP:
        """
        Generates Problem 8: **T1 (MMDTLZ, F18) vs T2 (MMZDT, F20)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F18 (Cigar + HGBat + Rastrigin).
        - T2: MMZDT-type with linear f1, hybrid function F20 (HGBat + Discus + ExGriewRosen + Rastrigin), and concave h.
        - Relationship: Different task types and different g-functions.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_8/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_8/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_8/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_8/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F18 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F18)
            g = self.funcs.eval_F18(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F20 g-function, and concave h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F20)
            g = self.funcs.eval_F20(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Concave h-function
            f2 = g * (1 - (f1 / g) ** 2)
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P9(self) -> MTOP:
        """
        Generates Problem 9: **T1 (MMDTLZ, F11) vs T2 (MMZDT, F18)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F11 (Modified Schwefel).
        - T2: MMZDT-type with linear f1, hybrid function F18 (Cigar + HGBat + Rastrigin), and concave h.
        - Relationship: Different task types and different g-functions.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_9/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_9/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_9/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_9/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F11 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F11 - Modified Schwefel)
            g = self.funcs.eval_F11(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F18 g-function, and concave h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F18)
            g = self.funcs.eval_F18(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Concave h-function
            f2 = g * (1 - (f1 / g) ** 2)
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P10(self) -> MTOP:
        """
        Generates Problem 10: **T1 (MMDTLZ, F15) vs T2 (MMZDT, F17)**.

        Both tasks are 2-objective, 50-dimensional.

        - T1: MMDTLZ-type with hybrid function F15 (ExGriewRosen).
        - T2: MMZDT-type with linear f1, hybrid function F17 (Modified Schwefel + Rastrigin + Elliptic), and concave h.
        - Relationship: Different task types and different g-functions.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 50

        # Load rotation matrices and shift vectors
        rotation1 = np.loadtxt(f'{self.mat_dir}/benchmark_10/matrix_1', dtype=float)
        shift1 = np.loadtxt(f'{self.mat_dir}/benchmark_10/bias_1', dtype=float)
        rotation2 = np.loadtxt(f'{self.mat_dir}/benchmark_10/matrix_2', dtype=float)
        shift2 = np.loadtxt(f'{self.mat_dir}/benchmark_10/bias_2', dtype=float)

        def T1(x):
            """Task 1: MMDTLZ with F15 g-function"""
            x = np.atleast_2d(x)
            # Evaluate g-function (F15 - ExGriewRosen)
            g = self.funcs.eval_F15(x[:, 1:], shift1, rotation1)
            # MMDTLZ objectives
            f1 = (1 + g) * np.cos(x[:, 0] * 0.5 * np.pi)
            f2 = (1 + g) * np.sin(x[:, 0] * 0.5 * np.pi)
            return np.vstack([f1, f2]).T

        def T2(x):
            """Task 2: MMZDT with linear f1, F17 g-function, and concave h"""
            x = np.atleast_2d(x)
            # Evaluate f1 (linear)
            f1 = self.funcs.eval_f1_linear(x[:, 0:1])
            # Evaluate g-function (F17)
            g = self.funcs.eval_F17(x[:, 1:], shift2, rotation2)
            g = g + 1
            # Concave h-function
            f2 = g * (1 - (f1 / g) ** 2)
            return np.vstack([f1, f2]).T

        # Bounds
        lb = np.concatenate([[0.0], -100.0 * np.ones(49)])
        ub = np.concatenate([[1.0], 100.0 * np.ones(49)])

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem



# --- True Pareto Front (PF) Functions ---

def _circle_PF(N, M=2) -> np.ndarray:
    """
    Quarter circle arc PF: unit circle in first quadrant.
    Used for MMDTLZ tasks.
    """
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])


def _convex_PF(N, M=2) -> np.ndarray:
    """
    Convex PF: f_2 = 1 - sqrt(f_1).
    Used for MMZDT tasks with convex h-function.
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def _concave_PF(N, M=2) -> np.ndarray:
    """
    Concave PF: f_2 = 1 - f_1^2.
    Used for MMZDT tasks with concave h-function.
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


# Problem-specific PF assignments
P1_T1_PF = _circle_PF      # MMDTLZ, F17
P1_T2_PF = _circle_PF      # MMZDT, F17, concave (special case: circle)

P2_T1_PF = _circle_PF      # MMDTLZ, F19
P2_T2_PF = _circle_PF      # MMDTLZ, F19

P3_T1_PF = _circle_PF      # MMDTLZ, F22
P3_T2_PF = _convex_PF      # MMZDT, F22, convex

P4_T1_PF = _convex_PF      # MMZDT, F15, convex
P4_T2_PF = _convex_PF      # MMZDT, F15, convex

P5_T1_PF = _circle_PF      # MMDTLZ, F4
P5_T2_PF = _concave_PF     # MMZDT, F4, concave

P6_T1_PF = _circle_PF      # MMDTLZ, F9
P6_T2_PF = _circle_PF      # MMDTLZ, F9

P7_T1_PF = _circle_PF      # MMDTLZ, F8
P7_T2_PF = _convex_PF      # MMZDT, F8, convex

P8_T1_PF = _circle_PF      # MMDTLZ, F18
P8_T2_PF = _concave_PF     # MMZDT, F20, concave

P9_T1_PF = _circle_PF      # MMDTLZ, F11
P9_T2_PF = _concave_PF     # MMZDT, F18, concave

P10_T1_PF = _circle_PF     # MMDTLZ, F15
P10_T2_PF = _concave_PF    # MMZDT, F17, concave


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
    'P10': {'T1': P10_T1_PF, 'T2': P10_T2_PF},
}





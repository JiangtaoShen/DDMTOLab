import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point


class CF:
    """
    Implementation of the CF test suite (CEC 2009) for constrained multi-objective optimization.

    The CF test problems (CF1 to CF10) are standard constrained multi-objective
    optimization benchmarks from the CEC 2009 competition.

    References
    ----------
    Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu, and S. Tiwari.
    "Multiobjective optimization test instances for the CEC 2009 special session and competition."
    University of Essex, Colchester, UK, Tech. Rep. CES-487, 2009.
    """

    def CF1(self, M=2, dim=None) -> MTOP:
        """
        Generates the **CF1** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(2, D, 2)
            J2 = np.arange(1, D, 2)

            # Exponents for the transformation
            y1_exp = 0.5 * (1 + 3 * (J1 + 1 - 2) / (D - 2))
            y2_exp = 0.5 * (1 + 3 * (J2 + 1 - 2) / (D - 2))

            y1 = x[:, J1] - x[:, 0:1] ** y1_exp
            y2 = x[:, J2] - x[:, 0:1] ** y2_exp

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0] + (2 * np.mean(y1 ** 2, axis=1) if J1.size > 0 else 0)
            obj[:, 1] = 1 - x[:, 0] + (2 * np.mean(y2 ** 2, axis=1) if J2.size > 0 else 0)
            return obj

        def C1(x):
            obj = T1(x)
            # Constraint: f1 + f2 >= 1 + abs(sin(...))
            # Transformed to <= 0 form: 1 + abs(...) - f1 - f2 <= 0
            return 1 - obj[:, 0] - obj[:, 1] + np.abs(np.sin(10 * np.pi * (obj[:, 0] - obj[:, 1] + 1)))

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF2(self, M=2, dim=None) -> MTOP:
        """
        Generates the **CF2** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(2, D, 2)
            J2 = np.arange(1, D, 2)

            term1 = (x[:, J1] - np.sin(6 * np.pi * x[:, 0:1] + (J1 + 1) * np.pi / D)) ** 2
            term2 = (x[:, J2] - np.cos(6 * np.pi * x[:, 0:1] + (J2 + 1) * np.pi / D)) ** 2

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0] + (2 * np.mean(term1, axis=1) if J1.size > 0 else 0)
            obj[:, 1] = 1 - np.sqrt(x[:, 0]) + (2 * np.mean(term2, axis=1) if J2.size > 0 else 0)
            return obj

        def C1(x):
            obj = T1(x)
            t = obj[:, 1] + np.sqrt(obj[:, 0]) - np.sin(2 * np.pi * (np.sqrt(obj[:, 0]) - obj[:, 1] + 1)) - 1
            # PlatEMO: PopCon = -t / (1 + exp(4*abs(t)))
            # Returns <= 0 if feasible (t >= 0)
            return -t / (1 + np.exp(4 * np.abs(t)))

        lb = np.zeros(dim)
        lb[1:] = -1.0
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF3(self, M=2, dim=None) -> MTOP:
        """
        Generates the **CF3** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(2, D, 2)
            J2 = np.arange(1, D, 2)

            Y = x - np.sin(6 * np.pi * x[:, 0:1] + (np.arange(D) + 1) * np.pi / D)

            # For J1
            term1 = 4 * np.sum(Y[:, J1] ** 2, axis=1) - 2 * np.prod(np.cos(20 * Y[:, J1] * np.pi / np.sqrt(J1 + 1)),
                                                                    axis=1) + 2
            # For J2
            term2 = 4 * np.sum(Y[:, J2] ** 2, axis=1) - 2 * np.prod(np.cos(20 * Y[:, J2] * np.pi / np.sqrt(J2 + 1)),
                                                                    axis=1) + 2

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0] + (2 / J1.size * term1 if J1.size > 0 else 0)
            obj[:, 1] = 1 - x[:, 0] ** 2 + (2 / J2.size * term2 if J2.size > 0 else 0)
            return obj

        def C1(x):
            obj = T1(x)
            # Constraint: f2 + f1^2 >= 1 + sin(...)
            return 1 - obj[:, 1] - obj[:, 0] ** 2 + np.sin(2 * np.pi * (obj[:, 0] ** 2 - obj[:, 1] + 1))

        lb = np.zeros(dim)
        lb[1:] = -2.0
        ub = np.ones(dim)
        ub[1:] = 2.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF4(self, M=2, dim=None) -> MTOP:
        """
        Generates the **CF4** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(2, D, 2)
            J2 = np.arange(1, D, 2)

            Y = x - np.sin(6 * np.pi * x[:, 0:1] + (np.arange(D) + 1) * np.pi / D)

            h = Y ** 2
            # Special handling for h[:, 1] (index 1)
            temp = Y[:, 1] < 1.5 * (1 - np.sqrt(0.5))
            h[temp, 1] = np.abs(Y[temp, 1])
            h[~temp, 1] = 0.125 + (Y[~temp, 1] - 1) ** 2

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0] + np.sum(h[:, J1], axis=1)
            obj[:, 1] = 1 - x[:, 0] + np.sum(h[:, J2], axis=1)
            return obj

        def C1(x):
            # PlatEMO: t = x2 - sin(...) - 0.5*x1 + 0.25
            t = x[:, 1] - np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / x.shape[1]) - 0.5 * x[:, 0] + 0.25
            return -t / (1 + np.exp(4 * np.abs(t)))

        lb = np.zeros(dim)
        lb[1:] = -2.0
        ub = np.ones(dim)
        ub[1:] = 2.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF5(self, M=2, dim=None) -> MTOP:
        """
        Generates the **CF5** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(2, D, 2)
            J2 = np.arange(1, D, 2)

            Y = np.zeros_like(x)
            Y[:, J1] = x[:, J1] - 0.8 * x[:, 0:1] * np.cos(6 * np.pi * x[:, 0:1] + (J1 + 1) * np.pi / D)
            Y[:, J2] = x[:, J2] - 0.8 * x[:, 0:1] * np.sin(6 * np.pi * x[:, 0:1] + (J2 + 1) * np.pi / D)

            h = 2 * Y ** 2 - np.cos(4 * np.pi * Y) + 1
            # Special handling for Y[:, 1]
            temp = Y[:, 1] < 1.5 * (1 - np.sqrt(0.5))
            h[temp, 1] = np.abs(Y[temp, 1])
            h[~temp, 1] = 0.125 + (Y[~temp, 1] - 1) ** 2

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0] + np.sum(h[:, J1], axis=1)
            obj[:, 1] = 1 - x[:, 0] + np.sum(h[:, J2], axis=1)
            return obj

        def C1(x):
            # PlatEMO: -x2 + 0.8*x1*sin(...) + 0.5*x1 - 0.25
            return -x[:, 1] + 0.8 * x[:, 0] * np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / x.shape[1]) + 0.5 * x[
                :, 0] - 0.25

        lb = np.zeros(dim)
        lb[1:] = -2.0
        ub = np.ones(dim)
        ub[1:] = 2.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF6(self, M=2, dim=None) -> MTOP:
        """
        Generates the **CF6** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(2, D, 2)
            J2 = np.arange(1, D, 2)

            Y = np.zeros_like(x)
            Y[:, J1] = x[:, J1] - 0.8 * x[:, 0:1] * np.cos(6 * np.pi * x[:, 0:1] + (J1 + 1) * np.pi / D)
            Y[:, J2] = x[:, J2] - 0.8 * x[:, 0:1] * np.sin(6 * np.pi * x[:, 0:1] + (J2 + 1) * np.pi / D)

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0] + np.sum(Y[:, J1] ** 2, axis=1)
            obj[:, 1] = (1 - x[:, 0]) ** 2 + np.sum(Y[:, J2] ** 2, axis=1)
            return obj

        def C1(x):
            D = x.shape[1]
            term1 = 0.5 * (1 - x[:, 0]) - (1 - x[:, 0]) ** 2
            term2 = 0.25 * np.sqrt(1 - x[:, 0]) - 0.5 * (1 - x[:, 0])

            c1 = -x[:, 1] + 0.8 * x[:, 0] * np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / D) + \
                 np.sign(term1) * np.sqrt(np.abs(term1))

            c2 = -x[:, 3] + 0.8 * x[:, 0] * np.sin(6 * np.pi * x[:, 0] + 4 * np.pi / D) + \
                 np.sign(term2) * np.sqrt(np.abs(term2))

            return np.column_stack([c1, c2])

        lb = np.zeros(dim)
        lb[1:] = -2.0
        ub = np.ones(dim)
        ub[1:] = 2.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF7(self, M=2, dim=None) -> MTOP:
        """
        Generates the **CF7** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(2, D, 2)
            J2 = np.arange(1, D, 2)

            Y = np.zeros_like(x)
            Y[:, J1] = x[:, J1] - np.cos(6 * np.pi * x[:, 0:1] + (J1 + 1) * np.pi / D)
            Y[:, J2] = x[:, J2] - np.sin(6 * np.pi * x[:, 0:1] + (J2 + 1) * np.pi / D)

            h = 2 * Y ** 2 - np.cos(4 * np.pi * Y) + 1
            # Special handling for h[:, 1] (index 1) and h[:, 3] (index 3)
            h[:, 1] = Y[:, 1] ** 2
            h[:, 3] = Y[:, 3] ** 2

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0] + np.sum(h[:, J1], axis=1)
            obj[:, 1] = (1 - x[:, 0]) ** 2 + np.sum(h[:, J2], axis=1)
            return obj

        def C1(x):
            D = x.shape[1]
            term1 = 0.5 * (1 - x[:, 0]) - (1 - x[:, 0]) ** 2
            term2 = 0.25 * np.sqrt(1 - x[:, 0]) - 0.5 * (1 - x[:, 0])

            c1 = -x[:, 1] + np.sin(6 * np.pi * x[:, 0] + 2 * np.pi / D) + \
                 np.sign(term1) * np.sqrt(np.abs(term1))

            c2 = -x[:, 3] + np.sin(6 * np.pi * x[:, 0] + 4 * np.pi / D) + \
                 np.sign(term2) * np.sqrt(np.abs(term2))

            return np.column_stack([c1, c2])

        lb = np.zeros(dim)
        lb[1:] = -2.0
        ub = np.ones(dim)
        ub[1:] = 2.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF8(self, M=3, dim=None) -> MTOP:
        """
        Generates the **CF8** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(3, D, 3)  # Indices 3, 6, 9...
            J2 = np.arange(4, D, 3)  # Indices 4, 7, 10...
            J3 = np.arange(2, D, 3)  # Indices 2, 5, 8...

            Y = x - 2 * x[:, 1:2] * np.sin(2 * np.pi * x[:, 0:1] + (np.arange(D) + 1) * np.pi / D)

            obj = np.zeros((N, M))
            obj[:, 0] = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + (
                2 * np.mean(Y[:, J1] ** 2, axis=1) if J1.size > 0 else 0)
            obj[:, 1] = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + (
                2 * np.mean(Y[:, J2] ** 2, axis=1) if J2.size > 0 else 0)
            obj[:, 2] = np.sin(0.5 * x[:, 0] * np.pi) + (2 * np.mean(Y[:, J3] ** 2, axis=1) if J3.size > 0 else 0)
            return obj

        def C1(x):
            obj = T1(x)
            # Constraint involves objectives: 1 - (f1^2+f2^2)/(1-f3^2) + 4*abs(...)
            num = obj[:, 0] ** 2 + obj[:, 1] ** 2
            den = 1 - obj[:, 2] ** 2

            # Safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                term = num / den
            term = np.nan_to_num(term)

            val = 1 - term + 4 * np.abs(np.sin(2 * np.pi * ((obj[:, 0] ** 2 - obj[:, 1] ** 2) / den + 1)))
            return val

        lb = np.zeros(dim)
        ub = np.ones(dim)
        lb[2:] = -4.0
        ub[2:] = 4.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF9(self, M=3, dim=None) -> MTOP:
        """
        Generates the **CF9** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(3, D, 3)
            J2 = np.arange(4, D, 3)
            J3 = np.arange(2, D, 3)

            Y = x - 2 * x[:, 1:2] * np.sin(2 * np.pi * x[:, 0:1] + (np.arange(D) + 1) * np.pi / D)

            obj = np.zeros((N, M))
            obj[:, 0] = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + (
                2 * np.mean(Y[:, J1] ** 2, axis=1) if J1.size > 0 else 0)
            obj[:, 1] = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + (
                2 * np.mean(Y[:, J2] ** 2, axis=1) if J2.size > 0 else 0)
            obj[:, 2] = np.sin(0.5 * x[:, 0] * np.pi) + (2 * np.mean(Y[:, J3] ** 2, axis=1) if J3.size > 0 else 0)
            return obj

        def C1(x):
            obj = T1(x)
            num = obj[:, 0] ** 2 + obj[:, 1] ** 2
            den = 1 - obj[:, 2] ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                term = num / den
            term = np.nan_to_num(term)

            val = 1 - term + 3 * np.sin(2 * np.pi * ((obj[:, 0] ** 2 - obj[:, 1] ** 2) / den + 1))
            return val

        lb = np.zeros(dim)
        ub = np.ones(dim)
        lb[2:] = -2.0
        ub[2:] = 2.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def CF10(self, M=3, dim=None) -> MTOP:
        """
        Generates the **CF10** problem.
        """
        if dim is None:
            dim = 10

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape
            J1 = np.arange(3, D, 3)
            J2 = np.arange(4, D, 3)
            J3 = np.arange(2, D, 3)

            Y = x - 2 * x[:, 1:2] * np.sin(2 * np.pi * x[:, 0:1] + (np.arange(D) + 1) * np.pi / D)

            h_J1 = 4 * Y[:, J1] ** 2 - np.cos(8 * np.pi * Y[:, J1]) + 1
            h_J2 = 4 * Y[:, J2] ** 2 - np.cos(8 * np.pi * Y[:, J2]) + 1
            h_J3 = 4 * Y[:, J3] ** 2 - np.cos(8 * np.pi * Y[:, J3]) + 1

            obj = np.zeros((N, M))
            obj[:, 0] = np.cos(0.5 * x[:, 0] * np.pi) * np.cos(0.5 * x[:, 1] * np.pi) + (
                2 * np.mean(h_J1, axis=1) if J1.size > 0 else 0)
            obj[:, 1] = np.cos(0.5 * x[:, 0] * np.pi) * np.sin(0.5 * x[:, 1] * np.pi) + (
                2 * np.mean(h_J2, axis=1) if J2.size > 0 else 0)
            obj[:, 2] = np.sin(0.5 * x[:, 0] * np.pi) + (2 * np.mean(h_J3, axis=1) if J3.size > 0 else 0)
            return obj

        def C1(x):
            # Same as CF9 constraint
            obj = T1(x)
            num = obj[:, 0] ** 2 + obj[:, 1] ** 2
            den = 1 - obj[:, 2] ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                term = num / den
            term = np.nan_to_num(term)

            val = 1 - term + np.sin(2 * np.pi * ((obj[:, 0] ** 2 - obj[:, 1] ** 2) / den + 1))
            return val

        lb = np.zeros(dim)
        ub = np.ones(dim)
        lb[2:] = -2.0
        ub[2:] = 2.0

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem


# --- Pareto Front (PF) Functions ---

def CF1_PF(N: int, M: int = 2) -> np.ndarray:
    """Pareto Front for CF1."""
    f1 = np.linspace(0, 1, 21)
    f2 = 1 - f1
    return np.column_stack([f1, f2])


def CF2_PF(N: int, M: int = 2) -> np.ndarray:
    """Pareto Front for CF2."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    # Filter infeasible
    mask = ((f1 > 0) & (f1 < 1 / 16)) | ((f1 > 1 / 4) & (f1 < 9 / 16))
    return np.column_stack([f1[~mask], f2[~mask]])


def CF3_PF(N: int, M: int = 2) -> np.ndarray:
    """Pareto Front for CF3."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    # Filter infeasible
    mask = ((f1 > 0) & (f1 < 0.5)) | ((f1 > np.sqrt(0.5)) & (f1 < np.sqrt(0.75)))
    return np.column_stack([f1[~mask], f2[~mask]])


def CF4_PF(N: int, M: int = 2) -> np.ndarray:
    """Pareto Front for CF4."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1
    mask1 = (f1 > 0.5) & (f1 <= 0.75)
    f2[mask1] = -0.5 * f1[mask1] + 0.75
    mask2 = f1 > 0.75
    f2[mask2] = 1 - f1[mask2] + 0.125
    return np.column_stack([f1, f2])


def CF5_PF(N: int, M: int = 2) -> np.ndarray:
    """Pareto Front for CF5."""
    return CF4_PF(N, M)


def CF6_PF(N: int, M: int = 2) -> np.ndarray:
    """Pareto Front for CF6."""
    f1 = np.linspace(0, 1, N)
    f2 = (1 - f1) ** 2
    mask1 = (f1 > 0.5) & (f1 <= 0.75)
    f2[mask1] = 0.5 * (1 - f1[mask1])
    mask2 = f1 > 0.75
    f2[mask2] = 0.25 * np.sqrt(1 - f1[mask2])
    return np.column_stack([f1, f2])


def CF7_PF(N: int, M: int = 2) -> np.ndarray:
    """Pareto Front for CF7."""
    return CF6_PF(N, M)


def CF8_PF(N: int, M: int = 3) -> np.ndarray:
    """Pareto Front for CF8."""
    n_per_group = max(1, N // 5)
    R = []
    for i in range(5):
        r3 = np.sin(np.linspace(0, 1, n_per_group) * np.pi / 2)
        r1 = np.sqrt(i / 4.0 * (1 - r3 ** 2))
        r2 = np.sqrt(np.maximum(1 - r1 ** 2 - r3 ** 2, 0))
        R.append(np.column_stack([r1, r2, r3]))
    return np.vstack(R)


def CF9_PF(N: int, M: int = 3) -> np.ndarray:
    """Pareto Front for CF9."""
    W, _ = uniform_point(N, 3)
    norm = np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))
    R = W / norm
    term = 1 - R[:, 2] ** 2
    c1 = np.sqrt(term / 4.0)
    c2 = np.sqrt(term / 2.0)
    c3 = np.sqrt(3 * term / 4.0)
    mask = ((R[:, 0] > 1e-5) & (R[:, 0] < c1)) | \
           ((R[:, 0] > c2) & (R[:, 0] < c3))
    return R[~mask]


def CF10_PF(N: int, M: int = 3) -> np.ndarray:
    """Pareto Front for CF10."""
    return CF9_PF(N, M)


SETTINGS = {
    'metric': 'IGD',
    'n_ref': 10000,
    'CF1': {'T1': CF1_PF},
    'CF2': {'T1': CF2_PF},
    'CF3': {'T1': CF3_PF},
    'CF4': {'T1': CF4_PF},
    'CF5': {'T1': CF5_PF},
    'CF6': {'T1': CF6_PF},
    'CF7': {'T1': CF7_PF},
    'CF8': {'T1': CF8_PF},
    'CF9': {'T1': CF9_PF},
    'CF10': {'T1': CF10_PF},
}
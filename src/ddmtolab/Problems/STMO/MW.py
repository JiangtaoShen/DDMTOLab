import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.algo_utils import nd_sort
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point


class MW:
    """
    Implementation of the MW test suite for constrained multi-objective optimization.

    The MW test problems are standard constrained multi-objective
    optimization benchmarks proposed by Ma and Wang (2019).

    References
    ----------
    Z. Ma and Y. Wang. "Evolutionary constrained multiobjective optimization:
    Test suite construction and performance comparisons."
    IEEE Transactions on Evolutionary Computation, 2019, 23(6): 972-986.
    """

    def MW1(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW1** problem.

        MW1 features a linear Pareto front with a nonlinear constraint boundary
        that creates a challenging feasible region.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW1 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function
            # g = 1 + sum(1 - exp(-10*((x_i^(D-M)) - 0.5 - (i-1)/(2*D))^2))
            indices = np.arange(M, D)
            terms = (x[:, M:] ** (D - M) - 0.5 - (indices - M) / (2 * D)) ** 2
            g = 1 + np.sum(1 - np.exp(-10 * terms), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0]
            obj[:, 1] = g * (1 - 0.85 * obj[:, 0] / g)

            return obj

        def C1(x):
            obj = T1(x)

            # Constraint: f1 + f2 - 1 - 0.5*sin(2*pi*l)^8 <= 0
            # where l = sqrt(2)*f2 - sqrt(2)*f1
            l = np.sqrt(2) * obj[:, 1] - np.sqrt(2) * obj[:, 0]
            con = np.sum(obj, axis=1) - 1 - 0.5 * np.sin(2 * np.pi * l) ** 8

            return con

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW2(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW2** problem.

        MW2 features a linear Pareto front (f2 = 1 - f1) with a multi-modal
        g function and a nonlinear constraint boundary.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW2 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate z transformation
            # z = 1 - exp(-10*(x_i - (i-1)/D)^2)
            indices = np.arange(M, D)
            z = 1 - np.exp(-10 * (x[:, M:] - (indices - M) / D) ** 2)

            # Calculate g function with multi-modal term
            # g = 1 + sum(1.5 + (0.1/D)*z^2 - 1.5*cos(2*pi*z))
            g = 1 + np.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * np.cos(2 * np.pi * z), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0]
            obj[:, 1] = g * (1 - obj[:, 0] / g)

            return obj

        def C1(x):
            obj = T1(x)

            # Constraint: f1 + f2 - 1 - 0.5*sin(3*pi*l)^8 <= 0
            # where l = sqrt(2)*f2 - sqrt(2)*f1
            l = np.sqrt(2) * obj[:, 1] - np.sqrt(2) * obj[:, 0]
            con = np.sum(obj, axis=1) - 1 - 0.5 * np.sin(3 * np.pi * l) ** 8

            return con

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW3(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW3** problem.

        MW3 features a linear Pareto front (f2 = 1 - f1) with two constraints
        that create a complex feasible region.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW3 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function
            # g = 1 + sum(2*(x_i + (x_{i-1} - 0.5)^2 - 1)^2)
            # Note: x_{i-1} for i >= M means x[:, M-1:end-1]
            term = x[:, M:] + (x[:, M - 1:-1] - 0.5) ** 2 - 1
            g = 1 + np.sum(2 * term ** 2, axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = x[:, 0]
            obj[:, 1] = g * (1 - obj[:, 0] / g)

            return obj

        def C1(x):
            obj = T1(x)

            # l = sqrt(2)*f2 - sqrt(2)*f1
            l = np.sqrt(2) * obj[:, 1] - np.sqrt(2) * obj[:, 0]

            # Two constraints:
            # c1: f1 + f2 - 1.05 - 0.45*sin(0.75*pi*l)^6 <= 0
            # c2: 0.85 - f1 - f2 + 0.3*sin(0.75*pi*l)^2 <= 0
            c1 = np.sum(obj, axis=1) - 1.05 - 0.45 * np.sin(0.75 * np.pi * l) ** 6
            c2 = 0.85 - np.sum(obj, axis=1) + 0.3 * np.sin(0.75 * np.pi * l) ** 2

            return np.column_stack([c1, c2])

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW4(self, M=3, dim=None) -> MTOP:
        """
        Generates the **MW4** problem.

        MW4 is a multi/many-objective constrained problem with a simplex-shaped
        Pareto front and a nonlinear constraint.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW4 task.
        """
        if dim is None:
            dim = 15

        # Store M as a local variable to ensure it's captured correctly
        num_obj = M

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function
            # g = sum(1 - exp(-10*((x_i^(D-M)) - 0.5 - (i-1)/(2*D))^2))
            indices = np.arange(num_obj, D)
            terms = (x[:, num_obj:] ** (D - num_obj) - 0.5 - (indices - num_obj) / (2 * D)) ** 2
            g = np.sum(1 - np.exp(-10 * terms), axis=1)

            # Calculate objectives using cumulative product
            # PopObj = (1+g) .* flip(cumprod([ones, x(:,1:M-1)], 2), 2) .* [ones, 1-x(:,M-1:-1:1)]
            obj = np.zeros((N, num_obj))

            for i in range(num_obj):
                obj[:, i] = 1 + g
                # Cumulative product part
                for j in range(num_obj - i - 1):
                    obj[:, i] *= x[:, j]
                # (1 - x) part
                if i > 0:
                    obj[:, i] *= (1 - x[:, num_obj - i - 1])

            return obj

        def C1(x):
            obj = T1(x)

            # l = f_M - sum(f_1, ..., f_{M-1})
            l = obj[:, -1] - np.sum(obj[:, :-1], axis=1)

            # Constraint: sum(f) - (1 + 0.4*sin(2.5*pi*l)^8) <= 0
            con = np.sum(obj, axis=1) - (1 + 0.4 * np.sin(2.5 * np.pi * l) ** 8)

            return con

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW5(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW5** problem.

        MW5 features a quarter-circle Pareto front with three constraints
        that create a complex feasible region with disconnected segments.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW5 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function
            indices = np.arange(M, D)
            terms = (x[:, M:] ** (D - M) - 0.5 - (indices - M) / (2 * D)) ** 2
            g = 1 + np.sum(1 - np.exp(-10 * terms), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0]
            obj[:, 1] = g * np.sqrt(1 - (obj[:, 0] / g) ** 2)

            return obj

        def C1(x):
            obj = T1(x)

            # Calculate angles
            l1 = np.arctan2(obj[:, 1], obj[:, 0])
            l2 = 0.5 * np.pi - 2 * np.abs(l1 - 0.25 * np.pi)

            # Three constraints
            c1 = obj[:, 0] ** 2 + obj[:, 1] ** 2 - (1.7 - 0.2 * np.sin(2 * l1)) ** 2
            c2 = (1 + 0.5 * np.sin(6 * l2 ** 3)) ** 2 - obj[:, 0] ** 2 - obj[:, 1] ** 2
            c3 = (1 - 0.45 * np.sin(6 * l2 ** 3)) ** 2 - obj[:, 0] ** 2 - obj[:, 1] ** 2

            return np.column_stack([c1, c2, c3])

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW6(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW6** problem.

        MW6 features an elliptical Pareto front with a complex constraint
        based on angular position.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW6 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate z transformation
            indices = np.arange(M, D)
            z = 1 - np.exp(-10 * (x[:, M:] - (indices - M) / D) ** 2)

            # Calculate g function with multi-modal term
            g = 1 + np.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * np.cos(2 * np.pi * z), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0] * 1.0999
            obj[:, 1] = g * np.sqrt(1.1 * 1.1 - (obj[:, 0] / g) ** 2)

            return obj

        def C1(x):
            obj = T1(x)

            # Calculate l factor based on angle
            l = np.cos(6 * np.arctan2(obj[:, 1], obj[:, 0]) ** 4) ** 10

            # Constraint: (f1/(1+0.15*l))^2 + (f2/(1+0.75*l))^2 - 1 <= 0
            con = (obj[:, 0] / (1 + 0.15 * l)) ** 2 + (obj[:, 1] / (1 + 0.75 * l)) ** 2 - 1

            return con

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW7(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW7** problem.

        MW7 features a quarter-circle Pareto front with two constraints
        that create a complex angular-dependent feasible region.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW7 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function (same as MW3)
            term = x[:, M:] + (x[:, M - 1:-1] - 0.5) ** 2 - 1
            g = 1 + np.sum(2 * term ** 2, axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0]
            obj[:, 1] = g * np.sqrt(1 - (obj[:, 0] / g) ** 2)

            return obj

        def C1(x):
            obj = T1(x)

            # Calculate angle
            l = np.arctan2(obj[:, 1], obj[:, 0])

            # Two constraints
            c1 = obj[:, 0] ** 2 + obj[:, 1] ** 2 - (1.2 + 0.4 * np.sin(4 * l) ** 16) ** 2
            c2 = (1.15 - 0.2 * np.sin(4 * l) ** 8) ** 2 - obj[:, 0] ** 2 - obj[:, 1] ** 2

            return np.column_stack([c1, c2])

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW8(self, M=3, dim=None) -> MTOP:
        """
        Generates the **MW8** problem.

        MW8 is a multi/many-objective constrained problem with a normalized
        spherical Pareto front and a constraint based on the angular position
        of the last objective.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW8 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate z transformation
            # z = 1 - exp(-10*(x_i - (i-1)/D)^2)
            indices = np.arange(M, D)
            z = 1 - np.exp(-10 * (x[:, M:] - (indices - M) / D) ** 2)

            # Calculate g function with multi-modal term
            # g = sum(1.5 + (0.1/D)*z^2 - 1.5*cos(2*pi*z))
            g = np.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * np.cos(2 * np.pi * z), axis=1)

            # Calculate objectives using cumulative product
            # PopObj = (1+g) .* flip(cumprod([ones, cos(x(:,1:M-1)*pi/2)], 2), 2) .* [ones, sin(x(:,M-1:-1:1)*pi/2)]
            obj = np.zeros((N, M))

            for i in range(M):
                obj[:, i] = 1 + g
                # Cumulative product of cosines
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x[:, j] * np.pi / 2)
                # Sine part
                if i > 0:
                    obj[:, i] *= np.sin(x[:, M - i - 1] * np.pi / 2)

            return obj

        def C1(x):
            obj = T1(x)

            # Calculate l = asin(f_M / sqrt(sum(f^2)))
            l = np.arcsin(obj[:, -1] / np.sqrt(np.sum(obj ** 2, axis=1)))

            # Constraint: sum(f^2) - (1.25 - 0.5*sin(6*l)^2)^2 <= 0
            con = np.sum(obj ** 2, axis=1) - (1.25 - 0.5 * np.sin(6 * l) ** 2) ** 2

            return con

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW9(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW9** problem.

        MW9 features a concave Pareto front (f2 = 1 - f1^0.6) with three
        constraints that create a complex feasible region with multiple
        disconnected segments.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW9 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function
            # g = 1 + sum(1 - exp(-10*((x_i^(D-M)) - 0.5 - (i-1)/(2*D))^2))
            indices = np.arange(M, D)
            terms = (x[:, M:] ** (D - M) - 0.5 - (indices - M) / (2 * D)) ** 2
            g = 1 + np.sum(1 - np.exp(-10 * terms), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0]
            obj[:, 1] = g * (1 - (obj[:, 0] / g) ** 0.6)

            return obj

        def C1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Recalculate g function
            indices = np.arange(M, D)
            terms = (x[:, M:] ** (D - M) - 0.5 - (indices - M) / (2 * D)) ** 2
            g = 1 + np.sum(1 - np.exp(-10 * terms), axis=1)

            # Recalculate objectives
            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0]
            obj[:, 1] = g * (1 - (obj[:, 0] / g) ** 0.6)

            # Three constraint terms
            # T1 = (1 - 0.64*f1^2 - f2) * (1 - 0.36*f1^2 - f2)
            T1 = (1 - 0.64 * obj[:, 0] ** 2 - obj[:, 1]) * \
                 (1 - 0.36 * obj[:, 0] ** 2 - obj[:, 1])

            # T2 = 1.35^2 - (f1 + 0.35)^2 - f2
            T2 = 1.35 ** 2 - (obj[:, 0] + 0.35) ** 2 - obj[:, 1]

            # T3 = 1.15^2 - (f1 + 0.15)^2 - f2
            T3 = 1.15 ** 2 - (obj[:, 0] + 0.15) ** 2 - obj[:, 1]

            # Constraint: min(T1, T2*T3) <= 0
            con = np.minimum(T1, T2 * T3)

            return con

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW10(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW10** problem.

        MW10 features a convex Pareto front (f2 = 1 - f1^2) with three
        constraints that create a complex feasible region with multiple
        disconnected segments.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW10 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate z transformation
            # z = 1 - exp(-10*(x_i - (i-1)/D)^2)
            indices = np.arange(M, D)
            z = 1 - np.exp(-10 * (x[:, M:] - (indices - M) / D) ** 2)

            # Calculate g function with multi-modal term
            # g = 1 + sum(1.5 + (0.1/D)*z^2 - 1.5*cos(2*pi*z))
            g = 1 + np.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * np.cos(2 * np.pi * z), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * (x[:, 0] ** D)
            obj[:, 1] = g * (1 - (obj[:, 0] / g) ** 2)

            return obj

        def C1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Recalculate z transformation
            indices = np.arange(M, D)
            z = 1 - np.exp(-10 * (x[:, M:] - (indices - M) / D) ** 2)

            # Recalculate g function
            g = 1 + np.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * np.cos(2 * np.pi * z), axis=1)

            # Recalculate objectives
            obj = np.zeros((N, M))
            obj[:, 0] = g * (x[:, 0] ** D)
            obj[:, 1] = g * (1 - (obj[:, 0] / g) ** 2)

            # Three constraints
            # c1: -(2 - 4*f1^2 - f2) * (2 - 8*f1^2 - f2) <= 0
            c1 = -(2 - 4 * obj[:, 0] ** 2 - obj[:, 1]) * \
                 (2 - 8 * obj[:, 0] ** 2 - obj[:, 1])

            # c2: (2 - 2*f1^2 - f2) * (2 - 16*f1^2 - f2) <= 0
            c2 = (2 - 2 * obj[:, 0] ** 2 - obj[:, 1]) * \
                 (2 - 16 * obj[:, 0] ** 2 - obj[:, 1])

            # c3: (1 - f1^2 - f2) * (1.2 - 1.2*f1^2 - f2) <= 0
            c3 = (1 - obj[:, 0] ** 2 - obj[:, 1]) * \
                 (1.2 - 1.2 * obj[:, 0] ** 2 - obj[:, 1])

            return np.column_stack([c1, c2, c3])

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW11(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW11** problem.

        MW11 features a quarter-circle Pareto front with four constraints
        that create a highly complex feasible region with multiple
        disconnected segments.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW11 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function (same as MW3 and MW7)
            # g = 1 + sum(2*(x_i + (x_{i-1} - 0.5)^2 - 1)^2)
            term = x[:, M:] + (x[:, M - 1:-1] - 0.5) ** 2 - 1
            g = 1 + np.sum(2 * term ** 2, axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0] * np.sqrt(1.9999)
            obj[:, 1] = g * np.sqrt(2 - (obj[:, 0] / g) ** 2)

            return obj

        def C1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Recalculate g function
            term = x[:, M:] + (x[:, M - 1:-1] - 0.5) ** 2 - 1
            g = 1 + np.sum(2 * term ** 2, axis=1)

            # Recalculate objectives
            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0] * np.sqrt(1.9999)
            obj[:, 1] = g * np.sqrt(2 - (obj[:, 0] / g) ** 2)

            # Four constraints
            # c1: -(3 - f1^2 - f2) * (3 - 2*f1^2 - f2) <= 0
            c1 = -(3 - obj[:, 0] ** 2 - obj[:, 1]) * \
                 (3 - 2 * obj[:, 0] ** 2 - obj[:, 1])

            # c2: (3 - 0.625*f1^2 - f2) * (3 - 7*f1^2 - f2) <= 0
            c2 = (3 - 0.625 * obj[:, 0] ** 2 - obj[:, 1]) * \
                 (3 - 7 * obj[:, 0] ** 2 - obj[:, 1])

            # c3: -(1.62 - 0.18*f1^2 - f2) * (1.125 - 0.125*f1^2 - f2) <= 0
            c3 = -(1.62 - 0.18 * obj[:, 0] ** 2 - obj[:, 1]) * \
                 (1.125 - 0.125 * obj[:, 0] ** 2 - obj[:, 1])

            # c4: (2.07 - 0.23*f1^2 - f2) * (0.63 - 0.07*f1^2 - f2) <= 0
            c4 = (2.07 - 0.23 * obj[:, 0] ** 2 - obj[:, 1]) * \
                 (0.63 - 0.07 * obj[:, 0] ** 2 - obj[:, 1])

            return np.column_stack([c1, c2, c3, c4])

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW12(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW12** problem.

        MW12 features a complex oscillating Pareto front with two constraints
        involving sinusoidal terms that create intricate feasible regions.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW12 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate g function
            # g = 1 + sum(1 - exp(-10*((x_i^(D-M)) - 0.5 - (i-1)/(2*D))^2))
            indices = np.arange(M, D)
            terms = (x[:, M:] ** (D - M) - 0.5 - (indices - M) / (2 * D)) ** 2
            g = 1 + np.sum(1 - np.exp(-10 * terms), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0]
            obj[:, 1] = g * (0.85 - 0.8 * (obj[:, 0] / g) -
                             0.08 * np.abs(np.sin(3.2 * np.pi * (obj[:, 0] / g))))

            return obj

        def C1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Recalculate g function
            indices = np.arange(M, D)
            terms = (x[:, M:] ** (D - M) - 0.5 - (indices - M) / (2 * D)) ** 2
            g = 1 + np.sum(1 - np.exp(-10 * terms), axis=1)

            # Recalculate objectives
            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0]
            obj[:, 1] = g * (0.85 - 0.8 * (obj[:, 0] / g) -
                             0.08 * np.abs(np.sin(3.2 * np.pi * (obj[:, 0] / g))))

            # Two constraints with sinusoidal terms
            # c1: (1 - 0.8*f1 - f2 + 0.08*sin(2*pi*(f2 - f1/1.5))) *
            #     (1.8 - 1.125*f1 - f2 + 0.08*sin(2*pi*(f2/1.8 - f1/1.6))) <= 0
            c1 = (1 - 0.8 * obj[:, 0] - obj[:, 1] +
                  0.08 * np.sin(2 * np.pi * (obj[:, 1] - obj[:, 0] / 1.5))) * \
                 (1.8 - 1.125 * obj[:, 0] - obj[:, 1] +
                  0.08 * np.sin(2 * np.pi * (obj[:, 1] / 1.8 - obj[:, 0] / 1.6)))

            # c2: -(1 - 0.625*f1 - f2 + 0.08*sin(2*pi*(f2 - f1/1.6))) *
            #      (1.4 - 0.875*f1 - f2 + 0.08*sin(2*pi*(f2/1.4 - f1/1.6))) <= 0
            c2 = -(1 - 0.625 * obj[:, 0] - obj[:, 1] +
                   0.08 * np.sin(2 * np.pi * (obj[:, 1] - obj[:, 0] / 1.6))) * \
                 (1.4 - 0.875 * obj[:, 0] - obj[:, 1] +
                  0.08 * np.sin(2 * np.pi * (obj[:, 1] / 1.4 - obj[:, 0] / 1.6)))

            return np.column_stack([c1, c2])

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW13(self, M=2, dim=None) -> MTOP:
        """
        Generates the **MW13** problem.

        MW13 features a complex Pareto front involving exponential and
        sinusoidal terms with two constraints that create intricate
        feasible regions.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 2).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW13 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Calculate z transformation
            # z = 1 - exp(-10*(x_i - (i-1)/D)^2)
            indices = np.arange(M, D)
            z = 1 - np.exp(-10 * (x[:, M:] - (indices - M) / D) ** 2)

            # Calculate g function with multi-modal term
            # g = 1 + sum(1.5 + (0.1/D)*z^2 - 1.5*cos(2*pi*z))
            g = 1 + np.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * np.cos(2 * np.pi * z), axis=1)

            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0] * 1.5
            obj[:, 1] = g * (5 - np.exp(obj[:, 0] / g) -
                             np.abs(0.5 * np.sin(3 * np.pi * obj[:, 0] / g)))

            return obj

        def C1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Recalculate z transformation
            indices = np.arange(M, D)
            z = 1 - np.exp(-10 * (x[:, M:] - (indices - M) / D) ** 2)

            # Recalculate g function
            g = 1 + np.sum(1.5 + (0.1 / D) * z ** 2 - 1.5 * np.cos(2 * np.pi * z), axis=1)

            # Recalculate objectives
            obj = np.zeros((N, M))
            obj[:, 0] = g * x[:, 0] * 1.5
            obj[:, 1] = g * (5 - np.exp(obj[:, 0] / g) -
                             np.abs(0.5 * np.sin(3 * np.pi * obj[:, 0] / g)))

            # Two constraints
            # c1: (5 - exp(f1) - 0.5*sin(3*pi*f1) - f2) *
            #     (5 - (1 + 0.4*f1) - 0.5*sin(3*pi*f1) - f2) <= 0
            c1 = (5 - np.exp(obj[:, 0]) - 0.5 * np.sin(3 * np.pi * obj[:, 0]) - obj[:, 1]) * \
                 (5 - (1 + 0.4 * obj[:, 0]) - 0.5 * np.sin(3 * np.pi * obj[:, 0]) - obj[:, 1])

            # c2: -(5 - (1 + f1 + 0.5*f1^2) - 0.5*sin(3*pi*f1) - f2) *
            #      (5 - (1 + 0.7*f1) - 0.5*sin(3*pi*f1) - f2) <= 0
            c2 = -(5 - (1 + obj[:, 0] + 0.5 * obj[:, 0] ** 2) -
                   0.5 * np.sin(3 * np.pi * obj[:, 0]) - obj[:, 1]) * \
                 (5 - (1 + 0.7 * obj[:, 0]) - 0.5 * np.sin(3 * np.pi * obj[:, 0]) - obj[:, 1])

            return np.column_stack([c1, c2])

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def MW14(self, M=3, dim=None) -> MTOP:
        """
        Generates the **MW14** problem.

        MW14 is a multi/many-objective constrained problem with a complex
        Pareto front involving exponential and sinusoidal terms, and a
        single constraint creating a disconnected feasible region.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, it is set to 15 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the MW14 task.
        """
        if dim is None:
            dim = 15

        def T1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Scale x by 1.5
            X = 1.5 * x

            # Calculate g function (same as MW3, MW7, MW11)
            # g = sum(2*(X_i + (X_{i-1} - 0.5)^2 - 1)^2)
            term = X[:, M:] + (X[:, M - 1:-1] - 0.5) ** 2 - 1
            g = np.sum(2 * term ** 2, axis=1)

            obj = np.zeros((N, M))
            # First M-1 objectives are simply the scaled decision variables
            obj[:, :M - 1] = X[:, :M - 1]

            # Last objective
            # f_M = ((1+g)/(M-1)) * sum(6 - exp(f_i) - 1.5*sin(1.1*pi*f_i^2)) for i=1 to M-1
            obj[:, M - 1] = ((1 + g) / (M - 1)) * \
                            np.sum(6 - np.exp(obj[:, :M - 1]) -
                                   1.5 * np.sin(1.1 * np.pi * obj[:, :M - 1] ** 2), axis=1)

            return obj

        def C1(x):
            x = np.atleast_2d(x)
            N, D = x.shape

            # Scale x by 1.5
            X = 1.5 * x

            # Recalculate g function
            term = X[:, M:] + (X[:, M - 1:-1] - 0.5) ** 2 - 1
            g = np.sum(2 * term ** 2, axis=1)

            # Recalculate objectives
            obj = np.zeros((N, M))
            obj[:, :M - 1] = X[:, :M - 1]
            obj[:, M - 1] = ((1 + g) / (M - 1)) * \
                            np.sum(6 - np.exp(obj[:, :M - 1]) -
                                   1.5 * np.sin(1.1 * np.pi * obj[:, :M - 1] ** 2), axis=1)

            # Constraint:
            # a = 1 + f_i + 0.5*f_i^2 + 1.5*sin(1.1*pi*f_i^2) for i=1 to M-1
            # con = f_M - 1/(M-1) * sum(6.1 - a) <= 0
            a = 1 + obj[:, :M - 1] + 0.5 * obj[:, :M - 1] ** 2 + \
                1.5 * np.sin(1.1 * np.pi * obj[:, :M - 1] ** 2)

            con = obj[:, M - 1] - (1 / (M - 1)) * np.sum(6.1 - a, axis=1)

            return con

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem


# --- Pareto Front (PF) Functions ---

def MW1_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW1.

    The PF is a linear segment f2 = 1 - 0.85*f1, filtered by the constraint.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the PF points (N' <= N due to filtering).
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 1 - 0.85 * R[:, 0]

    # Calculate constraint value
    l = np.sqrt(2) * R[:, 1] - np.sqrt(2) * R[:, 0]
    c = 1 - R[:, 0] - R[:, 1] + 0.5 * np.sin(2 * np.pi * l) ** 8

    # Remove points where c < 0
    R = R[c >= 0, :]

    return R


def MW2_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW2.

    The PF is a linear segment f2 = 1 - f1 (complete line).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1

    return np.column_stack([f1, f2])


def MW3_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW3.

    The PF is a linear segment f2 = 1 - f1, adjusted to satisfy constraint c2.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 1 - R[:, 0]

    # Check constraint c2: 0.85 - f1 - f2 + 0.3*sin(0.75*pi*sqrt(2)*(f2-f1))^2 > 0
    invalid = (0.85 - R[:, 0] - R[:, 1] +
               0.3 * np.sin(0.75 * np.pi * np.sqrt(2) * (R[:, 1] - R[:, 0])) ** 2) > 0

    # Adjust invalid points
    max_iterations = 1000
    iteration = 0
    while np.any(invalid) and iteration < max_iterations:
        R[invalid, :] = R[invalid, :] * 1.001
        invalid = (0.85 - R[:, 0] - R[:, 1] +
                   0.3 * np.sin(0.75 * np.pi * np.sqrt(2) * (R[:, 1] - R[:, 0])) ** 2) > 0
        iteration += 1

    return R


def MW4_PF(N: int, M: int = 3) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW4.

    The PF is a simplex with sum(f) = 1, filtered by the constraint.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the PF points (N' <= N due to filtering).
    """
    R, _ = uniform_point(N, M)

    # Calculate l = f_M - sum(f_1, ..., f_{M-1})
    l = R[:, -1] - np.sum(R[:, :-1], axis=1)

    # Calculate constraint: (1 + 0.4*sin(2.5*pi*l)^8) - sum(R)
    c = (1 + 0.4 * np.sin(2.5 * np.pi * l) ** 8) - np.sum(R, axis=1)

    # Remove points where c < 0
    R = R[c >= 0, :]

    return R


def MW5_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW5.

    The PF consists of specific disconnected segments on a quarter circle.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF (not used, returns fixed points).
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (16, M) representing the PF points.
    """
    # First half of the PF
    R = np.array([
        [0.0000, 1.0000],
        [0.3922, 0.9199],
        [0.4862, 0.8739],
        [0.5490, 0.8358],
        [0.5970, 0.8023],
        [0.6359, 0.7719],
        [0.6686, 0.7436],
        [0.6969, 0.7174]
    ])

    # Second half (flipped)
    R_flip = np.flip(R, axis=1)

    # Combine both halves
    R = np.vstack([R, R_flip])

    return R


def MW6_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW6.

    The PF is an arc normalized to radius 1.1, filtered by the constraint.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the PF points (N' <= N due to filtering).
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 1 - R[:, 0]

    # Normalize to radius 1.1 (since sqrt(sum(R^2,2)/1.21) = sqrt(R^2)/1.1)
    R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True) / 1.21)

    # Calculate l factor
    l = np.cos(6 * np.arctan2(R[:, 1], R[:, 0]) ** 4) ** 10

    # Calculate constraint
    c = 1 - (R[:, 0] / (1 + 0.15 * l)) ** 2 - (R[:, 1] / (1 + 0.75 * l)) ** 2

    # Remove points where c < 0
    R = R[c >= 0, :]

    return R


# def MW7_PF(N: int, M: int = 2) -> np.ndarray:
#     """
#     Computes the Pareto Front (PF) for MW7.
#
#     The PF is a quarter circle normalized to unit radius, adjusted to satisfy
#     constraint c2, and then filtered by non-dominated sorting.
#
#     Parameters
#     ----------
#     N : int
#         Number of points to generate on the PF.
#     M : int, optional
#         Number of objectives (default is 2).
#
#     Returns
#     -------
#     np.ndarray
#         Array of shape (N', M) representing the non-dominated PF points.
#     """
#     R = np.zeros((N, M))
#     R[:, 0] = np.linspace(0, 1, N)
#     R[:, 1] = 1 - R[:, 0]
#
#     # Normalize to unit circle
#     R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))
#
#     # Check constraint c2: (1.15 - 0.2*sin(4*atan(f2/f1))^8)^2 - f1^2 - f2^2 > 0
#     invalid = ((1.15 - 0.2 * np.sin(4 * np.arctan2(R[:, 1], R[:, 0])) ** 8) ** 2 -
#                R[:, 0] ** 2 - R[:, 1] ** 2) > 0
#
#     # Adjust invalid points
#     max_iterations = 1000
#     iteration = 0
#     while np.any(invalid) and iteration < max_iterations:
#         R[invalid, :] = R[invalid, :] * 1.001
#         invalid = ((1.15 - 0.2 * np.sin(4 * np.arctan2(R[:, 1], R[:, 0])) ** 8) ** 2 -
#                    R[:, 0] ** 2 - R[:, 1] ** 2) > 0
#         iteration += 1
#
#     # Keep only non-dominated solutions (front_no == 1)
#     front_no, _ = nd_sort(R, N)
#     R = R[front_no == 1, :]
#
#     return R


def MW7_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW7.

    The PF is a quarter circle normalized to unit radius, adjusted to satisfy
    constraint c2, and then filtered by non-dominated sorting.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the non-dominated PF points.
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 1 - R[:, 0]

    # Normalize to unit circle
    norms = np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))
    R = R / norms

    # Pre-compute angle (only needs to be updated for invalid points)
    angle = np.arctan2(R[:, 1], R[:, 0])

    # Check constraint c2: (1.15 - 0.2*sin(4*angle)^8)^2 - f1^2 - f2^2 > 0
    sin_term = np.sin(4 * angle) ** 8
    r_squared = R[:, 0] ** 2 + R[:, 1] ** 2
    invalid = ((1.15 - 0.2 * sin_term) ** 2 - r_squared) > 0

    # Adjust invalid points
    max_iterations = 1000
    iteration = 0
    while np.any(invalid) and iteration < max_iterations:
        R[invalid, :] *= 1.001
        # Only recompute for invalid points
        angle[invalid] = np.arctan2(R[invalid, 1], R[invalid, 0])
        sin_term[invalid] = np.sin(4 * angle[invalid]) ** 8
        r_squared[invalid] = R[invalid, 0] ** 2 + R[invalid, 1] ** 2
        invalid = ((1.15 - 0.2 * sin_term) ** 2 - r_squared) > 0
        iteration += 1

    # Keep only non-dominated solutions (front_no == 1)
    front_no, _ = nd_sort(R, N)
    R = R[front_no == 1, :]

    return R


def MW8_PF(N: int, M: int = 3) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW8.

    The PF is a normalized sphere filtered by the constraint based on
    the angular position of the last objective.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the PF points (N' <= N due to filtering).
    """
    # Generate uniformly distributed points on a simplex
    R, _ = uniform_point(N, M)

    # Normalize to unit sphere
    R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))

    # Calculate l = asin(f_M)
    l = np.arcsin(R[:, -1])

    # Calculate constraint: 1 - (1.25 - 0.5*sin(6*l)^2)^2 > 0
    c = 1 - (1.25 - 0.5 * np.sin(6 * l) ** 2) ** 2

    # Remove points where c > 0 (constraint violated)
    R = R[c <= 0, :]

    return R


def MW9_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW9.

    The PF is a concave curve f2 = 1 - f1^0.6, adjusted to satisfy
    the complex three-part constraint, and filtered by non-dominated sorting.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the non-dominated PF points.
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 1 - R[:, 0] ** 0.6

    # Calculate constraint terms
    T1 = (1 - 0.64 * R[:, 0] ** 2 - R[:, 1]) * \
         (1 - 0.36 * R[:, 0] ** 2 - R[:, 1])
    T2 = 1.35 ** 2 - (R[:, 0] + 0.35) ** 2 - R[:, 1]
    T3 = 1.15 ** 2 - (R[:, 0] + 0.15) ** 2 - R[:, 1]

    # Check constraint: min(T1, T2*T3) > 0
    invalid = np.minimum(T1, T2 * T3) > 0

    # Adjust invalid points
    max_iterations = 1000
    iteration = 0
    while np.any(invalid) and iteration < max_iterations:
        R[invalid, :] = R[invalid, :] * 1.001

        # Recalculate constraint terms for invalid points
        T1[invalid] = (1 - 0.64 * R[invalid, 0] ** 2 - R[invalid, 1]) * \
                      (1 - 0.36 * R[invalid, 0] ** 2 - R[invalid, 1])
        T2[invalid] = 1.35 ** 2 - (R[invalid, 0] + 0.35) ** 2 - R[invalid, 1]
        T3[invalid] = 1.15 ** 2 - (R[invalid, 0] + 0.15) ** 2 - R[invalid, 1]

        invalid = np.minimum(T1, T2 * T3) > 0
        iteration += 1

    # Keep only non-dominated solutions (front_no == 1)
    front_no, _ = nd_sort(R, N)
    R = R[front_no == 1, :]

    return R


def MW10_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW10.

    The PF is a convex curve f2 = 1 - f1^2, adjusted to satisfy
    three complex constraints, and filtered by non-dominated sorting.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the non-dominated PF points.
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 1 - R[:, 0] ** 2

    # Calculate constraint terms
    c1 = (2 - 4 * R[:, 0] ** 2 - R[:, 1]) * (2 - 8 * R[:, 0] ** 2 - R[:, 1])
    c2 = (2 - 2 * R[:, 0] ** 2 - R[:, 1]) * (2 - 16 * R[:, 0] ** 2 - R[:, 1])
    c3 = (1 - R[:, 0] ** 2 - R[:, 1]) * (1.2 - 1.2 * R[:, 0] ** 2 - R[:, 1])

    # Check constraints: c1 < 0 OR c2 > 0 OR c3 > 0 means invalid
    invalid = (c1 < 0) | (c2 > 0) | (c3 > 0)

    # Adjust invalid points
    max_iterations = 1000
    iteration = 0
    while np.any(invalid) and iteration < max_iterations:
        R[invalid, :] = R[invalid, :] * 1.001

        # Remove points that exceed threshold
        mask = np.any(R > 1.3, axis=1)
        R = R[~mask, :]

        if len(R) == 0:
            break

        # Recalculate constraint terms
        c1 = (2 - 4 * R[:, 0] ** 2 - R[:, 1]) * (2 - 8 * R[:, 0] ** 2 - R[:, 1])
        c2 = (2 - 2 * R[:, 0] ** 2 - R[:, 1]) * (2 - 16 * R[:, 0] ** 2 - R[:, 1])
        c3 = (1 - R[:, 0] ** 2 - R[:, 1]) * (1.2 - 1.2 * R[:, 0] ** 2 - R[:, 1])

        invalid = (c1 < 0) | (c2 > 0) | (c3 > 0)
        iteration += 1

    if len(R) > 0:
        # Keep only non-dominated solutions (front_no == 1)
        front_no, _ = nd_sort(R, len(R))
        R = R[front_no == 1, :]

    return R


def MW11_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW11.

    The PF is a quarter circle normalized to radius sqrt(2), adjusted to
    satisfy four complex constraints, and filtered by non-dominated sorting.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the non-dominated PF points.
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 1 - R[:, 0]

    # Normalize to radius sqrt(2)
    R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True) / 2)

    # Calculate constraint terms
    c1 = (3 - R[:, 0] ** 2 - R[:, 1]) * (3 - 2 * R[:, 0] ** 2 - R[:, 1])
    c2 = (3 - 0.625 * R[:, 0] ** 2 - R[:, 1]) * (3 - 7 * R[:, 0] ** 2 - R[:, 1])
    c3 = (1.62 - 0.18 * R[:, 0] ** 2 - R[:, 1]) * (1.125 - 0.125 * R[:, 0] ** 2 - R[:, 1])
    c4 = (2.07 - 0.23 * R[:, 0] ** 2 - R[:, 1]) * (0.63 - 0.07 * R[:, 0] ** 2 - R[:, 1])

    # Check constraints: c1 < 0 OR c2 > 0 OR c3 < 0 OR c4 > 0 means invalid
    invalid = (c1 < 0) | (c2 > 0) | (c3 < 0) | (c4 > 0)

    # Adjust invalid points
    max_iterations = 1000
    iteration = 0
    while np.any(invalid) and iteration < max_iterations:
        R[invalid, :] = R[invalid, :] * 1.001

        # Remove points that exceed threshold
        mask = np.any(R > 2.2, axis=1)
        R = R[~mask, :]

        if len(R) == 0:
            break

        # Recalculate constraint terms
        c1 = (3 - R[:, 0] ** 2 - R[:, 1]) * (3 - 2 * R[:, 0] ** 2 - R[:, 1])
        c2 = (3 - 0.625 * R[:, 0] ** 2 - R[:, 1]) * (3 - 7 * R[:, 0] ** 2 - R[:, 1])
        c3 = (1.62 - 0.18 * R[:, 0] ** 2 - R[:, 1]) * (1.125 - 0.125 * R[:, 0] ** 2 - R[:, 1])
        c4 = (2.07 - 0.23 * R[:, 0] ** 2 - R[:, 1]) * (0.63 - 0.07 * R[:, 0] ** 2 - R[:, 1])

        invalid = (c1 < 0) | (c2 > 0) | (c3 < 0) | (c4 > 0)
        iteration += 1

    # Add the point [1, 1] to complete the front
    if len(R) > 0:
        R = np.vstack([R, [1, 1]])

        # Keep only non-dominated solutions (front_no == 1)
        front_no, _ = nd_sort(R, len(R))
        R = R[front_no == 1, :]

    return R


def MW12_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW12.

    The PF is an oscillating curve f2 = 0.85 - 0.8*f1 - 0.08*|sin(3.2*pi*f1)|,
    adjusted to satisfy the first constraint.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the PF points.
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1, N)
    R[:, 1] = 0.85 - 0.8 * R[:, 0] - 0.08 * np.abs(np.sin(3.2 * np.pi * R[:, 0]))

    # Calculate constraint c1
    c1 = (1 - 0.8 * R[:, 0] - R[:, 1] +
          0.08 * np.sin(2 * np.pi * (R[:, 1] - R[:, 0] / 1.5))) * \
         (1.8 - 1.125 * R[:, 0] - R[:, 1] +
          0.08 * np.sin(2 * np.pi * (R[:, 1] / 1.8 - R[:, 0] / 1.6)))

    # Check constraint: c1 > 0 means invalid
    invalid = c1 > 0

    # Adjust invalid points
    max_iterations = 1000
    iteration = 0
    while np.any(invalid) and iteration < max_iterations:
        R[invalid, :] = R[invalid, :] * 1.001

        # Recalculate constraint c1
        c1[invalid] = (1 - 0.8 * R[invalid, 0] - R[invalid, 1] +
                       0.08 * np.sin(2 * np.pi * (R[invalid, 1] - R[invalid, 0] / 1.5))) * \
                      (1.8 - 1.125 * R[invalid, 0] - R[invalid, 1] +
                       0.08 * np.sin(2 * np.pi * (R[invalid, 1] / 1.8 - R[invalid, 0] / 1.6)))

        invalid = c1 > 0
        iteration += 1

    return R


def MW13_PF(N: int, M: int = 2) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW13.

    The PF is a complex curve f2 = 5 - exp(f1) - 0.5*|sin(3*pi*f1)|,
    adjusted to satisfy the first constraint and filtered by non-dominated sorting.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the non-dominated PF points.
    """
    R = np.zeros((N, M))
    R[:, 0] = np.linspace(0, 1.5, N)
    R[:, 1] = 5 - np.exp(R[:, 0]) - 0.5 * np.abs(np.sin(3 * np.pi * R[:, 0]))

    # Calculate constraint c1
    c1 = (5 - np.exp(R[:, 0]) - 0.5 * np.sin(3 * np.pi * R[:, 0]) - R[:, 1]) * \
         (5 - (1 + 0.4 * R[:, 0]) - 0.5 * np.sin(3 * np.pi * R[:, 0]) - R[:, 1])

    # Check constraint: c1 > 0 means invalid
    invalid = c1 > 0

    # Adjust invalid points
    max_iterations = 1000
    iteration = 0
    while np.any(invalid) and iteration < max_iterations:
        R[invalid, :] = R[invalid, :] * 1.001

        # Recalculate constraint c1
        c1[invalid] = (5 - np.exp(R[invalid, 0]) -
                       0.5 * np.sin(3 * np.pi * R[invalid, 0]) - R[invalid, 1]) * \
                      (5 - (1 + 0.4 * R[invalid, 0]) -
                       0.5 * np.sin(3 * np.pi * R[invalid, 0]) - R[invalid, 1])

        invalid = c1 > 0
        iteration += 1

    # Keep only non-dominated solutions (front_no == 1)
    if len(R) > 0:
        front_no, _ = nd_sort(R, len(R))
        R = R[front_no == 1, :]

    return R


def MW14_PF(N: int, M: int = 3) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for MW14.

    The PF has a specific structure with disconnected regions defined
    by intervals [0, 0.731], [1.331, 1.5] for each objective dimension.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int, optional
        Number of objectives (default is 3).

    Returns
    -------
    np.ndarray
        Array of shape (N', M) representing the PF points.
    """
    # Define intervals for the disconnected PF
    interval = np.array([0.0, 0.731, 1.331, 1.5])
    median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])

    # Generate uniform grid points in [0,1]^(M-1)
    X, _ = uniform_point(N, M - 1, method='grid')

    # Map points to the disconnected intervals
    mask_low = X <= median
    mask_high = X > median

    X[mask_low] = X[mask_low] * (interval[1] - interval[0]) / median + interval[0]
    X[mask_high] = (X[mask_high] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]

    # Calculate the last objective
    # f_M = 1/(M-1) * sum(6 - exp(f_i) - 1.5*sin(1.1*pi*f_i^2))
    f_M = (1 / (M - 1)) * np.sum(6 - np.exp(X) - 1.5 * np.sin(1.1 * np.pi * X ** 2), axis=1)

    # Combine into full objective array
    R = np.column_stack([X, f_M])

    return R


SETTINGS = {
    'metric': 'IGD',
    'n_ref': 2000,
    'MW1': {'T1': MW1_PF},
    'MW2': {'T1': MW2_PF},
    'MW3': {'T1': MW3_PF},
    'MW4': {'T1': MW4_PF},
    'MW5': {'T1': MW5_PF},
    'MW6': {'T1': MW6_PF},
    'MW7': {'T1': MW7_PF},
    'MW8': {'T1': MW8_PF},
    'MW9': {'T1': MW9_PF},
    'MW10': {'T1': MW10_PF},
    'MW11': {'T1': MW11_PF},
    'MW12': {'T1': MW12_PF},
    'MW13': {'T1': MW13_PF},
    'MW14': {'T1': MW14_PF}
}
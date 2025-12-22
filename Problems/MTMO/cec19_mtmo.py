import numpy as np
from Methods.mtop import MTOP


class LZ09:
    """
    LZ09 test function framework for multi-objective optimization.

    Parameters
    ----------
    dim : int
        Dimension of decision variables.
    num_of_objective : int
        Number of objectives.
    ltype : int
        Link function type (21, 22, 23, 24, 25, 26, 32).
    dtype : int
        Distance function type (1, 2, 3, 4).
    ptype : int
        Position function type (21, 22, 23, 24 for 2-obj; 31, 32, 33, 34 for 3-obj).
    """

    def __init__(self, dim, num_of_objective, ltype, dtype, ptype):
        self.dim = dim
        self.num_of_objective = num_of_objective
        self.ltype = ltype
        self.dtype = dtype
        self.ptype = ptype

    def psfunc(self, x, x1, css):
        """
        Position-related link function for 2-objective problems.

        Parameters
        ----------
        x : np.ndarray
            Subset of decision variables (odd or even indexed).
        x1 : np.ndarray
            First decision variable.
        css : int
            Class of index (1 or 2).

        Returns
        -------
        beta : np.ndarray
            Transformed variables.
        """
        if self.ltype == 21:
            if x.shape[1] % 2 == 0:
                qq = np.arange(3, self.dim + 1, 2)
            else:
                qq = np.arange(2, self.dim + 1, 2)
            x = 2.0 * (x - 0.5)
            beta = x - x1[:, np.newaxis] ** (0.5 * (self.dim + 3.0 * qq - 8) / (self.dim - 2))

        elif self.ltype == 22:
            if x.shape[1] % 2 == 0:
                qq = np.arange(3, self.dim + 1, 2)
            else:
                qq = np.arange(2, self.dim + 1, 2)
            x = 2.0 * (x - 0.5)
            theta = np.sin(6 * np.pi * x1[:, np.newaxis] + (np.pi * qq) / self.dim)
            beta = x - theta

        elif self.ltype == 23:
            if x.shape[1] % 2 == 0:
                qq = np.arange(3, self.dim + 1, 2)
            else:
                qq = np.arange(2, self.dim + 1, 2)
            theta = 6 * np.pi * x1[:, np.newaxis] + (np.pi * qq) / self.dim
            ra = 0.8 * x1[:, np.newaxis]
            x = 2.0 * (x - 0.5)
            if css == 1:
                beta = x - ra * np.cos(theta)
            else:
                beta = x - ra * np.sin(theta)

        elif self.ltype == 24:
            if x.shape[1] % 2 == 0:
                qq = np.arange(3, self.dim + 1, 2)
            else:
                qq = np.arange(2, self.dim + 1, 2)
            theta = 6 * np.pi * x1[:, np.newaxis] + (np.pi * qq) / self.dim
            ra = 0.8 * x1[:, np.newaxis]
            x = 2.0 * (x - 0.5)
            if css == 1:
                beta = x - ra * np.cos(theta / 3)
            else:
                beta = x - ra * np.sin(theta)

        elif self.ltype == 25:
            rho = 0.8
            phi = np.pi * x1
            if x.shape[1] % 2 == 0:
                qq = np.arange(3, self.dim + 1, 2)
            else:
                qq = np.arange(2, self.dim + 1, 2)
            theta = 6 * np.pi * x1[:, np.newaxis] + (np.pi * qq) / self.dim
            x = 2.0 * (x - 0.5)
            if css == 1:
                beta = x - rho * np.sin(phi)[:, np.newaxis] * np.sin(theta)
            elif css == 2:
                beta = x - rho * np.sin(phi)[:, np.newaxis] * np.cos(theta)
            else:
                beta = x - rho * np.cos(phi)[:, np.newaxis]

        elif self.ltype == 26:
            if x.shape[1] % 2 == 0:
                qq = np.arange(3, self.dim + 1, 2)
            else:
                qq = np.arange(2, self.dim + 1, 2)
            theta = 6 * np.pi * x1[:, np.newaxis] + (np.pi * qq) / self.dim
            ra = 0.3 * x1[:, np.newaxis] * (x1[:, np.newaxis] * np.cos(4 * theta) + 2)
            x = 2.0 * (x - 0.5)
            if css == 1:
                beta = x - ra * np.cos(theta)
            else:
                beta = x - ra * np.sin(theta)

        return beta

    def psfunc3(self, x, x1, x2, order):
        """
        Position-related link function for 3-objective problems.

        Parameters
        ----------
        x : np.ndarray
            Subset of decision variables (J1, J2, or J3).
        x1 : np.ndarray
            First decision variable.
        x2 : np.ndarray
            Second decision variable.
        order : np.ndarray
            Indices of x in original decision variables.

        Returns
        -------
        beta : np.ndarray
            Transformed variables.
        """
        if self.ltype == 32:
            theta = 2 * np.pi * x1[:, np.newaxis] + np.pi * order / self.dim
            x = 4.0 * (x - 0.5)
            beta = x - 2 * x2[:, np.newaxis] * np.sin(theta)
        return beta

    def alpha_function(self, x):
        """
        Shape function (PF shape).

        Parameters
        ----------
        x : np.ndarray
            Decision variables.

        Returns
        -------
        alpha : np.ndarray
            Shape function values for each objective.
        """
        alpha = np.zeros((x.shape[0], self.num_of_objective))

        if self.num_of_objective == 2:
            if self.ptype == 21:
                alpha[:, 0] = x[:, 0]
                alpha[:, 1] = 1 - np.sqrt(x[:, 0])
            elif self.ptype == 22:
                alpha[:, 0] = x[:, 0]
                alpha[:, 1] = 1 - x[:, 0] ** 2
            elif self.ptype == 23:
                alpha[:, 0] = x[:, 0]
                alpha[:, 1] = 1 - np.sqrt(x[:, 0]) - x[:, 0] * np.sin(10 * x[:, 0] ** 2 * np.pi)
            elif self.ptype == 24:
                alpha[:, 0] = x[:, 0]
                alpha[:, 1] = 1 - x[:, 0] - 0.05 * np.sin(4 * np.pi * x[:, 0])
        else:  # 3 objectives
            if self.ptype == 31:
                alpha[:, 0] = np.cos(x[:, 0] * np.pi / 2) * np.cos(x[:, 1] * np.pi / 2)
                alpha[:, 1] = np.cos(x[:, 0] * np.pi / 2) * np.sin(x[:, 1] * np.pi / 2)
                alpha[:, 2] = np.sin(x[:, 0] * np.pi / 2)
            elif self.ptype == 32:
                alpha[:, 0] = 1 - np.cos(x[:, 0] * np.pi / 2) * np.cos(x[:, 1] * np.pi / 2)
                alpha[:, 1] = 1 - np.cos(x[:, 0] * np.pi / 2) * np.sin(x[:, 1] * np.pi / 2)
                alpha[:, 2] = 1 - np.sin(x[:, 0] * np.pi / 2)
            elif self.ptype == 33:
                alpha[:, 0] = x[:, 0]
                alpha[:, 1] = x[:, 1]
                alpha[:, 2] = 3 - (np.sin(3 * np.pi * x[:, 0]) + np.sin(3 * np.pi * x[:, 1])) - 2 * (x[:, 0] + x[:, 1])
            elif self.ptype == 34:
                alpha[:, 0] = x[:, 0] * x[:, 1]
                alpha[:, 1] = x[:, 0] * (1 - x[:, 1])
                alpha[:, 2] = 1 - x[:, 0]

        return alpha

    def beta_function(self, odd_even_x):
        """
        Distance function.

        Parameters
        ----------
        odd_even_x : np.ndarray
            Subset of decision variables (odd or even indexed).

        Returns
        -------
        beta : np.ndarray
            Distance function values.
        """
        dim = odd_even_x.shape[1]

        if self.dtype == 1:
            beta = np.sum(odd_even_x ** 2, axis=1)
            beta = 2 * beta / dim
        elif self.dtype == 2:
            a = np.sqrt(np.arange(1, dim + 1))
            beta = np.sum(odd_even_x ** 2 * a, axis=1)
            beta = 2 * beta / dim
        elif self.dtype == 3:
            beta = np.sum((2 * odd_even_x) ** 2, axis=1) - np.sum(np.cos(4 * np.pi * odd_even_x), axis=1)
            beta = beta + dim
            beta = 2 * beta / dim
        elif self.dtype == 4:
            sum1 = np.sum((2 * odd_even_x) ** 2, axis=1)
            a = np.sqrt(np.arange(1, dim + 1))
            cos_terms = np.cos(10 * np.pi * (2 * odd_even_x) / a)
            # MATLAB bug: 按列累乘，取矩阵最后一个元素（标量）
            prod_cumsum = np.cumprod(cos_terms, axis=0)  # 按列累乘
            prod_value = prod_cumsum[-1, -1]  # 取最后一个元素（标量）
            beta = 2 * (sum1 - 2 * prod_value + 2) / dim  # 所有样本用同一个prod值

        return beta

    def objective_function(self, x):
        """
        Evaluate objective functions.

        Parameters
        ----------
        x : np.ndarray
            Decision variables, shape (n_samples, dim).

        Returns
        -------
        fitness : np.ndarray
            Objective values, shape (n_samples, num_of_objective).
        """
        x = np.atleast_2d(x)
        ltype_table = [21, 22, 23, 24, 26]

        if self.num_of_objective == 2:
            if self.ltype in ltype_table:
                # Extract odd indices (3, 5, 7, ...) - Python uses 0-indexing
                J1 = x[:, 2::2]  # indices 2, 4, 6, ... (corresponds to 3, 5, 7, ... in MATLAB)
                a = self.psfunc(J1, x[:, 0], 1)

                # Extract even indices (2, 4, 6, ...) - Python uses 0-indexing
                J2 = x[:, 1::2]  # indices 1, 3, 5, ... (corresponds to 2, 4, 6, ... in MATLAB)
                b = self.psfunc(J2, x[:, 0], 2)

                g = self.beta_function(a)
                h = self.beta_function(b)
                alpha = self.alpha_function(x)

                fitness_1 = alpha[:, 0] + h
                fitness_2 = alpha[:, 1] + g
                fitness = np.column_stack([fitness_1, fitness_2])
            else:  # ltype == 25
                # Split into 3 groups
                J1 = x[:, 3::3]  # indices 3, 6, 9, ... (corresponds to 4, 7, 10, ... in MATLAB)
                a = self.psfunc(J1, x[:, 0], 1)

                J2 = x[:, 1::3]  # indices 1, 4, 7, ... (corresponds to 2, 5, 8, ... in MATLAB)
                b = self.psfunc(J2, x[:, 0], 2)

                J3 = x[:, 2::3]  # indices 2, 5, 8, ... (corresponds to 3, 6, 9, ... in MATLAB)
                c = self.psfunc(J3, x[:, 0], 3)

                # Combine: a with odd indices of c, b with even indices of c
                a = np.column_stack([a, c[:, 0::2]])
                b = np.column_stack([b, c[:, 1::2]])

                g = self.beta_function(a)
                h = self.beta_function(b)
                alpha = self.alpha_function(x)

                fitness_1 = alpha[:, 0] + h
                fitness_2 = alpha[:, 1] + g
                fitness = np.column_stack([fitness_1, fitness_2])
        else:  # 3 objectives
            J1 = x[:, 3::3]  # indices 3, 6, 9, ...
            J2 = x[:, 4::3]  # indices 4, 7, 10, ...
            J3 = x[:, 2::3]  # indices 2, 5, 8, ...

            order1 = np.arange(4, x.shape[1] + 1, 3)  # 4, 7, 10, ... (1-indexed for algorithm)
            order2 = np.arange(5, x.shape[1] + 1, 3)  # 5, 8, 11, ...
            order3 = np.arange(3, x.shape[1] + 1, 3)  # 3, 6, 9, ...

            a = self.psfunc3(J1, x[:, 0], x[:, 1], order1)
            b = self.psfunc3(J2, x[:, 0], x[:, 1], order2)
            c = self.psfunc3(J3, x[:, 0], x[:, 1], order3)

            g = self.beta_function(a)
            h = self.beta_function(b)
            e = self.beta_function(c)
            alpha = self.alpha_function(x)

            fitness_1 = alpha[:, 0] + g
            fitness_2 = alpha[:, 1] + h
            fitness_3 = alpha[:, 2] + e
            fitness = np.column_stack([fitness_1, fitness_2, fitness_3])

        return fitness


class CEC19MTMO:
    """
    Implementation of the CEC 2019 Competition on Evolutionary Multi-Task Multi-Objective
    Optimization (MTMO) benchmark problems.

    These problems are based on the LZ09 test suite and consist of multiple tasks
    with different configurations designed to test knowledge transfer in multi-objective
    optimization scenarios.

    Parameters
    ----------
    None

    Attributes
    ----------
    None
    """

    def __init__(self):
        pass

    def P1(self) -> MTOP:
        """
        Generates Problem P1 (CPLX1): **T1 (LZ09_F1) vs T2 (LZ09_F2)**.

        Both tasks are 2-objective, 10-dimensional.

        - T1: LZ09_F1 with ptype=21, dtype=1, ltype=21
        - T2: LZ09_F2 with ptype=21, dtype=1, ltype=22
        - Relationship: Same PF shape (ptype=21), different link functions

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 10
        num_obj = 2

        # Task 1: LZ09_F1
        lz09_f1 = LZ09(dim=dim, num_of_objective=num_obj, ltype=21, dtype=1, ptype=21)

        def T1(x):
            return lz09_f1.objective_function(x)

        # Task 2: LZ09_F2
        lz09_f2 = LZ09(dim=dim, num_of_objective=num_obj, ltype=22, dtype=1, ptype=21)

        def T2(x):
            return lz09_f2.objective_function(x)

        lb = -100.0 * np.ones(dim)
        ub = 100.0 * np.ones(dim)
        lb[0] = 0.0
        ub[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self) -> MTOP:
        """
        Generates Problem P2 (CPLX2): **T1 (LZ09_F1) vs T2 (LZ09_F7)**.

        Both tasks are 2-objective, 10-dimensional.

        - T1: LZ09_F1 with ptype=21, dtype=1, ltype=21
        - T2: LZ09_F7 with ptype=21, dtype=3, ltype=21  # 修正：dtype=3
        - Relationship: Same PF shape (ptype=21), different distance functions

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 10
        num_obj = 2

        # Task 1: LZ09_F1
        lz09_f1 = LZ09(dim=dim, num_of_objective=num_obj, ltype=21, dtype=1, ptype=21)

        def T1(x):
            return lz09_f1.objective_function(x)

        # Task 2: LZ09_F7
        lz09_f7 = LZ09(dim=dim, num_of_objective=num_obj, ltype=21, dtype=3, ptype=21)  # 修正

        def T2(x):
            return lz09_f7.objective_function(x)

        lb = -5.0 * np.ones(dim)
        ub = 5.0 * np.ones(dim)
        lb[0] = 0.0
        ub[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self) -> MTOP:
        """
        Generates Problem P3 (CPLX3): **T1 (LZ09_F2) vs T2 (LZ09_F4)**.

        Both tasks are 2-objective, 30-dimensional.

        - T1: LZ09_F2 with ptype=21, dtype=1, ltype=22
        - T2: LZ09_F4 with ptype=21, dtype=1, ltype=24
        - Relationship: Same PF shape (ptype=21), different link functions and search spaces

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        num_obj = 2

        # Task 1: LZ09_F2, 30-dimensional
        dim1 = 30
        lz09_f2 = LZ09(dim=dim1, num_of_objective=num_obj, ltype=22, dtype=1, ptype=21)

        def T1(x):
            return lz09_f2.objective_function(x)

        lb1 = -2.0 * np.ones(dim1)
        ub1 = 2.0 * np.ones(dim1)
        lb1[0] = 0.0
        ub1[0] = 1.0

        # Task 2: LZ09_F4, 30-dimensional
        dim2 = 30
        lz09_f4 = LZ09(dim=dim2, num_of_objective=num_obj, ltype=24, dtype=1, ptype=21)

        def T2(x):
            return lz09_f4.objective_function(x)

        lb2 = -1.0 * np.ones(dim2)
        ub2 = 1.0 * np.ones(dim2)
        lb2[0] = 0.0
        ub2[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim1, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim2, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P4(self) -> MTOP:
        """
        Generates Problem P4 (CPLX4): **T1 (LZ09_F2) vs T2 (LZ09_F9)**.

        Both tasks are 2-objective, 30-dimensional.

        - T1: LZ09_F2 with ptype=21, dtype=1, ltype=22
        - T2: LZ09_F9 with ptype=22, dtype=1, ltype=22
        - Relationship: Different PF shapes (ptype=21 vs ptype=22), same link function

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 30
        num_obj = 2

        # Task 1: LZ09_F2
        lz09_f2 = LZ09(dim=dim, num_of_objective=num_obj, ltype=22, dtype=1, ptype=21)

        def T1(x):
            return lz09_f2.objective_function(x)

        # Task 2: LZ09_F9
        lz09_f9 = LZ09(dim=dim, num_of_objective=num_obj, ltype=22, dtype=1, ptype=22)

        def T2(x):
            return lz09_f9.objective_function(x)

        lb = -100.0 * np.ones(dim)
        ub = 100.0 * np.ones(dim)
        lb[0] = 0.0
        ub[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self) -> MTOP:
        """
        Generates Problem P5 (CPLX5): **T1 (LZ09_F3, 2-obj) vs T2 (LZ09_F6, 3-obj)**.

        Tasks have different objectives and dimensions.

        - T1: LZ09_F3, 2-objective, 30-dimensional, ptype=21, dtype=1, ltype=23
        - T2: LZ09_F6, 3-objective, 10-dimensional, ptype=31, dtype=1, ltype=32
        - Relationship: Different number of objectives and dimensions

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        # Task 1: LZ09_F3, 2-objective, 30-dimensional
        dim1 = 30
        num_obj1 = 2
        lz09_f3 = LZ09(dim=dim1, num_of_objective=num_obj1, ltype=23, dtype=1, ptype=21)

        def T1(x):
            return lz09_f3.objective_function(x)

        lb1 = np.zeros(dim1)
        ub1 = np.ones(dim1)

        # Task 2: LZ09_F6, 3-objective, 10-dimensional
        dim2 = 10
        num_obj2 = 3
        lz09_f6 = LZ09(dim=dim2, num_of_objective=num_obj2, ltype=32, dtype=1, ptype=31)

        def T2(x):
            return lz09_f6.objective_function(x)

        lb2 = np.zeros(dim2)
        ub2 = np.ones(dim2)

        problem = MTOP()
        problem.add_task(T1, dim=dim1, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim2, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P6(self) -> MTOP:
        """
        Generates Problem P6 (CPLX6): **T1 (LZ09_F3) vs T2 (LZ09_F9)**.

        Both tasks are 2-objective, 30-dimensional.

        - T1: LZ09_F3 with ptype=21, dtype=1, ltype=23
        - T2: LZ09_F9 with ptype=22, dtype=1, ltype=22
        - Relationship: Different PF shapes and search spaces

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 30
        num_obj = 2

        # Task 1: LZ09_F3
        lz09_f3 = LZ09(dim=dim, num_of_objective=num_obj, ltype=23, dtype=1, ptype=21)

        def T1(x):
            return lz09_f3.objective_function(x)

        lb1 = -50.0 * np.ones(dim)
        ub1 = 50.0 * np.ones(dim)
        lb1[0] = 0.0
        ub1[0] = 1.0

        # Task 2: LZ09_F9
        lz09_f9 = LZ09(dim=dim, num_of_objective=num_obj, ltype=22, dtype=1, ptype=22)

        def T2(x):
            return lz09_f9.objective_function(x)

        lb2 = -100.0 * np.ones(dim)
        ub2 = 100.0 * np.ones(dim)
        lb2[0] = 0.0
        ub2[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P7(self) -> MTOP:
        """
        Generates Problem P7 (CPLX7): **T1 (LZ09_F4) vs T2 (LZ09_F5)**.

        Both tasks are 2-objective, 30-dimensional.

        - T1: LZ09_F4 with ptype=21, dtype=1, ltype=24
        - T2: LZ09_F5 with ptype=21, dtype=1, ltype=26
        - Relationship: Same PF shape (ptype=21), different link functions

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 30
        num_obj = 2

        # Task 1: LZ09_F4
        lz09_f4 = LZ09(dim=dim, num_of_objective=num_obj, ltype=24, dtype=1, ptype=21)

        def T1(x):
            return lz09_f4.objective_function(x)

        # Task 2: LZ09_F5
        lz09_f5 = LZ09(dim=dim, num_of_objective=num_obj, ltype=26, dtype=1, ptype=21)

        def T2(x):
            return lz09_f5.objective_function(x)

        lb = -80.0 * np.ones(dim)
        ub = 80.0 * np.ones(dim)
        lb[0] = 0.0
        ub[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self) -> MTOP:
        """
        Generates Problem P8 (CPLX8): **T1 (LZ09_F5) vs T2 (LZ09_F7)**.

        Both tasks are 2-objective with different dimensions.

        - T1: LZ09_F5, 30-dimensional, ptype=21, dtype=1, ltype=26
        - T2: LZ09_F7, 10-dimensional, ptype=21, dtype=3, ltype=21
        - Relationship: Same PF shape (ptype=21), different dimensions and link functions
        - Note: x1 and x2 both in [0,1] for both tasks

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        num_obj = 2

        # Task 1: LZ09_F5, 30-dimensional
        dim1 = 30
        lz09_f5 = LZ09(dim=dim1, num_of_objective=num_obj, ltype=26, dtype=1, ptype=21)

        def T1(x):
            return lz09_f5.objective_function(x)

        lb1 = -20.0 * np.ones(dim1)
        ub1 = 20.0 * np.ones(dim1)
        lb1[0:2] = 0.0  # x1 and x2 in [0,1]
        ub1[0:2] = 1.0

        # Task 2: LZ09_F7, 10-dimensional
        dim2 = 10
        lz09_f7 = LZ09(dim=dim2, num_of_objective=num_obj, ltype=21, dtype=3, ptype=21)

        def T2(x):
            return lz09_f7.objective_function(x)

        lb2 = -20.0 * np.ones(dim2)
        ub2 = 20.0 * np.ones(dim2)
        lb2[0:2] = 0.0  # x1 and x2 in [0,1]
        ub2[0:2] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim1, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim2, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P9(self) -> MTOP:
        """
        Generates Problem P9 (CPLX9): **T1 (LZ09_F6, 3-obj) vs T2 (LZ09_F9, 2-obj)**.

        Tasks have different objectives and dimensions.

        - T1: LZ09_F6, 3-objective, 10-dimensional, ptype=31, dtype=1, ltype=32
        - T2: LZ09_F9, 2-objective, 30-dimensional, ptype=22, dtype=1, ltype=22
        - Relationship: Different number of objectives, dimensions, and PF shapes
        - Note: x1 and x2 both in [0,1] for both tasks

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        # Task 1: LZ09_F6, 3-objective, 10-dimensional
        dim1 = 10
        num_obj1 = 3
        lz09_f6 = LZ09(dim=dim1, num_of_objective=num_obj1, ltype=32, dtype=1, ptype=31)

        def T1(x):
            return lz09_f6.objective_function(x)

        lb1 = -50.0 * np.ones(dim1)
        ub1 = 50.0 * np.ones(dim1)
        lb1[0:2] = 0.0  # x1 and x2 in [0,1]
        ub1[0:2] = 1.0

        # Task 2: LZ09_F9, 2-objective, 30-dimensional
        dim2 = 30
        num_obj2 = 2
        lz09_f9 = LZ09(dim=dim2, num_of_objective=num_obj2, ltype=22, dtype=1, ptype=22)

        def T2(x):
            return lz09_f9.objective_function(x)

        lb2 = -100.0 * np.ones(dim2)
        ub2 = 100.0 * np.ones(dim2)
        lb2[0:2] = 0.0  # x1 and x2 in [0,1]
        ub2[0:2] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim1, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim2, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P10(self) -> MTOP:
        """
        Generates Problem P10 (CPLX10): **T1 (LZ09_F7) vs T2 (LZ09_F8)**.

        Both tasks are 2-objective, 10-dimensional.

        - T1: LZ09_F7 with ptype=21, dtype=3, ltype=21
        - T2: LZ09_F8 with ptype=21, dtype=4, ltype=21
        - Relationship: Same PF shape (ptype=21), different distance functions and search spaces

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 10
        num_obj = 2

        # Task 1: LZ09_F7
        lz09_f7 = LZ09(dim=dim, num_of_objective=num_obj, ltype=21, dtype=3, ptype=21)

        def T1(x):
            return lz09_f7.objective_function(x)

        lb1 = np.zeros(dim)
        ub1 = np.ones(dim)

        # Task 2: LZ09_F8
        lz09_f8 = LZ09(dim=dim, num_of_objective=num_obj, ltype=21, dtype=4, ptype=21)

        def T2(x):
            return lz09_f8.objective_function(x)

        lb2 = -10.0 * np.ones(dim)
        ub2 = 10.0 * np.ones(dim)
        lb2[0] = 0.0
        ub2[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem


# --- True Pareto Front (PF) Functions ---


def P1_T1_PF(N, M=2) -> np.ndarray:
    """Computes the True Pareto Front (PF) for P1, Task 1."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P1_T2_PF(N, M=2) -> np.ndarray:
    """Computes the True Pareto Front (PF) for P1, Task 2."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P2_T1_PF(N, M=2) -> np.ndarray:
    """Computes the True Pareto Front (PF) for P2, Task 1."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P2_T2_PF(N, M=2) -> np.ndarray:
    """Computes the True Pareto Front (PF) for P2, Task 2."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])

def P3_T1_PF(N, M=2) -> np.ndarray:
    """Computes the True Pareto Front (PF) for P3, Task 1."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P3_T2_PF(N, M=2) -> np.ndarray:
    """Computes the True Pareto Front (PF) for P3, Task 2."""
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P4_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P4, Task 1.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P4_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P4, Task 2.

    ptype=22: f2 = 1 - f1^2
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P5_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P5, Task 1.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P5_T2_PF(N, M=3) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P5, Task 2.

    ptype=31: Unit sphere surface in 3D
    """
    # Generate uniform points on unit sphere
    n_sqrt = int(np.sqrt(N))
    theta = np.linspace(0, np.pi / 2, n_sqrt)
    phi = np.linspace(0, np.pi / 2, n_sqrt)

    points = []
    for t in theta:
        for p in phi:
            # Spherical coordinates to Cartesian
            f1 = np.cos(t) * np.cos(p)
            f2 = np.cos(t) * np.sin(p)
            f3 = np.sin(t)
            points.append([f1, f2, f3])

    pf = np.array(points)
    # Normalize to ensure on unit sphere
    pf = pf / np.sqrt(np.sum(pf ** 2, axis=1, keepdims=True))
    return pf


def P6_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P6, Task 1.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P6_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P6, Task 2.

    ptype=22: f2 = 1 - f1^2
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P7_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P7, Task 1.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P7_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P7, Task 2.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P8_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P8, Task 1.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P8_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P8, Task 2.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P9_T1_PF(N, M=3) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P9, Task 1.

    ptype=31: Unit sphere surface in 3D
    """
    # Generate uniform points on unit sphere
    n_sqrt = int(np.sqrt(N))
    theta = np.linspace(0, np.pi / 2, n_sqrt)
    phi = np.linspace(0, np.pi / 2, n_sqrt)

    points = []
    for t in theta:
        for p in phi:
            # Spherical coordinates to Cartesian
            f1 = np.cos(t) * np.cos(p)
            f2 = np.cos(t) * np.sin(p)
            f3 = np.sin(t)
            points.append([f1, f2, f3])

    pf = np.array(points)
    # Normalize to ensure on unit sphere
    pf = pf / np.sqrt(np.sum(pf ** 2, axis=1, keepdims=True))
    return pf


def P9_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P9, Task 2.

    ptype=22: f2 = 1 - f1^2
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


def P10_T1_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P10, Task 1.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def P10_T2_PF(N, M=2) -> np.ndarray:
    """
    Computes the True Pareto Front (PF) for P10, Task 2.

    ptype=21: f2 = 1 - sqrt(f1)
    """
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])

SETTINGS = {
    'metric': 'IGD',
    'n_pf': 10000,
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
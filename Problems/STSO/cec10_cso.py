import numpy as np
import scipy.io
from Methods.mtop import MTOP


class CEC10_CSO:
    """
    CEC 2010 Competition on Constrained Real-Parameter Optimization (CSO) benchmark problems.

    This class provides constrained single-objective optimization benchmark functions
    configured as Multi-Task Optimization Problems (MTOPs) with only one task.

    Reference:
    ----------
    Mallipeddi, Rammohan and Suganthan, Ponnuthurai.
    "Problem Definitions and Evaluation Criteria for the CEC 2010 Competition
    on Constrained Real-parameter Optimization." (2010)

    Attributes
    ----------
    delta : float
        Tolerance for equality constraints (default: 1e-4).
    """

    def __init__(self, mat_dir='../Problems/STSO/data_cec10cso'):
        self.delta = 1e-4
        self.mat_dir = mat_dir

    def P1(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 1.

        This is a constrained optimization problem with:
        - 1 objective function
        - 2 inequality constraints
        - Search space: [0, 10] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 1.
        """
        delta = self.delta

        # Offset vector for P1
        O = np.array([
            0.030858718087483, -0.078632292353156, 0.048651146638038,
            -0.069089831066354, -0.087918542941928, 0.088982639811141,
            0.074143235639847, -0.086527593580149, -0.020616531903907,
            0.055586106499231, 0.059285954883598, -0.040671485554685,
            -0.087399911887693, -0.01842585125741, -0.005184912793062,
            -0.039892037937026, 0.036509229387458, 0.026046414854433,
            -0.067133862936029, 0.082780189144943, -0.049336722577062,
            0.018503188080959, 0.051610619131255, 0.018613117768432,
            0.093448598181657, -0.071208840780873, -0.036535677894572,
            -0.03126128526933, 0.099243805247963, 0.053872445945574
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 1"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            p = x - np.tile(o, (ps, 1))

            # Objective function
            numerator = np.abs(
                np.sum(np.cos(p) ** 4, axis=1) -
                2 * np.prod(np.cos(p) ** 2, axis=1)
            )
            denominator = np.sqrt(
                np.sum(np.arange(1, dim + 1) * (p ** 2), axis=1)
            )

            f = -numerator / denominator

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 1"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            p = x - np.tile(o, (ps, 1))

            # Inequality constraints (g <= 0)
            g1 = 0.75 - np.prod(p, axis=1)
            g2 = np.sum(p, axis=1) - 7.5 * dim

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            g2 = np.where(g2 < 0, 0, g2)

            # Equality constraint (initialized as zeros, matching MATLAB behavior)
            h = np.zeros(ps)

            # Combine constraints [g, h]
            cons = np.column_stack([g1, g2, h])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.zeros(dim)
        ub = np.full(dim, 10.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P2(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 2.

        This is a constrained optimization problem with:
        - 1 objective function (max function)
        - 2 inequality constraints
        - 1 equality constraint
        - Search space: [-5.12, 5.12] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 2.
        """
        delta = self.delta

        # Offset vector for P2
        O = np.array([
            -0.066939099286697, 0.470966419894494, -0.490528349401176,
            -0.312203454689423, -0.124759576300523, -0.247823908806285,
            -0.448077079941866, 0.326494954650117, 0.493435908752668,
            0.061699778818925, -0.30251101183711, -0.274045146932175,
            -0.432969960330318, 0.062239193145781, -0.188163731545079,
            -0.100709842052095, -0.333528971180922, -0.496627672944882,
            -0.288650116941944, 0.435648113198148, -0.348261107144255,
            0.456550427329479, -0.286843419772511, 0.145639015401174,
            -0.038656025783381, 0.333291935226012, -0.293687524888766,
            -0.347859473554797, -0.089300971656411, 0.142027393193559
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 2"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Objective function: max function
            f = np.max(z, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 2"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))
            y = z - 0.5

            # Inequality constraints (g <= 0)
            rastrigin_z = z ** 2 - 10 * np.cos(2 * np.pi * z) + 10
            g1 = 10 - (1 / dim) * np.sum(rastrigin_z, axis=1)
            g2 = (1 / dim) * np.sum(rastrigin_z, axis=1) - 15

            # Equality constraint (|h| <= delta)
            rastrigin_y = y ** 2 - 10 * np.cos(2 * np.pi * y) + 10
            h1 = np.abs((1 / dim) * np.sum(rastrigin_y, axis=1) - 20) - delta

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            g2 = np.where(g2 < 0, 0, g2)
            h1 = np.where(h1 < 0, 0, h1)

            # Combine constraints
            g = np.column_stack([g1, g2, h1])

            # Handle NaN values
            g[np.isnan(g)] = np.inf

            return g

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -5.12)
        ub = np.full(dim, 5.12)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P3(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 3.

        This is a constrained optimization problem with:
        - 1 objective function (Rosenbrock)
        - 1 equality constraint
        - Search space: [-1000, 1000] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 3.
        """
        delta = self.delta

        # Offset vector for P3
        O = np.array([
            111.17633500088529, 92.07880492633424, 417.9818592609036,
            253.16188128024302, 363.5279986597767, 314.334093889305,
            187.32739056163342, 240.4363027535162, 422.60090880560665,
            327.63042902581515, 62.04762897064405, 25.435663968682125,
            360.56773191905114, 154.9226721156832, 33.161292034425806,
            177.8091733067186, 262.58198940407755, 436.9800562237075,
            476.6400624069227, 331.2167787340325, 75.205948242522,
            484.33624811710115, 258.4696246506982, 419.8919566566751,
            357.51468895930395, 166.3771729386268, 47.59455935830133,
            188.20606700809785, 184.7964918401363, 267.9201349178807
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 3 (Rosenbrock)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Rosenbrock function
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]

            f = np.sum(100 * (z_D_1 ** 2 - z_2_D) ** 2 + (z_D_1 - 1) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 3"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Inequality constraint (initialized as zeros, matching MATLAB)
            g = np.zeros(ps)

            # Equality constraint (|h| <= delta)
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]
            h = np.abs(np.sum((z_D_1 - z_2_D) ** 2, axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            h = np.where(h < 0, 0, h)

            # Combine constraints [g, h]
            cons = np.column_stack([g, h])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -1000.0)
        ub = np.full(dim, 1000.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P4(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 4.

        This is a constrained optimization problem with:
        - 1 objective function (max function)
        - 4 equality constraints
        - Search space: [-50, 50] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 4.
        """
        delta = self.delta

        # Offset vector for P4
        O = np.array([
            0.820202353727904, 5.260154140335203, -1.694610371739177,
            -5.589298730330406, -0.141736605495543, 9.454675508078164,
            8.795744608532939, 9.687346331423548, -3.246522827444976,
            6.647399971577617, 1.434490229836026, -0.506531215086801,
            0.558594225280784, 7.919942423520642, 1.383716002673571,
            -1.520153615528276, -2.266737465474915, 6.48052999726508,
            -8.893207968949003, -3.528743044935322, 6.063486037065154,
            -4.51585211274229, 7.320477892009357, -8.990263774675665,
            9.446412007392851, -6.41068985463494, -9.135251626491991,
            2.07763837492787, 8.051026378030816, -1.002691032064544
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 4 (max function)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Objective function: max function
            f = np.max(z, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 4"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Inequality constraint (initialized as zeros, matching MATLAB)
            g = np.zeros(ps)

            # Equality constraints (|h| <= delta)
            h1 = np.abs((1 / dim) * np.sum(z * np.cos(np.sqrt(np.abs(z))), axis=1)) - delta

            # Split z into two halves for h2 and h3
            half = dim // 2
            z_first_half = z[:, :half - 1]
            z_second_half = z[:, 1:half]
            h2 = np.abs(np.sum((z_first_half - z_second_half) ** 2, axis=1)) - delta

            z_third_part = z[:, half:-1]
            z_fourth_part = z[:, half + 1:]
            h3 = np.abs(np.sum((z_third_part ** 2 - z_fourth_part) ** 2, axis=1)) - delta

            h4 = np.abs(np.sum(z, axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            h1 = np.where(h1 < 0, 0, h1)
            h2 = np.where(h2 < 0, 0, h2)
            h3 = np.where(h3 < 0, 0, h3)
            h4 = np.where(h4 < 0, 0, h4)

            # Combine constraints [g, h1, h2, h3, h4]
            cons = np.column_stack([g, h1, h2, h3, h4])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -50.0)
        ub = np.full(dim, 50.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P5(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 5.

        This is a constrained optimization problem with:
        - 1 objective function (max function)
        - 2 equality constraints
        - Search space: [-600, 600] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 5.
        """
        delta = self.delta

        # Offset vector for P5
        O = np.array([
            72.10900225247575, 9.007673762322495, 51.86632637302316,
            41.365704820161, 93.18768763916974, 74.53341902482204,
            63.745479932407655, 7.496986033468282, 56.16729598807964,
            17.71630810614085, 28.009655663065143, 29.36357615570272,
            26.966653374740996, 6.892189514516317, 44.29071160734624,
            84.35803966449319, 81.16906730972529, 92.76919270133271,
            3.826058034047476, 7.231864548985054, 14.446069444832405,
            46.49943418775763, 22.155722253817412, 69.11723738661682,
            88.99628570349459, 58.74823912291344, 52.265369214509846,
            47.030120955005074, 53.23321779503931, 5.778976086909701
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 5 (max function)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Objective function: max function
            f = np.max(z, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 5"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Inequality constraint (initialized as zeros, matching MATLAB)
            g = np.zeros(ps)

            # Equality constraints (|h| <= delta)
            h1 = np.abs((1 / dim) * np.sum(-z * np.sin(np.sqrt(np.abs(z))), axis=1)) - delta
            h2 = np.abs((1 / dim) * np.sum(-z * np.cos(0.5 * np.sqrt(np.abs(z))), axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            h1 = np.where(h1 < 0, 0, h1)
            h2 = np.where(h2 < 0, 0, h2)

            # Combine constraints [g, h1, h2]
            cons = np.column_stack([g, h1, h2])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -600.0)
        ub = np.full(dim, 600.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P6(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 6.

        This is a constrained optimization problem with:
        - 1 objective function (max function)
        - 2 equality constraints
        - Uses rotation matrix M
        - Search space: [-600, 600] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 6.
        """

        delta = self.delta

        # Offset vector for P6
        O = np.array([
            -1.595515627742907, 7.633467047559741, -5.764100483472472,
            8.103197246263033, -0.059515969417191, -0.737189363693078,
            -9.190862358160823, 4.22087353933443, -1.745435308213725,
            9.499044614342985, 3.82068618551277, 2.569334886907409,
            9.354368119489862, -0.852114934846258, 4.714177466874696,
            6.775420647884232, -9.074204717422479, -3.760650327490145,
            -0.77805530989772, -7.487007842931314, 4.435061566086135,
            -6.952711886757461, -8.752326993212105, -2.411334215593357,
            -6.149894283287328, 1.049303005795593, -6.049093253116644,
            0.950328133373404, 1.443084229017085, -0.163829799788475
        ])
        o = O[:dim]

        # Load rotation matrix from MAT file
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/P6_M.mat')
        M1 = mat_data['M1']
        M2 = mat_data['M2']

        # Select appropriate rotation matrix
        if dim == 10:
            M = M1
        else:
            M = M2[:dim, :dim]

        def Task(x):
            """Objective function for Problem 6 (max function)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Objective function: max function
            f = np.max(z, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 6"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Rotate: MATLAB uses (z + offset) * M
            # In NumPy, this is (z + offset) @ M (NO transpose!)
            y = (z + 483.6106156535) @ M - 483.6106156535

            # Inequality constraint (initialized as zeros, matching MATLAB)
            g = np.zeros(ps)

            # Equality constraints (|h| <= delta)
            h1 = np.abs((1 / dim) * np.sum(-y * np.sin(np.sqrt(np.abs(y))), axis=1)) - delta
            h2 = np.abs((1 / dim) * np.sum(-y * np.cos(0.5 * np.sqrt(np.abs(y))), axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            h1 = np.where(h1 < 0, 0, h1)
            h2 = np.where(h2 < 0, 0, h2)

            # Combine constraints [g, h1, h2]
            cons = np.column_stack([g, h1, h2])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -600.0)
        ub = np.full(dim, 600.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P7(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 7.

        This is a constrained optimization problem with:
        - 1 objective function (Rosenbrock)
        - 1 inequality constraint
        - Search space: [-140, 140] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 7.
        """
        delta = self.delta

        # Offset vector for P7
        O = np.array([
            -1.46823271282738, 47.51401860909492, -30.421056514069576,
            -7.707941671844303, -21.74698421666629, -17.88116387879569,
            5.274442455807971, 18.71403753778708, -36.959734507345146,
            -20.72950462154263, 25.4701966548936, -25.439992885801573,
            1.054563129830697, -31.556579857545657, -19.320382777005047,
            17.16774285348282, 34.66536814401755, -31.803705714749462,
            -12.926898387712775, 25.489686517508602, -45.23000430753644,
            36.31774710581284, -18.38690515559357, 34.86816378160691,
            -37.530671214167334, 19.288852618585977, 0.684612418754519,
            -12.636795982748637, 15.005454148879409, -40.468678588994315
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 7 (Rosenbrock)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1))
            z = y + 1

            # Rosenbrock function
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]

            f = np.sum(100 * (z_D_1 ** 2 - z_2_D) ** 2 + (z_D_1 - 1) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 7"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1))

            # Inequality constraint (g <= 0)
            g1 = 0.5 - np.exp(-0.1 * np.sqrt((1 / dim) * np.sum(y ** 2, axis=1))) - \
                 3 * np.exp((1 / dim) * np.sum(np.cos(0.1 * y), axis=1)) + np.exp(1)

            # Equality constraint (initialized as zeros, matching MATLAB)
            h = np.zeros(ps)

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)

            # Combine constraints [g, h]
            cons = np.column_stack([g1, h])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -140.0)
        ub = np.full(dim, 140.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P8(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 8.

        This is a constrained optimization problem with:
        - 1 objective function (Rosenbrock)
        - 1 inequality constraint
        - Uses rotation matrix M
        - Search space: [-140, 140] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 8.
        """
        import scipy.io
        import os
        delta = self.delta

        # Offset vector for P8
        O = np.array([
            -1.46823271282738, 47.51401860909492, -30.421056514069576,
            -7.707941671844303, -21.74698421666629, -17.88116387879569,
            5.274442455807971, 18.71403753778708, -36.959734507345146,
            -20.72950462154263, 25.4701966548936, -25.439992885801573,
            1.054563129830697, -31.556579857545657, -19.320382777005047,
            17.16774285348282, 34.66536814401755, -31.803705714749462,
            -12.926898387712775, 25.489686517508602, -45.23000430753644,
            36.31774710581284, -18.38690515559357, 34.86816378160691,
            -37.530671214167334, 19.288852618585977, 0.684612418754519,
            -12.636795982748637, 15.005454148879409, -40.468678588994315
        ])
        o = O[:dim]

        # Load rotation matrix from MAT file
        mat_path = os.path.join(self.mat_dir, 'P8_M.mat')
        mat_data = scipy.io.loadmat(mat_path)
        M1 = mat_data['M1']
        M2 = mat_data['M2']

        # Select appropriate rotation matrix
        if dim == 10:
            M = M1
        else:
            M = M2[:dim, :dim]

        def Task(x):
            """Objective function for Problem 8 (Rosenbrock)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1)) + 1

            # Rotate
            y = (z - 1) @ M

            # Rosenbrock function
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]

            f = np.sum(100 * (z_D_1 ** 2 - z_2_D) ** 2 + (z_D_1 - 1) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 8"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift and rotate
            z = x - np.tile(o, (ps, 1)) + 1
            y = (z - 1) @ M

            # Inequality constraint (g <= 0)
            g1 = 0.5 - np.exp(-0.1 * np.sqrt((1 / dim) * np.sum(y ** 2, axis=1))) - \
                 3 * np.exp((1 / dim) * np.sum(np.cos(0.1 * y), axis=1)) + np.exp(1)

            # Equality constraint (initialized as zeros, matching MATLAB)
            h = np.zeros(ps)

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)

            # Combine constraints [g, h]
            cons = np.column_stack([g1, h])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -140.0)
        ub = np.full(dim, 140.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P9(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 9.

        This is a constrained optimization problem with:
        - 1 objective function (Rosenbrock)
        - 1 equality constraint
        - Search space: [-500, 500] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 9.
        """
        delta = self.delta

        # Offset vector for P9
        O = np.array([
            -41.03250252873486, -35.70280591875908, -48.66938576680659,
            94.51946988004894, 31.68700466174738, 99.69508270219342,
            30.778279925351967, -31.041222172110807, -46.21010370947247,
            27.26190010072706, -2.093622677920422, 22.246274570582585,
            -42.887366421312436, 89.88377145577851, -6.731523713182725,
            97.86439204258224, 49.49993772881544, 23.210695390854696,
            -81.36716857155828, -20.15688556597543, 36.692155371634726,
            44.37408948075327, -15.984549833405907, -49.68391424581281,
            98.3715576810595, 0.127593155843627, 61.709914317965655,
            -84.0189999580673, -35.39565398431638, -5.143979333218638
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 9 (Rosenbrock)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1))
            z = y + 1

            # Rosenbrock function
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]

            f = np.sum(100 * (z_D_1 ** 2 - z_2_D) ** 2 + (z_D_1 - 1) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 9"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1))

            # Inequality constraint (initialized as zeros, matching MATLAB)
            g = np.zeros(ps)

            # Equality constraint (|h| <= delta)
            h1 = np.abs(np.sum(y * np.sin(np.sqrt(np.abs(y))), axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            h1 = np.where(h1 < 0, 0, h1)

            # Combine constraints [g, h]
            cons = np.column_stack([g, h1])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -500.0)
        ub = np.full(dim, 500.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P10(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 10.

        This is a constrained optimization problem with:
        - 1 objective function (Rosenbrock)
        - 1 equality constraint
        - Uses rotation matrix M
        - Search space: [-500, 500] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 10.
        """
        import scipy.io
        import os
        delta = self.delta

        # Offset vector for P10
        O = np.array([
            -41.03250252873486, -35.70280591875908, -48.66938576680659,
            94.51946988004894, 31.68700466174738, 99.69508270219342,
            30.778279925351967, -31.041222172110807, -46.21010370947247,
            27.26190010072706, -2.093622677920422, 22.246274570582585,
            -42.887366421312436, 89.88377145577851, -6.731523713182725,
            97.86439204258224, 49.49993772881544, 23.210695390854696,
            -81.36716857155828, -20.15688556597543, 36.692155371634726,
            44.37408948075327, -15.984549833405907, -49.68391424581281,
            98.3715576810595, 0.127593155843627, 61.709914317965655,
            -84.0189999580673, -35.39565398431638, -5.143979333218638
        ])
        o = O[:dim]

        # Load rotation matrix from MAT file
        mat_path = os.path.join(self.mat_dir, 'P10_M.mat')
        mat_data = scipy.io.loadmat(mat_path)
        M1 = mat_data['M1']
        M2 = mat_data['M2']

        # Select appropriate rotation matrix
        if dim == 10:
            M = M1
        else:
            M = M2[:dim, :dim]

        def Task(x):
            """Objective function for Problem 10 (Rosenbrock)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1)) + 1

            # Rotate
            y = (z - 1) @ M

            # Rosenbrock function
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]

            f = np.sum(100 * (z_D_1 ** 2 - z_2_D) ** 2 + (z_D_1 - 1) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 10"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift and rotate
            z = x - np.tile(o, (ps, 1)) + 1
            y = (z - 1) @ M

            # Inequality constraint (initialized as zeros, matching MATLAB)
            g = np.zeros(ps)

            # Equality constraint (|h| <= delta)
            h1 = np.abs(np.sum(y * np.sin(np.sqrt(np.abs(y))), axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            h1 = np.where(h1 < 0, 0, h1)

            # Combine constraints [g, h]
            cons = np.column_stack([g, h1])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -500.0)
        ub = np.full(dim, 500.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P11(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 11.

        This is a constrained optimization problem with:
        - 1 objective function (modified cosine function)
        - 1 equality constraint (Rosenbrock)
        - Uses rotation matrix M
        - Search space: [-100, 100] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 11.
        """
        import scipy.io
        import os
        delta = self.delta

        # Offset vector for P11
        O = np.array([
            0.786412832829728, 0.224970263937584, 0.534200883186777,
            0.708371248558908, 0.163080926857473, 0.768660589584868,
            0.1762692231182, 0.310310542254487, 0.279811607250377,
            0.825543830090833, 0.847363744014823, 0.442103825982325,
            0.84951329245954, 0.523004716844064, 0.044699072032802,
            0.792400388660219, 0.292824262720788, 0.178722825110973,
            0.549380820517875, 0.352736549012222, 0.080102993555225,
            0.853135372349337, 0.790965386853156, 0.951634097517732,
            0.809945865440195, 0.313724260202943, 0.241711589286433,
            0.546972335229794, 0.270900015013911, 0.389639306011642
        ])
        o = O[:dim]

        # Load rotation matrix from MAT file
        mat_path = os.path.join(self.mat_dir, 'P11_M.mat')
        mat_data = scipy.io.loadmat(mat_path)
        M1 = mat_data['M1']
        M2 = mat_data['M2']

        # Select appropriate rotation matrix
        if dim == 10:
            M = M1
        else:
            M = M2[:dim, :dim]

        def Task(x):
            """Objective function for Problem 11 (modified cosine)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1)) + 1

            # Rotate
            z = (y - 1) @ M

            # Objective function
            f = (1 / dim) * np.sum(-z * np.cos(2 * np.sqrt(np.abs(z))), axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 11"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1)) + 1

            # Inequality constraint (initialized as zeros, matching MATLAB)
            g = np.zeros(ps)

            # Equality constraint (|h| <= delta) - Rosenbrock
            y_D_1 = y[:, :-1]
            y_2_D = y[:, 1:]
            h1 = np.abs(np.sum(100 * (y_D_1 ** 2 - y_2_D) ** 2 + (y_D_1 - 1) ** 2, axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            h1 = np.where(h1 < 0, 0, h1)

            # Combine constraints [g, h]
            cons = np.column_stack([g, h1])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -100.0)
        ub = np.full(dim, 100.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P12(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 12.

        This is a constrained optimization problem with:
        - 1 objective function (Schwefel)
        - 1 equality constraint
        - 1 inequality constraint
        - Search space: [-1000, 1000] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 12.
        """
        delta = self.delta

        # Offset vector for P12
        O = np.array([
            18.889635068428205, -59.265426383246485, 33.25998466165768,
            20.152694275194037, -10.734106238462502, -90.85053128520764,
            -12.073899411249897, 59.72307696259165, -37.44193247323578,
            25.963111555782035, 6.251460324561279, 41.478172862575434,
            86.54258849813075, 34.94822787072172, 26.864471649916382,
            79.55580868986908, -44.66218241775459, -7.305741544994362,
            87.75843366209835, 33.836473236958284, 84.53385936725138,
            80.89850629751817, 48.46967726645195, -82.0758049330533,
            -98.54273249151939, 19.55069746505636, 8.33657824668768,
            88.54888769408086, -79.08282398956031, 63.254014133387614
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 12 (Schwefel)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Schwefel function
            f = np.sum(z * np.sin(np.sqrt(np.abs(z))), axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 12"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Equality constraint (|h| <= delta)
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]
            h1 = np.abs(np.sum((z_D_1 ** 2 - z_2_D) ** 2, axis=1)) - delta

            # Inequality constraint (g <= 0)
            g1 = np.sum(z - 100 * np.cos(0.1 * z) + 10, axis=1)

            # Set negative values to 0 (satisfied constraints)
            h1 = np.where(h1 < 0, 0, h1)
            g1 = np.where(g1 < 0, 0, g1)

            # Combine constraints [g, h]
            cons = np.column_stack([g1, h1])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -1000.0)
        ub = np.full(dim, 1000.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P13(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 13.

        This is a constrained optimization problem with:
        - 1 objective function (modified Schwefel)
        - 3 inequality constraints
        - Search space: [-500, 500] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 13.
        """
        delta = self.delta

        # Offset vector for P13
        O = np.array([
            69.69311714880897, 1.509803311435702, 67.6746198312362,
            80.43173609273597, 80.47622449424348, 51.21092936019716,
            52.7723719926014, 17.248465789326257, 52.40150903116374,
            39.64846247456716, 89.86375903333635, 32.079301315169474,
            43.192499277837946, 70.79294586561508, 1.48440984483988,
            19.8566700417119, 29.502667246412756, 34.256788127976684,
            12.643016541338264, 78.57234385195876, 26.51647349482587,
            97.06430708087798, 10.180504722002471, 82.90799886855778,
            63.540231382573154, 74.78243308676124, 87.20817289266436,
            50.779655804893764, 43.05412185616204, 33.862234518700916
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 13 (modified Schwefel)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Modified Schwefel function
            f = (1 / dim) * np.sum(-z * np.sin(np.sqrt(np.abs(z))), axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 13"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Inequality constraints (g <= 0)
            g1 = -50 + (1 / (100 * dim)) * np.sum(z ** 2, axis=1)
            g2 = (50 / dim) * np.sum(np.sin(0.02 * np.pi * z), axis=1)

            # Griewank-based constraint
            indices = np.arange(1, dim + 1)
            g3 = 75 - 50 * (np.sum(z ** 2 / 4000, axis=1) -
                            np.prod(np.cos(z / np.sqrt(indices)), axis=1) + 1)

            # Equality constraint (initialized as zeros, matching MATLAB)
            h = np.zeros(ps)

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            g2 = np.where(g2 < 0, 0, g2)
            g3 = np.where(g3 < 0, 0, g3)

            # Combine constraints [g1, g2, g3, h]
            cons = np.column_stack([g1, g2, g3, h])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -500.0)
        ub = np.full(dim, 500.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P14(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 14.

        This is a constrained optimization problem with:
        - 1 objective function (Rosenbrock)
        - 3 inequality constraints
        - Search space: [-1000, 1000] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 14.
        """
        delta = self.delta

        # Offset vector for P14
        O = np.array([
            -31.718907007204272, -39.536680684207184, -46.033718058035944,
            -42.2004014684422, -28.331307546159135, -38.64403177375364,
            -11.313371899853626, -11.717383190039943, -43.345049558717875,
            -31.46016185891229, -35.57742732758397, -45.49638850141341,
            -4.177473725277878, -26.974808661067083, -46.30991533784743,
            -45.997883193212814, -29.479673271045964, -4.336542960830036,
            -43.66244285780764, -22.43896852522004, -25.89273808052249,
            -24.221450510218993, -30.3952886350567, -31.170730638052895,
            -9.859463575974534, -16.727846507426452, -44.35226340706524,
            -33.10843069426064, -7.175153678947718, -4.601421202670486
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 14 (Rosenbrock)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1))
            z = y + 1

            # Rosenbrock function
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]

            f = np.sum(100 * (z_D_1 ** 2 - z_2_D) ** 2 + (z_D_1 - 1) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 14"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            y = x - np.tile(o, (ps, 1))

            # Inequality constraints (g <= 0)
            g1 = np.sum(-y * np.cos(np.sqrt(np.abs(y))), axis=1) - dim
            g2 = np.sum(y * np.cos(np.sqrt(np.abs(y))), axis=1) - dim
            g3 = np.sum(y * np.sin(np.sqrt(np.abs(y))), axis=1) - 10 * dim

            # Equality constraint (initialized as zeros, matching MATLAB)
            h = np.zeros(ps)

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            g2 = np.where(g2 < 0, 0, g2)
            g3 = np.where(g3 < 0, 0, g3)

            # Combine constraints [g1, g2, g3, h]
            cons = np.column_stack([g1, g2, g3, h])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -1000.0)
        ub = np.full(dim, 1000.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P15(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 15.

        This is a constrained optimization problem with:
        - 1 objective function (Rosenbrock)
        - 3 inequality constraints
        - Uses rotation matrix M
        - Search space: [-1000, 1000] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 15.
        """
        import scipy.io
        import os
        delta = self.delta

        # Offset vector for P15
        O = np.array([
            -31.718907007204272, -39.536680684207184, -46.033718058035944,
            -42.2004014684422, -28.331307546159135, -38.64403177375364,
            -11.313371899853626, -11.717383190039943, -43.345049558717875,
            -31.46016185891229, -35.57742732758397, -45.49638850141341,
            -4.177473725277878, -26.974808661067083, -46.30991533784743,
            -45.997883193212814, -29.479673271045964, -4.336542960830036,
            -43.66244285780764, -22.43896852522004, -25.89273808052249,
            -24.221450510218993, -30.3952886350567, -31.170730638052895,
            -9.859463575974534, -16.727846507426452, -44.35226340706524,
            -33.10843069426064, -7.175153678947718, -4.601421202670486
        ])
        o = O[:dim]

        # Load rotation matrix from MAT file
        mat_path = os.path.join(self.mat_dir, 'P15_M.mat')
        mat_data = scipy.io.loadmat(mat_path)
        M1 = mat_data['M1']
        M2 = mat_data['M2']

        # Select appropriate rotation matrix
        if dim == 10:
            M = M1
        else:
            M = M2[:dim, :dim]

        def Task(x):
            """Objective function for Problem 15 (Rosenbrock)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1)) + 1

            # Rotate
            y = (z - 1) @ M

            # Rosenbrock function
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]

            f = np.sum(100 * (z_D_1 ** 2 - z_2_D) ** 2 + (z_D_1 - 1) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 15"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift and rotate
            z = x - np.tile(o, (ps, 1)) + 1
            y = (z - 1) @ M

            # Inequality constraints (g <= 0)
            g1 = np.sum(-y * np.cos(np.sqrt(np.abs(y))), axis=1) - dim
            g2 = np.sum(y * np.cos(np.sqrt(np.abs(y))), axis=1) - dim
            g3 = np.sum(y * np.sin(np.sqrt(np.abs(y))), axis=1) - 10 * dim

            # Equality constraint (initialized as zeros, matching MATLAB)
            h = np.zeros(ps)

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            g2 = np.where(g2 < 0, 0, g2)
            g3 = np.where(g3 < 0, 0, g3)

            # Combine constraints [g1, g2, g3, h]
            cons = np.column_stack([g1, g2, g3, h])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -1000.0)
        ub = np.full(dim, 1000.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P16(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 16.

        This is a constrained optimization problem with:
        - 1 objective function (Griewank)
        - 1 inequality constraint
        - 2 equality constraints
        - Search space: [-10, 10] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 16.
        """
        delta = self.delta

        # Offset vector for P16
        O = np.array([
            0.365972807627352, 0.429881383400138, -0.420917679577772,
            0.984265986788929, 0.324792771198785, 0.463737106835568,
            0.989554882052943, 0.307453878359996, 0.625094764380575,
            -0.358589007202526, 0.24624504504104, -0.96149609569083,
            -0.184146201911073, -0.030609388103067, 0.13366054512765,
            0.450280168292005, -0.662063233352676, 0.720384516339946,
            0.518473305175091, -0.969074121149791, -0.221655317677079,
            0.327361832246864, -0.695097713581401, -0.671724285177815,
            -0.534907819936839, -0.003991036739113, 0.486452090756303,
            -0.689962754053575, -0.138437260109118, -0.626943354458217
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 16 (Griewank)"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Griewank function
            indices = np.arange(1, dim + 1)
            f = np.sum(z ** 2 / 4000, axis=1) - np.prod(np.cos(z / np.sqrt(indices)), axis=1) + 1

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 16"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Inequality constraint (g <= 0)
            g1 = np.sum(z ** 2 - 100 * np.cos(np.pi * z) + 10, axis=1)

            # Product constraint
            g2 = np.prod(z, axis=1)

            # Equality constraints (|h| <= delta)
            h1 = np.abs(np.sum(z * np.sin(np.sqrt(np.abs(z))), axis=1)) - delta
            h2 = np.abs(np.sum(-z * np.sin(np.sqrt(np.abs(z))), axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            g2 = np.where(g2 < 0, 0, g2)
            h1 = np.where(h1 < 0, 0, h1)
            h2 = np.where(h2 < 0, 0, h2)

            # Combine constraints [g1, g2, h1, h2]
            cons = np.column_stack([g1, g2, h1, h2])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -10.0)
        ub = np.full(dim, 10.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P17(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 17.

        This is a constrained optimization problem with:
        - 1 objective function (sum of squared differences)
        - 2 inequality constraints
        - 1 equality constraint
        - Search space: [-10, 10] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 17.
        """
        delta = self.delta

        # Offset vector for P17
        O = np.array([
            -0.628245703945122, 0.331024455127249, 0.402617203423807,
            0.462742527496583, -0.513329779137884, 0.288191632492259,
            0.41479349370103, 0.916196063289011, -0.427742767473712,
            0.811971694633694, -0.202953396286476, 0.786617208861492,
            -0.583805982901842, 0.91666360939369, -0.602135912772221,
            0.503807046950863, -0.196264987447976, -0.565579687152807,
            0.540878947793462, 0.183666358669345, -0.303576255198908,
            -0.896405440407756, -0.101939801890135, -0.049819872322279,
            0.434240825173134, 0.946552963504364, -0.32578927683003,
            -0.154255792477949, 0.577967633549953, -0.573697797217518
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 17"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Sum of squared differences
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]
            f = np.sum((z_D_1 - z_2_D) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 17"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Inequality constraints (g <= 0)
            g1 = np.prod(z, axis=1)
            g2 = np.sum(z, axis=1)

            # Equality constraint (|h| <= delta)
            h1 = np.abs(np.sum(z * np.sin(4 * np.sqrt(np.abs(z))), axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            g2 = np.where(g2 < 0, 0, g2)
            h1 = np.where(h1 < 0, 0, h1)

            # Combine constraints [g1, g2, h1]
            cons = np.column_stack([g1, g2, h1])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -10.0)
        ub = np.full(dim, 10.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem

    def P18(self, dim=10) -> MTOP:
        """
        Generates CEC10_CSO Problem 18.

        This is a constrained optimization problem with:
        - 1 objective function (sum of squared differences)
        - 1 inequality constraint
        - 1 equality constraint
        - Search space: [-50, 50] for all dimensions

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the search space (default is 10 or 30).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Problem 18.
        """
        delta = self.delta

        # Offset vector for P18
        O = np.array([
            -2.494401436611803, -0.306408781638572, -2.271946840536718,
            0.381278325914122, 2.394875929583502, 0.418708663782934,
            -2.082663588220074, 0.776060342716238, -0.374312845903175,
            0.352372662321828, 1.172942728375508, -0.24450210952894,
            1.049793874089803, -1.716285448140795, -1.026167671845868,
            -1.223031642604231, 0.924946651665792, 0.93270056541258,
            -2.312880521655027, -0.671857644927313, -0.312276658254605,
            -0.973986111708943, -0.454151248193331, 2.420597958989111,
            0.050346805172393, 1.050203106200361, -0.05420584346617,
            -0.081533357726523, -0.968176219532845, 1.682281307624435
        ])
        o = O[:dim]

        def Task(x):
            """Objective function for Problem 18"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Sum of squared differences
            z_D_1 = z[:, :-1]
            z_2_D = z[:, 1:]
            f = np.sum((z_D_1 - z_2_D) ** 2, axis=1)

            return f.reshape(-1, 1)

        def Constraint(x):
            """Constraint functions for Problem 18"""
            x = np.atleast_2d(x)
            ps = x.shape[0]

            # Shift
            z = x - np.tile(o, (ps, 1))

            # Inequality constraint (g <= 0)
            g1 = (1 / dim) * np.sum(-z * np.sin(np.sqrt(np.abs(z))), axis=1)

            # Equality constraint (|h| <= delta)
            h1 = np.abs((1 / dim) * np.sum(-z * np.sin(np.sqrt(np.abs(z))), axis=1)) - delta

            # Set negative values to 0 (satisfied constraints)
            g1 = np.where(g1 < 0, 0, g1)
            h1 = np.where(h1 < 0, 0, h1)

            # Combine constraints [g1, h1]
            cons = np.column_stack([g1, h1])

            # Handle NaN values
            cons[np.isnan(cons)] = np.inf

            return cons

        # Create MTOP instance
        problem = MTOP()
        lb = np.full(dim, -50.0)
        ub = np.full(dim, 50.0)
        problem.add_task(
            Task,
            dim=dim,
            constraint_func=Constraint,
            lower_bound=lb,
            upper_bound=ub
        )

        return problem
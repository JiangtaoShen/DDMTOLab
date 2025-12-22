import numpy as np
from Methods.mtop import MTOP


class SOPM:
    """
    Implementation of Synchronous Optimal Pulse-width Modulation (SOPM) benchmark
    problems for Multi-Task Multi-Objective Optimization (MTMO).

    These problems involve optimizing switching angles for multilevel inverters
    to minimize Total Harmonic Distortion (THD) and maintain desired fundamental
    voltage component while satisfying monotonicity constraints on switching angles.

    References
    ----------
    [1] Y. Li, W. Gong, "Multiobjective Multitask Optimization with Multiple Knowledge
        Types and Transfer Adaptation," IEEE Trans. Evol. Comput., 2024.
    [2] A. Kumar et al., "A Benchmark-suite of Real-world Constrained Multi-objective
        Optimization Problems and Some Baseline Results," Swarm and Evolutionary
        Computation, vol. 67, 2021.

    Notes
    -----
    All problems are constrained 2-objective optimization problems where:
    - Objective 1: Total Harmonic Distortion (THD)
    - Objective 2: Squared deviation from target modulation index
    - Constraints: Monotonically decreasing switching angles
    """

    def __init__(self):
        """Initialize SOPM MTMO problem suite."""
        pass

    def P1(self) -> MTOP:
        """
        Generates SOPM MTMO Problem 1: **[3, 5, 7]-level Inverters**.

        Three tasks optimizing switching angles for different inverter levels.

        - T1 (3-level): 2-objective, 25-dimensional
          * Decision variables: 25 switching angles in [0, 90] degrees
          * Target modulation index: m = 0.32
          * Constraints: 24 monotonicity constraints (α_i ≥ α_{i+1})

        - T2 (5-level): 2-objective, 25-dimensional
          * Decision variables: 25 switching angles in [0, 90] degrees
          * Target modulation index: m = 0.32
          * Constraints: 24 monotonicity constraints

        - T3 (7-level): 2-objective, 25-dimensional
          * Decision variables: 25 switching angles in [0, 90] degrees
          * Target modulation index: m = 0.36
          * Constraints: 24 monotonicity constraints

        - Relationship: Different inverter levels with similar structure but
          different harmonic patterns. Tests knowledge transfer across inverter
          configurations.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 25

        # Harmonic orders (31 harmonics)
        k = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47,
                      49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])

        # Precompute 1/k^4 sum for normalization
        k4_sum = np.sum(1.0 / k ** 4)

        def T1_3level(x):
            """
            Task 1: 3-level inverter optimization

            Parameters
            ----------
            x : array-like, shape (n_samples, 25)
                Switching angles in degrees [0, 90]

            Returns
            -------
            objs : ndarray, shape (n_samples, 2)
                Objectives: [THD, (fundamental - target)^2]
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            # 3-level sign pattern: alternating signs
            s = (-np.ones(25)) ** np.arange(2, 27)
            m = 0.32  # target modulation index

            # Convert degrees to radians
            x_rad = x * np.pi / 180.0

            objs = np.zeros((n_samples, 2))

            for i in range(n_samples):
                # Compute THD (Objective 1)
                thd_sum = 0.0
                for j, kj in enumerate(k):
                    # Sum over switching angles for this harmonic
                    harmonic_sum = np.sum(s * np.cos(kj * x_rad[i, :]))
                    thd_sum += (harmonic_sum ** 2) / (kj ** 4)

                objs[i, 0] = np.sqrt(thd_sum) / np.sqrt(k4_sum)

                # Compute fundamental component deviation (Objective 2)
                fundamental = np.sum(s * np.cos(x_rad[i, :]))
                objs[i, 1] = (fundamental - m) ** 2

            return objs

        def T1_constraint(x):
            """
            Monotonicity constraints for 3-level inverter

            Returns
            -------
            cons : ndarray, shape (n_samples, 24)
                Constraint violations: g_i = x_i - x_{i+1} + 1e-6
                (should be >= 0 for feasibility)
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            cons = np.zeros((n_samples, dim - 1))

            for i in range(dim - 1):
                cons[:, i] = x[:, i] - x[:, i + 1] + 1e-6
            cons[cons < 0] = 0

            return cons

        def T2_5level(x):
            """
            Task 2: 5-level inverter optimization

            Parameters
            ----------
            x : array-like, shape (n_samples, 25)
                Switching angles in degrees [0, 90]

            Returns
            -------
            objs : ndarray, shape (n_samples, 2)
                Objectives: [THD, (fundamental - target)^2]
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            # 5-level sign pattern
            s = np.array([1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1,
                          1, -1, 1, -1, -1, 1, -1, 1, 1, -1])
            m = 0.32

            x_rad = x * np.pi / 180.0
            objs = np.zeros((n_samples, 2))

            for i in range(n_samples):
                # Compute THD
                thd_sum = 0.0
                for j, kj in enumerate(k):
                    harmonic_sum = np.sum(s * np.cos(kj * x_rad[i, :]))
                    thd_sum += (harmonic_sum ** 2) / (kj ** 4)

                objs[i, 0] = np.sqrt(thd_sum) / np.sqrt(k4_sum)

                # Compute fundamental deviation
                fundamental = np.sum(s * np.cos(x_rad[i, :]))
                objs[i, 1] = (fundamental - m) ** 2

            return objs

        def T2_constraint(x):
            """Monotonicity constraints for 5-level inverter"""
            x = np.atleast_2d(x)
            cons = np.zeros((x.shape[0], dim - 1))

            for i in range(dim - 1):
                cons[:, i] = x[:, i] - x[:, i + 1] + 1e-6
            cons[cons < 0] = 0

            return cons

        def T3_7level(x):
            """
            Task 3: 7-level inverter optimization

            Parameters
            ----------
            x : array-like, shape (n_samples, 25)
                Switching angles in degrees [0, 90]

            Returns
            -------
            objs : ndarray, shape (n_samples, 2)
                Objectives: [THD, (fundamental - target)^2]
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            # 7-level sign pattern
            s = np.array([1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1,
                          -1, -1, -1, 1, 1, -1, -1, 1, 1, 1])
            m = 0.36

            x_rad = x * np.pi / 180.0
            objs = np.zeros((n_samples, 2))

            for i in range(n_samples):
                # Compute THD
                thd_sum = 0.0
                for j, kj in enumerate(k):
                    harmonic_sum = np.sum(s * np.cos(kj * x_rad[i, :]))
                    thd_sum += (harmonic_sum ** 2) / (kj ** 4)

                objs[i, 0] = np.sqrt(thd_sum) / np.sqrt(k4_sum)

                # Compute fundamental deviation
                fundamental = np.sum(s * np.cos(x_rad[i, :]))
                objs[i, 1] = (fundamental - m) ** 2

            return objs

        def T3_constraint(x):
            """Monotonicity constraints for 7-level inverter"""
            x = np.atleast_2d(x)
            cons = np.zeros((x.shape[0], dim - 1))

            for i in range(dim - 1):
                cons[:, i] = x[:, i] - x[:, i + 1] + 1e-6
            cons[cons < 0] = 0

            return cons

        # Bounds for all tasks: [0, 90] degrees
        lb = np.zeros(dim)
        ub = 90.0 * np.ones(dim)

        problem = MTOP()
        problem.add_task(T1_3level, dim=dim, constraint_func=T1_constraint,
                         lower_bound=lb, upper_bound=ub)
        problem.add_task(T2_5level, dim=dim, constraint_func=T2_constraint,
                         lower_bound=lb, upper_bound=ub)
        problem.add_task(T3_7level, dim=dim, constraint_func=T3_constraint,
                         lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self) -> MTOP:
        """
        Generates SOPM MTMO Problem 2: **[9, 11, 13]-level Inverters**.

        Three tasks optimizing switching angles for higher-level inverters.

        - T1 (9-level): 2-objective, 30-dimensional
          * Decision variables: 30 switching angles in [0, 90] degrees
          * Target modulation index: m = 0.32
          * Constraints: 29 monotonicity constraints

        - T2 (11-level): 2-objective, 30-dimensional
          * Decision variables: 30 switching angles in [0, 90] degrees
          * Target modulation index: m = 0.3333
          * Constraints: 29 monotonicity constraints

        - T3 (13-level): 2-objective, 30-dimensional
          * Decision variables: 30 switching angles in [0, 90] degrees
          * Target modulation index: m = 0.32
          * Constraints: 29 monotonicity constraints

        - Relationship: Higher-level inverters with more complex harmonic patterns.
          Tests scalability and knowledge transfer for increased problem complexity.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        dim = 30

        # Harmonic orders (31 harmonics)
        k = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47,
                      49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])

        k4_sum = np.sum(1.0 / k ** 4)

        def T1_9level(x):
            """
            Task 1: 9-level inverter optimization

            Parameters
            ----------
            x : array-like, shape (n_samples, 30)
                Switching angles in degrees [0, 90]

            Returns
            -------
            objs : ndarray, shape (n_samples, 2)
                Objectives: [THD, (fundamental - target)^2]
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            # 9-level sign pattern
            s = np.array([1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1,
                          1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1])
            m = 0.32

            x_rad = x * np.pi / 180.0
            objs = np.zeros((n_samples, 2))

            for i in range(n_samples):
                # Compute THD
                thd_sum = 0.0
                for j, kj in enumerate(k):
                    harmonic_sum = np.sum(s * np.cos(kj * x_rad[i, :]))
                    thd_sum += (harmonic_sum ** 2) / (kj ** 4)

                objs[i, 0] = np.sqrt(thd_sum) / np.sqrt(k4_sum)

                # Compute fundamental deviation
                fundamental = np.sum(s * np.cos(x_rad[i, :]))
                objs[i, 1] = (fundamental - m) ** 2

            return objs

        def T1_constraint(x):
            """Monotonicity constraints for 9-level inverter"""
            x = np.atleast_2d(x)
            cons = np.zeros((x.shape[0], dim - 1))

            for i in range(dim - 1):
                cons[:, i] = x[:, i] - x[:, i + 1] + 1e-6
            cons[cons < 0] = 0

            return cons

        def T2_11level(x):
            """
            Task 2: 11-level inverter optimization

            Parameters
            ----------
            x : array-like, shape (n_samples, 30)
                Switching angles in degrees [0, 90]

            Returns
            -------
            objs : ndarray, shape (n_samples, 2)
                Objectives: [THD, (fundamental - target)^2]
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            # 11-level sign pattern
            s = np.array([1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
                          -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1])
            m = 0.3333

            x_rad = x * np.pi / 180.0
            objs = np.zeros((n_samples, 2))

            for i in range(n_samples):
                # Compute THD
                thd_sum = 0.0
                for j, kj in enumerate(k):
                    harmonic_sum = np.sum(s * np.cos(kj * x_rad[i, :]))
                    thd_sum += (harmonic_sum ** 2) / (kj ** 4)

                objs[i, 0] = np.sqrt(thd_sum) / np.sqrt(k4_sum)

                # Compute fundamental deviation
                fundamental = np.sum(s * np.cos(x_rad[i, :]))
                objs[i, 1] = (fundamental - m) ** 2

            return objs

        def T2_constraint(x):
            """Monotonicity constraints for 11-level inverter"""
            x = np.atleast_2d(x)
            cons = np.zeros((x.shape[0], dim - 1))

            for i in range(dim - 1):
                cons[:, i] = x[:, i] - x[:, i + 1] + 1e-6
            cons[cons < 0] = 0

            return cons

        def T3_13level(x):
            """
            Task 3: 13-level inverter optimization

            Parameters
            ----------
            x : array-like, shape (n_samples, 30)
                Switching angles in degrees [0, 90]

            Returns
            -------
            objs : ndarray, shape (n_samples, 2)
                Objectives: [THD, (fundamental - target)^2]
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            # 13-level sign pattern
            s = np.array([1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1,
                          -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1])
            m = 0.32

            x_rad = x * np.pi / 180.0
            objs = np.zeros((n_samples, 2))

            for i in range(n_samples):
                # Compute THD
                thd_sum = 0.0
                for j, kj in enumerate(k):
                    harmonic_sum = np.sum(s * np.cos(kj * x_rad[i, :]))
                    thd_sum += (harmonic_sum ** 2) / (kj ** 4)

                objs[i, 0] = np.sqrt(thd_sum) / np.sqrt(k4_sum)

                # Compute fundamental deviation
                fundamental = np.sum(s * np.cos(x_rad[i, :]))
                objs[i, 1] = (fundamental - m) ** 2

            return objs

        def T3_constraint(x):
            """Monotonicity constraints for 13-level inverter"""
            x = np.atleast_2d(x)
            cons = np.zeros((x.shape[0], dim - 1))

            for i in range(dim - 1):
                cons[:, i] = x[:, i] - x[:, i + 1] + 1e-6
            cons[cons < 0] = 0

            return cons

        # Bounds for all tasks: [0, 90] degrees
        lb = np.zeros(dim)
        ub = 90.0 * np.ones(dim)

        problem = MTOP()
        problem.add_task(T1_9level, dim=dim, constraint_func=T1_constraint,
                         lower_bound=lb, upper_bound=ub)
        problem.add_task(T2_11level, dim=dim, constraint_func=T2_constraint,
                         lower_bound=lb, upper_bound=ub)
        problem.add_task(T3_13level, dim=dim, constraint_func=T3_constraint,
                         lower_bound=lb, upper_bound=ub)
        return problem


# Settings for SOPM MTMO problems
SETTINGS = {
    'metric': 'HV',
    'n_pf': 1000,
    'pf_path': './MOReference',
    'P1': {
        'T1': [2.854987e-01, 1.024000e-01],
        'T2': [7.342985e-01, 1.024000e-01],
        'T3': [5.888136e-01, 1.296000e-01]
    },
    'P2': {
        'T1': [5.217146e-01, 1.023999e-01],
        'T2': [6.106382e-01, 1.110885e-01],
        'T3': [1.949969e+00, 1.107699e+01]
    }
}
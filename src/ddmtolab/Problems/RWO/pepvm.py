import numpy as np
from ddmtolab.Methods.mtop import MTOP


class PEPVM:
    """
    Parameter Extraction of Photovoltaic Models (PEPVM) benchmark problem.

    This problem consists of three single-objective optimization tasks for parameter
    extraction of different photovoltaic cell models using experimental I-V data.

    - Task 1: Single Diode Model (5 parameters)
    - Task 2: Double Diode Model (7 parameters)
    - Task 3: PV Module Model (5 parameters)

    The tasks share similar parameter extraction objectives but differ in model
    complexity and experimental conditions.

    References
    ----------
    [1] Li, S., Gu, Q., Gong, W., & Ning, B. (2020). An Enhanced Adaptive
        Differential Evolution Algorithm for Parameter Extraction of Photovoltaic
        Models. Energy Conversion and Management, 205, 112443.
    [2] Li, Y., Gong, W., & Li, S. (2022). Multitasking Optimization via an
        Adaptive Solver Multitasking Evolutionary Framework. Information Sciences.
    [3] Li, Y., Gong, W., & Li, S. (2023). Evolutionary Competitive Multitasking
        Optimization via Improved Adaptive Differential Evolution. Expert Systems
        with Applications, 119550.

    Attributes
    ----------
    None required for this benchmark.
    """

    def __init__(self):
        pass

    def P1(self) -> MTOP:
        """
        Generates PEPVM Problem: Three photovoltaic parameter extraction tasks.

        - Task 1: Single Diode Model (5-D, experimental data at 33°C)
        - Task 2: Double Diode Model (7-D, experimental data at 33°C)
        - Task 3: PV Module Model (5-D, experimental data at 45°C)

        All tasks minimize RMSE between measured and modeled I-V characteristics.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance with 3 tasks.
        """
        # Physical constants
        q = 1.60217646e-19  # Elementary charge (C)
        k = 1.3806503e-23  # Boltzmann constant (J/K)

        # Task 1: Single Diode Model
        def T1(x):
            """
            Single Diode Model at T=33°C

            Parameters: [I_ph, I_sd, R_s, R_sh, a]
            Returns: RMSE (shape: (N, 1))
            """
            x = np.atleast_2d(x)
            n = x.shape[0]

            T = 273.15 + 33.0
            V_t = k * T / q

            I_ph = x[:, 0:1]
            I_sd = x[:, 1:2]
            R_s = x[:, 2:3]
            R_sh = x[:, 3:4]
            a = x[:, 4:5]

            # 防止除零和过小值
            a = np.maximum(a, 1e-10)
            R_sh = np.maximum(R_sh, 1e-10)
            I_sd = np.maximum(I_sd, 1e-50)

            # Experimental I-V data
            V_L = np.array([-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185,
                            0.1678, 0.2132, 0.2545, 0.2924, 0.3269, 0.3585,
                            0.3873, 0.4137, 0.4373, 0.4590, 0.4784, 0.4960,
                            0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736,
                            0.5833, 0.5900])
            I_L = np.array([0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590,
                            0.7570, 0.7570, 0.7555, 0.7540, 0.7505, 0.7465,
                            0.7385, 0.7280, 0.7065, 0.6755, 0.6320, 0.5730,
                            0.4990, 0.4130, 0.3165, 0.2120, 0.1035, -0.0100,
                            -0.1230, -0.2100])

            # Calculate RMSE for each solution
            summ = np.zeros((n, 1))
            for i in range(len(V_L)):
                exp_arg = (V_L[i] + I_L[i] * R_s) / (a * V_t)
                exp_arg = np.clip(exp_arg, -700, 700)

                exp_term = np.exp(exp_arg) - 1
                # 限制exp_term避免后续计算溢出
                exp_term = np.clip(exp_term, -1e10, 1e10)

                y1 = (I_ph - I_sd * exp_term
                      - (V_L[i] + I_L[i] * R_s) / R_sh - I_L[i])

                # 限制y1的范围避免平方时溢出
                y1 = np.clip(y1, -1e10, 1e10)
                summ += y1 ** 2

            obj = np.sqrt(summ / len(V_L))
            return obj

        # Task 2: Double Diode Model
        def T2(x):
            """
            Double Diode Model at T=33°C

            Parameters: [I_ph, I_sd1, R_s, R_sh, a1, I_sd2, a2]
            Returns: RMSE (shape: (N, 1))
            """
            x = np.atleast_2d(x)
            n = x.shape[0]

            T = 273.15 + 33.0
            V_t = k * T / q

            I_ph = x[:, 0:1]
            I_sd1 = x[:, 1:2]
            R_s = x[:, 2:3]
            R_sh = x[:, 3:4]
            a1 = x[:, 4:5]
            I_sd2 = x[:, 5:6]
            a2 = x[:, 6:7]

            # 防止除零和过小值
            a1 = np.maximum(a1, 1e-10)
            a2 = np.maximum(a2, 1e-10)
            R_sh = np.maximum(R_sh, 1e-10)
            I_sd1 = np.maximum(I_sd1, 1e-50)
            I_sd2 = np.maximum(I_sd2, 1e-50)

            # Experimental I-V data
            V_L = np.array([-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185,
                            0.1678, 0.2132, 0.2545, 0.2924, 0.3269, 0.3585,
                            0.3873, 0.4137, 0.4373, 0.4590, 0.4784, 0.4960,
                            0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736,
                            0.5833, 0.5900])
            I_L = np.array([0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590,
                            0.7570, 0.7570, 0.7555, 0.7540, 0.7505, 0.7465,
                            0.7385, 0.7280, 0.7065, 0.6755, 0.6320, 0.5730,
                            0.4990, 0.4130, 0.3165, 0.2120, 0.1035, -0.0100,
                            -0.1230, -0.2100])

            # Calculate RMSE for each solution
            summ = np.zeros((n, 1))
            for i in range(len(V_L)):
                exp_arg1 = (V_L[i] + I_L[i] * R_s) / (a1 * V_t)
                exp_arg2 = (V_L[i] + I_L[i] * R_s) / (a2 * V_t)
                exp_arg1 = np.clip(exp_arg1, -700, 700)
                exp_arg2 = np.clip(exp_arg2, -700, 700)

                exp_term1 = np.exp(exp_arg1) - 1
                exp_term2 = np.exp(exp_arg2) - 1
                # 限制exp_term避免后续计算溢出
                exp_term1 = np.clip(exp_term1, -1e10, 1e10)
                exp_term2 = np.clip(exp_term2, -1e10, 1e10)

                y1 = (I_ph
                      - I_sd1 * exp_term1
                      - I_sd2 * exp_term2
                      - (V_L[i] + I_L[i] * R_s) / R_sh - I_L[i])

                # 限制y1的范围避免平方时溢出
                y1 = np.clip(y1, -1e10, 1e10)
                summ += y1 ** 2

            obj = np.sqrt(summ / len(V_L))
            return obj

        # Task 3: PV Module Model
        def T3(x):
            """
            PV Module Model at T=45°C

            Parameters: [I_ph, I_sd, R_s, R_sh, a]
            Returns: RMSE (shape: (N, 1))
            """
            x = np.atleast_2d(x)
            n = x.shape[0]

            T = 273.15 + 45.0
            V_t = k * T / q

            I_ph = x[:, 0:1]
            I_sd = x[:, 1:2]
            R_s = x[:, 2:3]
            R_sh = x[:, 3:4]
            a = x[:, 4:5]
            Ns = 1

            # 防止除零和过小值
            a = np.maximum(a, 1e-10)
            R_sh = np.maximum(R_sh, 1e-10)
            I_sd = np.maximum(I_sd, 1e-50)

            # Experimental I-V data
            V_L = np.array([0.1248, 1.8093, 3.3511, 4.7622, 6.0538, 7.2364,
                            8.3189, 9.3097, 10.2163, 11.0449, 11.8018, 12.4929,
                            13.1231, 13.6983, 14.2221, 14.6995, 15.1346, 15.5311,
                            15.8929, 16.2229, 16.5241, 16.7987, 17.0499, 17.2793,
                            17.4885])
            I_L = np.array([1.0315, 1.0300, 1.0260, 1.0220, 1.0180, 1.0155,
                            1.0140, 1.0100, 1.0035, 0.9880, 0.9630, 0.9255,
                            0.8725, 0.8075, 0.7265, 0.6345, 0.5345, 0.4275,
                            0.3185, 0.2085, 0.1010, -0.0080, -0.1110, -0.2090,
                            -0.3030])

            # Calculate RMSE for each solution
            summ = np.zeros((n, 1))
            for i in range(len(V_L)):
                exp_arg = (V_L[i] + I_L[i] * R_s) / (a * Ns * V_t)
                exp_arg = np.clip(exp_arg, -700, 700)

                exp_term = np.exp(exp_arg) - 1
                # 限制exp_term避免后续计算溢出
                exp_term = np.clip(exp_term, -1e10, 1e10)

                y1 = (I_ph - I_sd * exp_term
                      - (V_L[i] + I_L[i] * R_s) / R_sh - I_L[i])

                # 限制y1的范围避免平方时溢出
                y1 = np.clip(y1, -1e10, 1e10)
                summ += y1 ** 2

            obj = np.sqrt(summ / len(V_L))
            return obj

        # Define bounds for each task
        # Task 1: [I_ph, I_sd, R_s, R_sh, a]
        lb1 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        ub1 = np.array([1.0, 1e-6, 0.5, 100.0, 2.0])

        # Task 2: [I_ph, I_sd1, R_s, R_sh, a1, I_sd2, a2]
        lb2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        ub2 = np.array([1.0, 1e-6, 0.5, 100.0, 2.0, 1e-6, 2.0])

        # Task 3: [I_ph, I_sd, R_s, R_sh, a]
        lb3 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        ub3 = np.array([2.0, 5e-5, 2.0, 2000.0, 50.0])

        # Create MTOP instance
        problem = MTOP()
        problem.add_task(T1, dim=5, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=7, lower_bound=lb2, upper_bound=ub2)
        problem.add_task(T3, dim=5, lower_bound=lb3, upper_bound=ub3)

        return problem
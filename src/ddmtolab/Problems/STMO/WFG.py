import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.algo_utils import nd_sort
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point


class WFG:
    """
    Implementation of the WFG (Walking Fish Group) test suite for multi-objective optimization.

    The WFG test problems are scalable benchmarks designed to test various characteristics
    of multi-objective optimization algorithms, including bias, flatness, and mixed Pareto fronts.
    """

    # =============================================================================
    # Transformation Functions
    # =============================================================================

    @staticmethod
    def _transformation_shift_linear(value, shift=0.35):
        """Linear shift transformation."""
        result = np.abs(value - shift) / np.abs(np.floor(shift - value) + shift)
        return np.clip(result, 0, 1)

    @staticmethod
    def _transformation_shift_deceptive(y, A=0.35, B=0.005, C=0.05):
        """Deceptive shift transformation."""
        tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
        tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
        ret = 1.0 + (np.abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
        return np.clip(ret, 0, 1)

    @staticmethod
    def _transformation_shift_multi_modal(y, A, B, C):
        """Multi-modal shift transformation."""
        tmp1 = np.abs(y - C) / (2.0 * (np.floor(C - y) + C))
        tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
        ret = (1.0 + np.cos(tmp2) + 4.0 * B * np.power(tmp1, 2.0)) / (B + 2.0)
        return np.clip(ret, 0, 1)

    @staticmethod
    def _transformation_bias_flat(y, a, b, c):
        """Flat bias transformation."""
        ret = a + np.minimum(0, np.floor(y - b)) * (a * (b - y) / b) - \
              np.minimum(0, np.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
        return np.clip(ret, 0, 1)

    @staticmethod
    def _transformation_bias_poly(y, alpha):
        """Polynomial bias transformation."""
        return np.clip(y ** alpha, 0, 1)

    @staticmethod
    def _transformation_param_dependent(y, y_deg, A=0.98 / 49.98, B=0.02, C=50.0):
        """Parameter-dependent transformation."""
        aux = A - (1.0 - 2.0 * y_deg) * np.abs(np.floor(0.5 - y_deg) + A)
        ret = np.power(y, B + (C - B) * aux)
        return np.clip(ret, 0, 1)

    @staticmethod
    def _transformation_param_deceptive(y, A=0.35, B=0.001, C=0.05):
        """Parameter-deceptive transformation."""
        tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
        tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
        ret = 1.0 + (np.abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
        return np.clip(ret, 0, 1)

    # =============================================================================
    # Reduction Functions
    # =============================================================================

    @staticmethod
    def _reduction_weighted_sum(y, w):
        """Weighted sum reduction."""
        return np.clip(np.sum(y * w, axis=1) / np.sum(w), 0, 1)

    @staticmethod
    def _reduction_weighted_sum_uniform(y):
        """Uniform weighted sum reduction (mean)."""
        return np.clip(np.mean(y, axis=1), 0, 1)

    @staticmethod
    def _reduction_non_sep(y, A):
        """Non-separable reduction."""
        n_samples, m = y.shape
        val = np.ceil(A / 2.0)

        num = np.zeros(n_samples)
        for j in range(m):
            num += y[:, j]
            for k in range(int(A) - 1):
                num += np.abs(y[:, j] - y[:, (1 + j + k) % m])

        denom = m * val * (1.0 + 2.0 * A - 2 * val) / A
        return np.clip(num / denom, 0, 1)

    # =============================================================================
    # Shape Functions
    # =============================================================================

    @staticmethod
    def _shape_linear(x, m):
        """Linear shape function."""
        M = x.shape[1]
        if m == 1:
            ret = np.prod(x, axis=1)
        elif 1 < m <= M:
            ret = np.prod(x[:, :M - m + 1], axis=1)
            ret *= 1.0 - x[:, M - m + 1]
        else:
            ret = 1.0 - x[:, 0]
        return np.clip(ret, 0, 1)

    @staticmethod
    def _shape_convex(x, m):
        """Convex shape function."""
        M = x.shape[1]
        if m == 1:
            ret = np.prod(1.0 - np.cos(0.5 * x[:, :M] * np.pi), axis=1)
        elif 1 < m <= M:
            ret = np.prod(1.0 - np.cos(0.5 * x[:, :M - m + 1] * np.pi), axis=1)
            ret *= 1.0 - np.sin(0.5 * x[:, M - m + 1] * np.pi)
        else:
            ret = 1.0 - np.sin(0.5 * x[:, 0] * np.pi)
        return np.clip(ret, 0, 1)

    @staticmethod
    def _shape_concave(x, m):
        """Concave shape function."""
        M = x.shape[1]
        if m == 1:
            ret = np.prod(np.sin(0.5 * x[:, :M] * np.pi), axis=1)
        elif 1 < m <= M:
            ret = np.prod(np.sin(0.5 * x[:, :M - m + 1] * np.pi), axis=1)
            ret *= np.cos(0.5 * x[:, M - m + 1] * np.pi)
        else:
            ret = np.cos(0.5 * x[:, 0] * np.pi)
        return np.clip(ret, 0, 1)

    @staticmethod
    def _shape_mixed(x, A=5.0, alpha=1.0):
        """Mixed shape function."""
        aux = 2.0 * A * np.pi
        ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
        return np.clip(ret, 0, 1)

    @staticmethod
    def _shape_disconnected(x, alpha=1.0, beta=1.0, A=5.0):
        """Disconnected shape function."""
        aux = np.cos(A * np.pi * x ** beta)
        ret = 1.0 - x ** alpha * aux ** 2
        return np.clip(ret, 0, 1)

    # =============================================================================
    # Legacy Methods (for backward compatibility)
    # =============================================================================

    @staticmethod
    def _s_linear(y, A):
        """Shape function: linear transformation (legacy)."""
        return np.abs(y - A) / np.abs(np.floor(A - y) + A)

    @staticmethod
    def _b_flat(y, A, B, C):
        """Bias function: flat region (legacy)."""
        Output = A + np.minimum(0, np.floor(y - B)) * A * (B - y) / B - \
                 np.minimum(0, np.floor(C - y)) * (1 - A) * (y - C) / (1 - C)
        return np.clip(Output, 0, 1)

    @staticmethod
    def _b_poly(y, alpha):
        """Bias function: polynomial (legacy)."""
        return y ** alpha

    @staticmethod
    def _r_sum(y, w):
        """Reduction function: weighted sum (legacy)."""
        return np.sum(y * w, axis=1) / np.sum(w)

    @staticmethod
    def _r_sum_uniform(y):
        """Reduction function: uniform weighted sum (legacy)."""
        return np.mean(y, axis=1)

    @staticmethod
    def _r_nonsep(y, A):
        """Reduction function: non-separable (legacy)."""
        n_samples, m = y.shape
        val = np.ceil(A / 2.0)

        num = np.zeros(n_samples)
        for j in range(m):
            num += y[:, j]
            for k in range(int(A) - 1):
                num += np.abs(y[:, j] - y[:, (1 + j + k) % m])

        denom = m * val * (1.0 + 2.0 * A - 2 * val) / A
        return np.clip(num / denom, 0, 1)

    @staticmethod
    def _convex(x):
        """Convex shape function (legacy - returns all objectives)."""
        n_samples = x.shape[0]
        M = x.shape[1]
        Output = np.ones((n_samples, M))

        for i in range(M):
            for j in range(M - i - 1):
                Output[:, i] *= (1 - np.cos(x[:, j] * np.pi / 2))
            if i > 0:
                Output[:, i] *= (1 - np.sin(x[:, M - i - 1] * np.pi / 2))

        return Output

    @staticmethod
    def _mixed(x):
        """Mixed shape function (legacy)."""
        aux = 2.0 * 5.0 * np.pi
        return (1.0 - x[:, 0] - (np.cos(aux * x[:, 0] + 0.5 * np.pi) / aux))

    @staticmethod
    def _disconnected(x, alpha=1.0, beta=1.0, A=5.0):
        """Disconnected shape function (legacy)."""
        aux = np.cos(A * np.pi * x[:, 0] ** beta)
        return 1.0 - x[:, 0] ** alpha * aux ** 2

    # =============================================================================
    # WFG Problem Implementations
    # =============================================================================

    def WFG1(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG1** problem.

        WFG1 features a mixed Pareto front with both convex and non-convex regions,
        along with polynomial bias and flat region transformations.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG1 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            z01 = x / (2 * np.arange(1, dim + 1))

            t1 = np.zeros((n_samples, K + L))
            t1[:, :K] = z01[:, :K]
            t1[:, K:] = self._s_linear(z01[:, K:], 0.35)

            t2 = np.zeros((n_samples, K + L))
            t2[:, :K] = t1[:, :K]
            t2[:, K:] = self._b_flat(t1[:, K:], 0.8, 0.75, 0.85)

            t3 = self._b_poly(t2, 0.02)

            t4 = np.zeros((n_samples, M))
            for i in range(M - 1):
                start_idx = i * K // (M - 1)
                end_idx = (i + 1) * K // (M - 1)
                w = 2 * np.arange(start_idx + 1, end_idx + 1)
                t4[:, i] = self._r_sum(t3[:, start_idx:end_idx], w)

            w_last = 2 * np.arange(K + 1, K + L + 1)
            t4[:, M - 1] = self._r_sum(t3[:, K:K + L], w_last)

            x_final = np.zeros((n_samples, M))
            for i in range(M - 1):
                x_final[:, i] = np.maximum(t4[:, M - 1], A[i]) * (t4[:, i] - 0.5) + 0.5
            x_final[:, M - 1] = t4[:, M - 1]

            h = self._convex(x_final)
            h[:, M - 1] = self._mixed(x_final)

            obj = D * x_final[:, M - 1:M] + S * h

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG2(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG2** problem.

        WFG2 features a disconnected Pareto front and uses non-separable reduction
        functions. It tests an algorithm's ability to maintain diversity across
        disconnected regions.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG2 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if L % 2 != 0:
            raise ValueError('In WFG2 the distance-related parameter (L) must be divisible by 2.')

        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            z01 = x / (2 * np.arange(1, dim + 1))

            t1 = np.zeros((n_samples, K + L))
            t1[:, :K] = z01[:, :K]
            t1[:, K:] = self._s_linear(z01[:, K:], 0.35)

            t2_parts = [t1[:, i] for i in range(K)]
            ind_non_sep = K + L // 2
            for i in range(K, ind_non_sep):
                head = K + 2 * (i - K)
                tail = K + 2 * (i - K) + 2
                t2_parts.append(self._r_nonsep(t1[:, head:tail], 2))

            t2 = np.column_stack(t2_parts)

            ind_r_sum = K + (dim - K) // 2
            gap = K // (M - 1)

            t3 = np.zeros((n_samples, M))
            for i in range(M - 1):
                start_idx = i * gap
                end_idx = (i + 1) * gap
                t3[:, i] = self._r_sum_uniform(t2[:, start_idx:end_idx])

            t3[:, M - 1] = self._r_sum_uniform(t2[:, K:ind_r_sum])

            x_final = np.zeros((n_samples, M))
            for i in range(M - 1):
                x_final[:, i] = np.maximum(t3[:, M - 1], A[i]) * (t3[:, i] - 0.5) + 0.5
            x_final[:, M - 1] = t3[:, M - 1]

            h = self._convex(x_final)
            h[:, M - 1] = self._disconnected(x_final, alpha=1.0, beta=1.0, A=5.0)

            obj = D * x_final[:, M - 1:M] + S * h

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG3(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG3** problem.

        WFG3 features a linear Pareto front with a degenerate geometry, testing
        an algorithm's ability to handle problems with dependencies between objectives.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG3 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if L % 2 != 0:
            raise ValueError('In WFG3 the distance-related parameter (L) must be divisible by 2.')

        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)
        A[1:] = 0

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            xu = 2 * np.arange(1, dim + 1)
            y = x / xu

            y[:, K:dim] = self._transformation_shift_linear(y[:, K:dim], 0.35)

            y_list = [y[:, i] for i in range(K)]

            l = dim - K
            ind_non_sep = K + l // 2

            i = K + 1
            while i <= ind_non_sep:
                head = K + 2 * (i - K) - 2
                tail = K + 2 * (i - K)
                y_list.append(self._reduction_non_sep(y[:, head:tail], 2))
                i += 1

            y = np.column_stack(y_list)

            ind_r_sum = K + (dim - K) // 2
            gap = K // (M - 1)

            t = []
            for m in range(1, M):
                t.append(self._reduction_weighted_sum_uniform(y[:, (m - 1) * gap: m * gap]))
            t.append(self._reduction_weighted_sum_uniform(y[:, K:ind_r_sum]))

            y = np.column_stack(t)

            x_final = []
            for i in range(M - 1):
                x_final.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
            x_final.append(y[:, -1])
            y = np.column_stack(x_final)

            h = []
            for m in range(M):
                h.append(self._shape_linear(y[:, :-1], m + 1))

            obj = D * y[:, -1][:, None] + S * np.column_stack(h)

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG4(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG4** problem.

        WFG4 features a concave Pareto front with multi-modal transformations,
        testing an algorithm's ability to handle multi-modality.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG4 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            xu = 2 * np.arange(1, dim + 1)
            y = x / xu

            y = self._transformation_shift_multi_modal(y, 30.0, 10.0, 0.35)

            gap = K // (M - 1)
            t = []
            for m in range(1, M):
                t.append(self._reduction_weighted_sum_uniform(y[:, (m - 1) * gap: m * gap]))
            t.append(self._reduction_weighted_sum_uniform(y[:, K:]))

            y = np.column_stack(t)

            x_final = []
            for i in range(M - 1):
                x_final.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
            x_final.append(y[:, -1])
            y = np.column_stack(x_final)

            h = []
            for m in range(M):
                h.append(self._shape_concave(y[:, :-1], m + 1))

            obj = D * y[:, -1][:, None] + S * np.column_stack(h)

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG5(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG5** problem.

        WFG5 features a concave Pareto front with parameter-deceptive transformations,
        testing an algorithm's ability to handle deceptive fitness landscapes.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG5 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            xu = 2 * np.arange(1, dim + 1)
            y = x / xu

            y = self._transformation_param_deceptive(y, A=0.35, B=0.001, C=0.05)

            gap = K // (M - 1)
            t = []
            for m in range(1, M):
                t.append(self._reduction_weighted_sum_uniform(y[:, (m - 1) * gap: m * gap]))
            t.append(self._reduction_weighted_sum_uniform(y[:, K:]))

            y = np.column_stack(t)

            x_final = []
            for i in range(M - 1):
                x_final.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
            x_final.append(y[:, -1])
            y = np.column_stack(x_final)

            h = []
            for m in range(M):
                h.append(self._shape_concave(y[:, :-1], m + 1))

            obj = D * y[:, -1][:, None] + S * np.column_stack(h)

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG6(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG6** problem.

        WFG6 features a concave Pareto front with non-separable reduction functions,
        testing an algorithm's ability to handle non-separable problems.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG6 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            xu = 2 * np.arange(1, dim + 1)
            y = x / xu

            y[:, K:dim] = self._transformation_shift_linear(y[:, K:dim], 0.35)

            gap = K // (M - 1)
            t = []
            for m in range(1, M):
                t.append(self._reduction_non_sep(y[:, (m - 1) * gap: m * gap], gap))
            t.append(self._reduction_non_sep(y[:, K:], dim - K))

            y = np.column_stack(t)

            x_final = []
            for i in range(M - 1):
                x_final.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
            x_final.append(y[:, -1])
            y = np.column_stack(x_final)

            h = []
            for m in range(M):
                h.append(self._shape_concave(y[:, :-1], m + 1))

            obj = D * y[:, -1][:, None] + S * np.column_stack(h)

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG7(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG7** problem.

        WFG7 features a concave Pareto front with parameter-dependent transformations,
        testing an algorithm's ability to handle parameter dependencies.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG7 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            xu = 2 * np.arange(1, dim + 1)
            y = x / xu

            for i in range(K):
                aux = self._reduction_weighted_sum_uniform(y[:, i + 1:])
                y[:, i] = self._transformation_param_dependent(y[:, i], aux)

            y[:, K:dim] = self._transformation_shift_linear(y[:, K:dim], 0.35)

            gap = K // (M - 1)
            t = []
            for m in range(1, M):
                t.append(self._reduction_weighted_sum_uniform(y[:, (m - 1) * gap: m * gap]))
            t.append(self._reduction_weighted_sum_uniform(y[:, K:]))

            y = np.column_stack(t)

            x_final = []
            for i in range(M - 1):
                x_final.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
            x_final.append(y[:, -1])
            y = np.column_stack(x_final)

            h = []
            for m in range(M):
                h.append(self._shape_concave(y[:, :-1], m + 1))

            obj = D * y[:, -1][:, None] + S * np.column_stack(h)

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG8(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG8** problem.

        WFG8 features a concave Pareto front with parameter-dependent transformations
        on distance parameters, testing an algorithm's ability to handle complex
        parameter dependencies.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG8 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            xu = 2 * np.arange(1, dim + 1)
            y = x / xu

            ret = []
            for i in range(K, dim):
                aux = self._reduction_weighted_sum_uniform(y[:, :i])
                ret.append(self._transformation_param_dependent(y[:, i], aux, A=0.98 / 49.98, B=0.02, C=50.0))
            y[:, K:dim] = np.column_stack(ret)

            y[:, K:dim] = self._transformation_shift_linear(y[:, K:dim], 0.35)

            gap = K // (M - 1)
            t = []
            for m in range(1, M):
                t.append(self._reduction_weighted_sum_uniform(y[:, (m - 1) * gap: m * gap]))
            t.append(self._reduction_weighted_sum_uniform(y[:, K:]))

            y = np.column_stack(t)

            x_final = []
            for i in range(M - 1):
                x_final.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
            x_final.append(y[:, -1])
            y = np.column_stack(x_final)

            h = []
            for m in range(M):
                h.append(self._shape_concave(y[:, :-1], m + 1))

            obj = D * y[:, -1][:, None] + S * np.column_stack(h)

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def WFG9(self, M=3, K=None, dim=None) -> MTOP:
        """
        Generates the **WFG9** problem.

        WFG9 features a concave Pareto front with parameter-dependent transformations,
        deceptive and multi-modal shifts, and non-separable reduction functions,
        testing multiple characteristics simultaneously.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        K : int, optional
            Position parameter, which should be a multiple of M-1.
            If None, it is set to M-1 (default is None).
        dim : int, optional
            Number of decision variables. If None, it is set to K + 10
            (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG9 task.
        """
        if K is None:
            K = M - 1

        L = 10
        if dim is None:
            dim = K + L

        D = 1
        S = 2 * np.arange(1, M + 1)
        A = np.ones(M - 1)

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            xu = 2 * np.arange(1, dim + 1)
            y = x / xu

            ret = []
            for i in range(0, dim - 1):
                aux = self._reduction_weighted_sum_uniform(y[:, i + 1:])
                ret.append(self._transformation_param_dependent(y[:, i], aux))
            y[:, :dim - 1] = np.column_stack(ret)

            a = []
            for i in range(K):
                a.append(self._transformation_shift_deceptive(y[:, i], 0.35, 0.001, 0.05))
            b = []
            for i in range(K, dim):
                b.append(self._transformation_shift_multi_modal(y[:, i], 30.0, 95.0, 0.35))
            y = np.column_stack(a + b)

            gap = K // (M - 1)
            t = []
            for m in range(1, M):
                t.append(self._reduction_non_sep(y[:, (m - 1) * gap: m * gap], gap))
            t.append(self._reduction_non_sep(y[:, K:], dim - K))

            y = np.column_stack(t)

            x_final = []
            for i in range(M - 1):
                x_final.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
            x_final.append(y[:, -1])
            y = np.column_stack(x_final)

            h = []
            for m in range(M):
                h.append(self._shape_concave(y[:, :-1], m + 1))

            obj = D * y[:, -1][:, None] + S * np.column_stack(h)

            return obj

        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1, dtype=float)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

# Pareto Front Generation Functions
def WFG1_PF(N: int, M: int) -> np.ndarray:
    """
    Generate WFG1 Pareto Front using sampling approach.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    k = M - 1
    l = 10
    n_var = k + l

    # Generate optimal position parameters (heavily biased toward 0)
    K = np.power(np.random.random((N, k)), 50.0)

    # Complete solution with distance parameters set to 0.35
    suffix = np.full((N, l), 0.35)
    X = np.column_stack([K, suffix])

    # Scale by upper bounds
    xu = 2 * np.arange(1, n_var + 1)
    X = X * xu

    # Normalize
    y = X / xu

    # t1: Linear shift on distance parameters
    y[:, k:] = np.abs(y[:, k:] - 0.35) / np.abs(np.floor(0.35 - y[:, k:]) + 0.35)

    # t2: Flat bias on distance parameters
    def _b_flat(y, A, B, C):
        Output = A + np.minimum(0, np.floor(y - B)) * A * (B - y) / B - \
                 np.minimum(0, np.floor(C - y)) * (1 - A) * (y - C) / (1 - C)
        return np.clip(Output, 0, 1)

    y[:, k:] = _b_flat(y[:, k:], 0.8, 0.75, 0.85)

    # t3: Polynomial bias on all parameters
    y = y ** 0.02

    # t4: Weighted sum reduction
    w = np.arange(2, 2 * n_var + 1, 2)
    gap = k // (M - 1)
    t = []
    for m in range(M - 1):
        _y = y[:, m * gap:(m + 1) * gap]
        _w = w[m * gap:(m + 1) * gap]
        t.append(np.sum(_y * _w, axis=1) / np.sum(_w))
    t.append(np.sum(y[:, k:] * w[k:], axis=1) / np.sum(w[k:]))
    y = np.column_stack(t)

    # Post-processing with degeneracy
    A = np.ones(M - 1)
    x = []
    for i in range(M - 1):
        x.append(np.maximum(y[:, -1], A[i]) * (y[:, i] - 0.5) + 0.5)
    x.append(y[:, -1])
    y = np.column_stack(x)

    # Shape functions
    S = 2 * np.arange(1, M + 1)
    h = []

    # Convex shape for first M-1 objectives
    for m in range(M - 1):
        if m == 0:
            h_m = np.prod(1.0 - np.cos(0.5 * y[:, :M - 1] * np.pi), axis=1)
        else:
            h_m = np.prod(1.0 - np.cos(0.5 * y[:, :M - 1 - m] * np.pi), axis=1)
            h_m *= (1.0 - np.sin(0.5 * y[:, M - 1 - m] * np.pi))
        h.append(h_m)

    # Mixed shape for last objective
    aux = 2.0 * 5.0 * np.pi
    h_last = (1.0 - y[:, 0] - (np.cos(aux * y[:, 0] + 0.5 * np.pi) / aux))
    h.append(h_last)

    # Final objectives
    F = y[:, -1][:, None] + S * np.column_stack(h)

    return F


def WFG2_PF(N: int, M: int) -> np.ndarray:
    """
    Generate WFG2 Pareto Front using sampling approach.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    k = M - 1
    l = 10
    n_var = k + l

    # Generate optimal position parameters (uniform random in [0, 1])
    K = np.random.random((N, k))

    # Complete solution with distance parameters set to 0.35
    suffix = np.full((N, l), 0.35)
    X = np.column_stack([K, suffix])

    # Scale by upper bounds
    xu = 2 * np.arange(1, n_var + 1)
    X = X * xu

    # Normalize
    y = X / xu

    # t1: Linear shift on distance parameters
    y[:, k:] = np.abs(y[:, k:] - 0.35) / np.abs(np.floor(0.35 - y[:, k:]) + 0.35)

    # t2: Non-separable reduction
    def _r_nonsep(y, A):
        n_samples, m = y.shape
        val = np.ceil(A / 2.0)
        num = np.zeros(n_samples)
        for j in range(m):
            num += y[:, j]
            for k_idx in range(int(A) - 1):
                num += np.abs(y[:, j] - y[:, (1 + j + k_idx) % m])
        denom = m * val * (1.0 + 2.0 * A - 2 * val) / A
        return np.clip(num / denom, 0, 1)

    t2_parts = [y[:, i] for i in range(k)]
    ind_non_sep = k + l // 2
    for i in range(k, ind_non_sep):
        head = k + 2 * (i - k)
        tail = k + 2 * (i - k) + 2
        t2_parts.append(_r_nonsep(y[:, head:tail], 2))

    y = np.column_stack(t2_parts)

    # t3: Uniform weighted sum reduction to M objectives
    ind_r_sum = k + (n_var - k) // 2
    gap = k // (M - 1)

    t3 = np.zeros((N, M))
    for i in range(M - 1):
        start_idx = i * gap
        end_idx = (i + 1) * gap
        t3[:, i] = np.mean(y[:, start_idx:end_idx], axis=1)

    t3[:, M - 1] = np.mean(y[:, k:ind_r_sum], axis=1)

    # Post-processing with degeneracy
    A = np.ones(M - 1)
    x = []
    for i in range(M - 1):
        x.append(np.maximum(t3[:, -1], A[i]) * (t3[:, i] - 0.5) + 0.5)
    x.append(t3[:, -1])
    y = np.column_stack(x)

    # Shape functions
    S = 2 * np.arange(1, M + 1)
    h = []

    # Convex shape for first M-1 objectives
    for m in range(M - 1):
        if m == 0:
            h_m = np.prod(1.0 - np.cos(0.5 * y[:, :M - 1] * np.pi), axis=1)
        else:
            h_m = np.prod(1.0 - np.cos(0.5 * y[:, :M - 1 - m] * np.pi), axis=1)
            h_m *= (1.0 - np.sin(0.5 * y[:, M - 1 - m] * np.pi))
        h.append(h_m)

    # Disconnected shape for last objective
    aux = np.cos(5.0 * np.pi * y[:, 0] ** 1.0)
    h_last = 1.0 - y[:, 0] ** 1.0 * aux ** 2
    h.append(h_last)

    # Final objectives
    F = y[:, -1][:, None] + S * np.column_stack(h)

    front_no, max_fno = nd_sort(F, len(F))
    first_front_indices = np.where(front_no == 1)[0]
    F = F[first_front_indices]

    return F


def WFG3_PF(N: int, M: int) -> np.ndarray:
    """
    Generate WFG3 Pareto Front using sampling approach.

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    k = M - 1
    l = 10
    n_var = k + l

    # Generate optimal position parameters (uniform random in [0, 1])
    K = np.random.random((N, k))

    # Complete solution with distance parameters set to 0.35
    suffix = np.full((N, l), 0.35)
    X = np.column_stack([K, suffix])

    # Scale by upper bounds
    xu = 2 * np.arange(1, n_var + 1)
    X = X * xu

    # Normalize
    y = X / xu

    # t1: Linear shift on distance parameters
    y[:, k:] = np.abs(y[:, k:] - 0.35) / np.abs(np.floor(0.35 - y[:, k:]) + 0.35)

    # t2: Non-separable reduction
    def _r_nonsep(y, A):
        n_samples, m = y.shape
        val = np.ceil(A / 2.0)
        num = np.zeros(n_samples)
        for j in range(m):
            num += y[:, j]
            for k_idx in range(int(A) - 1):
                num += np.abs(y[:, j] - y[:, (1 + j + k_idx) % m])
        denom = m * val * (1.0 + 2.0 * A - 2 * val) / A
        return np.clip(num / denom, 0, 1)

    t2_parts = [y[:, i] for i in range(k)]
    ind_non_sep = k + l // 2
    for i in range(k, ind_non_sep):
        head = k + 2 * (i - k)
        tail = k + 2 * (i - k) + 2
        t2_parts.append(_r_nonsep(y[:, head:tail], 2))

    y = np.column_stack(t2_parts)

    # t3: Uniform weighted sum reduction to M objectives
    ind_r_sum = k + (n_var - k) // 2
    gap = k // (M - 1)

    t3 = np.zeros((N, M))
    for i in range(M - 1):
        start_idx = i * gap
        end_idx = (i + 1) * gap
        t3[:, i] = np.mean(y[:, start_idx:end_idx], axis=1)

    t3[:, M - 1] = np.mean(y[:, k:ind_r_sum], axis=1)

    # Post-processing with degeneracy (A[1:] = 0 for WFG3)
    A = np.ones(M - 1)
    A[1:] = 0
    x = []
    for i in range(M - 1):
        x.append(np.maximum(t3[:, -1], A[i]) * (t3[:, i] - 0.5) + 0.5)
    x.append(t3[:, -1])
    y = np.column_stack(x)

    # Shape functions - Linear for all objectives
    S = 2 * np.arange(1, M + 1)
    h = []

    for m in range(M):
        if m == 0:
            h_m = np.prod(y[:, :M - 1], axis=1)
        elif 0 < m < M - 1:
            h_m = np.prod(y[:, :M - 1 - m], axis=1) * (1.0 - y[:, M - 1 - m])
        else:
            h_m = 1.0 - y[:, 0]
        h.append(h_m)

    # Final objectives
    F = y[:, -1][:, None] + S * np.column_stack(h)

    return F


def WFG4_PF(N: int, M: int) -> np.ndarray:
    """
    Generate WFG4 Pareto Front (1/8 sphere surface).

    Parameters
    ----------
    N : int
        Number of points to generate on the PF.
    M : int
        Number of objectives.

    Returns
    -------
    np.ndarray
        Array of shape (N, M) representing the PF points.
    """
    # Generate uniform points on simplex
    ref_dirs, _ = uniform_point(N, M)
    # ref_dirs = np.random.random((N, M))
    ref_dirs = ref_dirs / np.sum(ref_dirs, axis=1, keepdims=True)

    # Normalize to unit sphere
    R = ref_dirs / np.sqrt(np.sum(ref_dirs ** 2, axis=1, keepdims=True))

    # Scale by S
    S = 2 * np.arange(1, M + 1)
    R = R * S

    return R

WFG5_PF = WFG4_PF
WFG6_PF = WFG4_PF
WFG7_PF = WFG4_PF
WFG8_PF = WFG4_PF
WFG9_PF = WFG4_PF

# SETTINGS dictionary for data analysis
SETTINGS = {
    'metric': 'IGD',
    'n_ref': 2000,
    'WFG1': {'T1': WFG1_PF},
    'WFG2': {'T1': WFG2_PF},
    'WFG3': {'T1': WFG3_PF},
    'WFG4': {'T1': WFG4_PF},
    'WFG5': {'T1': WFG5_PF},
    'WFG6': {'T1': WFG6_PF},
    'WFG7': {'T1': WFG7_PF},
    'WFG8': {'T1': WFG8_PF},
    'WFG9': {'T1': WFG9_PF},
}
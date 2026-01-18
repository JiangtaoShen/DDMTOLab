import numpy as np
from Methods.mtop import MTOP
from Methods.Algo_Methods.uniform_point import uniform_point


class WFG:
    """
    Implementation of the WFG test suite for multi-objective optimization.

    The WFG (Walking Fish Group) test problems are a set of scalable,
    multi-objective benchmark problems with various properties including
    convex, concave, linear, and mixed Pareto fronts.

    Each method in this class generates a Multi-Task Optimization Problem (MTOP)
    instance containing a single WFG task.
    """

    def WFG1(self, M=3, dim=None) -> MTOP:
        """
        Generates the **WFG1** problem.

        WFG1 has a convex Pareto front and mixed separability.
        It uses several transformation functions including s_linear, b_flat, b_poly.

        Parameters
        ----------
        M : int, optional
            Number of objectives (default is 3).
        dim : int, optional
            Number of decision variables. If None, set to K + 10 where K = M-1 (default is None).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the WFG1 task.
        """
        # Position parameter, should be a multiple of (M-1)
        K = M - 1
        if dim is None:
            dim = K + 10

        L = dim - K  # Distance-related parameters

        def T1(x):
            """
            Calculate objective values for WFG1.

            Parameters
            ----------
            x : array-like
                Decision variables, shape (n_samples, dim)

            Returns
            -------
            array
                Objective values, shape (n_samples, M)
            """
            x = np.atleast_2d(x)
            n_samples = x.shape[0]

            # Step 1: Normalize decision variables to [0, 1]
            # Divide by 2:4:2*dim (MATLAB: 2:2:2*D)
            z = x / (2 * np.arange(1, dim + 1))

            # Step 2: Apply transformation functions
            # t1: first K variables unchanged, rest apply s_linear
            t1 = np.zeros((n_samples, dim))
            t1[:, :K] = z[:, :K]
            t1[:, K:] = s_linear(z[:, K:], 0.35)

            # t2: first K variables from t1, rest apply b_flat
            t2 = np.zeros((n_samples, dim))
            t2[:, :K] = t1[:, :K]
            t2[:, K:] = b_flat(t1[:, K:], 0.8, 0.75, 0.85)

            # t3: apply b_poly transformation
            t3 = b_poly(t2, 0.02)

            # Step 3: Weighted sum reduction
            t4 = np.zeros((n_samples, M))

            # For first M-1 objectives
            for i in range(M - 1):
                start_idx = int(i * K / (M - 1))
                end_idx = int((i + 1) * K / (M - 1))
                # Weights: 2*((i-1)*K/(M-1)+1):2:2*i*K/(M-1) in MATLAB
                weights = 2 * np.arange(start_idx + 1, end_idx + 1)
                t4[:, i] = r_sum(t3[:, start_idx:end_idx], weights)

            # Last objective uses the distance-related parameters
            # Weights: 2*(K+1):2:2*(K+L) in MATLAB
            weights = 2 * np.arange(K + 1, K + L + 1)
            t4[:, M - 1] = r_sum(t3[:, K:K + L], weights)

            # Step 4: Shape functions
            x_norm = np.zeros((n_samples, M))
            A = np.ones(M - 1)  # All ones for WFG1

            for i in range(M - 1):
                x_norm[:, i] = np.maximum(t4[:, M - 1], A[i]) * (t4[:, i] - 0.5) + 0.5
            x_norm[:, M - 1] = t4[:, M - 1]

            # Calculate shape functions
            h = convex_shape(x_norm)
            h[:, M - 1] = mixed_shape(x_norm)

            # Step 5: Final objectives
            D = 1
            S = 2 * np.arange(1, M + 1)  # 2:2:2*M in MATLAB
            pop_obj = D * x_norm[:, M - 1].reshape(-1, 1) + S.reshape(1, -1) * h

            return pop_obj

        # Lower and upper bounds
        # Lower bound: zeros
        # Upper bound: 2:2:2*dim (MATLAB: 2:2:2*D)
        lb = np.zeros(dim)
        ub = 2 * np.arange(1, dim + 1)

        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem


# --- WFG Transformation Functions ---

def s_linear(y, A):
    """
    s_linear transformation function.

    Parameters
    ----------
    y : array-like
        Input values, shape (n_samples, n_vars)
    A : float
        Parameter A

    Returns
    -------
    array
        Transformed values
    """
    y = np.atleast_2d(y)
    return np.abs(y - A) / np.abs(np.floor(A - y) + A)


def b_flat(y, A, B, C):
    """
    b_flat transformation function.

    Parameters
    ----------
    y : array-like
        Input values, shape (n_samples, n_vars)
    A, B, C : float
        Parameters

    Returns
    -------
    array
        Transformed values
    """
    y = np.atleast_2d(y)
    n_samples = y.shape[0]

    result = np.zeros_like(y)
    for i in range(n_samples):
        for j in range(y.shape[1]):
            y_val = y[i, j]

            # MATLAB: A+min(0,floor(y-B))*A.*(B-y)/B-min(0,floor(C-y))*(1-A).*(y-C)/(1-C)
            term1 = A
            term2 = min(0, np.floor(y_val - B)) * A * (B - y_val) / B
            term3 = min(0, np.floor(C - y_val)) * (1 - A) * (y_val - C) / (1 - C)

            result[i, j] = term1 + term2 - term3

    # Round to avoid numerical issues (MATLAB rounds to 4 decimal places)
    return np.round(result * 1e4) / 1e4


def b_poly(y, alpha):
    """
    b_poly transformation function.

    Parameters
    ----------
    y : array-like
        Input values, shape (n_samples, n_vars)
    alpha : float
        Exponent parameter

    Returns
    -------
    array
        Transformed values
    """
    return y ** alpha


def r_sum(y, w):
    """
    Weighted sum reduction function.

    Parameters
    ----------
    y : array-like
        Input values, shape (n_samples, n_vars)
    w : array-like
        Weights, shape (n_vars,) or scalar

    Returns
    -------
    array
        Weighted sum for each sample, shape (n_samples,)
    """
    y = np.atleast_2d(y)
    w = np.asarray(w)

    if w.ndim == 0:  # Scalar weight
        w = np.full(y.shape[1], w)

    # Ensure w has the right shape for broadcasting
    if w.shape[0] != y.shape[1]:
        raise ValueError(f"Weight dimension {w.shape[0]} doesn't match y dimension {y.shape[1]}")

    # Weighted sum: sum(y * w) / sum(w)
    weighted_sum = np.sum(y * w.reshape(1, -1), axis=1)
    total_weight = np.sum(w)

    return weighted_sum / total_weight


def convex_shape(x):
    """
    Convex shape function for WFG problems.

    Parameters
    ----------
    x : array-like
        Normalized variables, shape (n_samples, M)

    Returns
    -------
    array
        Convex shape values, shape (n_samples, M)
    """
    x = np.atleast_2d(x)
    M = x.shape[1]

    if M == 1:
        return np.ones((x.shape[0], 1))

    # MATLAB: fliplr(cumprod([ones(size(x,1),1),1-cos(x(:,1:end-1)*pi/2)],2)).*[ones(size(x,1),1),1-sin(x(:,end-1:-1:1)*pi/2)]

    # First part: cumprod of 1 - cos(x_i * pi/2) for i=1 to M-1
    cos_terms = 1 - np.cos(x[:, :M - 1] * np.pi / 2)
    cumprod_result = np.cumprod(np.hstack([np.ones((x.shape[0], 1)), cos_terms]), axis=1)

    # Flip left-right
    flipped = np.fliplr(cumprod_result)

    # Second part: 1 - sin(x_{M-i} * pi/2) for i=1 to M-1
    sin_terms = 1 - np.sin(np.fliplr(x[:, :M - 1]) * np.pi / 2)
    sin_part = np.hstack([np.ones((x.shape[0], 1)), sin_terms])

    return flipped * sin_part


def mixed_shape(x):
    """
    Mixed shape function for WFG1.

    Parameters
    ----------
    x : array-like
        Normalized variables, shape (n_samples, M)

    Returns
    -------
    array
        Mixed shape values, shape (n_samples,)
    """
    x = np.atleast_2d(x)
    # MATLAB: 1-x(:,1)-cos(10*pi*x(:,1)+pi/2)/10/pi
    return 1 - x[:, 0] - np.cos(10 * np.pi * x[:, 0] + np.pi / 2) / (10 * np.pi)


# --- Pareto Front (PF) Functions ---

def WFG1_PF(N: int, M: int) -> np.ndarray:
    """
    Computes the Pareto Front (PF) for WFG1.

    The PF is convex and computed using the algorithm from the original WFG paper.

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
    # 首先生成均匀点，注意uniform_point可能返回少于N的点
    W, actual_N = uniform_point(N, M, method='NBI')  # 使用NBI方法

    # 如果实际点数小于N，尝试其他方法或调整参数
    if actual_N < N:
        # 尝试生成更多点
        W2, actual_N2 = uniform_point(int(N * 1.5), M, method='grid')
        if actual_N2 >= N:
            W = W2
            actual_N = actual_N2
        # 如果还不够，直接复制已有的点
        elif actual_N < N:
            repeat_times = (N // actual_N) + 1
            W = np.tile(W, (repeat_times, 1))[:N]
            actual_N = N

    # 确保我们有足够的数据点
    if actual_N > N:
        # 随机选择N个点
        indices = np.random.choice(actual_N, N, replace=False)
        W = W[indices]
    elif actual_N < N:
        # 如果有缺失的点，用最后一个点填充
        missing = N - actual_N
        W = np.vstack([W, np.tile(W[-1:], (missing, 1))])

    # 现在W有N个点，继续计算
    # Step 1: 确保W都是正数（避免除以0）
    W = np.maximum(W, 1e-10)

    # Step 2: Initialize c matrix (all ones)
    c = np.ones((N, M))

    # Step 3: Calculate c values according to WFG1 PF formula
    for i in range(N):
        for j in range(2, M + 1):  # j from 2 to M
            # Calculate product term
            prod_term = 1.0
            start_idx = M - j + 1
            end_idx = M - 1

            if start_idx <= end_idx and start_idx >= 0 and end_idx < M:
                prod_term = np.prod(1 - c[i, start_idx:end_idx])

            # Calculate temp value
            # 注意：W[i, j-1]是第j个目标值，W[i, 0]是第一个目标值
            temp = (W[i, j - 1] / (W[i, 0] + 1e-10)) * prod_term

            # Update c value (adjust indices for 0-based)
            c_idx = M - j  # c(i, M-j+1) in 1-based -> c[i, M-j] in 0-based
            if temp <= 0:
                c[i, c_idx] = 0
            else:
                numerator = temp ** 2 - temp + np.sqrt(2 * temp)
                denominator = temp ** 2 + 1
                c[i, c_idx] = numerator / denominator

    # Step 4: Calculate x values from c
    # 限制c的范围在[-1, 1]之间，避免数值问题
    c_clipped = np.clip(c, -1 + 1e-10, 1 - 1e-10)
    x = np.arccos(c_clipped) * 2 / np.pi

    # Step 5: Additional optimization for x[:, 0] (first variable)
    if M >= 2:
        # 确保分母不为0
        safe_denominator = np.where(W[:, M - 2] == 0, 1e-10, W[:, M - 2])
        temp_val = (1 - np.sin(np.pi / 2 * x[:, 1])) * W[:, M - 1] / safe_denominator

        # Discretize search space for a
        a_grid = np.linspace(0, 1, 10001)  # 0:0.0001:1

        # Calculate error for each a value
        # E = |temp * (1-cos(pi/2*a)) - 1 + (a + cos(10*pi*a+pi/2)/(10*pi))|
        cos_term = np.cos(np.pi / 2 * a_grid)
        complex_term = a_grid + np.cos(10 * np.pi * a_grid + np.pi / 2) / (10 * np.pi)

        # Calculate error matrix
        temp_val_reshaped = temp_val.reshape(-1, 1)
        cos_term_reshaped = cos_term.reshape(1, -1)
        complex_term_reshaped = complex_term.reshape(1, -1)

        E = np.abs(temp_val_reshaped * (1 - cos_term_reshaped) -
                   1 + complex_term_reshaped)

        # Find the a value with minimum error for each point
        # Take the best candidate
        best_indices = np.argmin(E, axis=1)

        # Update x[:, 0] with optimal a values
        x[:, 0] = a_grid[best_indices]

    # Step 6: Apply shape functions
    h = convex_shape(x)
    h[:, M - 1] = mixed_shape(x)

    # Step 7: Scale by S = 2:2:2*M
    S = 2 * np.arange(1, M + 1)  # [2, 4, 6, ..., 2*M]

    # Final PF points
    pf_points = S.reshape(1, -1) * h

    return pf_points


SETTINGS = {
    'metric': 'IGD',
    'n_ref': 10000,
    'WFG1': {'T1': WFG1_PF},
}
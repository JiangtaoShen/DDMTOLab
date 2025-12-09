"""
UniformPoint - Generate a set of uniformly distributed points.

Reference:
[1] Y. Tian, X. Xiang, X. Zhang, R. Cheng, and Y. Jin. Sampling reference points on the Pareto fronts of benchmark
    multi-objective optimization problems. Proceedings of the IEEE Congress on Evolutionary Computation, 2018.
[2] T. Takagi, K. Takadama, and H. Sato. Incremental lattice design of weight vector set. Proceedings of the Genetic
    and Evolutionary Computation Conference Companion, 2020, 1486-1494.
"""

import numpy as np
from scipy.special import comb
from itertools import combinations
from math import gcd


def uniform_point(N: int, M: int, method: str = 'NBI') -> tuple[np.ndarray, int]:
    """
    Generate a set of uniformly distributed points.

    Parameters
    ----------
    N : int
        Approximate number of points to generate
    M : int
        Number of objectives/dimensions
    method : str, optional
        Sampling method to use (default: 'NBI'). Options:
        - 'NBI': Normal-boundary intersection method
        - 'ILD': Incremental lattice design
        - 'MUD': Mixture uniform design
        - 'grid': Grid sampling
        - 'Latin': Latin hypercube sampling

    Returns
    -------
    W : np.ndarray
        Array of uniformly distributed points, shape (N_actual, M)
    N_actual : int
        Actual number of points generated
    """
    method_map = {
        'NBI': nbi_method,
        'ILD': ild_method,
        'MUD': mud_method,
        'grid': grid_method,
        'Latin': latin_method
    }

    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")

    return method_map[method](N, M)


def nbi_method(N: int, M: int) -> tuple[np.ndarray, int]:
    """
    Generate uniformly distributed points using Normal-Boundary Intersection method.

    This method uses a two-layer approach to generate approximately N uniformly
    distributed points on the unit hyperplane.

    Parameters
    ----------
    N : int
        Approximate number of points to generate
    M : int
        Number of dimensions (objectives)

    Returns
    -------
    W : np.ndarray
        Weight vectors, shape (n_points, M)
    n_points : int
        Actual number of points generated
    """
    # First layer
    H1 = 1
    while comb(H1 + M, M - 1, exact=True) <= N:
        H1 += 1

    # Generate first layer points
    W = np.array(list(combinations(range(1, H1 + M), M - 1)))
    W = W - np.tile(np.arange(M - 1), (W.shape[0], 1)) - 1
    W = np.column_stack([W, np.full(W.shape[0], H1)]) - np.column_stack([np.zeros((W.shape[0], 1)), W])
    W = W / H1

    # Second layer (if needed)
    if H1 < M:
        H2 = 0
        while comb(H1 + M - 1, M - 1, exact=True) + comb(H2 + M, M - 1, exact=True) <= N:
            H2 += 1

        if H2 > 0:
            W2 = np.array(list(combinations(range(1, H2 + M), M - 1)))
            W2 = W2 - np.tile(np.arange(M - 1), (W2.shape[0], 1)) - 1
            W2 = np.column_stack([W2, np.full(W2.shape[0], H2)]) - np.column_stack([np.zeros((W2.shape[0], 1)), W2])
            W2 = W2 / H2
            W = np.vstack([W, W2 / 2 + 1 / (2 * M)])

    W = np.maximum(W, 1e-6)
    return W, W.shape[0]


def ild_method(N: int, M: int) -> tuple[np.ndarray, int]:
    """
    Generate uniformly distributed points using Incremental Lattice Design method.

    Parameters
    ----------
    N : int
        Approximate number of points to generate
    M : int
        Number of dimensions (objectives)

    Returns
    -------
    W : np.ndarray
        Weight vectors normalized to sum to 1, shape (n_points, M)
    n_points : int
        Actual number of points generated
    """
    I = M * np.eye(M)
    W = np.zeros((1, M))
    edge_W = W.copy()

    while W.shape[0] < N:
        # Repeat edge_W for each dimension and add identity
        edge_W = np.repeat(edge_W, M, axis=0) + np.tile(I, (edge_W.shape[0], 1))

        # Remove duplicates and points with all positive coordinates
        edge_W = np.unique(edge_W, axis=0)
        edge_W = edge_W[np.min(edge_W, axis=1) == 0]

        W = np.vstack([W + 1, edge_W])

    # Normalize to sum to 1
    W = W / np.sum(W, axis=1, keepdims=True)
    W = np.maximum(W, 1e-6)

    return W, W.shape[0]


def mud_method(N: int, M: int) -> tuple[np.ndarray, int]:
    """
    Generate uniformly distributed points using Mixture Uniform Design method.

    Parameters
    ----------
    N : int
        Exact number of points to generate
    M : int
        Number of dimensions (objectives)

    Returns
    -------
    W : np.ndarray
        Weight vectors on the unit hyperplane, shape (N, M)
    N : int
        Number of points generated (same as input N)
    """
    X = good_lattice_point(N, M - 1) ** (1.0 / np.tile(np.arange(M - 1, 0, -1), (N, 1)))
    X = np.maximum(X, 1e-6)

    W = np.zeros((N, M))
    W[:, :M - 1] = (1 - X) * np.cumprod(X, axis=1) / X
    W[:, M - 1] = np.prod(X, axis=1)

    return W, N


def grid_method(N: int, M: int) -> tuple[np.ndarray, int]:
    """
    Generate uniformly distributed points using Grid Sampling method.

    Parameters
    ----------
    N : int
        Approximate number of points to generate
    M : int
        Number of dimensions (objectives)

    Returns
    -------
    W : np.ndarray
        Grid points in the unit hypercube, shape (n_points, M)
    n_points : int
        Actual number of points generated
    """
    gap = np.linspace(0, 1, int(np.ceil(N ** (1.0 / M))))

    # Create meshgrid for all dimensions
    grids = np.meshgrid(*[gap] * M, indexing='ij')
    W = np.column_stack([grid.ravel() for grid in grids])

    return W, W.shape[0]


def latin_method(N: int, M: int) -> tuple[np.ndarray, int]:
    """
    Generate randomly distributed points using Latin Hypercube Sampling method.

    Parameters
    ----------
    N : int
        Exact number of points to generate
    M : int
        Number of dimensions (objectives)

    Returns
    -------
    W : np.ndarray
        Randomly sampled points in the unit hypercube, shape (N, M)
    N : int
        Number of points generated (same as input N)
    """
    W = np.argsort(np.random.rand(N, M), axis=0)
    W = (np.random.rand(N, M) + W) / N

    return W, N


def good_lattice_point(N: int, M: int) -> np.ndarray:
    """
    Generate good lattice points for the MUD method.

    Parameters
    ----------
    N : int
        Number of points to generate
    M : int
        Number of dimensions

    Returns
    -------
    Data : np.ndarray
        Good lattice points normalized to [0, 1], shape (N, M)
    """
    # Find coprime numbers with N
    hm = [i for i in range(1, N + 1) if gcd(i, N) == 1]

    # Generate udt matrix
    udt = np.zeros((N, len(hm)), dtype=int)
    for i, h in enumerate(hm):
        udt[:, i] = np.mod(np.arange(1, N + 1) * h, N)
        udt[udt[:, i] == 0, i] = N

    n_combination = comb(len(hm), M, exact=True)

    if n_combination < 1e4:
        # Try all combinations
        combination_list = list(combinations(range(len(hm)), M))
        CD2 = np.zeros(len(combination_list))

        for i, combo in enumerate(combination_list):
            UT = udt[:, combo]
            CD2[i] = calc_cd2(UT)

        min_index = np.argmin(CD2)
        Data = udt[:, combination_list[min_index]]
    else:
        # Use power sequence approach
        CD2 = np.zeros(N)
        for i in range(N):
            UT = np.mod(np.outer(np.arange(1, N + 1), i ** np.arange(M)), N)
            UT[UT == 0] = N
            CD2[i] = calc_cd2(UT)

        min_index = np.argmin(CD2)
        Data = np.mod(np.outer(np.arange(1, N + 1), min_index ** np.arange(M)), N)
        Data[Data == 0] = N

    Data = (Data - 1) / (N - 1)
    return Data


def calc_cd2(UT: np.ndarray) -> float:
    """
    Calculate the Centered Discrepancy (CD2) criterion.

    Parameters
    ----------
    UT : np.ndarray
        Input matrix, shape (N, S)

    Returns
    -------
    CD2 : float
        Centered discrepancy value (lower is better)
    """
    N, S = UT.shape
    X = (2 * UT - 1) / (2 * N)

    # Calculate CS1
    CS1 = np.sum(np.prod(2 + np.abs(X - 0.5) - (X - 0.5) ** 2, axis=1))

    # Calculate CS2
    CS2 = 0
    for i in range(N):
        X_diff = np.abs(np.tile(X[i:i + 1, :], (N, 1)) - X)
        term = 1 + 0.5 * np.abs(np.tile(X[i:i + 1, :], (N, 1)) - 0.5) + 0.5 * np.abs(X - 0.5) - 0.5 * X_diff
        CS2 += np.sum(np.prod(term, axis=1))

    CD2 = (13.0 / 12.0) ** S - 2 ** (1 - S) / N * CS1 + 1 / (N ** 2) * CS2

    return CD2
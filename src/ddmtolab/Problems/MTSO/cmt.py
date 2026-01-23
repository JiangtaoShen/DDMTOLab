"""
CMT Constrained Function Library
==================================

This module contains all the constrained benchmark functions used in CMT problems.
Each function has both objective and constraint components.
"""

import numpy as np
from typing import Tuple
from typing import Optional
from ddmtolab.Methods.mtop import MTOP


# ==============================================================================
# Constrained Ackley Functions
# ==============================================================================

def C_Ackley1(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
              opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ackley function with Type 1 constraint.

    Constraint Type 1: Complex periodic constraint
    g(x) = sum(x^2 - 5000*cos(0.1*pi*x) - 4000) <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Ackley objective
    sum1 = np.sum(x ** 2, axis=1) / D
    sum2 = np.sum(np.cos(2 * np.pi * x), axis=1) / D
    Obj = -20 * np.exp(-0.2 * np.sqrt(sum1)) - np.exp(sum2) + 20 + np.e

    # Compute constraint (Type 1: complex periodic)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2 - 5000 * np.cos(0.1 * np.pi * x_con) - 4000, axis=1)
    g[g < 0] = 0
    Con = g

    return Obj, Con


def C_Ackley2(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
              opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ackley function with Type 2 constraint.

    Constraint Type 2: Spherical constraint
    g(x) = sum(x^2) - 100*D <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Ackley objective
    sum1 = np.sum(x ** 2, axis=1) / D
    sum2 = np.sum(np.cos(2 * np.pi * x), axis=1) / D
    Obj = -20 * np.exp(-0.2 * np.sqrt(sum1)) - np.exp(sum2) + 20 + np.e

    # Compute constraint (Type 2: spherical)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2, axis=1) - 100 * D
    g[g < 0] = 0
    Con = g

    return Obj, Con


# ==============================================================================
# Constrained Griewank Functions
# ==============================================================================

def C_Griewank1(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Griewank function with Type 1 constraint.

    Constraint Type 1: Complex periodic constraint
    g(x) = sum(x^2 - 5000*cos(0.1*pi*x) - 4000) <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Griewank objective
    sum1 = np.sum(x ** 2, axis=1)
    sum2 = np.ones(ps)
    for i in range(D):
        sum2 *= np.cos(x[:, i] / np.sqrt(i + 1))
    Obj = 1 + sum1 / 4000 - sum2

    # Compute constraint (Type 1: complex periodic)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = var - opt_con[:D]
    g = np.sum(x_con ** 2 - 5000 * np.cos(0.1 * np.pi * x_con) - 4000, axis=1)
    g[g < 0] = 0
    Con = g

    return Obj, Con


def C_Griewank2(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Griewank function with Type 2 constraint.

    Constraint Type 2: Spherical constraint
    g(x) = sum(x^2) - 100*D <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Griewank objective
    sum1 = np.sum(x ** 2, axis=1)
    sum2 = np.ones(ps)
    for i in range(D):
        sum2 *= np.cos(x[:, i] / np.sqrt(i + 1))
    Obj = 1 + sum1 / 4000 - sum2

    # Compute constraint (Type 2: spherical)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = var - opt_con[:D]
    g = np.sum(x_con ** 2, axis=1) - 100 * D
    g[g < 0] = 0
    Con = g

    return Obj, Con


# ==============================================================================
# Constrained Rastrigin Functions
# ==============================================================================

def C_Rastrigin1(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                 opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rastrigin function with Type 1 constraint.

    Constraint Type 1: Complex periodic constraint
    g(x) = sum(x^2 - 5000*cos(0.1*pi*x) - 4000) <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Rastrigin objective
    Obj = 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)

    # Compute constraint (Type 1: complex periodic)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2 - 5000 * np.cos(0.1 * np.pi * x_con) - 4000, axis=1)
    g[g < 0] = 0
    Con = g

    return Obj, Con


def C_Rastrigin2(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                 opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rastrigin function with Type 2 constraint.

    Constraint Type 2: Spherical constraint
    g(x) = sum(x^2) - 100*D <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Rastrigin objective
    Obj = 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)

    # Compute constraint (Type 2: spherical)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2, axis=1) - 100 * D
    g[g < 0] = 0
    Con = g

    return Obj, Con


def C_Rastrigin4(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                 opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rastrigin function with Type 4 constraint (equality-like).

    Constraint Type 4: Equality-like constraint
    h(x) = |sum(-x * sin(0.1*pi*x))| - 1e-4 <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Rastrigin objective
    Obj = 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)

    # Compute constraint (Type 4: equality-like)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 2 * (var - opt_con[:D])
    h = -np.sum(x_con * np.sin(0.1 * np.pi * x_con), axis=1)
    h = np.abs(h) - 1e-4
    h[h < 0] = 0
    Con = h

    return Obj, Con


# ==============================================================================
# Constrained Rosenbrock Functions
# ==============================================================================

def C_Rosenbrock1(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                  opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rosenbrock function with Type 1 constraint.

    Constraint Type 1: Complex periodic constraint
    g(x) = sum(x^2 - 5000*cos(0.1*pi*x) - 4000) <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Rosenbrock objective
    if D == 1:
        Obj = 100 * (x[:, 0] - x[:, 0] ** 2) ** 2 + (x[:, 0] - 1) ** 2
    else:
        Obj = np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 +
                     (x[:, :-1] - 1) ** 2, axis=1)

    # Compute constraint (Type 1: complex periodic)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2 - 5000 * np.cos(0.1 * np.pi * x_con) - 4000, axis=1)
    g[g < 0] = 0
    Con = g

    return Obj, Con


def C_Rosenbrock2(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                  opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rosenbrock function with Type 2 constraint.

    Constraint Type 2: Spherical constraint
    g(x) = sum(x^2) - 100*D <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Rosenbrock objective
    if D == 1:
        Obj = 100 * (x[:, 0] - x[:, 0] ** 2) ** 2 + (x[:, 0] - 1) ** 2
    else:
        Obj = np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 +
                     (x[:, :-1] - 1) ** 2, axis=1)

    # Compute constraint (Type 2: spherical)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2, axis=1) - 100 * D
    g[g < 0] = 0
    Con = g

    return Obj, Con


# ==============================================================================
# Constrained Schwefel Functions
# ==============================================================================

def C_Schwefel1(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Schwefel function with Type 1 constraint.

    Constraint Type 1: Complex periodic constraint (scaled by 0.2)
    g(x) = sum(x^2 - 5000*cos(0.1*pi*x) - 4000) <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Schwefel objective
    sum1 = np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    Obj = 418.9829 * D - sum1

    # Compute constraint (Type 1: complex periodic, scaled by 0.2)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 0.2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2 - 5000 * np.cos(0.1 * np.pi * x_con) - 4000, axis=1)
    g[g < 0] = 0
    Con = g

    return Obj, Con


def C_Schwefel2(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Schwefel function with Type 2 constraint.

    Constraint Type 2: Spherical constraint (scaled by 0.2)
    g(x) = sum(x^2) - 100*D <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Schwefel objective
    sum1 = np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    Obj = 418.9829 * D - sum1

    # Compute constraint (Type 2: spherical, scaled by 0.2)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 0.2 * (var - opt_con[:D])
    g = np.sum(x_con ** 2, axis=1) - 100 * D
    g[g < 0] = 0
    Con = g

    return Obj, Con


# ==============================================================================
# Constrained Sphere Function
# ==============================================================================

def C_Sphere1(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
              opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sphere function with Type 1 constraint.

    Constraint Type 1: Complex periodic constraint
    g(x) = sum(x^2 - 5000*cos(0.1*pi*x) - 4000) <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples,).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Sphere objective
    Obj = np.sum(x ** 2, axis=1)

    # Compute constraint (Type 1: complex periodic)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = var - opt_con[:D]
    g = np.sum(x_con ** 2 - 5000 * np.cos(0.1 * np.pi * x_con) - 4000, axis=1)
    g[g < 0] = 0
    Con = g

    return Obj, Con


# ==============================================================================
# Constrained Weierstrass Function
# ==============================================================================

def C_Weierstrass3(var: np.ndarray, M: np.ndarray, opt: np.ndarray,
                   opt_con: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weierstrass function with Type 3 constraint (two constraints).

    Constraint Type 3: Two constraints
    g1(x) = -sum(|x|) + 12*D <= 0
    g2(x) = sum(x^2) - 500*D <= 0

    Parameters
    ----------
    var : np.ndarray
        Design variables, shape (n_samples, dim).
    M : np.ndarray
        Rotation matrix, shape (dim, dim) or (1, dim).
    opt : np.ndarray
        Shift vector for objective, shape (dim,) or (1, dim).
    opt_con : np.ndarray
        Shift vector for constraint, shape (dim,) or (1, dim).

    Returns
    -------
    Obj : np.ndarray
        Objective values, shape (n_samples,).
    Con : np.ndarray
        Constraint violations, shape (n_samples, 2).
    """
    var = np.atleast_2d(var)
    ps, D = var.shape

    # Handle rotation matrix
    if M.size == 1 or (M.ndim == 2 and M.shape[0] == 1):
        M = np.eye(D) if M.size == 1 else float(M[0, 0]) * np.eye(D)

    # Handle shift vectors
    if opt.ndim == 1 or opt.shape[0] == 1:
        opt = np.atleast_1d(opt).flatten()
        if opt.size == 1:
            opt = np.full(D, opt[0])

    # Apply rotation and shift for objective
    x = (M[:D, :D] @ (var - opt[:D]).T).T

    # Compute Weierstrass objective
    a = 0.5
    b = 3
    kmax = 20
    Obj = np.zeros(ps)

    for i in range(D):
        for k in range(kmax + 1):
            Obj += a ** k * np.cos(2 * np.pi * b ** k * (x[:, i] + 0.5))

    for k in range(kmax + 1):
        Obj -= D * a ** k * np.cos(2 * np.pi * b ** k * 0.5)

    # Compute constraints (Type 3: two constraints, scaled by 200)
    if opt_con.ndim == 1 or opt_con.shape[0] == 1:
        opt_con = np.atleast_1d(opt_con).flatten()
        if opt_con.size == 1:
            opt_con = np.full(D, opt_con[0])

    x_con = 200 * (var - opt_con[:D])
    g1 = -np.sum(np.abs(x_con), axis=1) + 12 * D
    g2 = np.sum(x_con ** 2, axis=1) - 500 * D

    g1[g1 < 0] = 0
    g2[g2 < 0] = 0

    Con = np.column_stack([g1, g2])

    return Obj, Con



class CMT:
    def __init__(self, default_dim: int = 50):
        self.default_dim = default_dim

    def CMT1(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = np.zeros(dim)
        opt_con1 = -40 * np.ones(dim)

        M2 = np.array([[1.0]])
        opt2 = np.zeros(dim)
        opt_con2 = 20 * np.ones(dim)

        def task1_func(x):
            obj, con = C_Griewank1(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Rastrigin1(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-100,
            upper_bound=100
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-50,
            upper_bound=50
        )

        return problem

    def CMT2(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = np.zeros(dim)
        opt_con1 = -4 * np.ones(dim)

        M2 = np.array([[1.0]])
        opt2 = np.zeros(dim)
        opt_con2 = 4 * np.ones(dim)

        def task1_func(x):
            obj, con = C_Ackley2(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Rastrigin2(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-50,
            upper_bound=50
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-50,
            upper_bound=50
        )

        return problem

    def CMT3(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = 42.096 * np.ones(dim)
        opt_con1 = 40 * np.ones(dim)

        M2 = np.array([[1.0]])
        opt2 = np.zeros(dim)
        opt_con2 = 400 * np.ones(dim)

        def task1_func(x):
            obj, con = C_Ackley2(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Schwefel1(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-50,
            upper_bound=50
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-500,
            upper_bound=500
        )

        return problem

    def CMT4(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = np.zeros(dim)
        opt_con1 = -20 * np.ones(dim)

        M2 = np.array([[1.0]])
        opt2 = np.zeros(dim)
        opt_con2 = 30 * np.ones(dim)

        def task1_func(x):
            obj, con = C_Rastrigin1(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Sphere1(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-50,
            upper_bound=50
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-100,
            upper_bound=100
        )

        return problem

    def CMT5(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = np.zeros(dim)
        opt_con1 = -30 * np.ones(dim)

        M2 = np.array([[1.0]])
        opt2 = np.zeros(dim)
        opt_con2 = np.zeros(dim)

        def task1_func(x):
            obj, con = C_Ackley1(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Rosenbrock2(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-50,
            upper_bound=50
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-50,
            upper_bound=50
        )

        return problem

    def CMT6(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = 2 * np.ones(dim)
        opt_con1 = np.zeros(dim)

        M2 = np.array([[1.0]])
        opt2 = 0.1 * np.ones(dim)
        opt_con2 = np.zeros(dim)

        def task1_func(x):
            obj, con = C_Ackley2(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Weierstrass3(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-50,
            upper_bound=50
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-0.5,
            upper_bound=0.5
        )

        return problem

    # 添加到 cmt.py 中的 CMT 类

    def CMT7(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = -30 * np.ones(dim)
        opt_con1 = -35 * np.ones(dim)

        M2 = np.array([[1.0]])
        opt2 = 35 * np.ones(dim)
        opt_con2 = 40 * np.ones(dim)

        def task1_func(x):
            obj, con = C_Rosenbrock1(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Rastrigin1(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-50,
            upper_bound=50
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-50,
            upper_bound=50
        )

        return problem

    def CMT8(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = np.zeros(dim)
        opt_con1 = -30 * np.ones(dim)

        M2 = np.array([[1.0]])
        opt2 = np.zeros(dim)
        opt_con2 = 0.2 * np.ones(dim)

        def task1_func(x):
            obj, con = C_Griewank2(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Weierstrass3(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-100,
            upper_bound=100
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-0.5,
            upper_bound=0.5
        )

        return problem


    def CMT9(self, dim: Optional[int] = None) -> MTOP:
        if dim is None:
            dim = self.default_dim

        M1 = np.array([[1.0]])
        opt1 = -10 * np.ones(dim)
        opt_con1 = np.zeros(dim)

        M2 = np.array([[1.0]])
        opt2 = np.zeros(dim)
        opt_con2 = 100 * np.ones(dim)

        def task1_func(x):
            obj, con = C_Rastrigin4(x, M1, opt1, opt_con1)
            return obj, con

        def task2_func(x):
            obj, con = C_Schwefel2(x, M2, opt2, opt_con2)
            return obj, con

        def task1_objective(x):
            obj, _ = task1_func(x)
            return obj

        def task1_constraint(x):
            _, con = task1_func(x)
            return con

        def task2_objective(x):
            obj, _ = task2_func(x)
            return obj

        def task2_constraint(x):
            _, con = task2_func(x)
            return con

        problem = MTOP()
        problem.add_task(
            objective_func=task1_objective,
            dim=dim,
            constraint_func=task1_constraint,
            lower_bound=-50,
            upper_bound=50
        )
        problem.add_task(
            objective_func=task2_objective,
            dim=dim,
            constraint_func=task2_constraint,
            lower_bound=-500,
            upper_bound=500
        )

        return problem
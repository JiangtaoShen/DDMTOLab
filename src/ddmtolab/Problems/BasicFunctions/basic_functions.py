"""
Base functions for single objective tasks.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.15
Version: 1.0

References:
[1] Da, Bingshui, et al. "Evolutionary multitasking for single-objective continuous optimization: Benchmark problems, performance metric, and baseline results." arXiv preprint arXiv:1706.03470 (2017).
"""
import numpy as np


def _rotate_and_shift(var, M, opt):
    """
    Apply the shift then the rotation, the way the MToP base functions do.

    Mirrors the three steps every ``Base/*.m`` in MToP performs before it
    evaluates a landscape::

        if size(M, 1) == 1, M = M * eye(D); end
        if size(opt, 2) == 1, opt = opt * ones(1, D); end
        var = (M(1:D, 1:D) * (var - repmat(opt(1:D), ps, 1))')';

    so a scalar rotation or offset broadcasts to the task dimension and an
    oversized rotation matrix is cropped to it. Passing a full ``(D, D)``
    matrix and a ``(1, D)`` offset, as every shipped suite does, leaves the
    result bit for bit what it was.

    Parameters
    ----------
    var : np.ndarray
        Decision variables, shape (n_samples, D).
    M : np.ndarray or float
        Rotation matrix, at least (D, D), or a scalar standing for M * I.
    opt : np.ndarray or float
        Shift vector, at least (D,), or a scalar standing for opt * ones(D).

    Returns
    -------
    np.ndarray
        The rotated, shifted variables, shape (n_samples, D).
    """
    D = var.shape[1]
    M = np.asarray(M, dtype=float)
    opt = np.asarray(opt, dtype=float)
    if M.size == 1:
        M = float(M) * np.eye(D)
    if opt.size == 1:
        opt = float(opt) * np.ones(D)
    return (M[:D, :D] @ (var - np.ravel(opt)[:D]).T).T



def Ackley(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    sum1 = np.sum(var ** 2, axis=1)
    sum2 = np.sum(np.cos(2 * np.pi * var), axis=1)
    avgsum1 = sum1 / D
    avgsum2 = sum2 / D
    Obj = -20 * np.exp(-0.2 * np.sqrt(avgsum1)) - np.exp(avgsum2) + 20 + np.exp(1) + g
    return Obj.reshape(-1, 1)

def Elliptic(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    a = 1e+6
    Obj = np.zeros((ps, 1))
    if D == 1:
        Obj = a * var**2
    else:
        for i in range(D):
            Obj = Obj + (a**((i) / (D - 1))) * (var[:, i]**2).reshape(-1, 1)
    Obj = Obj + g
    return Obj.reshape(-1, 1)

def Griewank(var: np.ndarray, M: np.ndarray, opt: np.ndarray, g: float ) -> np.ndarray:
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    sum1 = np.sum(var ** 2, axis=1)
    i = np.arange(1, D + 1)
    sum2 = np.prod(np.cos(var / np.sqrt(i)), axis=1)
    Obj = 1 + (1 / 4000) * sum1 - sum2 + g
    return Obj.reshape(-1, 1)

def Rastrigin(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    rastrigin_sum = np.sum(var ** 2 - 10 * np.cos(2 * np.pi * var), axis=1)
    Obj = 10 * D + rastrigin_sum + g
    return Obj.reshape(-1, 1)

def Rosenbrock(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    sum1 = np.zeros((ps, 1))
    if D == 1:
        sum1 = 100 * (var[:, 0] - var[:, 0]**2)**2 + (var[:, 0] - 1)**2
        if sum1.ndim == 1:
            sum1 = sum1.reshape(-1, 1)
    else:
        for ii in range(D - 1):
            xi = var[:, ii]
            xnext = var[:, ii + 1]
            new = 100 * (xnext - xi**2)**2 + (xi - 1)**2
            sum1 = sum1 + new.reshape(-1, 1)
    Obj = sum1 + g
    return Obj.reshape(-1, 1)

def Schwefel(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    sum1 = np.sum(var * np.sin(np.sqrt(np.abs(var))), axis=1)
    Obj = 418.9829 * D - sum1
    Obj = Obj + g
    return Obj.reshape(-1, 1)

def Schwefel2(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    Obj = np.zeros(ps)
    for i in range(D):
        Obj += np.sum(var[:, :i+1], axis=1)**2
    Obj = Obj + g
    return Obj.reshape(-1, 1)

def Sphere(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    Obj = np.sum(var**2, axis=1)
    Obj = Obj + g
    return Obj.reshape(-1, 1)

def Weierstrass(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = _rotate_and_shift(var, M, opt)
    a = 0.5
    b = 3
    kmax = 20
    Obj = np.zeros((ps, 1))
    for i in range(D):
        for k in range(kmax + 1):
            Obj = Obj + a ** k * np.cos(2 * np.pi * b ** k * (var[:, i].reshape(-1, 1) + 0.5))
    for k in range(kmax + 1):
        Obj = Obj - D * a ** k * np.cos(2 * np.pi * b ** k * 0.5)
    Obj = Obj + g
    return Obj.reshape(-1, 1)

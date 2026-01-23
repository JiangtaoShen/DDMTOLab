"""
Base functions for single objective tasks.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.15
Version: 1.0

References:
[1] Da, Bingshui, et al. "Evolutionary multitasking for single-objective continuous optimization: Benchmark problems,
    performance metric, and baseline results." arXiv preprint arXiv:1706.03470 (2017).
"""
import numpy as np


def Ackley(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = (M @ (var - opt).T).T
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
    var = (M @ (var - opt).T).T
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
    var = (M @ (var - opt).T).T
    sum1 = np.sum(var ** 2, axis=1)
    i = np.arange(1, D + 1)
    sum2 = np.prod(np.cos(var / np.sqrt(i)), axis=1)
    Obj = 1 + (1 / 4000) * sum1 - sum2 + g
    return Obj.reshape(-1, 1)

def Rastrigin(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = (M @ (var - opt).T).T
    rastrigin_sum = np.sum(var ** 2 - 10 * np.cos(2 * np.pi * var), axis=1)
    Obj = 10 * D + rastrigin_sum + g
    return Obj.reshape(-1, 1)

def Rosenbrock(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = (M @ (var - opt).T).T
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
    var = (M @ (var - opt).T).T
    sum1 = np.sum(var * np.sin(np.sqrt(np.abs(var))), axis=1)
    Obj = 418.9829 * D - sum1
    Obj = Obj + g
    return Obj.reshape(-1, 1)

def Schwefel2(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = (M @ (var - opt).T).T
    Obj = np.zeros(ps)
    for i in range(D):
        Obj += np.sum(var[:, :i+1], axis=1)**2
    Obj = Obj + g
    return Obj.reshape(-1, 1)

def Sphere(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = (M @ (var - opt).T).T
    Obj = np.sum(var**2, axis=1)
    Obj = Obj + g
    return Obj.reshape(-1, 1)

def Weierstrass(var, M, opt, g):
    if var.ndim != 2:
        raise ValueError("Input 'var' must be a 2D array: (n_samples, n_features)")
    ps, D = var.shape
    var = (M @ (var - opt).T).T
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
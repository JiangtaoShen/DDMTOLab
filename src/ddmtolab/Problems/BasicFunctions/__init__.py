"""Shared landscape primitives used to build the synthetic problem suites.

Every function has the same signature ``f(var, M, opt, g)``:

===========  ==========================================================
``var``      decision variables, shape ``(n_samples, D)``
``M``        rotation matrix, shape ``(D, D)``
``opt``      shift (global optimum) vector, shape ``(1, D)`` or ``(D,)``
``g``        constant bias added to the objective
===========  ==========================================================

and returns objective values of shape ``(n_samples, 1)``.

Examples
--------
>>> import numpy as np
>>> from ddmtolab.Problems.BasicFunctions import Sphere
>>> Sphere(np.zeros((1, 3)), np.eye(3), np.zeros(3), 0.0)
array([[0.]])
"""

from ddmtolab.Problems.BasicFunctions.basic_functions import (
    Ackley, Elliptic, Griewank, Rastrigin, Rosenbrock,
    Schwefel, Schwefel2, Sphere, Weierstrass,
)

__all__ = [
    "Ackley", "Elliptic", "Griewank", "Rastrigin", "Rosenbrock",
    "Schwefel", "Schwefel2", "Sphere", "Weierstrass",
]

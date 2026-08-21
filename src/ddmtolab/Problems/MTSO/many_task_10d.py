"""
Many-Task Optimization Benchmark Problems (10D).

4 problems with 3 or 5 tasks each, all 10-dimensional.

Problem Overview
----------------
- P1 (3 tasks): Griewank / Rastrigin / Ackley, same optimum
- P2 (3 tasks): Rosenbrock / Rastrigin / Sphere, close optima
- P3 (5 tasks): Ackley / Rastrigin / Griewank, same optimum with different rotations
- P4 (5 tasks): Sphere / Rastrigin / Ackley, grouped similar tasks with slightly shifted optima

"""
import numpy as np
from ddmtolab.Problems.BasicFunctions.basic_functions import (
    Ackley, Griewank, Rastrigin, Rosenbrock, Sphere,
)
from ddmtolab.Methods.mtop import MTOP


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random orthogonal matrix, Haar distributed, via QR decomposition."""
    H = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(H)
    # Fixing the sign of R's diagonal makes the decomposition unique, which is
    # what makes Q Haar distributed. It does not force det(Q) = +1: half the
    # draws are reflections. Either way Q is an isometry, so the landscape it
    # induces is a rigid transform of the base function.
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


class ManyTask_10D:
    """
    Many-Task Optimization Benchmark (10D).

    Contains 4 problems:
    - P1, P2: 3 tasks each
    - P3, P4: 5 tasks each

    All tasks are 10-dimensional single-objective unconstrained problems.

    Attributes
    ----------
    problem_information : dict
        Metadata about the benchmark suite.
    """

    problem_information = {
        'n_cases': 4,
        'n_tasks': '[3, 5]',
        'n_dims': '10',
        'n_objs': '1',
        'n_cons': '0',
        'type': 'synthetic',
    }

    def P1(self) -> MTOP:
        """
        Problem 1 (3 tasks, high similarity).

        - T1: Griewank ([-100, 100]), optimum at origin
        - T2: Rastrigin ([-50, 50]), optimum at origin
        - T3: Ackley ([-50, 50]), optimum at origin

        All three tasks share the same global optimum at the origin,
        resulting in complete intersection and strong positive transfer.

        Returns
        -------
        MTOP
            A 3-task optimization problem.
        """
        I = np.eye(10, dtype=float)
        opt_zero = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Griewank(x, I, opt_zero, 0.)

        def T2(x):
            return Rastrigin(x, I, opt_zero, 0.)

        def T3(x):
            return Ackley(x, I, opt_zero, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10,
                         lower_bound=np.full(10, -100),
                         upper_bound=np.full(10, 100))
        problem.add_task(T2, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T3, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        return problem

    def P2(self) -> MTOP:
        """
        Problem 2 (3 tasks, high similarity).

        - T1: Rosenbrock ([-50, 50]), shifted to (1, ..., 1)
        - T2: Rastrigin ([-50, 50]), shifted to (1, ..., 1)
        - T3: Sphere ([-100, 100]), shifted to (1, ..., 1)

        All tasks have global optima near (1, ..., 1) in the original
        decision space, yielding close optima in the [0, 1] unified space.

        Returns
        -------
        MTOP
            A 3-task optimization problem.
        """
        I = np.eye(10, dtype=float)
        opt_ones = np.ones((1, 10), dtype=float)

        def T1(x):
            return Rosenbrock(x, I, opt_ones, 0.)

        def T2(x):
            return Rastrigin(x, I, opt_ones, 0.)

        def T3(x):
            return Sphere(x, I, opt_ones, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T3, dim=10,
                         lower_bound=np.full(10, -100),
                         upper_bound=np.full(10, 100))
        return problem

    def P3(self) -> MTOP:
        """
        Problem 3 (5 tasks, medium-high similarity).

        - T1: Ackley ([-50, 50]), rotated (R1), optimum at origin
        - T2: Ackley ([-50, 50]), rotated (R2), optimum at origin
        - T3: Rastrigin ([-50, 50]), rotated (R3), optimum at origin
        - T4: Rastrigin ([-50, 50]), rotated (R4), optimum at origin
        - T5: Griewank ([-100, 100]), identity rotation, optimum at origin

        All five tasks share the same global optimum at the origin.
        Different rotation matrices create landscape diversity while
        preserving complete intersection of optima.

        Returns
        -------
        MTOP
            A 5-task optimization problem.
        """
        rng = np.random.default_rng(seed=42)
        R1 = _random_rotation(10, rng)
        R2 = _random_rotation(10, rng)
        R3 = _random_rotation(10, rng)
        R4 = _random_rotation(10, rng)
        I = np.eye(10, dtype=float)
        opt_zero = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Ackley(x, R1, opt_zero, 0.)

        def T2(x):
            return Ackley(x, R2, opt_zero, 0.)

        def T3(x):
            return Rastrigin(x, R3, opt_zero, 0.)

        def T4(x):
            return Rastrigin(x, R4, opt_zero, 0.)

        def T5(x):
            return Griewank(x, I, opt_zero, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T3, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T4, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T5, dim=10,
                         lower_bound=np.full(10, -100),
                         upper_bound=np.full(10, 100))
        return problem

    def P4(self) -> MTOP:
        """
        Problem 4 (5 tasks, medium similarity).

        - T1: Sphere ([-100, 100]), optimum at origin
        - T2: Sphere ([-100, 100]), optimum at (2, ..., 2)
        - T3: Rastrigin ([-50, 50]), optimum at origin
        - T4: Rastrigin ([-50, 50]), optimum at (2, ..., 2)
        - T5: Ackley ([-50, 50]), optimum at (1, ..., 1)

        Tasks form two groups (Sphere pair, Rastrigin pair) with an Ackley
        task bridging them. Within each group, optima are close in the
        [0, 1] unified space (0.50 vs 0.51 for Sphere, 0.50 vs 0.52 for
        Rastrigin), encouraging selective knowledge transfer.

        Returns
        -------
        MTOP
            A 5-task optimization problem.
        """
        I = np.eye(10, dtype=float)
        opt_zero = np.zeros((1, 10), dtype=float)
        opt_two = 2.0 * np.ones((1, 10), dtype=float)
        opt_one = np.ones((1, 10), dtype=float)

        def T1(x):
            return Sphere(x, I, opt_zero, 0.)

        def T2(x):
            return Sphere(x, I, opt_two, 0.)

        def T3(x):
            return Rastrigin(x, I, opt_zero, 0.)

        def T4(x):
            return Rastrigin(x, I, opt_two, 0.)

        def T5(x):
            return Ackley(x, I, opt_one, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10,
                         lower_bound=np.full(10, -100),
                         upper_bound=np.full(10, 100))
        problem.add_task(T2, dim=10,
                         lower_bound=np.full(10, -100),
                         upper_bound=np.full(10, 100))
        problem.add_task(T3, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T4, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        problem.add_task(T5, dim=10,
                         lower_bound=np.full(10, -50),
                         upper_bound=np.full(10, 50))
        return problem

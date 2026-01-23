import scipy.io
import pkgutil
import io
from ddmtolab.Problems.BasicFunctions.basic_functions import *
from ddmtolab.Methods.mtop import MTOP
import numpy as np

class CEC17MTSO_10D:
    """
    Implementation of the 10-Dimensional (10D) versions of the CEC 2017
    Multi-Task Optimization (MTSO) benchmark problems P1 to P9.

    These problems maintain the same underlying functions and global optima
    relationships as the original 50D CEC17 MTSO set but are configured with
    a reduced search space dimensionality (D=10) for both tasks.

    Attributes
    ----------
    data_dir : str
        The directory path for 10D problem data files.
    """

    def __init__(self):
        self.data_dir = 'data_cec17mtso_10d'

    def P1(self) -> MTOP:
        """
        Generates Problem 1 (10D): **T1: Griewank, T2: Rastrigin**.

        - T1: Griewank (Dim 10, [-100, 100]) - Standard
        - T2: Rastrigin (Dim 10, [-50, 50]) - Standard
        - Relationship: Global optima are at **origin (0)** for both tasks (Complete Intersection).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        rotation_task1 = np.eye(10, dtype=float)
        go_task1 = np.zeros((1, 10), dtype=float)
        rotation_task2 = np.eye(10, dtype=float)
        go_task2 = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Griewank(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rastrigin(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -100), upper_bound=np.full(10, 100))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        return problem

    def P2(self) -> MTOP:
        """
        Generates Problem 2 (10D): **T1: Rosenbrock, T2: Rastrigin**.

        - T1: Rosenbrock (Dim 10, [-50, 50]) - Shifted to **(1, ..., 1)**
        - T2: Rastrigin (Dim 10, [-50, 50]) - Shifted to **(1, ..., 1)**
        - Relationship: Global optima are at **(1, ..., 1)** for both tasks (Complete Intersection).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        rotation_task1 = np.eye(10, dtype=float)
        go_task1 = np.ones((1, 10), dtype=float)
        rotation_task2 = np.eye(10, dtype=float)
        go_task2 = np.ones((1, 10), dtype=float)

        def T1(x):
            return Rosenbrock(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rastrigin(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        return problem

    def P3(self) -> MTOP:
        """
        Generates Problem 3 (10D): **T1: Griewank, T2: Weierstrass**.

        - T1: Griewank (Dim 10, [-100, 100]) - Shifted to **(10, ..., 10)**
        - T2: Weierstrass (Dim 10, [-0.5, 0.5]) - Shifted to **(1, ..., 1)**
        - Relationship: Global optima are **misaligned** (No Intersection).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        rotation_task1 = np.eye(10, dtype=float)
        go_task1 = 10*np.ones((1, 10), dtype=float)
        rotation_task2 = np.eye(10, dtype=float)
        go_task2 = np.ones((1, 10), dtype=float)

        def T1(x):
            return Griewank(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Weierstrass(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -100), upper_bound=np.full(10, 100))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -0.5), upper_bound=np.full(10, 0.5))
        return problem

    def P4(self) -> MTOP:
        """
        Generates Problem 4 (10D): **T1: Ackley, T2: Rosenbrock**.

        - T1: Ackley (Dim 10, [-50, 50]) - Rotated and Shifted (Data loaded from P4.mat)
        - T2: Rosenbrock (Dim 10, [-50, 50]) - Standard (optimum at **origin** $\mathbf{0}$)
        - Relationship: **Partial Intersection** of global optima due to rotation/shift in T1.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/P4.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1_ld'].squeeze()
        go_task1 = mat_data['GO_Task1_ld'].squeeze()
        rotation_task2 = np.eye(10, dtype=float)
        go_task2 = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rosenbrock(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        return problem

    def P5(self) -> MTOP:
        """
        Generates Problem 5 (10D): **T1: Rastrigin, T2: Sphere**.

        - T1: Rastrigin (Dim 10, [-50, 50]) - Rotated and Shifted (Data loaded from P5.mat)
        - T2: Sphere (Dim 10, [-100, 100]) - Shifted (Data loaded from P5.mat)
        - Relationship: **Partial Intersection** of global optima.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/P5.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1_ld'].squeeze()
        go_task1 = mat_data['GO_Task1_ld'].squeeze()
        rotation_task2 = np.eye(10, dtype=float)
        go_task2 = mat_data['GO_Task2_ld'].squeeze()

        def T1(x):
            return Rastrigin(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Sphere(x, rotation_task2, go_task2, 0)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -100), upper_bound=np.full(10, 100))
        return problem

    def P6(self) -> MTOP:
        """
        Generates Problem 6 (10D): **T1: Ackley, T2: Rastrigin**.

        - T1: Ackley (Dim 10, [-50, 50]) - Rotated and Shifted (Data loaded from P6.mat)
        - T2: Rastrigin (Dim 10, [-50, 50]) - Rotated and Shifted (Data loaded from P6.mat)
        - Relationship: **Complete Intersection** of global optima, but different **rotations** (Data files define this).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/P6.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1_ld'].squeeze()
        go_task1 = mat_data['GO_Task1_ld'].squeeze()
        rotation_task2 = mat_data['Rotation_Task2_ld'].squeeze()
        go_task2 = mat_data['GO_Task2_ld'].squeeze()

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rastrigin(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        return problem

    def P7(self) -> MTOP:
        """
        Generates Problem 7 (10D): **T1: Ackley, T2: Schwefel**.

        - T1: Ackley (Dim 10, [-50, 50]) - Rotated and Shifted (Data loaded from P7.mat)
        - T2: Schwefel (Dim 10, [-500, 500]) - Standard (optimum at **origin** $\mathbf{0}$)
        - Relationship: **Partial Intersection** of global optima.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/P7.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1_ld'].squeeze()
        go_task1 = mat_data['GO_Task1_ld'].squeeze()
        rotation_task2 = np.eye(10, dtype=float)
        go_task2 = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Schwefel(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -500), upper_bound=np.full(10, 500))
        return problem

    def P8(self) -> MTOP:
        """
        Generates Problem 8 (10D): **T1: Ackley, T2: Weierstrass**.

        - T1: Ackley (Dim 10, [-50, 50]) - Rotated and Shifted (Data loaded from P8.mat)
        - T2: Weierstrass (Dim 10, [-0.5, 0.5]) - Rotated (Data loaded from P8.mat)
        - Relationship: **Partial Intersection** of global optima.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/P8.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1_ld'].squeeze()
        go_task1 = mat_data['GO_Task1_ld'].squeeze()
        rotation_task2 = 5*np.eye(10, dtype=float) # Note: Rotation matrix scaled by 5 in the original 10D data
        go_task2 = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Weierstrass(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -0.5), upper_bound=np.full(10, 0.5))
        return problem

    def P9(self) -> MTOP:
        """
        Generates Problem 9 (10D): **T1: Rastrigin, T2: Schwefel**.

        - T1: Rastrigin (Dim 10, [-50, 50]) - Rotated and Shifted (Data loaded from P9.mat)
        - T2: Schwefel (Dim 10, [-500, 500]) - Standard (optimum at **origin** $\mathbf{0}$)
        - Relationship: **Partial Intersection** of global optima.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the two tasks.
        """
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/P9.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1_ld'].squeeze()
        go_task1 = mat_data['GO_Task1_ld'].squeeze()
        rotation_task2 = np.eye(10, dtype=float)
        go_task2 = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Rastrigin(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Schwefel(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -500), upper_bound=np.full(10, 500))
        return problem
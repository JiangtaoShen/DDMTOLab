import pkgutil
import scipy.io
import io
from ddmtolab.Problems.BasicFunctions.basic_functions import *
from ddmtolab.Methods.mtop import MTOP
import numpy as np

class CEC17MTSO:
    """
    Implementation of the CEC 2017 Competition on Evolutionary Multi-Task Optimization (EMTO)
    benchmark problems P1 to P9.

    These problems are two-task optimization scenarios designed to test the ability
    of algorithms to leverage knowledge transfer under various relationships
    between tasks (similarity of global optima and search spaces).

    Attributes
    ----------
    mat_dir : str
        The directory path for problem data files.
    """

    def __init__(self):
        self.data_dir = 'data_cec17mtso'

    def P1(self) -> MTOP:
        """
        Generates Problem 1: **CI-HS** (Complete Intersection - High Similarity).

        - Task 1: Rotated and shifted **Griewank** (Dim 50, [-100, 100])
        - Task 2: Rotated and shifted **Rastrigin** (Dim 50, [-50, 50])
        - Characteristic: **Complete Overlap** of the global optima and **High Similarity**
          of the solution space structures (Griewank, Rastrigin are both multi-modal).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # CI-HS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/CI_H.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)

        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = mat_data['Rotation_Task2'].squeeze()
        go_task2 = mat_data['GO_Task2'].squeeze()

        def T1(x):
            return Griewank(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rastrigin(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -100), upper_bound=np.full(50, 100))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        return problem

    def P2(self) -> MTOP:
        """
        Generates Problem 2: **CI-MS** (Complete Intersection - Medium Similarity).

        - Task 1: Rotated and shifted **Ackley** (Dim 50, [-50, 50])
        - Task 2: Rotated and shifted **Rastrigin** (Dim 50, [-50, 50])
        - Characteristic: **Complete Overlap** of the global optima and **Medium Similarity**
          of the solution space structures (Ackley is generally smoother than Rastrigin).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # CI-MS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/CI_M.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = mat_data['Rotation_Task2'].squeeze()
        go_task2 = mat_data['GO_Task2'].squeeze()

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rastrigin(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        return problem

    def P3(self) -> MTOP:
        """
        Generates Problem 3: **CI-LS** (Complete Intersection - Low Similarity).

        - Task 1: Rotated and shifted **Ackley** (Dim 50, [-50, 50])
        - Task 2: Standard **Schwefel** (Dim 50, [-500, 500])
        - Characteristic: **Complete Overlap** of the global optima and **Low Similarity**
          of the solution space structures (Schwefel is very difficult, Ackley is moderate).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # CI-LS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/CI_L.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = np.eye(50, dtype=float)
        go_task2 = np.zeros((1, 50), dtype=float)

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Schwefel(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -500), upper_bound=np.full(50, 500))
        return problem

    def P4(self) -> MTOP:
        """
        Generates Problem 4: **PI-HS** (Partial Intersection - High Similarity).

        - Task 1: Rotated and shifted **Rastrigin** (Dim 50, [-50, 50])
        - Task 2: Shifted **Sphere** (Dim 50, [-100, 100])
        - Characteristic: **Partial Overlap** of the global optima and **High Similarity**
          of the solution space structures (Rastrigin is multi-modal, Sphere is uni-modal).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # PI-HS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/PI_H.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = np.eye(50, dtype=float)
        go_task2 = mat_data['GO_Task2'].squeeze()

        def T1(x):
            return Rastrigin(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Sphere(x, rotation_task2, go_task2, 0)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -100), upper_bound=np.full(50, 100))
        return problem

    def P5(self) -> MTOP:
        """
        Generates Problem 5: **PI-MS** (Partial Intersection - Medium Similarity).

        - Task 1: Rotated and shifted **Ackley** (Dim 50, [-50, 50])
        - Task 2: Standard **Rosenbrock** (Dim 50, [-50, 50])
        - Characteristic: **Partial Overlap** of the global optima and **Medium Similarity**
          of the solution space structures (Ackley is multi-modal, Rosenbrock is uni-modal and valley-shaped).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # PI-MS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/PI_M.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = np.eye(50, dtype=float)
        go_task2 = np.zeros((1, 50), dtype=float)

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rosenbrock(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        return problem

    def P6(self) -> MTOP:
        """
        Generates Problem 6: **PI-LS** (Partial Intersection - Low Similarity).

        - Task 1: Rotated and shifted **Ackley** (Dim 50, [-50, 50])
        - Task 2: Rotated and shifted **Weierstrass** (Dim 25, [-0.5, 0.5])
        - Characteristic: **Partial Overlap** of the global optima, **Unequal Dimensions** (50 vs 25),
          and **Low Similarity** (Weierstrass is highly complex and non-differentiable).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # PI-LS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/PI_L.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = mat_data['Rotation_Task2'].squeeze()
        go_task2 = mat_data['GO_Task2'].squeeze()

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Weierstrass(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        problem.add_task(T2, dim=25, lower_bound=np.full(25, -0.5), upper_bound=np.full(25, 0.5))
        return problem

    def P7(self) -> MTOP:
        """
        Generates Problem 7: **NI-HS** (No Intersection - High Similarity).

        - Task 1: Standard **Rosenbrock** (Dim 50, [-50, 50])
        - Task 2: Rotated and shifted **Rastrigin** (Dim 50, [-50, 50])
        - Characteristic: **No Overlap** of the global optima and **High Similarity**
          of the solution space structures.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # NI-HS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/NI_H.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = np.eye(50, dtype=float)
        go_task1 = np.zeros((1, 50), dtype=float)
        rotation_task2 = mat_data['Rotation_Task2'].squeeze()
        go_task2 = mat_data['GO_Task2'].squeeze()

        def T1(x):
            return Rosenbrock(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Rastrigin(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        return problem

    def P8(self) -> MTOP:
        """
        Generates Problem 8: **NI-MS** (No Intersection - Medium Similarity).

        - Task 1: Rotated and shifted **Griewank** (Dim 50, [-100, 100])
        - Task 2: Rotated and shifted **Weierstrass** (Dim 50, [-0.5, 0.5])
        - Characteristic: **No Overlap** of the global optima and **Medium Similarity**
          of the solution space structures.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # NI-MS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/NI_M.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = mat_data['Rotation_Task2'].squeeze()
        go_task2 = mat_data['GO_Task2'].squeeze()

        def T1(x):
            return Griewank(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Weierstrass(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -100), upper_bound=np.full(50, 100))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -0.5), upper_bound=np.full(50, 0.5))
        return problem

    def P9(self) -> MTOP:
        """
        Generates Problem 9: **NI-LS** (No Intersection - Low Similarity).

        - Task 1: Rotated and shifted **Rastrigin** (Dim 50, [-50, 50])
        - Task 2: Standard **Schwefel** (Dim 50, [-500, 500])
        - Characteristic: **No Overlap** of the global optima and **Low Similarity**
          of the solution space structures (Rastrigin is multi-modal, Schwefel is highly complex/difficult).

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing Task 1 and Task 2.
        """
        # NI-LS
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO', f'{self.data_dir}/NI_L.mat')
        mat_file = io.BytesIO(data_bytes)
        mat_data = scipy.io.loadmat(mat_file)
        rotation_task1 = mat_data['Rotation_Task1'].squeeze()
        go_task1 = mat_data['GO_Task1'].squeeze()
        rotation_task2 = np.eye(50, dtype=float)
        go_task2 = np.zeros((1, 50), dtype=float)

        def T1(x):
            return Rastrigin(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Schwefel(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=50, lower_bound=np.full(50, -50), upper_bound=np.full(50, 50))
        problem.add_task(T2, dim=50, lower_bound=np.full(50, -500), upper_bound=np.full(50, 500))
        return problem
import scipy.io
from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP
import numpy as np

class CEC19MaTSO:
    """
    Implementation of the CEC 2019 Competition on Massive Multi-Task Optimization (MaTSO)
    benchmark problems P1 to P6.

    These problems are designed to challenge algorithms with a large number of
    optimization tasks (task_size, typically 100 or more) derived from the same
    underlying function, but with different rotations and shifts, thereby testing
    transfer learning across many similar, but distinct tasks.

    Parameters
    ----------
    task_size : int, optional
        The number of tasks to generate for the problem (default is 10, but typically
        much larger for MaTSO benchmarks, e.g., 100).
    mat_dir : str, optional
        Directory path to the MAT files containing rotation matrices and global
        optima (GO) vectors for the tasks (default: '../Problems/MTSO/data_cec19matso').

    Attributes
    ----------
    task_size : int
        The configured number of tasks.
    dim : int
        The dimensionality of the search space for all tasks (fixed at 50).
    mat_dir : str
        The directory path for problem data files.
    """

    def __init__(self, task_size=10, mat_dir='../Problems/MTSO/data_cec19matso'):
        self.task_size = task_size
        self.dim = 50
        self.mat_dir = mat_dir

    def P1(self) -> MTOP:
        """
        Generates Problem 1 (MaTSO): **Rosenbrock** tasks.

        Each task is a 50D **Rosenbrock** function, rotated and shifted, with
        the number of tasks determined by ``self.task_size``.

        - Function: Rosenbrock
        - Dimensions: 50D
        - Bounds: [-50, 50]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``self.task_size`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask1.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask1.mat')
        # Load the array of GO vectors and rotation matrices
        go_task1 = go_data['GoTask1']
        rotation_task1 = rotation_data['RotationTask1']

        problem = MTOP()

        for i in range(self.task_size):
            # rotation_task1 is a 1xN cell array in MATLAB, accessed via [0, i]
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            # Closure to capture rotation and go vectors for each task function
            def create_task_function(rot, go):
                return lambda x: Rosenbrock(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -50),
                             upper_bound=np.full(self.dim, 50))
        return problem

    def P2(self) -> MTOP:
        """
        Generates Problem 2 (MaTSO): **Ackley** tasks.

        Each task is a 50D **Ackley** function, rotated and shifted, with
        the number of tasks determined by ``self.task_size``.

        - Function: Ackley
        - Dimensions: 50D
        - Bounds: [-50, 50]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``self.task_size`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask2.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask2.mat')
        go_task1 = go_data['GoTask2']
        rotation_task1 = rotation_data['RotationTask2']

        problem = MTOP()

        for i in range(self.task_size):
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            def create_task_function(rot, go):
                return lambda x: Ackley(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -50),
                             upper_bound=np.full(self.dim, 50))
        return problem

    def P3(self) -> MTOP:
        """
        Generates Problem 3 (MaTSO): **Rastrigin** tasks.

        Each task is a 50D **Rastrigin** function, rotated and shifted, with
        the number of tasks determined by ``self.task_size``.

        - Function: Rastrigin
        - Dimensions: 50D
        - Bounds: [-50, 50]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``self.task_size`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask3.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask3.mat')
        go_task1 = go_data['GoTask3']
        rotation_task1 = rotation_data['RotationTask3']

        problem = MTOP()

        for i in range(self.task_size):
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            def create_task_function(rot, go):
                return lambda x: Rastrigin(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -50),
                             upper_bound=np.full(self.dim, 50))
        return problem

    def P4(self) -> MTOP:
        """
        Generates Problem 4 (MaTSO): **Griewank** tasks.

        Each task is a 50D **Griewank** function, rotated and shifted, with
        the number of tasks determined by ``self.task_size``.

        - Function: Griewank
        - Dimensions: 50D
        - Bounds: [-100, 100]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``self.task_size`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask4.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask4.mat')
        go_task1 = go_data['GoTask4']
        rotation_task1 = rotation_data['RotationTask4']

        problem = MTOP()

        for i in range(self.task_size):
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            def create_task_function(rot, go):
                return lambda x: Griewank(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -100),
                             upper_bound=np.full(self.dim, 100))
        return problem

    def P5(self) -> MTOP:
        """
        Generates Problem 5 (MaTSO): **Weierstrass** tasks.

        Each task is a 50D **Weierstrass** function, rotated and shifted, with
        the number of tasks determined by ``self.task_size``.

        - Function: Weierstrass
        - Dimensions: 50D
        - Bounds: [-0.5, 0.5]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``self.task_size`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask5.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask5.mat')
        go_task1 = go_data['GoTask5']
        rotation_task1 = rotation_data['RotationTask5']

        problem = MTOP()

        for i in range(self.task_size):
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            def create_task_function(rot, go):
                return lambda x: Weierstrass(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -0.5),
                             upper_bound=np.full(self.dim, 0.5))
        return problem

    def P6(self) -> MTOP:
        """
        Generates Problem 6 (MaTSO): **Schwefel** tasks.

        Each task is a 50D **Schwefel** function, rotated and shifted, with
        the number of tasks determined by ``self.task_size``.

        - Function: Schwefel
        - Dimensions: 50D
        - Bounds: [-500, 500]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``self.task_size`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask6.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask6.mat')
        go_task1 = go_data['GoTask6']
        rotation_task1 = rotation_data['RotationTask6']

        problem = MTOP()

        for i in range(self.task_size):
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            def create_task_function(rot, go):
                return lambda x: Schwefel(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -500),
                             upper_bound=np.full(self.dim, 500))
        return problem
import scipy.io
from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP
import numpy as np

class CEC19MaTSO:
    """
    Implementation of the CEC 2019 Competition on Massive Multi-Task Optimization (MaTSO)
    benchmark problems P1 to P6.

    These problems are designed to challenge algorithms with a large number of
    optimization tasks (typically 100 or more) derived from the same
    underlying function, but with different rotations and shifts, thereby testing
    transfer learning across many similar, but distinct tasks.

    Parameters
    ----------
    mat_dir : str, optional
        Directory path to the MAT files containing rotation matrices and global
        optima (GO) vectors for the tasks (default: '../Problems/MTSO/data_cec19matso').

    Attributes
    ----------
    dim : int
        The dimensionality of the search space for all tasks (fixed at 50).
    mat_dir : str
        The directory path for problem data files.
    """

    def __init__(self, mat_dir='../Problems/MTSO/data_cec19matso'):
        self.dim = 50
        self.mat_dir = mat_dir

    def P1(self, task_num=10) -> MTOP:
        """
        Generates Problem 1 (MaTSO): **Rosenbrock** tasks.

        Each task is a 50D **Rosenbrock** function, rotated and shifted.

        Parameters
        ----------
        task_num : int, optional
            Number of tasks to create (default: 10).

        - Function: Rosenbrock
        - Dimensions: 50D
        - Bounds: [-50, 50]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``task_num`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask1.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask1.mat')
        go_task1 = go_data['GoTask1']
        rotation_task1 = rotation_data['RotationTask1']

        problem = MTOP()

        for i in range(task_num):
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            def create_task_function(rot, go):
                return lambda x: Rosenbrock(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -50),
                             upper_bound=np.full(self.dim, 50))
        return problem

    def P2(self, task_num=10) -> MTOP:
        """
        Generates Problem 2 (MaTSO): **Ackley** tasks.

        Each task is a 50D **Ackley** function, rotated and shifted.

        Parameters
        ----------
        task_num : int, optional
            Number of tasks to create (default: 10).

        - Function: Ackley
        - Dimensions: 50D
        - Bounds: [-50, 50]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``task_num`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask2.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask2.mat')
        go_task2 = go_data['GoTask2']
        rotation_task2 = rotation_data['RotationTask2']

        problem = MTOP()

        for i in range(task_num):
            rotation_matrix = rotation_task2[0, i]
            go_vector = go_task2[i, :]

            def create_task_function(rot, go):
                return lambda x: Ackley(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -50),
                             upper_bound=np.full(self.dim, 50))
        return problem

    def P3(self, task_num=10) -> MTOP:
        """
        Generates Problem 3 (MaTSO): **Rastrigin** tasks.

        Each task is a 50D **Rastrigin** function, rotated and shifted.

        Parameters
        ----------
        task_num : int, optional
            Number of tasks to create (default: 10).

        - Function: Rastrigin
        - Dimensions: 50D
        - Bounds: [-50, 50]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``task_num`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask3.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask3.mat')
        go_task3 = go_data['GoTask3']
        rotation_task3 = rotation_data['RotationTask3']

        problem = MTOP()

        for i in range(task_num):
            rotation_matrix = rotation_task3[0, i]
            go_vector = go_task3[i, :]

            def create_task_function(rot, go):
                return lambda x: Rastrigin(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -50),
                             upper_bound=np.full(self.dim, 50))
        return problem

    def P4(self, task_num=10) -> MTOP:
        """
        Generates Problem 4 (MaTSO): **Griewank** tasks.

        Each task is a 50D **Griewank** function, rotated and shifted.

        Parameters
        ----------
        task_num : int, optional
            Number of tasks to create (default: 10).

        - Function: Griewank
        - Dimensions: 50D
        - Bounds: [-100, 100]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``task_num`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask4.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask4.mat')
        go_task4 = go_data['GoTask4']
        rotation_task4 = rotation_data['RotationTask4']

        problem = MTOP()

        for i in range(task_num):
            rotation_matrix = rotation_task4[0, i]
            go_vector = go_task4[i, :]

            def create_task_function(rot, go):
                return lambda x: Griewank(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -100),
                             upper_bound=np.full(self.dim, 100))
        return problem

    def P5(self, task_num=10) -> MTOP:
        """
        Generates Problem 5 (MaTSO): **Weierstrass** tasks.

        Each task is a 50D **Weierstrass** function, rotated and shifted.

        Parameters
        ----------
        task_num : int, optional
            Number of tasks to create (default: 10).

        - Function: Weierstrass
        - Dimensions: 50D
        - Bounds: [-0.5, 0.5]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``task_num`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask5.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask5.mat')
        go_task5 = go_data['GoTask5']
        rotation_task5 = rotation_data['RotationTask5']

        problem = MTOP()

        for i in range(task_num):
            rotation_matrix = rotation_task5[0, i]
            go_vector = go_task5[i, :]

            def create_task_function(rot, go):
                return lambda x: Weierstrass(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -0.5),
                             upper_bound=np.full(self.dim, 0.5))
        return problem

    def P6(self, task_num=10) -> MTOP:
        """
        Generates Problem 6 (MaTSO): **Schwefel** tasks.

        Each task is a 50D **Schwefel** function, rotated and shifted.

        Parameters
        ----------
        task_num : int, optional
            Number of tasks to create (default: 10).

        - Function: Schwefel
        - Dimensions: 50D
        - Bounds: [-500, 500]

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing ``task_num`` tasks.
        """
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask6.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask6.mat')
        go_task6 = go_data['GoTask6']
        rotation_task6 = rotation_data['RotationTask6']

        problem = MTOP()

        for i in range(task_num):
            rotation_matrix = rotation_task6[0, i]
            go_vector = go_task6[i, :]

            def create_task_function(rot, go):
                return lambda x: Schwefel(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -500),
                             upper_bound=np.full(self.dim, 500))
        return problem
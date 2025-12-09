import scipy.io
from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP


class CEC19MaTSO:
    def __init__(self, task_size=10, mat_dir='../Problems/MTSO/data_cec19matso'):
        self.task_size = task_size
        self.dim = 50
        self.mat_dir = mat_dir

    def P1(self):
        go_data = scipy.io.loadmat(f'{self.mat_dir}/GoTask1.mat')
        rotation_data = scipy.io.loadmat(f'{self.mat_dir}/RotationTask1.mat')
        go_task1 = go_data['GoTask1']
        rotation_task1 = rotation_data['RotationTask1']

        problem = MTOP()

        for i in range(self.task_size):
            rotation_matrix = rotation_task1[0, i]
            go_vector = go_task1[i, :]

            def create_task_function(rot, go):
                return lambda x: Rosenbrock(x, rot, go, 0)

            task_function = create_task_function(rotation_matrix, go_vector)
            problem.add_task(task_function, dim=self.dim,
                             lower_bound=np.full(self.dim, -50),
                             upper_bound=np.full(self.dim, 50))
        return problem

    def P2(self):
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

    def P3(self):
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

    def P4(self):
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

    def P5(self):
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

    def P6(self):
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
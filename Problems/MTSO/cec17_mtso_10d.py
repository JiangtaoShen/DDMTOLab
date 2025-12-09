import scipy.io
from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP

class CEC17MTSO_10D:

    def __init__(self, mat_dir='../Problems/MTSO/data_cec17mtso_10d'):
        self.mat_dir = mat_dir

    def P1(self):
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

    def P2(self):
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

    def P3(self):
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

    def P4(self):
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/P4.mat')
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

    def P5(self):
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/P5.mat')
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

    def P6(self):
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/P6.mat')
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

    def P7(self):
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/P7.mat')
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

    def P8(self):
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/P8.mat')
        rotation_task1 = mat_data['Rotation_Task1_ld'].squeeze()
        go_task1 = mat_data['GO_Task1_ld'].squeeze()
        rotation_task2 = 5*np.eye(10, dtype=float)
        go_task2 = np.zeros((1, 10), dtype=float)

        def T1(x):
            return Ackley(x, rotation_task1, go_task1, 0.)

        def T2(x):
            return Weierstrass(x, rotation_task2, go_task2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=10, lower_bound=np.full(10, -50), upper_bound=np.full(10, 50))
        problem.add_task(T2, dim=10, lower_bound=np.full(10, -0.5), upper_bound=np.full(10, 0.5))
        return problem

    def P9(self):
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/P9.mat')
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
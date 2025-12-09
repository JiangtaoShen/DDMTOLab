import scipy.io
from Problems.BasicFunctions.basic_functions import *
from Methods.mtop import MTOP

class CEC17MTSO:

    def __init__(self, mat_dir='../Problems/MTSO/data_cec17mtso'):
        self.mat_dir = mat_dir

    def P1(self):
        # CI-HS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/CI_H.mat')
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

    def P2(self):
        # CI-MS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/CI_M.mat')
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

    def P3(self):
        # CI-LS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/CI_L.mat')
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

    def P4(self):
        # PI-HS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/PI_H.mat')
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

    def P5(self):
        # PI-MS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/PI_M.mat')
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

    def P6(self):
        # PI-LS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/PI_L.mat')
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

    def P7(self):
        # NI-HS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/NI_H.mat')
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

    def P8(self):
        # NI-MS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/NI_M.mat')
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

    def P9(self):
        # NI-LS
        mat_data = scipy.io.loadmat(f'{self.mat_dir}/NI_L.mat')
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
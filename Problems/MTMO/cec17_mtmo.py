import scipy.io
import numpy as np
from Methods.mtop import MTOP

class CEC17MTMO:

    def __init__(self, mat_dir='../Problems/MTMO/data_cec17mtmo'):
        self.mat_dir = mat_dir

    def P1(self):
        dim = 50

        def T1(x):
            x = np.atleast_2d(x)
            q = 1.0 + np.sum(x[:, 1:] ** 2, axis=1)
            x1 = x[:, 0]
            f1 = q * np.cos(np.pi * x1 / 2)
            f2 = q * np.sin(np.pi * x1 / 2)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            q = 1.0 + 9.0 / (dim - 1) * np.sum(np.abs(x[:, 1:]), axis=1)
            x1 = x[:, 0]
            f1 = x1
            f2 = q * (1.0 - (x1 / q) ** 2)
            return np.vstack([f1, f2]).T

        lb = np.array([0.0] + [-100.0] * (dim - 1))
        ub = np.array([1.0] + [100.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self):
        dim = 10
        Mcm2 = scipy.io.loadmat(f'{self.mat_dir}/Mcm2.mat')['Mcm2']
        Scm2 = scipy.io.loadmat(f'{self.mat_dir}/Scm2.mat')['Scm2'].flatten()

        def T1(x):
            x = np.atleast_2d(x)
            q = 1 + np.sum(100 * ((x[:, 1:-1] ** 2 - x[:, 2:]) ** 2 + (1 - x[:, 1:-1]) ** 2), axis=1)
            f1 = x[:, 0]
            f2 = q * (1 - (x[:, 0] / q) ** 2)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            z = (x[:, 1:] - Scm2) @ Mcm2.T
            q = 1 + 9 / (dim - 1) * np.sum(np.abs(z), axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2)
            f2 = q * np.sin(np.pi * x[:, 0] / 2)
            return np.vstack([f1, f2]).T

        lb = np.array([0.0] + [-5.0] * (dim - 1))
        ub = np.array([1.0] + [5.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self):
        dim = 50

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 1:]
            q = 1.0 + np.sum(part ** 2 - 10.0 * np.cos(2.0 * np.pi * part) + 10.0, axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            r = x.shape[1]
            part = x[:, 1:]
            term1 = 21.0 + np.e
            sum_sq = np.sum(part ** 2, axis=1)
            sum_cos = np.sum(np.cos(2.0 * np.pi * part), axis=1)
            q = term1 - 20.0 * np.exp(-0.2 * np.sqrt((1.0 / (r - 1)) * sum_sq)) - np.exp((1.0 / (r - 1)) * sum_cos)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        lb1 = np.concatenate(([0.0], -2.0 * np.ones(dim - 1)))
        ub1 = np.concatenate(([1.0], 2.0 * np.ones(dim - 1)))

        lb2 = np.concatenate(([0.0], -1.0 * np.ones(dim - 1)))
        ub2 = np.concatenate(([1.0], 1.0 * np.ones(dim - 1)))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P4(self):
        dim = 50
        Sph2 = scipy.io.loadmat(f'{self.mat_dir}/Sph2.mat')['Sph2'].flatten()

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 1:]
            q = 1.0 + np.sum(part ** 2, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            z = x[:, 1:] - Sph2
            q = 1.0 + np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        lb = np.array([0.0] + [-100.0] * (dim - 1))
        ub = np.array([1.0] + [100.0] * (dim - 1))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self):
        dim = 50
        Mpm1 = scipy.io.loadmat(f'{self.mat_dir}/Mpm1.mat')['Mpm1']
        Spm1 = scipy.io.loadmat(f'{self.mat_dir}/Spm1.mat')['Spm1'].flatten()
        Mpm2 = scipy.io.loadmat(f'{self.mat_dir}/Mpm2.mat')['Mpm2']

        def T1(x):
            x = np.atleast_2d(x)
            z = (x[:, 1:] - Spm1) @ Mpm1.T
            q = 1.0 + np.sum(z ** 2, axis=1)
            a = np.cos(np.pi * x[:, 0] / 2.0)
            b = np.sin(np.pi * x[:, 0] / 2.0)
            f1 = q * a
            f2 = q * b
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            z = x[:, 1:] @ Mpm2.T
            term = z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0
            q = 1.0 + np.sum(term, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - (x[:, 0] / q) ** 2)
            return np.vstack([f1, f2]).T

        lb = np.zeros(dim)
        ub = np.ones(dim)

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P6(self):
        dim = 50
        Spl2 = scipy.io.loadmat(f'{self.mat_dir}/Spl2.mat')['Spl2'].flatten()

        def T1(x):

            x = np.atleast_2d(x)
            r = x.shape[1]

            sum_sq = np.sum(x[:, 1:] ** 2, axis=1)
            denom = np.sqrt(np.arange(1, r))
            cos_terms = np.cos(x[:, 1:] / denom)
            prod_cos = np.prod(cos_terms, axis=1)
            q = 2.0 + (1.0 / 4000.0) * sum_sq - prod_cos
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        def T2(x):

            x = np.atleast_2d(x)
            r = x.shape[1]
            z = x[:, 1:] - Spl2
            sum_sq = np.sum(z ** 2, axis=1)
            sum_cos = np.sum(np.cos(2.0 * np.pi * z), axis=1)
            q = 21.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt((1.0 / (r - 1)) * sum_sq)) - np.exp(
                (1.0 / (r - 1)) * sum_cos)
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        lb1 = np.concatenate(([0.0], -50.0 * np.ones(dim - 1)))
        ub1 = np.concatenate(([1.0], 50.0 * np.ones(dim - 1)))

        lb2 = np.concatenate(([0.0], -100.0 * np.ones(dim - 1)))
        ub2 = np.concatenate(([1.0], 100.0 * np.ones(dim - 1)))

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim, lower_bound=lb2, upper_bound=ub2)
        return problem

    def P7(self):
        dim = 50

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 1:-1]
            next_part = x[:, 2:]
            q = 1.0 + np.sum(100.0 * (part ** 2 - next_part) ** 2 + (1.0 - part) ** 2, axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2.0)
            f2 = q * np.sin(np.pi * x[:, 0] / 2.0)
            return np.vstack([f1, f2]).T

        def T2(x):
            x = np.atleast_2d(x)
            q = 1.0 + np.sum(x[:, 1:] ** 2, axis=1)
            f1 = x[:, 0]
            f2 = q * (1.0 - np.sqrt(x[:, 0] / q))
            return np.vstack([f1, f2]).T

        lb = np.full(dim, -80.0)
        ub = np.full(dim, 80.0)
        lb[0] = 0.0
        ub[0] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self):
        dim = 20
        Mnm2 = scipy.io.loadmat(f'{self.mat_dir}/Mnm2.mat')['Mnm2']

        def T1(x):
            x = np.atleast_2d(x)
            part = x[:, 2:-1]
            next_part = x[:, 3:]
            q = 1.0 + np.sum(100 * (part ** 2 - next_part) ** 2 + (1.0 - part) ** 2, axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2) * np.cos(np.pi * x[:, 1] / 2)
            f2 = q * np.cos(np.pi * x[:, 0] / 2) * np.sin(np.pi * x[:, 1] / 2)
            f3 = q * np.sin(np.pi * x[:, 0] / 2)
            return np.vstack([f1, f2, f3]).T

        def T2(x):
            x = np.atleast_2d(x)
            z = x[:, 2:] @ Mnm2.T
            q = 1.0 + np.sum(z ** 2, axis=1)
            s = 0.5 * np.sum(x[:, :2], axis=1)
            f1 = s
            f2 = q * (1.0 - (s / q) ** 2)
            return np.vstack([f1, f2]).T

        lb = np.full(dim, -20.0)
        ub = np.full(dim, 20.0)
        lb[:2] = 0.0
        ub[:2] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
        problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P9(self):
        Snl1 = scipy.io.loadmat(f'{self.mat_dir}/Snl1.mat')['Snl1'].flatten()
        dim1 = 25
        dim2 = 50

        def T1(x):
            x = np.atleast_2d(x)
            z = x[:, 2:25] - Snl1
            a = np.arange(1, 24)
            q = 2.0 + (1.0 / 4000.0) * np.sum(z ** 2, axis=1) - np.prod(np.cos(z / np.sqrt(a)), axis=1)
            f1 = q * np.cos(np.pi * x[:, 0] / 2) * np.cos(np.pi * x[:, 1] / 2)
            f2 = q * np.cos(np.pi * x[:, 0] / 2) * np.sin(np.pi * x[:, 1] / 2)
            f3 = q * np.sin(np.pi * x[:, 0] / 2)
            return np.vstack([f1, f2, f3]).T

        def T2(x):
            x = np.atleast_2d(x)
            r = x.shape[1]
            q = 21.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(x[:, 2:] ** 2, axis=1) / (r - 2))) \
                - np.exp(np.sum(np.cos(2.0 * np.pi * x[:, 2:]), axis=1) / (r - 2))
            s = 0.5 * np.sum(x[:, :2], axis=1)
            f1 = s
            f2 = q * (1.0 - (s / q) ** 2)
            return np.vstack([f1, f2]).T

        lb1 = np.full(dim1, -50.0)
        ub1 = np.full(dim1, 50.0)
        lb1[:2] = 0.0
        ub1[:2] = 1.0

        lb2 = np.full(dim2, -100.0)
        ub2 = np.full(dim2, 100.0)
        lb2[:2] = 0.0
        ub2[:2] = 1.0

        problem = MTOP()
        problem.add_task(T1, dim=dim1, lower_bound=lb1, upper_bound=ub1)
        problem.add_task(T2, dim=dim2, lower_bound=lb2, upper_bound=ub2)
        return problem


def P1_T1_PF(N, M):
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])

def P1_T2_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])

def P2_T1_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])

def P2_T2_PF(N, M):
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])

def P3_T1_PF(N, M):
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])

def P3_T2_PF(N, M):
    f1 = np.linspace(0, 1, N, M)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])

def P4_T1_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])

def P4_T2_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])

def P5_T1_PF(N, M):
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])

def P5_T2_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])

def P6_T1_PF(N, M):
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])

def P6_T2_PF(N, M):
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])

def P7_T1_PF(N, M):
    theta = np.linspace(0, np.pi / 2, N)
    f1 = np.cos(theta)
    f2 = np.sin(theta)
    return np.column_stack([f1, f2])

def P7_T2_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])

def P8_T1_PF(N, M):
    n_sqrt = int(np.sqrt(N))
    theta = np.linspace(0, np.pi / 2, n_sqrt)
    phi = np.linspace(0, np.pi / 2, n_sqrt)

    points = []
    for t in theta:
        for p in phi:
            f1 = np.sin(t) * np.cos(p)
            f2 = np.sin(t) * np.sin(p)
            f3 = np.cos(t)
            points.append([f1, f2, f3])

    return np.array(points)

def P8_T2_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])

def P9_T1_PF(N, M):
    n_sqrt = int(np.sqrt(N))
    theta = np.linspace(0, np.pi / 2, n_sqrt)
    phi = np.linspace(0, np.pi / 2, n_sqrt)

    points = []
    for t in theta:
        for p in phi:
            f1 = np.sin(t) * np.cos(p)
            f2 = np.sin(t) * np.sin(p)
            f3 = np.cos(t)
            points.append([f1, f2, f3])

    return np.array(points)

def P9_T2_PF(N, M):
    f1 = np.linspace(0, 1, N)
    f2 = 1 - f1 ** 2
    return np.column_stack([f1, f2])


SETTINGS = {
    'metric': 'IGD',
    'n_pf': 1000,
    'pf_path':'./MOReference',
    'P1': {'T1': P1_T1_PF, 'T2': P1_T2_PF},
    'P2': {'T1': P2_T1_PF, 'T2': P2_T2_PF},
    'P3': {'T1': P3_T1_PF, 'T2': P3_T2_PF},
    'P4': {'T1': P4_T1_PF, 'T2': P4_T2_PF},
    'P5': {'T1': P5_T1_PF, 'T2': P5_T2_PF},
    'P6': {'T1': P6_T1_PF, 'T2': P6_T2_PF},
    'P7': {'T1': P7_T1_PF, 'T2': P7_T2_PF},
    'P8': {'T1': P8_T1_PF, 'T2': P8_T2_PF},
    'P9': {'T1': P9_T1_PF, 'T2': P9_T2_PF},
}
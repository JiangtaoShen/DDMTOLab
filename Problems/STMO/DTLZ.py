import numpy as np
from Methods.mtop import MTOP


class DTLZ:
    def DTLZ1(self, M=3, dim=None):
        k = 5
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 100 * (dim - M + 1 + np.sum((xM - 0.5) ** 2 - np.cos(20 * np.pi * (xM - 0.5)), axis=1))
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = 0.5 * (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= x[:, j]
                if i > 0:
                    obj[:, i] *= (1 - x[:, M - i - 1])
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ2(self, M=3, dim=None):
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = np.sum((xM - 0.5) ** 2, axis=1)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ3(self, M=3, dim=None):
        k = 10
        if dim is None:
            dim = M + k - 1
        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 100 * (k + np.sum((xM - 0.5) ** 2 - np.cos(20 * np.pi * (xM - 0.5)), axis=1))
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ4(self, M=3, dim=None, alpha=100):
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            x_modified = x.copy()
            x_modified[:, :M - 1] = x[:, :M - 1] ** alpha
            xM = x[:, M - 1:]
            g = np.sum((xM - 0.5) ** 2, axis=1)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x_modified[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x_modified[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ5(self, M=3, dim=None):
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = np.sum((xM - 0.5) ** 2, axis=1)
            x_modified = x.copy()
            if M > 2:
                Temp = np.tile(g.reshape(-1, 1), (1, M - 2))
                x_modified[:, 1:M - 1] = (1 + 2 * Temp * x[:, 1:M - 1]) / (2 + 2 * Temp)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x_modified[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x_modified[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ6(self, M=3, dim=None):
        k = 10
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = np.sum(xM ** 0.1, axis=1)
            x_modified = x.copy()
            if M > 2:
                Temp = np.tile(g.reshape(-1, 1), (1, M - 2))
                x_modified[:, 1:M - 1] = (1 + 2 * Temp * x[:, 1:M - 1]) / (2 + 2 * Temp)
            obj = np.zeros((n_samples, M))
            for i in range(M):
                obj[:, i] = (1 + g)
                for j in range(M - i - 1):
                    obj[:, i] *= np.cos(x_modified[:, j] * np.pi / 2)
                if i > 0:
                    obj[:, i] *= np.sin(x_modified[:, M - i - 1] * np.pi / 2)
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ7(self, M=3, dim=None):
        k = 20
        if dim is None:
            dim = M + k - 1

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            xM = x[:, M - 1:]
            g = 1 + 9 * np.mean(xM, axis=1)
            obj = np.zeros((n_samples, M))
            obj[:, :M - 1] = x[:, :M - 1]
            h = M - np.sum(
                obj[:, :M - 1] / (1 + g.reshape(-1, 1)) *
                (1 + np.sin(3 * np.pi * obj[:, :M - 1])),
                axis=1
            )
            obj[:, M - 1] = (1 + g) * h
            return obj

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ8(self, M=3, dim=None):
        k = 10
        if dim is None:
            dim = M * k

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            obj = np.zeros((n_samples, M))
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.mean(x[:, start_idx:end_idx], axis=1)
            return obj

        def C1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            obj = np.zeros((n_samples, M))
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.mean(x[:, start_idx:end_idx], axis=1)
            cons = np.zeros((n_samples, M))
            cons[:, :M - 1] = 1 - np.tile(obj[:, M - 1:M], (1, M - 1)) - 4 * obj[:, :M - 1]
            if M == 2:
                cons[:, M - 1] = 0
            else:
                sorted_obj = np.sort(obj[:, :M - 1], axis=1)
                cons[:, M - 1] = 1 - 2 * obj[:, M - 1] - np.sum(sorted_obj[:, :2], axis=1)
            return cons

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem

    def DTLZ9(self, M=3, dim=None):
        k = 10
        if dim is None:
            dim = M * k

        def T1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            x_transformed = x ** 0.1
            obj = np.zeros((n_samples, M))
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.sum(x_transformed[:, start_idx:end_idx], axis=1)

            return obj

        def C1(x):
            x = np.atleast_2d(x)
            n_samples = x.shape[0]
            D = x.shape[1]
            x_transformed = x ** 0.1
            obj = np.zeros((n_samples, M))
            for m in range(M):
                start_idx = m * D // M
                end_idx = (m + 1) * D // M
                obj[:, m] = np.sum(x_transformed[:, start_idx:end_idx], axis=1)
            cons = 1 - np.tile(obj[:, M - 1:M] ** 2, (1, M - 1)) - obj[:, :M - 1] ** 2
            return cons

        lb = np.zeros(dim)
        ub = np.ones(dim)
        problem = MTOP()
        problem.add_task(objective_func=T1, dim=dim, constraint_func=C1, lower_bound=lb, upper_bound=ub)
        return problem


import numpy as np
from Methods.Algo_Methods.uniform_point import uniform_point


def DTLZ1_PF(N, M):
    W, _ = uniform_point(N, M)
    return W/2

def DTLZ2_PF(N, M):
    W, _ = uniform_point(N, M)
    norms = np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))
    return W / norms

def DTLZ3_PF(N, M):
    W, _ = uniform_point(N, M)
    norms = np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))
    return W / norms

def DTLZ4_PF(N, M):
    W, _ = uniform_point(N, M)
    norms = np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))
    return W / norms

def DTLZ5_PF(N, M):
    t = np.linspace(0, 1, N)
    R = np.column_stack([t, 1 - t])
    norms = np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))
    R = R / norms
    if M > 2:
        first_col_repeated = np.tile(R[:, 0:1], (1, M - 2))
        R = np.hstack([first_col_repeated, R])
        powers = np.concatenate([[M - 2], np.arange(M - 2, -1, -1)])
        scale_factors = np.sqrt(2) ** powers
        R = R / scale_factors.reshape(1, -1)
    return R

def DTLZ6_PF(N, M):
    t = np.linspace(0, 1, N)
    R = np.column_stack([t, 1 - t])
    norms = np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))
    R = R / norms
    if M > 2:
        first_col_repeated = np.tile(R[:, 0:1], (1, M - 2))
        R = np.hstack([first_col_repeated, R])
        powers = np.concatenate([[M - 2], np.arange(M - 2, -1, -1)])
        scale_factors = np.sqrt(2) ** powers
        R = R / scale_factors.reshape(1, -1)
    return R

def DTLZ7_PF(N, M):
    interval = np.array([0, 0.251412, 0.631627, 0.859401])
    median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])
    X, _ = uniform_point(N, M - 1, 'grid')
    mask_low = X <= median
    X[mask_low] = X[mask_low] * (interval[1] - interval[0]) / median + interval[0]
    mask_high = X > median
    X[mask_high] = (X[mask_high] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
    h = M - np.sum(X / 2 * (1 + np.sin(3 * np.pi * X)), axis=1, keepdims=True)
    optimum = np.hstack([X, 2 * h])
    return optimum

def DTLZ8_PF(N, M):
    if M == 2:
        temp = np.linspace(0, 1, N).reshape(-1, 1)
        optimum = np.hstack([(1 - temp) / 4, temp])
    else:
        temp, _ = uniform_point(N // (M - 1), 3)
        temp[:, 2] = temp[:, 2] / 2
        mask = (temp[:, 0] >= (1 - temp[:, 2]) / 4) & \
               (temp[:, 0] <= temp[:, 1]) & \
               (temp[:, 2] <= 1 / 3)
        temp = temp[mask, :]
        n_temp = temp.shape[0]
        optimum = np.zeros((n_temp * (M - 1), M))
        for i in range(M - 1):
            start_idx = i * n_temp
            end_idx = (i + 1) * n_temp
            optimum[start_idx:end_idx, :M - 1] = np.tile(temp[:, 1], (M - 1, 1)).T
            optimum[start_idx:end_idx, M - 1] = temp[:, 2]
            optimum[start_idx:end_idx, i] = temp[:, 0]
        gap_values = np.unique(optimum[:, M - 1])
        if len(gap_values) > 1:
            gap = np.sort(gap_values)[1] - np.sort(gap_values)[0]
            temp_extra = np.arange(1 / 3, 1 + gap, gap).reshape(-1, 1)
            extra_points = np.hstack([
                np.tile((1 - temp_extra) / 4, (1, M - 1)),
                temp_extra
            ])
            optimum = np.vstack([optimum, extra_points])
        optimum = np.unique(optimum, axis=0)
    return optimum

def DTLZ9_PF(N, M):
    Temp = np.linspace(0, 1, N).reshape(-1, 1)
    optimum = np.hstack([
        np.tile(np.cos(0.5 * np.pi * Temp), (1, M - 1)),
        np.sin(0.5 * np.pi * Temp)
    ])
    return optimum


SETTINGS = {
    'metric': 'IGD',
    'n_ref': 10000,
    'DTLZ1': {'T1': DTLZ1_PF},
    'DTLZ2': {'T1': DTLZ2_PF},
    'DTLZ3': {'T1': DTLZ3_PF},
    'DTLZ4': {'T1': DTLZ4_PF},
    'DTLZ5': {'T1': DTLZ5_PF},
    'DTLZ6': {'T1': DTLZ6_PF},
    'DTLZ7': {'T1': DTLZ7_PF},
    'DTLZ8': {'T1': DTLZ8_PF},
    'DTLZ9': {'T1': DTLZ9_PF},
}

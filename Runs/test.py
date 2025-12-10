import numpy as np
from Methods.mtop import MTOP
from Algorithms.STSO.GA import GA
from Algorithms.STSO.BO import BO


def t1(x):  # Sphere
    return np.sum(x**2)

def t2(x):  # Rastrigin
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

problem = MTOP()
problem.add_task(t1, dim=2, lower_bound=np.array([-5, -5]), upper_bound=np.array([5, 5]))
problem.add_task(t2, dim=3, lower_bound=np.array([-5.12]*3), upper_bound=np.array([5.12]*3))

GA(problem, n=[10, 10], max_nfes=[100, 50]).optimize()
BO(problem, n_initial=[20, 10], max_nfes=[100, 50]).optimize()

from Methods.test_data_analysis import TestDataAnalyzer
TestDataAnalyzer().run()
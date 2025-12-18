"""
Reference Vector Guided Evolutionary Algorithm (RVEA)

This module implements RVEA for many-objective optimization problems.

References
----------
    [1] Cheng, Ran, et al. "A reference vector guided evolutionary algorithm for many-objective \
        optimization." IEEE transactions on evolutionary computation 20.5 (2016): 773-791.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.25
Version: 1.0
"""
from tqdm import tqdm
import time
from scipy.spatial.distance import cdist
from Methods.Algo_Methods.uniform_point import uniform_point
from Methods.Algo_Methods.algo_utils import *


class RVEA:
    """
    Reference Vector Guided Evolutionary Algorithm for many-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, alpha=2.0, fr=0.1, save_data=True, save_path='./TestData',
                 name='RVEA_test', disable_tqdm=True):
        """
        Initialize RVEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        alpha : float, optional
            Parameter controlling the rate of change of penalty (default: 2.0)
        fr : float, optional
            Frequency of employing reference vector adaptation (default: 0.1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'RVEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.alpha = alpha
        self.fr = fr
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the RVEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        no = problem.n_objs
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate uniformly distributed reference vectors for each task
        v0 = []
        for i in range(nt):
            v_i, n = uniform_point(n_per_task[i], no[i])
            v0.append(v_i)
            n_per_task[i] = n

        # Initialize adaptive reference vectors (will be scaled during optimization)
        v = [v_i.copy() for v_i in v0]

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Random parent selection
                matingpool = np.random.randint(0, decs[i].shape[0], size=n_per_task[i])

                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i][matingpool, :], muc=20.0, mum=15.0)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs),
                                                          (cons[i], off_cons))

                # Environmental selection using angle-penalized distance
                index = rvea_selection(objs[i], cons[i], v[i], (nfes_per_task[i] / max_nfes_per_task[i]) ** self.alpha)
                objs[i], decs[i], cons[i] = select_by_index(index, objs[i], decs[i], cons[i])

                # Periodically adapt reference vectors based on population range
                current_gen = int(np.ceil(nfes_per_task[i] / n_per_task[i]))
                update_interval = int(np.ceil(self.fr * max_nfes_per_task[i] / n_per_task[i]))
                if current_gen % update_interval == 0:
                    v[i] = v0[i] * (objs[i].max(axis=0) - objs[i].min(axis=0))

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def rvea_selection(objs, cons, v, theta):
    """
    Environmental selection using angle-penalized distance metric.

    Parameters
    ----------
    objs : np.ndarray
        Objective value matrix of shape (N, M)
    cons : np.ndarray
        Constraint value matrix of shape (N, K)
    v : np.ndarray
        Reference vectors of shape (NV, M)
    theta : float
        Penalty parameter controlling the balance between convergence and diversity

    Returns
    -------
    index : np.ndarray
        Indices of selected solutions of shape (n_selected,)

    Notes
    -----
    The angle-penalized distance (APD) combines both convergence and diversity:
    APD = (1 + M * theta * angle / gamma) * distance
    where M is the number of objectives, angle is the angle between solution and reference vector,
    gamma is the smallest angle between reference vectors, and distance is the Euclidean norm.
    """
    N, M = objs.shape
    NV = v.shape[0]

    # Translate the population to make minimum objective values zero
    objs = objs - np.min(objs, axis=0)

    # Calculate constraint violation for each solution
    CV = np.sum(np.maximum(0, cons), axis=1)

    # Calculate the smallest angle between each reference vector and others
    cosine = 1 - cdist(v, v, metric='cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    gamma = np.maximum(gamma, 1e-6)

    # Associate each solution to its nearest reference vector
    angle = np.arccos(1 - cdist(objs, v, metric='cosine'))
    associate = np.argmin(angle, axis=1)

    # Select one solution for each reference vector
    Next = np.full(NV, -1, dtype=int)
    for i in np.unique(associate):
        current1 = np.where((associate == i) & (CV == 0))[0]
        current2 = np.where((associate == i) & (CV != 0))[0]

        if len(current1) > 0:
            # Calculate angle-penalized distance and select the minimum
            APD = (1 + M * theta * angle[current1, i] / gamma[i]) * np.sqrt(np.sum(objs[current1, :] ** 2, axis=1))
            best = np.argmin(APD)
            Next[i] = current1[best]
        elif len(current2) > 0:
            # Select the solution with minimum constraint violation
            best = np.argmin(CV[current2])
            Next[i] = current2[best]

    # Extract valid selected indices
    index = Next[Next != -1]

    return index
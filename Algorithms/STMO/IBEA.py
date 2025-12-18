"""
Indicator-Based Evolutionary Algorithm (IBEA)

This module implements IBEA for multi-objective optimization problems.

References
----------
    [1] Zitzler, Eckart, and Simon Künzli. "Indicator-based selection in multiobjective \
        search." International conference on parallel problem solving from nature. 2004.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.13
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from Methods.Algo_Methods.algo_utils import *


class IBEA:
    """
    Indicator-Based Evolutionary Algorithm for multi-objective optimization.

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
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, kappa=0.05, muc=20.0, mum=15.0, save_data=True,
                 save_path='./TestData', name='IBEA_test', disable_tqdm=True):
        """
        Initialize IBEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        kappa : float, optional
            Fitness scaling factor (default: 0.05)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'IBEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.kappa = kappa
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the IBEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs = init_history(decs, objs)

        # Calculate initial fitness for each task
        fitness = []
        for i in range(nt):
            fitness_i, _, _ = ibea_fitness(objs[i], self.kappa)
            fitness.append(fitness_i.copy())

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Parent selection via binary tournament based on fitness
                matingpool = tournament_selection(2, n_per_task[i], -fitness[i])

                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i][matingpool, :], muc=self.muc, mum=self.mum)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs))

                # Environmental selection: keep best n individuals based on fitness
                index, fitness[i] = ibea_selection(objs[i], n_per_task[i], self.kappa)
                objs[i], decs[i]  = select_by_index(index, objs[i], decs[i])

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def ibea_fitness(objs, kappa):
    """
    Calculate fitness values for the population using IBEA indicator.

    Parameters
    ----------
    objs : ndarray
        Objective values with shape (N, M), where N is the number of
        individuals and M is the number of objectives.
    kappa : float
        Fitness scaling factor.

    Returns
    -------
    fitness : ndarray
        Fitness values for each individual with shape (N,).
    I : ndarray
        Indicator matrix with shape (N, N).
    C : ndarray
        Normalization constants with shape (N,).
    """
    N = objs.shape[0]

    # Normalize objective values to [0, 1]
    min_val = np.min(objs, axis=0)
    max_val = np.max(objs, axis=0)

    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    objs_norm = (objs - min_val) / range_val

    # Calculate indicator matrix I where I[i,j] is the max difference between individual i and j
    I = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            I[i, j] = np.max(objs_norm[i, :] - objs_norm[j, :])

    # Calculate normalization constants (max absolute value in each column)
    C = np.max(np.abs(I), axis=0)

    # Calculate fitness: sum along columns after scaling by C and kappa
    C_matrix = np.tile(C, (N, 1))
    fitness = np.sum(-np.exp(-I / C_matrix / kappa), axis=0) + 1

    return fitness, I, C


def ibea_selection(objs, N, kappa):
    """
    Environmental selection for IBEA algorithm.

    Parameters
    ----------
    objs : ndarray
        Objective values with shape (n, m), where n is the population
        size and m is the number of objectives.
    N : int
        Number of individuals to select.
    kappa : float
        Fitness scaling factor.

    Returns
    -------
    index : ndarray
        Indices of selected individuals with shape (N,).
    selected_fitness : ndarray
        Fitness values of selected individuals with shape (N,).
    """
    n, m = objs.shape
    remaining = list(range(n))

    # Calculate initial fitness values
    fitness, I, C = ibea_fitness(objs, kappa)

    # Iteratively remove the worst individual until N individuals remain
    while len(remaining) > N:
        # Find the individual with minimum fitness
        fitness_remaining = fitness[remaining]
        worst_idx = np.argmin(fitness_remaining)
        removed_idx = remaining[worst_idx]

        # Update fitness values after removal
        fitness = fitness + np.exp(-I[removed_idx, :] / C[removed_idx] / kappa)

        # Remove the individual
        remaining.pop(worst_idx)

    # Return selected indices and their fitness values
    index = np.array(remaining)
    selected_fitness = fitness[index]

    return index, selected_fitness
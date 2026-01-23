"""
Multi-objective Evolutionary Algorithm Based on Decomposition and Dominance (MOEA/DD)

This module implements MOEA/DD for multi/many-objective optimization problems.

References
----------
    [1] Li, Ke, et al. "An evolutionary many-objective optimization algorithm based on dominance and decomposition." \
        IEEE transactions on evolutionary computation 19.5 (2014): 694-716.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.18
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MOEADD:
    """
    Multi-objective Evolutionary Algorithm Based on Decomposition and Dominance.

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

    def __init__(self, problem, n=None, max_nfes=None, delta=0.9, muc=20.0, mum=15.0, save_data=True,
                 save_path='./TestData', name='MOEADD_test', disable_tqdm=True):
        """
        Initialize MOEA/DD.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        delta : float, optional
            Probability of choosing parents locally (default: 0.9)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MOEADD_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.delta = delta
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MOEA/DD algorithm.

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

        # Generate uniformly distributed weight vectors for each task
        W = []
        T = []  # Neighborhood size
        B = []  # Neighbor indices
        for i in range(nt):
            w_i, n = uniform_point(n_per_task[i], no[i])
            W.append(w_i)
            n_per_task[i] = n

            # Set neighborhood size to 10% of population
            T.append(int(np.ceil(n / 10)))

            # Detect the neighbors of each weight vector based on Euclidean distance
            distances = squareform(pdist(w_i))
            neighbors = np.argsort(distances, axis=1)[:, :T[i]]
            B.append(neighbors)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize region for each task
        region = []
        for i in range(nt):
            # Calculate cosine similarity: 1 - cosine distance
            norm_objs = objs[i] / (np.linalg.norm(objs[i], axis=1, keepdims=True) + 1e-10)
            norm_w = W[i] / (np.linalg.norm(W[i], axis=1, keepdims=True) + 1e-10)
            cosine_similarity = np.dot(norm_objs, norm_w.T)
            region_i = np.argmax(cosine_similarity, axis=1)
            region.append(region_i)

        # Initialize front_no for each task
        front_no = []
        for i in range(nt):
            front_no_i, _ = nd_sort(objs[i], cons[i], n_per_task[i])
            front_no.append(front_no_i)

        # Initialize ideal point Z for each task
        Z = []
        for i in range(nt):
            Z.append(np.min(objs[i], axis=0))

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        # Main loop
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for task_id in active_tasks:
                # For each solution
                for i in range(n_per_task[task_id]):
                    # Choose the parents
                    ei = np.where(np.isin(region[task_id], B[task_id][i]))[0]

                    if np.random.rand() < self.delta and len(ei) >= 2:
                        # Local selection
                        cv_ei = np.sum(np.maximum(0, cons[task_id][ei]), axis=1) if cons[
                                                                                        task_id] is not None else np.zeros(
                            len(ei))
                        parents = tournament_selection(2, 2, cv_ei)
                        parents = ei[parents]
                    else:
                        # Global selection
                        cv = np.sum(np.maximum(0, cons[task_id]), axis=1) if cons[task_id] is not None else np.zeros(
                            n_per_task[task_id])
                        parents = tournament_selection(2, 2, cv)

                    # Generate an offspring
                    parent_decs = decs[task_id][parents, :]
                    off_decs = ga_generation(parent_decs, muc=self.muc, mum=self.mum)
                    off_objs, off_cons = evaluation_single(problem, off_decs[:1], task_id)

                    # Assign offspring to region
                    norm_off_obj = off_objs[0] / (np.linalg.norm(off_objs[0]) + 1e-10)
                    norm_w = W[task_id] / (np.linalg.norm(W[task_id], axis=1, keepdims=True) + 1e-10)
                    cosine_similarity = np.dot(norm_off_obj, norm_w.T)
                    off_region = np.argmax(cosine_similarity)

                    # Add the offspring to the population
                    decs[task_id] = np.vstack([decs[task_id], off_decs[0]])
                    objs[task_id] = np.vstack([objs[task_id], off_objs[0]])
                    if cons[task_id] is not None and off_cons is not None:
                        cons[task_id] = np.vstack([cons[task_id], off_cons[0]])
                    region[task_id] = np.append(region[task_id], off_region)

                    # Update front_no (add mode)
                    front_no[task_id] = update_front(objs[task_id], front_no[task_id])

                    # Calculate constraint violations
                    cv = np.sum(np.maximum(0, cons[task_id]), axis=1) if cons[task_id] is not None else np.zeros(
                        len(objs[task_id]))

                    # Update the ideal point
                    Z[task_id] = np.minimum(Z[task_id], off_objs[0])

                    # Delete a solution from the population
                    if np.any(cv > 0):
                        S = np.argsort(cv)[::-1]
                        S = S[:np.sum(cv > 0)]
                        flag = False
                        x = None

                        for j in range(len(S)):
                            if np.sum(region[task_id] == region[task_id][S[j]]) > 1:
                                flag = True
                                x = S[j]
                                break

                        if not flag:
                            x = S[0]

                    elif np.max(front_no[task_id]) == 1:
                        x = locate_worst(objs[task_id], W[task_id], region[task_id], front_no[task_id], Z[task_id])

                    else:
                        fl = np.where(front_no[task_id] == np.max(front_no[task_id]))[0]

                        if len(fl) == 1:
                            if np.sum(region[task_id] == region[task_id][fl[0]]) > 1:
                                x = fl[0]
                            else:
                                x = locate_worst(objs[task_id], W[task_id], region[task_id], front_no[task_id],
                                                 Z[task_id])
                        else:
                            sub_region = np.unique(region[task_id][fl])
                            crowd = np.bincount(region[task_id][np.isin(region[task_id], sub_region)],
                                                minlength=W[task_id].shape[0])
                            phi = np.where(crowd == np.max(crowd))[0]

                            pbi = cal_pbi(objs[task_id], W[task_id], region[task_id], Z[task_id],
                                          np.isin(region[task_id], phi))
                            pbi_sum = np.zeros(W[task_id].shape[0])

                            for j in range(len(pbi)):
                                pbi_sum[region[task_id][j]] += pbi[j]

                            phi = np.argmax(pbi_sum)
                            phih = np.where(region[task_id] == phi)[0]

                            if len(phih) > 1:
                                x = phih[np.argmax(pbi[phih])]
                            else:
                                x = locate_worst(objs[task_id], W[task_id], region[task_id], front_no[task_id],
                                                 Z[task_id])

                    # Update front_no before removing (delete mode)
                    front_no[task_id] = update_front(objs[task_id], front_no[task_id], x)

                    # Remove the worst solution
                    decs[task_id] = np.delete(decs[task_id], x, axis=0)
                    objs[task_id] = np.delete(objs[task_id], x, axis=0)
                    if cons[task_id] is not None:
                        cons[task_id] = np.delete(cons[task_id], x, axis=0)
                    region[task_id] = np.delete(region[task_id], x)

                    # Update evaluation count
                    nfes_per_task[task_id] += 1
                    pbar.update(1)

                    # Check if evaluation budget is exhausted
                    if nfes_per_task[task_id] >= max_nfes_per_task[task_id]:
                        break

                # Update history
                append_history(all_decs[task_id], decs[task_id], all_objs[task_id],
                               objs[task_id], all_cons[task_id], cons[task_id])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, all_cons=all_cons,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def update_front(pop_obj, front_no, x=None):
    """
    Update the front number of each solution when a solution is added or deleted.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values of shape (N, M)
    front_no : np.ndarray
        Front numbers
    x : int, optional
        Index of solution to delete. If None, assumes a new solution is added at the end (default: None)

    Returns
    -------
    FrontNo : np.ndarray
        Updated front numbers
    """
    N, M = pop_obj.shape

    if x is None:
        # Add a new solution (has been stored in the last of PopObj)
        front_no = np.append(front_no, 0)
        move = np.zeros(N, dtype=bool)
        move[N - 1] = True
        current_f = 1

        # Locate the front No. of the new solution
        while True:
            dominated = False
            for i in range(N - 1):
                if front_no[i] == current_f:
                    m = 0
                    while m < M and pop_obj[i, m] <= pop_obj[N - 1, m]:
                        m += 1
                    dominated = (m == M)
                    if dominated:
                        break

            if not dominated:
                break
            else:
                current_f += 1

        # Move down the dominated solutions front by front
        while np.any(move):
            next_move = np.zeros(N, dtype=bool)
            for i in range(N):
                if front_no[i] == current_f:
                    dominated = False
                    for j in range(N):
                        if move[j]:
                            m = 0
                            while m < M and pop_obj[j, m] <= pop_obj[i, m]:
                                m += 1
                            dominated = (m == M)
                            if dominated:
                                break
                    next_move[i] = dominated

            front_no[move] = current_f
            current_f += 1
            move = next_move

    else:
        # Delete the x-th solution
        x = int(x)
        move = np.zeros(N, dtype=bool)
        move[x] = True
        current_f = int(front_no[x]) + 1

        while np.any(move):
            next_move = np.zeros(N, dtype=bool)
            for i in range(N):
                if front_no[i] == current_f:
                    dominated = False
                    for j in range(N):
                        if move[j]:
                            m = 0
                            while m < M and pop_obj[j, m] <= pop_obj[i, m]:
                                m += 1
                            dominated = (m == M)
                            if dominated:
                                break
                    next_move[i] = dominated

            for i in range(N):
                if next_move[i]:
                    dominated = False
                    for j in range(N):
                        if front_no[j] == current_f - 1 and not move[j]:
                            m = 0
                            while m < M and pop_obj[j, m] <= pop_obj[i, m]:
                                m += 1
                            dominated = (m == M)
                            if dominated:
                                break
                    next_move[i] = not dominated

            front_no[move] = current_f - 2
            current_f += 1
            move = next_move

        front_no = np.delete(front_no, x)

    return front_no


def locate_worst(pop_obj, W, region, front_no, Z):
    """
    Detect the worst solution in the population.

    Parameters
    ----------
    pop_obj : ndarray
        Population objective values, shape (n_solutions, n_objectives).
    W : ndarray
        Weight vectors, shape (n_weights, n_objectives).
    region : ndarray
        Region assignment for each solution, shape (n_solutions,).
    front_no : ndarray
        Pareto front number for each solution, shape (n_solutions,).
    Z : ndarray
        Ideal point (reference point), shape (n_objectives,).

    Returns
    -------
    int
        Index of the worst solution in the population.
    """
    crowd = np.bincount(region, minlength=W.shape[0])
    phi = np.where(crowd == np.max(crowd))[0]
    pbi = cal_pbi(pop_obj, W, region, Z, np.isin(region, phi))
    pbi_sum = np.zeros(W.shape[0])
    for j in range(len(pbi)):
        pbi_sum[region[j]] += pbi[j]
    phi = np.argmax(pbi_sum)
    phih = np.where(region == phi)[0]
    R = phih[front_no[phih] == np.max(front_no[phih])]
    x = R[np.argmax(pbi[R])]
    return int(x)


def cal_pbi(pop_obj, W, region, Z, sub):
    """
    Calculate the PBI (Penalty-based Boundary Intersection) value between
    each solution and its associated weight vector.

    Parameters
    ----------
    pop_obj : ndarray
        Population objective values, shape (n_solutions, n_objectives).
    W : ndarray
        Weight vectors, shape (n_weights, n_objectives).
    region : ndarray
        Region assignment for each solution, shape (n_solutions,).
    Z : ndarray
        Ideal point (reference point), shape (n_objectives,).
    sub : ndarray of bool
        Boolean mask indicating which solutions to calculate PBI for,
        shape (n_solutions,).

    Returns
    -------
    pbi : ndarray
        PBI values for all solutions, shape (n_solutions,).
        Only solutions where sub is True have non-zero values.
    """
    m = W.shape[1]
    z_rep = np.tile(Z, (np.sum(sub), 1))
    w_sub = W[region[sub], :]
    norm_w = np.sqrt(np.sum(w_sub ** 2, axis=1))
    d1 = np.abs(np.sum((pop_obj[sub, :] - z_rep) * w_sub, axis=1)) / norm_w
    d1_norm_w = d1 / norm_w
    projection = z_rep + w_sub * d1_norm_w[:, np.newaxis]
    d2 = np.sqrt(np.sum((pop_obj[sub, :] - projection) ** 2, axis=1))
    pbi = np.zeros(pop_obj.shape[0])
    pbi[sub] = d1 + 5 * d2
    return pbi
"""
Self-Regulated Evolutionary Multitask Optimization (SREMTO)

This module implements SREMTO for multi-task single-objective optimization problems.

References
----------
    [1] Zheng, Xiaolong, A. K. Qin, Maoguo Gong, and Deyun Zhou. "Self-Regulated Evolutionary Multitask Optimization." IEEE Transactions on Evolutionary Computation 24.1 (2020): 16-28. https://doi.org/10.1109/TEVC.2019.2904696

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.28
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from Methods.Algo_Methods.algo_utils import *


class SREMTO:
    """
    Self-Regulated Evolutionary Multitask Optimization.

    This algorithm features:
    - Ability vector for self-regulated knowledge transfer
    - Two-line segment ability calculation based on ranking
    - Combined SBX crossover with differential mutation
    - Multi-factorial evaluation based on ability probability

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, th=0.3, p_alpha=0.7, p_beta=1.0,
                 muc=1.0, mum=39.0, save_data=True, save_path='./TestData',
                 name='SREMTO_test', disable_tqdm=True):
        """
        Initialize SREMTO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        th : float, optional
            Threshold for two-line segments point (default: 0.3)
        p_alpha : float, optional
            Probability of crossover (default: 0.7)
        p_beta : float, optional
            Probability of differential mutation (default: 1.0)
        muc : float, optional
            Distribution index for SBX crossover (default: 1.0)
        mum : float, optional
            Distribution index for polynomial mutation (default: 39.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'SREMTO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.th = th
        self.p_alpha = p_alpha
        self.p_beta = p_beta
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SREMTO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d_max = max(dims)
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Two-line segment parameters for ability calculation
        # Line 1: for ranks 1 to n (within top-n for a task)
        a1 = (self.th - 1) / (n - 1)
        b1 = (n - self.th) / (n - 1)
        # Line 2: for ranks > n (outside top-n for a task)
        a2 = (-self.th) / (n * (nt - 1))
        b2 = (n * nt * self.th) / (n * (nt - 1))

        # Initialize unified population (all tasks share same individuals)
        # Each individual has: dec, mf_obj (obj for each task), mf_cv, mf_rank, ability
        pop_size = n * nt  # Total population size
        pop_decs = np.random.rand(pop_size, d_max)

        # Initialize multi-factorial objectives and constraints
        pop_mf_objs = np.full((pop_size, nt), np.inf)
        pop_mf_cvs = np.full((pop_size, nt), np.inf)
        pop_mf_ranks = np.zeros((pop_size, nt), dtype=int)
        pop_abilities = np.zeros((pop_size, nt))

        # Evaluate initial population on all tasks
        nfes = 0
        for i in range(pop_size):
            for t in range(nt):
                dec_t = pop_decs[i, :dims[t]].reshape(1, -1)
                obj_t, con_t = evaluation_single(problem, dec_t, t)
                pop_mf_objs[i, t] = obj_t[0, 0]
                cv_t = np.sum(np.maximum(0, con_t[0])) if con_t is not None and con_t.size > 0 else 0
                pop_mf_cvs[i, t] = cv_t
                nfes += 1

        # Calculate initial rankings for each task
        for t in range(nt):
            pop_mf_ranks[:, t] = self._calculate_ranks(pop_mf_objs[:, t], pop_mf_cvs[:, t])

        # Calculate initial ability vectors
        pop_abilities = self._calculate_abilities(pop_mf_ranks, a1, b1, a2, b2, n)

        # Track best solutions for each task
        best_decs = [None] * nt
        best_objs = [np.inf] * nt
        for t in range(nt):
            best_idx = np.argmin(pop_mf_objs[:, t])
            best_decs[t] = pop_decs[best_idx].copy()
            best_objs[t] = pop_mf_objs[best_idx, t]

        # Initialize history storage
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        # Store initial population per task
        for t in range(nt):
            task_indices = np.where(pop_mf_ranks[:, t] <= n)[0]
            if len(task_indices) > 0:
                all_decs[t].append(pop_decs[task_indices, :dims[t]].copy())
                all_objs[t].append(pop_mf_objs[task_indices, t].reshape(-1, 1).copy())
                all_cons[t].append(pop_mf_cvs[task_indices, t].reshape(-1, 1).copy())

        # Progress bar
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while nfes < max_nfes:
            int_pop_decs = pop_decs.copy()
            int_pop_mf_objs = pop_mf_objs.copy()
            int_pop_mf_cvs = pop_mf_cvs.copy()
            int_pop_abilities = pop_abilities.copy()

            # Generate offspring for each task
            for t in range(nt):
                # Select parents: individuals with rank <= n for task t
                parent_indices = np.where(pop_mf_ranks[:, t] <= n)[0]
                if len(parent_indices) < 2:
                    continue

                parent_decs = pop_decs[parent_indices]
                parent_abilities = pop_abilities[parent_indices]

                # Generate offspring
                off_decs, off_abilities = self._generation(
                    parent_decs, parent_abilities, best_decs[t], d_max
                )

                # Evaluate offspring on tasks based on ability
                off_mf_objs = np.full((len(off_decs), nt), np.inf)
                off_mf_cvs = np.full((len(off_decs), nt), np.inf)

                for i in range(len(off_decs)):
                    for k in range(nt):
                        # Evaluate on task k if: k == t (always) or random < ability[k]
                        if k == t or np.random.rand() < off_abilities[i, k]:
                            dec_k = off_decs[i, :dims[k]].reshape(1, -1)
                            obj_k, con_k = evaluation_single(problem, dec_k, k)
                            off_mf_objs[i, k] = obj_k[0, 0]
                            cv_k = np.sum(np.maximum(0, con_k[0])) if con_k is not None and con_k.size > 0 else 0
                            off_mf_cvs[i, k] = cv_k
                            nfes += 1
                            pbar.update(1)

                            if nfes >= max_nfes:
                                break
                    if nfes >= max_nfes:
                        break

                # Merge offspring with intermediate population
                int_pop_decs = np.vstack([int_pop_decs, off_decs])
                int_pop_mf_objs = np.vstack([int_pop_mf_objs, off_mf_objs])
                int_pop_mf_cvs = np.vstack([int_pop_mf_cvs, off_mf_cvs])
                int_pop_abilities = np.vstack([int_pop_abilities, off_abilities])

                if nfes >= max_nfes:
                    break

            # Selection: calculate ranks and select top-n per task
            int_pop_mf_ranks = np.zeros((len(int_pop_decs), nt), dtype=int)
            for t in range(nt):
                int_pop_mf_ranks[:, t] = self._calculate_ranks(
                    int_pop_mf_objs[:, t], int_pop_mf_cvs[:, t]
                )

            # Select individuals that are in top-n for at least one task
            selected_indices = set()
            for t in range(nt):
                top_n_indices = np.where(int_pop_mf_ranks[:, t] <= n)[0]
                selected_indices.update(top_n_indices)
            selected_indices = np.array(list(selected_indices))

            if len(selected_indices) > 0:
                pop_decs = int_pop_decs[selected_indices]
                pop_mf_objs = int_pop_mf_objs[selected_indices]
                pop_mf_cvs = int_pop_mf_cvs[selected_indices]
                pop_abilities = int_pop_abilities[selected_indices]

                # Recalculate ranks for selected population
                pop_mf_ranks = np.zeros((len(pop_decs), nt), dtype=int)
                for t in range(nt):
                    pop_mf_ranks[:, t] = self._calculate_ranks(
                        pop_mf_objs[:, t], pop_mf_cvs[:, t]
                    )

                # Update abilities
                pop_abilities = self._calculate_abilities(pop_mf_ranks, a1, b1, a2, b2, n)

                # Update best solutions
                for t in range(nt):
                    valid_mask = pop_mf_objs[:, t] < np.inf
                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        best_valid_idx = valid_indices[np.argmin(pop_mf_objs[valid_indices, t])]
                        if pop_mf_objs[best_valid_idx, t] < best_objs[t]:
                            best_decs[t] = pop_decs[best_valid_idx].copy()
                            best_objs[t] = pop_mf_objs[best_valid_idx, t]

                # Store history per task
                for t in range(nt):
                    task_indices = np.where(pop_mf_ranks[:, t] <= n)[0]
                    if len(task_indices) > 0:
                        all_decs[t].append(pop_decs[task_indices, :dims[t]].copy())
                        all_objs[t].append(pop_mf_objs[task_indices, t].reshape(-1, 1).copy())
                        all_cons[t].append(pop_mf_cvs[task_indices, t].reshape(-1, 1).copy())

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=max_nfes_per_task, all_cons=all_cons,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _calculate_ranks(self, objs, cvs):
        """
        Calculate ranks based on constraint violation first, then objective value.

        Parameters
        ----------
        objs : np.ndarray
            Objective values, shape (n,)
        cvs : np.ndarray
            Constraint violations, shape (n,)

        Returns
        -------
        ranks : np.ndarray
            Ranks (1-based), shape (n,)
        """
        n = len(objs)
        # Sort by CV first, then by objective
        indices = np.lexsort((objs, cvs))
        ranks = np.zeros(n, dtype=int)
        ranks[indices] = np.arange(1, n + 1)
        return ranks

    def _calculate_abilities(self, mf_ranks, a1, b1, a2, b2, n):
        """
        Calculate ability vectors using two-line segment formula.

        Parameters
        ----------
        mf_ranks : np.ndarray
            Multi-factorial ranks, shape (pop_size, nt)
        a1, b1 : float
            Parameters for line segment 1 (rank <= n)
        a2, b2 : float
            Parameters for line segment 2 (rank > n)
        n : int
            Population size per task

        Returns
        -------
        abilities : np.ndarray
            Ability vectors, shape (pop_size, nt)
        """
        pop_size, nt = mf_ranks.shape
        abilities = np.zeros((pop_size, nt))

        for t in range(nt):
            for i in range(pop_size):
                rank = mf_ranks[i, t]
                if rank <= n:
                    # Line 1: high ability for top-ranked individuals
                    abilities[i, t] = a1 * rank + b1
                else:
                    # Line 2: lower ability for lower-ranked individuals
                    abilities[i, t] = a2 * rank + b2

        # Clip abilities to [0, 1]
        abilities = np.clip(abilities, 0, 1)
        return abilities

    def _generation(self, parent_decs, parent_abilities, best_dec, d_max):
        """
        Generate offspring using SBX crossover and differential mutation.

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables, shape (n_parents, d_max)
        parent_abilities : np.ndarray
            Parent ability vectors, shape (n_parents, nt)
        best_dec : np.ndarray
            Best solution for the current task, shape (d_max,)
        d_max : int
            Maximum dimension

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (n_parents, d_max)
        off_abilities : np.ndarray
            Offspring ability vectors (inherited from parents), shape (n_parents, nt)
        """
        n_parents = len(parent_decs)
        nt = parent_abilities.shape[1]

        off_decs = np.zeros((n_parents, d_max))
        off_abilities = np.zeros((n_parents, nt))

        # Shuffle indices for pairing
        ind_order = np.random.permutation(n_parents)

        count = 0
        for i in range(n_parents // 2):
            p1 = ind_order[i]
            p2 = ind_order[i + n_parents // 2]

            if np.random.rand() < self.p_alpha:
                # Crossover
                off_dec1, off_dec2 = crossover(parent_decs[p1], parent_decs[p2], mu=self.muc)

                # Differential mutation
                if np.random.rand() < self.p_beta:
                    r = np.random.rand()
                    off_dec1 = off_dec1 + r * (best_dec - off_dec1 + parent_decs[p1] - parent_decs[p2])
                    off_dec2 = off_dec2 + r * (best_dec - off_dec2 + parent_decs[p2] - parent_decs[p1])
            else:
                # Mutation only
                off_dec1 = mutation(parent_decs[p1].copy(), mu=self.mum)
                off_dec2 = mutation(parent_decs[p2].copy(), mu=self.mum)

            # Inherit abilities from parents (imitation)
            off_abilities[count] = parent_abilities[p1].copy()
            off_abilities[count + 1] = parent_abilities[p2].copy()

            # Boundary handling
            off_decs[count] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1] = np.clip(off_dec2, 0, 1)

            count += 2

        # Handle odd number of parents
        if n_parents % 2 == 1:
            last_idx = ind_order[-1]
            off_decs[count] = mutation(parent_decs[last_idx].copy(), mu=self.mum)
            off_decs[count] = np.clip(off_decs[count], 0, 1)
            off_abilities[count] = parent_abilities[last_idx].copy()
            count += 1

        return off_decs[:count], off_abilities[:count]
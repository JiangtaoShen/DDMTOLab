"""
Evolutionary Multi-task with Effective Transfer (EMT-ET)

This module implements EMT-ET for multi-task multi-objective optimization problems.

References
----------
    [1] Lin, Jiabin, Hai-Lin Liu, Kay Chen Tan, and Fangqing Gu. "An Effective Knowledge Transfer Approach for Multiobjective Multitasking Optimization." IEEE Transactions on Cybernetics 51.6 (2021): 3238-3248.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.16
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class EMTET:
    """
    Evolutionary Multi-task with Effective Transfer.

    This algorithm features:
    - Adaptive knowledge transfer based on successful transferred solutions
    - Transfer solutions selected from Pareto front of source tasks
    - Distribution-based perturbation for transferred solutions
    - NSGA-II based environmental selection

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, G=8, P=0.5, muc=20.0, mum=15.0,
                 save_data=True, save_path='./TestData', name='EMTET_test', disable_tqdm=True):
        """
        Initialize EMT-ET algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        G : int, optional
            Number of transfer solutions per generation (default: 8)
        P : float, optional
            Probability of distribution-based perturbation (default: 0.5)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EMTET_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.G = G
        self.P = P
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMT-ET algorithm.

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
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize additional attributes for transfer tracking
        # is_trans[t]: boolean array indicating if solution came from transfer
        # origin_task[t]: source task index for transferred solutions
        # front_no[t]: front number from NSGA-II sorting
        is_trans = []
        origin_task = []
        front_no = []
        rank = []

        for t in range(nt):
            is_trans.append(np.zeros(n_per_task[t], dtype=bool))
            origin_task.append(np.full(n_per_task[t], -1, dtype=int))

            # Initial NSGA-II sorting
            rank_t, front_no_t, _ = self._nsga2_sort(objs[t], cons[t])
            front_no.append(front_no_t)
            rank.append(rank_t)

            # Sort population by rank
            sorted_indices = np.argsort(rank_t)
            decs[t] = decs[t][sorted_indices]
            objs[t] = objs[t][sorted_indices]
            cons[t] = cons[t][sorted_indices] if cons[t] is not None else None
            is_trans[t] = is_trans[t][sorted_indices]
            origin_task[t] = origin_task[t][sorted_indices]
            front_no[t] = front_no[t][sorted_indices]

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            active_tasks = [t for t in range(nt) if nfes_per_task[t] < max_nfes_per_task[t]]
            if not active_tasks:
                break

            for t in active_tasks:
                # === Step 1: Transfer ===
                transfer_decs, transfer_is_trans, transfer_origin = self._transfer(
                    decs, objs, is_trans, origin_task, front_no, t, n_per_task
                )

                # Align dimensions for transferred solutions
                target_dim = problem.dims[t]
                transfer_decs_aligned = np.zeros((len(transfer_decs), target_dim))
                for i in range(len(transfer_decs)):
                    transfer_decs_aligned[i] = self._align_dimensions(transfer_decs[i], target_dim)

                # Evaluate transferred solutions
                transfer_objs, transfer_cons = evaluation_single(problem, transfer_decs_aligned, t)
                nfes_per_task[t] += len(transfer_decs_aligned)
                pbar.update(len(transfer_decs_aligned))

                # Reset is_trans for current population (they are no longer "newly transferred")
                is_trans[t] = np.zeros(len(decs[t]), dtype=bool)

                # Merge current population with transferred solutions
                decs[t] = np.vstack([decs[t], transfer_decs_aligned])
                objs[t] = np.vstack([objs[t], transfer_objs])
                cons[t] = np.vstack([cons[t], transfer_cons]) if cons[t] is not None else transfer_cons
                is_trans[t] = np.concatenate([is_trans[t], transfer_is_trans])
                origin_task[t] = np.concatenate([origin_task[t], transfer_origin])

                # Update front numbers for merged population
                rank_t, front_no_t, _ = self._nsga2_sort(objs[t], cons[t])
                front_no[t] = front_no_t

                # === Step 2: Generation ===
                # Create mating pool: first N individuals + G transferred (with rank 1 for transferred)
                current_pop_size = n_per_task[t]
                n_offspring = n_per_task[t] - self.G

                # Tournament selection for mating
                # Use indices [0, N-1] for original population, with transferred solutions having advantage
                mating_pool_size = n_offspring

                mating_ranks = np.concatenate([
                    np.arange(1, current_pop_size + 1),
                    np.ones(self.G)
                ])

                mating_pool = tournament_selection(2, mating_pool_size, mating_ranks)

                # Generate offspring
                parent_decs = decs[t][mating_pool]
                off_decs = ga_generation(parent_decs, muc=self.muc, mum=self.mum)

                # Evaluate offspring
                off_objs, off_cons = evaluation_single(problem, off_decs, t)
                nfes_per_task[t] += len(off_decs)
                pbar.update(len(off_decs))

                # === Step 3: Selection ===
                # Merge all: current (with transferred) + offspring
                decs[t] = np.vstack([decs[t], off_decs])
                objs[t] = np.vstack([objs[t], off_objs])
                cons[t] = np.vstack([cons[t], off_cons]) if cons[t] is not None else off_cons

                # Extend is_trans and origin_task for offspring (not transferred)
                is_trans[t] = np.concatenate([is_trans[t], np.zeros(len(off_decs), dtype=bool)])
                origin_task[t] = np.concatenate([origin_task[t], np.full(len(off_decs), -1, dtype=int)])

                # NSGA-II sorting and selection
                rank_t, front_no_t, _ = self._nsga2_sort(objs[t], cons[t])

                # Select top N individuals
                sorted_indices = np.argsort(rank_t)[:n_per_task[t]]

                decs[t] = decs[t][sorted_indices]
                objs[t] = objs[t][sorted_indices]
                cons[t] = cons[t][sorted_indices] if cons[t] is not None else None
                is_trans[t] = is_trans[t][sorted_indices]
                origin_task[t] = origin_task[t][sorted_indices]
                front_no[t] = front_no_t[sorted_indices]
                rank[t] = rank_t[sorted_indices]

                # Append to history
                append_history(all_decs[t], decs[t], all_objs[t], objs[t], all_cons[t], cons[t])

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _transfer(self, decs, objs, is_trans, origin_task, front_no, t, n_per_task):
        """
        Perform knowledge transfer to task t.

        The transfer strategy:
        1. If there are successful transferred solutions (is_trans=True and FrontNo<2),
           find nearest neighbors from their origin tasks
        2. Otherwise, randomly select solutions from other tasks

        Parameters
        ----------
        decs : list of np.ndarray
            Decision variables for all tasks
        objs : list of np.ndarray
            Objective values for all tasks
        is_trans : list of np.ndarray
            Transfer flags for all tasks
        origin_task : list of np.ndarray
            Origin task indices for all tasks
        front_no : list of np.ndarray
            Front numbers for all tasks
        t : int
            Target task index
        n_per_task : list of int
            Population sizes for all tasks

        Returns
        -------
        transfer_decs : np.ndarray
            Decision variables of transfer solutions
        transfer_is_trans : np.ndarray
            Transfer flags (all True)
        transfer_origin : np.ndarray
            Origin task indices
        """
        nt = len(decs)

        # Find successful transferred solutions (is_trans=True and in first front)
        successful_indices = np.where((is_trans[t] == True) & (front_no[t] < 2))[0]

        transfer_decs_list = []
        transfer_origin_list = []

        if len(successful_indices) > 0:
            # Strategy 1: Find nearest neighbors from origin tasks
            G_temp = int(np.ceil(self.G / len(successful_indices)))

            for s_idx in successful_indices:
                ot = origin_task[t][s_idx]  # Origin task
                if ot < 0 or ot >= nt:
                    continue

                # Calculate distances to all solutions in origin task
                # Need to align dimensions first
                s_dec = decs[t][s_idx]
                ot_decs = decs[ot]

                # Align dimensions for distance calculation
                min_dim = min(len(s_dec), ot_decs.shape[1])
                s_dec_aligned = s_dec[:min_dim]
                ot_decs_aligned = ot_decs[:, :min_dim]

                distances = np.sqrt(np.sum((ot_decs_aligned - s_dec_aligned) ** 2, axis=1))
                nearest_indices = np.argsort(distances)

                # Select G_temp nearest neighbors
                for j in range(min(G_temp, len(nearest_indices))):
                    if len(transfer_decs_list) >= self.G:
                        break
                    idx = nearest_indices[j]
                    transfer_decs_list.append(decs[ot][idx].copy())
                    transfer_origin_list.append(ot)

                if len(transfer_decs_list) >= self.G:
                    break

        # If not enough, use random selection (Strategy 2)
        if len(transfer_decs_list) < self.G:
            task_pool = [k for k in range(nt) if k != t]

            while len(transfer_decs_list) < self.G:
                ot = task_pool[np.random.randint(len(task_pool))]
                idx = np.random.randint(n_per_task[ot])
                transfer_decs_list.append(decs[ot][idx].copy())
                transfer_origin_list.append(ot)

        # Trim to exactly G solutions
        transfer_decs_list = transfer_decs_list[:self.G]
        transfer_origin_list = transfer_origin_list[:self.G]

        # Apply distribution-based perturbation
        for i in range(len(transfer_decs_list)):
            if np.random.rand() < self.P:
                # Perturb: dec = 2 * rand() * dec, then clip to [0, 1]
                transfer_decs_list[i] = 2 * np.random.rand() * transfer_decs_list[i]
                transfer_decs_list[i] = np.clip(transfer_decs_list[i], 0, 1)

        # Convert to arrays
        # Note: transfer_decs may have different dimensions, handle in caller
        transfer_is_trans = np.ones(len(transfer_decs_list), dtype=bool)
        transfer_origin = np.array(transfer_origin_list, dtype=int)

        return transfer_decs_list, transfer_is_trans, transfer_origin

    def _align_dimensions(self, dec, target_dim):
        """
        Align decision variable dimensions to target dimension.

        Parameters
        ----------
        dec : np.ndarray
            Decision variable of shape (dim,)
        target_dim : int
            Target dimension

        Returns
        -------
        aligned_dec : np.ndarray
            Aligned decision variable of shape (target_dim,)
        """
        current_dim = len(dec)
        if current_dim == target_dim:
            return dec.copy()
        elif current_dim < target_dim:
            # Pad with random values
            padding = np.random.rand(target_dim - current_dim)
            return np.concatenate([dec, padding])
        else:
            # Truncate
            return dec[:target_dim].copy()

    def _nsga2_sort(self, objs, cons=None):
        """
        Sort solutions based on NSGA-II criteria.

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (pop_size, n_obj)
        cons : np.ndarray, optional
            Constraint values of shape (pop_size, n_con)

        Returns
        -------
        rank : np.ndarray
            Ranking of each solution
        front_no : np.ndarray
            Front number of each solution
        crowd_dis : np.ndarray
            Crowding distance of each solution
        """
        pop_size = objs.shape[0]

        # Non-dominated sorting
        if cons is not None and cons.size > 0:
            front_no, _ = nd_sort(objs, cons, pop_size)
        else:
            front_no, _ = nd_sort(objs, pop_size)

        # Crowding distance
        crowd_dis = crowding_distance(objs, front_no)

        # Sort by front number (ascending), then by crowding distance (descending)
        sorted_indices = np.lexsort((-crowd_dis, front_no))

        # Create rank array
        rank = np.empty(pop_size, dtype=int)
        rank[sorted_indices] = np.arange(pop_size)

        return rank, front_no, crowd_dis
"""
Evolutionary Multi-task with Population Distribution-based Transfer (EMT-PD)

This module implements EMT-PD for multi-task multi-objective optimization problems.

References
----------
    [1] Liang, Zhengping, Weiqi Liang, Zhiqiang Wang, Xiaoliang Ma, Ling Liu, and Zexuan Zhu. "Multiobjective Evolutionary Multitasking With Two-Stage Adaptive Knowledge Transfer Based on Population Distribution." IEEE Transactions on Systems, Man, and Cybernetics: Systems (2021): 1-13.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.13
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from Methods.Algo_Methods.algo_utils import *


class EMTPD:
    """
    Evolutionary Multi-task with Population Distribution-based Transfer.

    This algorithm features:
    - Two-stage adaptive knowledge transfer based on population distribution
    - Covariance-based distribution alignment between tasks
    - Multifactorial evolutionary framework with RMP
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
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, G=5, muc=20.0, mum=15.0,
                 save_data=True, save_path='./TestData', name='EMTPD_test', disable_tqdm=True):
        """
        Initialize EMT-PD algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.3)
        G : int, optional
            Transfer gap - perform distribution-based transfer every G generations (default: 5)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EMTPD_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.G = G
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMT-PD algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        max_dim = max(dims)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population in unified space (max dimension)
        # Each task's population is padded to max_dim
        decs = []
        for t in range(nt):
            dec_t = np.random.rand(n, max_dim)
            decs.append(dec_t)

        # Evaluate in native space
        objs = []
        cons = []
        for t in range(nt):
            dec_native = decs[t][:, :dims[t]]
            obj_t, con_t = evaluation_single(problem, dec_native, t)
            objs.append(obj_t)
            cons.append(con_t)

        nfes_per_task = [n] * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize MFFactor (skill factor) for each individual
        mf_factors = []
        for t in range(nt):
            mf_factors.append(np.full(n, t, dtype=int))

        # Initial NSGA-II sorting
        ranks = []
        for t in range(nt):
            rank_t, _, _ = self._nsga2_sort(objs[t], cons[t])
            ranks.append(rank_t)
            # Sort population by rank
            sorted_indices = np.argsort(rank_t)
            decs[t] = decs[t][sorted_indices]
            objs[t] = objs[t][sorted_indices]
            cons[t] = cons[t][sorted_indices] if cons[t] is not None else None
            mf_factors[t] = mf_factors[t][sorted_indices]

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        gen = 0
        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            gen += 1

            if gen % self.G != 0:
                # === Standard Generation (MOMFEA-style) ===
                # Merge all populations
                all_pop_decs = np.vstack(decs)
                all_mf_factors = np.concatenate(mf_factors)

                # Tournament selection across all tasks
                all_ranks = np.concatenate([np.arange(n) for _ in range(nt)])  # Use position as rank
                mating_pool = tournament_selection(2, n * nt, all_ranks)

                # Generate offspring
                off_decs, off_mf_factors = self._generation(
                    all_pop_decs[mating_pool], all_mf_factors[mating_pool], max_dim
                )
            else:
                # === Distribution-based Transfer ===
                off_decs, off_mf_factors = self._transfer(decs, n, nt, max_dim)

            # Evaluate offspring for each task
            for t in range(nt):
                # Get offspring belonging to task t
                task_mask = off_mf_factors == t
                off_decs_t = off_decs[task_mask]

                if len(off_decs_t) == 0:
                    continue

                # Evaluate in native space
                off_decs_native = off_decs_t[:, :dims[t]]
                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_native, t)
                nfes_per_task[t] += len(off_decs_t)
                pbar.update(len(off_decs_t))

                # Merge with current population
                decs[t] = np.vstack([decs[t], off_decs_t])
                objs[t] = np.vstack([objs[t], off_objs_t])
                cons[t] = np.vstack([cons[t], off_cons_t]) if cons[t] is not None else off_cons_t
                mf_factors[t] = np.concatenate([mf_factors[t], np.full(len(off_decs_t), t, dtype=int)])

                # NSGA-II selection
                rank_t, _, _ = self._nsga2_sort(objs[t], cons[t])
                sorted_indices = np.argsort(rank_t)[:n]

                decs[t] = decs[t][sorted_indices]
                objs[t] = objs[t][sorted_indices]
                cons[t] = cons[t][sorted_indices] if cons[t] is not None else None
                mf_factors[t] = mf_factors[t][sorted_indices]

            # Convert back to native space for history
            decs_native = []
            for t in range(nt):
                decs_native.append(decs[t][:, :dims[t]])

            append_history(all_decs, decs_native, all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Convert final decs to native space
        final_decs = []
        for t in range(nt):
            final_decs.append(decs[t][:, :dims[t]])

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, parent_decs, parent_mf_factors, max_dim):
        """
        Generate offspring using MOMFEA-style assortative mating.

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables of shape (pop_size, max_dim)
        parent_mf_factors : np.ndarray
            Parent skill factors of shape (pop_size,)
        max_dim : int
            Maximum dimension

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables
        off_mf_factors : np.ndarray
            Offspring skill factors
        """
        pop_size = len(parent_decs)
        off_decs = np.zeros((pop_size, max_dim))
        off_mf_factors = np.zeros(pop_size, dtype=int)

        # Shuffle for random pairing
        shuffled_indices = np.random.permutation(pop_size)

        count = 0
        for i in range(0, pop_size - 1, 2):
            p1_idx = shuffled_indices[i]
            p2_idx = shuffled_indices[i + 1]

            p1_dec = parent_decs[p1_idx]
            p2_dec = parent_decs[p2_idx]
            sf1 = parent_mf_factors[p1_idx]
            sf2 = parent_mf_factors[p2_idx]

            if sf1 == sf2 or np.random.rand() < self.rmp:
                # Crossover
                off_dec1, off_dec2 = crossover(p1_dec, p2_dec, mu=self.muc)
                # Mutation
                off_dec1 = mutation(off_dec1, mu=self.mum)
                off_dec2 = mutation(off_dec2, mu=self.mum)
                # Random skill factor assignment
                off_sf1 = np.random.choice([sf1, sf2])
                off_sf2 = sf1 if off_sf1 == sf2 else sf2
            else:
                # Only mutation (no crossover)
                off_dec1 = mutation(p1_dec.copy(), mu=self.mum)
                off_dec2 = mutation(p2_dec.copy(), mu=self.mum)
                off_sf1 = sf1
                off_sf2 = sf2

            # Boundary handling
            off_dec1 = np.clip(off_dec1, 0, 1)
            off_dec2 = np.clip(off_dec2, 0, 1)

            off_decs[count] = off_dec1
            off_decs[count + 1] = off_dec2
            off_mf_factors[count] = off_sf1
            off_mf_factors[count + 1] = off_sf2
            count += 2

        # Handle odd population size
        if pop_size % 2 == 1:
            last_idx = shuffled_indices[-1]
            off_decs[count] = mutation(parent_decs[last_idx].copy(), mu=self.mum)
            off_decs[count] = np.clip(off_decs[count], 0, 1)
            off_mf_factors[count] = parent_mf_factors[last_idx]

        return off_decs, off_mf_factors

    def _transfer(self, decs, n, nt, max_dim):
        """
        Perform distribution-based knowledge transfer.

        This method uses covariance matrices to align population distributions
        between tasks and generate transferred solutions.

        Parameters
        ----------
        decs : list of np.ndarray
            Decision variables for all tasks
        n : int
            Population size per task
        nt : int
            Number of tasks
        max_dim : int
            Maximum dimension

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables
        off_mf_factors : np.ndarray
            Offspring skill factors
        """
        model_size = min(n, 40)
        off_decs_list = []
        off_mf_factors_list = []

        for t in range(nt):
            # P: top model_size solutions from task t (transposed: max_dim x model_size)
            P = decs[t][:model_size].T

            # Select a random source task
            task_pool = [k for k in range(nt) if k != t]
            k = task_pool[np.random.randint(len(task_pool))]

            # Q: top model_size solutions from task k
            Q = decs[k][:model_size].T

            # Compute covariance matrices A_t and A_k
            A_t = np.zeros((max_dim, max_dim))
            A_k = np.zeros((max_dim, max_dim))

            for dim1 in range(max_dim):
                a = P[dim1, :]
                c = Q[dim1, :]
                for dim2 in range(max_dim):
                    b = P[dim2, :]
                    d = Q[dim2, :]

                    # Covariance between dimensions
                    cov_ab = np.cov(a, b)
                    A_t[dim1, dim2] = cov_ab[0, 1] if cov_ab.shape == (2, 2) else 0

                    cov_cd = np.cov(c, d)
                    A_k[dim1, dim2] = cov_cd[0, 1] if cov_cd.shape == (2, 2) else 0

            # Compute combined distribution parameters
            # A = inv(inv(A_t) + inv(A_k))
            # avg_n = A * (inv(A_t) * avg_P + inv(A_k) * avg_Q)
            avg_P = np.mean(P, axis=1)
            avg_Q = np.mean(Q, axis=1)

            # Add regularization to avoid singular matrices
            reg = 1e-6 * np.eye(max_dim)
            A_t_reg = A_t + reg
            A_k_reg = A_k + reg

            try:
                inv_A_t = np.linalg.inv(A_t_reg)
                inv_A_k = np.linalg.inv(A_k_reg)
                A = np.linalg.inv(inv_A_t + inv_A_k)
                avg_n = A @ (inv_A_t @ avg_P + inv_A_k @ avg_Q)
            except np.linalg.LinAlgError:
                # Fallback to simple average if matrix inversion fails
                avg_n = (avg_P + avg_Q) / 2

            # Normalize avg_n to [0, 1]
            max_n = np.max(avg_n)
            min_n = np.min(avg_n)
            if max_n - min_n > 1e-10:
                avg_n = (avg_n - min_n) / (max_n - min_n)
            else:
                avg_n = np.full(max_dim, 0.5)

            # Compute weight vector
            w1 = avg_P - avg_n

            # Generate offspring
            for i in range(n):
                a_idx = np.random.randint(model_size)
                b_idx = np.random.randint(model_size)

                # Weighted combination
                off_dec = w1 * P[:, a_idx] + (1 - w1) * Q[:, b_idx]

                # Mutation
                off_dec = mutation(off_dec, mu=self.mum)

                # Boundary handling
                off_dec = np.clip(off_dec, 0, 1)

                # Handle NaN values
                nan_mask = np.isnan(off_dec)
                if np.any(nan_mask):
                    off_dec[nan_mask] = np.random.rand(np.sum(nan_mask))

                off_decs_list.append(off_dec)
                off_mf_factors_list.append(t)

        off_decs = np.array(off_decs_list)
        off_mf_factors = np.array(off_mf_factors_list, dtype=int)

        return off_decs, off_mf_factors

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
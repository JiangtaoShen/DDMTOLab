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
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def _sbx_crossover_unclipped(par_dec1, par_dec2, mu):
    """
    Simulated binary crossover (MTO-Platform ``GA_Crossover``).

    Unlike the shared ``crossover`` helper, the offspring are NOT clipped to
    [0, 1] here: SREMTO applies its differential mutation to the raw crossover
    output and clips only once, at the end of Generation.
    """
    d = par_dec1.shape[0]
    u = np.random.rand(d)
    beta = np.zeros(d)
    mask = u <= 0.5
    beta[mask] = (2 * u[mask]) ** (1 / (mu + 1))
    beta[~mask] = (2 * (1 - u[~mask])) ** (-1 / (mu + 1))
    beta *= (-1.0) ** np.random.randint(0, 2, size=d)
    beta[np.random.rand(d) < 0.5] = 1.0

    off_dec1 = 0.5 * ((1 + beta) * par_dec1 + (1 - beta) * par_dec2)
    off_dec2 = 0.5 * ((1 + beta) * par_dec2 + (1 - beta) * par_dec1)
    return off_dec1, off_dec2


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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, th=0.3, p_alpha=0.7, p_beta=1.0,
                 muc=1.0, mum=39.0, save_data=True, save_path='./Data',
                 name='SREMTO', disable_tqdm=True):
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
        ncons = problem.n_cons
        c_max = max(ncons)
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

        # Initialize unified population (all tasks share the same individuals). Each
        # individual carries: dec, mf_obj / mf_cv per task, mf_rank per task, ability.
        pop_size = n * nt
        pop_decs = np.vstack(space_transfer(problem=problem, decs=initialization(problem, n),
                                            type='uni', padding='random'))

        # Track the best solution found so far for each task (feasibility first); it
        # drives the differential mutation and is refreshed on every evaluation
        best_decs = [None] * nt
        best_objs = [np.inf] * nt
        best_cvs = [np.inf] * nt

        def evaluate_on_task(sub_decs, t):
            """Evaluate a block of unified decisions on task t."""
            objs_t, cons_t = evaluation_single(problem, sub_decs[:, :dims[t]], t, unified=True)
            cons_t = cons_t[:, :c_max] if c_max > 0 else np.zeros((sub_decs.shape[0], 0))
            cvs_t = np.sum(np.maximum(0, cons_t), axis=1)
            return objs_t[:, 0], cvs_t, cons_t

        def update_best(sub_decs, sub_objs, sub_cvs, t):
            """Feasibility-first incumbent update for task t (MToP ``Algo.Best``)."""
            idx = constrained_sort(sub_objs, sub_cvs)[0]
            if (sub_cvs[idx], sub_objs[idx]) <= (best_cvs[t], best_objs[t]):
                best_decs[t] = sub_decs[idx].copy()
                best_objs[t] = sub_objs[idx]
                best_cvs[t] = sub_cvs[idx]

        # Evaluate the initial population on every task (multifactorial initialization)
        pop_mf_objs = np.full((pop_size, nt), np.inf)
        pop_mf_cvs = np.full((pop_size, nt), np.inf)
        pop_mf_cons = np.zeros((pop_size, nt, c_max))
        for t in range(nt):
            objs_t, cvs_t, cons_t = evaluate_on_task(pop_decs, t)
            pop_mf_objs[:, t] = objs_t
            pop_mf_cvs[:, t] = cvs_t
            pop_mf_cons[:, t, :] = cons_t
            update_best(pop_decs, objs_t, cvs_t, t)
        nfes = pop_size * nt

        # Calculate initial factorial ranks and ability vectors
        pop_mf_ranks = self._rank_pool(pop_mf_objs, pop_mf_cvs)
        pop_abilities = self._calculate_abilities(pop_mf_ranks, a1, b1, a2, b2, n)

        # Initialize history storage with the top-n subpopulation of each task
        all_decs, all_objs, all_cons = init_history(
            *self._task_views(pop_decs, pop_mf_objs, pop_mf_cons, pop_mf_ranks, dims, ncons, nt, n))

        # Progress bar
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while nfes < max_nfes:
            int_pop_decs = pop_decs.copy()
            int_pop_mf_objs = pop_mf_objs.copy()
            int_pop_mf_cvs = pop_mf_cvs.copy()
            int_pop_mf_cons = pop_mf_cons.copy()
            int_pop_abilities = pop_abilities.copy()

            # Generate offspring for each task
            for t in range(nt):
                # Select parents: the subpopulation of task t, i.e. rank <= n
                parent_indices = np.where(pop_mf_ranks[:, t] <= n)[0]
                if len(parent_indices) < 2:
                    continue

                # Generate offspring
                off_decs, off_abilities = self._generation(
                    pop_decs[parent_indices], pop_abilities[parent_indices], best_decs[t]
                )

                n_off = off_decs.shape[0]
                off_mf_objs = np.full((n_off, nt), np.inf)
                off_mf_cvs = np.full((n_off, nt), np.inf)
                off_mf_cons = np.zeros((n_off, nt, c_max))

                # Decide which offspring is assessed on which task. Task t is always
                # assessed; any other task k only with probability Ability(k). The
                # random draws follow the reference's (individual, task) order.
                eval_mask = np.zeros((n_off, nt), dtype=bool)
                for i in range(n_off):
                    for k in range(nt):
                        if k == t or np.random.rand() < off_abilities[i, k]:
                            eval_mask[i, k] = True

                for k in range(nt):
                    rows = np.where(eval_mask[:, k])[0]
                    if rows.size == 0:
                        continue
                    objs_k, cvs_k, cons_k = evaluate_on_task(off_decs[rows], k)
                    off_mf_objs[rows, k] = objs_k
                    off_mf_cvs[rows, k] = cvs_k
                    off_mf_cons[rows, k, :] = cons_k
                    nfes += rows.size
                    pbar.update(rows.size)
                    update_best(off_decs[rows], objs_k, cvs_k, k)

                # Merge offspring with intermediate population
                int_pop_decs = np.vstack([int_pop_decs, off_decs])
                int_pop_mf_objs = np.vstack([int_pop_mf_objs, off_mf_objs])
                int_pop_mf_cvs = np.vstack([int_pop_mf_cvs, off_mf_cvs])
                int_pop_mf_cons = np.vstack([int_pop_mf_cons, off_mf_cons])
                int_pop_abilities = np.vstack([int_pop_abilities, off_abilities])

            # Selection: rank the whole intermediate pool on every task
            int_pop_mf_ranks = self._rank_pool(int_pop_mf_objs, int_pop_mf_cvs)

            # Keep every individual that is in the top-n of at least one task
            selected_indices = np.unique(np.concatenate(
                [np.where(int_pop_mf_ranks[:, t] <= n)[0] for t in range(nt)]))

            pop_decs = int_pop_decs[selected_indices]
            pop_mf_objs = int_pop_mf_objs[selected_indices]
            pop_mf_cvs = int_pop_mf_cvs[selected_indices]
            pop_mf_cons = int_pop_mf_cons[selected_indices]
            # The survivors keep the ranks they earned in the intermediate pool; they
            # are NOT re-ranked among themselves, otherwise every ability would shift
            pop_mf_ranks = int_pop_mf_ranks[selected_indices]
            pop_abilities = self._calculate_abilities(pop_mf_ranks, a1, b1, a2, b2, n)

            # Store history per task
            task_decs, task_objs, task_cons = self._task_views(
                pop_decs, pop_mf_objs, pop_mf_cons, pop_mf_ranks, dims, ncons, nt, n)
            append_history(all_decs, task_decs, all_objs, task_objs, all_cons, task_cons)

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=max_nfes_per_task, all_cons=all_cons,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    @staticmethod
    def _rank_pool(mf_objs, mf_cvs):
        """
        Rank every member of a pool on every task (constraint violation first).

        Parameters
        ----------
        mf_objs : np.ndarray
            Multifactorial objective values, shape (pool_size, nt)
        mf_cvs : np.ndarray
            Multifactorial constraint violations, shape (pool_size, nt)

        Returns
        -------
        ranks : np.ndarray
            1-based factorial ranks, shape (pool_size, nt)
        """
        pool_size, nt = mf_objs.shape
        ranks = np.zeros((pool_size, nt), dtype=int)
        for t in range(nt):
            order = constrained_sort(mf_objs[:, t], mf_cvs[:, t])
            ranks[order, t] = np.arange(1, pool_size + 1)
        return ranks

    @staticmethod
    def _task_views(pop_decs, mf_objs, mf_cons, mf_ranks, dims, ncons, nt, n):
        """
        Extract the top-n subpopulation of every task in its native search space.

        Parameters
        ----------
        pop_decs : np.ndarray
            Unified decision variables, shape (pop_size, d_max)
        mf_objs : np.ndarray
            Multifactorial objective values, shape (pop_size, nt)
        mf_cons : np.ndarray
            Multifactorial constraint values, shape (pop_size, nt, c_max)
        mf_ranks : np.ndarray
            1-based factorial ranks, shape (pop_size, nt)
        dims : list[int]
            Decision-space dimension of every task
        ncons : list[int]
            Number of constraints of every task
        nt : int
            Number of tasks
        n : int
            Subpopulation size per task

        Returns
        -------
        task_decs, task_objs, task_cons : list[np.ndarray]
            Per-task decision, objective and constraint matrices
        """
        task_decs, task_objs, task_cons = [], [], []
        for t in range(nt):
            idx = np.where(mf_ranks[:, t] <= n)[0]
            task_decs.append(pop_decs[idx][:, :dims[t]].copy())
            task_objs.append(mf_objs[idx, t].reshape(-1, 1).copy())
            task_cons.append(mf_cons[idx, t, :ncons[t]].copy())
        return task_decs, task_objs, task_cons

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
        # Line 1 for the top-n of a task, line 2 for everybody else. The two segments
        # meet at rank n (value TH) and line 2 reaches zero at rank n * nt; ranks past
        # that point would go negative, which the clip turns into "never assessed".
        abilities = np.where(mf_ranks <= n, a1 * mf_ranks + b1, a2 * mf_ranks + b2)
        return np.clip(abilities, 0, 1)

    def _generation(self, parent_decs, parent_abilities, best_dec):
        """
        Generate offspring using SBX crossover and differential mutation.

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables, shape (n_parents, d_max)
        parent_abilities : np.ndarray
            Parent ability vectors, shape (n_parents, nt)
        best_dec : np.ndarray
            Best solution found so far for the current task, shape (d_max,)

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (2 * ceil(n_parents / 2), d_max)
        off_abilities : np.ndarray
            Offspring ability vectors (inherited from parents), same row count
        """
        n_parents, d_max = parent_decs.shape
        nt = parent_abilities.shape[1]
        half = n_parents // 2
        n_pairs = int(np.ceil(n_parents / 2))

        off_decs = np.zeros((2 * n_pairs, d_max))
        off_abilities = np.zeros((2 * n_pairs, nt))

        # Shuffle indices for pairing: the reference pairs order[i] with
        # order[i + floor(N / 2)] for i = 1..ceil(N / 2)
        ind_order = np.random.permutation(n_parents)

        count = 0
        for i in range(n_pairs):
            p1 = ind_order[i]
            p2 = ind_order[i + half]

            if np.random.rand() < self.p_alpha:
                # Crossover
                off_dec1, off_dec2 = _sbx_crossover_unclipped(parent_decs[p1], parent_decs[p2], self.muc)

                # Differential mutation towards the task incumbent; the reference draws
                # an independent scaling factor for each of the two children
                if np.random.rand() < self.p_beta:
                    off_dec1 = off_dec1 + np.random.rand() * (
                        best_dec - off_dec1 + parent_decs[p1] - parent_decs[p2])
                    off_dec2 = off_dec2 + np.random.rand() * (
                        best_dec - off_dec2 + parent_decs[p2] - parent_decs[p1])
            else:
                # Mutation only
                off_dec1 = mutation(parent_decs[p1].copy(), mu=self.mum)
                off_dec2 = mutation(parent_decs[p2].copy(), mu=self.mum)

            # Inherit abilities from parents (imitation)
            off_abilities[count] = parent_abilities[p1].copy()
            off_abilities[count + 1] = parent_abilities[p2].copy()

            # Boundary handling (single repair, after crossover and differential mutation)
            off_decs[count] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1] = np.clip(off_dec2, 0, 1)

            count += 2

        return off_decs, off_abilities
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
Version: 1.1
"""
import time
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
class EMT_PD:
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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, G=5, muc=20.0, mum=15.0,
                 save_data=True, save_path='./Data', name='EMT-PD', disable_tqdm=True):
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
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EMT-PD')
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
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Population lives in the unified [0, 1] space of dimension max(dims); the
        # genes beyond a task's own dimension are initialized at random and keep
        # evolving, exactly as in the MATLAB reference (Dec = rand(1, max(D))).
        decs = space_transfer(problem, initialization(problem, n), type='uni', padding='random')

        objs, cons = [], []
        for t in range(nt):
            obj_t, con_t = evaluation_single(problem, decs[t][:, :dims[t]], t)
            objs.append(obj_t)
            cons.append(con_t)
        nfes = n * nt

        # Sort each task's population by non-dominated rank then crowding distance
        for t in range(nt):
            rank_t, _, _ = nsga2_sort(objs[t], cons[t])
            order = np.argsort(rank_t)
            decs[t], objs[t], cons[t] = decs[t][order], objs[t][order], cons[t][order]

        all_decs, all_objs, all_cons = init_history(
            [decs[t][:, :dims[t]] for t in range(nt)], objs, cons)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        # Skill factor of the concatenated population: task-ordered blocks of size n
        pop_sfs = np.repeat(np.arange(nt), n)
        # Tournament fitness is the position inside the own (sorted) sub-population
        pool_fitness = np.tile(np.arange(n), nt)

        # MTO-Platform increments Algo.Gen inside notTerminated before the loop
        # body runs, so the first executed generation already sees Gen == 2
        gen = 1
        while nfes < max_nfes:
            gen += 1

            if gen % self.G != 0:
                # Generation
                mating_pool = platemo_tournament_selection(2, n * nt, pool_fitness)
                par_decs = np.vstack(decs)[mating_pool, :]
                par_sfs = pop_sfs[mating_pool]
                off_decs, off_sfs = self._generation(par_decs, par_sfs)
            else:
                # Transfer
                off_decs, off_sfs = self._transfer(decs, n, nt)

            for t in range(nt):
                # Evaluation
                mask = off_sfs == t
                if not np.any(mask):
                    continue
                off_decs_t = off_decs[mask, :]
                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_t[:, :dims[t]], t)
                nfes += off_decs_t.shape[0]
                pbar.update(off_decs_t.shape[0])

                # Selection: NSGA-II sorting on the merged parent + offspring pool
                merged_decs, merged_objs, merged_cons = vstack_groups(
                    (decs[t], off_decs_t), (objs[t], off_objs_t), (cons[t], off_cons_t))
                rank_t, _, _ = nsga2_sort(merged_objs, merged_cons)
                index = np.argsort(rank_t)[:n]
                decs[t], objs[t], cons[t] = select_by_index(index, merged_decs, merged_objs, merged_cons)

            append_history(all_decs, [decs[t][:, :dims[t]] for t in range(nt)],
                           all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=max_nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, par_decs, par_sfs):
        """
        Multifactorial offspring generation (MTO-Platform ``EMT_PD.Generation``).

        Parents are paired deterministically as (i, i + floor(L/2)); the mating
        pool order is already randomized by tournament selection. Assortative
        mating applies SBX + polynomial mutation when both parents share a skill
        factor or when a random draw falls below ``rmp``, and polynomial mutation
        alone otherwise. Each child imitates the skill factor of one of the two
        parents drawn independently at random.

        Parameters
        ----------
        par_decs : np.ndarray
            Mating pool decision variables in unified space, shape (L, d_uni)
        par_sfs : np.ndarray
            Mating pool skill factors, shape (L,)

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (2 * ceil(L / 2), d_uni)
        off_sfs : np.ndarray
            Offspring skill factors, shape (2 * ceil(L / 2),)
        """
        length, d = par_decs.shape
        half = length // 2
        n_pairs = int(np.ceil(length / 2))
        off_decs = np.empty((2 * n_pairs, d))
        off_sfs = np.empty(2 * n_pairs, dtype=int)

        count = 0
        for i in range(n_pairs):
            p1, p2 = i, i + half
            sf1, sf2 = int(par_sfs[p1]), int(par_sfs[p2])

            if sf1 == sf2 or np.random.rand() < self.rmp:
                # Crossover
                off_dec1, off_dec2 = sbx_crossover_unclipped(par_decs[p1, :], par_decs[p2, :], self.muc)
                # Mutation
                off_dec1 = poly_mutation_unclipped(off_dec1, self.mum)
                off_dec2 = poly_mutation_unclipped(off_dec2, self.mum)
                # Imitation: each child picks one of the two parents independently
                pair = (sf1, sf2)
                off_sfs[count] = pair[np.random.randint(2)]
                off_sfs[count + 1] = pair[np.random.randint(2)]
            else:
                # Mutation only
                off_dec1 = poly_mutation_unclipped(par_decs[p1, :], self.mum)
                off_dec2 = poly_mutation_unclipped(par_decs[p2, :], self.mum)
                # Imitation
                off_sfs[count] = sf1
                off_sfs[count + 1] = sf2

            off_decs[count, :] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1, :] = np.clip(off_dec2, 0, 1)
            count += 2

        return off_decs, off_sfs

    def _transfer(self, decs, n, nt):
        """
        Population distribution based knowledge transfer (MTO-Platform ``EMT_PD.Transfer``).

        For every task the elite front of its own population and of a randomly
        chosen partner task are modelled by their sample covariance matrices.
        The precision-weighted mean of the two Gaussians is min-max normalized
        and its distance to the own mean defines a per-variable blend weight
        used to recombine one solution from each task.

        Parameters
        ----------
        decs : list of np.ndarray
            Sorted population of each task in unified space, shape (n, d_uni)
        n : int
            Population size per task
        nt : int
            Number of tasks

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (n * nt, d_uni)
        off_sfs : np.ndarray
            Offspring skill factors, shape (n * nt,)
        """
        d = decs[0].shape[1]
        model_size = min(n, 40)
        off_decs = np.empty((n * nt, d))
        off_sfs = np.empty(n * nt, dtype=int)

        count = 0
        for t in range(nt):
            p = decs[t][:model_size, :].T
            task_pool = [k for k in range(nt) if k != t]
            k = task_pool[np.random.randint(len(task_pool))]
            q = decs[k][:model_size, :].T

            # Sample covariance matrices over the unified decision variables
            a_t = np.cov(p)
            a_k = np.cov(q)
            avg_p = np.mean(p, axis=1)
            avg_q = np.mean(q, axis=1)

            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                try:
                    # MATLAB inv() of a rank deficient covariance returns Inf and
                    # the resulting NaN decision variables are resampled below
                    inv_a_t = np.linalg.inv(a_t)
                    inv_a_k = np.linalg.inv(a_k)
                    a = np.linalg.inv(inv_a_t + inv_a_k)
                    avg_n = a @ (inv_a_t @ avg_p + inv_a_k @ avg_q)
                except np.linalg.LinAlgError:
                    avg_n = np.full(d, np.nan)

                avg_n = (avg_n - np.min(avg_n)) / (np.max(avg_n) - np.min(avg_n))
                w1 = avg_p - avg_n

            for _ in range(n):
                a_idx = np.random.randint(model_size)
                b_idx = np.random.randint(model_size)
                with np.errstate(over='ignore', invalid='ignore'):
                    off_dec = w1 * p[:, a_idx] + (1 - w1) * q[:, b_idx]
                off_dec = poly_mutation_unclipped(off_dec, self.mum)
                off_dec = np.clip(off_dec, 0, 1)
                nan_mask = np.isnan(off_dec)
                if np.any(nan_mask):
                    off_dec[nan_mask] = np.random.rand(int(np.sum(nan_mask)))
                off_decs[count, :] = off_dec
                off_sfs[count] = t
                count += 1

        return off_decs, off_sfs

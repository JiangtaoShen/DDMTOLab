"""
Multi-task Evolutionary Algorithm with Self-adaptive Solvers (MTEA-SaO)

This module implements MTEA-SaO for multi-task single-objective optimization problems.

References
----------
    [1] Li, Yanchi, Wenyin Gong, and Shuijia Li. "Multitasking Optimization via an Adaptive Solver Multitasking Evolutionary Framework." Information Sciences (2022). https://doi.org/10.1016/j.ins.2022.10.099

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.02
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from Methods.Algo_Methods.algo_utils import *


class MTEASaO:
    """
    Multi-task Evolutionary Algorithm with Self-adaptive Solvers for single-objective optimization.

    This algorithm features:
    - Two solver strategies: GA (SBX + PM) and DE (DE/rand/1/bin)
    - Self-adaptive solver selection based on success/failure history
    - Knowledge transfer between tasks via random solution injection
    - Adaptive population partitioning among solvers

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
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, t_gap=10, t_num=10, sa_gap=70,
                 memory=30, ga_muc=2.0, ga_mum=5.0, de_f=0.5, de_cr=0.9,
                 save_data=True, save_path='./TestData', name='MTEASaO_test', disable_tqdm=True):
        """
        Initialize MTEA-SaO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        t_gap : int, optional
            Transfer gap - perform knowledge transfer every t_gap generations (default: 10)
        t_num : int, optional
            Number of solutions to transfer (default: 10)
        sa_gap : int, optional
            Self-adaptive gap - update solver allocation every sa_gap generations (default: 70)
        memory : int, optional
            Memory length for success/failure history (default: 30)
        ga_muc : float, optional
            Distribution index for GA crossover (SBX) (default: 2.0)
        ga_mum : float, optional
            Distribution index for GA mutation (PM) (default: 5.0)
        de_f : float, optional
            DE scaling factor (default: 0.5)
        de_cr : float, optional
            DE crossover probability (default: 0.9)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MTEASaO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.t_gap = t_gap
        self.t_num = t_num
        self.sa_gap = sa_gap
        self.memory = memory
        self.ga_muc = ga_muc
        self.ga_mum = ga_mum
        self.de_f = de_f
        self.de_cr = de_cr
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTEA-SaO algorithm.

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

        # Strategy settings
        st_num = 2  # Number of strategies (GA, DE)

        # Initialize strategy population sizes (equal split)
        # stn[t] = [size_strategy_1, size_strategy_2]
        stn = []
        for t in range(nt):
            base_size = n_per_task[t] // st_num
            sizes = [base_size] * st_num
            sizes[-1] = n_per_task[t] - sum(sizes[:-1])  # Adjust last strategy size
            stn.append(sizes)

        # Success/failure history
        succ_history = []  # List of arrays with shape (nt, st_num)
        fail_history = []

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        gen = 0
        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            gen += 1
            succ_iter = np.zeros((nt, st_num))
            fail_iter = np.zeros((nt, st_num))

            for t in range(nt):
                if nfes_per_task[t] >= max_nfes_per_task[t]:
                    continue

                # Calculate median objectives and CV for comparison
                cvs = np.sum(np.maximum(0, cons[t]), axis=1) if cons[t] is not None and cons[t].size > 0 else np.zeros(len(objs[t]))
                median_obj = np.median(objs[t][:, 0])
                median_cv = np.median(cvs)

                # Prepare parent population (with potential transfer)
                parent_decs = decs[t].copy()

                # Knowledge Transfer
                # Condition: t_num > 0 AND within the window before memory period AND at transfer gap
                if (self.t_num > 0 and
                    (gen - 1) % self.sa_gap + 1 < (self.sa_gap - self.memory) and
                    gen % self.t_gap == 0):
                    # Get transfer solutions from other tasks
                    transfer_decs = self._transfer(decs, t, problem.dims)
                    if len(transfer_decs) > 0:
                        # Randomly replace some parents with transferred solutions
                        replace_indices = np.random.permutation(len(parent_decs))[:len(transfer_decs)]
                        for i, idx in enumerate(replace_indices):
                            parent_decs[idx] = transfer_decs[i]

                # Process each strategy
                start_idx = 0
                for st in range(st_num):
                    end_idx = start_idx + stn[t][st]
                    if end_idx <= start_idx:
                        start_idx = end_idx
                        continue

                    st_indices = list(range(start_idx, end_idx))
                    st_parent_decs = parent_decs[st_indices]

                    if st == 0:
                        # Strategy 1: GA (SBX + PM) with elitist selection
                        off_decs = self._generation_ga(st_parent_decs)
                        off_objs, off_cons = evaluation_single(problem, off_decs, t)
                        nfes_per_task[t] += len(off_decs)
                        pbar.update(len(off_decs))

                        # Elitist selection: compare parent vs offspring one-by-one
                        for i, idx in enumerate(st_indices):
                            parent_cv = cvs[idx]
                            parent_obj = objs[t][idx, 0]
                            off_cv = np.sum(np.maximum(0, off_cons[i])) if off_cons is not None and off_cons[i].size > 0 else 0
                            off_obj = off_objs[i, 0]

                            # Constrained comparison: prefer lower CV, then lower objective
                            if off_cv < parent_cv or (off_cv == parent_cv and off_obj < parent_obj):
                                decs[t][idx] = off_decs[i]
                                objs[t][idx] = off_objs[i]
                                if cons[t] is not None:
                                    cons[t][idx] = off_cons[i]

                    else:
                        # Strategy 2: DE with tournament selection
                        off_decs = self._generation_de(st_parent_decs)
                        off_objs, off_cons = evaluation_single(problem, off_decs, t)
                        nfes_per_task[t] += len(off_decs)
                        pbar.update(len(off_decs))

                        # Tournament selection: compare parent vs offspring one-by-one
                        for i, idx in enumerate(st_indices):
                            parent_cv = cvs[idx]
                            parent_obj = objs[t][idx, 0]
                            off_cv = np.sum(np.maximum(0, off_cons[i])) if off_cons is not None and off_cons[i].size > 0 else 0
                            off_obj = off_objs[i, 0]

                            # Constrained comparison: prefer lower CV, then lower objective
                            if off_cv < parent_cv or (off_cv == parent_cv and off_obj <= parent_obj):
                                decs[t][idx] = off_decs[i]
                                objs[t][idx] = off_objs[i]
                                if cons[t] is not None:
                                    cons[t][idx] = off_cons[i]

                    # Calculate success/failure for this strategy
                    # Recalculate CVs after potential updates
                    current_cvs = np.sum(np.maximum(0, cons[t][st_indices]), axis=1) if cons[t] is not None and cons[t].size > 0 else np.zeros(len(st_indices))
                    current_objs = objs[t][st_indices, 0]

                    # Success: CV < median_cv OR (CV == median_cv AND obj < median_obj)
                    succ_mask = (current_cvs < median_cv) | (
                        (current_cvs == median_cv) & (current_objs < median_obj)
                    )
                    succ_iter[t, st] = np.sum(succ_mask)

                    # Failure: CV > median_cv OR (CV == median_cv AND obj > median_obj)
                    fail_mask = (current_cvs > median_cv) | (
                        (current_cvs == median_cv) & (current_objs > median_obj)
                    )
                    fail_iter[t, st] = np.sum(fail_mask)

                    start_idx = end_idx

                # Append to history
                append_history(all_decs[t], decs[t], all_objs[t], objs[t], all_cons[t], cons[t])

            # Update success/failure history
            succ_history.append(succ_iter)
            fail_history.append(fail_iter)

            # Trim history to memory length (per-task adjustment)
            max_history_len = self.memory * nt
            if len(succ_history) > max_history_len:
                succ_history = succ_history[-max_history_len:]
                fail_history = fail_history[-max_history_len:]

            # Update strategy population sizes
            for t in range(nt):
                # Aggregate history for task t (every nt-th entry belongs to all tasks)
                succ_t = np.zeros(st_num)
                fail_t = np.zeros(st_num)
                for i in range(len(succ_history)):
                    succ_t += succ_history[i][t]
                    fail_t += fail_history[i][t]

                # Calculate success probability
                succ_p = np.zeros(st_num)
                for st in range(st_num):
                    total = succ_t[st] + fail_t[st]
                    if total == 0:
                        succ_p[st] = 0.01
                    else:
                        succ_p[st] = succ_t[st] / total + 0.01

                # Combine with previous allocation (momentum)
                succ_old = np.array(stn[t]) / sum(stn[t])
                succ_p = succ_old / 2 + succ_p
                succ_p = succ_p / np.sum(succ_p)

                # Update allocation every sa_gap generations
                if gen % self.sa_gap == 0:
                    new_sizes = (succ_p * n_per_task[t]).astype(int)
                    new_sizes[-1] = n_per_task[t] - np.sum(new_sizes[:-1])
                    stn[t] = list(new_sizes)

                    # Shuffle population to redistribute among strategies
                    shuffle_indices = np.random.permutation(n_per_task[t])
                    decs[t] = decs[t][shuffle_indices]
                    objs[t] = objs[t][shuffle_indices]
                    if cons[t] is not None:
                        cons[t] = cons[t][shuffle_indices]

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _transfer(self, decs, t, dims):
        """
        Perform random knowledge transfer from other tasks.

        Parameters
        ----------
        decs : list of np.ndarray
            Decision variables for all tasks
        t : int
            Current task index
        dims : list of int
            Dimensions for all tasks

        Returns
        -------
        transfer_decs : np.ndarray
            Transferred decision variables aligned to target task dimension
        """
        nt = len(decs)
        if nt <= 1:
            return np.array([])

        transfer_decs = []
        other_tasks = [k for k in range(nt) if k != t]

        for _ in range(self.t_num):
            # Random task and random individual
            rand_t = other_tasks[np.random.randint(len(other_tasks))]
            rand_p = np.random.randint(len(decs[rand_t]))
            dec = decs[rand_t][rand_p].copy()

            # Align dimensions to target task
            dec = self._align_dimensions(dec, dims[t])
            transfer_decs.append(dec)

        return np.array(transfer_decs)

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
            # Pad with random values in [0, 1]
            padding = np.random.rand(target_dim - current_dim)
            return np.concatenate([dec, padding])
        else:
            # Truncate to target dimension
            return dec[:target_dim].copy()

    def _generation_ga(self, parent_decs):
        """
        Generate offspring using GA (SBX crossover + polynomial mutation).

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables of shape (pop_size, dim)

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables of shape (pop_size, dim)
        """
        pop_size = len(parent_decs)

        if pop_size <= 1:
            # Only mutation for very small population
            off_decs = np.array([mutation(parent_decs[i].copy(), mu=self.ga_mum) for i in range(pop_size)])
            return np.clip(off_decs, 0, 1)

        dim = parent_decs.shape[1]
        off_decs = np.zeros((pop_size, dim))

        # Shuffle indices for pairing
        ind_order = np.random.permutation(pop_size)

        count = 0
        for i in range(pop_size // 2):
            p1 = ind_order[i]
            p2 = ind_order[i + pop_size // 2]

            # Crossover
            off_dec1, off_dec2 = crossover(parent_decs[p1], parent_decs[p2], mu=self.ga_muc)

            # Mutation
            off_dec1 = mutation(off_dec1, mu=self.ga_mum)
            off_dec2 = mutation(off_dec2, mu=self.ga_mum)

            # Boundary handling
            off_decs[count] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1] = np.clip(off_dec2, 0, 1)
            count += 2

        # Handle odd population size
        if pop_size % 2 == 1:
            off_decs[count] = mutation(parent_decs[ind_order[-1]].copy(), mu=self.ga_mum)
            off_decs[count] = np.clip(off_decs[count], 0, 1)

        return off_decs[:pop_size]

    def _generation_de(self, parent_decs):
        """
        Generate offspring using DE (DE/rand/1/bin).

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables of shape (pop_size, dim)

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables of shape (pop_size, dim)
        """
        pop_size = len(parent_decs)
        dim = parent_decs.shape[1]

        if pop_size < 4:
            # Fallback to mutation only when population too small for DE
            off_decs = np.array([mutation(parent_decs[i].copy(), mu=self.ga_mum) for i in range(pop_size)])
            return np.clip(off_decs, 0, 1)

        off_decs = np.zeros((pop_size, dim))

        for i in range(pop_size):
            # Select 3 different individuals (different from i)
            candidates = list(range(pop_size))
            candidates.remove(i)
            selected = np.random.choice(candidates, 3, replace=False)
            x1, x2, x3 = selected

            # DE/rand/1 mutation
            mutant = parent_decs[x1] + self.de_f * (parent_decs[x2] - parent_decs[x3])

            # DE binomial crossover
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < self.de_cr
            mask[j_rand] = True  # Ensure at least one dimension from mutant
            off_dec = np.where(mask, mutant, parent_decs[i])

            # Boundary handling
            off_decs[i] = np.clip(off_dec, 0, 1)

        return off_decs
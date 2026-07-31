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
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def _sbx_crossover_unclipped(par_dec1, par_dec2, mu):
    """
    Simulated binary crossover (MTO-Platform ``GA_Crossover``).

    Unlike the shared ``crossover`` helper the offspring are NOT clipped to
    [0, 1]: the MATLAB reference clips only once at the end of Generation_GA,
    after mutation has acted on the raw crossover output.
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


def _poly_mutation_unclipped(dec, mu):
    """
    Polynomial mutation (MTO-Platform ``GA_Mutation``) with prob 1/D per gene.

    Operates on the possibly out-of-bounds crossover output and does NOT
    clip; the caller clips once afterwards, matching the MATLAB reference.
    """
    d = dec.shape[0]
    dec = dec.copy()
    prob_m = 1 / d
    for j in range(d):
        if np.random.rand() < prob_m:
            u = np.random.rand()
            if u <= 0.5:
                delta = (2 * u + (1 - 2 * u) * (1 - dec[j]) ** (mu + 1)) ** (1 / (mu + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * dec[j] ** (mu + 1)) ** (1 / (mu + 1))
            dec[j] += delta
    return dec


class MTEA_SaO:
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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
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
                 save_data=True, save_path='./Data', name='MTEA-SaO', disable_tqdm=True):
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
        dims = problem.dims
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # MToP evolves one unified genome of length max(D) per individual and
        # truncates to D(t) only at evaluation time. Keeping the unified space
        # here matters because a transferred individual is injected verbatim
        # into another task, so its genes beyond the source dimension become
        # active decision variables of the target task. Padding is U[0, 1] to
        # match MToP's rand(1, max(D)) initialization.
        pop_decs = space_transfer(problem=problem, decs=decs, type='uni', padding='random')

        # Strategy settings
        st_num = 2  # Number of strategies (GA, DE)

        # Initialize strategy population sizes (equal split, remainder to last)
        stn = []
        for t in range(nt):
            sizes = [n_per_task[t] // st_num] * st_num
            sizes[-1] = n_per_task[t] - sum(sizes[:-1])
            stn.append(sizes)

        # Success/failure history. MToP stores a flat matrix that receives T
        # rows (one per task) every generation and trims it with
        # succ(end - Memory * T : end, :), i.e. it keeps Memory * T + 1 rows;
        # the per-task slices are then read back as succ(t:T:end, :). The
        # structure is ported verbatim so the memory window matches.
        succ_rows = np.zeros((0, st_num))
        fail_rows = np.zeros((0, st_num))

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        # MToP increments Algo.Gen once before entering the loop body, so the
        # first generation the reference executes already has Gen == 2.
        gen = 1
        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            gen += 1
            succ_iter = np.zeros((nt, st_num))
            fail_iter = np.zeros((nt, st_num))

            for t in range(nt):
                if nfes_per_task[t] >= max_nfes_per_task[t]:
                    continue

                # Median objective / CV of the whole task population, taken
                # before the transferred solutions are injected
                cvs = np.sum(np.maximum(0, cons[t]), axis=1)
                median_obj = np.median(objs[t][:, 0])
                median_cv = np.median(cvs)

                # Parent pool used only to generate offspring
                parent_decs = pop_decs[t].copy()

                # Knowledge Transfer
                if (self.t_num > 0 and
                        (gen - 1) % self.sa_gap + 1 < (self.sa_gap - self.memory) and
                        gen % self.t_gap == 0):
                    transfer_decs = self._transfer(pop_decs, t)
                    n_transfer = min(len(transfer_decs), len(parent_decs))
                    if n_transfer > 0:
                        replace_indices = np.random.permutation(len(parent_decs))[:n_transfer]
                        parent_decs[replace_indices] = transfer_decs[:n_transfer]

                # Process each strategy
                start_idx = 0
                for st in range(st_num):
                    end_idx = start_idx + stn[t][st]
                    if end_idx <= start_idx:
                        start_idx = end_idx
                        continue

                    st_indices = np.arange(start_idx, end_idx)
                    st_parent_decs = parent_decs[st_indices]

                    if st == 0:
                        # Strategy 1: GA (SBX + PM) with elitist selection over
                        # the strategy sub-population (MToP Selection_Elit)
                        off_decs = self._generation_ga(st_parent_decs)
                        off_objs, off_cons = evaluation_single(problem, off_decs[:, :dims[t]], t)
                        nfes_per_task[t] += len(off_decs)
                        pbar.update(len(off_decs))

                        merged_decs = np.vstack([pop_decs[t][st_indices], off_decs])
                        merged_objs = np.vstack([objs[t][st_indices], off_objs])
                        merged_cons = np.vstack([cons[t][st_indices], off_cons])
                        sel = selection_elit(merged_objs, len(st_indices), merged_cons)
                        pop_decs[t][st_indices] = merged_decs[sel]
                        objs[t][st_indices] = merged_objs[sel]
                        cons[t][st_indices] = merged_cons[sel]
                    else:
                        # Strategy 2: DE (DE/rand/1/bin) with pairwise
                        # tournament selection (MToP Selection_Tournament, Ep=0)
                        off_decs = self._generation_de(st_parent_decs)
                        off_objs, off_cons = evaluation_single(problem, off_decs[:, :dims[t]], t)
                        nfes_per_task[t] += len(off_decs)
                        pbar.update(len(off_decs))

                        pop_cv = np.sum(np.maximum(0, cons[t][st_indices]), axis=1)
                        off_cv = np.sum(np.maximum(0, off_cons), axis=1)
                        pop_obj = objs[t][st_indices, 0]
                        off_obj = off_objs[:, 0]

                        # Both infeasible: the offspring must reduce the CV.
                        # Both feasible: the offspring must strictly improve
                        # the objective. A feasible offspring never replaces an
                        # infeasible parent, exactly as in MToP.
                        replace_cv = (pop_cv > off_cv) & (pop_cv > 0) & (off_cv > 0)
                        equal_cv = (pop_cv <= 0) & (off_cv <= 0)
                        replace = (equal_cv & (pop_obj > off_obj)) | replace_cv

                        rep_idx = st_indices[replace]
                        pop_decs[t][rep_idx] = off_decs[replace]
                        objs[t][rep_idx] = off_objs[replace]
                        cons[t][rep_idx] = off_cons[replace]

                    # Success / failure of this strategy, measured on the
                    # updated sub-population against the pre-generation medians
                    current_cvs = np.sum(np.maximum(0, cons[t][st_indices]), axis=1)
                    current_objs = objs[t][st_indices, 0]

                    succ_iter[t, st] = np.sum(
                        (current_cvs < median_cv) |
                        ((current_cvs == median_cv) & (current_objs < median_obj))
                    )
                    fail_iter[t, st] = np.sum(
                        (current_cvs > median_cv) |
                        ((current_cvs == median_cv) & (current_objs > median_obj))
                    )

                    start_idx = end_idx

                # Append to history in the task's own (real) decision space
                append_history(all_decs[t], pop_decs[t][:, :dims[t]],
                               all_objs[t], objs[t], all_cons[t], cons[t])

            # Update success/failure history (T rows per generation)
            succ_rows = np.vstack([succ_rows, succ_iter])
            fail_rows = np.vstack([fail_rows, fail_iter])
            if succ_rows.shape[0] > self.memory * nt:
                keep = self.memory * nt + 1
                succ_rows = succ_rows[-keep:, :]
                fail_rows = fail_rows[-keep:, :]

            # Update strategy population sizes
            for t in range(nt):
                succ_t = succ_rows[t::nt, :]
                fail_t = fail_rows[t::nt, :]

                # Calculate success probability
                succ_p = np.zeros(st_num)
                for st in range(st_num):
                    total = succ_t[:, st].sum() + fail_t[:, st].sum()
                    if total == 0:
                        succ_p[st] = 0.01
                    else:
                        succ_p[st] = succ_t[:, st].sum() / total + 0.01

                # Combine with the previous allocation
                succ_old = np.array(stn[t], dtype=float) / sum(stn[t])
                succ_p = succ_old / 2 + succ_p
                succ_p = succ_p / np.sum(succ_p)

                # Update allocation every sa_gap generations
                if gen % self.sa_gap == 0:
                    new_sizes = (succ_p * n_per_task[t]).astype(int)
                    new_sizes[-1] = n_per_task[t] - np.sum(new_sizes[:-1])
                    stn[t] = list(new_sizes)

                    # Shuffle population to redistribute among strategies
                    shuffle_indices = np.random.permutation(n_per_task[t])
                    pop_decs[t] = pop_decs[t][shuffle_indices]
                    objs[t] = objs[t][shuffle_indices]
                    cons[t] = cons[t][shuffle_indices]

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _transfer(self, pop_decs, t):
        """
        Perform random knowledge transfer from other tasks.

        MToP builds an archive holding the populations of every other task and
        draws ``TNum`` individuals from it, each time picking a random task and
        then a random individual. The unified genome is copied verbatim.

        Parameters
        ----------
        pop_decs : list of np.ndarray
            Decision variables (unified space) for all tasks
        t : int
            Current task index

        Returns
        -------
        transfer_decs : np.ndarray
            Transferred decision variables of shape (t_num, d_max)
        """
        dim = pop_decs[t].shape[1]
        archive = [k for k in range(len(pop_decs)) if k != t]
        if not archive:
            return np.zeros((0, dim))

        transfer_decs = np.empty((self.t_num, dim))
        for i in range(self.t_num):
            rand_t = archive[np.random.randint(len(archive))]
            rand_p = np.random.randint(pop_decs[rand_t].shape[0])
            transfer_decs[i] = pop_decs[rand_t][rand_p]
        return transfer_decs

    def _generation_ga(self, parent_decs):
        """
        Generate offspring using GA (SBX crossover + polynomial mutation).

        Matches MToP ``Generation_GA``: for a sub-population of at most one
        individual only polynomial mutation is applied; otherwise a random
        permutation is split in halves, parent i is paired with parent
        i + floor(N/2) for i = 1..ceil(N/2), the single clip to [0, 1] happens
        after mutation, and the offspring list is truncated back to N.

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables of shape (pop_size, dim)

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables of shape (pop_size, dim)
        """
        pop_size, dim = parent_decs.shape

        if pop_size <= 1:
            off_decs = np.empty((pop_size, dim))
            for i in range(pop_size):
                off_decs[i] = _poly_mutation_unclipped(parent_decs[i], self.ga_mum)
            return off_decs

        order = np.random.permutation(pop_size)
        half = pop_size // 2
        n_pairs = int(np.ceil(pop_size / 2))
        off_decs = np.empty((2 * n_pairs, dim))

        count = 0
        for i in range(n_pairs):
            p1 = parent_decs[order[i]]
            p2 = parent_decs[order[i + half]]

            c1, c2 = _sbx_crossover_unclipped(p1, p2, self.ga_muc)
            c1 = _poly_mutation_unclipped(c1, self.ga_mum)
            c2 = _poly_mutation_unclipped(c2, self.ga_mum)

            off_decs[count] = np.clip(c1, 0, 1)
            off_decs[count + 1] = np.clip(c2, 0, 1)
            count += 2

        return off_decs[:pop_size]

    def _generation_de(self, parent_decs):
        """
        Generate offspring using DE (DE/rand/1/bin).

        Matches MToP ``Generation_DE``: sub-populations with fewer than four
        individuals fall back to polynomial mutation only.

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables of shape (pop_size, dim)

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables of shape (pop_size, dim)
        """
        pop_size, dim = parent_decs.shape

        if pop_size < 4:
            off_decs = np.empty((pop_size, dim))
            for i in range(pop_size):
                off_decs[i] = _poly_mutation_unclipped(parent_decs[i], self.ga_mum)
            return off_decs

        off_decs = np.empty((pop_size, dim))
        for i in range(pop_size):
            candidates = np.arange(pop_size)
            candidates = candidates[candidates != i]
            x1, x2, x3 = np.random.choice(candidates, 3, replace=False)

            # DE/rand/1 mutation
            trial = parent_decs[x1] + self.de_f * (parent_decs[x2] - parent_decs[x3])

            # DE binomial crossover (MToP DE_Crossover)
            replace = np.random.rand(dim) > self.de_cr
            replace[np.random.randint(dim)] = False
            trial = np.where(replace, parent_decs[i], trial)

            # Boundary handling
            off_decs[i] = np.clip(trial, 0, 1)

        return off_decs

"""
Multi-objective Multi-task Evolutionary Algorithm with Progressive Auto-Encoding (MO-MTEA-PAE)

This module implements MO-MTEA-PAE for multi-task multi-objective optimization problems.

References
----------
    [1] Q. Gu, Y. Li, W. Gong, Z. Yuan, B. Ning, C. Hu, and J. Wu, "Progressive Auto-Encoding for Domain Adaptation in Evolutionary Multi-Task Optimization," Applied Soft Computing, vol. 175, p. 113916, 2025.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def _lhsdesign(n, d):
    """Latin hypercube sample of shape (n, d) in [0, 1] (MATLAB ``lhsdesign``)."""
    edges = np.linspace(0, 1, n + 1)
    matrix = np.zeros((n, d))
    for j in range(d):
        samples = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(samples)
        matrix[:, j] = samples
    return matrix
class MO_MTEA_PAE:
    """
    Multi-objective Multi-task Evolutionary Algorithm with Progressive Auto-Encoding.

    This algorithm features:

    - Kernelized autoencoding (NFC) for cross-task knowledge transfer
    - Two transfer strategies: segment transfer (historical distribution) and
      stochastic replacement transfer (current distribution)
    - Adaptive selection between DE and GA offspring generation
    - Adaptive selection between transfer types based on success rates
    - SPEA2 environmental selection per task
    - Elite solution transfer across tasks

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
        # NFC regresses the target population onto the source population, so
        # the two design matrices must share their row count
        'n': 'equal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None,
                 Seg=10, TNum=20, TGap=5,
                 F=0.5, CR=0.9, MuC=20, MuM=15,
                 save_data=True, save_path='./Data',
                 name='MO-MTEA-PAE', disable_tqdm=True):
        """
        Initialize MO-MTEA-PAE algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        Seg : int, optional
            Number of segments for the DisPop update schedule (default: 10)
        TNum : int, optional
            Number of transfer solutions per transfer event (default: 20)
        TGap : int, optional
            Transfer gap in generations (default: 5)
        F : float, optional
            DE mutation factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.9)
        MuC : float, optional
            SBX crossover distribution index (default: 20)
        MuM : float, optional
            PM mutation distribution index (default: 15)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MO-MTEA-PAE')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.Seg = Seg
        self.TNum = TNum
        self.TGap = TGap
        self.F = F
        self.CR = CR
        self.MuC = MuC
        self.MuM = MuM
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MO-MTEA-PAE algorithm.

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
        dims = problem.dims
        d_max = max(dims)

        # TNum = min(TNum, fix(N / 2))
        TNum = min(self.TNum, min(n_per_task) // 2)

        # SegGap = fix(maxFE / (T * N * Seg)); DDMTOLab's per-task budget times
        # T is MToP's Prob.maxFE, so the T factor cancels.
        SegGap = max(1, max_nfes_per_task[0] // (n_per_task[0] * self.Seg))

        # Initialize populations in the unified space (Latin hypercube, as in
        # MToP's lhsdesign(Prob.N, max(Prob.D)))
        decs = []
        objs = []
        cons = []
        for t in range(nt):
            decs_t = _lhsdesign(n_per_task[t], d_max)
            objs_t, cons_t = evaluation_single(problem, decs_t[:, :dims[t]], t)
            decs.append(decs_t)
            objs.append(objs_t)
            cons.append(cons_t)
        nfes_per_task = list(n_per_task)

        # SPEA2 selection of the initial population (sorts it by fitness)
        fit = []
        for t in range(nt):
            sel_idx, fit_t = self._spea2_select(objs[t], cons[t], n_per_task[t])
            decs[t] = decs[t][sel_idx]
            objs[t] = objs[t][sel_idx]
            cons[t] = cons[t][sel_idx]
            fit.append(fit_t[sel_idx])

        # History in each task's own decision space
        all_decs = [[decs[t][:, :dims[t]].copy()] for t in range(nt)]
        all_objs = [[objs[t].copy()] for t in range(nt)]
        all_cons = [[cons[t].copy()] for t in range(nt)]

        # Archive and DisPop
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]
        arc_cons = [c.copy() for c in cons]
        dis_decs = [d.copy() for d in decs]
        dis_objs = [o.copy() for o in objs]
        dis_cons = [c.copy() for c in cons]

        # KT / OP flags of the current population (parents always carry 0)
        kt_flags = [np.zeros(n_per_task[t], dtype=int) for t in range(nt)]
        op_flags = [np.zeros(n_per_task[t], dtype=int) for t in range(nt)]

        # Success tracking (cumulative)
        succ_t = np.full((nt, 2), float(TNum))                              # [segment, stochastic]
        sum_t = succ_t.copy()
        succ_g = np.array([[float(n_per_task[t])] * 2 for t in range(nt)])  # [DE, GA]
        sum_g = succ_g.copy()

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # MToP increments Algo.Gen once before entering the loop body, so the
        # first generation the reference executes already has Gen == 2.
        gen = 1

        while sum(nfes_per_task) < total_nfes:
            gen += 1

            # === Step 1: Generate offspring ===
            off_decs_list = []
            off_kt_list = []
            off_op_list = []

            for t in range(nt):
                Nt = n_per_task[t]
                pG = succ_g[t] / sum_g[t]

                off_decs_t, off_op_t = self._generation(decs[t], Nt, pG, d_max)
                off_decs_list.append(off_decs_t)
                off_kt_list.append(np.zeros(Nt, dtype=int))
                off_op_list.append(off_op_t)

            # === Step 2: Knowledge transfer ===
            if TNum > 0 and gen % self.TGap == 0:
                # MToP saves Arc, overwrites Arc{t} with the (re-sorted)
                # current population so the stochastic branch regresses on it,
                # and restores Arc at the end of the block. The stored archive
                # is therefore unchanged by the transfer, so it is simply kept
                # aside here and the sorted populations are held separately.
                sorted_pop_decs = []
                for t in range(nt):
                    # Re-sort DisPop by SPEA2 fitness
                    sel_idx, _ = self._spea2_select(dis_objs[t], dis_cons[t], n_per_task[t])
                    dis_decs[t] = dis_decs[t][sel_idx]
                    dis_objs[t] = dis_objs[t][sel_idx]
                    dis_cons[t] = dis_cons[t][sel_idx]

                    # Arc{t} = Pop{t}, then re-sorted by SPEA2 fitness
                    sel_idx2, _ = self._spea2_select(objs[t], cons[t], n_per_task[t])
                    sorted_pop_decs.append(decs[t][sel_idx2])

                for t in range(nt):
                    k = np.random.randint(nt)
                    while k == t:
                        k = np.random.randint(nt)

                    Nk = n_per_task[k]

                    # --- Segment transfer (historical distributions) ---
                    nd_idx = np.where(fit[k] < 1)[0]
                    if len(nd_idx) < TNum:
                        s_best_idx = np.arange(min(TNum, Nk))
                    else:
                        s_best_idx = nd_idx[np.random.permutation(len(nd_idx))[:TNum]]
                    s_best_decs = decs[k][s_best_idx, :dims[k]]

                    seg_dec = self._nfc(dis_decs[t][:, :dims[t]],
                                        dis_decs[k][:, :dims[k]],
                                        s_best_decs)
                    if dims[t] < d_max:
                        seg_dec = np.hstack([seg_dec,
                                             np.random.rand(seg_dec.shape[0], d_max - dims[t])])

                    # --- Stochastic replacement transfer (current populations) ---
                    s_best_decs2 = sorted_pop_decs[k][:TNum, :dims[k]]
                    sto_dec = self._nfc(sorted_pop_decs[t][:, :dims[t]],
                                        sorted_pop_decs[k][:, :dims[k]],
                                        s_best_decs2)
                    if dims[t] < d_max:
                        sto_dec = np.hstack([sto_dec,
                                             np.random.rand(sto_dec.shape[0], d_max - dims[t])])

                    # Choose between the two transfer types per solution
                    pT = succ_t[t] / sum_t[t]
                    tr_decs = np.zeros((TNum, d_max))
                    tr_kt = np.zeros(TNum, dtype=int)
                    for i in range(TNum):
                        if np.random.rand() < pT[0] / (pT[0] + pT[1]):
                            tr_decs[i] = seg_dec[i]
                            tr_kt[i] = 1  # segment
                        else:
                            tr_decs[i] = sto_dec[i]
                            tr_kt[i] = 2  # stochastic
                    tr_decs = np.clip(tr_decs, 0, 1)

                    replace_idx = np.random.permutation(off_decs_list[t].shape[0])[:TNum]
                    off_decs_list[t][replace_idx] = tr_decs
                    off_kt_list[t][replace_idx] = tr_kt
                    off_op_list[t][replace_idx] = 0

            # Refresh the historical distribution every SegGap generations
            if gen % SegGap == 0:
                dis_decs = [d.copy() for d in decs]
                dis_objs = [o.copy() for o in objs]
                dis_cons = [c.copy() for c in cons]

            # === Step 3: Environmental selection ===
            for t in range(nt):
                if nfes_per_task[t] >= max_nfes_per_task[t]:
                    continue

                Nt = n_per_task[t]

                # Parents carry no credit into the next comparison
                kt_flags[t] = np.zeros(Nt, dtype=int)
                op_flags[t] = np.zeros(Nt, dtype=int)

                # Elite solution transfer: one random offspring is replaced by
                # the best individual of another task
                k = np.random.randint(nt)
                while k == t:
                    k = np.random.randint(nt)
                rnd_idx = np.random.randint(Nt)
                off_decs_list[t][rnd_idx] = decs[k][0].copy()
                off_kt_list[t][rnd_idx] = 3
                off_op_list[t][rnd_idx] = 0

                # Attempt counters (updated before evaluation, as in MToP)
                sum_t[t, 0] += np.sum(off_kt_list[t] == 1)
                sum_t[t, 1] += np.sum(off_kt_list[t] == 2)
                sum_g[t, 0] += np.sum(off_op_list[t] == 1)
                sum_g[t, 1] += np.sum(off_op_list[t] == 2)

                off_objs_t, off_cons_t = evaluation_single(
                    problem, off_decs_list[t][:, :dims[t]], t)
                nfes_per_task[t] += off_decs_list[t].shape[0]
                pbar.update(off_decs_list[t].shape[0])

                merged_decs = np.vstack([decs[t], off_decs_list[t]])
                merged_objs = np.vstack([objs[t], off_objs_t])
                merged_cons = np.vstack([cons[t], off_cons_t])
                merged_kt = np.concatenate([kt_flags[t], off_kt_list[t]])
                merged_op = np.concatenate([op_flags[t], off_op_list[t]])

                sel_idx, fit_all = self._spea2_select(merged_objs, merged_cons, Nt)

                failed_mask = np.ones(merged_decs.shape[0], dtype=bool)
                failed_mask[sel_idx] = False
                failed_idx = np.where(failed_mask)[0]

                # Success counters
                succ_t[t, 0] += np.sum(merged_kt[sel_idx] == 1)
                succ_t[t, 1] += np.sum(merged_kt[sel_idx] == 2)
                succ_g[t, 0] += np.sum(merged_op[sel_idx] == 1)
                succ_g[t, 1] += np.sum(merged_op[sel_idx] == 2)

                decs[t] = merged_decs[sel_idx]
                objs[t] = merged_objs[sel_idx]
                cons[t] = merged_cons[sel_idx]
                fit[t] = fit_all[sel_idx]
                kt_flags[t] = merged_kt[sel_idx]
                op_flags[t] = merged_op[sel_idx]

                # Archive receives the discarded solutions, then is resampled
                # down to N (MToP always applies randperm, so it also shuffles
                # an archive that is already of size N)
                if failed_idx.size > 0:
                    arc_decs[t] = np.vstack([arc_decs[t], merged_decs[failed_idx]])
                    arc_objs[t] = np.vstack([arc_objs[t], merged_objs[failed_idx]])
                    arc_cons[t] = np.vstack([arc_cons[t], merged_cons[failed_idx]])

                perm = np.random.permutation(arc_decs[t].shape[0])[:Nt]
                arc_decs[t] = arc_decs[t][perm]
                arc_objs[t] = arc_objs[t][perm]
                arc_cons[t] = arc_cons[t][perm]

                append_history(all_decs[t], decs[t][:, :dims[t]],
                               all_objs[t], objs[t],
                               all_cons[t], cons[t])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, pop_decs, N, pG, d_max):
        """
        Generate offspring with the adaptive DE / GA operator pool.

        Matches MToP ``Generation``: the DE branch draws its two difference
        parents with rank-proportional acceptance ``(N - rank) / N`` and its
        base index uniformly, the GA branch pairs the tournament order i with
        i + floor(N/2), and every offspring is clipped once at the end.

        Parameters
        ----------
        pop_decs : np.ndarray, shape (N, d_max)
            Parent population, sorted by SPEA2 fitness (index 0 = best)
        N : int
            Population size
        pG : np.ndarray, shape (2,)
            Success rates of [DE, GA]
        d_max : int
            Unified genome width

        Returns
        -------
        off_decs : np.ndarray, shape (N, d_max)
        op_flags : np.ndarray, shape (N,)
            1 = DE, 2 = GA
        """
        off_decs = np.zeros((N, d_max))
        op_flags = np.zeros(N, dtype=int)

        # Tournament order for the GA branch (fitness = positional rank 1..N)
        rank_vals = np.arange(1, N + 1)
        ind_order = platemo_tournament_selection(2, 2 * N, rank_vals)

        denom = pG[0] + pG[1]
        p_de = pG[0] / denom if denom > 0 else 0.5

        for i in range(N):
            if np.random.rand() < p_de:
                # DE/rand/1 with rank-proportional parent acceptance
                x1 = self._rank_selection(N, exclude=(i,))
                x2 = self._rank_selection(N, exclude=(i, x1))
                x3 = self._random_selection(N, exclude=(i, x1, x2))

                trial = pop_decs[x1] + self.F * (pop_decs[x2] - pop_decs[x3])

                # DE binomial crossover (MToP DE_Crossover)
                replace = np.random.rand(d_max) > self.CR
                replace[np.random.randint(d_max)] = False
                off_decs[i] = np.where(replace, pop_decs[i], trial)
                op_flags[i] = 1
            else:
                # GA: SBX crossover + polynomial mutation, single clip below
                p1 = ind_order[i]
                p2 = ind_order[i + N // 2]
                child, _ = sbx_crossover_unclipped(pop_decs[p1], pop_decs[p2], self.MuC)
                off_decs[i] = poly_mutation_unclipped(child, self.MuM)
                op_flags[i] = 2

            off_decs[i] = np.clip(off_decs[i], 0, 1)

        return off_decs, op_flags

    @staticmethod
    def _rank_selection(N, exclude=()):
        """
        Rank-proportional parent draw (lower index = better rank).

        MToP repeats ``x = randi(N)`` while ``rand() > (N - rank(x)) / N`` or
        ``x`` is excluded, so index x (rank x + 1 in 0-based terms) is accepted
        with probability (N - rank) / N.
        """
        for _ in range(1000):
            x = np.random.randint(N)
            if x not in exclude and np.random.rand() < (N - (x + 1)) / N:
                return x
        candidates = [i for i in range(N) if i not in exclude]
        return int(np.random.choice(candidates)) if candidates else 0

    @staticmethod
    def _random_selection(N, exclude=()):
        """Uniform draw over the indices that are not excluded."""
        candidates = [i for i in range(N) if i not in exclude]
        return int(np.random.choice(candidates)) if candidates else 0

    @staticmethod
    def _spea2_select(objs, cons, N, epsilon=0.0):
        """
        SPEA2 environmental selection (MToP ``Selection_SPEA2``).

        Parameters
        ----------
        objs : np.ndarray
            Objective values, shape (n_pop, n_obj)
        cons : np.ndarray or None
            Constraint values, shape (n_pop, n_con)
        N : int
            Target population size
        epsilon : float, optional
            Epsilon-constraint tolerance (default: 0.0)

        Returns
        -------
        sel_idx : np.ndarray
            Indices of the survivors, sorted by ascending fitness
        fitness : np.ndarray
            SPEA2 fitness of every input solution
        """
        pop_size = objs.shape[0]
        N = max(0, min(N, pop_size))

        if cons is None or cons.size == 0:
            cv = np.zeros(pop_size)
        else:
            cv = np.sum(np.maximum(0, cons), axis=1)
        cv = np.where(cv < epsilon, 0.0, cv)

        fitness = spea2_fitness(objs, cv[:, None])
        if N == 0:
            return np.zeros(0, dtype=int), fitness

        next_mask = fitness < 1
        n_selected = int(np.sum(next_mask))

        if n_selected < N:
            order = np.argsort(fitness, kind='stable')
            next_mask = np.zeros(pop_size, dtype=bool)
            next_mask[order[:N]] = True
        elif n_selected > N:
            selected_idx = np.where(next_mask)[0]
            keep_idx = spea2_truncation(objs[selected_idx], N)
            next_mask = np.zeros(pop_size, dtype=bool)
            next_mask[selected_idx[keep_idx]] = True

        sel_idx = np.where(next_mask)[0]
        sel_idx = sel_idx[np.argsort(fitness[sel_idx], kind='stable')]

        return sel_idx, fitness

    @staticmethod
    def _nfc(target_pop, source_pop, source_best, kernel='poly'):
        """
        Kernelized autoencoding transfer (MToP ``NFC``).

        Maps ``source_best`` from the source task to the target task space
        using a kernel ridge map learned between the two population
        distributions.

        Parameters
        ----------
        target_pop : np.ndarray, shape (N, D_target)
            Target task population distribution
        source_pop : np.ndarray, shape (N, D_source)
            Source task population distribution
        source_best : np.ndarray, shape (TNum, D_source)
            Solutions to transfer from the source task
        kernel : str, optional
            Kernel type (default: 'poly')

        Returns
        -------
        mapped : np.ndarray, shape (TNum, D_target)
        """
        D_target = target_pop.shape[1]
        D_source = source_pop.shape[1]

        T_H = np.asarray(target_pop, dtype=np.float64)
        S_H = np.asarray(source_pop, dtype=np.float64)
        if D_target < D_source:
            T_H = np.hstack([T_H, np.zeros((T_H.shape[0], D_source - D_target))])
        elif D_target > D_source:
            S_H = np.hstack([S_H, np.zeros((S_H.shape[0], D_target - D_source))])

        # Features in rows, samples in columns
        S_H_T = S_H.T
        T_H_T = T_H.T

        kk = MO_MTEA_PAE._kernelmatrix(kernel, S_H_T, S_H_T)

        d = kk.shape[0]
        Q0 = kk @ kk.T
        P = T_H_T @ kk.T
        reg = 1e-5 * np.eye(d)
        W = P @ np.linalg.pinv(Q0 + reg)

        S_Best = np.asarray(source_best, dtype=np.float64)
        if D_target <= D_source:
            K_map = MO_MTEA_PAE._kernelmatrix(kernel, S_H_T, S_Best.T)
            return (W @ K_map).T[:, :D_target]

        S_Best = np.hstack([S_Best, np.zeros((S_Best.shape[0], D_target - D_source))])
        K_map = MO_MTEA_PAE._kernelmatrix(kernel, S_H_T, S_Best.T)
        return (W @ K_map).T

    @staticmethod
    def _kernelmatrix(kernel, X, X2):
        """
        Kernel matrix between column-format data (MToP ``kernelmatrix``).

        Parameters
        ----------
        kernel : str
            Kernel type ('poly', 'lin', 'rbf')
        X : np.ndarray, shape (dim, N1)
        X2 : np.ndarray, shape (dim, N2)

        Returns
        -------
        K : np.ndarray, shape (N1, N2)
        """
        d1 = X.shape[0]
        d2 = X2.shape[0]
        if d1 < d2:
            X = np.vstack([X, np.zeros((d2 - d1, X.shape[1]))])
        elif d1 > d2:
            X2 = np.vstack([X2, np.zeros((d1 - d2, X2.shape[1]))])

        if kernel == 'poly':
            b, d = 0.1, 5
            return (X.T @ X2 + b) ** d
        elif kernel == 'lin':
            return X.T @ X2
        elif kernel == 'rbf':
            n1sq = np.sum(X ** 2, axis=0)
            n2sq = np.sum(X2 ** 2, axis=0)
            D = n1sq[:, None] + n2sq[None, :] - 2 * X.T @ X2
            return np.exp(-D / (2 * 0.1 ** 2))
        else:
            raise ValueError(f'Unsupported kernel: {kernel}')

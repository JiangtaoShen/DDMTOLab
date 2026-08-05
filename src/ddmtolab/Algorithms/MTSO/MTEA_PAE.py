"""
Multi-Task Evolutionary Algorithm with Progressive Auto-Encoding (MTEA-PAE)

This module implements MTEA-PAE for multi-task optimization using kernelized
autoencoding (NFC) for cross-task knowledge transfer with adaptive operator
and transfer type selection.

References
----------
    [1] Gu, Qiong, et al. "Progressive Auto-Encoding for Domain Adaptation
        in Evolutionary Multi-Task Optimization." Applied Soft Computing,
        113916, 2025.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def _lhsdesign(n, d):
    """Latin hypercube sample of shape (n, d) in [0, 1] (MATLAB lhsdesign)."""
    edges = np.linspace(0, 1, n + 1)
    matrix = np.zeros((n, d))
    for j in range(d):
        samples = np.random.uniform(edges[:-1], edges[1:])
        np.random.shuffle(samples)
        matrix[:, j] = samples
    return matrix


def _kernelmatrix_poly(X, X2):
    """Polynomial kernel matrix K = (X^T @ X2 + 0.1)^5.

    Parameters
    ----------
    X : np.ndarray, shape (D, N1)
        Feature matrix (features x samples)
    X2 : np.ndarray, shape (D, N2)
        Feature matrix (features x samples)

    Returns
    -------
    K : np.ndarray, shape (N1, N2)
    """
    d1, d2 = X.shape[0], X2.shape[0]
    if d1 < d2:
        X = np.vstack([X, np.zeros((d2 - d1, X.shape[1]))])
    elif d2 < d1:
        X2 = np.vstack([X2, np.zeros((d1 - d2, X2.shape[1]))])
    return (X.T @ X2 + 0.1) ** 5


def _nfc(T_H, S_H, S_Best):
    """Kernelized autoencoding transfer (NFC).

    Transfers source solutions to target domain using kernel ridge regression
    learned from historical population distributions.

    Parameters
    ----------
    T_H : np.ndarray, shape (N_t, D_t)
        Target history population
    S_H : np.ndarray, shape (N_s, D_k)
        Source history population
    S_Best : np.ndarray, shape (M, D_k)
        Source solutions to transfer

    Returns
    -------
    result : np.ndarray, shape (M, D_t)
        Transferred solutions in target space
    """
    D_t = T_H.shape[1]
    D_k = S_H.shape[1]

    # Pad dimensions to match
    T_H_pad = T_H.copy()
    S_H_pad = S_H.copy()
    if D_t < D_k:
        T_H_pad = np.hstack([T_H_pad,
                             np.zeros((T_H_pad.shape[0], D_k - D_t))])
    elif D_t > D_k:
        S_H_pad = np.hstack([S_H_pad,
                             np.zeros((S_H_pad.shape[0], D_t - D_k))])

    # Transpose to features x samples
    S_T = S_H_pad.T
    T_T = T_H_pad.T

    # Self-kernel K(S, S)
    K = _kernelmatrix_poly(S_T, S_T)

    # Solve for mapping W = P * pinv(Q0 + reg)
    Q0 = K @ K.T
    P = T_T @ K.T
    reg = 1e-5 * np.eye(Q0.shape[0])
    W = P @ np.linalg.pinv(Q0 + reg)

    # Transform source best
    if D_t <= D_k:
        K_new = _kernelmatrix_poly(S_T, S_Best.T)
        result = (W @ K_new).T
        return result[:, :D_t]
    else:
        S_Best_pad = np.hstack([S_Best,
                                np.zeros((S_Best.shape[0], D_t - D_k))])
        K_new = _kernelmatrix_poly(S_T, S_Best_pad.T)
        return (W @ K_new).T


def _constrained_sort(objs, cons):
    """Sort indices by constraint violation then objective (ascending).

    Parameters
    ----------
    objs : np.ndarray, shape (N, 1)
    cons : np.ndarray, shape (N, C)

    Returns
    -------
    rank : np.ndarray
        Sorted indices
    """
    if cons.shape[1] > 0:
        cv = np.sum(np.maximum(0, cons), axis=1)
    else:
        cv = np.zeros(len(objs))
    return np.lexsort((objs.flatten(), cv))


class MTEA_PAE:
    """
    Multi-Task Evolutionary Algorithm with Progressive Auto-Encoding.

    Uses kernelized autoencoding (NFC) for cross-task knowledge transfer
    with two transfer strategies: segment transfer (using current distribution)
    and stochastic replacement transfer (using archive). Adaptive selection
    between DE/rand/1/bin and GA (SBX+PM) for offspring generation.

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

    def __init__(self, problem, n=None, max_nfes=None, Seg=10, TNum=20,
                 TGap=5, F=0.5, CR=0.9, MuC=2, MuM=5,
                 save_data=True, save_path='./Data', name='MTEA-PAE',
                 disable_tqdm=True):
        """
        Initialize MTEA-PAE algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        Seg : int, optional
            Number of segments for distribution snapshots (default: 10)
        TNum : int, optional
            Number of solutions to transfer (default: 20)
        TGap : int, optional
            Generation gap between transfers (default: 5)
        F : float, optional
            DE mutation scale factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.9)
        MuC : float, optional
            SBX distribution index (default: 2)
        MuM : float, optional
            PM distribution index (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTEA-PAE')
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
        Execute the MTEA-PAE algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives,
            and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes = self.max_nfes * nt
        maxD = max(dims)
        TNum = min(self.TNum, n // 2)
        # Generation gap between distribution snapshots
        SegGap = max(1, self.max_nfes // (n * self.Seg))

        # Initialize populations by Latin hypercube sampling in the unified
        # [0, 1]^maxD space (MATLAB: Decs = lhsdesign(Prob.N, max(Prob.D)))
        pop_decs = [_lhsdesign(n, maxD) for _ in range(nt)]
        decs = [pop_decs[t][:, :dims[t]] for t in range(nt)]
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        # Report what each task actually consumed rather than the requested
        # budget, which the analysis tools use to scale convergence curves
        nfes_per_task = [n] * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pop_objs = [o.copy() for o in objs]
        pop_cons = [c.copy() for c in cons]

        # Sort each population by [CVs, Objs]
        for t in range(nt):
            rank = _constrained_sort(pop_objs[t], pop_cons[t])
            pop_decs[t] = pop_decs[t][rank]
            pop_objs[t] = pop_objs[t][rank]
            pop_cons[t] = pop_cons[t][rank]

        # Archive and distribution reference (copies of initial sorted pop)
        arc_decs = [p.copy() for p in pop_decs]
        arc_objs = [o.copy() for o in pop_objs]
        arc_cons = [c.copy() for c in pop_cons]
        dis_decs = [p.copy() for p in pop_decs]
        dis_objs = [o.copy() for o in pop_objs]
        dis_cons = [c.copy() for c in pop_cons]

        # Adaptive success tracking
        SuccT = TNum * np.ones((nt, 2))
        SumT = SuccT.copy()
        SuccG = n * np.ones((nt, 2))
        SumG = SuccG.copy()

        # MATLAB's Algo.Gen is already 2 inside the first loop body, because
        # notTerminated() counted the (evaluated) initialization generation.
        gen = 1
        pbar = tqdm(total=max_nfes, initial=nfes, desc=self.name,
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            gen += 1

            # --- Generate offspring per task ---
            off_decs = []
            off_kt = []
            off_op = []
            for t in range(nt):
                pG = SuccG[t] / SumG[t]
                od, oop = self._generation(pop_decs[t], pG, n, maxD)
                off_decs.append(od)
                off_kt.append(np.zeros(n, dtype=int))
                off_op.append(oop)

            # --- Knowledge transfer every TGap generations ---
            if TNum > 0 and gen % self.TGap == 0:
                # Backup archive before temporary modifications
                arc_bk_d = [a.copy() for a in arc_decs]
                arc_bk_o = [a.copy() for a in arc_objs]
                arc_bk_c = [a.copy() for a in arc_cons]

                # Sort DisPop and merge Arc with Pop
                for t in range(nt):
                    rank = _constrained_sort(dis_objs[t], dis_cons[t])
                    dis_decs[t] = dis_decs[t][rank]
                    dis_objs[t] = dis_objs[t][rank]
                    dis_cons[t] = dis_cons[t][rank]

                    arc_decs[t] = np.vstack([arc_decs[t], pop_decs[t]])
                    arc_objs[t] = np.vstack([arc_objs[t], pop_objs[t]])
                    arc_cons[t] = np.vstack([arc_cons[t], pop_cons[t]])
                    rank = _constrained_sort(arc_objs[t], arc_cons[t])
                    arc_decs[t] = arc_decs[t][rank]
                    arc_objs[t] = arc_objs[t][rank]
                    arc_cons[t] = arc_cons[t][rank]

                # Perform NFC transfer for each task
                for t in range(nt):
                    k = np.random.randint(nt)
                    while k == t:
                        k = np.random.randint(nt)

                    # Segment Transfer using DisPop
                    sBest_seg = pop_decs[k][:TNum, :dims[k]]
                    tDis = dis_decs[t][:, :dims[t]]
                    kDis = dis_decs[k][:, :dims[k]]
                    SegDec = _nfc(tDis, kDis, sBest_seg)
                    if dims[t] < maxD:
                        SegDec = np.hstack([SegDec,
                                            np.random.rand(TNum,
                                                           maxD - dims[t])])

                    # Stochastic Transfer using Archive
                    sBest_sto = arc_decs[k][:TNum, :dims[k]]
                    tArc = arc_decs[t][:, :dims[t]]
                    kArc = arc_decs[k][:, :dims[k]]
                    StoDec = _nfc(tArc, kArc, sBest_sto)
                    if dims[t] < maxD:
                        StoDec = np.hstack([StoDec,
                                            np.random.rand(TNum,
                                                           maxD - dims[t])])

                    # Probabilistic selection between transfer types
                    pT = SuccT[t] / SumT[t]
                    p_seg = pT[0] / (pT[0] + pT[1])

                    replace_idx = np.random.choice(n, TNum, replace=False)
                    for i, idx in enumerate(replace_idx):
                        if np.random.rand() < p_seg:
                            off_decs[t][idx] = np.clip(SegDec[i], 0, 1)
                            off_kt[t][idx] = 1
                        else:
                            off_decs[t][idx] = np.clip(StoDec[i], 0, 1)
                            off_kt[t][idx] = 2
                        off_op[t][idx] = 0

                # Restore archive from backup
                arc_decs = arc_bk_d
                arc_objs = arc_bk_o
                arc_cons = arc_bk_c

            # --- Environmental selection ---
            for t in range(nt):
                # Elite solution transfer from random source task
                k = np.random.randint(nt)
                while k == t:
                    k = np.random.randint(nt)
                rnd_idx = np.random.randint(n)
                off_decs[t][rnd_idx] = pop_decs[k][0].copy()
                off_kt[t][rnd_idx] = 3
                off_op[t][rnd_idx] = 0

                # Count submissions
                SumT[t, 0] += np.sum(off_kt[t] == 1)
                SumT[t, 1] += np.sum(off_kt[t] == 2)
                SumG[t, 0] += np.sum(off_op[t] == 1)
                SumG[t, 1] += np.sum(off_op[t] == 2)

                # Evaluate offspring
                off_objs_t, off_cons_t = evaluation_single(
                    problem, off_decs[t][:, :dims[t]], t)
                nfes += n
                nfes_per_task[t] += n
                pbar.update(n)

                # Combine parent + offspring
                merged_decs = np.vstack([pop_decs[t], off_decs[t]])
                merged_objs = np.vstack([pop_objs[t], off_objs_t])
                merged_cons = np.vstack([pop_cons[t], off_cons_t])
                merged_kt = np.concatenate(
                    [np.zeros(n, dtype=int), off_kt[t]])
                merged_op = np.concatenate(
                    [np.zeros(n, dtype=int), off_op[t]])

                # Elitist selection
                sel = selection_elit(objs=merged_objs, n=n,
                                     cons=merged_cons)

                # Sort selected population (needed for rank-biased DE)
                sub_rank = _constrained_sort(merged_objs[sel],
                                             merged_cons[sel])
                sel_sorted = sel[sub_rank]

                # Count successes from surviving offspring
                SuccT[t, 0] += np.sum(merged_kt[sel_sorted] == 1)
                SuccT[t, 1] += np.sum(merged_kt[sel_sorted] == 2)
                SuccG[t, 0] += np.sum(merged_op[sel_sorted] == 1)
                SuccG[t, 1] += np.sum(merged_op[sel_sorted] == 2)

                # Failed individuals (not selected)
                all_idx = set(range(2 * n))
                failed_idx = np.array(sorted(all_idx - set(sel_sorted)))

                # Update archive with failed individuals
                arc_decs[t] = np.vstack([arc_decs[t],
                                         merged_decs[failed_idx]])
                arc_objs[t] = np.vstack([arc_objs[t],
                                         merged_objs[failed_idx]])
                arc_cons[t] = np.vstack([arc_cons[t],
                                         merged_cons[failed_idx]])
                # Random sample archive to N
                perm = np.random.permutation(len(arc_decs[t]))[:n]
                arc_decs[t] = arc_decs[t][perm]
                arc_objs[t] = arc_objs[t][perm]
                arc_cons[t] = arc_cons[t][perm]

                # Update population (sorted)
                pop_decs[t] = merged_decs[sel_sorted]
                pop_objs[t] = merged_objs[sel_sorted]
                pop_cons[t] = merged_cons[sel_sorted]

            # Update distribution reference every SegGap generations
            if gen % SegGap == 0:
                dis_decs = [p.copy() for p in pop_decs]
                dis_objs = [o.copy() for o in pop_objs]
                dis_cons = [c.copy() for c in pop_cons]

            # Record history in task-specific dimensions
            real_decs = [pop_decs[t][:, :dims[t]] for t in range(nt)]
            append_history(all_decs, real_decs, all_objs, pop_objs,
                           all_cons, pop_cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, pop_decs, pG, n, maxD):
        """Generate offspring with adaptive DE/GA selection.

        Parameters
        ----------
        pop_decs : np.ndarray, shape (n, maxD)
            Current sorted population
        pG : np.ndarray, shape (2,)
            Success ratio for [DE, GA]
        n : int
            Population size
        maxD : int
            Maximum dimensionality

        Returns
        -------
        off_decs : np.ndarray, shape (n, maxD)
        op : np.ndarray, shape (n,)
            Operator type (1=DE, 2=GA)
        """
        off_decs = pop_decs.copy()
        op = np.zeros(n, dtype=int)

        # Pre-compute tournament selection for GA parents
        indorder = np.zeros(2 * n, dtype=int)
        for j in range(2 * n):
            a, b = np.random.randint(n, size=2)
            indorder[j] = min(a, b)  # lower index = better (sorted pop)

        p_de = pG[0] / (pG[0] + pG[1])

        for i in range(n):
            if np.random.rand() < p_de:
                # DE/rand/1/bin with rank-biased donor selection
                x1 = np.random.randint(n)
                itr = 0
                while ((np.random.rand() > (n - 1 - x1) / n or x1 == i)
                       and itr < 1000):
                    x1 = np.random.randint(n)
                    itr += 1

                x2 = np.random.randint(n)
                itr = 0
                while ((np.random.rand() > (n - 1 - x2) / n
                        or x2 == i or x2 == x1) and itr < 1000):
                    x2 = np.random.randint(n)
                    itr += 1

                x3 = np.random.randint(n)
                while x3 == i or x3 == x1 or x3 == x2:
                    x3 = np.random.randint(n)

                # DE/rand/1 mutation + binomial crossover
                v = pop_decs[x1] + self.F * (pop_decs[x2] - pop_decs[x3])
                u = pop_decs[i].copy()
                j_rand = np.random.randint(maxD)
                mask = np.random.rand(maxD) < self.CR
                mask[j_rand] = True
                u[mask] = v[mask]
                off_decs[i] = u
                op[i] = 1
            else:
                # GA: SBX crossover + polynomial mutation
                # NOTE: crossover()/mutation() take 1-D decision vectors; a
                # (1, D) array would make SBX draw a single scalar beta and
                # polynomial mutation use prob_m = 1/1.
                p1 = indorder[i]
                p2 = indorder[i + n // 2]
                child, _ = crossover(pop_decs[p1], pop_decs[p2], mu=self.MuC)
                child = mutation(child, mu=self.MuM)
                off_decs[i] = child
                op[i] = 2

            off_decs[i] = np.clip(off_decs[i], 0, 1)

        return off_decs, op

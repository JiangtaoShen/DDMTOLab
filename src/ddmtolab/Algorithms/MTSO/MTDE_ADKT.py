"""
Multitask Differential Evolution with Adaptive Dual Knowledge Transfer (MTDE-ADKT)

This module implements MTDE-ADKT for multi-task optimization using SHADE-based
adaptive parameters, distribution-aligned knowledge transfer, and adaptive
transfer probability control.

References
----------
    [1] Zhang, Tingyu, et al. "Multitask Differential Evolution with Adaptive
        Dual Knowledge Transfer." Applied Soft Computing, 165: 112040, 2024.

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


class MTDE_ADKT:
    """
    Multitask DE with Adaptive Dual Knowledge Transfer.

    Combines SHADE-based adaptive F/CR with two knowledge transfer modes:
    - Type 1: Distribution-aligned transfer via covariance whitening/coloring
    - Type 2: Direct transfer from source task
    - Type 3: Standard DE/current-to-pbest/1 (no transfer)

    Transfer probabilities (RMP1, RMP2) are adaptively adjusted based on
    the relative success rates of each transfer type.

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

    def __init__(self, problem, n=None, max_nfes=None, P=0.1, H=100,
                 Gap=50, Alpha=0.25, RMP0=0.15, Beta=0.9, TGap=1,
                 save_data=True, save_path='./Data', name='MTDE-ADKT',
                 disable_tqdm=True):
        """
        Initialize MTDE-ADKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        P : float, optional
            Top P fraction for pbest selection (default: 0.1)
        H : int, optional
            SHADE success history memory size (default: 100)
        Gap : int, optional
            RMP adaptation period in generations (default: 50)
        Alpha : float, optional
            Population reduction timing as fraction of budget (default: 0.25)
        RMP0 : float, optional
            Initial random mating probability (default: 0.15)
        Beta : float, optional
            EMA smoothing factor for centroid tracking (default: 0.9)
        TGap : int, optional
            Transfer frequency: transfer every TGap generations (default: 1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTDE-ADKT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.P = P
        self.H = H
        self.Gap = Gap
        self.Alpha = Alpha
        self.RMP0 = RMP0
        self.Beta = Beta
        self.TGap = TGap
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTDE-ADKT algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Convert to unified space
        pop_decs, pop_cons = space_transfer(
            problem=problem, decs=decs, cons=cons, type='uni', padding='mid')
        pop_objs = objs
        maxD = pop_decs[0].shape[1]
        maxC = pop_cons[0].shape[1]

        # SHADE memory
        MF = [0.5 * np.ones(self.H) for _ in range(nt)]
        MCR = [0.5 * np.ones(self.H) for _ in range(nt)]
        Hidx = [0] * nt

        # Archives (unified space decs)
        archives = [np.empty((0, maxD)) for _ in range(nt)]

        # Success rate histories (initialized with 1.0 as in MATLAB)
        r_suc1 = [[1.0] for _ in range(nt)]
        r_suc2 = [[1.0] for _ in range(nt)]
        r_suc3 = [[1.0] for _ in range(nt)]

        # Adaptive RMP
        RMP1 = np.full(nt, self.RMP0)
        RMP2 = np.full(nt, self.RMP0)
        delta_rmp = self.Gap / 500.0

        # EMA centroid
        mDec = 0.5 * np.ones((nt, maxD))
        for t in range(nt):
            mDec[t] = np.mean(pop_decs[t], axis=0)

        reduce_flag = False
        # MToP increments Algo.Gen inside notTerminated *before* the loop body
        # runs, so the first executed generation is Gen = 2. Matching the
        # counter keeps the mod(Gen, TGap) / mod(Gen, Gap) schedules and the
        # r_suc window slices identical to the reference.
        gen = 2

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            for t in range(nt):
                n_t = len(pop_decs[t])

                # --- SHADE: generate adaptive F and CR ---
                F_arr = np.zeros(n_t)
                CR_arr = np.zeros(n_t)
                for i in range(n_t):
                    idx = np.random.randint(self.H)
                    f = MF[t][idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                    while f <= 0:
                        f = MF[t][idx] + 0.1 * np.tan(
                            np.pi * (np.random.rand() - 0.5))
                    F_arr[i] = min(f, 1.0)
                    cr = np.random.normal(MCR[t][idx], 0.1)
                    CR_arr[i] = np.clip(cr, 0, 1)

                # Source task selection
                k = np.random.randint(nt)
                while k == t:
                    k = np.random.randint(nt)

                # Union: population + archive
                if len(archives[t]) > 0:
                    union_decs = np.vstack([pop_decs[t], archives[t]])
                else:
                    union_decs = pop_decs[t]

                # --- Partition individuals into 3 transfer types ---
                par1_idx, par2_idx, par3_idx = [], [], []
                if gen % self.TGap == 0:
                    for i in range(n_t):
                        if np.random.rand() < RMP1[t]:
                            par1_idx.append(i)
                        elif np.random.rand() < RMP2[t]:
                            par2_idx.append(i)
                        else:
                            par3_idx.append(i)
                else:
                    par3_idx = list(range(n_t))

                n1, n2, n3 = len(par1_idx), len(par2_idx), len(par3_idx)

                # pbest indices (sorted by CV then obj)
                cvs = np.sum(np.maximum(0, pop_cons[t]), axis=1) \
                    if maxC > 0 else np.zeros(n_t)
                sort_order = np.lexsort((pop_objs[t][:, 0], cvs))
                n_pbest = max(round(self.P * n_t), 1)
                pop_pbest = sort_order[:n_pbest]

                # Transfer individuals from source
                n_transfer = n1 + n2
                n_source = len(pop_decs[k])
                if n_transfer > 0:
                    t_perm = np.random.permutation(n_source)
                    transfer_idx = t_perm[:min(n_transfer, n_source)]
                    if len(transfer_idx) < n_transfer:
                        extra = np.random.randint(
                            n_source, size=n_transfer - len(transfer_idx))
                        transfer_idx = np.concatenate([transfer_idx, extra])
                else:
                    transfer_idx = np.array([], dtype=int)

                # Offspring storage
                off_decs = np.zeros_like(pop_decs[t])
                off_objs = np.full_like(pop_objs[t], np.inf)
                off_cons = np.zeros_like(pop_cons[t])
                replace_all = np.zeros(n_t, dtype=bool)

                # === Type 1: Distribution-aligned transfer ===
                if n1 > 0:
                    idx1 = np.array(par1_idx)
                    t_idx1 = transfer_idx[:n1]

                    # D_Align: compute transformation matrix once
                    mus = np.mean(pop_decs[k], axis=0)
                    mut = np.mean(pop_decs[t], axis=0)
                    Cs = np.cov((pop_decs[k] - mus), rowvar=False) \
                        + np.eye(maxD)
                    Ct = np.cov((pop_decs[t] - mut), rowvar=False) \
                        + np.eye(maxD)
                    inv_sqrt_Cs = _matrix_inv_sqrt(Cs)
                    sqrt_Ct = _matrix_sqrt(Ct)
                    T_mat = inv_sqrt_Cs @ sqrt_Ct

                    # Transform source individuals
                    source_centered = pop_decs[k][t_idx1] - mus
                    transpop1 = source_centered @ T_mat + mDec[t]

                    off1 = _generation_transfer(
                        pop_decs[t][idx1], transpop1,
                        F_arr[idx1], CR_arr[idx1],
                        pop_decs[t], union_decs, pop_pbest, maxD)

                    o1, c1_real = evaluation_single(
                        problem, off1[:, :dims[t]], t)
                    c1 = np.zeros((n1, maxC))
                    if maxC > 0 and c1_real.shape[1] > 0:
                        c1[:, :c1_real.shape[1]] = c1_real
                    nfes += n1
                    pbar.update(n1)

                    off_decs[idx1] = off1
                    off_objs[idx1] = o1
                    off_cons[idx1] = c1

                    rep1 = _tournament(pop_objs[t][idx1], pop_cons[t][idx1],
                                       o1, c1, maxC)
                    replace_all[idx1] = rep1
                    r_suc1[t].append(np.sum(rep1) / n1)
                else:
                    r_suc1[t].append(0.0)

                # === Type 2: Direct transfer (no D_Align) ===
                if n2 > 0:
                    idx2 = np.array(par2_idx)
                    t_idx2 = transfer_idx[n1:n1 + n2]
                    transpop2 = pop_decs[k][t_idx2].copy()

                    off2 = _generation_transfer(
                        pop_decs[t][idx2], transpop2,
                        F_arr[idx2], CR_arr[idx2],
                        pop_decs[t], union_decs, pop_pbest, maxD)

                    o2, c2_real = evaluation_single(
                        problem, off2[:, :dims[t]], t)
                    c2 = np.zeros((n2, maxC))
                    if maxC > 0 and c2_real.shape[1] > 0:
                        c2[:, :c2_real.shape[1]] = c2_real
                    nfes += n2
                    pbar.update(n2)

                    off_decs[idx2] = off2
                    off_objs[idx2] = o2
                    off_cons[idx2] = c2

                    rep2 = _tournament(pop_objs[t][idx2], pop_cons[t][idx2],
                                       o2, c2, maxC)
                    replace_all[idx2] = rep2
                    r_suc2[t].append(np.sum(rep2) / n2)
                else:
                    r_suc2[t].append(0.0)

                # === Type 3: Standard DE/current-to-pbest/1 ===
                if n3 > 0:
                    idx3 = np.array(par3_idx)

                    off3 = _generation_standard(
                        pop_decs[t][idx3],
                        F_arr[idx3], CR_arr[idx3],
                        pop_decs[t], union_decs, pop_pbest, maxD)

                    o3, c3_real = evaluation_single(
                        problem, off3[:, :dims[t]], t)
                    c3 = np.zeros((n3, maxC))
                    if maxC > 0 and c3_real.shape[1] > 0:
                        c3[:, :c3_real.shape[1]] = c3_real
                    nfes += n3
                    pbar.update(n3)

                    off_decs[idx3] = off3
                    off_objs[idx3] = o3
                    off_cons[idx3] = c3

                    rep3 = _tournament(pop_objs[t][idx3], pop_cons[t][idx3],
                                       o3, c3, maxC)
                    replace_all[idx3] = rep3
                    r_suc3[t].append(np.sum(rep3) / n3)
                else:
                    r_suc3[t].append(0.0)

                # === SHADE update ===
                if np.any(replace_all):
                    SF = F_arr[replace_all]
                    SCR = CR_arr[replace_all]

                    p_cv = np.sum(np.maximum(0, pop_cons[t][replace_all]),
                                  axis=1) if maxC > 0 else np.zeros(
                        np.sum(replace_all))
                    o_cv = np.sum(np.maximum(0, off_cons[replace_all]),
                                  axis=1) if maxC > 0 else np.zeros(
                        np.sum(replace_all))

                    dif = p_cv - o_cv
                    dif_obj = pop_objs[t][replace_all, 0] \
                        - off_objs[replace_all, 0]
                    dif_obj = np.maximum(dif_obj, 0)
                    dif[dif <= 0] = dif_obj[dif <= 0]

                    sum_dif = np.sum(dif)
                    if sum_dif > 0:
                        dif = dif / sum_dif
                        MF[t][Hidx[t]] = np.sum(dif * SF ** 2) / \
                            np.sum(dif * SF)
                        MCR[t][Hidx[t]] = np.sum(dif * SCR)
                    else:
                        MF[t][Hidx[t]] = MF[t][(Hidx[t] - 1) % self.H]
                        MCR[t][Hidx[t]] = MCR[t][(Hidx[t] - 1) % self.H]
                else:
                    MF[t][Hidx[t]] = MF[t][(Hidx[t] - 1) % self.H]
                    MCR[t][Hidx[t]] = MCR[t][(Hidx[t] - 1) % self.H]
                Hidx[t] = (Hidx[t] + 1) % self.H

                # === Archive update ===
                if np.any(replace_all):
                    replaced_decs = pop_decs[t][replace_all]
                    if len(archives[t]) > 0:
                        archives[t] = np.vstack(
                            [archives[t], replaced_decs])
                    else:
                        archives[t] = replaced_decs.copy()
                    # MATLAB caps the archive at the *current* population size
                    # (length(population{t})), which halves after reduction.
                    if len(archives[t]) > n_t:
                        perm = np.random.permutation(len(archives[t]))[:n_t]
                        archives[t] = archives[t][perm]

                # === Selection: replace winners ===
                pop_decs[t][replace_all] = off_decs[replace_all]
                pop_objs[t][replace_all] = off_objs[replace_all]
                pop_cons[t][replace_all] = off_cons[replace_all]

            # === RMP adaptation ===
            # r_suc*[t] holds the MATLAB array with a leading 1 at index 0, so
            # MATLAB's r_suc{t}((Gen-Gap+1):Gen) is r_suc[t][gen-Gap:gen] here.
            for t in range(nt):
                if gen % self.Gap == 0 and gen >= self.Gap:
                    r1_avg = np.mean(r_suc1[t][gen - self.Gap:gen])
                    r3_avg = np.mean(r_suc3[t][gen - self.Gap:gen])
                    if r1_avg * self.TGap >= r3_avg:
                        RMP1[t] = min(RMP1[t] + delta_rmp, 0.45)
                    else:
                        RMP1[t] = max(RMP1[t] - delta_rmp, 0.01)

                    r2_avg = np.mean(r_suc2[t][gen - self.Gap:gen])
                    if r2_avg * self.TGap >= r3_avg:
                        RMP2[t] = min(RMP2[t] + delta_rmp, 0.45)
                    else:
                        RMP2[t] = max(RMP2[t] - delta_rmp, 0.02)

            # === Population reduction ===
            if not reduce_flag and nfes >= max_nfes * self.Alpha:
                new_n = round(n / 2)
                for t in range(nt):
                    cvs_t = np.sum(np.maximum(0, pop_cons[t]), axis=1) \
                        if maxC > 0 else np.zeros(len(pop_decs[t]))
                    rank_t = np.lexsort((pop_objs[t][:, 0], cvs_t))

                    # Save removed to archive
                    removed = rank_t[new_n:]
                    if len(removed) > 0:
                        if len(archives[t]) > 0:
                            archives[t] = np.vstack(
                                [archives[t], pop_decs[t][removed]])
                        else:
                            archives[t] = pop_decs[t][removed].copy()
                        if len(archives[t]) > n:
                            perm = np.random.permutation(
                                len(archives[t]))[:n]
                            archives[t] = archives[t][perm]

                    # Keep top half
                    keep = rank_t[:new_n]
                    pop_decs[t] = pop_decs[t][keep]
                    pop_objs[t] = pop_objs[t][keep]
                    pop_cons[t] = pop_cons[t][keep]

                reduce_flag = True

            # === EMA centroid update ===
            for t in range(nt):
                mDec[t] = (1 - self.Beta) * mDec[t] + \
                    self.Beta * np.mean(pop_decs[t], axis=0)

            # Record history
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs,
                           all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results


# ============================================================
# Helper functions
# ============================================================

def _matrix_sqrt(A):
    """Compute matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _matrix_inv_sqrt(A):
    """Compute inverse matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _tournament(parent_objs, parent_cons, off_objs, off_cons, maxC):
    """
    One-to-one selection matching MTO-Platform ``Selection_Tournament``
    (epsilon = 0): the parent is replaced only if

    - both are infeasible and the offspring has a strictly lower CV, or
    - both are feasible and the offspring has a strictly lower objective.

    A feasible offspring therefore never displaces an infeasible parent, and
    two infeasible individuals with equal CV keep the parent.
    """
    n_ind = len(parent_objs)
    p_cv = np.sum(np.maximum(0, parent_cons), axis=1) \
        if maxC > 0 else np.zeros(n_ind)
    o_cv = np.sum(np.maximum(0, off_cons), axis=1) \
        if maxC > 0 else np.zeros(n_ind)

    replace_cv = (p_cv > o_cv) & (p_cv > 0) & (o_cv > 0)
    equal_cv = (p_cv <= 0) & (o_cv <= 0)
    replace_f = parent_objs[:, 0] > off_objs[:, 0]
    return (equal_cv & replace_f) | replace_cv


def _generation_transfer(parent_decs, transpop_decs, F, CR,
                         pop_decs, union_decs, pbest_idx, D):
    """
    DE/current-to-pbest/1 with transfer base vector.

    Mutation: v = transpop_i + F * (pbest - transpop_i) + F * (x1 - x2)
    Crossover: binomial with transpop_i
    Boundary: midpoint repair toward transpop_i
    """
    n_ind = len(parent_decs)
    n_pop = len(pop_decs)
    n_union = len(union_decs)
    off = np.zeros((n_ind, D))

    for i in range(n_ind):
        pb = pbest_idx[np.random.randint(len(pbest_idx))]
        x1 = np.random.randint(n_pop)
        while x1 == pb:
            x1 = np.random.randint(n_pop)
        x2 = np.random.randint(n_union)
        while x2 == x1 or x2 == pb:
            x2 = np.random.randint(n_union)

        v = transpop_decs[i] + \
            F[i] * (pop_decs[pb] - transpop_decs[i]) + \
            F[i] * (pop_decs[x1] - union_decs[x2])

        # Binomial crossover with transpop (not parent)
        u = transpop_decs[i].copy()
        j_rand = np.random.randint(D)
        mask = np.random.rand(D) < CR[i]
        mask[j_rand] = True
        u[mask] = v[mask]

        # Midpoint boundary repair toward transpop
        vio_low = u < 0
        u[vio_low] = transpop_decs[i][vio_low] / 2
        vio_up = u > 1
        u[vio_up] = (transpop_decs[i][vio_up] + 1) / 2

        off[i] = u
    return off


def _generation_standard(parent_decs, F, CR, pop_decs, union_decs,
                         pbest_idx, D):
    """
    Standard DE/current-to-pbest/1 (no transfer).

    Mutation: v = parent_i + F * (pbest - parent_i) + F * (x1 - x2)
    Crossover: binomial with parent_i
    Boundary: midpoint repair toward parent_i
    """
    n_ind = len(parent_decs)
    n_pop = len(pop_decs)
    n_union = len(union_decs)
    off = np.zeros((n_ind, D))

    for i in range(n_ind):
        pb = pbest_idx[np.random.randint(len(pbest_idx))]
        x1 = np.random.randint(n_pop)
        while x1 == pb:
            x1 = np.random.randint(n_pop)
        x2 = np.random.randint(n_union)
        while x2 == x1 or x2 == pb:
            x2 = np.random.randint(n_union)

        v = parent_decs[i] + \
            F[i] * (pop_decs[pb] - parent_decs[i]) + \
            F[i] * (pop_decs[x1] - union_decs[x2])

        # Binomial crossover with parent
        u = parent_decs[i].copy()
        j_rand = np.random.randint(D)
        mask = np.random.rand(D) < CR[i]
        mask[j_rand] = True
        u[mask] = v[mask]

        # Midpoint boundary repair toward parent
        vio_low = u < 0
        u[vio_low] = parent_decs[i][vio_low] / 2
        vio_up = u > 1
        u[vio_up] = (parent_decs[i][vio_up] + 1) / 2

        off[i] = u
    return off

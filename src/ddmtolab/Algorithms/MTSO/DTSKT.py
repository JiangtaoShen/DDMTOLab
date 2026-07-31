"""
Distribution Direction-Assisted Two-Stage Knowledge Transfer (DTSKT)

This module implements DTSKT for many-task single-objective optimization with
distribution-based two-stage knowledge transfer.

References
----------
    [1] Zhang, Tingyu, et al. "Distribution Direction-Assisted Two-Stage Knowledge
        Transfer for Many-Task Optimization." IEEE Transactions on Systems, Man,
        and Cybernetics: Systems (2025): 1-15.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class DTSKT:
    """
    Distribution Direction-Assisted Two-Stage Knowledge Transfer for Many-Task Optimization.

    Uses Gaussian EDA with two-stage knowledge transfer:
    - Exploring stage: shifts sampling mean along the best source task's search path
    - Exploiting stage: uses combined source-target distribution for refined search

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'equal',
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

    def __init__(self, problem, n=None, max_nfes=None, A=0.35, beta=0.6, rmp=0.5, topn=2,
                 save_data=True, save_path='./Data', name='DTSKT', disable_tqdm=True):
        """
        Initialize DTSKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance (all tasks must have equal dimensions)
        n : int, optional
            Population size per task (default: 200)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        A : float, optional
            Elite ratio for weighted mean computation (default: 0.35)
        beta : float, optional
            Stage transition ratio - fraction of budget for exploring stage (default: 0.6)
        rmp : float, optional
            Probability of knowledge transfer in offspring generation (default: 0.5)
        topn : int, optional
            Number of top source tasks for multi-source transfer (default: 2)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'DTSKT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 200
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.A = A
        self.beta = beta
        self.rmp = rmp
        self.topn = topn
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the DTSKT algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d = dims[0]
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        total_max_nfes = sum(max_nfes_per_task)
        elit_n = max(1, int(np.floor(self.A * n + 0.5)))

        # Initialize population and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = [n] * nt
        total_nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Log-weights for weighted mean computation
        log_weights = np.log(elit_n + 1) - np.log(np.arange(1, elit_n + 1))

        # Initialize distribution parameters for each task
        M = [None] * nt        # weighted mean of elite
        M_old = [None] * nt    # population mean
        S = [None] * nt        # element-wise std from elite
        path = [None] * nt     # search direction
        rank = [None] * nt     # sort order

        for t in range(nt):
            rank[t] = _constrained_rank(objs[t], cons[t])
            elite_decs = decs[t][rank[t][:elit_n]]
            M[t] = np.average(elite_decs, axis=0, weights=log_weights)
            M_old[t] = np.mean(decs[t], axis=0)
            QQ = elite_decs - M[t]
            S[t] = np.sqrt(np.mean(QQ ** 2, axis=0))
            path[t] = decs[t][rank[t][0]] - M_old[t]

        pbar = tqdm(total=total_max_nfes, initial=total_nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while total_nfes < total_max_nfes:
            # Compute cosine similarity matrix CO
            CO = np.zeros((nt, nt))
            for t in range(nt - 1):
                for k in range(t + 1, nt):
                    norm_t = np.linalg.norm(path[t])
                    norm_k = np.linalg.norm(path[k])
                    if norm_t > 0 and norm_k > 0:
                        cos_val = np.dot(path[t], path[k]) / (norm_t * norm_k)
                    else:
                        cos_val = 0.0
                    CO[t, k] = np.exp(cos_val)
                    CO[k, t] = CO[t, k]
            # Diagonal stays 0 (matches MATLAB's 1/inf initialization)

            # MToP checks termination only at the top of the generation loop, so
            # every task is always processed to completion; the overshoot is
            # removed from the recorded history by trim_excess_evaluations.
            for t in range(nt):
                # --- Evaluate M{t} ---
                # MATLAB evaluates the distribution mean unclipped (M can leave
                # [0, 1] through the 4-point correction below).
                M_dec = M[t].reshape(1, -1)
                OO_obj, OO_con = evaluation_single(problem, M_dec, t)
                nfes_per_task[t] += 1
                total_nfes += 1
                pbar.update(1)

                flag = 0
                idx_set = set()
                mDec_t = np.mean(decs[t], axis=0)

                # Find most similar task
                co_row = CO[t, :]
                k = np.argmax(co_row)
                max_co = co_row[k]
                co_sum = np.sum(co_row)
                msr = max_co / co_sum if co_sum > 0 else 0.0

                # --- Determine transfer strategy ---
                if msr > 1.5 / nt:
                    # Single dominant source task
                    PATH = path[k].copy()
                    elite_k = decs[k][rank[k][:elit_n]]
                    shifted_k = elite_k - M_old[k] + mDec_t
                    elite_t = decs[t][rank[t][:elit_n]]
                    POPM = np.vstack([shifted_k, elite_t]) - M[t]
                    S1 = np.sqrt(np.mean(POPM ** 2, axis=0))
                else:
                    # Multi-source transfer
                    PATH = np.zeros(d)
                    CO2 = np.argsort(-CO[k, :], kind='stable')
                    TOP_idx = [int(i) for i in CO2[:self.topn]]
                    if t in TOP_idx:
                        TOP_idx.remove(t)
                    # MATLAB appends k unconditionally, so k is counted twice
                    # (and therefore double-weighted) when it is already a
                    # top-TOPN neighbour of itself.
                    TOP_idx.append(int(k))

                    # Compute weights from CO(t, :)
                    w = np.array([CO[t, ti] for ti in TOP_idx])
                    w_sum = np.sum(w)
                    w = w / w_sum if w_sum > 0 else np.ones(len(TOP_idx)) / len(TOP_idx)

                    # Weighted path
                    for i, ti in enumerate(TOP_idx):
                        PATH += w[i] * path[ti]

                    # Weighted combined std
                    DDD = np.zeros(d)
                    for i, kk in enumerate(TOP_idx):
                        elite_kk = decs[kk][rank[kk][:elit_n]]
                        shifted_kk = elite_kk - M_old[kk] + mDec_t
                        elite_t = decs[t][rank[t][:elit_n]]
                        POPM = np.vstack([shifted_kk, elite_t]) - M[t]
                        WW = np.sqrt(np.mean(POPM ** 2, axis=0))
                        DDD += w[i] * WW
                    S1 = DDD

                # --- Generate offspring ---
                offspring_decs = np.zeros((n, d))
                offspring_evaluated = np.full(n, False)
                offspring_objs = np.full((n, objs[t].shape[1]), np.inf)
                n_cons_t = cons[t].shape[1]
                offspring_cons = np.zeros((n, n_cons_t))

                for i in range(n - 2):
                    if flag == 0:
                        # Generate from distribution
                        if np.random.rand() < self.rmp:
                            if total_nfes < self.beta * total_max_nfes:
                                # Exploring stage
                                offspring_decs[i] = np.random.normal(
                                    M[t] + np.random.rand() * PATH, S[t])
                            else:
                                # Exploiting stage
                                offspring_decs[i] = np.random.normal(M[t], S1)
                        else:
                            offspring_decs[i] = np.random.normal(M[t], S[t])

                        offspring_decs[i] = np.clip(offspring_decs[i], 0, 1)

                        # Evaluate
                        off_obj, off_con = evaluation_single(
                            problem, offspring_decs[i:i + 1], t)
                        offspring_objs[i] = off_obj[0]
                        offspring_cons[i] = off_con[0]
                        offspring_evaluated[i] = True
                        nfes_per_task[t] += 1
                        total_nfes += 1
                        pbar.update(1)

                        # Check if worse than OO
                        if offspring_objs[i, 0] > OO_obj[0, 0]:
                            flag = 1
                        idx_set.add(i)
                    else:
                        # Mirror strategy
                        offspring_decs[i] = 2 * M[t] - offspring_decs[i - 1]
                        flag = 0

                    offspring_decs[i] = np.clip(offspring_decs[i], 0, 1)

                # Add best individual and OO as last two
                offspring_decs[n - 2] = decs[t][rank[t][0]]  # best from current pop
                offspring_decs[n - 1] = M_dec[0]              # OO point

                # Evaluate non-evaluated offspring (mirrors + best + OO).
                # MATLAB re-evaluates the elite copy and OO here as well.
                non_eval_indices = [i for i in range(n) if not offspring_evaluated[i]]
                if non_eval_indices:
                    ne_decs = offspring_decs[non_eval_indices]
                    ne_objs, ne_cons = evaluation_single(problem, ne_decs, t)
                    for j, ni in enumerate(non_eval_indices):
                        offspring_objs[ni] = ne_objs[j]
                        offspring_cons[ni] = ne_cons[j]
                        offspring_evaluated[ni] = True
                    nfes_per_task[t] += len(non_eval_indices)
                    total_nfes += len(non_eval_indices)
                    pbar.update(len(non_eval_indices))

                # Assemble new population (idx first, then rest - order doesn't matter after sort)
                decs[t] = offspring_decs.copy()
                objs[t] = offspring_objs.copy()
                cons[t] = offspring_cons.copy()

                # --- Selection (sort) ---
                rank[t] = _constrained_rank(objs[t], cons[t])

                # --- Update M with 4-point correction ---
                M_O = M[t].copy()
                elite_decs = decs[t][rank[t][:elit_n]]
                M[t] = np.average(elite_decs, axis=0, weights=log_weights)
                D_M = M[t] - M_O

                # MATLAB evaluates the four probe points unclipped
                O_decs = np.array([
                    M[t] + 2 * D_M,   # O1: overshoot
                    M[t],             # O2: current
                    M_O,              # O3: old
                    M[t] - 0.5 * D_M  # O4: between
                ])
                O_objs, _ = evaluation_single(problem, O_decs, t)
                nfes_per_task[t] += 4
                total_nfes += 4
                pbar.update(4)

                # Adjust M based on objective trend
                if O_objs[0, 0] < O_objs[1, 0] and O_objs[1, 0] < O_objs[2, 0]:
                    # Improving trend: use overshoot
                    M[t] = O_decs[0]
                elif O_objs[1, 0] > max(O_objs[2, 0], O_objs[3, 0]):
                    # M went wrong: fallback
                    M[t] = O_decs[3]
                # else: keep M[t] as is

                # --- Update S, path, M_old ---
                elite_decs = decs[t][rank[t][:elit_n]]
                QQ = elite_decs - M[t]
                S[t] = np.sqrt(np.mean(QQ ** 2, axis=0))

                # Path from random top-3 individual to old mean
                top3_idx = np.random.randint(0, min(3, n))
                path[t] = decs[t][rank[t][top3_idx]] - M_old[t]
                M_old[t] = np.mean(decs[t], axis=0)

            # Append history for all tasks
            append_history(all_decs, decs, all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Trim excess evaluations
        all_decs, all_objs, nfes_per_task, all_cons = trim_excess_evaluations(
            all_decs, all_objs, nt, max_nfes_per_task, nfes_per_task, all_cons)

        # Save results
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results


def _constrained_rank(objs, cons):
    """
    Sort individuals by constraint violation then objective value (ascending).

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (n, 1)
    cons : np.ndarray
        Constraint values, shape (n, c)

    Returns
    -------
    rank : np.ndarray
        Sort indices, shape (n,)
    """
    n = objs.shape[0]
    if cons is not None and cons.shape[1] > 0:
        cv = np.sum(np.maximum(0, cons), axis=1)
    else:
        cv = np.zeros(n)
    return np.lexsort((objs[:, 0], cv))

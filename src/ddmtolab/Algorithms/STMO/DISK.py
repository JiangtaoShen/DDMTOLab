"""
Distribution-based Kriging-assisted evolutionary algorithm (DISK)

This module implements DISK for computationally expensive multi/many-objective
optimization. It uses Distribution-based Probabilistic Dominance (DIPD) that combines
prediction uncertainty from Kriging with the Gaussian density fitted on the currently
non-dominated decision vectors. It features an adaptive local search, triggered when
the newly evaluated candidates fail to dominate the archive front, that is guided by
the reference vector pointing at the widest gap of the current front.

References
----------
    [1] Z. Zhang, Y. Wang, G. Sun, and T. Pang. A distribution information based Kriging-assisted evolutionary algorithm for expensive many-objective optimization problems. IEEE Transactions on Evolutionary Computation, 2024.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, multivariate_normal
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
import warnings

warnings.filterwarnings("ignore")


class DISK:
    """
    Distribution-Informed Surrogate-assisted Kriging for expensive
    multi-objective optimization with adaptive local search.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None,
                 wmax=60, alpha=5,
                 save_data=True, save_path='./Data', name='DISK', disable_tqdm=True):
        """
        Initialize DISK algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1). This is
            MATLAB's ``NI = Problem.N``: it is simultaneously the size of the
            initial Latin hypercube design, the size of the archive A1 kept by
            ``EnvironmentalSelection`` and the size of the surrogate population.
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        wmax : int, optional
            Surrogate evolution generations (default: 60)
        alpha : int, optional
            Number of candidates per iteration (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'DISK')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.wmax = wmax
        self.alpha = alpha
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the DISK algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        # MATLAB: NI = Problem.N is both the DoE size and the archive size
        n_per_task = list(n_initial_per_task)

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Initialize working populations
        pop_indices = []
        for i in range(nt):
            N = n_per_task[i]
            if decs[i].shape[0] <= N:
                pop_indices.append(np.arange(decs[i].shape[0]))
            else:
                idx = _env_selection_real(objs[i], N)
                pop_indices.append(np.where(idx)[0])

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Safety guard: MATLAB loops forever if NewSelect never returns a
        # candidate (no expensive evaluation is consumed in that case).
        stagnation = 0

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break
            nfes_before_sweep = sum(nfes_per_task)

            for i in active_tasks:
                M = n_objs[i]
                D = dims[i]
                N = n_per_task[i]

                # ===== Learn distribution from Pareto front =====
                front_no, _ = nd_sort(objs[i], objs[i].shape[0])
                pf_decs = decs[i][front_no == 1]
                if pf_decs.shape[0] <= 1:
                    pf_decs2 = decs[i][front_no == 2] if np.any(front_no == 2) else np.empty((0, D))
                    pf_decs = np.vstack([pf_decs, pf_decs2]) if pf_decs2.shape[0] > 0 else pf_decs

                if pf_decs.shape[0] <= 1:
                    dist_mu = np.mean(decs[i], axis=0)
                    dist_K = np.cov(decs[i].T) + 1e-6 * np.eye(D)
                else:
                    dist_mu = np.mean(pf_decs, axis=0)
                    dist_K = np.cov(pf_decs.T)
                    if dist_K.ndim == 0:
                        dist_K = np.array([[dist_K]])
                    dist_K += 1e-6 * np.eye(D)

                # ===== Build GP models =====
                try:
                    obj_models = []
                    for j in range(M):
                        model = gp_build(decs[i], objs[i][:, j:j + 1], data_type)
                        obj_models.append(model)
                except Exception:
                    continue

                # ===== Evolutionary search on surrogates =====
                A1_decs = decs[i][pop_indices[i]]
                OP_decs, OP_objs, OP_mse = _surrogate_evolution(
                    A1_decs, obj_models, M, N, self.wmax, data_type, dist_mu, dist_K
                )

                # ===== Candidate selection =====
                n_budget = min(self.alpha, max_nfes_per_task[i] - nfes_per_task[i])
                if n_budget <= 0:
                    continue

                cand_decs, cand_objs = _new_select(
                    OP_decs, OP_objs, OP_mse,
                    decs[i], objs[i],
                    n_budget, problem, i, dist_mu, dist_K
                )

                if cand_decs is None or cand_decs.shape[0] == 0:
                    continue

                # ===== Check improvement for local search =====
                flag = _judge_ls(cand_objs, objs[i])

                # Update database
                decs[i] = np.vstack([decs[i], cand_decs])
                objs[i] = np.vstack([objs[i], cand_objs])
                nfes_per_task[i] += cand_decs.shape[0]
                pbar.update(cand_decs.shape[0])

                # ===== Adaptive local search =====
                if flag == 1 and nfes_per_task[i] < max_nfes_per_task[i]:
                    # Re-train models
                    try:
                        obj_models = []
                        for j in range(M):
                            model = gp_build(decs[i], objs[i][:, j:j + 1], data_type)
                            obj_models.append(model)
                    except Exception:
                        pass
                    else:
                        W, ideal = _identify_w(objs[i], N, M)
                        ls_dec, ls_obj = _local_search(
                            OP_decs, OP_objs, OP_mse, W, ideal,
                            obj_models, M, N, self.wmax, data_type,
                            decs[i], problem, i
                        )
                        if ls_dec is not None:
                            decs[i] = np.vstack([decs[i], ls_dec])
                            objs[i] = np.vstack([objs[i], ls_obj])
                            nfes_per_task[i] += ls_dec.shape[0]
                            pbar.update(ls_dec.shape[0])

                # ===== Update working population =====
                idx = _env_selection_real(objs[i], N)
                pop_indices[i] = np.where(idx)[0]

            if sum(nfes_per_task) == nfes_before_sweep:
                stagnation += 1
                if stagnation >= 50:
                    break
            else:
                stagnation = 0

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.alpha)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# Surrogate-based Evolutionary Search
# =============================================================================

def _surrogate_evolution(pop_decs, obj_models, M, N, wmax, data_type, dist_mu, dist_K):
    """
    Evolutionary search guided by GP surrogates with DIPD selection.

    Parameters
    ----------
    pop_decs : np.ndarray
        Initial population decisions
    obj_models : list
        GP models for each objective
    M : int
        Number of objectives
    N : int
        Population size
    wmax : int
        Number of generations
    data_type : torch.dtype
        Data type for GP
    dist_mu, dist_K : np.ndarray
        Distribution parameters for DIPD

    Returns
    -------
    pop_decs, pop_objs, pop_mse : np.ndarray
        Evolved population
    """
    pop_objs, pop_mse = _gp_predict_all(pop_decs, obj_models, M, data_type)

    for w in range(wmax):
        off_decs = ga_generation(pop_decs, muc=20, mum=20)
        pop_decs = np.vstack([pop_decs, off_decs])
        pop_objs, pop_mse = _gp_predict_all(pop_decs, obj_models, M, data_type)

        # DIPD-based environmental selection
        selected = _s_env_selection(pop_decs, pop_objs, pop_mse, N, dist_mu, dist_K)
        pop_decs = pop_decs[selected]
        pop_objs = pop_objs[selected]
        pop_mse = pop_mse[selected]

    return pop_decs, pop_objs, pop_mse


def _gp_predict_all(pop_decs, obj_models, M, data_type):
    """Predict objectives and MSE for all solutions."""
    N = pop_decs.shape[0]
    pop_objs = np.zeros((N, M))
    pop_mse = np.zeros((N, M))

    for j in range(M):
        pred, std = gp_predict(obj_models[j], pop_decs, data_type)
        pop_objs[:, j] = pred.flatten()
        pop_mse[:, j] = np.abs(std.flatten()) ** 2

    pop_objs = np.real(pop_objs)
    pop_mse = np.abs(np.real(pop_mse))
    return pop_objs, pop_mse


# =============================================================================
# Distribution-Informed Probabilistic Dominance Sorting
# =============================================================================

def _ndsort_dipd(pop_decs, pop_objs, obj_mse, n_sort, dist_mu, dist_K):
    """
    Non-dominated sorting with Distribution-Informed Probabilistic Dominance.

    Combines prediction uncertainty with distribution likelihood from Pareto front.

    Parameters
    ----------
    pop_decs : np.ndarray
        Decision variables, shape (N, D)
    pop_objs : np.ndarray
        Predicted objectives, shape (N, M)
    obj_mse : np.ndarray
        Prediction MSE, shape (N, M)
    n_sort : int
        Number of solutions to sort
    dist_mu : np.ndarray
        Distribution mean, shape (D,)
    dist_K : np.ndarray
        Distribution covariance, shape (D, D)

    Returns
    -------
    front_no : np.ndarray
        Front assignments
    max_fno : int
        Last front number
    """
    N, M = pop_objs.shape

    # Compute probability density under learned distribution
    try:
        rv = multivariate_normal(mean=dist_mu, cov=dist_K, allow_singular=True)
        Pro = rv.pdf(pop_decs)
    except Exception:
        Pro = np.ones(N)

    Pro = np.maximum(Pro, 1e-300)

    # Pairwise probabilistic dominance
    # mean_diff[i,j,k] = obj_i_k - obj_j_k
    mean_diff = pop_objs[:, np.newaxis, :] - pop_objs[np.newaxis, :, :]
    sigma_sq = obj_mse[:, np.newaxis, :] + obj_mse[np.newaxis, :, :]
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-20))

    # P(obj_i <= obj_j) for each objective
    z = -mean_diff / sigma
    x_PD = norm.cdf(z)       # P(i is better than j)
    y_PD = 1.0 - x_PD        # P(j is better than i)

    # Weight by distribution probability — MATLAB:
    # x_PD = -x_PD .* Pro[i], y_PD = -y_PD .* Pro[j]
    # Dominance: all(-x*Pro[i] <= -y*Pro[j]) ↔ all(x*Pro[i] >= y*Pro[j])
    x_PD = x_PD * Pro[:, np.newaxis, np.newaxis]    # weighted by Pro[i]
    y_PD = y_PD * Pro[np.newaxis, :, np.newaxis]    # weighted by Pro[j]

    # Dominance: i dominates j if all(x_PD[i,j,:] >= y_PD[i,j,:]) and not all equal
    dominates = np.all(x_PD >= y_PD, axis=2) & ~np.all(
        np.abs(x_PD - y_PD) < 1e-10, axis=2)
    np.fill_diagonal(dominates, False)

    # Non-dominated sorting
    front_no = np.full(N, np.inf)
    max_fno = 0
    remaining = np.ones(N, dtype=bool)

    n_assigned = 0
    while n_assigned < min(n_sort, N):
        max_fno += 1
        current = np.where(remaining)[0]
        if len(current) == 0:
            break

        sub_dom = dominates[np.ix_(current, current)]
        dom_count = np.sum(sub_dom, axis=0)
        min_count = np.min(dom_count)
        frontal = current[dom_count == min_count]

        front_no[frontal] = max_fno
        remaining[frontal] = False
        n_assigned += len(frontal)

    return front_no, max_fno


# =============================================================================
# Environmental Selection
# =============================================================================

def _s_env_selection(pop_decs, pop_objs, pop_mse, N, dist_mu, dist_K):
    """Surrogate environmental selection using DIPD + angle-based diversity."""
    n_total = pop_objs.shape[0]
    front_no, max_fno = _ndsort_dipd(pop_decs, pop_objs, pop_mse, N, dist_mu, dist_K)

    # Normalize objectives
    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    obj_range = np.maximum(zmax - zmin, 1e-9)
    norm_objs = (pop_objs - zmin) / obj_range

    selected = np.where(front_no < max_fno)[0]
    last_front = np.where(front_no == max_fno)[0]
    n_remaining = N - len(selected)

    if n_remaining <= 0:
        return selected[:N]

    if len(last_front) <= n_remaining:
        selected = np.concatenate([selected, last_front])
    elif max_fno == 1:
        chosen = _truncation_angle(norm_objs[last_front], n_remaining)
        selected = np.concatenate([selected, last_front[chosen]])
    else:
        chosen = _dist_selection_angle(norm_objs[selected], norm_objs[last_front], n_remaining)
        selected = np.concatenate([selected, last_front[chosen]])

    return selected


def _env_selection_real(pop_objs, N):
    """Real evaluation environmental selection (standard ND + angle diversity)."""
    n_total = pop_objs.shape[0]
    if n_total <= N:
        return np.ones(n_total, dtype=bool)

    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    obj_range = np.maximum(zmax - zmin, 1e-9)
    norm_objs = (pop_objs - zmin) / obj_range

    front_no, max_fno = nd_sort(pop_objs, N)

    next_mask = np.zeros(n_total, dtype=bool)
    next_mask[front_no < max_fno] = True
    last_front = np.where(front_no == max_fno)[0]
    n_remaining = N - np.sum(next_mask)

    if n_remaining <= 0:
        return next_mask

    if len(last_front) <= n_remaining:
        next_mask[last_front] = True
    elif max_fno == 1:
        chosen = _truncation_angle(norm_objs[last_front], n_remaining)
        next_mask[last_front[chosen]] = True
    else:
        sel_objs = norm_objs[next_mask]
        chosen = _dist_selection_angle(sel_objs, norm_objs[last_front], n_remaining)
        next_mask[last_front[chosen]] = True

    return next_mask


def _angle_distance(A, B):
    """Compute pairwise angle-based distance: arccos(1 - cosine_distance)."""
    # Cosine similarity
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    norm_A = np.maximum(norm_A, 1e-10)
    norm_B = np.maximum(norm_B, 1e-10)
    cos_sim = (A @ B.T) / (norm_A @ norm_B.T)
    cos_sim = np.clip(cos_sim, -1, 1)
    return np.arccos(cos_sim)


def _truncation_angle(pop_objs, K):
    """Select K solutions by removing most crowded (angle-based distance)."""
    N = pop_objs.shape[0]
    if N <= K:
        return np.arange(N)

    dist = _angle_distance(pop_objs, pop_objs)
    np.fill_diagonal(dist, np.inf)

    deleted = np.zeros(N, dtype=bool)
    while np.sum(~deleted) > K:
        remaining = np.where(~deleted)[0]
        sub_dist = dist[np.ix_(remaining, remaining)]
        sorted_d = np.sort(sub_dist, axis=1)
        # MATLAB: [~,Rank] = sortrows(Temp) -> column 1 (the nearest-neighbour
        # distance) is the primary key, np.lexsort takes the LAST key as
        # primary, hence the column reversal.
        order = np.lexsort(sorted_d[:, ::-1].T)
        deleted[remaining[order[0]]] = True

    return np.where(~deleted)[0]


def _dist_selection_angle(selected_objs, candidate_objs, n_select):
    """Select n_select candidates maximizing min angle-distance to selected."""
    N1 = selected_objs.shape[0]
    N2 = candidate_objs.shape[0]
    if N2 <= n_select:
        return np.arange(N2)

    dist = _angle_distance(candidate_objs, selected_objs)
    min_dist = np.min(dist, axis=1)
    remaining = np.ones(N2, dtype=bool)
    chosen = []

    for _ in range(n_select):
        avail = np.where(remaining)[0]
        if len(avail) == 0:
            break
        pos = int(avail[int(np.argmax(min_dist[avail]))])
        chosen.append(pos)
        remaining[pos] = False
        # the newly selected candidate joins the reference set
        new_d = _angle_distance(candidate_objs, candidate_objs[pos:pos + 1]).ravel()
        min_dist = np.minimum(min_dist, new_d)

    return np.array(chosen, dtype=int)


# =============================================================================
# Candidate Selection (NewSelect)
# =============================================================================

def _new_select(OP_decs, OP_objs, OP_mse, db_decs, db_objs,
                alpha, problem, task_idx, dist_mu, dist_K):
    """
    Select alpha candidates from surrogate-evolved population.

    Uses DIPD filtering then angle-based diversity selection.
    Evaluates candidates iteratively, updating Pareto front.

    Returns
    -------
    cand_decs, cand_objs : np.ndarray or None
        Selected and evaluated candidates
    """
    # Find novel solutions
    if db_decs.shape[0] > 0:
        dist_to_db = cdist(OP_decs, db_decs)
        min_dist = np.min(dist_to_db, axis=1)
        novel_mask = min_dist > 1e-50
    else:
        novel_mask = np.ones(OP_decs.shape[0], dtype=bool)

    novel_idx = np.where(novel_mask)[0]
    if len(novel_idx) == 0:
        return None, None

    if len(novel_idx) <= alpha:
        pop_new = OP_decs[novel_idx]
        new_objs, _ = evaluation_single(problem, pop_new, task_idx)
        return pop_new, new_objs

    # Extract and normalize
    pop_decs = OP_decs[novel_idx]
    pop_objs = OP_objs[novel_idx]
    pop_mse = OP_mse[novel_idx]

    # Get database Pareto front
    n_db = db_objs.shape[0]
    front_no_db, _ = nd_sort(db_objs, n_db)
    pf_objs = db_objs[front_no_db == 1]
    pf_objs = np.unique(pf_objs, axis=0) if pf_objs.shape[0] > 1 else pf_objs

    # Normalize
    all_objs_combined = np.vstack([pf_objs, pop_objs])
    zmin = np.min(all_objs_combined, axis=0)
    zmax = np.max(all_objs_combined, axis=0)
    obj_range = np.maximum(zmax - zmin, 1e-9)

    norm_pop_objs = (pop_objs - zmin) / obj_range
    norm_pop_mse = pop_mse / (obj_range ** 2)
    norm_pf_objs = (pf_objs - zmin) / obj_range

    # DIPD sorting - first front only (MATLAB: nSort=1)
    front_no, _ = _ndsort_dipd(pop_decs, norm_pop_objs, norm_pop_mse, 1, dist_mu, dist_K)
    first_front = np.where(front_no == 1)[0]

    if len(first_front) == 0:
        first_front = np.arange(min(alpha, pop_decs.shape[0]))

    if len(first_front) <= alpha:
        sel_decs = pop_decs[first_front]
        sel_objs, _ = evaluation_single(problem, sel_decs, task_idx)
        return sel_decs, sel_objs

    # Diversity-based iterative selection
    cand_decs_list = []
    cand_objs_list = []
    selectable = np.ones(len(first_front), dtype=bool)
    current_pf = norm_pf_objs.copy()

    for _ in range(alpha):
        avail = np.where(selectable)[0]
        if len(avail) == 0:
            break

        avail_objs = norm_pop_objs[first_front[avail]]
        # Angle distance to current Pareto front
        if current_pf.shape[0] > 0:
            angle_dist = _angle_distance(avail_objs, current_pf)
            min_angles = np.min(angle_dist, axis=1)
        else:
            min_angles = np.ones(len(avail))

        best = np.argmax(min_angles)
        best_global = first_front[avail[best]]

        # Evaluate
        new_dec = pop_decs[best_global:best_global + 1]
        new_obj, _ = evaluation_single(problem, new_dec, task_idx)
        cand_decs_list.append(new_dec[0])
        cand_objs_list.append(new_obj[0])

        # Update Pareto front with new solution
        all_real_objs = np.vstack([db_objs] + [np.array(cand_objs_list)])
        fno, _ = nd_sort(all_real_objs, all_real_objs.shape[0])
        new_pf = all_real_objs[fno == 1]
        new_pf = np.unique(new_pf, axis=0) if new_pf.shape[0] > 1 else new_pf
        current_pf = (new_pf - zmin) / obj_range

        selectable[avail[best]] = False

    if len(cand_decs_list) == 0:
        return None, None

    return np.array(cand_decs_list), np.array(cand_objs_list)


# =============================================================================
# Weight Vector Identification
# =============================================================================

def _identify_w(db_objs, N, M):
    """
    Identify weight vector pointing to gap in Pareto front.

    Parameters
    ----------
    db_objs : np.ndarray
        All evaluated objectives
    N : int
        Population size (for generating reference vectors)
    M : int
        Number of objectives

    Returns
    -------
    W : np.ndarray
        Weight vector, shape (M,)
    ideal : np.ndarray
        Adjusted ideal point, shape (M,)
    """
    # Generate candidate weight vectors (MATLAB: UniformPoint(10*N,M))
    V, _ = uniform_point(10 * N, M)

    # Extract Pareto front
    n_all = db_objs.shape[0]
    front_no, _ = nd_sort(db_objs, n_all)
    pf_objs = db_objs[front_no == 1]

    if pf_objs.shape[0] == 0:
        return np.ones(M) / M, np.zeros(M)

    # Compute adjusted ideal point
    nadir = np.max(pf_objs, axis=0)
    ideal_raw = np.min(pf_objs, axis=0)
    ideal = ideal_raw - (nadir - ideal_raw) / 10 - 0.1

    # Shift objectives
    shifted_pf = pf_objs - ideal
    shifted_pf = np.maximum(shifted_pf, 1e-10)

    # Compute angles between weight vectors and PF solutions
    angle_dist = _angle_distance(V, shifted_pf)

    # Find vector with max min-angle to any PF solution (ties broken at random,
    # matching MATLAB's index(randperm(length(index),1)))
    min_angles = np.min(angle_dist, axis=1)
    tied = np.where(min_angles == np.max(min_angles))[0]
    best = tied[np.random.randint(len(tied))] if len(tied) > 1 else tied[0]

    W = V[best]
    return W, ideal


# =============================================================================
# Local Search
# =============================================================================

def _local_search(OP_decs, OP_objs, OP_mse, W, ideal, obj_models, M, N,
                  wmax, data_type, db_decs, problem, task_idx):
    """
    Focused local search in direction W using weighted Chebyshev + uncertainty.

    fitness = max(|obj - ideal| * W) - 2 * mean(std)

    Returns
    -------
    ls_dec, ls_obj : np.ndarray or None
        Best solution found and its true objective
    """
    pop_decs = OP_decs.copy()
    pop_objs = OP_objs.copy()
    pop_mse = OP_mse.copy()

    for w in range(wmax):
        # 4 offspring operators
        off1 = ga_generation(pop_decs, muc=20, mum=20)
        off2 = _de_current_rand_1(pop_decs)
        off3 = _de_rand_1(pop_decs)
        off4 = _de_current_rand_1(pop_decs)

        # MATLAB: P.decs = unique(P.decs,'rows') - exact row deduplication,
        # the decision values themselves must not be perturbed by rounding
        pop_decs = np.vstack([pop_decs, off1, off2, off3, off4])
        pop_decs = np.unique(pop_decs, axis=0)

        pop_objs, pop_mse = _gp_predict_all(pop_decs, obj_models, M, data_type)
        pop_std = np.sqrt(np.maximum(pop_mse, 0))

        # Directional fitness: weighted Chebyshev - exploration bonus
        fitness = np.max(np.abs(pop_objs - ideal) * W, axis=1) - 2 * np.mean(pop_std, axis=1)

        # Keep top N
        sorted_idx = np.argsort(fitness)
        n_keep = min(N, len(sorted_idx))
        pop_decs = pop_decs[sorted_idx[:n_keep]]
        pop_objs = pop_objs[sorted_idx[:n_keep]]
        pop_mse = pop_mse[sorted_idx[:n_keep]]

    # Select best
    pop_std = np.sqrt(np.maximum(pop_mse, 0))
    fitness = np.max(np.abs(pop_objs - ideal) * W, axis=1) - 2 * np.mean(pop_std, axis=1)
    best_idx = np.argmin(fitness)
    best_dec = pop_decs[best_idx:best_idx + 1]

    # Check for duplicate
    if db_decs.shape[0] > 0:
        min_d = np.min(cdist(best_dec, db_decs))
        if min_d <= 1e-50:
            return None, None

    best_obj, _ = evaluation_single(problem, best_dec, task_idx)
    return best_dec, best_obj


def _judge_ls(cand_objs, db_objs):
    """
    Judge if local search should be triggered.
    Returns 1 if candidates improved the Pareto front.
    """
    front_no_c, _ = nd_sort(cand_objs, cand_objs.shape[0])
    front_no_db, _ = nd_sort(db_objs, db_objs.shape[0])

    pf_c = cand_objs[front_no_c == 1]
    pf_db = db_objs[front_no_db == 1]

    # Check if any candidate dominates any archive PF member
    for c_sol in pf_c:
        for db_sol in pf_db:
            if np.all(c_sol <= db_sol) and not np.all(c_sol == db_sol):
                return 0  # Candidate dominates archive → no improvement needed

    return 1  # No domination → trigger local search


# =============================================================================
# DE Operators
# =============================================================================

def _de_current_rand_1(pop_decs):
    """DE/current-to-rand/1 with polynomial mutation."""
    N, D = pop_decs.shape
    Fm_set = np.array([0.6, 0.8, 1.0])
    CRm_set = np.array([0.1, 0.2, 1.0])

    F = Fm_set[np.random.randint(0, 3, size=N)].reshape(-1, 1)
    CR = CRm_set[np.random.randint(0, 3, size=N)].reshape(-1, 1)

    F_mat = np.tile(F, (1, D))
    site = np.random.rand(N, D) < CR
    p1 = pop_decs[np.random.permutation(N)]
    p2 = pop_decs[np.random.permutation(N)]
    p3 = pop_decs[np.random.permutation(N)]

    # DE/current-to-rand/1: x + F*(r1-x) + F*(r2-r3)
    mutant = pop_decs + F_mat * (p1 - pop_decs) + F_mat * (p2 - p3)
    offspring = pop_decs.copy()
    offspring[site] = mutant[site]

    offspring = np.clip(offspring, 0, 1)
    offspring = _poly_mutation(offspring, mum=20)
    return offspring


def _de_rand_1(pop_decs):
    """DE/rand/1 with polynomial mutation."""
    N, D = pop_decs.shape
    Fm_set = np.array([0.6, 0.8, 1.0])
    CRm_set = np.array([0.1, 0.2, 1.0])

    F = Fm_set[np.random.randint(0, 3, size=N)].reshape(-1, 1)
    CR = CRm_set[np.random.randint(0, 3, size=N)].reshape(-1, 1)

    F_mat = np.tile(F, (1, D))
    site = np.random.rand(N, D) < CR
    p1 = pop_decs[np.random.permutation(N)]
    p2 = pop_decs[np.random.permutation(N)]
    p3 = pop_decs[np.random.permutation(N)]

    # DE/rand/1: r1 + F*(r2-r3)
    mutant = p1 + F_mat * (p2 - p3)
    offspring = pop_decs.copy()
    offspring[site] = mutant[site]

    offspring = np.clip(offspring, 0, 1)
    offspring = _poly_mutation(offspring, mum=20)
    return offspring


def _poly_mutation(pop_decs, mum=20):
    """Polynomial mutation with proM=1, bounds [0,1]."""
    N, D = pop_decs.shape
    site = np.random.rand(N, D) < 1.0 / D
    mu = np.random.rand(N, D)

    temp_low = site & (mu <= 0.5)
    temp_high = site & (mu > 0.5)

    offspring = pop_decs.copy()

    if np.any(temp_low):
        x = offspring[temp_low]
        delta = (2 * mu[temp_low] + (1 - 2 * mu[temp_low]) *
                 (1 - x) ** (mum + 1)) ** (1.0 / (mum + 1)) - 1
        offspring[temp_low] = x + delta

    if np.any(temp_high):
        x = offspring[temp_high]
        delta = 1 - (2 * (1 - mu[temp_high]) + 2 * (mu[temp_high] - 0.5) *
                      (1 - (1 - x)) ** (mum + 1)) ** (1.0 / (mum + 1))
        offspring[temp_high] = x + delta

    return np.clip(offspring, 0, 1)

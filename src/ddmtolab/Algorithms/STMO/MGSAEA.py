"""
Multigranularity Surrogate-Assisted Evolutionary Algorithm (MGSAEA)

This module implements MGSAEA for computationally expensive constrained multi-objective
optimization. It uses a two-stage framework:

- Stage 1 (convergence stage): Builds surrogates for objectives only, ignoring constraints
- Stage 2 (constraint stage): Adaptively selects constraint handling strategy based on
  constraint satisfaction status (all violated, partially violated, all satisfied)

Stage transition is triggered when the ideal point change rate drops below a threshold,
indicating convergence of the unconstrained search.

References
----------
    [1] Y. Zhang, H. Jiang, Y. Tian, H. Ma, and X. Zhang. Multigranularity surrogate modeling for evolutionary multiobjective optimization with expensive constraints. IEEE Transactions on Neural Networks and Learning Systems, 2024, 35(3): 2956-2968.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
import warnings

warnings.filterwarnings("ignore")


def _spea2_select(fitness, pop_obj, N, use_real_obj=None):
    """
    SPEA2 selection: prefer fitness < 1, fill/truncate to N.

    Parameters
    ----------
    fitness : np.ndarray
    pop_obj : np.ndarray, used for truncation distance
    N : int, target size
    use_real_obj : np.ndarray, optional, use this for truncation distance instead

    Returns
    -------
    next_mask : np.ndarray, bool
    """
    n_total = len(fitness)
    next_mask = fitness < 1
    if np.sum(next_mask) < N:
        rank = np.argsort(fitness)
        next_mask[:] = False
        next_mask[rank[:min(N, n_total)]] = True
    elif np.sum(next_mask) > N:
        trunc_obj = use_real_obj[next_mask] if use_real_obj is not None else pop_obj[next_mask]
        kept_indices = spea2_truncation(trunc_obj, N)
        temp = np.where(next_mask)[0]
        next_mask[:] = False
        next_mask[temp[kept_indices]] = True
    return next_mask


def _env_selection(pop_dec, pop_obj, NI, M, status=None):
    """
    Environmental selection using SPEA2 fitness and truncation.

    Parameters
    ----------
    pop_dec : np.ndarray, shape (N, D)
    pop_obj : np.ndarray, shape (N, M_ext)
    NI : int, target size
    M : int, number of real objectives
    status : int or None, constraint handling mode (1/2/3 or None for unconstrained)

    Returns
    -------
    pop_dec, pop_obj, fitness
    """
    # Remove duplicates on real objectives
    _, unique_idx = np.unique(pop_obj[:, :M], axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    pop_dec = pop_dec[unique_idx]
    pop_obj = pop_obj[unique_idx]

    N = pop_dec.shape[0]
    if N == 0:
        return pop_dec, pop_obj, np.array([])

    # Calculate fitness based on status
    if status is None:
        # Stage 1: unconstrained
        fitness = spea2_fitness(pop_obj)
        next_mask = _spea2_select(fitness, pop_obj, NI)
    elif status == 1:
        real_obj = pop_obj[:, :M]
        cv = np.maximum(0, pop_obj[:, -1:])
        fitness = spea2_fitness(real_obj, cv)
        next_mask = _spea2_select(fitness, pop_obj, NI, use_real_obj=real_obj)
    elif status == 2:
        real_obj = pop_obj[:, :M]
        pop_con = pop_obj[:, M:]
        cv = np.sum(np.maximum(0, pop_con), axis=1, keepdims=True)
        fitness = spea2_fitness(np.hstack([real_obj, cv]))
        next_mask = _spea2_select(fitness, pop_obj, NI, use_real_obj=real_obj)
    else:  # status == 3
        fitness = spea2_fitness(pop_obj)
        next_mask = _spea2_select(fitness, pop_obj, NI)

    return pop_dec[next_mask], pop_obj[next_mask], fitness[next_mask]


# ============================================================================
# Archive and Population Update
# ============================================================================

def _update_archive(arc_decs, arc_objs, arc_cons, N):
    """Update archive with SPEA2 fitness-based selection and truncation."""
    _, unique_idx = np.unique(arc_objs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    arc_decs = arc_decs[unique_idx]
    arc_objs = arc_objs[unique_idx]
    arc_cons = arc_cons[unique_idx]

    if arc_decs.shape[0] > N:
        fitness = spea2_fitness(arc_objs, arc_cons)
        next_mask = _spea2_select(fitness, arc_objs, N)
        arc_decs = arc_decs[next_mask]
        arc_objs = arc_objs[next_mask]
        arc_cons = arc_cons[next_mask]

    return arc_decs, arc_objs, arc_cons


def _update_population(pop_decs, pop_objs, pop_cons, new_decs, new_objs, new_cons,
                        N, status=None):
    """
    Update population: remove duplicates with new, SPEA2 selection to N, append new.

    Parameters
    ----------
    status : int or None
        None for Stage 1 (unconstrained), 1/2/3 for Stage 2
    """
    # Remove solutions duplicating new
    if new_objs.shape[0] > 0:
        keep = []
        for i in range(pop_objs.shape[0]):
            is_dup = np.any(np.all(np.abs(pop_objs[i] - new_objs) < 1e-10, axis=1))
            if not is_dup:
                keep.append(i)
        if len(keep) > 0:
            keep = np.array(keep)
            pop_decs = pop_decs[keep]
            pop_objs = pop_objs[keep]
            pop_cons = pop_cons[keep]
        else:
            pop_decs = np.empty((0, pop_decs.shape[1]))
            pop_objs = np.empty((0, pop_objs.shape[1]))
            pop_cons = np.empty((0, pop_cons.shape[1]))

    # Deduplicate within population
    if pop_objs.shape[0] > 0:
        _, unique_idx = np.unique(pop_objs, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        pop_decs = pop_decs[unique_idx]
        pop_objs = pop_objs[unique_idx]
        pop_cons = pop_cons[unique_idx]

    # SPEA2 selection to N
    if pop_decs.shape[0] > N:
        if status is None or status == 3:
            # Unconstrained or all constraints satisfied
            fitness = spea2_fitness(pop_objs, pop_cons)
        elif status == 1:
            fitness = spea2_fitness(pop_objs, pop_cons)
        elif status == 2:
            cv = np.sum(np.maximum(0, pop_cons), axis=1, keepdims=True)
            fitness = spea2_fitness(np.hstack([pop_objs, cv]))

        next_mask = _spea2_select(fitness, pop_objs, N)
        pop_decs = pop_decs[next_mask]
        pop_objs = pop_objs[next_mask]
        pop_cons = pop_cons[next_mask]

    # Append new
    if new_decs.shape[0] > 0:
        pop_decs = np.vstack([pop_decs, new_decs])
        pop_objs = np.vstack([pop_objs, new_objs])
        pop_cons = np.vstack([pop_cons, new_cons])

    return pop_decs, pop_objs, pop_cons


# ============================================================================
# Constraint Normalization
# ============================================================================

def _normalize_cv(pop_con):
    """Normalize constraint violations and return aggregated scalar CV."""
    pop_con = np.maximum(0, pop_con)
    cmin = np.min(pop_con, axis=0)
    cmax = np.max(pop_con, axis=0)
    denom = cmax - cmin
    denom[denom == 0] = 1.0
    pop_con_norm = (pop_con - cmin) / denom
    pop_con_norm = np.nan_to_num(pop_con_norm, nan=0.0)
    return np.sum(pop_con_norm, axis=1, keepdims=True)


# ============================================================================
# Ideal Point Change Rate
# ============================================================================

def _calc_max_change(ideal_points, current_iter, gap):
    """
    Calculate the maximum change rate of ideal points over the last `gap` iterations.

    Parameters
    ----------
    ideal_points : list of np.ndarray
        Ideal points at each iteration.
    current_iter : int
        Current iteration index (0-based).
    gap : int
        Window size for change rate computation.

    Returns
    -------
    max_change : float
    """
    delta = 1e-6
    current = ideal_points[current_iter]
    previous = ideal_points[current_iter - gap]
    rz = np.abs((current - previous) / np.maximum(np.abs(previous), delta))
    return np.max(rz)


# ============================================================================
# Main Algorithm
# ============================================================================

class MGSAEA:
    """
    Multigranularity Surrogate-Assisted Evolutionary Algorithm for expensive
    constrained multi-objective optimization.

    Two-stage approach:
    - Stage 1: Objective-only surrogates until ideal points converge
    - Stage 2: Constraint-aware surrogates with adaptive granularity

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100,
                 wmax=20, mu=5, gap=20, lam=1e-3,
                 save_data=True, save_path='./Data', name='MGSAEA', disable_tqdm=True):
        """
        Initialize MGSAEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Archive/population size per task (default: 100)
        wmax : int, optional
            Number of inner GA generations on surrogates (default: 20)
        mu : int, optional
            Number of real evaluated solutions per iteration (default: 5)
        gap : int, optional
            Window for ideal point change rate computation (default: 20)
        lam : float, optional
            Threshold for stage transition (default: 1e-3)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MGSAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.wmax = wmax
        self.mu = mu
        self.gap = gap
        self.lam = lam
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MGSAEA algorithm.

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
        n_cons = problem.n_cons

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # History tracking
        has_cons = any(c.shape[1] > 0 for c in cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Per-task optimization
        for task_i in range(nt):
            m = n_objs[task_i]
            dim = dims[task_i]
            nc = n_cons[task_i]
            NI = n_initial_per_task[task_i]
            N_archive = n_per_task[task_i]

            # Population and archive for this task
            pop_decs = decs[task_i].copy()
            pop_objs = objs[task_i].copy()
            pop_cons = cons[task_i].copy() if nc > 0 else np.zeros((NI, 0))

            # Initialize archive
            arc_decs = pop_decs.copy()
            arc_objs = pop_objs.copy()
            arc_cons = pop_cons.copy() if nc > 0 else np.zeros((NI, 0))
            if nc > 0:
                arc_cons_for_update = arc_cons.copy()
            else:
                arc_cons_for_update = np.zeros((arc_decs.shape[0], 1))
            arc_decs, arc_objs, arc_cons_for_update = _update_archive(
                arc_decs, arc_objs, arc_cons_for_update, N_archive
            )
            if nc > 0:
                arc_cons = arc_cons_for_update
            else:
                arc_cons = np.zeros((arc_decs.shape[0], 0))

            # Stage flag: 0 = Stage 1 (convergence), 1 = Stage 2 (constraint)
            flag = 0
            iteration = 0
            ideal_points = []

            while nfes_per_task[task_i] < max_nfes_per_task[task_i]:
                # Track ideal points
                ideal_points.append(np.min(pop_objs, axis=0))

                # Check stage transition
                if iteration > self.gap and flag == 0:
                    max_change = _calc_max_change(ideal_points, iteration, self.gap)
                    if max_change <= self.lam:
                        flag = 1

                if flag == 0:
                    # ===== Stage 1: Objective-only surrogates =====
                    surr_objs = pop_objs.copy()
                    M_surr = m
                    models = []
                    for j in range(M_surr):
                        gp = gp_build(pop_decs, surr_objs[:, j:j+1], data_type)
                        models.append(gp)

                    fitness = spea2_fitness(pop_objs)

                    # Inner GA loop on surrogates
                    inner_decs = pop_decs.copy()
                    inner_objs = surr_objs.copy()
                    inner_fitness = fitness.copy()

                    for w in range(self.wmax):
                        mating_pool = tournament_selection(2, NI, -inner_fitness)
                        off_decs = ga_generation(inner_decs[mating_pool], muc=20.0, mum=20.0)
                        inner_decs = np.vstack([inner_decs, off_decs])

                        N_inner = inner_decs.shape[0]
                        pred_objs = np.zeros((N_inner, M_surr))
                        for j in range(M_surr):
                            pred_j, _ = gp_predict(models[j], inner_decs, data_type)
                            pred_objs[:, j] = pred_j.ravel()
                        inner_objs = pred_objs

                        inner_decs, inner_objs, inner_fitness = _env_selection(
                            inner_decs, inner_objs, NI, m, status=None
                        )

                    # Select mu solutions for real evaluation
                    sel_decs, _, _ = _env_selection(inner_decs, inner_objs, self.mu, m, status=None)
                    status_for_update = None

                else:
                    # ===== Stage 2: Constraint-aware surrogates =====
                    if nc > 0:
                        max_con_per_col = np.max(np.maximum(0, pop_cons), axis=0)
                        n_inf = np.sum(max_con_per_col > 0)
                    else:
                        n_inf = 0

                    if nc > 0 and n_inf == nc:
                        # Status 1: All constraints violated
                        status = 1
                        cv_col = _normalize_cv(pop_cons)
                        surr_objs = np.hstack([pop_objs, cv_col])
                        M_surr = m + 1
                        models = []
                        for j in range(M_surr):
                            gp = gp_build(pop_decs, surr_objs[:, j:j+1], data_type)
                            models.append(gp)
                        fitness = spea2_fitness(pop_objs, pop_cons)

                    elif nc > 0 and 0 < n_inf < nc:
                        # Status 2: Partial constraint violation
                        status = 2
                        raw_con = np.maximum(0, pop_cons)
                        max_con_per_col = np.max(raw_con, axis=0)
                        active_idx = np.where(max_con_per_col > 0)[0]
                        active_con = raw_con[:, active_idx]
                        cmin = np.min(active_con, axis=0)
                        cmax = np.max(active_con, axis=0)
                        denom = cmax - cmin
                        denom[denom == 0] = 1.0
                        active_con_norm = (active_con - cmin) / denom
                        surr_objs = np.hstack([pop_objs, active_con_norm])
                        M_surr = surr_objs.shape[1]
                        models = []
                        for j in range(M_surr):
                            gp = gp_build(pop_decs, surr_objs[:, j:j+1], data_type)
                            models.append(gp)
                        cv_agg = np.sum(np.maximum(0, pop_cons), axis=1, keepdims=True)
                        fitness = spea2_fitness(np.hstack([pop_objs, cv_agg]))

                    else:
                        # Status 3: All constraints satisfied (or unconstrained)
                        status = 3
                        surr_objs = pop_objs.copy()
                        M_surr = m
                        models = []
                        for j in range(M_surr):
                            gp = gp_build(pop_decs, surr_objs[:, j:j+1], data_type)
                            models.append(gp)
                        fitness = spea2_fitness(pop_objs)

                    # Inner GA loop on surrogates
                    inner_decs = pop_decs.copy()
                    inner_objs = surr_objs.copy()
                    inner_fitness = fitness.copy()

                    for w in range(self.wmax):
                        mating_pool = tournament_selection(2, NI, -inner_fitness)
                        off_decs = ga_generation(inner_decs[mating_pool], muc=20.0, mum=20.0)
                        inner_decs = np.vstack([inner_decs, off_decs])

                        N_inner = inner_decs.shape[0]
                        pred_objs = np.zeros((N_inner, M_surr))
                        for j in range(M_surr):
                            pred_j, _ = gp_predict(models[j], inner_decs, data_type)
                            pred_objs[:, j] = pred_j.ravel()
                        inner_objs = pred_objs

                        inner_decs, inner_objs, inner_fitness = _env_selection(
                            inner_decs, inner_objs, NI, m, status=status
                        )

                    # Select mu solutions for real evaluation
                    sel_decs, _, _ = _env_selection(
                        inner_decs, inner_objs, self.mu, m, status=status
                    )
                    status_for_update = status

                # Remove duplicates
                sel_decs = remove_duplicates(sel_decs, decs[task_i])
                if sel_decs.shape[0] == 0:
                    sel_decs = np.random.rand(self.mu, dim)
                    sel_decs = remove_duplicates(sel_decs, decs[task_i])
                    if sel_decs.shape[0] == 0:
                        iteration += 1
                        continue

                # Expensive evaluation
                new_objs, new_cons = evaluation_single(problem, sel_decs, task_i)
                n_new = sel_decs.shape[0]

                # Update cumulative dataset
                decs[task_i] = np.vstack([decs[task_i], sel_decs])
                objs[task_i] = np.vstack([objs[task_i], new_objs])
                if nc > 0:
                    cons[task_i] = np.vstack([cons[task_i], new_cons])

                # Update population
                pop_cons_for_update = pop_cons if nc > 0 else np.zeros((pop_decs.shape[0], max(1, nc)))
                new_cons_for_update = new_cons if nc > 0 else np.zeros((n_new, max(1, nc)))
                pop_decs, pop_objs, pop_cons_updated = _update_population(
                    pop_decs, pop_objs, pop_cons_for_update,
                    sel_decs, new_objs, new_cons_for_update,
                    NI - self.mu, status=status_for_update
                )
                if nc == 0:
                    pop_cons = np.zeros((pop_decs.shape[0], 0))
                else:
                    pop_cons = pop_cons_updated

                # Update archive
                combined_arc_decs = np.vstack([arc_decs, sel_decs])
                combined_arc_objs = np.vstack([arc_objs, new_objs])
                if nc > 0:
                    combined_arc_cons = np.vstack([arc_cons, new_cons])
                else:
                    combined_arc_cons = np.zeros((combined_arc_objs.shape[0], 1))
                arc_decs, arc_objs, arc_cons_full = _update_archive(
                    combined_arc_decs, combined_arc_objs, combined_arc_cons, N_archive
                )
                if nc == 0:
                    arc_cons = np.zeros((arc_decs.shape[0], 0))
                else:
                    arc_cons = arc_cons_full

                nfes_per_task[task_i] += n_new
                pbar.update(n_new)
                iteration += 1


        pbar.close()
        runtime = time.time() - start_time

        if has_cons:
            all_decs, all_objs, all_cons = build_staircase_history(decs, objs, k=self.mu, db_cons=cons)
        else:
            all_decs, all_objs = build_staircase_history(decs, objs, k=self.mu)
            all_cons = None
        results = build_save_results(
            all_decs, all_objs, runtime, max_nfes_per_task,
            all_cons=all_cons if has_cons else None,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )
        return results

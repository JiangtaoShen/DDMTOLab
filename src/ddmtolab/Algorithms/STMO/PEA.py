"""
Pareto-based Kriging-assisted constrained multi-objective evolutionary algorithm (PEA)

This module implements PEA for expensive constrained multi-objective optimization.
Kriging (Gaussian process) surrogates are built for every objective and every
constraint, and the surrogate-based search is driven by Constrained Probabilistic
Pareto Dominance (CPPD): the probability that solution i is better than solution j on
an objective is weighted by the least probability of feasibility of the two
solutions, and the resulting weighted probabilities define the dominance relation.

References
----------
    [1] Z. Zhang, Y. Wang, G. Sun, and K. Tang. Constrained probabilistic Pareto dominance for expensive constrained multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2024.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.17
Version: 2.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, gp_predict
import warnings

warnings.filterwarnings("ignore")

_EPS_VAR = 1e-20


class PEA:
    """
    Pareto-based Kriging-assisted constrained multi-objective evolutionary
    algorithm for expensive optimization, based on Constrained Probabilistic
    Pareto Dominance.
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

    def __init__(self, problem, n_initial=None, max_nfes=None,
                 wmax=20, mu=5,
                 save_data=True, save_path='./Data', name='PEA', disable_tqdm=True):
        """
        Initialize PEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1). This is
            MATLAB's ``NI = Problem.N``: it is simultaneously the size of the
            initial Latin hypercube design and the size of the working
            population maintained by ``Pop_Update``.
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        wmax : int, optional
            Generations of the surrogate-based evolutionary search (default: 20)
        mu : int, optional
            Number of candidates selected per iteration (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'PEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.wmax = wmax
        self.mu = mu
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the PEA algorithm.

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

        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # MATLAB: NI = Problem.N is both the DoE size and the population size
        ni_per_task = list(n_initial_per_task)

        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()
        has_cons = any(nc > 0 for nc in n_cons)

        sample_success = [1] * nt
        models_obj = [None] * nt
        models_con = [None] * nt
        pop_indices = [np.arange(decs[i].shape[0]) for i in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        stagnation = 0
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break
            nfes_before_sweep = sum(nfes_per_task)

            for i in active_tasks:
                M = n_objs[i]
                NI = ni_per_task[i]
                nc = n_cons[i]

                # ---------------- Model management ----------------
                if sample_success[i]:
                    keep = _distinct_rows(decs[i])
                    try:
                        models_obj[i] = mo_gp_build(decs[i][keep], objs[i][keep], data_type)
                    except Exception:
                        continue
                    if nc > 0:
                        try:
                            models_con[i] = mo_gp_build(decs[i][keep], cons[i][keep], data_type)
                        except Exception:
                            models_con[i] = None
                    else:
                        models_con[i] = None

                # ---------------- Evolutionary search ----------------
                P_decs = decs[i][pop_indices[i]]
                P_objs = objs[i][pop_indices[i]]
                P_cons = cons[i][pop_indices[i]] if nc > 0 else np.zeros((P_decs.shape[0], 0))

                pop_decs, pop_objs, pop_cons, obj_mse, con_mse = _evo_search(
                    P_decs, P_objs, P_cons, self.wmax,
                    models_obj[i], models_con[i], M, data_type
                )

                # ---------------- Candidate selection ----------------
                db_cons = cons[i] if nc > 0 else np.zeros((decs[i].shape[0], 0))
                candidates = _candi_select(
                    pop_decs, pop_objs, pop_cons, obj_mse, con_mse,
                    decs[i], objs[i], db_cons, self.mu, NI
                )

                sample_success[i] = 0
                if candidates is not None and candidates.shape[0] > 0:
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    candidates = candidates[:remaining]

                    cand_objs, cand_cons = evaluation_single(problem, candidates, i)

                    decs[i] = np.vstack([decs[i], candidates])
                    objs[i] = np.vstack([objs[i], cand_objs])
                    cons[i] = np.vstack([cons[i], cand_cons])

                    upd_cons = cons[i] if nc > 0 else None
                    pop_indices[i] = np.where(_pop_update(objs[i], upd_cons, NI))[0]
                    sample_success[i] = 1

                    nfes_per_task[i] += candidates.shape[0]
                    pbar.update(candidates.shape[0])

            if sum(nfes_per_task) == nfes_before_sweep:
                stagnation += 1
                if stagnation >= 50:
                    break
            else:
                stagnation = 0

        pbar.close()
        runtime = time.time() - start_time

        if has_cons:
            all_decs, all_objs, all_cons = build_staircase_history(
                decs, objs, k=self.mu, db_cons=cons)
            results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                         max_nfes=nfes_per_task, all_cons=all_cons,
                                         bounds=problem.bounds,
                                         save_path=self.save_path, filename=self.name,
                                         save_data=self.save_data)
        else:
            all_decs, all_objs = build_staircase_history(decs, objs, k=self.mu)
            results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                         max_nfes=nfes_per_task, bounds=problem.bounds,
                                         save_path=self.save_path, filename=self.name,
                                         save_data=self.save_data)
        return results


# =============================================================================
# Helpers
# =============================================================================

def _distinct_rows(decs):
    """Indices of the first occurrence of each distinct decision vector."""
    _, idx = np.unique(np.round(decs, 10), axis=0, return_index=True)
    return np.sort(idx)


def _predict(models, x, n_out, data_type):
    """Predict mean and MSE with one GP per output. Returns zeros if no model."""
    n = x.shape[0]
    mean = np.zeros((n, n_out))
    mse = np.zeros((n, n_out))
    if models is None or n_out == 0:
        return mean, mse
    for j in range(n_out):
        pred, std = gp_predict(models[j], x, data_type)
        mean[:, j] = pred.flatten()
        mse[:, j] = std.flatten() ** 2
    return mean, mse


def _is_member_rows(a, b):
    """Boolean mask of the rows of ``a`` that occur exactly in ``b``."""
    if b.shape[0] == 0:
        return np.zeros(a.shape[0], dtype=bool)
    bset = set(map(tuple, b.tolist()))
    return np.array([tuple(row) in bset for row in a.tolist()], dtype=bool)


# =============================================================================
# Constrained Probabilistic Pareto Dominance (CPPD) Sorting
# =============================================================================

def _least_feasible_probability(pop_cons, con_mse):
    """MATLAB ``Feasible_Probability``: least probability of feasibility."""
    n = pop_cons.shape[0]
    if pop_cons.shape[1] == 0:
        return np.ones(n)
    p = norm.cdf((0.0 - pop_cons) / np.sqrt(np.maximum(con_mse, _EPS_VAR)))
    return np.minimum(np.min(p, axis=1), 1.0)


def _ndsort_cppd(pop_objs, obj_mse, pop_cons, con_mse, n_sort):
    """
    Non-dominated sorting based on Constrained Probabilistic Pareto Dominance.

    ``A[i,j,k]`` is the probability that solution i is better than solution j on
    objective k; it is weighted by the least feasibility probability ``LFP`` of
    the corresponding solution. Solution i dominates j when
    ``A[i,j,:]*LFP[i] >= (1-A[i,j,:])*LFP[j]`` component-wise and the two
    vectors are not identical. For unconstrained problems ``LFP == 1``, which
    reduces the relation to plain probabilistic dominance.

    Parameters
    ----------
    pop_objs : np.ndarray, shape (N, M)
        Predicted objectives.
    obj_mse : np.ndarray, shape (N, M)
        Prediction MSE of the objectives.
    pop_cons : np.ndarray, shape (N, C)
        Predicted constraints (may have zero columns).
    con_mse : np.ndarray, shape (N, C)
        Prediction MSE of the constraints.
    n_sort : int
        Number of solutions to sort.

    Returns
    -------
    front_no : np.ndarray, shape (N,)
        Front assignment. inf = unassigned.
    max_fno : int
        Last assigned front number.
    """
    N = pop_objs.shape[0]
    lfp = _least_feasible_probability(pop_cons, con_mse)

    sigma = np.sqrt(np.maximum(
        obj_mse[:, np.newaxis, :] + obj_mse[np.newaxis, :, :], _EPS_VAR))
    mean_diff = pop_objs[:, np.newaxis, :] - pop_objs[np.newaxis, :, :]
    A = norm.cdf(-mean_diff / sigma)
    B = 1.0 - A

    x_pd = A * lfp[:, np.newaxis, np.newaxis]
    y_pd = B * lfp[np.newaxis, :, np.newaxis]

    dominate = np.all(x_pd >= y_pd, axis=2) & ~np.all(x_pd == y_pd, axis=2)
    np.fill_diagonal(dominate, False)

    return _min_domination_sort(dominate, n_sort)


def _min_domination_sort(dominate, n_sort):
    """
    Front assignment used by the CPPD sorter: repeatedly extract the solutions
    with the minimum number of dominators among the remaining ones.
    """
    N = dominate.shape[0]
    dominate = dominate.copy()
    front_no = np.full(N, np.inf)
    max_fno = 0
    while np.sum(np.isfinite(front_no)) < min(n_sort, N):
        max_fno += 1
        current = np.where(~np.isfinite(front_no))[0]
        if len(current) == 0:
            break
        dom_count = np.sum(dominate[np.ix_(current, current)], axis=0)
        index = current[dom_count == np.min(dom_count)]
        front_no[index] = max_fno
        dominate[index, :] = False
    return front_no, max_fno


# =============================================================================
# Evolutionary Search
# =============================================================================

def _evo_search(P_decs, P_objs, P_cons, wmax, models_obj, models_con, M, data_type):
    """
    Surrogate-based evolutionary search using CPPD environmental selection.

    The parent population keeps its real objective/constraint values and zero
    prediction variance, exactly as in MATLAB.
    """
    N = P_decs.shape[0]
    n_con = P_cons.shape[1]

    pop_decs = P_decs.copy()
    pop_objs = P_objs.copy()
    pop_cons = P_cons.copy()
    obj_mse = np.zeros((N, M))
    con_mse = np.zeros((N, n_con))

    for _ in range(wmax):
        off_decs = ga_generation(pop_decs, muc=20, mum=20)
        off_objs, off_obj_mse = _predict(models_obj, off_decs, M, data_type)
        off_cons, off_con_mse = _predict(models_con, off_decs, n_con, data_type)

        pop_decs = np.vstack([pop_decs, off_decs])
        pop_objs = np.vstack([pop_objs, off_objs])
        pop_cons = np.vstack([pop_cons, off_cons])
        obj_mse = np.vstack([obj_mse, off_obj_mse])
        con_mse = np.vstack([con_mse, off_con_mse])

        keep = _environmental_selection(pop_objs, obj_mse, pop_cons, con_mse, N)

        pop_decs = pop_decs[keep]
        pop_objs = pop_objs[keep]
        pop_cons = pop_cons[keep]
        obj_mse = obj_mse[keep]
        con_mse = con_mse[keep]

    return pop_decs, pop_objs, pop_cons, obj_mse, con_mse


def _environmental_selection(pop_objs, obj_mse, pop_cons, con_mse, N):
    """
    Environmental selection of the surrogate search. Note that, unlike TEA,
    PEA does *not* normalize the objectives here.
    """
    front_no, max_fno = _ndsort_cppd(pop_objs, obj_mse, pop_cons, con_mse, N)

    next_mask = front_no < max_fno
    last = np.where(front_no == max_fno)[0]

    if max_fno == 1:
        keep = spea2_truncation(pop_objs[last], N)
        next_mask[last[keep]] = True
    else:
        choose = _dist_selection(pop_objs[next_mask], pop_objs[last],
                                 N - int(np.sum(next_mask)))
        next_mask[last[choose]] = True

    return np.where(next_mask)[0]


# =============================================================================
# Candidate Selection
# =============================================================================

def _candi_select(pop_decs, pop_objs, pop_cons, obj_mse, con_mse,
                  db_decs, db_objs, db_cons, mu, NI):
    """
    Select at most ``mu`` candidates for expensive evaluation (MATLAB
    ``Candi_Select``).
    """
    in_db = _is_member_rows(pop_decs, db_decs)
    if np.all(in_db):
        return None

    novel = np.where(~in_db)[0]
    if len(novel) <= mu:
        return _filter_close(pop_decs[novel], db_decs)

    pop_decs = pop_decs[novel]
    pop_objs = pop_objs[novel]
    obj_mse = obj_mse[novel]
    pop_cons = pop_cons[novel]
    con_mse = con_mse[novel]

    zmin = np.min(np.vstack([db_objs, pop_objs]), axis=0)
    zmax = np.max(np.vstack([db_objs, pop_objs]), axis=0)
    rng = np.maximum(zmax - zmin, 10e-10)
    ref_obj = (db_objs - zmin) / rng
    pop_objs = (pop_objs - zmin) / rng
    obj_mse = obj_mse / (rng ** 2)

    # ---- Reference set -------------------------------------------------
    if db_cons.shape[1] > 0:
        num = int(np.sum(np.all(db_cons <= 0, axis=1)))
        front_no, _ = nd_sort(ref_obj, db_cons, ref_obj.shape[0])
    else:
        num = ref_obj.shape[0]
        front_no, _ = nd_sort(ref_obj, ref_obj.shape[0])

    if num >= NI:
        ref_obj = ref_obj[front_no == 1]
    else:
        mask = front_no == 1
        f = 1
        max_f = int(np.max(front_no[np.isfinite(front_no)])) if np.any(np.isfinite(front_no)) else 1
        while np.sum(mask) < NI and f <= max_f:
            mask = mask | (front_no == f)
            f += 1
        ref_obj = ref_obj[mask]

    # ---- Select mu points ----------------------------------------------
    front_no, max_fno = _ndsort_cppd(pop_objs, obj_mse, pop_cons, con_mse, mu)
    next_mask = front_no < max_fno
    last = list(np.where(front_no == max_fno)[0])
    n_need = mu - int(np.sum(next_mask))

    if len(last) == n_need:
        next_mask[np.array(last, dtype=int)] = True
    elif len(last) > n_need:
        ref = np.vstack([ref_obj, pop_objs[next_mask]])
        for _ in range(n_need):
            cand = pop_objs[np.array(last, dtype=int)]
            pos = int(np.argmax(np.min(cdist(cand, ref), axis=1)))
            next_mask[last[pos]] = True
            ref = np.vstack([ref, pop_objs[last[pos]].reshape(1, -1)])
            last.pop(pos)

    return _filter_close(pop_decs[next_mask], db_decs)


def _filter_close(cand, db_decs):
    """Drop candidates closer than 1e-5 to any already evaluated solution."""
    if cand.shape[0] == 0:
        return None
    keep = [cand[k] for k in range(cand.shape[0])
            if db_decs.shape[0] == 0 or np.min(cdist(cand[k:k + 1], db_decs)) > 1e-5]
    return np.array(keep) if keep else None


# =============================================================================
# Population Update
# =============================================================================

def _pop_update(pop_objs, pop_cons, N):
    """
    MATLAB ``Pop_Update``: rebuild the working population from the database
    with constrained non-dominated sorting.
    """
    n_total = pop_objs.shape[0]
    if n_total <= N:
        return np.ones(n_total, dtype=bool)

    if pop_cons is not None and pop_cons.shape[1] > 0:
        front_no, max_fno = nd_sort(pop_objs, pop_cons, N)
    else:
        front_no, max_fno = nd_sort(pop_objs, N)

    next_mask = front_no < max_fno
    last = np.where(front_no == max_fno)[0]

    if max_fno == 1:
        keep = spea2_truncation(pop_objs[last], N)
        next_mask[last[keep]] = True
    else:
        choose = _dist_selection(pop_objs[next_mask], pop_objs[last],
                                 N - int(np.sum(next_mask)))
        next_mask[last[choose]] = True

    return next_mask


# =============================================================================
# Diversity Helper
# =============================================================================

def _dist_selection(selected_objs, candidate_objs, n_select):
    """
    MATLAB ``Dist_Selection``: greedily add the candidate whose nearest
    neighbour among the already selected solutions is farthest away.

    Parameters
    ----------
    selected_objs : np.ndarray, shape (N1, M)
        Objectives of the already selected solutions.
    candidate_objs : np.ndarray, shape (N2, M)
        Objectives of the last-front candidates.
    n_select : int
        Number of candidates to add.

    Returns
    -------
    chosen : np.ndarray
        Indices into ``candidate_objs``.
    """
    N2 = candidate_objs.shape[0]
    if N2 <= n_select:
        return np.arange(N2)

    dist = cdist(candidate_objs, selected_objs)
    min_dist = np.min(dist, axis=1)
    remaining = np.ones(N2, dtype=bool)
    chosen = []

    for _ in range(n_select):
        avail = np.where(remaining)[0]
        if len(avail) == 0:
            break
        pos = avail[int(np.argmax(min_dist[avail]))]
        chosen.append(pos)
        remaining[pos] = False
        # the newly selected candidate joins the reference set
        new_d = cdist(candidate_objs, candidate_objs[pos:pos + 1]).ravel()
        min_dist = np.minimum(min_dist, new_d)

    return np.array(chosen, dtype=int)

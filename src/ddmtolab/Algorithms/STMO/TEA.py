"""
Two-phase Evolutionary Algorithm (TEA)

This module implements TEA for expensive constrained multi-objective optimization.
The search is driven by Kriging (Gaussian process) surrogates and a probabilistic
dominance relation (PDPD) that splits the objectives of every compared pair into a
"certain" group (the two dominance probabilities differ by more than epsilon) and an
"uncertain" group (aggregated by a product of probabilities). The algorithm runs in
two phases: phase 1 ignores the constraints, phase 2 additionally models the
constraints and uses the constrained PDPD relation. The transition is triggered when
the newly sampled candidates stop improving the feasible non-dominated set for
``ct_max`` consecutive iterations.

References
----------
    [1] Z. Zhang, Y. Wang, J. Liu, G. Sun, and K. Tang. A two-phase Kriging-assisted evolutionary algorithm for expensive constrained multiobjective optimization problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(8): 4579-4591.

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


class TEA:
    """
    Two-phase Evolutionary Algorithm for expensive (constrained) multi-objective
    optimization using PDPD sorting and Kriging surrogates.
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
                 save_data=True, save_path='./Data', name='TEA', disable_tqdm=True):
        """
        Initialize TEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1). This is
            MATLAB's ``NI = Problem.N``: it is simultaneously the size of the
            initial Latin hypercube design and the size of the working
            population maintained by ``Pop_Reselect``.
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
            Name for the experiment (default: 'TEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.wmax = wmax
        self.mu = mu
        self.ct_max = 2
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the TEA algorithm.

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

        # Per-task state (MATLAB globals phase/ct and the flag sample_success)
        phase = [1] * nt
        ct = [0] * nt
        sample_success = [1] * nt
        models_obj = [None] * nt
        models_con = [None] * nt
        # Working population P: indices into the database
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

                # ---------------- Surrogate construction ----------------
                if sample_success[i]:
                    keep = _distinct_rows(decs[i])
                    try:
                        models_obj[i] = mo_gp_build(decs[i][keep], objs[i][keep], data_type)
                    except Exception:
                        continue
                    if phase[i] == 2 and nc > 0:
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
                    P_decs, P_objs, P_cons, self.wmax, models_obj[i], models_con[i],
                    M, phase[i], data_type
                )

                # ---------------- Candidate selection ----------------
                db_cons = cons[i] if nc > 0 else np.zeros((decs[i].shape[0], 0))
                candidates = _candi_select(
                    pop_decs, pop_objs, pop_cons, obj_mse, con_mse,
                    decs[i], objs[i], db_cons, self.mu, phase[i], NI
                )

                sample_success[i] = 0
                if candidates is not None and candidates.shape[0] > 0:
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    candidates = candidates[:remaining]

                    cand_objs, cand_cons = evaluation_single(problem, candidates, i)
                    sample_success[i] = 1

                    # ---------------- Phase transition ----------------
                    phase[i], ct[i] = _phase_trans(
                        objs[i], db_cons,
                        cand_objs, cand_cons if nc > 0 else np.zeros((candidates.shape[0], 0)),
                        ct[i], self.ct_max, phase[i]
                    )

                    decs[i] = np.vstack([decs[i], candidates])
                    objs[i] = np.vstack([objs[i], cand_objs])
                    cons[i] = np.vstack([cons[i], cand_cons])

                    nfes_per_task[i] += candidates.shape[0]
                    pbar.update(candidates.shape[0])

                # ---------------- Population reselection ----------------
                db_cons = cons[i] if nc > 0 else None
                pop_indices[i] = np.where(_pop_reselect(objs[i], db_cons, NI, phase[i]))[0]

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
    _, idx = np.unique(np.round(decs, 12), axis=0, return_index=True)
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


# =============================================================================
# PDPD Non-dominated Sorting
# =============================================================================

def _feasible_probability(pop_cons, con_mse):
    """
    MATLAB ``Feasible_Probability``: least (LPoF) and total (TPoF) probability
    of feasibility of every solution.
    """
    n = pop_cons.shape[0]
    if pop_cons.shape[1] == 0:
        return np.ones(n), np.ones(n)
    p = norm.cdf((0.0 - pop_cons) / np.sqrt(np.maximum(con_mse, _EPS_VAR)))
    lpof = np.minimum(np.min(p, axis=1), 1.0)
    tpof = np.prod(p, axis=1)
    return lpof, tpof


def _ndsort_pdpd(pop_objs, obj_mse, n_sort, pop_cons=None, con_mse=None, epsilon=0.75):
    """
    Non-dominated sorting based on Probabilistic Dominant Product Dominance.

    For every compared pair the objectives are split into an "uncertain" group
    (``|Pi - Pj| <= epsilon``), which is aggregated by the product of the
    probabilities, and a "certain" group which is compared component-wise.
    Solution i dominates j when it is at least as good on every certain
    objective and on the aggregated product, with strict improvement somewhere.

    When ``pop_cons`` is given (phase 2) the constrained variant is used: pairs
    whose least feasibility probabilities are both >= 0.5 or exactly equal are
    compared with PDPD, otherwise the solution with the larger total
    feasibility probability dominates.

    Parameters
    ----------
    pop_objs : np.ndarray, shape (N, M)
        Predicted objectives (normalized).
    obj_mse : np.ndarray, shape (N, M)
        Prediction MSE (normalized).
    n_sort : int
        Number of solutions to sort.
    pop_cons, con_mse : np.ndarray, optional
        Predicted constraints and their MSE (phase 2 only).
    epsilon : float
        Threshold splitting certain/uncertain objectives (default: 0.75).

    Returns
    -------
    front_no : np.ndarray, shape (N,)
        Front assignment. inf = unassigned.
    max_fno : int
        Last assigned front number.
    """
    N = pop_objs.shape[0]

    sigma = np.sqrt(np.maximum(
        obj_mse[:, np.newaxis, :] + obj_mse[np.newaxis, :, :], _EPS_VAR))
    mean_diff = pop_objs[:, np.newaxis, :] - pop_objs[np.newaxis, :, :]
    # A[i,j,k] = P(solution i is better than solution j on objective k)
    A = norm.cdf(-mean_diff / sigma)
    B = 1.0 - A

    uncertain = np.abs(A - B) <= epsilon
    certain = ~uncertain

    # component-wise comparison on the certain objectives
    ge_cert = np.all(np.where(certain, A >= B, True), axis=2)
    eq_cert = np.all(np.where(certain, A == B, True), axis=2)
    # product dominance on the uncertain objectives
    pd_i = np.prod(np.where(uncertain, A, 1.0), axis=2)
    pd_j = np.prod(np.where(uncertain, B, 1.0), axis=2)

    pdpd_dom = ge_cert & (pd_i >= pd_j) & ~(eq_cert & (pd_i == pd_j))

    iu, ju = np.triu_indices(N, 1)
    if pop_cons is None or pop_cons.shape[1] == 0:
        flag1 = pdpd_dom[iu, ju]
        flag2 = pdpd_dom[ju, iu]
    else:
        lpof, tpof = _feasible_probability(pop_cons, con_mse)
        use_pdpd = ((lpof[iu] >= 0.5) & (lpof[ju] >= 0.5)) | (lpof[iu] == lpof[ju])
        # MATLAB: [~,flag] = max([TPoF(i),TPoF(j)]) -> ties resolved towards i
        tie_flag1 = tpof[iu] >= tpof[ju]
        flag1 = np.where(use_pdpd, pdpd_dom[iu, ju], tie_flag1)
        flag2 = np.where(use_pdpd, pdpd_dom[ju, iu], ~tie_flag1)

    dominate = np.zeros((N, N), dtype=bool)
    dominate[iu[flag1], ju[flag1]] = True
    dominate[ju[flag2], iu[flag2]] = True

    return _min_domination_sort(dominate, n_sort)


def _min_domination_sort(dominate, n_sort):
    """
    Front assignment used by the PDPD/CPPD/DIPD sorters: repeatedly extract the
    solutions with the minimum number of dominators among the remaining ones.
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

def _evo_search(P_decs, P_objs, P_cons, wmax, models_obj, models_con, M, phase, data_type):
    """
    Surrogate-based evolutionary search using PDPD environmental selection.

    The parent population enters the loop with its *real* objective and
    constraint values and zero prediction variance, exactly as in MATLAB.
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
        if phase == 2 and n_con > 0 and models_con is not None:
            off_cons, off_con_mse = _predict(models_con, off_decs, n_con, data_type)
        else:
            off_cons = np.zeros((off_decs.shape[0], n_con))
            off_con_mse = np.zeros((off_decs.shape[0], n_con))

        pop_decs = np.vstack([pop_decs, off_decs])
        pop_objs = np.vstack([pop_objs, off_objs])
        pop_cons = np.vstack([pop_cons, off_cons])
        obj_mse = np.vstack([obj_mse, off_obj_mse])
        con_mse = np.vstack([con_mse, off_con_mse])

        keep = _env_selection(pop_objs, obj_mse, pop_cons, con_mse, N, phase)

        pop_decs = pop_decs[keep]
        pop_objs = pop_objs[keep]
        pop_cons = pop_cons[keep]
        obj_mse = obj_mse[keep]
        con_mse = con_mse[keep]

    return pop_decs, pop_objs, pop_cons, obj_mse, con_mse


def _env_selection(pop_objs, obj_mse, pop_cons, con_mse, N, phase):
    """Environmental selection of the surrogate search (MATLAB Evo_Search)."""
    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    rng = np.maximum(zmax - zmin, 10e-10)
    norm_objs = (pop_objs - zmin) / rng
    norm_mse = obj_mse / (rng ** 2)

    if phase == 2 and pop_cons.shape[1] > 0:
        front_no, max_fno = _ndsort_pdpd(norm_objs, norm_mse, N, pop_cons, con_mse)
    else:
        front_no, max_fno = _ndsort_pdpd(norm_objs, norm_mse, N)

    next_mask = front_no < max_fno
    last = np.where(front_no == max_fno)[0]

    if max_fno == 1:
        keep = spea2_truncation(norm_objs[last], N)
        next_mask[last[keep]] = True
    else:
        choose = _dis_selection_evo(last, N - int(np.sum(next_mask)))
        next_mask[last[choose]] = True

    return np.where(next_mask)[0]


def _dis_selection_evo(last, mu):
    """
    Port of ``Dis_Selection`` as it appears inside PlatEMO's TEA/Evo_Search.

    The MATLAB routine allocates its distance matrix implicitly, which leaves a
    zero on the diagonal; the crowding measure ``D = 1./(Distance(:,1)+2)`` is
    therefore the constant 0.5 for every solution and MATLAB's stable ``sort``
    returns the identity permutation. The routine consequently keeps the first
    ``mu`` members of the last front in index order. (``Pop_Reselect`` contains
    a second copy of the same function which pre-allocates ``inf(N)`` and does
    perform the intended isolation-based selection - see
    ``_dis_selection_isolated``.)
    """
    return np.arange(min(mu, len(last)))


def _dis_selection_isolated(pop_objs, last, mu):
    """
    Port of ``Dis_Selection`` as it appears inside PlatEMO's TEA/Pop_Reselect,
    where the distance matrix is pre-allocated with ``inf(N)``: the ``mu``
    solutions of the last front with the largest nearest-neighbour distance are
    selected.
    """
    dist = cdist(pop_objs, pop_objs)
    np.fill_diagonal(dist, np.inf)
    d = 1.0 / (np.min(dist, axis=1)[last] + 2.0)
    return np.argsort(d, kind='stable')[:mu]


# =============================================================================
# Candidate Selection
# =============================================================================

def _candi_select(pop_decs, pop_objs, pop_cons, obj_mse, con_mse,
                  db_decs, db_objs, db_cons, mu, phase, NI):
    """
    Select at most ``mu`` candidates for expensive evaluation (MATLAB
    ``Candi_Select``).
    """
    in_db = _is_member_rows(pop_decs, db_decs)
    if np.all(in_db):
        return None

    idx = np.where(~in_db)[0]
    chosen = _selection(pop_objs[idx], obj_mse[idx], pop_cons[idx], con_mse[idx],
                        db_objs, db_cons, mu, phase, NI)
    cand = pop_decs[idx[chosen]]

    keep = []
    for k in range(cand.shape[0]):
        if np.min(cdist(cand[k:k + 1], db_decs)) > 1e-5:
            keep.append(cand[k])
    return np.array(keep) if keep else None


def _is_member_rows(a, b):
    """Boolean mask of the rows of ``a`` that occur exactly in ``b``."""
    if b.shape[0] == 0:
        return np.zeros(a.shape[0], dtype=bool)
    bset = set(map(tuple, b.tolist()))
    return np.array([tuple(row) in bset for row in a.tolist()], dtype=bool)


def _selection(pop_objs, obj_mse, pop_cons, con_mse, all_obj, all_con, mu, phase, NI):
    """MATLAB ``Selection`` nested in Candi_Select. Returns indices of the picks."""
    zmin = np.min(np.vstack([all_obj, pop_objs]), axis=0)
    zmax = np.max(np.vstack([all_obj, pop_objs]), axis=0)
    rng = np.maximum(zmax - zmin, 10e-10)
    ref_obj = (all_obj - zmin) / rng
    pop_objs = (pop_objs - zmin) / rng
    obj_mse = obj_mse / (rng ** 2)

    # ---- Reference set -------------------------------------------------
    if phase == 2 and all_con.shape[1] > 0:
        num = int(np.sum(np.all(all_con <= 0, axis=1)))
        front_no, _ = nd_sort(ref_obj, all_con, ref_obj.shape[0])
        if num > NI:
            ref_obj = ref_obj[front_no == 1]
        else:
            mask = front_no == 1
            f = 1
            max_f = int(np.max(front_no[np.isfinite(front_no)])) if np.any(np.isfinite(front_no)) else 1
            while np.sum(mask) <= NI and f <= max_f:
                mask = mask | (front_no == f)
                f += 1
            ref_obj = ref_obj[mask]
    else:
        front_no, _ = nd_sort(ref_obj, ref_obj.shape[0])
        ref_obj = ref_obj[front_no == 1]

    # ---- Select mu points ----------------------------------------------
    if phase == 2 and pop_cons.shape[1] > 0:
        front_no, max_fno = _ndsort_pdpd(pop_objs, obj_mse, mu, pop_cons, con_mse)
    else:
        front_no, max_fno = _ndsort_pdpd(pop_objs, obj_mse, mu)

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

    return np.where(next_mask)[0]


# =============================================================================
# Phase Transition
# =============================================================================

def _phase_trans(db_objs, db_cons, c_objs, c_cons, ct, ct_max, phase):
    """
    MATLAB ``Phase_Trans``: switch from phase 1 to phase 2 once the newly
    sampled candidates fail to improve the feasible non-dominated set for
    ``ct_max`` consecutive iterations.
    """
    if phase != 1:
        return phase, ct

    feasible = np.all(db_cons <= 0, axis=1) if db_cons.shape[1] > 0 \
        else np.ones(db_objs.shape[0], dtype=bool)
    if not np.any(feasible):
        return 1, ct

    if c_cons.shape[1] > 0:
        c_feasible = np.all(c_cons <= 0, axis=1)
    else:
        c_feasible = np.ones(c_objs.shape[0], dtype=bool)

    ref = db_objs[feasible]
    new_feas = c_objs[c_feasible]
    new_infeas = c_objs[~c_feasible]

    cond_a = (new_feas.shape[0] == 0) or (_set_dominate(new_feas, ref) == 3)
    cond_b = (new_infeas.shape[0] == 0) or (_set_dominate(new_infeas, ref) in (1, 3))

    index = 0
    if cond_a and cond_b:
        ct += 1
        if ct >= ct_max:
            index = 1
    else:
        ct = 0

    return (2 if index == 1 else 1), ct


def _set_dominate(a_obj, b_obj):
    """
    MATLAB ``set_dominate``: relation between the non-dominated fronts of two
    objective sets. 1 = A dominates B, 2 = B dominates A, 3 = mutually
    non-dominated, 4 = mixed.
    """
    fb, _ = nd_sort(b_obj, b_obj.shape[0])
    b_obj = b_obj[fb == 1]
    fa, _ = nd_sort(a_obj, a_obj.shape[0])
    a_obj = a_obj[fa == 1]

    row_flags = []
    for i in range(a_obj.shape[0]):
        vals = []
        for j in range(b_obj.shape[0]):
            if np.all(a_obj[i] == b_obj[j]):
                vals.append(3)
            elif np.all(a_obj[i] <= b_obj[j]):
                vals.append(1)
            elif np.all(a_obj[i] >= b_obj[j]):
                vals.append(2)
            else:
                vals.append(3)
        row_flags.append(_aggregate_flags(vals))
    return _aggregate_flags(row_flags)


def _aggregate_flags(vals):
    """Collapse a list of pairwise flags exactly like MATLAB's set_dominate."""
    uni = sorted(set(vals))
    if len(uni) == 1:
        return uni[0]
    if len(uni) == 2:
        if uni == [1, 3]:
            return 1
        if uni == [2, 3]:
            return 2
        return 4
    return 4


# =============================================================================
# Population Reselection
# =============================================================================

def _pop_reselect(pop_objs, pop_cons, N, phase):
    """
    MATLAB ``Pop_Reselect``: rebuild the working population from the database
    with standard (constrained in phase 2) non-dominated sorting.
    """
    n_total = pop_objs.shape[0]
    if n_total <= N:
        return np.ones(n_total, dtype=bool)

    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    rng = np.maximum(zmax - zmin, 10e-10)
    norm_objs = (pop_objs - zmin) / rng

    if phase == 2 and pop_cons is not None and pop_cons.shape[1] > 0:
        front_no, max_fno = nd_sort(norm_objs, pop_cons, N)
    else:
        front_no, max_fno = nd_sort(norm_objs, N)

    next_mask = front_no < max_fno
    last = np.where(front_no == max_fno)[0]

    if max_fno == 1:
        keep = spea2_truncation(norm_objs[last], N)
        next_mask[last[keep]] = True
    else:
        choose = _dis_selection_isolated(norm_objs, last, N - int(np.sum(next_mask)))
        next_mask[last[choose]] = True

    return next_mask

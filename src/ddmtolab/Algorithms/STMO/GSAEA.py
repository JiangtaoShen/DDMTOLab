"""
Grouping via Sensitivity Analysis Evolutionary Algorithm (GSAEA)

This module implements GSAEA for high-dimensional expensive multi-objective
optimization. Sobol' first order sensitivity indices, computed on Kriging models
of each objective, split the decision variables into convergence-related and
diversity-related subsets. Each subset is then searched by its own
surrogate-assisted operator: differential evolution driven by a predicted
convergence indicator, and a genetic algorithm driven by per-cluster objective
models. The reconstructed candidates are ranked by a predicted shift-based
density estimation value, and the best ones are evaluated exactly.

References
----------
    [1] W. Chen, Z. Li, J. Yu, and Y. Pu. Grouping via sensitivity analysis evolutionary algorithm for high-dimensional expensive multi-objective optimization. Complex & Intelligent Systems, 2026, 12: 66.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.14
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
import warnings

warnings.filterwarnings("ignore")


class GSAEA:
    """
    Grouping via Sensitivity Analysis Evolutionary Algorithm for high-dimensional
    expensive multi-objective optimization.
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=50,
                 k=5, n_clusters=None, wmax=20, n_sobol=256,
                 F=0.5, CR=0.9, muc=20, mum=20,
                 save_data=True, save_path='./Data', name='GSAEA', disable_tqdm=True):
        """
        Initialize GSAEA.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial LHS samples per task (default: 100, capped so that
            at least one infill iteration fits inside the budget)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        n : int or List[int], optional
            Population size per task (default: 50)
        k : int, optional
            Number of candidates re-evaluated per iteration (default: 5)
        n_clusters : int, optional
            Number of K-means clusters used for the local objective models.
            ``None`` follows the paper's default nc = M (default: None)
        wmax : int, optional
            Number of generations of the surrogate-assisted DE search on the
            convergence-related variables (default: 20)
        n_sobol : int, optional
            Base sample size of the Sobol' index estimator; the grouping costs
            ``n_sobol * (D + 2)`` cheap Kriging predictions and no real
            evaluations (default: 256)
        F : float, optional
            DE differential weight (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.9)
        muc : float, optional
            Distribution index of the SBX crossover (default: 20)
        mum : float, optional
            Distribution index of the polynomial mutation (default: 20)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'GSAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)

        Notes
        -----
        The paper fixes N = 50, NI = 100, FEs_max = 300, k = 5 and nc = M, and
        leaves ``wmax`` and the Sobol' sample size unspecified; the defaults
        above keep the surrogate search cost of the same order as the DE-based
        baselines the paper compares against.
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.n = n
        self.k = k
        self.n_clusters = n_clusters
        self.wmax = wmax
        self.n_sobol = n_sobol
        self.F = F
        self.CR = CR
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the GSAEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n_objs = problem.n_objs
        data_type = torch.float

        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        if self.n_initial is None:
            # NI = 100 in the paper; keep room for at least one infill iteration
            n_initial_per_task = [min(100, max(2 * n_objs[i] + 1, m - self.k))
                                  for i, m in enumerate(max_nfes_per_task)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # The grouping is computed once and then fixed for the whole run
        groups = [None] * nt
        # A task whose infill has run dry is retired; its budget stays unspent
        # rather than being reported as consumed
        exhausted = [False] * nt

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt)
                            if nfes_per_task[i] < max_nfes_per_task[i] and not exhausted[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                N = n_per_task[i]
                nc = self.n_clusters if self.n_clusters is not None else M

                # ===== Variable grouping via Sobol' sensitivity analysis =====
                if groups[i] is None:
                    groups[i] = _group_variables(decs[i], objs[i], self.n_sobol, data_type)
                cv_idx, dv_idx = groups[i]

                # ===== Construction of the surrogate models (Algorithm 2) =====
                obj_models, centers, con_model, sde_model = _build_surrogates(
                    decs[i], objs[i], nc, M
                )

                # The working population is the best N solutions of the archive
                pop_decs = decs[i][_env_selection(objs[i], N)]

                # ===== Optimization of CV by surrogate-assisted DE (Algorithm 4) =====
                sub_cv = _convergence_optimization(
                    pop_decs, cv_idx, con_model, self.wmax, self.F, self.CR
                )

                # ===== Optimization of DV by surrogate-assisted GA (Algorithm 5) =====
                sub_dv = _diversity_optimization(
                    pop_decs, dv_idx, con_model, obj_models, centers, M, N,
                    self.muc, self.mum
                )

                # ===== Infill sampling criterion =====
                candidates = _reconstruct(sub_cv, sub_dv, cv_idx, dv_idx)
                pred_sde = _predict(sde_model, candidates)
                # Larger SDE indicates a higher quality solution
                candidates = candidates[np.argsort(-pred_sde, kind='stable')]

                candidates = remove_duplicates(candidates, decs[i])
                candidates = candidates[:self.k]
                candidates = candidates[:max_nfes_per_task[i] - nfes_per_task[i]]
                if candidates.shape[0] == 0:
                    # Every candidate repeats a solution already evaluated, so the
                    # search has converged; retire the task instead of spinning
                    exhausted[i] = True
                    continue

                cand_objs, _ = evaluation_single(problem, candidates, i)
                decs[i] = np.vstack([decs[i], candidates])
                objs[i] = np.vstack([objs[i], cand_objs])

                nfes_per_task[i] += candidates.shape[0]
                pbar.update(candidates.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.k)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# Performance indicators
# =============================================================================

def _con_values(objs):
    """
    Convergence indicator Con, Eq. (2).

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (n, M)

    Returns
    -------
    con : np.ndarray
        Distance of each solution to the ideal point, shape (n,). Smaller is better.
    """
    ideal = np.min(objs, axis=0)
    return np.sqrt(np.sum((objs - ideal) ** 2, axis=1))


def _sde_values(objs):
    """
    Shift-based density estimation, Eqs. (3) and (4).

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (n, M)

    Returns
    -------
    sde : np.ndarray
        SDE value of each solution, shape (n,). Larger is better.
    """
    n = objs.shape[0]
    if n < 2:
        return np.zeros(n)

    fmin = np.min(objs, axis=0)
    fmax = np.max(objs, axis=0)
    span = fmax - fmin
    span[span == 0] = 1.0
    norm = (objs - fmin) / span

    sde = np.zeros(n)
    for p in range(n):
        # max{0, f'(y) - f'(x)} shifts every other solution towards x
        shifted = np.maximum(0.0, norm - norm[p])
        dist = np.sqrt(np.sum(shifted ** 2, axis=1))
        dist[p] = np.inf
        sde[p] = np.min(dist)
    return sde


# =============================================================================
# Variable grouping via Sobol' sensitivity analysis (Algorithm 3)
# =============================================================================

def _sobol_first_order(model, D, n_base, data_type, chunk=8192):
    """
    First order Sobol' indices of a Kriging model over the unit cube.

    Uses the Saltelli estimator ``S_d = mean(y_B * (y_C^d - y_A)) / Var(y)``,
    which needs ``n_base * (D + 2)`` cheap surrogate predictions and no real
    function evaluations.

    Parameters
    ----------
    model : SingleTaskGP
        Kriging model of one objective
    D : int
        Number of decision variables
    n_base : int
        Base sample size
    data_type : torch.dtype
        Tensor dtype used by the surrogate
    chunk : int, optional
        Maximum number of query points predicted at once (default: 8192)

    Returns
    -------
    s1 : np.ndarray
        First order indices clipped to [0, 1], shape (D,)
    """
    A = np.random.rand(n_base, D)
    B = np.random.rand(n_base, D)

    def predict(X):
        out = np.empty(X.shape[0])
        for start in range(0, X.shape[0], chunk):
            stop = min(start + chunk, X.shape[0])
            out[start:stop] = mo_gp_predict([model], X[start:stop], data_type)[:, 0]
        return out

    y_a = predict(A)
    y_b = predict(B)

    var = np.var(np.concatenate([y_a, y_b]))
    if var <= 0 or not np.isfinite(var):
        return np.zeros(D)

    s1 = np.zeros(D)
    for d in range(D):
        C = A.copy()
        C[:, d] = B[:, d]
        y_c = predict(C)
        s1[d] = np.mean(y_b * (y_c - y_a)) / var
    return np.clip(s1, 0.0, 1.0)


def _group_variables(arc_decs, arc_objs, n_sobol, data_type):
    """
    Split the decision variables into convergence- and diversity-related subsets.

    Parameters
    ----------
    arc_decs : np.ndarray
        Evaluated decisions, shape (n, D)
    arc_objs : np.ndarray
        Evaluated objectives, shape (n, M)
    n_sobol : int
        Base sample size of the Sobol' estimator
    data_type : torch.dtype
        Tensor dtype used by the surrogate

    Returns
    -------
    cv_idx : np.ndarray
        Indices of the convergence-related variables
    dv_idx : np.ndarray
        Indices of the diversity-related variables
    """
    D = arc_decs.shape[1]
    M = arc_objs.shape[1]

    models = mo_gp_build(arc_decs, arc_objs, data_type)
    s = np.zeros((M, D))
    for m in range(M):
        s[m] = _sobol_first_order(models[m], D, n_sobol, data_type)

    ms = np.mean(s, axis=0)                       # Eq. (9)
    gt = float(np.mean(ms))                       # Eq. (10)
    dv_mask = ms > gt                             # Eq. (11)

    # A degenerate split leaves one operator with nothing to search; fall back to
    # the median, which always yields two non-empty groups for D >= 2.
    if not dv_mask.any() or dv_mask.all():
        dv_mask = ms > np.median(ms)
    if not dv_mask.any() or dv_mask.all():
        dv_mask = np.zeros(D, dtype=bool)
        dv_mask[: max(1, D // 2)] = True

    dv_idx = np.where(dv_mask)[0]
    cv_idx = np.where(~dv_mask)[0]
    return cv_idx, dv_idx


# =============================================================================
# Construction of the surrogate models (Algorithm 2)
# =============================================================================

def _fit(X, y):
    """
    Fit one RBF model, returning it together with the design sites it needs at
    prediction time.

    Parameters
    ----------
    X : np.ndarray
        Inputs, shape (n, D)
    y : np.ndarray
        Responses, shape (n,)

    Returns
    -------
    bundle : tuple
        ``(model, design_sites)`` accepted by :func:`_predict`
    """
    mS, mY = dsmerge(X, y)
    return rbf_build(mS, mY), mS


def _predict(bundle, X):
    """
    Evaluate a bundle produced by :func:`_fit`.

    Parameters
    ----------
    bundle : tuple
        ``(model, design_sites)``
    X : np.ndarray
        Query points, shape (nq, D)

    Returns
    -------
    y : np.ndarray
        Predicted responses, shape (nq,)
    """
    model, mS = bundle
    return np.asarray(rbf_predict(model, mS, X)).ravel()


def _build_surrogates(arc_decs, arc_objs, nc, M):
    """
    Build the local objective models and the global Con and SDE models.

    Parameters
    ----------
    arc_decs : np.ndarray
        Evaluated decisions, shape (n, D)
    arc_objs : np.ndarray
        Evaluated objectives, shape (n, M)
    nc : int
        Number of K-means clusters
    M : int
        Number of objectives

    Returns
    -------
    obj_models : List[List[tuple]]
        ``obj_models[c][j]`` predicts objective j inside cluster c
    centers : np.ndarray
        Cluster centers in the decision space, shape (nc_eff, D)
    con_model : tuple
        Model of the Con values
    sde_model : tuple
        Model of the SDE values
    """
    n = arc_decs.shape[0]
    nc = int(max(1, min(nc, n)))

    # --- Local objective models ---
    if nc == 1:
        labels = np.zeros(n, dtype=int)
    else:
        labels = kmeans_clustering(arc_decs, nc)

    centers = []
    obj_models = []
    for c in range(nc):
        members = np.where(labels == c)[0]
        # An RBF interpolant needs a handful of distinct sites; tiny clusters
        # fall back to the whole archive so every center keeps a usable model.
        train_idx = members if members.shape[0] >= 3 else np.arange(n)
        centers.append(arc_decs[members].mean(axis=0) if members.shape[0] > 0
                       else arc_decs.mean(axis=0))
        obj_models.append([_fit(arc_decs[train_idx], arc_objs[train_idx, j])
                           for j in range(M)])

    # --- Convergence model ---
    con_model = _fit(arc_decs, _con_values(arc_objs))

    # --- Diversity and convergence model ---
    sde_model = _fit(arc_decs, _sde_values(arc_objs))

    return obj_models, np.vstack(centers), con_model, sde_model


def _predict_objs(decs, obj_models, centers):
    """
    Predict objectives with the local model of the nearest cluster center.

    Parameters
    ----------
    decs : np.ndarray
        Query points, shape (n, D)
    obj_models : List[List[tuple]]
        Per-cluster objective models
    centers : np.ndarray
        Cluster centers, shape (nc, D)

    Returns
    -------
    objs : np.ndarray
        Predicted objectives, shape (n, M)
    """
    M = len(obj_models[0])
    objs = np.zeros((decs.shape[0], M))
    assign = np.argmin(np.linalg.norm(decs[:, None, :] - centers[None, :, :], axis=2), axis=1)
    for c in range(len(obj_models)):
        rows = np.where(assign == c)[0]
        if rows.shape[0] == 0:
            continue
        for j in range(M):
            objs[rows, j] = _predict(obj_models[c][j], decs[rows])
    return objs


# =============================================================================
# Evolutionary optimization (Algorithms 4 and 5)
# =============================================================================

def _convergence_optimization(pop_decs, cv_idx, con_model, wmax, F, CR):
    """
    Surrogate-assisted DE on the convergence-related variables (Algorithm 4).

    Parameters
    ----------
    pop_decs : np.ndarray
        Current population, shape (N, D)
    cv_idx : np.ndarray
        Indices of the convergence-related variables
    con_model : tuple
        Model of the Con values
    wmax : int
        Number of generations
    F : float
        DE differential weight
    CR : float
        DE crossover rate

    Returns
    -------
    sub_cv : np.ndarray
        Population after the convergence search, shape (N, D)
    """
    pop = pop_decs.copy()
    if cv_idx.shape[0] == 0 or pop.shape[0] < 4:
        return pop

    con = _predict(con_model, pop)
    for _ in range(wmax):
        # Con is minimized, so the tournament ranks on its negation
        parents_idx = tournament_selection(2, pop.shape[0], -con)
        parents = pop[parents_idx]

        # The DE operator only touches CV; the remaining variables are inherited
        offspring = parents.copy()
        offspring[:, cv_idx] = de_generation(parents[:, cv_idx], F, CR)

        off_con = _predict(con_model, offspring)
        par_con = con[parents_idx]

        better = off_con < par_con
        pop[parents_idx[better]] = offspring[better]
        con[parents_idx[better]] = off_con[better]
    return pop


def _diversity_optimization(pop_decs, dv_idx, con_model, obj_models, centers,
                            M, N, muc, mum):
    """
    Surrogate-assisted GA on the diversity-related variables (Algorithm 5).

    Parameters
    ----------
    pop_decs : np.ndarray
        Current population, shape (N, D)
    dv_idx : np.ndarray
        Indices of the diversity-related variables
    con_model : tuple
        Model of the Con values
    obj_models : List[List[tuple]]
        Per-cluster objective models
    centers : np.ndarray
        Cluster centers, shape (nc, D)
    M : int
        Number of objectives
    N : int
        Population size
    muc : float
        Distribution index of the SBX crossover
    mum : float
        Distribution index of the polynomial mutation

    Returns
    -------
    sub_dv : np.ndarray
        Population after the diversity search, shape (N, D)
    """
    pop = pop_decs.copy()
    if dv_idx.shape[0] == 0 or pop.shape[0] < 2:
        return pop

    for _ in range(M):
        con = _predict(con_model, pop)
        parents_idx = tournament_selection(2, pop.shape[0], -con)
        parents = pop[parents_idx]

        offspring = parents.copy()
        offspring[:, dv_idx] = ga_generation(parents[:, dv_idx], muc, mum)

        merged = np.vstack([pop, offspring])
        merged_objs = _predict_objs(merged, obj_models, centers)
        pop = merged[_angular_selection(merged_objs, min(N, merged.shape[0]))]
    return pop


def _angular_selection(objs, n):
    """
    Keep the extreme solutions and then fill up by maximal angular diversity.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (n_total, M)
    n : int
        Number of solutions to keep

    Returns
    -------
    selected : np.ndarray
        Indices of the selected solutions, shape (n,)

    Notes
    -----
    Algorithm 5 phrases the fill-up step as adding "the solutions with the
    largest cosine value", while the surrounding text states the intent as
    "maximal angular diversity". The two readings are opposites; the text is
    followed here, so each step adds the solution whose largest cosine
    similarity to the already selected set is smallest.
    """
    n_total, M = objs.shape
    if n >= n_total:
        return np.arange(n_total)

    # Translate to the ideal point so the angles are measured in the positive
    # orthant, then normalize to unit length for the cosine similarity
    shifted = objs - np.min(objs, axis=0)
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = shifted / norms

    selected = list(np.unique(np.argmin(objs, axis=0)))[:n]
    remaining = [i for i in range(n_total) if i not in set(selected)]

    if selected:
        max_cos = np.max(unit[remaining] @ unit[selected].T, axis=1)
    else:
        max_cos = np.zeros(len(remaining))

    while len(selected) < n and remaining:
        pick = int(np.argmin(max_cos))
        chosen = remaining.pop(pick)
        max_cos = np.delete(max_cos, pick)
        selected.append(chosen)
        if remaining:
            max_cos = np.maximum(max_cos, unit[remaining] @ unit[chosen])

    return np.array(selected[:n], dtype=int)


def _reconstruct(sub_cv, sub_dv, cv_idx, dv_idx):
    """
    Concatenate the two optimized subsets into complete decision vectors.

    Parameters
    ----------
    sub_cv : np.ndarray
        Population returned by the convergence search, shape (N1, D)
    sub_dv : np.ndarray
        Population returned by the diversity search, shape (N2, D)
    cv_idx : np.ndarray
        Indices of the convergence-related variables
    dv_idx : np.ndarray
        Indices of the diversity-related variables

    Returns
    -------
    candidates : np.ndarray
        Reconstructed candidates, shape (min(N1, N2), D)
    """
    n = min(sub_cv.shape[0], sub_dv.shape[0])
    candidates = np.empty((n, sub_cv.shape[1]))
    candidates[:, cv_idx] = sub_cv[:n][:, cv_idx]
    candidates[:, dv_idx] = sub_dv[:n][:, dv_idx]
    return candidates


# =============================================================================
# Environmental selection
# =============================================================================

def _env_selection(objs, n):
    """
    Pick the working population from the archive by non-dominated sorting.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (n_total, M)
    n : int
        Target size

    Returns
    -------
    selected : np.ndarray
        Indices of the selected solutions
    """
    n_total = objs.shape[0]
    if n_total <= n:
        return np.arange(n_total)

    front_no, _ = nd_sort(objs, n)
    crowd_dis = crowding_distance(objs, front_no)
    order = np.lexsort((-crowd_dis, front_no))
    return order[:n]

"""
Ensemble-based Surrogate Model-Assisted Evolutionary Algorithm (EM-SAEA)

This module implements EM-SAEA for computationally expensive constrained/unconstrained
multi/many-objective optimization. It uses a two-stage approach:
- Stage 1 (objective-oriented, FE < 50% budget): RVMM-based search with two sub-populations
- Stage 2 (constraint-oriented, FE >= 50% budget): MOEA/D with ensemble constraint models

References
----------
    [1] Y. Li, X. Feng, and H. Yu. Enhancing landscape approximation with ensemble-based surrogate model for expensive constrained multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2025.

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
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict, gp_build, gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
import warnings

warnings.filterwarnings("ignore")


class EM_SAEA:
    """
    Ensemble-based Surrogate Model-Assisted Evolutionary Algorithm for expensive
    constrained/unconstrained multi/many-objective optimization.

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
                 wmax=20, lc_num=5, mu=5, alpha=2.0, kk=0.5,
                 save_data=True, save_path='./Data', name='EM-SAEA', disable_tqdm=True):
        """
        Initialize EM-SAEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size (number of reference vectors) per task (default: 100)
        wmax : int, optional
            Number of generations before updating surrogate models (default: 20)
        lc_num : int, optional
            Number of local constraint model clusters (default: 5)
        mu : int, optional
            Number of re-evaluated solutions per iteration in stage 2 (default: 5)
        alpha : float, optional
            Parameter controlling APD penalty rate (default: 2.0)
        kk : float, optional
            Uncertainty weighting factor for MSE augmentation (default: 0.5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EM-SAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.wmax = wmax
        self.lc_num = lc_num
        self.mu = mu
        self.alpha = alpha
        self.kk = kk
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EM-SAEA algorithm.

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

        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Reference/weight vectors
        W_list, V0_list = [], []
        for i in range(nt):
            w_i, actual_n = uniform_point(n_per_task[i], n_objs[i])
            W_list.append(w_i)
            V0_list.append(w_i.copy())
            n_per_task[i] = actual_n

        # [V0,NI] = UniformPoint(Problem.N,Problem.M): the size of the initial
        # Latin hypercube design equals the (adjusted) number of reference
        # vectors, so the very first CDP environmental selection receives
        # exactly Problem.N solutions.
        if self.n_initial is None:
            n_initial_per_task = n_per_task.copy()
        else:
            n_initial_per_task = par_list(self.n_initial, nt)

        # Cluster weight vectors for local constraint models
        ClW_list, lc_num_list = [], []
        for i in range(nt):
            clw_i, actual_lc = uniform_point(self.lc_num, n_objs[i])
            ClW_list.append(clw_i)
            lc_num_list.append(actual_lc)

        # Initialize by LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # History
        has_cons = any(c > 0 for c in n_cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        stages = [2] * nt  # Initial stage = 2 (constraint-oriented)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                dim = dims[i]
                n_con = n_cons[i]
                NI = n_initial_per_task[i]
                N = n_per_task[i]

                # Deduplicate training data
                _, unique_idx = np.unique(decs[i], axis=0, return_index=True)
                unique_idx = np.sort(unique_idx)
                train_decs = decs[i][unique_idx]
                train_objs = objs[i][unique_idx]
                train_cons = cons[i][unique_idx] if n_con > 0 else None

                # Build objective GP models
                obj_models = mo_gp_build(train_decs, train_objs, data_type)

                if stages[i] == 1:
                    new_decs = _stage1_objective(
                        train_decs, train_objs, obj_models,
                        V0_list[i], N, M, self.wmax, self.alpha, self.kk, data_type
                    )
                else:
                    new_decs = _stage2_constraint(
                        train_decs, train_objs, train_cons, obj_models,
                        W_list[i], V0_list[i], N, M, n_con, self.wmax, self.mu,
                        self.kk, self.alpha, lc_num_list[i], ClW_list[i],
                        nfes_per_task[i], max_nfes_per_task[i], data_type
                    )

                if new_decs is not None and len(new_decs) > 0:
                    new_decs = remove_duplicates(new_decs, decs[i])
                    # Respect the remaining budget
                    new_decs = new_decs[:max_nfes_per_task[i] - nfes_per_task[i]]

                if new_decs is not None and len(new_decs) > 0:
                    new_objs, new_cons = evaluation_single(problem, new_decs, i)
                    decs[i] = np.vstack([decs[i], new_decs])
                    objs[i] = np.vstack([objs[i], new_objs])
                    if n_con > 0:
                        cons[i] = np.vstack([cons[i], new_cons])

                    nfes_per_task[i] += new_decs.shape[0]
                    pbar.update(new_decs.shape[0])

                # Stage switch
                threshold = int(np.ceil(NI + 0.5 * (max_nfes_per_task[i] - NI)))
                stages[i] = 1 if nfes_per_task[i] < threshold else 2

        pbar.close()
        runtime = time.time() - start_time

        # Stage 1 infills a single solution (mu1 = 1) while stage 2 infills up to
        # mu, so the number of new evaluations per iteration is not constant and
        # the history must be built with a unit step.
        if has_cons:
            all_decs, all_objs, all_cons = build_staircase_history(decs, objs, k=1, db_cons=cons)
        else:
            all_decs, all_objs = build_staircase_history(decs, objs, k=1)
            all_cons = None
        kwargs = {}
        if has_cons:
            kwargs['all_cons'] = all_cons
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data, **kwargs
        )
        return results


# =============================================================================
# Stage 1: Objective-oriented optimization with RVMM
# =============================================================================

def _stage1_objective(train_decs, train_objs, obj_models, V0, N, M, wmax, alpha, kk, data_type):
    """
    Objective-oriented optimization stage using RVMM.

    Two sub-populations evolved on surrogates:
    1. Sub-pop with clustered reference vectors (V1) - convergence
    2. Sub-pop with full reference vectors (V) - diversity
    RVMM selects between convergence and diversity contributions.

    Returns
    -------
    new_decs : np.ndarray or None
        Selected decision variables for expensive re-evaluation
    """
    # Get non-dominated archive objectives
    front_no, _ = nd_sort(train_objs, train_objs.shape[0])
    A2Obj = train_objs[front_no == 1]
    if A2Obj.shape[0] == 0:
        A2Obj = train_objs.copy()

    # Scale for reference vector adaptation
    if A2Obj.shape[0] >= 2:
        scale = np.maximum(A2Obj.max(axis=0) - A2Obj.min(axis=0), 1e-6)
    else:
        scale = np.ones(M)

    # --- Sub-population 1: V1 (clustered active reference vectors) ---
    V_1 = V0 * scale
    A2Obj_temp = A2Obj - A2Obj.min(axis=0)
    angle_a2 = np.arccos(np.clip(1 - cdist(A2Obj_temp, V_1, 'cosine'), -1, 1))
    associate_a2 = np.argmin(angle_a2, axis=1)
    active = np.unique(associate_a2)
    Va = V_1[active]

    # Cluster active vectors into min(5, |Va|) groups
    n_cluster = min(5, Va.shape[0])
    if n_cluster > 1 and Va.shape[0] > n_cluster:
        labels = kmeans_clustering(V0[active], n_cluster)
    else:
        labels = np.arange(Va.shape[0]) % max(n_cluster, 1)

    # Select one random reference vector per cluster
    V1_list = []
    selected_active_ids = []
    for c in range(n_cluster):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) > 0:
            idx = cluster_idx[np.random.randint(len(cluster_idx))]
            V1_list.append(Va[idx])
            selected_active_ids.append(active[idx])
    V1 = np.array(V1_list) if V1_list else V_1[:min(5, V_1.shape[0])]

    # Pad to 5 if needed
    if V1.shape[0] < 5:
        not_selected = np.setdiff1d(np.arange(V_1.shape[0]),
                                     np.array(selected_active_ids) if selected_active_ids else [])
        n_add = min(5 - V1.shape[0], len(not_selected))
        if n_add > 0:
            add_idx = not_selected[np.random.permutation(len(not_selected))[:n_add]]
            V1 = np.vstack([V1, V_1[add_idx]])

    # Evolve sub-population 1 with V1 (mating pool always resampled)
    pop1_decs, pop1_objs, pop1_mse = _surrogate_evolve(
        train_decs, obj_models, V1, N, M, wmax, alpha, kk, data_type,
        scale.copy(), train_decs, accumulate=False, always_resample=True
    )

    # --- Sub-population 2: V (full reference vectors) ---
    V = V0 * scale
    pop2_decs, pop2_objs, pop2_mse = _surrogate_evolve(
        train_decs, obj_models, V, N, M, wmax, alpha, kk, data_type,
        scale.copy(), train_decs, accumulate=True, V0=V0
    )

    # --- RVMM Selection ---
    return _rvmm_select(pop2_decs, pop2_objs, pop1_decs, pop1_objs, A2Obj, scale, V0)


def _surrogate_evolve(init_decs, obj_models, V, N, M, wmax, alpha, kk, data_type,
                       scale, archive_decs, accumulate=False, V0=None,
                       always_resample=False):
    """
    Run GA evolution on surrogate models with K-RVEA environmental selection.

    Parameters
    ----------
    init_decs : np.ndarray
        Initial population decisions
    obj_models : list
        Trained GP models for objectives
    V : np.ndarray
        Reference vectors
    N : int
        Population size
    M : int
        Number of objectives
    wmax : int
        Number of generations
    alpha : float
        APD penalty parameter
    kk : float
        MSE weighting factor
    data_type : torch.dtype
        Data type for GP
    scale : np.ndarray
        Scale for reference vector adaptation
    archive_decs : np.ndarray
        Already-evaluated solutions (for duplicate checking)
    accumulate : bool
        If True, accumulate solutions from all generations
    V0 : np.ndarray or None
        Original reference vectors (needed if accumulate=True for V init)
    always_resample : bool
        If True the mating pool is always drawn with replacement
        (``randi(size(PopDec,1),1,Problem.N)``), as done for the first
        sub-population; otherwise the population itself is mated whenever it
        already holds at least N solutions.

    Returns
    -------
    final_decs, final_objs, final_mse : np.ndarray
        Evolved population (filtered: unique, not in archive)
    """
    pop_decs = init_decs.copy()
    if V0 is not None:
        V0 = V0.copy()
    all_gen_decs, all_gen_objs, all_gen_mse = [], [], []

    for w in range(1, wmax + 1):
        # Generate offspring
        if always_resample or pop_decs.shape[0] < N:
            pool = np.random.randint(0, pop_decs.shape[0], N)
            off_decs = ga_generation(pop_decs[pool], muc=20, mum=20)
        else:
            off_decs = ga_generation(pop_decs, muc=20, mum=20)

        # Remove near-duplicates against archive
        off_decs = remove_duplicates(off_decs, init_decs)
        if off_decs.shape[0] == 0:
            if accumulate:
                all_gen_decs.append(pop_decs.copy())
                pred_o, pred_m = mo_gp_predict(obj_models, pop_decs, data_type, mse=True)
                all_gen_objs.append(pred_o)
                all_gen_mse.append(pred_m)
            continue

        pop_decs = np.vstack([pop_decs, off_decs])

        # Predict objectives and MSE
        pop_objs, pop_mse = mo_gp_predict(obj_models, pop_decs, data_type, mse=True)
        pop_mse = _process_mse(pop_mse)

        pop_objs_orig = pop_objs.copy()
        pop_mse_orig = pop_mse.copy()

        # Augment objectives with uncertainty
        pop_objs_aug = pop_objs + kk * pop_mse

        # Initialize V on first generation for full-vector mode
        if accumulate and V0 is not None and w == 1:
            fno, _ = nd_sort(pop_objs_aug, pop_objs_aug.shape[0])
            nd_objs = pop_objs_aug[fno == 1]
            if nd_objs.shape[0] > 0:
                w_scale = nd_objs.max(axis=0) - nd_objs.min(axis=0)
                w_scale[w_scale == 0] = 1e-6
                V = V0 * w_scale
                scale = w_scale

                # Augment V0 with normalized non-dominated directions whenever
                # fewer than N reference vectors are active.
                ang = np.arccos(np.clip(1 - cdist(nd_objs, V, 'cosine'), -1, 1))
                active = np.unique(np.argmin(ang, axis=1))
                if active.shape[0] < N:
                    nd_norm = (nd_objs - nd_objs.min(axis=0)) / w_scale
                    n_add = min(N - active.shape[0], nd_norm.shape[0])
                    add_idx = np.random.permutation(nd_norm.shape[0])[:n_add]
                    V0 = np.vstack([V0, nd_norm[add_idx]])
                    V = V0 * w_scale

        # K-RVEA environmental selection
        cons_zero = np.zeros((pop_decs.shape[0], 1))
        index = _k_env_selection(pop_objs_aug, V, (w / wmax) ** alpha)
        if len(index) == 0:
            if accumulate:
                all_gen_decs.append(pop_decs.copy())
                all_gen_objs.append(pop_objs_aug)
                all_gen_mse.append(pop_mse)
            continue

        pop_decs = pop_decs[index]
        pop_objs_aug = pop_objs_aug[index]
        pop_mse = pop_mse[index]

        # Fallback: if all selected are already in archive, add ND offspring
        new_unique = remove_duplicates(pop_decs, archive_decs)
        if new_unique.shape[0] == 0 and off_decs.shape[0] > 0:
            n_total = pop_objs_orig.shape[0]
            off_start = n_total - off_decs.shape[0]
            off_obj = pop_objs_orig[off_start:]
            off_mse_v = pop_mse_orig[off_start:]
            fno_off, _ = nd_sort(off_obj, off_obj.shape[0])
            nd_idx = np.where(fno_off == 1)[0]
            if len(nd_idx) > 0:
                pop_decs = np.vstack([pop_decs, off_decs[nd_idx]])
                pop_objs_aug = np.vstack([pop_objs_aug, off_obj[nd_idx] + kk * off_mse_v[nd_idx]])
                pop_mse = np.vstack([pop_mse, off_mse_v[nd_idx]])

        # Adapt reference vectors
        adapt_interval = max(1, int(np.ceil(wmax * 0.1)))
        if w % adapt_interval == 0 and len(np.unique(pop_objs_aug, axis=0)) > 2:
            if accumulate and V0 is not None:
                V = V0 * np.maximum(pop_objs_aug.max(axis=0) - pop_objs_aug.min(axis=0), 1e-6)
            else:
                V = V / np.maximum(scale, 1e-6)
                scale = np.maximum(pop_objs_aug.max(axis=0) - pop_objs_aug.min(axis=0), 1e-6)
                V = V * scale

        if accumulate:
            all_gen_decs.append(pop_decs.copy())
            all_gen_objs.append(pop_objs_aug.copy())
            all_gen_mse.append(pop_mse.copy())

    # Finalize
    if accumulate and all_gen_decs:
        pop_decs = np.vstack(all_gen_decs)
        pop_objs_aug = np.vstack(all_gen_objs)
        pop_mse = np.vstack(all_gen_mse)

    # Remove MSE augmentation
    pop_objs = pop_objs_aug - kk * pop_mse

    # Deduplicate
    _, uidx = np.unique(pop_objs, axis=0, return_index=True)
    uidx = np.sort(uidx)
    pop_decs = pop_decs[uidx]
    pop_objs = pop_objs[uidx]
    pop_mse = pop_mse[uidx]

    # Remove already-evaluated
    keep = _not_in_archive(pop_decs, archive_decs)
    pop_decs = pop_decs[keep]
    pop_objs = pop_objs[keep]
    pop_mse = pop_mse[keep]

    # Keep only non-dominated
    if pop_decs.shape[0] > 0:
        fno, _ = nd_sort(pop_objs, pop_objs.shape[0])
        nd_mask = fno == 1
        pop_decs = pop_decs[nd_mask]
        pop_objs = pop_objs[nd_mask]
        pop_mse = pop_mse[nd_mask]

    return pop_decs, pop_objs, pop_mse


# =============================================================================
# Stage 2: Constraint-oriented optimization with ensemble models
# =============================================================================

def _stage2_constraint(train_decs, train_objs, train_cons, obj_models,
                       W, V0, N, M, n_con, wmax, mu, kk, alpha,
                       lc_num, ClW, nfes, max_nfes, data_type):
    """
    Constraint-oriented optimization stage with MOEA/D and ensemble constraint models.

    Returns
    -------
    new_decs : np.ndarray or None
        Selected decision variables for expensive re-evaluation
    """
    n_train = train_decs.shape[0]

    # --- CDP environmental selection to get initial population ---
    # Zmin = min objs of the feasible part of the archive (ArcFZ)
    if train_cons is not None and train_cons.shape[1] > 0:
        feasible_arc = np.all(train_cons <= 0, axis=1)
    else:
        feasible_arc = np.ones(n_train, dtype=bool)
    arc_fz = train_objs[feasible_arc].min(axis=0) if np.any(feasible_arc) else None

    if n_train <= N:
        pop_decs = train_decs.copy()
    else:
        pop_idx = _cdp_environmental_selection(train_objs, train_cons, N, W, arc_fz)
        pop_decs = train_decs[pop_idx]

    # --- Build constraint models ---
    con_models_global = []
    con_models_local = []  # [cluster][constraint]
    if n_con > 0 and train_cons is not None:
        # Cluster training data for local models
        z_min_obj = train_objs.min(axis=0)
        norm_objs = train_objs - z_min_obj
        cluster_size = int(np.ceil(n_train / lc_num))

        # Assign clusters: angle-based to ClW
        cluster_assign = np.full((n_train, lc_num), -1, dtype=int)
        for c in range(lc_num):
            angles_c = np.arccos(np.clip(1 - cdist(norm_objs, ClW[c:c + 1], 'cosine'), -1, 1)).flatten()
            sorted_idx = np.argsort(angles_c)
            cluster_assign[sorted_idx[:cluster_size], c] = c

        # Round-robin for remaining
        norm_objs_rr = norm_objs.copy()
        remaining = n_train
        while remaining > 0:
            for c in range(lc_num):
                angles_c = np.arccos(np.clip(1 - cdist(norm_objs_rr, ClW[c:c + 1], 'cosine'), -1, 1)).flatten()
                loc = np.argmin(angles_c)
                cluster_assign[loc, c] = c
                norm_objs_rr[loc] = np.inf
                remaining -= 1
                if remaining == 0:
                    break

        # Final cluster: min of two assignments
        cluster_final = np.min(cluster_assign, axis=1)

        # Train local constraint models
        for c in range(lc_num):
            c_models = []
            c_idx = np.where(cluster_final == c)[0]
            if len(c_idx) < 2:
                c_idx = np.arange(n_train)  # fallback to global
            for j in range(n_con):
                try:
                    model = gp_build(train_decs[c_idx], train_cons[c_idx, j:j + 1], data_type)
                except Exception:
                    model = gp_build(train_decs, train_cons[:, j:j + 1], data_type)
                c_models.append(model)
            con_models_local.append(c_models)

        # Train global constraint models
        for j in range(n_con):
            model = gp_build(train_decs, train_cons[:, j:j + 1], data_type)
            con_models_global.append(model)

    # --- Predict initial population's objectives and constraints ---
    pop_objs = mo_gp_predict(obj_models, pop_decs, data_type, mse=False)
    pop_N = pop_decs.shape[0]

    if n_con > 0 and con_models_global:
        pop_cons = np.zeros((pop_N, n_con))
        for j in range(n_con):
            pred, _ = gp_predict(con_models_global[j], pop_decs, data_type)
            pop_cons[:, j] = pred.flatten()
    else:
        pop_cons = np.zeros((pop_N, 0))

    pop_cv = np.sum(np.maximum(0, pop_cons), axis=1) if n_con > 0 else np.zeros(pop_N)
    pf = np.mean(pop_cv == 0)  # fraction of feasible solutions

    # --- Neighbor structure for MOEA/D ---
    T = max(2, int(np.ceil(pop_N / 10)))
    nr = max(1, int(np.ceil(pop_N / 100)))
    B_dist = cdist(W[:pop_N], W[:pop_N])
    B = np.argsort(B_dist, axis=1)[:, :T]

    # Angle thresholds for VCDP
    angle_ww = np.arccos(np.clip(1 - cdist(W[:pop_N], W[:pop_N], 'cosine'), -1, 1))
    np.fill_diagonal(angle_ww, np.inf)
    theta_min = np.min(angle_ww, axis=1)
    theta_angle = theta_min * 0.5

    Z = train_objs.min(axis=0)  # ideal point

    # --- MOEA/D evolution ---
    for _gen in range(wmax):
        for ii in range(pop_N):
            # Choose parents
            if np.random.rand() < 0.9:
                P = B[ii, np.random.permutation(B.shape[1])]
            else:
                P = np.random.permutation(pop_N)

            # Generate offspring from first 2 parents
            p1, p2 = pop_decs[P[0]], pop_decs[P[1 % len(P)]]
            off_dec1, _ = crossover(p1, p2, mu=20)
            off_dec = mutation(off_dec1, mu=20)
            off_dec = np.clip(off_dec, 0, 1)

            # Predict offspring objectives
            off_dec_2d = off_dec.reshape(1, -1)
            off_obj = mo_gp_predict(obj_models, off_dec_2d, data_type, mse=False)
            off_obj = off_obj.flatten()

            # Update ideal point
            Z = np.minimum(Z, off_obj)

            # Predict offspring constraints with ensemble
            if n_con > 0 and con_models_global:
                norm_off = off_obj - Z
                angles_to_clw = np.arccos(np.clip(
                    1 - cdist(norm_off.reshape(1, -1), ClW, 'cosine'), -1, 1)).flatten()
                off_cluster = np.argmin(angles_to_clw)

                off_con_gc = np.zeros(n_con)
                off_mse_gc = np.zeros(n_con)
                off_con_lc = np.zeros(n_con)
                off_mse_lc = np.zeros(n_con)

                for j in range(n_con):
                    pred_gc, std_gc = gp_predict(con_models_global[j], off_dec_2d, data_type)
                    off_con_gc[j] = pred_gc.flatten()[0]
                    off_mse_gc[j] = (std_gc.flatten()[0]) ** 2

                    if off_cluster < len(con_models_local):
                        pred_lc, std_lc = gp_predict(
                            con_models_local[off_cluster][j], off_dec_2d, data_type)
                        off_con_lc[j] = pred_lc.flatten()[0]
                        off_mse_lc[j] = (std_lc.flatten()[0]) ** 2
                    else:
                        off_con_lc[j] = off_con_gc[j]
                        off_mse_lc[j] = off_mse_gc[j]

                # Choose model with lower MSE
                off_con = off_con_gc if np.mean(off_mse_gc) < np.mean(off_mse_lc) else off_con_lc
                off_cv = np.sum(np.maximum(0, off_con))
            else:
                off_con = np.zeros(0)
                off_cv = 0.0

            # PBI values for neighbors
            P_valid = P[P < pop_N]
            if len(P_valid) == 0:
                continue

            W_p = W[P_valid]
            obj_p = pop_objs[P_valid]
            cv_p = pop_cv[P_valid]

            # PBI decomposition: g = d1 + 5*d2
            norm_w = np.sqrt(np.sum(W_p ** 2, axis=1))
            diff_p = obj_p - Z
            norm_p = np.sqrt(np.sum(diff_p ** 2, axis=1))
            cos_p = np.sum(diff_p * W_p, axis=1) / (norm_w * np.maximum(norm_p, 1e-10))
            cos_p = np.clip(cos_p, -1, 1)
            g_old = norm_p * cos_p + 5 * norm_p * np.sqrt(np.maximum(1 - cos_p ** 2, 0))

            diff_o = off_obj - Z
            norm_o = np.sqrt(np.sum(diff_o ** 2))
            cos_o = np.sum(diff_o * W_p, axis=1) / (norm_w * max(norm_o, 1e-10))
            cos_o = np.clip(cos_o, -1, 1)
            g_new = norm_o * cos_o + 5 * norm_o * np.sqrt(np.maximum(1 - cos_o ** 2, 0))

            # VCDP replacement
            if off_cv + np.sum(cv_p) == 0:
                # All feasible: replace by PBI
                replace_mask = g_old >= g_new
            else:
                # CDP: replace if (better PBI and same CV) or (lower CV)
                replace_mask = ((g_old >= g_new) & (cv_p == off_cv)) | (cv_p > off_cv)

            replace_idx = np.where(replace_mask)[0][:nr]
            for r in replace_idx:
                idx = P_valid[r]
                pop_decs[idx] = off_dec
                pop_objs[idx] = off_obj
                if n_con > 0:
                    pop_cons[idx] = off_con
                pop_cv[idx] = off_cv

    # --- Select solutions for re-evaluation ---
    # Try ArchiveUpdate: select feasible non-dominated
    new_decs = _archive_update(pop_decs, pop_objs, pop_cons if n_con > 0 else None, mu)

    if new_decs is None or len(new_decs) == 0:
        # Fallback: KrigingSelect (APD-based with constraints)
        theta_ks = (nfes / max(max_nfes, 1)) ** 2
        new_decs = _kriging_select_con(pop_decs, pop_objs, W[:pop_N], mu, theta_ks,
                                        pop_cons if n_con > 0 else None)

    return new_decs


# =============================================================================
# Helper Functions
# =============================================================================

def _cdp_environmental_selection(pop_obj, pop_con, N, Z, Zmin):
    """
    NSGA-III environmental selection with the constrained-dominance principle.

    Port of ``CDPEnvironmentalSelection``: constrained non-dominated sorting
    followed by the NSGA-III reference-point niching on the last front.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (n, M)
    pop_con : np.ndarray or None
        Constraint values, shape (n, C)
    N : int
        Number of solutions to keep
    Z : np.ndarray
        Reference points, shape (NZ, M)
    Zmin : np.ndarray or None
        Ideal point of the feasible archive; ``ones(1,M)`` when empty

    Returns
    -------
    index : np.ndarray
        Ascending indices of the selected solutions
    """
    n = pop_obj.shape[0]
    if Zmin is None:
        Zmin = np.ones(Z.shape[1])

    if pop_con is not None and pop_con.shape[1] > 0:
        front_no, max_fno = nd_sort(pop_obj, pop_con, N)
    else:
        front_no, max_fno = nd_sort(pop_obj, N)

    next_mask = front_no < max_fno
    last = np.where(front_no == max_fno)[0]
    K = N - int(np.sum(next_mask))
    if K > 0 and len(last) > 0:
        choose = _nsga3_last_selection(pop_obj[next_mask], pop_obj[last], K, Z, Zmin)
        next_mask[last[choose]] = True

    index = np.where(next_mask)[0]
    if index.shape[0] > N:
        index = index[:N]
    elif index.shape[0] < N:
        # Numerical fallback (never triggered by the reference, which always
        # has enough solutions in the last front)
        rest = np.setdiff1d(np.arange(n), index)
        index = np.sort(np.concatenate([index, rest[:N - index.shape[0]]]))
    return index


def _nsga3_last_selection(objs_before, objs_last, K, Z, Zmin):
    """
    NSGA-III niching on the last front (``LastSelection``).

    Parameters
    ----------
    objs_before : np.ndarray
        Objectives already selected, shape (N1, M)
    objs_last : np.ndarray
        Objectives of the last front, shape (N2, M)
    K : int
        Number of solutions to pick from the last front
    Z : np.ndarray
        Reference points, shape (NZ, M)
    Zmin : np.ndarray
        Ideal point, shape (M,)

    Returns
    -------
    choose : np.ndarray
        Boolean mask over the last front, shape (N2,)
    """
    pop_obj = np.vstack([objs_before, objs_last]) - Zmin
    n, M = pop_obj.shape
    N1 = objs_before.shape[0]
    N2 = objs_last.shape[0]
    NZ = Z.shape[0]

    # --- Normalization ---
    extreme = np.zeros(M, dtype=int)
    w = np.eye(M) + 1e-6
    for i in range(M):
        extreme[i] = np.argmin(np.max(pop_obj / w[i], axis=1))
    try:
        hyperplane = np.linalg.solve(pop_obj[extreme, :], np.ones(M))
        with np.errstate(divide='ignore'):
            a = 1.0 / hyperplane
        if np.any(~np.isfinite(a)):
            a = np.max(pop_obj, axis=0)
    except np.linalg.LinAlgError:
        a = np.max(pop_obj, axis=0)
    a = np.where(np.abs(a) < 1e-12, 1e-12, a)
    pop_obj = pop_obj / a

    # --- Association ---
    cosine = 1 - cdist(pop_obj, Z, metric='cosine')
    cosine = np.clip(np.nan_to_num(cosine), -1, 1)
    distance = np.linalg.norm(pop_obj, axis=1, keepdims=True) * np.sqrt(np.maximum(1 - cosine ** 2, 0))
    pi = np.argmin(distance, axis=1)
    d = np.min(distance, axis=1)

    rho = np.bincount(pi[:N1], minlength=NZ)

    choose = np.zeros(N2, dtype=bool)
    zchoose = np.ones(NZ, dtype=bool)
    while np.sum(choose) < K:
        temp = np.where(zchoose)[0]
        if len(temp) == 0:
            break
        jmin = temp[rho[temp] == np.min(rho[temp])]
        j = jmin[np.random.randint(len(jmin))]
        I = np.where((~choose) & (pi[N1:] == j))[0]
        if len(I) > 0:
            s = int(np.argmin(d[N1 + I])) if rho[j] == 0 else np.random.randint(len(I))
            choose[I[s]] = True
            rho[j] += 1
        else:
            zchoose[j] = False
    return choose


def _k_env_selection(pop_obj, V, theta):
    """
    K-RVEA environmental selection using Angle-Penalized Distance (APD).

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    V : np.ndarray
        Reference vectors, shape (NV, M)
    theta : float
        APD penalty parameter

    Returns
    -------
    index : np.ndarray
        Selected solution indices
    """
    N, M = pop_obj.shape
    NV = V.shape[0]

    # Translate
    pop_obj = pop_obj - pop_obj.min(axis=0)

    # Gamma: smallest angle between each reference vector and others
    cosine = 1 - cdist(V, V, 'cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    gamma = np.maximum(gamma, 1e-6)

    # Associate each solution to nearest reference vector
    angle = np.arccos(np.clip(1 - cdist(pop_obj, V, 'cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)

    # Select one solution per reference vector by minimum APD
    Next = np.full(NV, -1, dtype=int)
    for i in np.unique(associate):
        current = np.where(associate == i)[0]
        APD = (1 + M * theta * angle[current, i] / gamma[i]) * \
              np.sqrt(np.sum(pop_obj[current] ** 2, axis=1))
        best = np.argmin(APD)
        Next[i] = current[best]

    return Next[Next != -1]


def _process_mse(mse):
    """Transform MSE: sqrt for values <= 1, keep raw for values > 1."""
    mse = np.maximum(mse, 0)
    s = np.sqrt(mse)
    return np.where(mse <= 1, s, mse)


def _not_in_archive(decs, archive, tol=1e-6):
    """Return boolean mask of solutions NOT in archive."""
    if archive.shape[0] == 0:
        return np.ones(decs.shape[0], dtype=bool)
    dists = cdist(decs, archive)
    min_dists = np.min(dists, axis=1)
    return min_dists > tol


def _archive_update(pop_dec, pop_obj, pop_con, n):
    """
    Select feasible non-dominated solutions with niche-based diversity.

    Parameters
    ----------
    pop_dec : np.ndarray
        Decision variables
    pop_obj : np.ndarray
        Objective values
    pop_con : np.ndarray or None
        Constraint values
    n : int
        Number of solutions to select

    Returns
    -------
    selected_decs : np.ndarray or None
        Selected decision variables
    """
    # Select feasible solutions
    if pop_con is not None and pop_con.shape[1] > 0:
        feasible = np.all(pop_con <= 0, axis=1)
    else:
        feasible = np.ones(pop_dec.shape[0], dtype=bool)

    f_dec = pop_dec[feasible]
    f_obj = pop_obj[feasible]

    if f_dec.shape[0] == 0:
        return None

    # Select non-dominated
    front_no, _ = nd_sort(f_obj, f_obj.shape[0])
    nd_mask = front_no == 1
    nd_dec = f_dec[nd_mask]
    nd_obj = f_obj[nd_mask]

    if nd_dec.shape[0] == 0:
        return None

    # Random permutation
    perm = np.random.permutation(nd_dec.shape[0])
    nd_dec = nd_dec[perm]
    nd_obj = nd_obj[perm]

    if nd_dec.shape[0] <= n:
        return nd_dec

    # Niche-based truncation
    fmax = nd_obj.max(axis=0)
    fmin = nd_obj.min(axis=0)
    rng = fmax - fmin
    rng[rng == 0] = 1.0
    norm_obj = (nd_obj - fmin) / rng

    d = cdist(norm_obj, norm_obj)
    np.fill_diagonal(d, np.inf)
    sd = np.sort(d, axis=1)
    r = np.median(sd[:, min(norm_obj.shape[1] - 1, sd.shape[1] - 1)])
    r = max(r, 1e-10)
    R = np.minimum(d / r, 1.0)

    # Delete worst one by one
    remaining = list(range(nd_dec.shape[0]))
    while len(remaining) > n:
        niche_vals = 1 - np.prod(R[np.ix_(remaining, remaining)], axis=1)
        worst = np.argmax(niche_vals)
        remaining.pop(worst)

    return nd_dec[remaining]


def _kriging_select_con(pop_dec, pop_obj, V, mu, theta, pop_con=None):
    """
    Constraint-aware Kriging selection using APD.

    Parameters
    ----------
    pop_dec : np.ndarray
        Population decisions
    pop_obj : np.ndarray
        Predicted objectives
    V : np.ndarray
        Weight/reference vectors
    mu : int
        Number of solutions to select
    theta : float
        APD penalty parameter
    pop_con : np.ndarray or None
        Predicted constraints

    Returns
    -------
    selected_decs : np.ndarray
        Selected decision variables
    """
    N = pop_obj.shape[0]
    M = pop_obj.shape[1]
    CV = np.sum(np.maximum(0, pop_con), axis=1) if pop_con is not None and pop_con.shape[1] > 0 else np.zeros(N)

    # Active reference vectors
    pop_obj_t = pop_obj - pop_obj.min(axis=0)
    angle_all = np.arccos(np.clip(1 - cdist(pop_obj_t, V, 'cosine'), -1, 1))
    associate = np.argmin(angle_all, axis=1)
    active = np.unique(associate)
    Va = V[active]

    n_cluster = min(mu, len(active))
    if n_cluster == 0:
        idx = np.random.choice(N, size=min(mu, N), replace=False)
        return pop_dec[idx]

    if n_cluster < len(active):
        labels = kmeans_clustering(Va, n_cluster)
    else:
        labels = np.arange(len(active))

    # Gamma and APD
    if Va.shape[0] > 1:
        cosine = 1 - cdist(Va, Va, 'cosine')
        np.fill_diagonal(cosine, 0)
        gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
        gamma = np.maximum(gamma, 1e-6)
    else:
        gamma = np.array([1.0])

    angle_va = np.arccos(np.clip(1 - cdist(pop_obj_t, Va, 'cosine'), -1, 1))
    assoc_va = np.argmin(angle_va, axis=1)

    APD = np.ones(N)
    for j in np.unique(assoc_va):
        cur = np.where(assoc_va == j)[0]
        if len(cur) > 0:
            APD[cur] = (1 + M * theta * angle_va[cur, j] / gamma[j]) * \
                       np.sqrt(np.sum(pop_obj_t[cur] ** 2, axis=1))

    cindex = labels[assoc_va]

    selected = []
    for c in np.unique(cindex):
        cur = np.where(cindex == c)[0]
        if len(cur) == 0:
            continue

        # Prefer feasible solutions
        feas = cur[CV[cur] == 0]
        if len(feas) > 0:
            t = np.unique(assoc_va[feas])
            best_per_vec = []
            for j in t:
                cur_j = np.where((assoc_va == j) & (CV == 0))[0]
                if len(cur_j) > 0:
                    best_per_vec.append(cur_j[np.argmin(APD[cur_j])])
            if best_per_vec:
                best_arr = np.array(best_per_vec)
                selected.append(best_arr[np.argmin(APD[best_arr])])
        else:
            # Select by minimum CV
            selected.append(cur[np.argmin(CV[cur])])

    if len(selected) == 0:
        idx = np.random.choice(N, size=min(mu, N), replace=False)
        return pop_dec[idx]

    return pop_dec[selected]


def _rvmm_select(pop2_dec, pop2_obj, pop1_dec, pop1_obj, A2Obj, scale, V0):
    """
    RVMM-based infill selection combining convergence and diversity.

    Parameters
    ----------
    pop2_dec, pop2_obj : np.ndarray
        Sub-population 2 (full vectors) - for diversity
    pop1_dec, pop1_obj : np.ndarray
        Sub-population 1 (clustered vectors) - for convergence
    A2Obj : np.ndarray
        Archive (real-evaluated) non-dominated objectives
    scale : np.ndarray
        Objective scale
    V0 : np.ndarray
        Original reference vectors

    Returns
    -------
    new_decs : np.ndarray or None
        Selected solution(s) for evaluation
    """
    if pop1_dec.shape[0] == 0 and pop2_dec.shape[0] == 0:
        return None

    # --- Convergence metric (cd) from sub-pop 1 ---
    cbest_dec = None
    if pop1_obj.shape[0] > 0:
        zmin = np.minimum(A2Obj.min(axis=0), pop1_obj.min(axis=0))
        cd = np.zeros(pop1_obj.shape[0])
        for idx in range(pop1_obj.shape[0]):
            sol = pop1_obj[idx]
            dist_to_archive = cdist((sol - zmin).reshape(1, -1),
                                     A2Obj - zmin, 'euclidean').flatten()
            nearest_id = np.argmin(dist_to_archive)
            # Check if solution dominates nearest archive member
            dominates = np.any(sol < A2Obj[nearest_id]) and not np.any(sol > A2Obj[nearest_id])
            cd[idx] = dist_to_archive[nearest_id] if dominates else 0.0

        # flagMu == 1: cbest = a uniformly random index among the maxima of cd
        # (cid = find(cd == cd(cbest)); cbest = cid(randperm(end,1))).  The tie
        # is resolved randomly even when every cd is zero.
        ties = np.where(cd == np.max(cd))[0]
        cbest = int(ties[np.random.randint(len(ties))])
        cbest_dec = pop1_dec[cbest:cbest + 1]
        cbest_obj = pop1_obj[cbest]

    # --- Diversity metric (dd) from sub-pop 2 ---
    dbest_dec = None
    if pop2_obj.shape[0] > 0:
        zmin_all = A2Obj.min(axis=0)
        if pop1_obj.shape[0] > 0:
            zmin_all = np.minimum(zmin_all, pop1_obj.min(axis=0))
        if pop2_obj.shape[0] > 0:
            zmin_all = np.minimum(zmin_all, pop2_obj.min(axis=0))

        scale_safe = np.maximum(scale, 1e-6)
        pop2_norm = np.maximum(pop2_obj - zmin_all, 0) / scale_safe
        A2_norm1 = np.maximum(A2Obj - zmin_all, 0) / scale_safe
        A2_norm2 = np.maximum(A2Obj - A2Obj.min(axis=0), 0) / scale_safe

        # Choose reference based on IGD
        igd1 = _rvmm_igd(pop2_norm, A2_norm1)
        igd2 = _rvmm_igd(pop2_norm, A2_norm2)
        A2_ref = A2_norm1 if igd1 < igd2 else A2_norm2

        # Compute angle-based diversity
        if A2_ref.shape[0] > 0 and pop2_norm.shape[0] > 0:
            angle = np.arccos(np.clip(1 - cdist(pop2_norm, A2_ref, 'cosine'), -1, 1))
            min_angle = np.min(angle, axis=1)
            dd = min_angle / max(np.max(min_angle), 1e-10)
            dbest = np.argmax(dd)
            dbest_dec = pop2_dec[dbest:dbest + 1]

    # --- Final selection: convergence or diversity ---
    if cbest_dec is not None:
        # Check if convergence solution is on Pareto front 1 w.r.t. archive
        combined = np.vstack([cbest_obj.reshape(1, -1), A2Obj])
        front_no, _ = nd_sort(combined, combined.shape[0])
        if front_no[0] == 1:
            return cbest_dec
    if dbest_dec is not None:
        return dbest_dec
    if cbest_dec is not None:
        return cbest_dec
    return None


def _rvmm_igd(pop_obj, optimum):
    """Compute Inverted Generational Distance."""
    if pop_obj.shape[1] != optimum.shape[1] or pop_obj.shape[0] == 0 or optimum.shape[0] == 0:
        return np.inf
    return np.mean(np.min(cdist(optimum, pop_obj), axis=1))

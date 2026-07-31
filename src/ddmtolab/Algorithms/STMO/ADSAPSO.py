"""
Adaptive Dropout based Surrogate-Assisted Particle Swarm Optimization (ADSAPSO)

This module implements ADSAPSO for computationally expensive multi/many-objective optimization.
It uses adaptive dropout to select important decision variables, builds RBF surrogate models
in the reduced space, and applies PSO on surrogates to find promising solutions.

References
----------
    [1] J. Lin, C. He, and R. Cheng. Adaptive dropout for high-dimensional expensive multiobjective optimization. Complex & Intelligent Systems, 2022, 8(1): 271-285.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class ADSAPSO:
    """
    Adaptive Dropout based Surrogate-Assisted Particle Swarm Optimization
    for expensive multi/many-objective optimization.

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

    def __init__(self, problem, n_initial=100, max_nfes=None, n=100, k=5, beta=0.5,
                 n_a=200, n_s=50, save_data=True, save_path='./Data', name='ADSAPSO',
                 disable_tqdm=True):
        """
        Initialize ADSAPSO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size for PSO per task (default: 100)
        k : int, optional
            Number of re-evaluated solutions per generation (default: 5)
        beta : float, optional
            Percentage of dropout (fraction of dimensions to keep) (default: 0.5)
        n_a : int, optional
            Number of solutions for building surrogate models (default: 200)
        n_s : int, optional
            Number of well/poorly performing solutions for dimension analysis (default: 50)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'ADSAPSO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.k = k
        self.beta = beta
        self.n_a = n_a
        self.n_s = n_s
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the ADSAPSO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                D = dims[i]

                # Main operator: dimension dropout + RBF + PSO
                offspring_decs = _operator(
                    decs[i], objs[i], self.k, self.beta, self.n_a, self.n_s,
                    n_per_task[i], M, D, nfes_per_task[i], max_nfes_per_task[i]
                )

                # Drop points already evaluated; PlatEMO does not deduplicate, so keep
                # the raw candidates when the filter would empty the batch (the PSO
                # inner loop is deterministic given the archive and can repeat points)
                filtered = remove_duplicates(offspring_decs, decs[i])
                if filtered.shape[0] > 0:
                    offspring_decs = filtered

                if offspring_decs.shape[0] > 0:
                    # Limit to remaining budget
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if offspring_decs.shape[0] > remaining:
                        offspring_decs = offspring_decs[:remaining]

                    # Evaluate with expensive objective function
                    offspring_objs, _ = evaluation_single(problem, offspring_decs, i)

                    # Update archive
                    decs[i] = np.vstack([decs[i], offspring_decs])
                    objs[i] = np.vstack([objs[i], offspring_objs])

                    nfes_per_task[i] += offspring_decs.shape[0]
                    pbar.update(offspring_decs.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.k)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results


# =============================================================================
# Main Operator
# =============================================================================

def _operator(arc_decs, arc_objs, k, beta, n_a, n_s, n, M, D, nfes, max_nfes):
    """
    Main operator: environmental selection, dimension dropout, RBF construction, PSO.

    Parameters
    ----------
    arc_decs : np.ndarray
        Archive decision variables, shape (N_arc, D)
    arc_objs : np.ndarray
        Archive objective values, shape (N_arc, M)
    k : int
        Number of re-evaluated solutions
    beta : float
        Dropout percentage
    n_a : int
        Archive size for surrogate building
    n_s : int
        Number of well/poorly performing solutions
    n : int
        Population size for PSO
    M : int
        Number of objectives
    D : int
        Number of decision variables
    nfes : int
        Current function evaluation count
    max_nfes : int
        Maximum function evaluations

    Returns
    -------
    candidate_decs : np.ndarray
        New candidate solutions, shape (k, D)
    """
    N_arc = arc_decs.shape[0]

    # Step 1: Environmental Selection to get N_a solutions
    n_a_actual = min(n_a, N_arc)
    sel_decs, sel_objs, front_no, crowd_dis = _environmental_selection(
        arc_decs, arc_objs, n_a_actual
    )
    N_sel = sel_decs.shape[0]

    # Step 2: Sort by [front_no ascending, crowd_dis descending]
    sort_idx = np.lexsort((-crowd_dis, front_no))

    # Step 3: Select top k as candidate solutions
    k_actual = min(k, N_sel)
    candidate_decs = sel_decs[sort_idx[:k_actual]].copy()

    # Step 4: Identify well and poorly performing solutions
    n_s_actual = min(n_s, N_sel)
    index_well = sort_idx[:n_s_actual]
    index_poor = sort_idx[-n_s_actual:]

    # Step 5: Compute dimension importance via mean difference
    model_dif = np.mean(sel_decs[index_well], axis=0) - np.mean(sel_decs[index_poor], axis=0)
    abs_dif = np.abs(model_dif)

    # Select top ceil(beta*D) dimensions
    n_selected_dims = max(1, int(np.ceil(beta * D)))
    sorted_dif = np.sort(abs_dif)[::-1]
    threshold = sorted_dif[min(n_selected_dims - 1, len(sorted_dif) - 1)]
    # Match MATLAB: find(abs(Model_Dif) >= threshold) - may select more due to ties
    index_dif = np.where(abs_dif >= threshold)[0]

    # Step 6: Build RBF models for selected dimensions
    decs_surrogate = sel_decs[:, index_dif]

    rbf_models = []
    for j in range(M):
        model = _rbf_create(decs_surrogate, sel_objs[:, j])
        rbf_models.append(model)

    # Step 7: Environmental selection for PSO population
    n_actual = min(n, N_sel)
    pop_decs_full, pop_objs, _, _ = _environmental_selection(sel_decs, sel_objs, n_actual)
    pop_decs_reduced = pop_decs_full[:, index_dif]

    # Step 8: PSO on surrogates in reduced dimension space
    offspring_decs_reduced, offspring_objs = _reproduction(
        pop_decs_reduced, pop_objs, rbf_models, M, nfes, max_nfes
    )

    # Step 9: Environmental selection on offspring to select top k
    k_out = min(k_actual, offspring_decs_reduced.shape[0])
    if k_out < offspring_decs_reduced.shape[0]:
        sel_off_decs, _, _, _ = _environmental_selection(
            offspring_decs_reduced, offspring_objs, k_out
        )
    else:
        sel_off_decs = offspring_decs_reduced

    # Step 10: Replace selected dimensions in candidate solutions
    n_replace = min(k_actual, sel_off_decs.shape[0])
    candidate_decs[np.ix_(np.arange(n_replace), index_dif)] = sel_off_decs[:n_replace]

    # Clip to [0, 1]
    candidate_decs = np.clip(candidate_decs, 0, 1)

    return candidate_decs


# =============================================================================
# Environmental Selection (NSGA-II style)
# =============================================================================

def _environmental_selection(pop_decs, pop_objs, n):
    """
    NSGA-II style environmental selection using non-dominated sorting
    and crowding distance.

    Parameters
    ----------
    pop_decs : np.ndarray
        Decision variables, shape (N, D)
    pop_objs : np.ndarray
        Objective values, shape (N, M)
    n : int
        Number of solutions to select

    Returns
    -------
    sel_decs : np.ndarray
        Selected decision variables
    sel_objs : np.ndarray
        Selected objective values
    front_no : np.ndarray
        Front numbers of selected solutions
    crowd_dis : np.ndarray
        Crowding distances of selected solutions
    """
    N = pop_decs.shape[0]
    n = min(n, N)

    # Non-dominated sorting
    front_no, max_fno = nd_sort(pop_objs, n)

    # Crowding distance
    crowd_dis = crowding_distance(pop_objs, front_no)

    # Select solutions from fronts below max_fno
    next_mask = front_no < max_fno

    # Fill remaining from last front based on crowding distance
    last = np.where(front_no == max_fno)[0]
    if last.size > 0 and np.sum(next_mask) < n:
        rank = np.argsort(crowd_dis[last])[::-1]
        n_remaining = n - np.sum(next_mask)
        next_mask[last[rank[:n_remaining]]] = True

    selected = np.where(next_mask)[0]
    return (pop_decs[selected], pop_objs[selected],
            front_no[selected], crowd_dis[selected])


# =============================================================================
# RBF Surrogate Model
# =============================================================================

def _rbf_create(X, y, kernel='gaussian'):
    """
    Create RBF surrogate model with augmented linear system.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Training inputs
    y : np.ndarray, shape (N,) or (N, 1)
        Training targets
    kernel : str
        Kernel type: 'gaussian' or 'cubic'

    Returns
    -------
    para : dict
        RBF model parameters
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    N, D = X.shape

    # Normalize X to [-1, 1]
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    x_range = xmax - xmin
    x_range[x_range < 1e-10] = 1.0
    X_norm = 2.0 / x_range * (X - xmin) - 1.0

    # Normalize y to [-1, 1]
    ymin = y.min(axis=0)
    ymax = y.max(axis=0)
    y_range = ymax - ymin
    y_range[y_range < 1e-10] = 1.0
    y_norm = 2.0 / y_range * (y - ymin) - 1.0

    # Compute pairwise distance matrix
    r = cdist(X_norm, X_norm, 'euclidean')

    # Compute kernel matrix
    if kernel == 'gaussian':
        # MATLAB: radbas(sqrt(-log(0.5)) * r) = exp(-log(2) * r^2)
        Phi = np.exp(-np.log(2) * r ** 2)
    elif kernel == 'cubic':
        Phi = r ** 3
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Augmented linear system: [Phi, P; P', 0] * [alpha; beta] = [y; 0]
    P = np.hstack([np.ones((N, 1)), X_norm])
    A = np.block([
        [Phi, P],
        [P.T, np.zeros((D + 1, D + 1))]
    ])
    b = np.vstack([y_norm, np.zeros((D + 1, y_norm.shape[1]))])

    # Add small regularization for numerical stability
    A += np.eye(A.shape[0]) * 1e-8

    # Solve the linear system
    theta = np.linalg.lstsq(A, b, rcond=None)[0]

    return {
        'alpha': theta[:N],
        'beta': theta[N:],
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'nodes': X_norm,
        'kernel': kernel
    }


def _rbf_predict_single(X, para):
    """
    Predict using a single RBF model.

    Parameters
    ----------
    X : np.ndarray, shape (nx, D)
        Test inputs
    para : dict
        RBF model parameters

    Returns
    -------
    y : np.ndarray, shape (nx, 1)
        Predicted values
    """
    nx = X.shape[0]
    nodes = para['nodes']

    # Normalize X to [-1, 1]
    x_range = para['xmax'] - para['xmin']
    x_range[x_range < 1e-10] = 1.0
    X_norm = 2.0 / x_range * (X - para['xmin']) - 1.0

    # Compute distances to training nodes
    r = cdist(X_norm, nodes, 'euclidean')

    # Compute kernel values
    if para['kernel'] == 'gaussian':
        Phi = np.exp(-np.log(2) * r ** 2)
    elif para['kernel'] == 'cubic':
        Phi = r ** 3

    # Predict in normalized space
    P = np.hstack([np.ones((nx, 1)), X_norm])
    y_norm = Phi @ para['alpha'] + P @ para['beta']

    # Denormalize
    y_range = para['ymax'] - para['ymin']
    y = y_range / 2.0 * (y_norm + 1.0) + para['ymin']

    return y


def _rbf_predict(X, rbf_models, M):
    """
    Predict objectives using multiple RBF models.

    Parameters
    ----------
    X : np.ndarray, shape (N, D_reduced)
        Test inputs in reduced dimension space
    rbf_models : list
        List of RBF model dicts, one per objective
    M : int
        Number of objectives

    Returns
    -------
    pred_objs : np.ndarray, shape (N, M)
        Predicted objective values
    """
    N = X.shape[0]
    pred_objs = np.zeros((N, M))
    for j in range(M):
        pred_objs[:, j:j + 1] = _rbf_predict_single(X, rbf_models[j])
    return pred_objs


# =============================================================================
# PSO-based Reproduction on Surrogates
# =============================================================================

def _reproduction(pop_decs, pop_objs, rbf_models, M, nfes, max_nfes):
    """
    PSO-based reproduction on RBF surrogates with BFE-guided Gbest management.

    Parameters
    ----------
    pop_decs : np.ndarray, shape (N, D_reduced)
        Population in reduced dimension space
    pop_objs : np.ndarray, shape (N, M)
        Population objectives
    rbf_models : list
        RBF models for each objective
    M : int
        Number of objectives
    nfes : int
        Current function evaluation count
    max_nfes : int
        Maximum function evaluations

    Returns
    -------
    off_decs : np.ndarray, shape (N, D_reduced)
        Offspring decision variables in reduced space
    off_objs : np.ndarray, shape (N, M)
        Offspring surrogate-predicted objectives
    """
    N, D = pop_decs.shape

    # Initialize Pbest as current particles
    pbest_decs = pop_decs.copy()
    pbest_objs = pop_objs.copy()

    # Initialize Archive: non-dominated solutions sorted by BFE
    front_no_all, _ = nd_sort(pop_objs, N)
    nd_mask = front_no_all == 1
    archive_decs = pop_decs[nd_mask].copy()
    archive_objs = pop_objs[nd_mask].copy()

    if archive_decs.shape[0] > 0:
        bfe = _compute_bfe(archive_objs)
        rank = np.argsort(bfe)[::-1]
        archive_decs = archive_decs[rank]
        archive_objs = archive_objs[rank]
        if archive_decs.shape[0] > N:
            archive_decs = archive_decs[:N]
            archive_objs = archive_objs[:N]

    # Initialize Gbest: random from top 10% of archive
    arch_size = archive_decs.shape[0]
    if arch_size > 0:
        top_count = max(1, int(np.ceil(arch_size / 10)))
        gbest_idx = np.random.randint(0, top_count, size=N)
        gbest_decs = archive_decs[gbest_idx].copy()
        gbest_objs = archive_objs[gbest_idx].copy()
    else:
        gbest_decs = pop_decs.copy()
        gbest_objs = pop_objs.copy()

    # Initialize velocity and particles
    velocity = np.zeros((N, D))
    particle_decs = pop_decs.copy()

    off_decs = particle_decs.copy()
    off_objs = pop_objs.copy()

    # PSO loop (100 generations)
    for gen in range(100):
        W = 0.5
        r1 = np.random.rand(N, 1) * np.ones((1, D))
        r2 = np.random.rand(N, 1) * np.ones((1, D))

        off_vel = (W * velocity
                   + r1 * (pbest_decs - particle_decs)
                   + r2 * (gbest_decs - particle_decs))
        off_decs = particle_decs + off_vel

        # Predict objectives using RBF surrogates
        off_objs = _rbf_predict(off_decs, rbf_models, M)

        # Update particle state
        particle_decs = off_decs.copy()
        velocity = off_vel.copy()

        # Update Pbest: replace if new solution is better in at least one objective
        replace = ~np.all(off_objs >= pbest_objs, axis=1)
        pbest_decs[replace] = off_decs[replace]
        pbest_objs[replace] = off_objs[replace]

        # Update Gbest: keep non-dominated, sort by crowding distance
        if gbest_decs.shape[0] > 1:
            gbest_front, _ = nd_sort(gbest_objs, gbest_objs.shape[0])
            nd_g = gbest_front == 1
            gbest_decs = gbest_decs[nd_g]
            gbest_objs = gbest_objs[nd_g]

            if gbest_decs.shape[0] > 1:
                cd_g = _crowding_distance_same_front(gbest_objs)
                rank_g = np.argsort(cd_g)[::-1]
                keep = min(N, gbest_decs.shape[0])
                gbest_decs = gbest_decs[rank_g[:keep]]
                gbest_objs = gbest_objs[rank_g[:keep]]

        # Ensure N Gbest guides are available (pad by repeating if needed)
        if gbest_decs.shape[0] < N:
            indices = np.random.randint(0, gbest_decs.shape[0], size=N)
            gbest_decs = gbest_decs[indices]
            gbest_objs = gbest_objs[indices]

    # Apply polynomial mutation if late in optimization (>= 75% budget used)
    if nfes >= 0.75 * max_nfes:
        proM = 1
        disM = 20
        site = np.random.rand(N, D) < proM / D
        mu_arr = np.random.rand(N, D)

        # Clip to [0, 1] before mutation
        off_decs = np.clip(off_decs, 0.0, 1.0)

        # Polynomial mutation (bounds [0, 1])
        temp = site & (mu_arr <= 0.5)
        off_decs[temp] = off_decs[temp] + (
            (2 * mu_arr[temp] + (1 - 2 * mu_arr[temp])
             * (1 - off_decs[temp]) ** (disM + 1)) ** (1.0 / (disM + 1)) - 1
        )

        temp = site & (mu_arr > 0.5)
        off_decs[temp] = off_decs[temp] + (
            1 - (2 * (1 - mu_arr[temp]) + 2 * (mu_arr[temp] - 0.5)
                 * off_decs[temp] ** (disM + 1)) ** (1.0 / (disM + 1))
        )

        # Re-predict after mutation
        off_objs = _rbf_predict(off_decs, rbf_models, M)

    return off_decs, off_objs


# =============================================================================
# BFE (Balanceable Fitness Estimation)
# =============================================================================

def _compute_bfe(pop_objs):
    """
    Compute BFE values for a population of non-dominated solutions.
    Used for archive management in PSO Gbest initialization.

    Parameters
    ----------
    pop_objs : np.ndarray, shape (N, M)
        Objective values

    Returns
    -------
    bfe : np.ndarray, shape (N,)
        BFE values (higher is better)
    """
    N, M = pop_objs.shape
    if N <= 1:
        return np.ones(N)

    # Normalize objectives
    fmin = pop_objs.min(axis=0)
    fmax = pop_objs.max(axis=0)
    f_range = fmax - fmin
    f_range[f_range < 1e-10] = 1.0
    norm_objs = (pop_objs - fmin) / f_range

    # Compute shifted distance (SDE)
    sde = np.full((N, N), np.inf)
    for i in range(N):
        shifted = np.maximum(norm_objs, norm_objs[i:i + 1, :])
        diffs = norm_objs[i:i + 1, :] - shifted
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        sde[i, :] = dists
        sde[i, i] = np.inf

    # Cd: diversity metric based on minimum SDE
    SDE_min = np.min(sde, axis=1)
    sde_range = SDE_min.max() - SDE_min.min()
    if sde_range < 1e-10:
        Cd = np.zeros(N)
    else:
        Cd = (SDE_min - SDE_min.min()) / sde_range

    # Cv: convergence value
    dis = np.sqrt(np.sum(norm_objs ** 2, axis=1))
    Cv = 1 - dis

    # d1 and d2: projected distances
    ones_vec = np.ones((1, M))
    cosine = 1 - cdist(norm_objs, ones_vec, metric='cosine').flatten()
    cosine = np.clip(cosine, -1, 1)
    d1 = dis * cosine
    d2 = dis * np.sqrt(np.clip(1 - cosine ** 2, 0, None))

    # Determine alpha and beta for each solution (8 cases)
    alpha = np.zeros(N)
    beta_arr = np.zeros(N)
    meanCd = np.mean(Cd)
    meanCv = np.mean(Cv)
    meand1 = np.mean(d1)
    meand2 = np.mean(d2)

    case111 = (Cv > meanCv) & (d1 <= meand1) & (Cd <= meanCd)
    case112 = (Cv > meanCv) & (d1 <= meand1) & (Cd > meanCd)
    case121 = (Cv > meanCv) & (d1 > meand1) & (Cd <= meanCd)
    case122 = (Cv > meanCv) & (d1 > meand1) & (Cd > meanCd)
    case211 = (Cv <= meanCv) & (d1 <= meand1) & (d2 > meand2) & (Cd <= meanCd)
    case212 = (Cv <= meanCv) & (d1 <= meand1) & (d2 > meand2) & (Cd > meanCd)
    case221 = (Cv <= meanCv) & ((d1 > meand1) | (d2 <= meand2)) & (Cd <= meanCd)
    case222 = (Cv <= meanCv) & ((d1 > meand1) | (d2 <= meand2)) & (Cd > meanCd)

    alpha[case111] = np.random.rand(np.sum(case111)) * 0.3 + 0.8
    beta_arr[case111] = 1.0
    alpha[case112] = 1.0
    beta_arr[case112] = 1.0
    alpha[case121] = 0.6
    beta_arr[case121] = 1.0
    alpha[case122] = 0.9
    beta_arr[case122] = 1.0
    alpha[case211] = np.random.rand(np.sum(case211)) * 0.3 + 0.8
    beta_arr[case211] = np.random.rand(np.sum(case211)) * 0.3 + 0.8
    alpha[case212] = 1.0
    beta_arr[case212] = 1.0
    alpha[case221] = 0.2
    beta_arr[case221] = 0.2
    alpha[case222] = 1.0
    beta_arr[case222] = 0.2

    bfe = alpha * Cd + beta_arr * Cv
    return bfe


# =============================================================================
# Crowding Distance for Same Front
# =============================================================================

def _crowding_distance_same_front(pop_objs):
    """
    Calculate crowding distance for solutions assumed to be in the same front.

    Parameters
    ----------
    pop_objs : np.ndarray, shape (N, M)
        Objective values

    Returns
    -------
    crowd_dis : np.ndarray, shape (N,)
        Crowding distance values
    """
    N, M = pop_objs.shape
    if N <= 2:
        return np.full(N, np.inf)

    crowd_dis = np.zeros(N)
    fmax = pop_objs.max(axis=0)
    fmin = pop_objs.min(axis=0)

    for m in range(M):
        rank = np.argsort(pop_objs[:, m])
        crowd_dis[rank[0]] = np.inf
        crowd_dis[rank[-1]] = np.inf
        denom = fmax[m] - fmin[m]
        if denom < 1e-10:
            continue
        for j in range(1, N - 1):
            crowd_dis[rank[j]] += (pop_objs[rank[j + 1], m] - pop_objs[rank[j - 1], m]) / denom

    return crowd_dis

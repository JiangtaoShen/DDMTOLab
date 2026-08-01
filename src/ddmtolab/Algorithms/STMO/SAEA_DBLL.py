"""
Surrogate-Assisted Evolutionary Algorithm with Decomposition-Based Local Learning (SAEA-DBLL)

This module implements SAEA-DBLL for computationally expensive multi/many-objective optimization.
It uses RBF surrogate models with a decomposition-based local learning strategy, where a subset of
reference vectors defines neighborhoods for competitive swarm optimization, combined with adaptive
vector selection and APD-based environmental selection.

References
----------
    [1] J. Shen, P. Wang, H. Dong, W. Wang, and J. Li. Surrogate-assisted evolutionary algorithm with decomposition-based local learning for high-dimensional multi-objective optimization. Expert Systems with Applications, 2024, 240: 122575.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.18
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
import warnings

warnings.filterwarnings("ignore")


class SAEA_DBLL:
    """
    Surrogate-Assisted Evolutionary Algorithm with Decomposition-Based Local Learning
    for expensive multi/many-objective optimization.

    Uses RBF surrogates with neighborhood-aware competitive swarm optimization and
    adaptive sub-reference vector management.

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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=None, alpha=2.0,
                 wmax=20, mu=5, T=3, K=2,
                 save_data=True, save_path='./Data', name='SAEA-DBLL', disable_tqdm=True):
        """
        Initialize SAEA-DBLL algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: dim+50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Requested number of reference vectors per task
            (default: 55 when the task has 10 objectives, otherwise 50)
        alpha : float, optional
            Exponent for theta progression (default: 2.0)
        wmax : int, optional
            Number of inner surrogate evolution generations (default: 20)
        mu : int, optional
            Number of re-evaluated solutions per iteration (default: 5)
        T : int, optional
            Neighborhood size for sub-vectors (default: 3)
        K : int, optional
            Division factor for sub-vector count (default: 2)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SAEA-DBLL')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.alpha = alpha
        self.wmax = wmax
        self.mu = mu
        self.T = T
        self.K = K
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SAEA-DBLL algorithm.

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

        # Set default initial samples: dim + 50
        if self.n_initial is None:
            n_initial_per_task = [dims[i] + 50 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Requested number of reference vectors (MATLAB: NV = 55 if M == 10 else 50).
        # NV stays at the *requested* value in MATLAB because UniformPoint is called
        # with a single output argument, so the actual vector count never overwrites it.
        if self.n is None:
            nv_request = [55 if n_objs[i] == 10 else 50 for i in range(nt)]
        else:
            nv_request = par_list(self.n, nt)

        # Generate uniform reference vectors
        V0_list = []
        n_per_task = [0] * nt
        for i in range(nt):
            v_i, actual_n = uniform_point(nv_request[i], n_objs[i])
            V0_list.append(v_i)
            n_per_task[i] = actual_n

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # A1: archive of all evaluated solutions
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]

        # Initialize adapted reference vectors and sub-vectors
        V_list = [v.copy() for v in V0_list]
        Ve_list = []
        for i in range(nt):
            # MATLAB: Ve = V(randperm(size(V,1),ceil(NV/K)),:) with NV = requested count
            n_avail = V_list[i].shape[0]
            n_sub = max(1, int(np.ceil(nv_request[i] / self.K)))
            n_sub = min(n_sub, n_avail)
            perm = np.random.permutation(n_avail)[:n_sub]
            Ve_list.append(V_list[i][perm].copy())

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break
            nfes_before_pass = sum(nfes_per_task)

            for i in active_tasks:
                M = n_objs[i]
                dim = dims[i]
                NV = n_per_task[i]
                V0 = V0_list[i]
                V = V_list[i]
                Ve = Ve_list[i]

                # Compute theta (progress parameter)
                theta = (nfes_per_task[i] / max_nfes_per_task[i]) ** self.alpha

                # Compute neighborhood matrix B for sub-vectors
                Ne = min(Ve.shape[0], self.T)
                if Ve.shape[0] > 1:
                    B_dist = cdist(Ve, Ve, metric='euclidean')
                    B = np.argsort(B_dist, axis=1)[:, :Ne]
                else:
                    B = np.zeros((1, 1), dtype=int)

                A1Dec = arc_decs[i].copy()
                A1Obj = arc_objs[i].copy()

                # Build RBF models for each objective
                RModels = []
                mS_per_obj = []
                for j in range(M):
                    mS_j, mY_j = dsmerge(A1Dec, A1Obj[:, j])
                    rmodel = rbf_build(mS_j, mY_j)
                    RModels.append(rmodel)
                    mS_per_obj.append(mS_j)

                # Inner loop: surrogate-based evolution
                PopDec = A1Dec.copy()
                PopObj = A1Obj.copy()
                PopVel = np.zeros_like(PopDec)
                Ns = []

                for w in range(self.wmax):
                    # Reproduction: decomposition-based local learning
                    OffDec, OffVel = _reproduction_operator(
                        PopObj, PopDec, PopVel, B, Ve, theta, M
                    )
                    PopDec = np.vstack([PopDec, OffDec])
                    PopVel = np.vstack([PopVel, OffVel])

                    # Predict objectives using RBF models
                    N_pop = PopDec.shape[0]
                    PopObj = np.zeros((N_pop, M))
                    for j in range(M):
                        PopObj[:, j] = rbf_predict(RModels[j], mS_per_obj[j], PopDec)

                    # Environmental selection (RVEA-style APD)
                    index = _environmental_selection(PopObj, V, theta)
                    Ns.append(len(index))
                    PopDec = PopDec[index]
                    PopObj = PopObj[index]
                    PopVel = PopVel[index]

                # Adapt reference vectors
                V = V0 * np.maximum(PopObj.max(axis=0) - PopObj.min(axis=0), 1e-10)
                V_list[i] = V

                mean_Ns = np.mean(Ns) if len(Ns) > 0 else NV
                n_sub_new = max(1, int(np.ceil(mean_Ns / self.K)))
                Ve = _vector_adaption(V0, PopObj, n_sub_new)
                Ve_list[i] = Ve

                # Sample selection (infill)
                PopNew = _sample_selection(PopDec, PopObj, V, self.mu, theta)

                if PopNew is not None and PopNew.shape[0] > 0:
                    # Remove duplicates against all previously evaluated solutions.
                    # SAEADBLL.m evaluates PopNew verbatim and only folds duplicates
                    # out afterwards in UpdataArchive, spending evaluations on points
                    # whose objective values are already known; every expensive
                    # algorithm in DDMTOLab filters first instead, so a duplicate
                    # never consumes part of the budget.
                    PopNew = remove_duplicates(PopNew, decs[i])

                    # Limit to remaining budget
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if PopNew.shape[0] > remaining:
                        PopNew = PopNew[:remaining]

                    # Re-evaluate with expensive function
                    new_objs, _ = evaluation_single(problem, PopNew, i)

                    # Update archive A1 (MATLAB UpdataArchive: unique decision rows)
                    arc_decs[i], arc_objs[i] = merge_archive(
                        arc_decs[i], arc_objs[i], PopNew, new_objs
                    )

                    # Update cumulative dataset
                    decs[i] = np.vstack([decs[i], PopNew])
                    objs[i] = np.vstack([objs[i], new_objs])

                    nfes_per_task[i] += PopNew.shape[0]
                    pbar.update(PopNew.shape[0])

            if sum(nfes_per_task) == nfes_before_pass:
                # No task could produce a new sample in a full pass; stop
                break

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.mu)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results


# =============================================================================
# Environmental Selection (RVEA-style APD)
# =============================================================================

def _environmental_selection(pop_obj, V, theta):
    """
    Environmental selection using angle-penalized distance (APD).

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    V : np.ndarray
        Reference vectors, shape (NV, M)
    theta : float
        Penalty parameter

    Returns
    -------
    index : np.ndarray
        Selected solution indices
    """
    N, M = pop_obj.shape
    NV = V.shape[0]

    # Translate objectives
    pop_obj_t = pop_obj - pop_obj.min(axis=0)

    # Smallest angle between each reference vector and others
    cosine = 1 - cdist(V, V, metric='cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    gamma = np.maximum(gamma, 1e-6)

    # Associate each solution to nearest reference vector
    angle = np.arccos(np.clip(1 - cdist(pop_obj_t, V, metric='cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)

    # Select one solution per reference vector
    Next = np.full(NV, -1, dtype=int)
    for vi in np.unique(associate):
        current = np.where(associate == vi)[0]
        APD = (1 + M * theta * angle[current, vi] / gamma[vi]) * \
              np.sqrt(np.sum(pop_obj_t[current] ** 2, axis=1))
        best = np.argmin(APD)
        Next[vi] = current[best]

    return Next[Next != -1]


# =============================================================================
# Reproduction Operator (Decomposition-Based Local Learning)
# =============================================================================

def _reproduction_operator(pop_obj, pop_dec, pop_vel, B, Ve, theta, M):
    """
    Decomposition-based local learning reproduction operator.

    Winners are selected via APD environmental selection using sub-vectors Ve.
    Losers learn from neighborhood winners via CSO-style velocity update,
    followed by polynomial mutation.

    Parameters
    ----------
    pop_obj : np.ndarray
        Population objectives, shape (N, M)
    pop_dec : np.ndarray
        Population decisions, shape (N, D)
    pop_vel : np.ndarray
        Population velocities, shape (N, D)
    B : np.ndarray
        Neighborhood matrix for sub-vectors, shape (n_sub, Ne)
    Ve : np.ndarray
        Sub-reference vectors, shape (n_sub, M)
    theta : float
        Penalty parameter
    M : int
        Number of objectives

    Returns
    -------
    off_dec : np.ndarray
        Offspring decisions
    off_vel : np.ndarray
        Offspring velocities
    """
    N, D = pop_dec.shape

    # Environmental selection on sub-vectors to determine winners
    winner_idx, winner_v_idx = _env_selection_with_vectors(pop_obj, Ve, theta)

    loser_idx = np.setdiff1d(np.arange(N), winner_idx)
    NL = len(loser_idx)

    if NL == 0:
        # All solutions are winners: OffDec = WinnerDec, OffVel = WinnerVel (MATLAB)
        off_dec = _polynomial_mutation(pop_dec[winner_idx].copy())
        off_vel = pop_vel[winner_idx].copy()
        return off_dec, off_vel

    loser_dec = pop_dec[loser_idx]
    winner_dec = pop_dec[winner_idx]
    loser_vel = pop_vel[loser_idx]
    winner_vel = pop_vel[winner_idx]
    loser_obj = pop_obj[loser_idx] - pop_obj.min(axis=0)

    # Associate each loser to nearest sub-vector
    angle_loser = np.arccos(np.clip(1 - cdist(loser_obj, Ve, metric='cosine'), -1, 1))
    loser_associate = np.argmin(angle_loser, axis=1)

    # For each loser, find a winner from the neighborhood of its associated sub-vector
    temp_winner_dec = np.zeros((NL, D))
    for li in range(NL):
        ve_idx = loser_associate[li]
        # Get neighborhood sub-vector indices
        neighbors = B[ve_idx]
        # Find which winner_v_idx entries are in the neighborhood
        match_mask = np.isin(neighbors, winner_v_idx)
        match_indices = neighbors[match_mask]

        if len(match_indices) > 0:
            # Map back to winner indices: find which winner corresponds to each matched Ve index
            chosen_ve = match_indices[np.random.randint(len(match_indices))]
            w_pos = np.where(winner_v_idx == chosen_ve)[0]
            if len(w_pos) > 0:
                w_idx = w_pos[np.random.randint(len(w_pos))]
            else:
                w_idx = np.random.randint(len(winner_idx))
        else:
            w_idx = np.random.randint(len(winner_idx))

        temp_winner_dec[li] = winner_dec[w_idx]

    # CSO velocity and position update
    r1 = np.random.rand(NL, 1) * np.ones((1, D))
    r2 = np.random.rand(NL, 1) * np.ones((1, D))
    off_vel_loser = r1 * loser_vel + r2 * (temp_winner_dec - loser_dec)
    off_dec_loser = loser_dec + off_vel_loser + r1 * (off_vel_loser - loser_vel)

    # Combine with winners
    off_dec = np.vstack([off_dec_loser, winner_dec])
    off_vel = np.vstack([off_vel_loser, winner_vel])

    # Polynomial mutation
    off_dec = _polynomial_mutation(off_dec)

    return off_dec, off_vel


def _env_selection_with_vectors(pop_obj, V, theta):
    """
    Environmental selection returning both solution indices and associated vector indices.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    V : np.ndarray
        Reference vectors, shape (NV, M)
    theta : float
        Penalty parameter

    Returns
    -------
    index : np.ndarray
        Selected solution indices
    index_v : np.ndarray
        Associated reference vector indices for each selected solution
    """
    N, M = pop_obj.shape
    NV = V.shape[0]

    pop_obj_t = pop_obj - pop_obj.min(axis=0)

    cosine = 1 - cdist(V, V, metric='cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    gamma = np.maximum(gamma, 1e-6)

    angle = np.arccos(np.clip(1 - cdist(pop_obj_t, V, metric='cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)

    Next = np.full(NV, -1, dtype=int)
    NextV = np.full(NV, -1, dtype=int)
    for vi in np.unique(associate):
        current = np.where(associate == vi)[0]
        APD = (1 + M * theta * angle[current, vi] / gamma[vi]) * \
              np.sqrt(np.sum(pop_obj_t[current] ** 2, axis=1))
        best = np.argmin(APD)
        Next[vi] = current[best]
        NextV[vi] = vi

    valid = Next != -1
    return Next[valid], NextV[valid]


def _polynomial_mutation(off_dec, dis_m=20):
    """
    Polynomial mutation in [0, 1] space.

    Parameters
    ----------
    off_dec : np.ndarray
        Decision variables, shape (N, D)
    dis_m : float
        Distribution index (default: 20)

    Returns
    -------
    off_dec : np.ndarray
        Mutated decision variables
    """
    N, D = off_dec.shape
    off_dec = np.clip(off_dec, 0, 1)

    site = np.random.rand(N, D) < 1.0 / D
    mu = np.random.rand(N, D)

    temp1 = site & (mu <= 0.5)
    off_dec[temp1] = off_dec[temp1] + (
        (2.0 * mu[temp1] + (1 - 2.0 * mu[temp1]) *
         (1 - off_dec[temp1]) ** (dis_m + 1)) ** (1.0 / (dis_m + 1)) - 1
    )

    temp2 = site & (mu > 0.5)
    off_dec[temp2] = off_dec[temp2] + (
        1 - (2.0 * (1 - mu[temp2]) + 2.0 * (mu[temp2] - 0.5) *
             (1 - (1 - off_dec[temp2])) ** (dis_m + 1)) ** (1.0 / (dis_m + 1))
    )

    return np.clip(off_dec, 0, 1)


# =============================================================================
# Vector Adaption
# =============================================================================

def _vector_adaption(V0, pop_obj, k):
    """
    Adapt sub-reference vectors by clustering active vectors.

    Parameters
    ----------
    V0 : np.ndarray
        Original reference vectors, shape (NV, M)
    pop_obj : np.ndarray
        Population objectives, shape (N, M)
    k : int
        Target number of sub-vectors

    Returns
    -------
    Ve : np.ndarray
        Adapted sub-reference vectors
    """
    # Scale V0 by objective range
    obj_range = pop_obj.max(axis=0) - pop_obj.min(axis=0)
    obj_range = np.maximum(obj_range, 1e-10)
    V = V0 * obj_range

    # Find active vectors
    _, active = _no_active(pop_obj, V)

    if len(active) == 0:
        # Fallback: use random subset
        perm = np.random.permutation(V.shape[0])[:max(1, k)]
        return V[perm]

    Va = V[active]
    k = max(1, min(k, len(active)))

    if len(active) <= k:
        return Va

    # Cluster active vectors into k groups
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=None)
    labels = km.fit_predict(Va)
    centers = km.cluster_centers_

    # For each cluster, select the vector closest to the cluster center
    Vindex = []
    for c in np.unique(labels):
        current = np.where(labels == c)[0]
        if len(current) == 1:
            Vindex.append(current[0])
        else:
            Vc = Va[current]
            angle = np.arccos(np.clip(1 - cdist(Vc, centers[c:c + 1], metric='cosine'), -1, 1))
            best = np.argmin(angle.flatten())
            Vindex.append(current[best])

    return Va[Vindex]


def _no_active(pop_obj, V):
    """
    Detect inactive reference vectors.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    V : np.ndarray
        Reference vectors, shape (NV, M)

    Returns
    -------
    num_inactive : int
        Number of inactive reference vectors
    active : np.ndarray
        Indices of active reference vectors
    """
    pop_obj_t = pop_obj - pop_obj.min(axis=0)
    angle = np.arccos(np.clip(1 - cdist(pop_obj_t, V, metric='cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)
    active = np.unique(associate)
    num_inactive = V.shape[0] - len(active)
    return num_inactive, active


# =============================================================================
# Sample Selection (Infill Strategy)
# =============================================================================

def _sample_selection(pop_dec, pop_obj, V, mu, theta):
    """
    Select infill points via clustering of active reference vectors
    and APD-based selection within each cluster.

    Parameters
    ----------
    pop_dec : np.ndarray
        Population decisions, shape (N, D)
    pop_obj : np.ndarray
        Population objectives, shape (N, M)
    V : np.ndarray
        Reference vectors, shape (NV, M)
    mu : int
        Number of solutions to select
    theta : float
        Penalty parameter

    Returns
    -------
    pop_new : np.ndarray
        Selected decision variables for re-evaluation
    """
    M = pop_obj.shape[1]
    N = pop_obj.shape[0]

    if N == 0:
        return np.empty((0, pop_dec.shape[1]))

    # Find active vectors
    num_inactive, active_idx = _no_active(pop_obj, V)
    n_active = len(active_idx)

    if n_active == 0:
        # Fallback: return random subset
        idx = np.random.permutation(N)[:min(mu, N)]
        return pop_dec[idx]

    NCluster = min(mu, n_active)
    Va = V[active_idx]

    # Translate objectives
    pop_obj_t = pop_obj - pop_obj.min(axis=0)

    # Compute gamma for active vectors
    if Va.shape[0] > 1:
        cosine = 1 - cdist(Va, Va, metric='cosine')
        np.fill_diagonal(cosine, 0)
        gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
        gamma = np.maximum(gamma, 1e-6)
    else:
        # MATLAB: cosine matrix is [0] after zeroing the diagonal -> acos(0) = pi/2
        gamma = np.array([np.pi / 2])

    # Associate each solution to nearest active vector
    angle = np.arccos(np.clip(1 - cdist(pop_obj_t, Va, metric='cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)

    # Compute APD for each solution
    APD_S = np.ones(N)
    for vi in np.unique(associate):
        current = np.where(associate == vi)[0]
        if len(current) > 0:
            APD = (1 + M * theta * angle[current, vi] / gamma[vi]) * \
                  np.sqrt(np.sum(pop_obj_t[current] ** 2, axis=1))
            APD_S[current] = APD

    # Cluster active vectors
    if NCluster >= n_active:
        labels = np.arange(n_active)
        NCluster = n_active
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=NCluster, n_init=10, random_state=None)
        labels = km.fit_predict(Va)

    # Map solutions to clusters via their associated active vector
    Cindex = labels[associate]

    # Select one solution per cluster
    Next = np.full(NCluster, -1, dtype=int)
    for c in np.unique(Cindex):
        current = np.where(Cindex == c)[0]
        # For each active vector in this cluster, find the best APD solution
        solution_best = []
        t = np.unique(associate[current])
        for vi in t:
            current_s = np.where(associate == vi)[0]
            best_id = current_s[np.argmin(APD_S[current_s])]
            solution_best.append(best_id)
        solution_best = np.array(solution_best)
        # Among the per-vector bests, select the overall best
        best = solution_best[np.argmin(APD_S[solution_best])]
        Next[c] = best

    selected = Next[Next != -1]
    if len(selected) == 0:
        idx = np.random.permutation(N)[:min(mu, N)]
        return pop_dec[idx]

    return pop_dec[selected]

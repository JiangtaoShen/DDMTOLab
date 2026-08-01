"""
Multi-Model-based Ranking Aggregation Evolutionary Algorithm (MMRAEA)

This module implements MMRAEA for computationally expensive multi/many-objective optimization.
It uses three RBF surrogate models (objective approximation, dominance prediction, and fitness
prediction) combined with a ranking aggregation infill strategy and dual evolutionary
optimization (CSO + GA) for balanced convergence and diversity.

References
----------
    [1] J. Shen, X. Wang, R. He, Y. Tian, W. Wang, P. Wang, and Z. Wen. Optimization of high-dimensional expensive multi-objective problems using multi-mode radial basis functions. Complex & Intelligent Systems, 2025, 11(2): 147.

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
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class MMRAEA:
    """
    Multi-Model-based Ranking Aggregation Evolutionary Algorithm for expensive
    multi/many-objective optimization.

    Uses three RBF surrogate models (objective, dominance, fitness) with ranking
    aggregation infill strategy and dual evolutionary optimization (CSO + GA).

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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, wmax=20,
                 save_data=True, save_path='./Data', name='MMRAEA', disable_tqdm=True):
        """
        Initialize MMRAEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size per task (default: 100)
        wmax : int, optional
            Number of inner surrogate evolution generations (default: 20)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MMRAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.wmax = wmax
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MMRAEA algorithm.

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

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # A1: archive of all evaluated solutions
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                dim = dims[i]
                N = n_per_task[i]

                A1Dec = arc_decs[i].copy()
                A1Obj = arc_objs[i].copy()

                # === Build dominance prediction model ===
                front_no_a1, _ = nd_sort(A1Obj, len(A1Dec))
                DS, DY = dsmerge(A1Dec, front_no_a1.astype(float))
                Dmodel = rbf_build(DS, DY)

                # === Build RBF approximation models for each objective ===
                RModels = []
                mS_per_obj = []
                for j in range(M):
                    mS_j, mY_j = dsmerge(A1Dec, A1Obj[:, j])
                    rmodel = rbf_build(mS_j, mY_j)
                    RModels.append(rmodel)
                    mS_per_obj.append(mS_j)

                # === Build fitness prediction model ===
                fitness_vals = _cal_fitness(A1Obj)
                FS, FY = dsmerge(A1Dec, fitness_vals)
                Fmodel = rbf_build(FS, FY)

                # === Evolutionary Optimization (dual: CSO + GA) ===
                PopDec, PopObj = _ea_optimization(
                    A1Dec, A1Obj, self.wmax, N, M, RModels, Fmodel, mS_per_obj, FS
                )

                # === Infill Strategy ===
                PopNew = _infill_strategy(PopDec, PopObj, Dmodel, DS, Fmodel, FS, A1Obj)

                # Remove duplicates
                PopNew = remove_duplicates(PopNew, decs[i])

                if PopNew.shape[0] > 0:
                    # Limit to remaining budget
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if PopNew.shape[0] > remaining:
                        PopNew = PopNew[:remaining]

                    # Re-evaluate with expensive function
                    new_objs, _ = evaluation_single(problem, PopNew, i)

                    # Update archive (merge and deduplicate)
                    arc_decs[i], arc_objs[i] = merge_archive(
                        arc_decs[i], arc_objs[i], PopNew, new_objs
                    )

                    # Update cumulative dataset
                    decs[i] = np.vstack([decs[i], PopNew])
                    objs[i] = np.vstack([objs[i], new_objs])

                    nfes_per_task[i] += PopNew.shape[0]
                    pbar.update(PopNew.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [decs[i].copy() for i in range(nt)]
        db_objs = [objs[i].copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results


# =============================================================================
# Shift-Based Density Fitness (calFitness)
# =============================================================================

def _cal_fitness(pop_obj):
    """
    Calculate shift-based density fitness for each solution.

    For each solution i, compute the shifted objective of every other solution j
    as max(obj_j, obj_i) element-wise. The fitness is the minimum distance from
    solution i to any shifted solution j.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)

    Returns
    -------
    fitness : np.ndarray
        Fitness values, shape (N,). Higher is better (more isolated).
    """
    N = pop_obj.shape[0]
    if N <= 1:
        return np.ones(N)

    # Normalize objectives to [0, 1]
    fmax = pop_obj.max(axis=0)
    fmin = pop_obj.min(axis=0)
    obj_range = fmax - fmin
    obj_range[obj_range == 0] = 1.0
    norm_obj = (pop_obj - fmin) / obj_range

    # Compute shift-based distance matrix
    dis = np.full((N, N), np.inf)
    for i in range(N):
        # Shifted objectives: max(norm_obj[j], norm_obj[i]) for all j
        shifted = np.maximum(norm_obj, norm_obj[i])
        for j in range(N):
            if j != i:
                dis[i, j] = np.linalg.norm(norm_obj[i] - shifted[j])

    fitness = np.min(dis, axis=1)
    return fitness


# =============================================================================
# Competitive Swarm Optimizer (CSO)
# =============================================================================

def _cso(pop_decs, fitness, pop_vel):
    """
    Competitive Swarm Optimization operator.

    Pairs solutions randomly, losers learn from winners, and polynomial mutation
    is applied.

    Parameters
    ----------
    pop_decs : np.ndarray
        Population decisions, shape (N, D), values in [0, 1]
    fitness : np.ndarray
        Fitness values, shape (N,). Higher is better.
    pop_vel : np.ndarray
        Velocity matrix, shape (N, D)

    Returns
    -------
    off_decs : np.ndarray
        Offspring decisions, shape (2*floor(N/2), D)
    off_vel : np.ndarray
        Offspring velocities, shape (2*floor(N/2), D)
    """
    N, D = pop_decs.shape

    if N < 2:
        return pop_decs.copy(), pop_vel.copy()

    # Random pairing
    half = N // 2
    rank = np.random.permutation(N)[:half * 2]
    idx1 = rank[:half]
    idx2 = rank[half:]

    # Determine winners and losers (higher fitness wins)
    swap = fitness[idx1] >= fitness[idx2]
    loser = idx1.copy()
    winner = idx2.copy()
    loser[swap] = idx2[swap]
    winner[swap] = idx1[swap]

    loser_dec = pop_decs[loser]
    winner_dec = pop_decs[winner]
    loser_vel = pop_vel[loser]
    winner_vel = pop_vel[winner]

    # Update velocity and position
    r1 = np.random.rand(half, 1) * np.ones((1, D))
    r2 = np.random.rand(half, 1) * np.ones((1, D))
    off_vel = r1 * loser_vel + r2 * (winner_dec - loser_dec)
    off_dec = loser_dec + off_vel + r1 * (off_vel - loser_vel)

    # Combine with winners
    off_dec = np.vstack([off_dec, winner_dec])
    off_vel = np.vstack([off_vel, winner_vel])

    # Polynomial mutation (in [0,1] space)
    n_off = off_dec.shape[0]
    dis_m = 20
    site = np.random.rand(n_off, D) < 1.0 / D
    mu = np.random.rand(n_off, D)

    off_dec = np.clip(off_dec, 0, 1)

    temp1 = site & (mu <= 0.5)
    off_dec[temp1] = off_dec[temp1] + (1.0 - 0.0) * (
        (2.0 * mu[temp1] + (1 - 2.0 * mu[temp1]) *
         (1 - (off_dec[temp1] - 0.0) / (1.0 - 0.0)) ** (dis_m + 1)) ** (1.0 / (dis_m + 1)) - 1
    )

    temp2 = site & (mu > 0.5)
    off_dec[temp2] = off_dec[temp2] + (1.0 - 0.0) * (
        1 - (2.0 * (1 - mu[temp2]) + 2.0 * (mu[temp2] - 0.5) *
             (1 - (1.0 - off_dec[temp2]) / (1.0 - 0.0)) ** (dis_m + 1)) ** (1.0 / (dis_m + 1))
    )

    off_dec = np.clip(off_dec, 0, 1)

    return off_dec, off_vel


# =============================================================================
# PDR Environmental Selection (NSGA-II style: ND-sort + crowding distance)
# =============================================================================

def _es_pdr(pop_obj, N):
    """
    Environmental selection using non-dominated sorting and crowding distance.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (n, M)
    N : int
        Number to select

    Returns
    -------
    index : np.ndarray
        Selected indices
    """
    n = pop_obj.shape[0]
    if n <= N:
        return np.arange(n)

    front_no, max_fno = nd_sort(pop_obj, N)
    Next = front_no < max_fno
    remaining = N - np.sum(Next)

    if remaining > 0:
        last_front = np.where(front_no == max_fno)[0]
        cd = crowding_distance(pop_obj, front_no)
        sorted_last = last_front[np.argsort(-cd[last_front])]
        Next[sorted_last[:remaining]] = True

    return np.where(Next)[0]


# =============================================================================
# EA Optimization (Dual: CSO + GA)
# =============================================================================

def _ea_optimization(A1Dec, A1Obj, wmax, N, M, RModels, Fmodel, mS_per_obj, FS):
    """
    Dual inner evolutionary optimization on surrogates.

    Branch 1: CSO (Competitive Swarm Optimizer) guided by fitness model.
    Branch 2: GA (Genetic Algorithm) with standard operators.
    Both branches use RBF objective models for prediction and PDR for selection.

    Parameters
    ----------
    A1Dec : np.ndarray
        Archive decisions, shape (n, D)
    A1Obj : np.ndarray
        Archive objectives, shape (n, M)
    wmax : int
        Number of inner generations
    N : int
        Population size
    M : int
        Number of objectives
    RModels : list
        List of M RBF models for objective prediction
    Fmodel : dict
        RBF model for fitness prediction
    mS_per_obj : list
        Training data for each RBF objective model, list of M arrays
    FS : np.ndarray
        Training data for fitness model

    Returns
    -------
    PopDec : np.ndarray
        Combined population decisions
    PopObj : np.ndarray
        Combined predicted objectives
    """
    n_archive = A1Dec.shape[0]
    D = A1Dec.shape[1]

    # --- Branch 1: CSO ---
    pop_dec1 = A1Dec.copy()
    pop_vel = np.zeros((n_archive, D))

    for w in range(wmax):
        # Predict fitness for CSO
        fit = rbf_predict(Fmodel, FS, pop_dec1)

        # CSO operator
        off_dec1, off_vel = _cso(pop_dec1, fit, pop_vel)
        pop_vel = np.vstack([pop_vel, off_vel])
        pop_dec1 = np.vstack([pop_dec1, off_dec1])

        # Predict objectives
        pop_obj1 = _rbf_predict_multi(RModels, mS_per_obj, pop_dec1, M)

        # PDR selection
        idx1 = _es_pdr(pop_obj1, N)
        pop_dec1 = pop_dec1[idx1]
        pop_vel = pop_vel[idx1]
        pop_obj1 = pop_obj1[idx1]

    # --- Branch 2: GA ---
    pop_dec2 = A1Dec.copy()

    for w in range(wmax):
        off_dec2 = ga_generation(pop_dec2, muc=20.0, mum=20.0)
        pop_dec2 = np.vstack([pop_dec2, off_dec2])

        pop_obj2 = _rbf_predict_multi(RModels, mS_per_obj, pop_dec2, M)

        idx2 = _es_pdr(pop_obj2, N)
        pop_dec2 = pop_dec2[idx2]
        pop_obj2 = pop_obj2[idx2]

    # Combine both branches
    PopDec = np.vstack([pop_dec1, pop_dec2])
    PopObj = np.vstack([pop_obj1, pop_obj2])

    return PopDec, PopObj


# =============================================================================
# Infill Strategy
# =============================================================================

def _infill_strategy(PopDec, PopObj, Dmodel, DS, Fmodel, FS, A1Obj):
    """
    Ranking aggregation infill strategy for selecting new evaluation points.

    1. Combine predicted population with archive, keep non-dominated from predicted set.
    2. Compute 3 criteria: shift-based density fitness, dominance prediction, fitness prediction.
    3. ND-sort on [-Fit1, Fit2, -Fit3] to extract front-1 candidates.
    4. Rank each criterion, compute Quality (sum of ranks) and Uncertainty (rank differences).
    5. Select front-1 of [Quality, Uncertainty] plus the most uncertain solution.

    Parameters
    ----------
    PopDec : np.ndarray
        Predicted population decisions, shape (NP, D)
    PopObj : np.ndarray
        Predicted population objectives, shape (NP, M)
    Dmodel : dict
        RBF model for dominance prediction
    DS : np.ndarray
        Training data for dominance model
    Fmodel : dict
        RBF model for fitness prediction
    FS : np.ndarray
        Training data for fitness model
    A1Obj : np.ndarray
        Archive objectives, shape (NA, M)

    Returns
    -------
    PopNew : np.ndarray
        Selected decision variables for re-evaluation
    """
    N_pop = PopDec.shape[0]

    # Combine predicted population with archive objectives for ND-sorting
    CPopObj = np.vstack([PopObj, A1Obj])
    FN1, _ = nd_sort(CPopObj, CPopObj.shape[0])

    # Find front-1 solutions that are from the predicted population (not archive)
    f1_mask = FN1 == 1
    f1_from_pop = np.where(f1_mask[:N_pop])[0]

    if len(f1_from_pop) == 0:
        # Fallback: return the best predicted solution
        return PopDec[:1]

    PopDec = PopDec[f1_from_pop]
    PopObj = PopObj[f1_from_pop]

    if len(f1_from_pop) <= 1:
        return PopDec

    N = PopDec.shape[0]

    # Compute 3 fitness criteria
    Fit1 = _cal_fitness(PopObj)                       # Shift-based density (higher = better)
    Fit2 = rbf_predict(Dmodel, DS, PopDec)           # Dominance prediction (lower = better)
    Fit3 = rbf_predict(Fmodel, FS, PopDec)           # Fitness prediction (higher = better)

    # ND-sort on [-Fit1, Fit2, -Fit3] (all minimized)
    tri_obj = np.column_stack([-Fit1, Fit2, -Fit3])
    FN, _ = nd_sort(tri_obj, N)

    # Keep only front-1
    f1_mask2 = FN == 1
    PopDec = PopDec[f1_mask2]
    PopObj = PopObj[f1_mask2]
    Fit1 = Fit1[f1_mask2]
    Fit2 = Fit2[f1_mask2]
    Fit3 = Fit3[f1_mask2]

    if PopDec.shape[0] <= 1:
        return PopDec

    N = PopDec.shape[0]

    # Rank-based scoring (matching MATLAB implementation)
    # Fit1: descending (higher is better) -> rank 1 for highest
    s1 = np.argsort(-Fit1)
    # Fit2: ascending (lower is better) -> rank 1 for lowest
    s2 = np.argsort(Fit2)
    # Fit3: descending (higher is better) -> rank 1 for highest
    s3 = np.argsort(-Fit3)

    Q1 = np.empty(N)
    Q2 = np.empty(N)
    Q3 = np.empty(N)
    for rank, idx in enumerate(s1):
        Q1[idx] = rank + 1
    for rank, idx in enumerate(s2):
        Q2[idx] = rank + 1
    for rank, idx in enumerate(s3):
        Q3[idx] = rank + 1

    Quality = Q1 + Q2 + Q3
    Uncertainty = np.abs(Q1 - Q2) + np.abs(Q1 - Q3) + np.abs(Q2 - Q3)

    # ND-sort on [Quality, Uncertainty]
    qu_obj = np.column_stack([Quality, Uncertainty])
    FN_qu, _ = nd_sort(qu_obj, N)
    f1_indices = np.where(FN_qu == 1)[0]

    # Also add the most uncertain solution
    max_unc_idx = np.argmax(Uncertainty)
    selected = np.unique(np.concatenate([f1_indices, [max_unc_idx]]))

    return PopDec[selected]


def _rbf_predict_multi(RModels, mS_per_obj, X_query, M):
    """
    Predict multiple objectives using RBF models.

    Parameters
    ----------
    RModels : list
        List of M RBF models
    mS_per_obj : list
        Training data for each RBF model, list of M arrays
    X_query : np.ndarray
        Query points, shape (nq, d)
    M : int
        Number of objectives

    Returns
    -------
    pred_objs : np.ndarray
        Predicted objectives, shape (nq, M)
    """
    nq = X_query.shape[0]
    pred_objs = np.zeros((nq, M))
    for j in range(M):
        pred_objs[:, j] = rbf_predict(RModels[j], mS_per_obj[j], X_query)
    return pred_objs

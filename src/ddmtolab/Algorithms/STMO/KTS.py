"""
Kriging-Assisted Two-Archive Search (KTS)

This module implements KTS for computationally expensive constrained/unconstrained
multi-objective optimization. It adaptively switches between two search modes:

- Mode 0 (unconstrained/KTA2-style): two-archive CA/DA with convergence/diversity
- Mode 1 (constrained/KCCMO-style): SPEA2-based fitness with K-means sampling

The switching is based on the correlation between convergence metric Q and
constraint violation CV.

References
----------
    [1] Z. Song, H. Wang, and B. Xue. A Kriging-Assisted Two-Archive Evolutionary Algorithm for Expensive Multi-Objective Optimization. IEEE Transactions on Evolutionary Computation, 2024.

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
from scipy.stats import wilcoxon, rankdata
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
import warnings

warnings.filterwarnings("ignore")


class KTS:
    """
    Kriging-Assisted Two-Archive Search for expensive constrained/unconstrained
    multi-objective optimization with adaptive mode switching.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'equal',
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
                 tau=0.6, phi=-0.2, mu=20, phi1=0.1, wmax1=10, mu1=5,
                 save_data=True, save_path='./Data', name='KTS', disable_tqdm=True):
        """
        Initialize KTS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task. PlatEMO's KTS draws
            ``Problem.N`` Latin-hypercube samples, so the default is ``n``.
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population/archive size per task (default: 100)
        tau : float, optional
            Correlation threshold for mode 0 (default: 0.6)
        phi : float, optional
            Correlation threshold for mode 1 (default: -0.2)
        mu : int, optional
            Number of elite solutions for correlation (default: 20)
        phi1 : float, optional
            Uncertainty sampling fraction (default: 0.1)
        wmax1 : int, optional
            Inner surrogate evolution generations (default: 10)
        mu1 : int, optional
            Number of re-evaluated solutions per generation (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'KTS')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.tau = tau
        self.phi = phi
        self.mu = mu
        self.phi1 = phi1
        self.wmax1 = wmax1
        self.mu1 = mu1
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the KTS algorithm.

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
        n_cons = problem.n_cons if hasattr(problem, 'n_cons') else [0] * nt

        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # MATLAB: N = Problem.N; P = UniformPoint(N,Problem.D,'Latin')
        if self.n_initial is None:
            n_initial_per_task = list(n_per_task)
        else:
            n_initial_per_task = par_list(self.n_initial, nt)

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Initialize archives for each task
        CAs = []   # Convergence Archives (IBEA-based)
        DA1s = []  # Diversity Archives (ND+Lp)
        P1s = []   # Population 1 (SPEA2, no constraints)
        P2s = []   # Population 2 (SPEA2, with constraints)

        for i in range(nt):
            N = n_per_task[i]
            M = n_objs[i]
            p_i = 1.0 / M
            con_i = cons[i]

            # CA: IBEA-based -- MATLAB: CA = UpdateCA([],Population,CAsize)
            CAs.append(_update_CA(None, objs[i], decs[i], con_i, N))

            # DA1: ND + diversity -- MATLAB: DA1 = UpdateDA(Population,[],N,p)
            DA1s.append(_update_DA(None, objs[i], decs[i], con_i, N, p_i))

            # MATLAB seeds P1 and P2 with the raw initial population; no
            # environmental selection is applied before the first iteration.
            P1s.append((objs[i].copy(), decs[i].copy(),
                        con_i.copy() if con_i is not None else None))
            P2s.append((objs[i].copy(), decs[i].copy(),
                        con_i.copy() if con_i is not None else None))

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
                p_i = 1.0 / M
                N = n_per_task[i]
                C = n_cons[i]

                # ===== Determine Search Mode =====
                search_mode = self._determine_search_mode(
                    objs[i], cons[i], self.mu, self.tau, self.phi
                )

                # MATLAB: DA = DA1 (mode 0) or P1 (mode 1)
                DA_pool = DA1s[i] if search_mode == 0 else P1s[i]

                # ===== Build GP models for the objectives (trained on A1) =====
                obj_models = []
                for j in range(M):
                    model = gp_build(decs[i], objs[i][:, j:j + 1], data_type)
                    obj_models.append(model)

                # ===== Build GP models for the constraints =====
                # MATLAB trains them on the deduplicated union [DA;CA;P2], not on
                # the full evaluated archive.
                con_models = []
                if C > 0 and search_mode == 1:
                    con_train_decs = np.vstack([DA_pool[1], CAs[i][1], P2s[i][1]])
                    con_train_cons = np.vstack([DA_pool[2], CAs[i][2], P2s[i][2]])
                    _, uniq = np.unique(np.round(con_train_decs * 1e4) / 1e4,
                                        axis=0, return_index=True)
                    con_train_decs = con_train_decs[uniq]
                    con_train_cons = con_train_cons[uniq]
                    for j in range(C):
                        model = gp_build(con_train_decs, con_train_cons[:, j:j + 1], data_type)
                        con_models.append(model)

                # ===== Inner Loop: Surrogate-based Evolution =====
                CCA_objs = CAs[i][0].copy()
                CCA_decs = CAs[i][1].copy()
                CCA_mse = np.zeros((CCA_objs.shape[0], M))

                CDA_objs = DA_pool[0].copy()
                CDA_decs = DA_pool[1].copy()
                CDA_mse = np.zeros((CDA_objs.shape[0], M))

                CP2_objs = P2s[i][0].copy()
                CP2_decs = P2s[i][1].copy()
                CP2_cons = P2s[i][2].copy() if P2s[i][2] is not None else None
                CP2_mse = np.zeros((CP2_objs.shape[0], M))

                for w in range(self.wmax1):
                    if search_mode == 0:
                        # KTA2-style: crossover-only + mutation-only
                        parentC_decs, parentM_decs = _mating_selection(
                            CCA_objs, CCA_decs, CDA_objs, CDA_decs, N
                        )
                        off_C = _crossover_only(parentC_decs, muc=20)
                        off_M = _mutation_only(parentM_decs, mum=20)
                        off_decs = np.vstack([off_C, off_M])
                    else:
                        # KCCMO-style: full GA on each pool independently
                        fitness_p2 = spea2_fitness(CP2_objs, CP2_cons)
                        fitness_da = spea2_fitness(CDA_objs)
                        pool1 = _platemo_tournament_selection(2, N, fitness_p2)
                        pool2 = _platemo_tournament_selection(2, N, fitness_da)
                        off1 = _full_ga(CP2_decs[pool1])
                        off2 = _full_ga(CDA_decs[pool2])
                        off_decs = np.vstack([off1, off2])

                    # Predict objectives for offspring only
                    off_objs, off_mse = _predict_objectives(
                        off_decs, obj_models, M, data_type)

                    # Predict constraints for offspring if needed
                    off_cons = None
                    if search_mode == 1 and len(con_models) > 0:
                        off_cons = _predict_constraints(
                            off_decs, con_models, C, data_type)

                    # Update surrogate CA: CCA + offspring
                    ca_objs = np.vstack([CCA_objs, off_objs])
                    ca_decs = np.vstack([CCA_decs, off_decs])
                    ca_mse = np.vstack([CCA_mse, off_mse])
                    CCA_objs, CCA_decs, CCA_mse = _k_update_CA(
                        ca_objs, ca_decs, ca_mse, N)

                    # Update surrogate DA: CDA + offspring
                    da_objs = np.vstack([CDA_objs, off_objs])
                    da_decs = np.vstack([CDA_decs, off_decs])
                    da_mse = np.vstack([CDA_mse, off_mse])
                    if search_mode == 0:
                        CDA_objs, CDA_decs, CDA_mse = _k_update_DA(
                            da_objs, da_decs, da_mse, N, p_i)
                    else:
                        CDA_objs, CDA_decs, _, CDA_mse = _k_update_P(
                            da_objs, da_decs, None, da_mse, N,
                            is_origin=False)
                        # Update surrogate P2: CP2 + offspring
                        p2_objs = np.vstack([CP2_objs, off_objs])
                        p2_decs = np.vstack([CP2_decs, off_decs])
                        p2_mse = np.vstack([CP2_mse, off_mse])
                        p2_cons = None
                        if off_cons is not None and CP2_cons is not None:
                            p2_cons = np.vstack([CP2_cons, off_cons])
                        elif off_cons is not None:
                            p2_cons = off_cons
                        CP2_objs, CP2_decs, CP2_cons, CP2_mse = _k_update_P(
                            p2_objs, p2_decs, p2_cons, p2_mse, N,
                            is_origin=True)

                # ===== Adaptive Sampling =====
                if search_mode == 0:
                    # Remove solutions already evaluated (MATLAB: setxor)
                    keep_ca = _not_in_archive(CCA_decs, decs[i])
                    keep_da = _not_in_archive(CDA_decs, decs[i])
                    # KTA2-style sampling
                    offspring_decs = _adaptive_sampling(
                        CCA_objs[keep_ca], CDA_objs[keep_da],
                        CCA_decs[keep_ca], CDA_decs[keep_da],
                        CDA_mse[keep_da], DA_pool[0], DA_pool[1],
                        self.mu1, p_i, self.phi1
                    )
                else:
                    # KCCMO-style sampling with K-means
                    # MATLAB: KCCMO_sampling(CP2, P2, mu1)
                    ref_front = _get_best_objs(P2s[i][0], P2s[i][2])
                    offspring_decs = _kccmo_sampling(
                        CP2_objs, CP2_decs, CP2_cons,
                        ref_front, self.mu1
                    )

                # Remove duplicates
                offspring_decs = remove_duplicates(offspring_decs, decs[i])

                # Never exceed the remaining budget of this task
                remaining = max_nfes_per_task[i] - nfes_per_task[i]
                if offspring_decs.shape[0] > remaining:
                    offspring_decs = offspring_decs[:remaining]

                if offspring_decs.shape[0] > 0:
                    # Evaluate
                    off_objs, off_cons = evaluation_single(problem, offspring_decs, i)

                    # Update all evaluated data
                    decs[i] = np.vstack([decs[i], offspring_decs])
                    objs[i] = np.vstack([objs[i], off_objs])
                    if cons[i] is not None and off_cons is not None:
                        cons[i] = np.vstack([cons[i], off_cons])

                    # Update real archives
                    CAs[i] = _update_CA(CAs[i], off_objs, offspring_decs, off_cons, N)
                    DA1s[i] = _update_DA(DA1s[i], off_objs, offspring_decs, off_cons, N, p_i)

                    P1s[i] = _update_P_real(
                        P1s[i], off_objs, offspring_decs, off_cons, N, is_origin=False)
                    P2s[i] = _update_P_real(
                        P2s[i], off_objs, offspring_decs, off_cons, N, is_origin=True)

                    nfes_per_task[i] += offspring_decs.shape[0]
                    pbar.update(offspring_decs.shape[0])

            if sum(nfes_per_task) == nfes_before_pass:
                # No task could produce a new sample in a full pass; stop
                break

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.mu1)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results

    @staticmethod
    def _determine_search_mode(objs_all, cons_all, mu, tau, phi):
        """
        Determine search mode based on correlation between convergence Q and CV.

        Parameters
        ----------
        objs_all : np.ndarray
            All evaluated objectives
        cons_all : np.ndarray or None
            All evaluated constraints
        mu : int
            Number of elite solutions for correlation
        tau : float
            Threshold for mode 0
        phi : float
            Threshold for mode 1

        Returns
        -------
        search_mode : int
            0 for unconstrained/KTA2 mode, 1 for constrained/KCCMO mode
        """
        N = objs_all.shape[0]

        # Compute constraint violation
        if cons_all is None or cons_all.shape[1] == 0:
            # No constraints → always mode 0
            return 0

        cv = np.sum(np.maximum(0, cons_all), axis=1)

        # If all feasible, mode 0
        if np.all(cv < 1e-10):
            return 0

        # Compute convergence metric Q using IBEA indicator
        Q = _cal_Q(objs_all)

        # Sort Q descending and take last mu+1 (best convergence) — MATLAB:
        # [Q, index] = sort(Q,'descend'); CV = CV(index);
        # coef = corrcoef(Q(end-mu:end), CV(end-mu:end))
        sorted_indices = np.argsort(Q)[::-1]
        Q_sorted = Q[sorted_indices]
        cv_sorted = cv[sorted_indices]
        n_elite = min(mu + 1, N)
        Q_elite = Q_sorted[-n_elite:]
        cv_elite = cv_sorted[-n_elite:]

        # Correlation coefficient
        if np.std(Q_elite) < 1e-10 or np.std(cv_elite) < 1e-10:
            return 0

        try:
            r_coef = np.corrcoef(Q_elite, cv_elite)[0, 1]
        except Exception:
            return 0

        if np.isnan(r_coef):
            return 0

        # Mode switching
        if r_coef < phi:
            return 1
        elif r_coef >= tau:
            return 0
        else:
            return np.random.choice([0, 1])


# =============================================================================
# Helper Functions
# =============================================================================

def _cal_Q(objs):
    """
    Compute convergence metric Q for each solution using IBEA indicator.

    Q(i) = 1/F(i) where F(i) is the IBEA fitness (MATLAB: ``Cal_Q``).

    The IBEA fitness is negative by construction (``sum(-exp(...)) + 1``), so the
    sign must be preserved -- clamping it to a positive floor destroys the
    ordering that the subsequent correlation test relies on.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (N, M)

    Returns
    -------
    Q : np.ndarray
        Convergence metric, shape (N,)
    """
    fitness, _, _ = ibea_fitness(objs, kappa=0.05)
    sign = np.where(fitness < 0, -1.0, 1.0)
    safe = np.where(np.abs(fitness) < 1e-12, sign * 1e-12, fitness)
    return 1.0 / safe


def _cal_fitness_spea2_ref(pop_objs, pop_cons, ref_objs):
    """
    Modified SPEA2 fitness using distance to reference front instead of density.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives, shape (N, M)
    pop_cons : np.ndarray or None
        Constraints, shape (N, C)
    ref_objs : np.ndarray
        Reference Pareto front objectives

    Returns
    -------
    fitness : np.ndarray
        Fitness values, shape (N,)
    """
    N = pop_objs.shape[0]

    if pop_cons is not None:
        cv = np.sum(np.maximum(0, pop_cons), axis=1)
    else:
        cv = np.zeros(N)

    dominate = np.zeros((N, N), dtype=bool)
    for p in range(N):
        for q in range(p + 1, N):
            if cv[p] < cv[q]:
                dominate[p, q] = True
            elif cv[p] > cv[q]:
                dominate[q, p] = True
            else:
                any_less = np.any(pop_objs[p] < pop_objs[q])
                any_greater = np.any(pop_objs[p] > pop_objs[q])
                if any_less and not any_greater:
                    dominate[p, q] = True
                elif any_greater and not any_less:
                    dominate[q, p] = True

    S = np.sum(dominate, axis=1).astype(float)
    R = np.zeros(N)
    for p in range(N):
        dominators = np.where(dominate[:, p])[0]
        R[p] = np.sum(S[dominators])

    # Distance to reference front (min Euclidean distance)
    dist_to_ref = cdist(pop_objs, ref_objs)
    min_dist = np.min(dist_to_ref, axis=1)
    D = 1.0 / (min_dist + 2.0)

    fitness = R + D
    return fitness


def _predict_objectives(pop_decs, obj_models, M, data_type):
    """Predict objectives and MSE using GP models."""
    N = pop_decs.shape[0]
    pop_objs = np.zeros((N, M))
    pop_mse = np.zeros((N, M))

    for j in range(M):
        pred, std = gp_predict(obj_models[j], pop_decs, data_type)
        pop_objs[:, j] = pred.flatten()
        pop_mse[:, j] = (std.flatten()) ** 2

    return pop_objs, pop_mse


def _predict_constraints(pop_decs, con_models, C, data_type):
    """Predict constraints using GP models."""
    N = pop_decs.shape[0]
    pop_cons = np.zeros((N, C))

    for j in range(C):
        pred, _ = gp_predict(con_models[j], pop_decs, data_type)
        pop_cons[:, j] = pred.flatten()

    return pop_cons


def _mating_selection(CA_objs, CA_decs, DA_objs, DA_decs, N):
    """
    KTA2-style mating selection from CA and DA.

    Returns parents for crossover and mutation.
    """
    CA_n = CA_objs.shape[0]
    DA_n = DA_objs.shape[0]
    half_N = int(np.ceil(N / 2))

    # Select from CA with dominance comparison
    idx1 = np.random.randint(0, CA_n, size=half_N)
    idx2 = np.random.randint(0, CA_n, size=half_N)

    any_less = np.any(CA_objs[idx1] < CA_objs[idx2], axis=1)
    any_greater = np.any(CA_objs[idx1] > CA_objs[idx2], axis=1)
    dominate = any_less.astype(int) - any_greater.astype(int)

    selected_CA = np.where(dominate == 1, idx1, idx2)
    selected_DA = np.random.randint(0, DA_n, size=half_N)

    parentC_decs = np.vstack([CA_decs[selected_CA], DA_decs[selected_DA]])

    # Mutation parents from CA
    parentM_decs = CA_decs[np.random.randint(0, CA_n, size=N)]

    return parentC_decs, parentM_decs


def _platemo_tournament_selection(K, N, *fitness):
    """
    Exact port of PlatEMO's ``TournamentSelection``.

    Candidates are compared lexicographically on the given fitness keys (lower is
    better). Solutions with identical fitness share the same rank, so a tournament
    among tied candidates is decided by the (random) draw order, i.e. uniformly at
    random -- unlike a total order that would break ties by index.
    """
    fits = np.column_stack([np.asarray(f, dtype=float).ravel() for f in fitness])
    _, loc = np.unique(fits, axis=0, return_inverse=True)
    loc = loc.ravel()
    parents = np.random.randint(0, fits.shape[0], size=(K, N))
    best = np.argmin(loc[parents], axis=0)
    return parents[best, np.arange(N)]


def _stack_optional(a, b):
    """Vertically stack two optionally-None arrays."""
    if a is None:
        return b.copy() if b is not None else None
    if b is None:
        return a.copy()
    return np.vstack([a, b])


def _update_CA(CA, new_objs, new_decs, new_cons, max_size):
    """Update Convergence Archive using IBEA fitness (MATLAB: ``UpdateCA``)."""
    if CA is None:
        CA_objs = new_objs.copy()
        CA_decs = new_decs.copy()
        CA_cons = new_cons.copy() if new_cons is not None else None
    else:
        CA_objs = np.vstack([CA[0], new_objs])
        CA_decs = np.vstack([CA[1], new_decs])
        CA_cons = _stack_optional(CA[2], new_cons)

    N = CA_objs.shape[0]
    if N <= max_size:
        return CA_objs, CA_decs, CA_cons

    fitness, I, C = ibea_fitness(CA_objs, kappa=0.05)

    choose = list(range(N))
    while len(choose) > max_size:
        fit_values = fitness[choose]
        min_idx = np.argmin(fit_values)
        to_remove = choose[min_idx]

        if C[to_remove] > 1e-10:
            fitness += np.exp(-I[to_remove, :] / C[to_remove] / 0.05)

        choose.pop(min_idx)

    return (CA_objs[choose], CA_decs[choose],
            CA_cons[choose] if CA_cons is not None else None)


def _update_DA(DA, new_objs, new_decs, new_cons, max_size, p):
    """Update Diversity Archive with ND sorting and Lp truncation (``UpdateDA``)."""
    if DA is None:
        DA_objs = new_objs.copy()
        DA_decs = new_decs.copy()
        DA_cons = new_cons.copy() if new_cons is not None else None
    else:
        DA_objs = np.vstack([DA[0], new_objs])
        DA_decs = np.vstack([DA[1], new_decs])
        DA_cons = _stack_optional(DA[2], new_cons)

    N = DA_objs.shape[0]
    front_no, _ = nd_sort(DA_objs, N)
    nd_mask = front_no == 1
    DA_objs = DA_objs[nd_mask]
    DA_decs = DA_decs[nd_mask]
    if DA_cons is not None:
        DA_cons = DA_cons[nd_mask]

    N = DA_objs.shape[0]
    if N <= max_size:
        return DA_objs, DA_decs, DA_cons

    # Select extreme solutions
    choose = np.zeros(N, dtype=bool)
    for m in range(DA_objs.shape[1]):
        choose[np.argmin(DA_objs[:, m])] = True
        choose[np.argmax(DA_objs[:, m])] = True

    if np.sum(choose) > max_size:
        chosen_idx = np.where(choose)[0]
        to_remove = np.random.choice(chosen_idx, size=np.sum(choose) - max_size, replace=False)
        choose[to_remove] = False
    elif np.sum(choose) < max_size:
        diff = DA_objs[:, np.newaxis, :] - DA_objs[np.newaxis, :, :]
        dist = np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
        np.fill_diagonal(dist, np.inf)

        while np.sum(choose) < max_size:
            remaining = np.where(~choose)[0]
            chosen = np.where(choose)[0]
            if len(remaining) == 0:
                break
            min_dists = np.min(dist[np.ix_(remaining, chosen)], axis=1)
            best = np.argmax(min_dists)
            choose[remaining[best]] = True

    return (DA_objs[choose], DA_decs[choose],
            DA_cons[choose] if DA_cons is not None else None)


def _update_P(pop_objs, pop_decs, pop_cons, N, is_origin):
    """
    SPEA2-based population update.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives
    pop_decs : np.ndarray
        Decisions
    pop_cons : np.ndarray or None
        Constraints
    N : int
        Target size
    is_origin : bool
        If True, include constraints in fitness

    Returns
    -------
    selected_objs, selected_decs, selected_cons : np.ndarray
    """
    if is_origin and pop_cons is not None:
        fitness = spea2_fitness(pop_objs, pop_cons)
    else:
        fitness = spea2_fitness(pop_objs)

    n_pop = pop_objs.shape[0]

    # Select solutions with fitness < 1 (non-dominated in SPEA2 sense)
    next_mask = fitness < 1.0

    if np.sum(next_mask) < N:
        # Not enough: take top N by fitness
        sorted_idx = np.argsort(fitness)
        selected = sorted_idx[:min(N, n_pop)]
    elif np.sum(next_mask) > N:
        # Too many: truncate by distance
        candidates = np.where(next_mask)[0]
        selected = spea2_truncation(pop_objs[candidates], N)
        selected = candidates[selected]
    else:
        selected = np.where(next_mask)[0]

    # MATLAB's Update_P sorts the surviving population by ascending fitness
    selected = np.asarray(selected)[np.argsort(fitness[selected], kind='stable')]

    out_cons = pop_cons[selected] if pop_cons is not None else None
    return pop_objs[selected], pop_decs[selected], out_cons


def _update_P_real(P, new_objs, new_decs, new_cons, N, is_origin):
    """Update real population P with new solutions (MATLAB: ``Update_P``)."""
    P_objs = np.vstack([P[0], new_objs])
    P_decs = np.vstack([P[1], new_decs])
    P_cons = _stack_optional(P[2], new_cons)

    return _update_P(P_objs, P_decs, P_cons, N, is_origin)


def _k_update_CA(pop_objs, pop_decs, pop_mse, max_size):
    """Update predicted CA using IBEA fitness."""
    N = pop_objs.shape[0]
    if N <= max_size:
        return pop_objs.copy(), pop_decs.copy(), pop_mse.copy()

    fitness, I, C = ibea_fitness(pop_objs, kappa=0.05)

    choose = list(range(N))
    while len(choose) > max_size:
        fit_values = fitness[choose]
        min_idx = np.argmin(fit_values)
        to_remove = choose[min_idx]

        if C[to_remove] > 1e-10:
            fitness += np.exp(-I[to_remove, :] / C[to_remove] / 0.05)

        choose.pop(min_idx)

    return pop_objs[choose], pop_decs[choose], pop_mse[choose]


def _k_update_DA(pop_objs, pop_decs, pop_mse, max_size, p):
    """Update predicted DA with ND sort + Lp truncation."""
    N = pop_objs.shape[0]

    min_vals = np.min(pop_objs, axis=0)
    max_vals = np.max(pop_objs, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    pop_objs_norm = (pop_objs - min_vals) / range_vals

    front_no, _ = nd_sort(pop_objs, N)
    nd_mask = front_no == 1

    pop_objs = pop_objs[nd_mask]
    pop_decs = pop_decs[nd_mask]
    pop_mse = pop_mse[nd_mask]
    pop_objs_norm = pop_objs_norm[nd_mask]

    N = pop_objs.shape[0]
    if N <= max_size:
        return pop_objs, pop_decs, pop_mse

    M = pop_objs_norm.shape[1]
    choose = np.zeros(N, dtype=bool)
    select = np.random.permutation(M)
    if select[0] < N:
        choose[select[0]] = True
    else:
        choose[0] = True

    if np.sum(choose) < max_size:
        diff = pop_objs_norm[:, np.newaxis, :] - pop_objs_norm[np.newaxis, :, :]
        dist = np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
        np.fill_diagonal(dist, np.inf)

        while np.sum(choose) < max_size:
            remaining = np.where(~choose)[0]
            chosen = np.where(choose)[0]
            if len(remaining) == 0:
                break
            min_dists = np.min(dist[np.ix_(remaining, chosen)], axis=1)
            best = np.argmax(min_dists)
            choose[remaining[best]] = True

    return pop_objs[choose], pop_decs[choose], pop_mse[choose]


def _k_update_P(pop_objs, pop_decs, pop_cons, pop_mse, N, is_origin):
    """
    Update predicted population P using SPEA2 fitness.

    Returns (objs, decs, cons, mse) tuple.
    """
    if is_origin and pop_cons is not None:
        fitness = spea2_fitness(pop_objs, pop_cons)
    else:
        fitness = spea2_fitness(pop_objs)

    n_pop = pop_objs.shape[0]
    next_mask = fitness < 1.0

    if np.sum(next_mask) < N:
        sorted_idx = np.argsort(fitness)
        selected = sorted_idx[:min(N, n_pop)]
    elif np.sum(next_mask) > N:
        candidates = np.where(next_mask)[0]
        sel = spea2_truncation(pop_objs[candidates], N)
        selected = candidates[sel]
    else:
        selected = np.where(next_mask)[0]

    out_cons = pop_cons[selected] if pop_cons is not None else None
    return pop_objs[selected], pop_decs[selected], out_cons, pop_mse[selected]


def _adaptive_sampling(CA_objs, DA_objs, CA_decs, DA_decs, DA_mse,
                        real_DA_objs, real_DA_decs, mu, p, phi):
    """
    KTA2-style adaptive sampling with convergence/uncertainty/diversity strategies.
    """
    combined_objs = np.vstack([CA_objs, DA_objs])
    ideal_point = np.min(combined_objs, axis=0)

    # MATLAB: a diversity archive no larger than mu forces convergence sampling
    if DA_decs.shape[0] <= mu:
        flag = 1
    else:
        flag = _cal_convergence(CA_objs, DA_objs, ideal_point)

    if flag == 1:
        # Convergence sampling from CA
        N = CA_objs.shape[0]
        if N <= mu:
            return CA_decs.copy()

        fitness, I, C = ibea_fitness(CA_objs, kappa=0.05)

        choose = list(range(N))
        while len(choose) > mu:
            fit_values = fitness[choose]
            min_idx = np.argmin(fit_values)
            to_remove = choose[min_idx]

            if C[to_remove] > 1e-10:
                fitness += np.exp(-I[to_remove, :] / C[to_remove] / 0.05)

            choose.pop(min_idx)

        return CA_decs[choose]
    else:
        if DA_decs.shape[0] <= mu:
            return DA_decs.copy()

        pd_pred = _pure_diversity(DA_objs)
        pd_real = _pure_diversity(real_DA_objs)

        if pd_pred < pd_real:
            # Uncertainty sampling
            DA_n = DA_mse.shape[0]
            subset_size = max(1, int(np.ceil(phi * DA_n)))
            chosen = []

            for _ in range(mu):
                perm = np.random.permutation(DA_n)
                subset_idx = perm[:subset_size]
                n_obj_cols = min(DA_mse.shape[1], DA_objs.shape[1])
                uncertainty = np.mean(DA_mse[subset_idx, :n_obj_cols], axis=1)
                best = subset_idx[np.argmax(uncertainty)]
                chosen.append(best)

            return DA_decs[chosen]
        else:
            # Diversity sampling
            all_objs = np.vstack([DA_objs, real_DA_objs])
            min_vals = np.min(all_objs, axis=0)
            max_vals = np.max(all_objs, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0

            DA_Nor = (real_DA_objs - min_vals) / range_vals
            DA_Nor_pre = (DA_objs - min_vals) / range_vals

            N_real = DA_Nor.shape[0]
            Pop = np.vstack([DA_Nor, DA_Nor_pre])
            Pop_dec = np.vstack([real_DA_decs, DA_decs])
            NN = Pop.shape[0]

            choose = np.zeros(NN, dtype=bool)
            choose[:N_real] = True
            target_size = N_real + mu

            diff = Pop[:, np.newaxis, :] - Pop[np.newaxis, :, :]
            dist_matrix = np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
            np.fill_diagonal(dist_matrix, np.inf)

            offspring = []
            while np.sum(choose) < target_size:
                remaining = np.where(~choose)[0]
                chosen_idx = np.where(choose)[0]
                if len(remaining) == 0:
                    break
                min_dists = np.min(dist_matrix[np.ix_(remaining, chosen_idx)], axis=1)
                best = np.argmax(min_dists)
                choose[remaining[best]] = True
                offspring.append(Pop_dec[remaining[best]])

            if len(offspring) == 0:
                idx = np.random.choice(DA_decs.shape[0],
                                       size=min(mu, DA_decs.shape[0]), replace=False)
                return DA_decs[idx]

            return np.array(offspring)


def _kccmo_sampling(pop_objs, pop_decs, pop_cons, ref_objs, mu1):
    """
    KCCMO-style sampling using K-means clustering and fitness selection.

    Parameters
    ----------
    pop_objs : np.ndarray
        Population objectives
    pop_decs : np.ndarray
        Population decisions
    pop_cons : np.ndarray or None
        Population constraints
    ref_objs : np.ndarray
        Reference front objectives (from CA)
    mu1 : int
        Number of solutions to select

    Returns
    -------
    selected_decs : np.ndarray
        Selected decision variables
    """
    N = pop_objs.shape[0]
    n_clusters = min(mu1, N)

    if N <= mu1:
        return pop_decs.copy()

    # Compute fitness using reference front
    if ref_objs is not None and ref_objs.shape[0] > 0:
        fitness = _cal_fitness_spea2_ref(pop_objs, pop_cons, ref_objs)
    else:
        fitness = spea2_fitness(pop_objs, pop_cons)

    # K-means clustering on objectives
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=None)
        labels = kmeans.fit_predict(pop_objs)
    except ImportError:
        # Fallback: simple K-means
        labels = _simple_kmeans(pop_objs, n_clusters)

    # Select best from each cluster
    selected = []
    for c in range(n_clusters):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) == 0:
            continue
        best_in_cluster = cluster_idx[np.argmin(fitness[cluster_idx])]
        selected.append(best_in_cluster)

    if len(selected) == 0:
        idx = np.random.choice(N, size=min(mu1, N), replace=False)
        return pop_decs[idx]

    return pop_decs[selected]


def _simple_kmeans(X, k, max_iter=50):
    """Simple K-means fallback if sklearn unavailable."""
    N = X.shape[0]
    idx = np.random.choice(N, size=k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(N, dtype=int)

    for _ in range(max_iter):
        dist = cdist(X, centers)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                centers[c] = members.mean(axis=0)

    return labels


def _cal_convergence(pop_obj1, pop_obj2, z_min):
    """
    Wilcoxon signed-rank test comparing CA vs DA convergence.
    Returns 1 if CA converges significantly better.
    """
    N1 = pop_obj1.shape[0]
    N2 = pop_obj2.shape[0]

    if N1 != N2:
        return 0

    try:
        pop_obj = np.vstack([pop_obj1, pop_obj2]) - z_min
        denominator = np.max(pop_obj, axis=0) - z_min
        denominator[np.abs(denominator) < 1e-10] = 1.0
        pop_obj = pop_obj / denominator

        distance1 = np.sqrt(np.clip(np.sum(pop_obj[:N1], axis=1), 0, None))
        distance2 = np.sqrt(np.clip(np.sum(pop_obj[N1:], axis=1), 0, None))

        diff = distance1 - distance2
        abs_diff = np.abs(diff)
        eps_tol = np.finfo(float).eps * (np.abs(distance1) + np.abs(distance2))
        nonzero = abs_diff > eps_tol

        if np.sum(nonzero) < 2:
            return 0

        diff_nz = diff[nonzero]
        abs_diff_nz = abs_diff[nonzero]

        ranks = rankdata(abs_diff_nz)
        r1 = np.sum(ranks[diff_nz < 0])

        _, p_value = wilcoxon(distance1[nonzero], distance2[nonzero])

        flag = 1 if p_value <= 0.05 else 0
        r2 = np.sum(ranks) - r1

        if flag == 1 and (r1 - r2) < 0:
            flag = 0

        return flag
    except Exception:
        return 0


def _pure_diversity(pop_obj):
    """
    Pure diversity using maximum spanning tree with Minkowski(0.1) distance.
    """
    N = pop_obj.shape[0]
    if N <= 1:
        return 0.0

    C = np.eye(N, dtype=bool)
    D = cdist(pop_obj, pop_obj, metric='minkowski', p=0.1)
    np.fill_diagonal(D, np.inf)

    score = 0.0
    for k in range(N - 1):
        while True:
            d = np.min(D, axis=1)
            J = np.argmin(D, axis=1)
            i_node = np.argmax(d)
            j_node = J[i_node]

            if D[j_node, i_node] != -np.inf:
                D[j_node, i_node] = np.inf
            if D[i_node, j_node] != -np.inf:
                D[i_node, j_node] = np.inf

            P = C[i_node].copy()
            while not P[j_node]:
                new_P = np.any(C[P], axis=0)
                if np.array_equal(P, new_P):
                    break
                P = new_P

            if not P[j_node]:
                break

        C[i_node, j_node] = True
        C[j_node, i_node] = True
        D[i_node, :] = -np.inf
        score += d[i_node]

    return score


def _crossover_only(parents, muc=20):
    """
    SBX crossover only -- MATLAB ``OperatorGA(Problem,Parent,{1,20,0,0})``.

    PlatEMO splits the parent matrix into ``Parent(1:floor(N/2))`` and
    ``Parent(floor(N/2)+1:2*floor(N/2))`` without shuffling, so in mode 0 the
    ``[CA-selected ; DA-random]`` layout of ParentCdec crosses a CA parent with a
    DA parent. That structure must be preserved.
    """
    n, d = parents.shape
    num_pairs = n // 2
    if num_pairs == 0:
        return np.zeros((0, d))

    off1 = np.zeros((num_pairs, d))
    off2 = np.zeros((num_pairs, d))
    for j in range(num_pairs):
        c1, c2 = crossover(parents[j, :], parents[num_pairs + j, :], mu=muc)
        off1[j] = c1
        off2[j] = c2
    return np.vstack([off1, off2])


def _mutation_only(parents, mum=20):
    """
    Polynomial mutation only -- MATLAB ``OperatorGA(Problem,Parent,{0,0,1,20})``.

    With ``proC = 0`` the SBX step returns the parents unchanged and PlatEMO keeps
    only the first ``2*floor(N/2)`` of them.
    """
    n, d = parents.shape
    n_off = 2 * (n // 2)
    offdecs = np.zeros((n_off, d))
    for j in range(n_off):
        offdecs[j] = mutation(np.clip(parents[j, :], 0.0, 1.0), mu=mum)
    return offdecs


def _full_ga(parents, muc=20, mum=20):
    """Full GA (MATLAB ``OperatorGA`` defaults {1,20,1,20}) on a mating pool."""
    off = _crossover_only(parents, muc=muc)
    for j in range(off.shape[0]):
        off[j] = mutation(off[j], mu=mum)
    return off


def _not_in_archive(new_decs, archive_decs):
    """Return boolean mask of rows in new_decs NOT present in archive_decs."""
    if new_decs.shape[0] == 0 or archive_decs.shape[0] == 0:
        return np.ones(new_decs.shape[0], dtype=bool)
    dist = cdist(new_decs, archive_decs)
    return np.min(dist, axis=1) > 1e-5


def _get_best_objs(pop_objs, pop_cons):
    """
    Get non-dominated feasible objectives (MATLAB: Population.best.objs).

    Returns the Pareto front of feasible solutions, or all objectives if
    no feasible solutions exist.
    """
    N = pop_objs.shape[0]
    if pop_cons is not None:
        cv = np.sum(np.maximum(0, pop_cons), axis=1)
        feasible = cv < 1e-10
        if np.any(feasible):
            feas_objs = pop_objs[feasible]
            front_no, _ = nd_sort(feas_objs, feas_objs.shape[0])
            return feas_objs[front_no == 1]
    front_no, _ = nd_sort(pop_objs, N)
    return pop_objs[front_no == 1]

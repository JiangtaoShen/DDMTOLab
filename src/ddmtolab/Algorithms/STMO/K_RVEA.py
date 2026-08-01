"""
Kriging-assisted Reference Vector Guided Evolutionary Algorithm (K-RVEA)

This module implements K-RVEA for computationally expensive multi/many-objective optimization.

References
----------
    [1] T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya. A surrogate-assisted reference vector guided evolutionary algorithm for computationally expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2018, 22(1): 129-142.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.11
Version: 2.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Algorithms.STMO.RVEA import rvea_selection
import warnings

warnings.filterwarnings("ignore")


class K_RVEA:
    """
    Kriging-assisted Reference Vector Guided Evolutionary Algorithm for expensive
    multi/many-objective optimization.

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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, alpha=2.0, wmax=20, mu=5,
                 save_data=True, save_path='./Data', name='K-RVEA', disable_tqdm=True):
        """
        Initialize K-RVEA algorithm.

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
        alpha : float, optional
            Parameter controlling the rate of change of penalty (default: 2.0)
        wmax : int, optional
            Number of generations before updating Kriging models (default: 20)
        mu : int, optional
            Number of re-evaluated solutions at each generation (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'K-RVEA')
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
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the K-RVEA algorithm.

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

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate uniformly distributed reference vectors for each task
        V0 = []
        for i in range(nt):
            v_i, actual_n = uniform_point(n_per_task[i], n_objs[i])
            V0.append(v_i)
            n_per_task[i] = actual_n

        # Initialize adaptive reference vectors
        V = [v.copy() for v in V0]

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # A1: training archive (maintained at ~NI size for model building)
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break
            nfes_before_pass = sum(nfes_per_task)

            for i in active_tasks:
                m = n_objs[i]
                dim = dims[i]
                NI = n_initial_per_task[i]

                # Build GP models for each objective using training archive
                models = mo_gp_build(arc_decs[i], arc_objs[i], data_type)

                # --- Inner optimization loop on surrogates ---
                pop_decs = arc_decs[i].copy()

                for w in range(1, self.wmax + 1):
                    # Generate offspring via GA operators
                    off_decs = ga_generation(pop_decs, muc=20.0, mum=20.0)

                    # Merge parent and offspring
                    pop_decs = np.vstack([pop_decs, off_decs])

                    # Predict objectives and MSE for all solutions
                    pop_objs, pop_mse = mo_gp_predict(models, pop_decs, data_type, mse=True)

                    # Environmental selection using angle-penalized distance
                    theta = (w / self.wmax) ** self.alpha
                    cons_zero = np.zeros((pop_decs.shape[0], 1))
                    index = rvea_selection(pop_objs, cons_zero, V[i], theta)

                    pop_decs = pop_decs[index]
                    pop_objs = pop_objs[index]
                    pop_mse = pop_mse[index]

                    # Adapt reference vectors periodically
                    adapt_interval = max(1, int(np.ceil(self.wmax * 0.1)))
                    if w % adapt_interval == 0:
                        obj_range = pop_objs.max(axis=0) - pop_objs.min(axis=0)
                        V[i] = V0[i] * obj_range

                # --- Infill selection: select mu solutions for re-evaluation ---
                num_inactive_archive = _count_inactive(arc_objs[i], V0[i])
                theta_final = ((self.wmax + 1) / self.wmax) ** self.alpha

                new_decs = _kriging_select(
                    pop_decs, pop_objs, pop_mse, V[i], V0[i],
                    num_inactive_archive, 0.05 * n_per_task[i], self.mu, theta_final
                )

                # Remove duplicates against all previously evaluated solutions
                new_decs = remove_duplicates(new_decs, decs[i])

                # Never exceed the remaining budget of this task
                remaining = max_nfes_per_task[i] - nfes_per_task[i]
                if new_decs.shape[0] > remaining:
                    new_decs = new_decs[:remaining]

                if new_decs.shape[0] > 0:
                    # Evaluate new solutions with expensive objective function
                    new_objs, _ = evaluation_single(problem, new_decs, i)

                    # Update cumulative dataset (A2 in MATLAB)
                    decs[i] = np.vstack([decs[i], new_decs])
                    objs[i] = np.vstack([objs[i], new_objs])

                    # Update training archive (A1 in MATLAB)
                    arc_decs[i], arc_objs[i] = _update_archive(
                        arc_decs[i], arc_objs[i], new_decs, new_objs,
                        V[i], self.mu, NI
                    )

                    nfes_per_task[i] += new_decs.shape[0]
                    pbar.update(new_decs.shape[0])

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


def _count_inactive(objs, V):
    """
    Count inactive reference vectors.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (N, M)
    V : np.ndarray
        Reference vectors, shape (NV, M)

    Returns
    -------
    num_inactive : int
        Number of inactive reference vectors
    """
    objs_translated = objs - objs.min(axis=0)
    angle = np.arccos(np.clip(1 - cdist(objs_translated, V, metric='cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)
    active = np.unique(associate)
    return V.shape[0] - len(active)


def _get_active_info(objs, V):
    """
    Get count of inactive vectors and indices of active vectors.

    Parameters
    ----------
    objs : np.ndarray
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
    objs_translated = objs - objs.min(axis=0)
    angle = np.arccos(np.clip(1 - cdist(objs_translated, V, metric='cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)
    active = np.unique(associate)
    num_inactive = V.shape[0] - len(active)
    return num_inactive, active


def _kriging_select(pop_decs, pop_objs, pop_mse, V, V0, num_v1, delta, mu, theta):
    """
    Select solutions for expensive re-evaluation (KrigingSelect in MATLAB).

    Switches between exploitation (APD-based) and exploration (uncertainty-based)
    depending on the change in inactive reference vectors.

    Parameters
    ----------
    pop_decs : np.ndarray
        Population decisions after surrogate optimization
    pop_objs : np.ndarray
        Predicted objectives, shape (N, M)
    pop_mse : np.ndarray
        Predicted MSE, shape (N, M)
    V : np.ndarray
        Adapted reference vectors
    V0 : np.ndarray
        Original reference vectors
    num_v1 : int
        Number of inactive vectors in archive (against V0)
    delta : float
        Threshold for switching between exploitation and exploration
    mu : int
        Number of solutions to select
    theta : float
        APD penalty parameter

    Returns
    -------
    selected_decs : np.ndarray
        Selected decision variables for re-evaluation
    """
    N, M = pop_objs.shape

    # Compute NumV2: inactive vectors in surrogate population against V0
    num_v2 = _count_inactive(pop_objs, V0)

    # Get active reference vectors against V (adapted)
    n_inactive_v, active_indices = _get_active_info(pop_objs, V)

    n_clusters = min(mu, len(active_indices))
    if n_clusters == 0:
        indices = np.random.choice(N, size=min(mu, N), replace=False)
        return pop_decs[indices]

    Va = V[active_indices]

    # Cluster active reference vectors
    if n_clusters >= len(active_indices):
        cluster_labels = np.arange(len(active_indices))
        n_clusters = len(active_indices)
    else:
        cluster_labels = kmeans_clustering(Va, n_clusters)

    # Translate objectives
    objs_translated = pop_objs - pop_objs.min(axis=0)

    # Compute gamma: smallest angle between each active vector and others.
    # MATLAB zeroes the diagonal of the cosine matrix before taking acos, so a
    # single active vector yields gamma = acos(0) = pi/2.
    cosine = 1 - cdist(Va, Va, metric='cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
    gamma = np.maximum(gamma, 1e-6)

    # Associate each solution to its nearest active reference vector
    angle = np.arccos(np.clip(1 - cdist(objs_translated, Va, metric='cosine'), -1, 1))
    associate = np.argmin(angle, axis=1)

    # Compute APD for all solutions
    apd = np.ones(N)
    for j in np.unique(associate):
        current = np.where(associate == j)[0]
        if len(current) > 0:
            apd[current] = (1 + M * theta * angle[current, j] / gamma[j]) * \
                           np.sqrt(np.sum(objs_translated[current] ** 2, axis=1))

    # Map solutions to clusters via their associated reference vector
    cindex = cluster_labels[associate]

    # Selection: exploitation or exploration based on inactive vector change
    flag = num_v2 - num_v1
    selected_indices = []

    for c in np.unique(cindex):
        current = np.where(cindex == c)[0]
        if len(current) == 0:
            continue

        if flag <= delta:
            # Exploitation: for each active vector in this cluster, find best APD solution,
            # then select the overall best among them
            t = np.unique(associate[current])
            solution_best = []
            for j in t:
                current_s = np.where(associate == j)[0]
                best_id = current_s[np.argmin(apd[current_s])]
                solution_best.append(best_id)
            solution_best = np.array(solution_best)
            best = solution_best[np.argmin(apd[solution_best])]
            selected_indices.append(best)
        else:
            # Exploration: select solution with highest mean MSE in this cluster
            uncertainty = np.mean(pop_mse[current], axis=1)
            best = current[np.argmax(uncertainty)]
            selected_indices.append(best)

    if len(selected_indices) == 0:
        indices = np.random.choice(N, size=min(mu, N), replace=False)
        return pop_decs[indices]

    return pop_decs[selected_indices]


def _update_archive(arc_decs, arc_objs, new_decs, new_objs, V, mu, NI):
    """
    Update training archive (UpdataArchive in MATLAB).

    Maintains archive size at approximately NI by keeping newly evaluated solutions
    and selecting diverse representatives from old solutions.

    Parameters
    ----------
    arc_decs : np.ndarray
        Current archive decision variables
    arc_objs : np.ndarray
        Current archive objective values
    new_decs : np.ndarray
        Newly evaluated decision variables
    new_objs : np.ndarray
        Newly evaluated objective values
    V : np.ndarray
        Adapted reference vectors
    mu : int
        Number of newly evaluated solutions
    NI : int
        Target archive size

    Returns
    -------
    updated_decs : np.ndarray
        Updated archive decision variables
    updated_objs : np.ndarray
        Updated archive objective values

    Notes
    -----
    Faithful port of PlatEMO's ``UpdataArchive.m``. In the first branch PlatEMO
    clusters the *inactive reference vectors* ``Via`` but then uses the
    within-cluster member positions as indices into ``Total`` (the solutions).
    That index mixing is reproduced here so the archive contents match the
    reference implementation.
    """
    # MATLAB: All = [A1.decs;New.decs]; [~,index] = unique(All,'rows');
    # numpy's unique(axis=0) also returns lexicographically sorted rows with the
    # index of the first occurrence, matching MATLAB's default behaviour.
    all_decs = np.vstack([arc_decs, new_decs])
    all_objs = np.vstack([arc_objs, new_objs])
    _, unique_idx = np.unique(all_decs, axis=0, return_index=True)
    total_decs = all_decs[unique_idx]
    total_objs = all_objs[unique_idx]

    if total_decs.shape[0] <= NI:
        return total_decs, total_objs

    # Reference vectors that are NOT associated with any newly evaluated solution
    _, active = _get_active_info(new_objs, V)
    inactive = np.setdiff1d(np.arange(V.shape[0]), active)
    Vi = V[inactive]

    # Remove the newly evaluated solutions from Total (they are appended later)
    keep = ~_rows_in(total_decs, new_decs)
    total_decs = total_decs[keep]
    total_objs = total_objs[keep]

    K = NI - mu
    if total_decs.shape[0] <= K:
        return np.vstack([total_decs, new_decs]), np.vstack([total_objs, new_objs])

    n_via = 0
    if Vi.shape[0] > 0:
        objs_translated = total_objs - total_objs.min(axis=0)
        angle = np.arccos(np.clip(1 - cdist(objs_translated, Vi, metric='cosine'), -1, 1))
        associate = np.argmin(angle, axis=1)
        Via = Vi[np.unique(associate)]
        n_via = Via.shape[0]

    if n_via > K:
        # Cluster the inactive reference vectors
        labels = kmeans_clustering(Via, K)
    else:
        # Cluster the solutions when too few active reference vectors remain
        labels = kmeans_clustering(total_objs, K)

    selected = []
    for c in np.unique(labels):
        current = np.where(labels == c)[0]
        best = np.random.randint(len(current)) if len(current) > 1 else 0
        selected.append(current[best])
    selected = np.array(selected, dtype=int)
    selected = selected[selected < total_decs.shape[0]]

    return (np.vstack([total_decs[selected], new_decs]),
            np.vstack([total_objs[selected], new_objs]))


def _rows_in(decs, ref_decs):
    """
    Boolean mask of rows of ``decs`` that occur exactly in ``ref_decs``.

    Equivalent to MATLAB's ``ismember(decs, ref_decs, 'rows')``.
    """
    if decs.shape[0] == 0 or ref_decs.shape[0] == 0:
        return np.zeros(decs.shape[0], dtype=bool)
    ref = {row.tobytes() for row in np.ascontiguousarray(ref_decs)}
    return np.array([row.tobytes() in ref for row in np.ascontiguousarray(decs)], dtype=bool)

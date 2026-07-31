"""
Expected Direction-based Hypervolume Improvement (DirHV-EI)

This module implements DirHV-EI for parallel expensive multi/many-objective optimization.
It uses GP surrogates with a MOEA/D-GR framework to maximize direction-based hypervolume
expected improvement, and a greedy batch selection strategy.

References
----------
    [1] L. Zhao and Q. Zhang. Hypervolume-guided decomposition for parallel expensive multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2024, 28(2): 432-444.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.16
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc
from scipy.spatial.distance import pdist, squareform
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
import warnings

warnings.filterwarnings("ignore")


class DirHV_EI:
    """
    Expected Direction-based Hypervolume Improvement for parallel expensive
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

    def __init__(self, problem, n_initial=None, max_nfes=None, batch_size=5,
                 save_data=True, save_path='./Data', name='DirHV-EI', disable_tqdm=True):
        """
        Initialize DirHV-EI algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        batch_size : int, optional
            Number of true function evaluations per iteration (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'DirHV-EI')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.batch_size = batch_size
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the DirHV-EI algorithm.

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
                m = n_objs[i]
                dim = dims[i]

                # Determine actual batch size (don't exceed remaining budget)
                remaining = max_nfes_per_task[i] - nfes_per_task[i]
                actual_batch = min(self.batch_size, remaining)

                # Scale objectives to [0, 1]
                ori_objs = objs[i]
                ymin = ori_objs.min(axis=0)
                ymax = ori_objs.max(axis=0)
                yrange = ymax - ymin
                yrange[yrange < 1e-12] = 1e-12
                train_y = (ori_objs - ymin) / yrange

                # Non-dominated solutions (scaled)
                front_no, _ = nd_sort(train_y, 1)
                train_y_nds = train_y[front_no == 1]

                # Build GP models for each objective on scaled data
                models = mo_gp_build(decs[i], train_y, data_type)

                # Optimize DirHV-EI and select batch of candidate points
                new_decs = _opt_dirhvei(
                    m, dim, models, train_y_nds, actual_batch, data_type
                )

                # Remove duplicates, then top up so that exactly `actual_batch`
                # points are evaluated per cycle (the greedy subset selection
                # may return the same candidate index more than once)
                new_decs = remove_duplicates(new_decs, decs[i])
                if new_decs.shape[0] > actual_batch:
                    new_decs = new_decs[:actual_batch]
                while new_decs.shape[0] < actual_batch:
                    pad = np.random.rand(actual_batch - new_decs.shape[0], dim)
                    pad = remove_duplicates(pad, np.vstack([decs[i], new_decs])
                                            if new_decs.shape[0] > 0 else decs[i])
                    if pad.shape[0] == 0:
                        continue
                    new_decs = np.vstack([new_decs, pad]) if new_decs.shape[0] > 0 else pad

                # Expensive evaluation
                new_objs, _ = evaluation_single(problem, new_decs, i)

                # Update dataset
                decs[i] = np.vstack([decs[i], new_decs])
                objs[i] = np.vstack([objs[i], new_objs])

                nfes_per_task[i] += new_decs.shape[0]
                pbar.update(new_decs.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.batch_size)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results


# ==================== Core DirHV-EI Functions ====================

def _lhs(n, d):
    """Latin hypercube sample in [0, 1]^d (MATLAB ``lhsdesign`` equivalent)."""
    return qmc.LatinHypercube(d=d).random(n=n)


def _opt_dirhvei(M, D, models, train_y_nds, batch_size, data_type):
    """
    Maximize DirHV-EI and select a batch of query points.

    Parameters
    ----------
    M : int
        Number of objectives
    D : int
        Number of decision variables
    models : list
        List of trained GP models (one per objective)
    train_y_nds : np.ndarray
        Scaled non-dominated objective values, shape (n_nds, M)
    batch_size : int
        Number of points to select
    data_type : torch.dtype
        Data type for GP prediction

    Returns
    -------
    new_x : np.ndarray
        Selected candidate points, shape (batch_size, D)
    """
    # Generate reference vectors
    num_weights = [200, 210, 295, 456, 462]
    if M <= 3:
        ref_vecs, _ = uniform_point(num_weights[M - 2], M)
    elif M <= 6:
        ref_vecs, _ = uniform_point(num_weights[M - 2], M, method='ILD')
    else:
        ref_vecs, _ = uniform_point(500, M)

    # Utopian point
    z = -0.01 * np.ones(M)

    # Calculate intersection points and direction vectors
    xis, dir_vecs = _get_xis(train_y_nds, ref_vecs, z)

    # Use MOEA/D-GR to maximize DirHV-EI
    candidate_x, candidate_mean, candidate_std = _moead_gr(
        D, models, dir_vecs, xis, data_type
    )

    # Discard duplicate candidates
    _, unique_idx = np.unique(candidate_x, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    candidate_x = candidate_x[unique_idx]
    candidate_mean = candidate_mean[unique_idx]
    candidate_std = candidate_std[unique_idx]

    # Compute DirHV-EI for all candidates against all directions
    n_candidates = candidate_x.shape[0]
    n_dirs = dir_vecs.shape[0]
    DirHVEIs = np.zeros((n_candidates, n_dirs))
    for j in range(n_candidates):
        u_rep = np.tile(candidate_mean[j], (n_dirs, 1))
        s_rep = np.tile(candidate_std[j], (n_dirs, 1))
        DirHVEIs[j, :] = _get_dirhvei(u_rep, s_rep, xis)

    # Greedy batch selection
    actual_batch = min(batch_size, n_candidates)
    Qb = _subset_selection(DirHVEIs, actual_batch)
    new_x = candidate_x[Qb]

    return new_x


def _get_xis(train_y_nds, ref_vecs, z):
    """
    Calculate intersection points and direction vectors.

    Parameters
    ----------
    train_y_nds : np.ndarray
        Non-dominated objectives (scaled), shape (n, M)
    ref_vecs : np.ndarray
        Reference vectors, shape (N, M)
    z : np.ndarray
        Utopian point, shape (M,)

    Returns
    -------
    xis : np.ndarray
        Intersection points, shape (N, M)
    dir_vecs : np.ndarray
        Direction vectors (unit L2 norm), shape (N, M)
    """
    M = ref_vecs.shape[1]

    # Eq. 25: direction vectors with unit L2 norm
    temp = 1.1 * ref_vecs - z
    norm_temp = np.sqrt(np.sum(temp ** 2, axis=1, keepdims=True))
    dir_vecs = temp / norm_temp

    # Eq. 11: intersection points
    div_dir = 1.0 / dir_vecs  # (N, M)
    train_y_translated = train_y_nds - z  # (n, M)

    # G[i,k] = max_j ( (train_y_translated[k,j]) / dir_vecs[i,j] )
    G = div_dir[:, 0:1] * train_y_translated[:, 0:1].T  # (N, n)
    for j in range(1, M):
        G = np.maximum(G, div_dir[:, j:j + 1] * train_y_translated[:, j:j + 1].T)

    # Minimum of mTch for each direction vector
    Lmin = G.min(axis=1, keepdims=True)  # (N, 1)

    # Intersection points
    xis = z + Lmin * dir_vecs  # (N, M)

    return xis, dir_vecs


def _moead_gr(D, models, dir_vecs, xis, data_type):
    """
    MOEA/D with Global Replacement to maximize DirHV-EI over direction vectors.

    Parameters
    ----------
    D : int
        Number of decision variables
    models : list
        List of GP models
    dir_vecs : np.ndarray
        Direction vectors, shape (N, M)
    xis : np.ndarray
        Intersection points, shape (N, M)
    data_type : torch.dtype
        Data type for GP prediction

    Returns
    -------
    pop_x : np.ndarray
        Candidate solutions, shape (N, D)
    pop_mean : np.ndarray
        GP predicted means, shape (N, M)
    pop_std : np.ndarray
        GP predicted standard deviations, shape (N, M)
    """
    max_iter = 50
    pop_size = dir_vecs.shape[0]

    # Neighbourhood
    T = max(2, int(np.ceil(pop_size / 10)))
    dist_mat = squareform(pdist(dir_vecs))
    B = np.argsort(dist_mat, axis=1)[:, :T]

    # Initial population via LHS in [0, 1]^D (lhsdesign in MATLAB)
    pop_x = _lhs(pop_size, D)

    # GP prediction for initial population
    pop_mean, pop_std = _gp_evaluate(pop_x, models, data_type)

    # DirHV-EI for each individual on its own direction
    pop_dirhvei = _get_dirhvei(pop_mean, pop_std, xis)

    # Main optimization loop
    for gen in range(max_iter - 1):
        for i in range(pop_size):
            # Select parents
            if np.random.rand() < 0.8:
                P = B[i, np.random.permutation(B.shape[1])]
            else:
                P = np.random.permutation(pop_size)

            # Generate offspring via DE + polynomial mutation
            off_x = _operator_de(pop_x[i:i + 1], pop_x[P[0]:P[0] + 1],
                                 pop_x[P[1]:P[1] + 1])
            off_x = off_x.reshape(1, -1)

            # GP prediction for offspring
            off_mean, off_std = _gp_evaluate(off_x, models, data_type)

            # Compute DirHV-EI of offspring for all directions
            off_mean_rep = np.tile(off_mean, (pop_size, 1))
            off_std_rep = np.tile(off_std, (pop_size, 1))
            off_dirhveis = _get_dirhvei(off_mean_rep, off_std_rep, xis)

            # Global Replacement: find best matching subproblem
            best_index = np.argmax(off_dirhveis)

            # Replacement neighbourhood
            P_replace = B[best_index]
            # Update solutions where offspring is better
            improve_mask = pop_dirhvei[P_replace] < off_dirhveis[P_replace]
            off_indices = P_replace[improve_mask]

            if len(off_indices) > 0:
                pop_x[off_indices] = off_x
                pop_mean[off_indices] = off_mean
                pop_std[off_indices] = off_std
                pop_dirhvei[off_indices] = off_dirhveis[off_indices]

    return pop_x, pop_mean, pop_std


def _gp_evaluate(X, models, data_type):
    """
    Predict GP posterior mean and std for candidate solutions.

    gp_predict returns original objectives (it negates internally to undo the
    training-time negation). We use those directly as the GP mean predictions.

    Parameters
    ----------
    X : np.ndarray
        Candidate solutions, shape (N, D)
    models : list
        List of GP models
    data_type : torch.dtype
        Data type

    Returns
    -------
    u : np.ndarray
        Predicted means, shape (N, M)
    s : np.ndarray
        Predicted standard deviations, shape (N, M)
    """
    from ddmtolab.Methods.Algo_Methods.bo_utils import gp_predict

    N = X.shape[0]
    M = len(models)
    u = np.zeros((N, M))
    s = np.zeros((N, M))

    for j in range(M):
        pred_obj, pred_std = gp_predict(models[j], X, data_type)
        # gp_predict already returns original objectives (negated back)
        u[:, j] = pred_obj.flatten()
        s[:, j] = pred_std.flatten()

    # Ensure non-negative variance
    s = np.maximum(s, 0.0)

    return u, s


def _get_dirhvei(u, sigma, xis):
    """
    Calculate direction-based hypervolume expected improvement (Eq. 23).

    Parameters
    ----------
    u : np.ndarray
        Predicted means, shape (N, M)
    sigma : np.ndarray
        Predicted standard deviations, shape (N, M)
    xis : np.ndarray
        Intersection points, shape (N, M)

    Returns
    -------
    dirhvei : np.ndarray
        DirHV-EI values, shape (N,)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)

    xi_minus_u = xis - u
    tau = xi_minus_u / sigma_safe

    # Precompute normal CDF and PDF
    normcdf_tau = norm.cdf(tau)
    normpdf_tau = norm.pdf(tau)

    temp = xi_minus_u * normcdf_tau + sigma_safe * normpdf_tau  # (N, M)
    dirhvei = np.prod(temp, axis=1)  # (N,)

    return dirhvei


def _subset_selection(DirHVEIs, batch_size):
    """
    Greedy batch selection via submodularity (Algorithm 3).

    Parameters
    ----------
    DirHVEIs : np.ndarray
        DirHV-EI matrix, shape (L, N) where L = candidates, N = directions
    batch_size : int
        Number of points to select

    Returns
    -------
    Qb : list
        Selected candidate indices
    """
    L, N = DirHVEIs.shape
    Qb = []
    temp = DirHVEIs.copy()
    beta = np.zeros(N)

    for _ in range(batch_size):
        # Select candidate with max sum of DirHV-EI across all directions
        sums = temp.sum(axis=1)
        index = np.argmax(sums)
        Qb.append(index)
        beta += temp[index]

        # Update: [EI_D(x|lambda) - beta]_+
        temp = DirHVEIs - beta
        temp[temp < 0] = 0

    return Qb


def _operator_de(parent1, parent2, parent3):
    """
    Differential Evolution operator with polynomial mutation.

    Operates in [0, 1]^D space (DDMTOLab convention).

    Parameters
    ----------
    parent1, parent2, parent3 : np.ndarray
        Parent solutions, shape (1, D)

    Returns
    -------
    offspring : np.ndarray
        Generated offspring, shape (1, D)
    """
    CR, F, proM, disM = 1.0, 0.5, 1.0, 20.0
    N, D = parent1.shape

    # DE mutation
    site = np.random.rand(N, D) < CR
    offspring = parent1.copy()
    offspring[site] = offspring[site] + F * (parent2[site] - parent3[site])

    # Clip to bounds [0, 1]
    offspring = np.clip(offspring, 0, 1)

    # Polynomial mutation
    site = np.random.rand(N, D) < proM / D
    mu = np.random.rand(N, D)

    temp = site & (mu <= 0.5)
    if np.any(temp):
        val = 2.0 * mu[temp] + (1.0 - 2.0 * mu[temp]) * \
              (1.0 - offspring[temp]) ** (disM + 1)
        delta = val ** (1.0 / (disM + 1)) - 1.0
        offspring[temp] = offspring[temp] + delta

    temp = site & (mu > 0.5)
    if np.any(temp):
        val = 2.0 * (1.0 - mu[temp]) + 2.0 * (mu[temp] - 0.5) * \
              (1.0 - (1.0 - offspring[temp])) ** (disM + 1)
        delta = 1.0 - val ** (1.0 / (disM + 1))
        offspring[temp] = offspring[temp] + delta

    offspring = np.clip(offspring, 0, 1)

    return offspring

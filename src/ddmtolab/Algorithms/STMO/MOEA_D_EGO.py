"""
MOEA/D with Efficient Global Optimization (MOEA/D-EGO)

This module implements MOEA/D-EGO for computationally expensive multi-objective optimization.
It uses clustering-based Gaussian Process models, MOEA/D-DE to maximize Expected Tchebycheff
Improvement (ETI) with Moment Matching Approximation, and K-means batch selection.

References
----------
    [1] Q. Zhang, W. Liu, E. Tsang, and B. Virginas. Expensive multiobjective optimization by MOEA/D with Gaussian process model. IEEE Transactions on Evolutionary Computation, 2010, 14(3): 456-474.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
import warnings

warnings.filterwarnings("ignore")


class MOEA_D_EGO:
    """
    MOEA/D with Efficient Global Optimization for expensive
    multi-objective optimization.
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
                 save_data=True, save_path='./Data', name='MOEA-D-EGO', disable_tqdm=True):
        """
        Initialize MOEA/D-EGO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        batch_size : int, optional
            Number of true evaluations per iteration (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MOEA-D-EGO')
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
        Execute the MOEA/D-EGO algorithm.

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

        n_initial_per_task = par_list(
            self.n_initial if self.n_initial is not None else [11 * d - 1 for d in dims], nt)
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
                M = n_objs[i]
                D = dims[i]
                remaining = max_nfes_per_task[i] - nfes_per_task[i]
                actual_batch = min(self.batch_size, remaining)

                # Build clustering-based GP models
                try:
                    models, centers = _build_gp_models_fcm(decs[i], objs[i])
                except Exception:
                    new_dec = np.random.rand(actual_batch, D)
                    new_obj, _ = evaluation_single(problem, new_dec, i)
                    decs[i] = np.vstack([decs[i], new_dec])
                    objs[i] = np.vstack([objs[i], new_obj])
                    nfes_per_task[i] += actual_batch
                    pbar.update(actual_batch)
                    continue

                # Optimize ETI and select batch of candidate points
                new_x = _opt_eti_fcm(M, D, actual_batch, decs[i], objs[i], models, centers)

                # Remove duplicates, then top up so that exactly `actual_batch`
                # points are evaluated per cycle (MATLAB always returns
                # Batch_size query points)
                new_x = remove_duplicates(new_x, decs[i])
                if new_x.shape[0] > actual_batch:
                    new_x = new_x[:actual_batch]
                while new_x.shape[0] < actual_batch:
                    pad = np.random.rand(actual_batch - new_x.shape[0], D)
                    pad = remove_duplicates(pad, np.vstack([decs[i], new_x])
                                            if new_x.shape[0] > 0 else decs[i])
                    if pad.shape[0] == 0:
                        continue
                    new_x = np.vstack([new_x, pad]) if new_x.shape[0] > 0 else pad

                # Expensive evaluation
                new_obj, _ = evaluation_single(problem, new_x, i)
                decs[i] = np.vstack([decs[i], new_x])
                objs[i] = np.vstack([objs[i], new_obj])
                nfes_per_task[i] += new_x.shape[0]
                pbar.update(new_x.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.batch_size)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# GP Model Building with Clustering
# =============================================================================

_L1 = 80  # Max number of training points per GP model
_L2 = 20  # Increment of archive size that adds one more cluster


def _lhs(n, d):
    """Latin hypercube sample in [0, 1]^d (MATLAB ``lhsdesign`` equivalent)."""
    return qmc.LatinHypercube(d=d).random(n=n)


def _fcm(data, cluster_n, expo=2.0, max_iter=100, min_impro=0.05):
    """Fuzzy c-means clustering.

    Faithful port of the ``FCM`` helper embedded in PlatEMO's ``GPmodelFCM.m``,
    invoked there as ``FCM(train_x, csize, [2 NaN 0.05 false])`` i.e. fuzzifier
    ``expo = 2``, default ``max_iter = 100``, minimum improvement ``0.05``.

    Parameters
    ----------
    data : np.ndarray
        Data to cluster, shape (n, d)
    cluster_n : int
        Number of clusters
    expo : float, optional
        Exponent for the fuzzy partition matrix (default: 2.0)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    min_impro : float, optional
        Minimum improvement of the objective before stopping (default: 0.05)

    Returns
    -------
    center : np.ndarray
        Cluster centers, shape (cluster_n, d)
    """
    data_n = data.shape[0]

    # initfcm: random partition matrix whose columns sum to one
    U = np.random.rand(cluster_n, data_n)
    U = U / U.sum(axis=0, keepdims=True)

    center = np.tile(data.mean(axis=0), (cluster_n, 1))
    obj_prev = None

    for _ in range(max_iter):
        # stepfcm
        mf = U ** expo
        center = (mf @ data) / mf.sum(axis=1, keepdims=True)
        dist = cdist(center, data)  # distfcm
        obj_fcn = np.sum((dist ** 2) * mf)

        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = dist ** (-2.0 / (expo - 1.0))
        # Singularity guard (a data point coinciding with a center)
        if not np.all(np.isfinite(tmp)):
            bad = ~np.isfinite(tmp)
            tmp[bad] = 1.0
            tmp[:, np.any(bad, axis=0)] = 0.0
            tmp[bad] = 1.0
        U = tmp / tmp.sum(axis=0, keepdims=True)

        if obj_prev is not None and abs(obj_fcn - obj_prev) < min_impro:
            break
        obj_prev = obj_fcn

    return center


def _build_gp_models_fcm(train_x, train_y, L1=_L1, L2=_L2):
    """Build GP models with fuzzy c-means clustering (GPmodelFCM.m).

    When K <= L1 all evaluated points are used directly for a single set of M
    GP models. Otherwise ``csize = 1 + ceil((K-L1)/L2)`` clusters are formed and
    for every cluster the L1 nearest training points are used to build M models.

    Returns
    -------
    models : list of list
        models[cluster_i][obj_j] is a GP model
    centers : np.ndarray
        Cluster centers, shape (csize, D)
    """
    K, M = train_y.shape

    if K <= L1:
        # All the K evaluated points are directly used for building GP models
        centers = _fcm(train_x, 1)
        cluster_models = []
        for j in range(M):
            model = gp_build(train_x, train_y[:, j:j + 1])
            cluster_models.append(model)
        return [cluster_models], centers
    else:
        # FuzzyCM
        csize = 1 + int(np.ceil((K - L1) / L2))
        centers = _fcm(train_x, csize)

        # For each cluster, use the L1 nearest training points
        dis = cdist(train_x, centers)
        sorted_idx = np.argsort(dis, axis=0)  # (K, csize)

        models = []
        for ci in range(csize):
            temp_idx = sorted_idx[:L1, ci]
            cluster_models = []
            for j in range(M):
                model = gp_build(train_x[temp_idx], train_y[temp_idx, j:j + 1])
                cluster_models.append(model)
            models.append(cluster_models)

        return models, centers


def _gp_evaluate_fcm(X, models, centers):
    """Predict using cluster-based GP models.

    Returns
    -------
    u : np.ndarray, shape (N, M) - predicted means
    s : np.ndarray, shape (N, M) - predicted stds
    """
    N = X.shape[0]
    M = len(models[0])
    u = np.zeros((N, M))
    s = np.zeros((N, M))

    if len(models) == 1:
        for j in range(M):
            pred, std = gp_predict(models[0][j], X)
            u[:, j] = pred.flatten()
            s[:, j] = std.flatten()
        return u, np.maximum(s, 0)

    dis = cdist(X, centers)
    nearest = np.argmin(dis, axis=1)

    for ci in range(len(models)):
        mask = nearest == ci
        if not np.any(mask):
            continue
        X_ci = X[mask]
        for j in range(M):
            pred, std = gp_predict(models[ci][j], X_ci)
            u[mask, j] = pred.flatten()
            s[mask, j] = std.flatten()

    s = np.maximum(s, 0)
    return u, s


def _gp_evaluate_mean_fcm(X, models, centers):
    """Predict mean only using cluster-based GP models."""
    u, _ = _gp_evaluate_fcm(X, models, centers)
    return u


# =============================================================================
# Main Infill Procedure
# =============================================================================

def _opt_eti_fcm(M, D, batch_size, train_x, train_y, models, centers):
    """Maximize ETI using MOEA/D and select batch of candidate points."""
    # Generate reference vectors
    num_weights = {2: 200, 3: 210, 4: 295, 5: 456, 6: 462}
    if M <= 3:
        ref_vecs, _ = uniform_point(num_weights.get(M, 200), M, method='NBI')
    elif M <= 6:
        ref_vecs, _ = uniform_point(num_weights.get(M, 295), M, method='ILD')
    else:
        ref_vecs, _ = uniform_point(500, M)

    # Estimate utopian point z
    z = _get_estimation_z(D, models, centers, ref_vecs, train_y.min(axis=0))

    # Calculate gmin for each weight vector
    gmin = _get_gmin(train_y, ref_vecs, z)

    # Maximize ETI using MOEA/D-DE
    pop_eti, candidate_x, _, _ = _moead_eti(D, models, centers, ref_vecs, gmin, z)

    # Filter: remove similar candidates and those with ETI <= 0
    Q = []
    Q_eti = []
    temp = train_x.copy()
    for idx in range(candidate_x.shape[0]):
        if candidate_x.shape[0] > 0 and temp.shape[0] > 0:
            min_dist = np.min(cdist(candidate_x[idx:idx + 1], temp))
        else:
            min_dist = np.inf
        if min_dist > 1e-5 and pop_eti[idx] > 0:
            Q.append(candidate_x[idx])
            Q_eti.append(pop_eti[idx])
            temp = np.vstack([temp, candidate_x[idx:idx + 1]])

    if len(Q) > 0:
        Q = np.array(Q)
        Q_eti = np.array(Q_eti)
    else:
        Q = np.empty((0, D))
        Q_eti = np.empty(0)

    # K-means batch selection
    new_x = _kmeans_batch_select(Q, batch_size, candidate_x, Q_eti)
    return new_x


# =============================================================================
# Utopian Point Estimation
# =============================================================================

def _get_estimation_z(D, models, centers, ref_vecs, z_init):
    """Estimate utopian point using MOEA/D to minimize GP posterior mean."""
    delta = 0.9
    nr = 2
    max_iter = 100
    pop_size = ref_vecs.shape[0]

    # Neighborhood
    T = max(2, int(np.ceil(pop_size / 10)))
    B_dist = cdist(ref_vecs, ref_vecs)
    B = np.argsort(B_dist, axis=1)[:, :T]

    # Initial population (lhsdesign in MATLAB)
    pop_x = _lhs(pop_size, D)
    pop_mean = _gp_evaluate_mean_fcm(pop_x, models, centers)
    z = np.minimum(pop_mean.min(axis=0), z_init)

    for gen in range(max_iter - 1):
        for i in range(pop_size):
            if np.random.rand() < delta:
                P = B[i, np.random.permutation(T)]
            else:
                P = np.random.permutation(pop_size)

            # Generate offspring using DE
            off_x = _operator_de(pop_x[i], pop_x[P[0]], pop_x[P[1]])
            off_mean = _gp_evaluate_mean_fcm(off_x.reshape(1, -1), models, centers).flatten()

            # Update z
            z = np.minimum(z, off_mean)

            # Tchebycheff scalarization
            g_old = np.max((pop_mean[P] - z) * ref_vecs[P], axis=1)
            g_new = np.max(np.tile(off_mean - z, (len(P), 1)) * ref_vecs[P], axis=1)

            # Replace at most nr neighbors
            improve_idx = np.where(g_old > g_new)[0][:nr]
            if len(improve_idx) > 0:
                replace_idx = P[improve_idx]
                pop_x[replace_idx] = off_x
                pop_mean[replace_idx] = off_mean

    return z


# =============================================================================
# ETI Maximization with MOEA/D-DE
# =============================================================================

def _get_gmin(objs, ref_vecs, z):
    """Calculate minimum Tchebycheff value for each weight vector."""
    translated = objs - z  # (N, M)
    G = ref_vecs[:, 0:1] * translated[:, 0:1].T  # (NW, N)
    for j in range(1, ref_vecs.shape[1]):
        G = np.maximum(G, ref_vecs[:, j:j + 1] * translated[:, j:j + 1].T)
    return G.min(axis=1)  # (NW,)


def _moead_eti(D, models, centers, ref_vecs, gmin, z):
    """Use MOEA/D-DE to maximize ETI for all subproblems."""
    delta = 0.9
    nr = 2
    max_iter = 50
    pop_size = ref_vecs.shape[0]

    # Neighborhood
    T = max(2, int(np.ceil(pop_size / 10)))
    B_dist = cdist(ref_vecs, ref_vecs)
    B = np.argsort(B_dist, axis=1)[:, :T]

    # Initial population (lhsdesign in MATLAB)
    pop_x = _lhs(pop_size, D)
    pop_mean, pop_std = _gp_evaluate_fcm(pop_x, models, centers)
    pop_eti = _get_eti(pop_mean, pop_std, ref_vecs, gmin, z)

    for gen in range(max_iter - 1):
        for i in range(pop_size):
            if np.random.rand() < delta:
                P = B[i, np.random.permutation(T)]
            else:
                P = np.random.permutation(pop_size)

            # Generate offspring using DE
            off_x = _operator_de(pop_x[i], pop_x[P[0]], pop_x[P[1]])
            off_mean, off_std = _gp_evaluate_fcm(off_x.reshape(1, -1), models, centers)
            off_mean = off_mean.flatten()
            off_std = off_std.flatten()

            # Compute ETI for offspring on neighbors' subproblems
            eti_new = _get_eti(
                np.tile(off_mean, (len(P), 1)),
                np.tile(off_std, (len(P), 1)),
                ref_vecs[P], gmin[P], z
            )

            # Replace at most nr neighbors where offspring has better ETI
            improve_idx = np.where(pop_eti[P] < eti_new)[0][:nr]
            if len(improve_idx) > 0:
                replace_idx = P[improve_idx]
                pop_x[replace_idx] = off_x
                pop_mean[replace_idx] = off_mean
                pop_std[replace_idx] = off_std
                pop_eti[replace_idx] = eti_new[improve_idx]

    return pop_eti, pop_x, pop_mean, pop_std


# =============================================================================
# Expected Tchebycheff Improvement (ETI)
# =============================================================================

def _get_eti(u, sigma, ref_vecs, gbest, z):
    """Compute Expected Tchebycheff Improvement.

    Uses Moment Matching Approximation to approximate max of weighted Gaussians.

    Parameters
    ----------
    u : (N, M) predicted means
    sigma : (N, M) predicted stds
    ref_vecs : (N, M) weight vectors
    gbest : (N,) best Tchebycheff values per subproblem
    z : (M,) utopian point
    """
    # Weighted Gaussian: g_j ~ N(w_j*(u_j-z_j), (w_j*sigma_j)^2)
    g_mu = ref_vecs * (u - z)  # (N, M)
    g_sig = ref_vecs * sigma  # (N, M)
    g_sig = np.maximum(g_sig, 0)
    g_sig2 = g_sig ** 2  # (N, M)

    # MMA: approximate max of M Gaussians recursively
    mma_mean = g_mu[:, 0].copy()
    mma_sigma2 = g_sig2[:, 0].copy()

    for j in range(1, g_mu.shape[1]):
        mu_pair = np.column_stack([mma_mean, g_mu[:, j]])
        sig2_pair = np.column_stack([mma_sigma2, g_sig2[:, j]])
        mma_mean, mma_sigma2 = _app_max_of_2_gaussian(mu_pair, sig2_pair)

    mma_std = np.sqrt(np.maximum(mma_sigma2, 1e-20))
    tau = (gbest - mma_mean) / mma_std

    # ETI = (gbest - mma_mean) * Phi(tau) + mma_std * phi(tau)
    eti = (gbest - mma_mean) * norm.cdf(tau) + mma_std * norm.pdf(tau)
    return eti


def _app_max_of_2_gaussian(mu, sig2):
    """Moment Matching Approximation for max of two Gaussians (Eq. 18 & 19).

    Parameters
    ----------
    mu : (N, 2) means
    sig2 : (N, 2) variances

    Returns
    -------
    y : (N,) approximated mean of max
    s2 : (N,) approximated variance of max
    """
    tau = np.sqrt(np.maximum(sig2[:, 0] + sig2[:, 1], 1e-20))
    alpha = (mu[:, 0] - mu[:, 1]) / tau

    phi_alpha = norm.pdf(alpha)
    Phi_alpha = norm.cdf(alpha)
    Phi_neg_alpha = norm.cdf(-alpha)

    # Eq. 18: mean of max
    y = mu[:, 0] * Phi_alpha + mu[:, 1] * Phi_neg_alpha + tau * phi_alpha

    # Eq. 19: second moment (corrected as per Appendix B)
    s2 = ((mu[:, 0] ** 2 + sig2[:, 0]) * Phi_alpha +
          (mu[:, 1] ** 2 + sig2[:, 1]) * Phi_neg_alpha +
          (mu[:, 0] + mu[:, 1]) * tau * phi_alpha)
    s2 = s2 - y ** 2
    s2 = np.maximum(s2, 0)

    return y, s2


# =============================================================================
# DE Operator
# =============================================================================

def _operator_de(parent1, parent2, parent3, CR=1.0, F=0.5, mum=20):
    """DE/rand/1/bin with polynomial mutation in [0,1] space."""
    D = len(parent1)

    # Differential evolution
    site = np.random.rand(D) < CR
    offspring = parent1.copy()
    offspring[site] = offspring[site] + F * (parent2[site] - parent3[site])

    # Polynomial mutation (prob = 1/D per dimension)
    site = np.random.rand(D) < 1.0 / D
    offspring = np.clip(offspring, 0, 1)
    mu = np.random.rand(D)
    for j in np.where(site)[0]:
        if mu[j] <= 0.5:
            offspring[j] = offspring[j] + (1 - 0) * (
                (2 * mu[j] + (1 - 2 * mu[j]) * (1 - offspring[j]) ** (mum + 1)) ** (1 / (mum + 1)) - 1)
        else:
            offspring[j] = offspring[j] + (1 - 0) * (
                1 - (2 * (1 - mu[j]) + 2 * (mu[j] - 0.5) * offspring[j] ** (mum + 1)) ** (1 / (mum + 1)))

    return np.clip(offspring, 0, 1)


# =============================================================================
# Batch Selection
# =============================================================================

def _kmeans_batch_select(Q, batch_size, candidate_x, Q_eti):
    """Select batch of solutions using K-means clustering on filtered candidates."""
    if Q.shape[0] == 0:
        # Fallback: random from candidates
        idx = np.random.choice(candidate_x.shape[0], size=min(batch_size, candidate_x.shape[0]),
                               replace=False)
        return candidate_x[idx]

    actual_batch = min(batch_size, Q.shape[0])

    try:
        kmeans = KMeans(n_clusters=actual_batch, n_init=10)
        labels = kmeans.fit_predict(Q)
    except Exception:
        idx = np.random.choice(Q.shape[0], size=actual_batch, replace=False)
        return Q[idx]

    selected = []
    for ci in range(actual_batch):
        cluster_idx = np.where(labels == ci)[0]
        if len(cluster_idx) > 0:
            best = cluster_idx[np.argmax(Q_eti[cluster_idx])]
            selected.append(best)

    result = Q[np.array(selected)] if selected else np.empty((0, Q.shape[1]))

    # Pad with random candidates if not enough
    if result.shape[0] < batch_size:
        n_pad = batch_size - result.shape[0]
        pad_idx = np.random.choice(candidate_x.shape[0], size=min(n_pad, candidate_x.shape[0]),
                                   replace=False)
        result = np.vstack([result, candidate_x[pad_idx]]) if result.shape[0] > 0 else candidate_x[
            pad_idx]

    return result

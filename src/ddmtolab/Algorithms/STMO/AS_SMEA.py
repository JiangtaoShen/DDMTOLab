"""
Surrogate-assisted Multi-objective Evolutionary Algorithm with Adaptive local
region Search (AS-SMEA)

This module implements AS-SMEA for high-dimensional expensive multi-objective
optimization. The decision space is partitioned into several hyper-ellipsoidal
local regions whose position and shape are carried by a covariance matrix
adaptation state. Each region is searched in parallel: local Gaussian processes
turn the expensive problem into K cheap approximate multi-objective problems,
one per acquisition function, each solved by NSGA-II. A multi-armed bandit then
decides which acquisition function's candidate set is worth the real
evaluations, and stagnating regions are restarted at a Thompson-sampled
hypervolume optimum far from the existing region centers.

References
----------
    [1] Q. Wang, H. Li, W. Zhang, Y. Zhang, D. Gong, F. Chu, A. W. Mohamed, and I. Muhammad. Surrogate-assisted evolutionary algorithm with adaptive local region search for high-dimensional expensive multi-objective optimization problems. Swarm and Evolutionary Computation, 2026, 100: 102232.

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
from scipy.spatial.distance import cdist
from scipy.stats import chi2, norm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
import warnings

warnings.filterwarnings("ignore")

#: The K acquisition functions of Eq. (7), each written so that lower is better
ACQUISITIONS = ('EI', 'UCB', 'TS', 'PI', 'PE')


class AS_SMEA:
    """
    Surrogate-assisted multi-objective evolutionary algorithm with adaptive local
    region search for high-dimensional expensive multi-objective optimization.
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

    def __init__(self, problem, n_initial=None, max_nfes=None,
                 n_regions=5, n_select=3, n=100, n_gen=50,
                 gamma=0.75, alpha=0.99735, kappa=2.0, shrinkage=0.1,
                 n_screen=20, n_fantasy=3, fantasy_size=10, n_hv_sample=4096,
                 restart_patience=3, n_restart_cand=2000,
                 muc=20, mum=20,
                 save_data=True, save_path='./Data', name='AS-SMEA', disable_tqdm=True):
        """
        Initialize AS-SMEA.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial LHS samples per task (default: min(300, max_nfes/2))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 509)
        n_regions : int, optional
            Number of local regions L (default: 5)
        n_select : int, optional
            Number of representative solutions really evaluated per region and
            iteration, S in Algorithm 4 (default: 3)
        n : int, optional
            Population size of the NSGA-II run on each approximate problem
            (default: 100)
        n_gen : int, optional
            Number of generations of that NSGA-II run (default: 50)
        gamma : float, optional
            Decay factor of the accumulated bandit reward, Eq. (8) (default: 0.75)
        alpha : float, optional
            Confidence level of the chi-square cutoff that closes the
            hyper-ellipsoidal region, Eq. (5) (default: 0.99735, the 3-sigma rule)
        kappa : float, optional
            Trade-off weight of the UCB acquisition function (default: 2.0)
        shrinkage : float, optional
            Weight of the identity matrix blended into the sample covariance of a
            region (default: 0.1)
        n_screen : int, optional
            Number of candidates kept per approximate problem before the HVKG
            search (default: 20)
        n_fantasy : int, optional
            Number of fantasy draws averaged by the HVKG criterion, N' in
            Eq. (10) (default: 3)
        fantasy_size : int, optional
            Size of the random subset V_i in Eq. (10) (default: 10)
        n_hv_sample : int, optional
            Number of Monte-Carlo samples used for the hypervolume of the bandit
            reward (default: 4096)
        restart_patience : int, optional
            Number of consecutive iterations without hypervolume improvement
            after which a region is restarted (default: 3)
        n_restart_cand : int, optional
            Size of the random pool scanned for the new region center in
            Algorithm 3 (default: 2000)
        muc : float, optional
            Distribution index of the SBX crossover (default: 20)
        mum : float, optional
            Distribution index of the polynomial mutation (default: 20)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'AS-SMEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)

        Notes
        -----
        The paper fixes L = 5, S = 3, gamma = 0.75, a 100/50 NSGA-II and
        FEs_max = 509 with min(Original, 300) initial samples. It leaves the
        restart condition, the covariance regularization and the cost of the
        HVKG search unspecified; ``restart_patience``, ``shrinkage``,
        ``n_screen``, ``n_fantasy`` and ``fantasy_size`` fill those gaps and are
        documented on the corresponding helpers.
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 509
        self.n_regions = n_regions
        self.n_select = n_select
        self.n = n
        self.n_gen = n_gen
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.shrinkage = shrinkage
        self.n_screen = n_screen
        self.n_fantasy = n_fantasy
        self.fantasy_size = fantasy_size
        self.n_hv_sample = n_hv_sample
        self.restart_patience = restart_patience
        self.n_restart_cand = n_restart_cand
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the AS-SMEA algorithm.

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
        data_type = torch.float

        max_nfes_per_task = par_list(self.max_nfes, nt)
        if self.n_initial is None:
            n_initial_per_task = [min(300, max(2 * n_objs[i] + 1, m // 2))
                                  for i, m in enumerate(max_nfes_per_task)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)

        # Initialization (Algorithm 1, line 3)
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Local region generation (Algorithm 1, line 4)
        regions = [_init_regions(decs[i], objs[i], self.n_regions, dims[i],
                                 self.alpha, self.shrinkage, len(ACQUISITIONS))
                   for i in range(nt)]

        # A task whose regions all run dry is retired; its budget stays unspent
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
                D = dims[i]
                M = n_objs[i]
                budget = max_nfes_per_task[i] - nfes_per_task[i]
                spent_before = nfes_per_task[i]

                for r, region in enumerate(regions[i]):
                    if budget <= 0:
                        break

                    members = region.members(decs[i])
                    reg_decs, reg_objs = decs[i][members], objs[i][members]

                    # Local surrogate models of the m objectives (line 7)
                    models = mo_gp_build(reg_decs, reg_objs, data_type)

                    # K approximate MOP models, each solved by NSGA-II (lines 8-9)
                    box = _HVBox(reg_objs, self.n_hv_sample)
                    cand_sets = []
                    for acq in ACQUISITIONS:
                        cand_sets.append(_solve_surrogate_mop(
                            models, acq, region, reg_decs, reg_objs,
                            self.n, self.n_gen, self.kappa, self.muc, self.mum,
                            data_type
                        ))

                    # MAB-guided adaptive solution selection (line 10, Algorithm 4)
                    new_decs = _mass_select(
                        cand_sets, models, reg_objs, box, region,
                        min(self.n_select, budget), self.gamma, self.n_screen,
                        self.n_fantasy, self.fantasy_size, data_type
                    )

                    new_decs = remove_duplicates(new_decs, decs[i])[:budget]
                    if new_decs.shape[0] == 0:
                        region.stall += 1
                    else:
                        # Real evaluation and archive update (line 11)
                        new_objs, _ = evaluation_single(problem, new_decs, i)
                        decs[i] = np.vstack([decs[i], new_decs])
                        objs[i] = np.vstack([objs[i], new_objs])
                        nfes_per_task[i] += new_decs.shape[0]
                        budget -= new_decs.shape[0]
                        pbar.update(new_decs.shape[0])

                        # Update the search distribution of the region (line 12)
                        region.update(np.vstack([reg_decs, new_decs]),
                                      np.vstack([reg_objs, new_objs]))

                        gain = box.increment(box.anchor(reg_objs)[1], new_objs)
                        region.stall = 0 if gain > 0 else region.stall + 1

                    # HV-based local region restart (lines 13-15, Algorithm 3)
                    if region.stall >= self.restart_patience and budget > 0:
                        centers = np.vstack([r.center for r in regions[i]])
                        fresh, used = _restart_region(
                            problem, i, decs, objs, centers, D, M, self.alpha,
                            self.shrinkage, len(ACQUISITIONS), self.n_restart_cand,
                            data_type
                        )
                        if fresh is not None:
                            # The new region replaces the stagnating one, which keeps
                            # the parallel search width at L; the paper states that a
                            # region is generated but not how many may coexist.
                            regions[i][r] = fresh
                            nfes_per_task[i] += used
                            budget -= used
                            pbar.update(used)

                if nfes_per_task[i] == spent_before:
                    # A whole sweep of the regions produced nothing new to
                    # evaluate; retire the task instead of spinning
                    exhausted[i] = True

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.n_select)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# CMA-based local region
# =============================================================================

class _Region:
    """
    One hyper-ellipsoidal local region carrying a CMA search distribution.

    The region is the set ``{x : Delta(x)^2 <= chi2_alpha(D)}`` of Eq. (5), where
    ``Delta`` is the Mahalanobis distance of Eq. (6) to ``N(m, sigma^2 C)``.

    Parameters
    ----------
    m : np.ndarray
        Mean vector of the search distribution, shape (D,)
    cov : np.ndarray
        Covariance of the initial sample set, shape (D, D)
    D : int
        Number of decision variables
    alpha : float
        Confidence level of the chi-square cutoff
    shrinkage : float
        Weight of the identity matrix blended into ``cov``
    n_arms : int
        Number of acquisition functions the bandit chooses between

    Notes
    -----
    The paper compares the Mahalanobis distance itself against the chi-square
    critical value. The squared distance is used here instead, which is the form
    that makes the cutoff the 3-sigma rule the paper appeals to.

    A local region holds far fewer points than there are variables, so its sample
    covariance is singular. It is blended with a scaled identity before use;
    without that the ellipsoid degenerates onto the affine hull of the region and
    the variation operators cannot leave it.
    """

    def __init__(self, m, cov, D, alpha, shrinkage, n_arms):
        self.dim = D
        self.radius = float(np.sqrt(chi2.ppf(alpha, D)))
        self.reward = np.zeros(n_arms)
        self.stall = 0

        self.m = np.asarray(m, dtype=float).ravel()
        self._set_covariance(cov, shrinkage)
        self.shrinkage = shrinkage

        # Standard CMA constants; only the rank-mu term and the evolution path of
        # Eqs. (1)-(3) are needed, so the shared cmaes_* helpers (which re-seed a
        # singular covariance to the identity) are not reused here.
        self.cm = 1.0
        self.cc = 4.0 / (D + 4.0)
        self.c1 = 2.0 / ((D + 1.3) ** 2 + 1.0)
        self.cmu = min(1.0 - self.c1, 2.0 / ((D + 2.0) ** 2))
        self.cs = 0.3
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt(1.0 / (D + 1.0)) - 1.0) + self.cs
        self.chi_n = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
        self.path = np.zeros(D)

    @property
    def center(self):
        """Mean vector of the search distribution."""
        return self.m

    def _set_covariance(self, cov, shrinkage):
        """Store a regularized covariance together with its eigendecomposition."""
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        scale = max(float(np.trace(cov)) / self.dim, 1e-12)
        cov = (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(self.dim)

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12 * max(float(eigvals.max()), 1e-12))

        self.sigma = float(np.sqrt(np.mean(eigvals)))
        self.C = cov / (self.sigma ** 2)
        self.B = eigvecs
        self.Dg = np.sqrt(eigvals) / self.sigma

    def sample(self, n):
        """
        Draw ``n`` points from the search distribution, clipped to the unit box.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        X : np.ndarray
            Samples, shape (n, D)
        """
        z = np.random.randn(n, self.dim)
        return np.clip(self.m + self.sigma * ((z * self.Dg) @ self.B.T), 0.0, 1.0)

    def distance(self, X):
        """
        Mahalanobis distance of Eq. (6).

        Parameters
        ----------
        X : np.ndarray
            Query points, shape (n, D)

        Returns
        -------
        delta : np.ndarray
            Distances, shape (n,)
        """
        z = ((np.atleast_2d(X) - self.m) @ self.B) / self.Dg
        return np.linalg.norm(z, axis=1) / self.sigma

    def repair(self, X):
        """
        Pull points outside the ellipsoid back onto its boundary.

        Parameters
        ----------
        X : np.ndarray
            Points to repair, shape (n, D)

        Returns
        -------
        X : np.ndarray
            Points inside the region and the unit box, shape (n, D)
        """
        X = np.atleast_2d(X).copy()
        delta = self.distance(X)
        outside = delta > self.radius
        if np.any(outside):
            shrink = (self.radius / delta[outside])[:, None]
            X[outside] = self.m + (X[outside] - self.m) * shrink
        return np.clip(X, 0.0, 1.0)

    def members(self, decs, min_size=None):
        """
        Indices of the evaluated solutions that fall inside the region.

        Parameters
        ----------
        decs : np.ndarray
            All evaluated decisions, shape (n, D)
        min_size : int, optional
            Minimum number of training points; the nearest solutions outside the
            region top the set up when it is too small to fit a model
            (default: ``D + 1`` capped at the archive size)

        Returns
        -------
        idx : np.ndarray
            Indices into ``decs``
        """
        if min_size is None:
            # D + 1 is the least that can pin down a trend in D dimensions; a
            # larger floor would drag in most of the archive and stop the region
            # from being local at all
            min_size = min(decs.shape[0], max(5, self.dim + 1))

        inside = np.where(self.distance(decs) <= self.radius)[0]
        if inside.shape[0] >= min_size:
            return inside

        order = np.argsort(np.linalg.norm(decs - self.m, axis=1), kind='stable')
        return order[:min_size]

    def update(self, decs, objs):
        """
        Move the search distribution towards the best solutions, Eqs. (1)-(3).

        Parameters
        ----------
        decs : np.ndarray
            Solutions of the region, shape (n, D)
        objs : np.ndarray
            Their objective values, shape (n, M)
        """
        n = decs.shape[0]
        if n < 2:
            return

        # x_{i:N} is ranked by non-dominated sorting, then by crowding distance
        front_no, _ = nd_sort(objs, n)
        crowd = crowding_distance(objs, front_no)
        order = np.lexsort((-crowd, front_no))

        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        best = decs[order[:mu]]

        m_old = self.m.copy()
        self.m = m_old + self.cm * (w @ (best - m_old))

        step = (self.m - m_old) / self.sigma
        self.path = (1 - self.cc) * self.path + np.sqrt(self.cc * (2 - self.cc)) * step

        y = (best - m_old) / self.sigma
        rank_mu = (y * w[:, None]).T @ y
        cov = (1 - self.c1 - self.cmu) * (self.sigma ** 2) * self.C \
            + self.cmu * (self.sigma ** 2) * rank_mu \
            + self.c1 * (self.sigma ** 2) * np.outer(self.path, self.path)

        # beta(t) of Eq. (3): the usual cumulative step-size adaptation
        beta = float(np.exp(self.cs / self.damps *
                            (np.linalg.norm(self.path) / self.chi_n - 1.0)))
        beta = float(np.clip(beta, 0.5, 2.0))

        self._set_covariance(cov, self.shrinkage)
        self.sigma *= beta


def _init_regions(decs, objs, n_regions, D, alpha, shrinkage, n_arms):
    """
    Local region initialization based on MaxMin distance (Algorithm 2).

    Parameters
    ----------
    decs : np.ndarray
        Evaluated decisions, shape (n, D)
    objs : np.ndarray
        Evaluated objectives, shape (n, M)
    n_regions : int
        Number of regions L
    D : int
        Number of decision variables
    alpha : float
        Confidence level of the chi-square cutoff
    shrinkage : float
        Weight of the identity matrix blended into a sample covariance
    n_arms : int
        Number of acquisition functions

    Returns
    -------
    regions : List[_Region]
    """
    front_no, _ = nd_sort(objs, decs.shape[0])
    ps0 = np.where(front_no == 1)[0]
    if ps0.shape[0] == 0:
        ps0 = np.arange(decs.shape[0])

    # The first center is the largest hypervolume contributor of PS0, the rest
    # are as far as possible from the centers already chosen, Eq. (4)
    contrib = _hv_contribution(objs[ps0])
    centers = [ps0[int(np.argmax(contrib))]]
    while len(centers) < min(n_regions, ps0.shape[0]):
        rest = np.setdiff1d(ps0, np.array(centers))
        d = cdist(decs[rest], decs[centers]).min(axis=1)
        centers.append(int(rest[int(np.argmax(d))]))

    delta = _neighbourhood_radius(decs)
    regions = []
    for c in centers:
        X = decs[np.linalg.norm(decs - decs[c], axis=1) <= delta]
        if X.shape[0] < 2:
            X = decs[np.argsort(np.linalg.norm(decs - decs[c], axis=1))[:2]]
        regions.append(_Region(X.mean(axis=0), np.cov(X, rowvar=False),
                               D, alpha, shrinkage, n_arms))

    # A degenerate PS0 can yield fewer centers than requested; clone the search
    # distribution around further samples so the parallel search keeps width L
    while len(regions) < n_regions:
        c = np.random.randint(decs.shape[0])
        X = decs[np.argsort(np.linalg.norm(decs - decs[c], axis=1))[:max(2, decs.shape[0] // 4)]]
        regions.append(_Region(X.mean(axis=0), np.cov(X, rowvar=False),
                               D, alpha, shrinkage, n_arms))
    return regions


def _neighbourhood_radius(decs, max_pairs=400):
    """
    Median pairwise distance of the dataset, the radius delta of Algorithm 2.

    Parameters
    ----------
    decs : np.ndarray
        Evaluated decisions, shape (n, D)
    max_pairs : int, optional
        Number of rows subsampled before the pairwise distances are formed
        (default: 400)

    Returns
    -------
    delta : float
    """
    X = decs
    if X.shape[0] > max_pairs:
        X = X[np.random.choice(X.shape[0], max_pairs, replace=False)]
    d = cdist(X, X)
    iu = np.triu_indices(X.shape[0], k=1)
    return float(np.median(d[iu])) if iu[0].size else 1.0


def _restart_region(problem, task, decs, objs, centers, D, M, alpha, shrinkage,
                    n_arms, n_cand, data_type):
    """
    HV-based local region restart (Algorithm 3).

    A global Gaussian process is Thompson-sampled for every objective, and the
    point maximizing the hypervolume scalarization ``S_w`` outside every existing
    region center becomes the center of a fresh region.

    Parameters
    ----------
    problem : MTOP
        Problem instance, used for the one real evaluation of the new center
    task : int
        Task index
    decs, objs : List[np.ndarray]
        Evaluated archive, updated in place with the new center
    centers : np.ndarray
        Centers of the existing regions, shape (L, D)
    D, M : int
        Numbers of decision variables and objectives
    alpha, shrinkage : float
        Region parameters
    n_arms : int
        Number of acquisition functions
    n_cand : int
        Size of the random pool scanned for the new center
    data_type : torch.dtype
        Tensor dtype used by the surrogate

    Returns
    -------
    region : _Region or None
        The new region, or None when no candidate lies outside every center
    used : int
        Number of real evaluations spent (1 on success)
    """
    models = mo_gp_build(decs[task], objs[task], data_type)

    pool = np.random.rand(n_cand, D)
    delta = _neighbourhood_radius(decs[task])
    far = np.all(cdist(pool, centers) > delta, axis=1)
    if not np.any(far):
        return None, 0
    pool = pool[far]

    # Thompson sampling of one posterior objective function per objective
    mu, mse = mo_gp_predict(models, pool, data_type, mse=True)
    sampled = mu + np.sqrt(np.maximum(mse, 0.0)) * np.random.randn(*mu.shape)

    # S_w[F] = min_i (max(f_i / w_i, 0))^m on the improvement over the nadir, so
    # that the scalarization is maximized on a minimization problem
    w = np.abs(np.random.randn(M))
    w = w / max(np.linalg.norm(w), 1e-12)
    gain = np.maximum(np.max(objs[task], axis=0) - sampled, 0.0)
    scalar = np.min(gain / np.maximum(w, 1e-12), axis=1) ** M

    x_new = pool[int(np.argmax(scalar))][None, :]
    y_new, _ = evaluation_single(problem, x_new, task)
    decs[task] = np.vstack([decs[task], x_new])
    objs[task] = np.vstack([objs[task], y_new])

    dist = np.linalg.norm(decs[task] - x_new, axis=1)
    X = decs[task][dist <= delta]
    if X.shape[0] < 2:
        X = decs[task][np.argsort(dist)[:2]]
    return _Region(X.mean(axis=0), np.cov(X, rowvar=False), D, alpha,
                   shrinkage, n_arms), 1


# =============================================================================
# Problem approximation and local region search
# =============================================================================

def _acquisition(name, mu, sigma, best, kappa):
    """
    Evaluate one acquisition function, written so that lower is better.

    Parameters
    ----------
    name : str
        One of ``ACQUISITIONS``
    mu : np.ndarray
        Posterior means, shape (n, M)
    sigma : np.ndarray
        Posterior standard deviations, shape (n, M)
    best : np.ndarray
        Best observed value of each objective in the region, shape (M,)
    kappa : float
        Trade-off weight of UCB

    Returns
    -------
    values : np.ndarray
        Acquisition values, shape (n, M)

    Notes
    -----
    The Thompson sample draws an independent normal deviate per point rather than
    one function from the posterior; the paper does not prescribe how TS is
    realized inside the surrogate problem.
    """
    sigma = np.maximum(sigma, 1e-12)
    if name == 'UCB':
        return mu - kappa * sigma
    if name == 'TS':
        return mu + sigma * np.random.randn(*mu.shape)
    if name == 'PE':
        return -sigma

    imp = best - mu
    z = imp / sigma
    if name == 'PI':
        return -norm.cdf(z)
    return -(imp * norm.cdf(z) + sigma * norm.pdf(z))     # EI


def _solve_surrogate_mop(models, acq, region, reg_decs, reg_objs,
                         n_pop, n_gen, kappa, muc, mum, data_type):
    """
    Solve one approximate MOP of Eq. (7) inside a region with NSGA-II.

    Parameters
    ----------
    models : list
        Local Gaussian processes, one per objective
    acq : str
        Acquisition function name
    region : _Region
        Region the search is confined to
    reg_decs, reg_objs : np.ndarray
        Evaluated solutions of the region
    n_pop : int
        Population size
    n_gen : int
        Number of generations
    kappa : float
        Trade-off weight of UCB
    muc, mum : float
        Distribution indices of the variation operators
    data_type : torch.dtype
        Tensor dtype used by the surrogate

    Returns
    -------
    candidates : np.ndarray
        Non-dominated set of the approximate problem, shape (n_nd, D)
    """
    best = np.min(reg_objs, axis=0)

    def surrogate(X):
        mu, mse = mo_gp_predict(models, X, data_type, mse=True)
        return _acquisition(acq, mu, np.sqrt(np.maximum(mse, 0.0)), best, kappa)

    # Seed with the solutions already evaluated in the region, then fill up from
    # its search distribution; listing them first keeps them in the population
    seed = region.repair(reg_decs)[:n_pop]
    pop = np.vstack([seed, region.sample(max(0, n_pop - seed.shape[0]))])
    pop_acq = surrogate(pop)

    for _ in range(n_gen):
        front_no, _ = nd_sort(pop_acq, pop.shape[0])
        crowd = crowding_distance(pop_acq, front_no)
        mating = tournament_selection(2, pop.shape[0], -front_no, crowd)

        offspring = region.repair(ga_generation(pop[mating], muc, mum))
        off_acq = surrogate(offspring)

        merged = np.vstack([pop, offspring])
        merged_acq = np.vstack([pop_acq, off_acq])
        keep = _nsga2_selection(merged_acq, n_pop)
        pop, pop_acq = merged[keep], merged_acq[keep]

    front_no, _ = nd_sort(pop_acq, pop.shape[0])
    return pop[front_no == 1]


def _nsga2_selection(objs, n):
    """
    NSGA-II environmental selection.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (n_total, M)
    n : int
        Target size

    Returns
    -------
    selected : np.ndarray
        Indices of the survivors
    """
    if objs.shape[0] <= n:
        return np.arange(objs.shape[0])
    front_no, _ = nd_sort(objs, n)
    crowd = crowding_distance(objs, front_no)
    return np.lexsort((-crowd, front_no))[:n]


# =============================================================================
# Hypervolume helpers
# =============================================================================

class _HVBox:
    """
    Reference point and Monte-Carlo sample shared by one region iteration.

    Eqs. (9) and (10) only ever ask what a handful of new points *add* to a fixed
    front, thousands of times per iteration. Fixing one sample set makes those
    differences consistent, and :meth:`anchor` reduces each later query to the
    samples the front leaves free, which is what keeps the criterion affordable.

    Parameters
    ----------
    objs : np.ndarray
        Objectives of the region, shape (n, M)
    n_sample : int
        Number of Monte-Carlo samples
    """

    def __init__(self, objs, n_sample):
        ideal = np.min(objs, axis=0)
        nadir = np.max(objs, axis=0)
        span = np.where(nadir - ideal <= 0, 1.0, nadir - ideal)
        self.ref = nadir + 0.1 * span
        self.samples = ideal + np.random.rand(n_sample, objs.shape[1]) * (self.ref - ideal)

    def _dominated(self, objs, samples):
        """Mask of the samples dominated by at least one point of a front."""
        if objs.shape[0] == 0 or samples.shape[0] == 0:
            return np.zeros(samples.shape[0], dtype=bool)
        inside = objs[np.all(objs <= self.ref, axis=1)]
        if inside.shape[0] == 0:
            return np.zeros(samples.shape[0], dtype=bool)
        return np.any(np.all(inside[None, :, :] <= samples[:, None, :], axis=2), axis=1)

    def value(self, objs):
        """
        Fraction of the box dominated by a front.

        Parameters
        ----------
        objs : np.ndarray
            Front, shape (n, M)

        Returns
        -------
        hv : float
            Value in [0, 1]
        """
        return float(np.mean(self._dominated(objs, self.samples)))

    def anchor(self, objs):
        """
        Value of a front together with the samples it leaves undominated.

        Parameters
        ----------
        objs : np.ndarray
            Front, shape (n, M)

        Returns
        -------
        hv : float
            Value of the front in [0, 1]
        free : np.ndarray
            Samples no point of the front dominates, shape (n_free, M)
        """
        dominated = self._dominated(objs, self.samples)
        return float(np.mean(dominated)), self.samples[~dominated]

    def increment(self, free, objs):
        """
        What a set of points adds to the front that left ``free`` undominated.

        Exact, because a sample is dominated by the union exactly when the front
        or the new points dominate it.

        Parameters
        ----------
        free : np.ndarray
            Samples returned by :meth:`anchor`, shape (n_free, M)
        objs : np.ndarray
            New points, shape (n, M)

        Returns
        -------
        gain : float
            Added fraction of the box, in [0, 1]
        """
        return float(np.sum(self._dominated(objs, free))) / self.samples.shape[0]


def _hv_contribution(objs):
    """
    Exclusive hypervolume contribution of every solution of a front.

    Parameters
    ----------
    objs : np.ndarray
        Front, shape (n, M)

    Returns
    -------
    contrib : np.ndarray
        Contributions, shape (n,)
    """
    n = objs.shape[0]
    if n == 1:
        return np.ones(1)
    box = _HVBox(objs, 4096)
    total = box.value(objs)
    return np.array([total - box.value(np.delete(objs, i, axis=0)) for i in range(n)])


# =============================================================================
# MAB-guided adaptive solution selection (Algorithm 4)
# =============================================================================

def _mass_select(cand_sets, models, reg_objs, box, region, n_select,
                 gamma, n_screen, n_fantasy, fantasy_size, data_type):
    """
    Pick the representative candidate set of the arm with the largest reward.

    Parameters
    ----------
    cand_sets : List[np.ndarray]
        One non-dominated set per acquisition function
    models : list
        Local Gaussian processes
    reg_objs : np.ndarray
        Objectives evaluated in the region, the current front Y_t
    box : _HVBox
        Hypervolume box of the region
    region : _Region
        Region whose accumulated rewards are updated in place
    n_select : int
        Number of solutions selected per arm, S
    gamma : float
        Decay factor of Eq. (8)
    n_screen : int
        Number of candidates kept per arm before the HVKG search
    n_fantasy : int
        Number of fantasy draws averaged by HVKG
    fantasy_size : int
        Size of the random subset V_i

    Returns
    -------
    best : np.ndarray
        Selected decisions, shape (<= n_select, D)
    """
    base_hv, free = box.anchor(reg_objs)
    picks, rewards = [], np.full(len(cand_sets), -np.inf)

    for j, X_C in enumerate(cand_sets):
        if X_C.shape[0] == 0:
            picks.append(np.zeros((0, region.dim)))
            continue

        X_B = _hvkg_select(X_C, models, box, free, n_select,
                           n_screen, n_fantasy, fantasy_size, data_type)
        picks.append(X_B)

        # R_{t+1}, the relative hypervolume improvement of Eq. (9)
        pred = mo_gp_predict(models, X_B, data_type)
        gain = box.increment(free, pred)
        rewards[j] = gain / base_hv if base_hv > 0 else gain

    finite = np.isfinite(rewards)
    if not np.any(finite):
        return np.zeros((0, region.dim))

    # h_{t+1} = gamma * h_t + R_{t+1}, Eq. (8)
    region.reward[finite] = gamma * region.reward[finite] + rewards[finite]
    order = np.argsort(-np.where(finite, region.reward, -np.inf), kind='stable')
    return picks[int(order[0])]


def _hvkg_select(X_C, models, box, free, n_select, n_screen,
                 n_fantasy, fantasy_size, data_type):
    """
    Greedy hypervolume knowledge-gradient selection, Eqs. (10) and (11).

    Parameters
    ----------
    X_C : np.ndarray
        Candidate set of one arm, shape (n, D)
    models : list
        Local Gaussian processes
    box : _HVBox
        Hypervolume box of the region
    free : np.ndarray
        Samples the current front Y_t leaves undominated, from
        :meth:`_HVBox.anchor`
    n_select : int
        Number of candidates to pick
    n_screen : int
        Number of candidates kept before the search
    n_fantasy : int
        Number of fantasy draws averaged per candidate, N'
    fantasy_size : int
        Size of the random subset V_i

    Returns
    -------
    X_B : np.ndarray
        Selected candidates, shape (<= n_select, D)

    Notes
    -----
    Eq. (10) conditions the models on ``(x, mu(x))``. For an exact Gaussian
    process that leaves the posterior mean unchanged everywhere, so the criterion
    would score every candidate identically. The usual knowledge-gradient
    fantasy ``y ~ N(mu(x), sigma(x)^2)`` is drawn instead, and the resulting
    posterior mean is obtained in closed form from the joint posterior over the
    candidate set, which avoids refitting a model per candidate.
    """
    # Keep a diverse screen of the candidate set: the criterion below costs
    # n_select * n_screen * n_fantasy hypervolume evaluations
    mu, mse = mo_gp_predict(models, X_C, data_type, mse=True)
    if X_C.shape[0] > n_screen:
        keep = _nsga2_selection(mu, n_screen)
        X_C, mu, mse = X_C[keep], mu[keep], mse[keep]

    n, M = mu.shape
    if n == 0:
        return X_C

    # Joint posterior covariance over the screened candidates, one per objective
    cov = np.stack([_posterior_covariance(models[j], X_C, data_type)
                    for j in range(M)], axis=0)
    sd = np.sqrt(np.maximum(np.stack([np.diag(cov[j]) for j in range(M)], axis=1), 1e-24))

    selected = []
    remaining = list(range(n))

    for _ in range(min(n_select, n)):
        scores = np.full(len(remaining), -np.inf)
        for pos, a in enumerate(remaining):
            others = [r for r in remaining if r != a]
            if not others:
                scores[pos] = 0.0
                continue
            gains = []
            for _ in range(n_fantasy):
                V = np.random.choice(others, size=min(fantasy_size, len(others)),
                                     replace=False)
                eps = np.random.randn(M)
                # Rank-one posterior mean update after observing (x_a, y_a)
                shift = cov[:, V, a].T * (eps / sd[a])
                # Only the gain over Y_t matters, so the fixed part never
                # has to be re-scanned
                gains.append(box.increment(free, mu[V] + shift))
            scores[pos] = float(np.mean(gains))

        pick = int(np.argmax(scores))
        selected.append(remaining.pop(pick))

    return X_C[np.array(selected, dtype=int)]


def _posterior_covariance(model, X, data_type):
    """
    Joint posterior covariance of one Gaussian process over a candidate set.

    Parameters
    ----------
    model : SingleTaskGP
        Trained model
    X : np.ndarray
        Query points, shape (n, D)
    data_type : torch.dtype
        Tensor dtype used by the surrogate

    Returns
    -------
    cov : np.ndarray
        Covariance matrix, shape (n, n)
    """
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(torch.tensor(X, dtype=data_type))
        cov = posterior.distribution.covariance_matrix.cpu().numpy()
    return np.atleast_2d(np.squeeze(cov))

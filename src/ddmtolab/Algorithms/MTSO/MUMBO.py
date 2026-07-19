"""
Multi-task Max-value Bayesian Optimization (MUMBO)

This module implements MUMBO for expensive multi-task optimization. The algorithm
uses an information-theoretic acquisition function based on mutual information
between candidate observations and the optimal objective value g*. A multi-task
Gaussian process (intrinsic coregionalization kernel, Matern 5/2 base) provides
cross-task knowledge transfer, and the MUMBO acquisition exploits the bivariate
predictive distribution between the target task and each candidate task through
the extended skew Gaussian (ESG) form of y | g < g*. The acquisition value is
divided by the evaluation cost of each task, selecting the evaluation with the
largest information gain per unit cost (paper Eq. 2).

Adaptation to multi-task optimization: the paper optimizes a single objective
with auxiliary information sources. Here every task must be optimized, so the
target task (z0 in the paper) rotates round-robin over the tasks with remaining
budget; all tasks with remaining budget act as candidate information sources at
every iteration. With equal costs the target mostly evaluates itself (rho is
maximal at the target, collapsing MUMBO to MES per paper Sec. 3.2) and transfer
happens through the shared coregionalized GP; a cost vector shifts evaluations
toward cheaper tasks via the alpha/cost ratio.

The GP is fitted on negated, per-task min-max normalized objectives
(mtgp_build convention), so the paper's maximization formulation applies
verbatim in the model space while the platform minimizes.

References
----------
    [1] Moss, Henry B., David S. Leslie, and Paul Rayson. "Mumbo: Multi-task
        max-value Bayesian optimization." Joint European Conference on Machine
        Learning and Knowledge Discovery in Databases. Springer, 2020.

    [2] Wang, Zi, and Stefanie Jegelka. "Max-value entropy search for efficient
        Bayesian optimization." ICML, 2017.

Notes
-----
Corrected against the paper source (arXiv:2006.12093):

- g* is sampled with the mean-field Gumbel method of [2] (product of posterior
  marginal CDFs over a random grid plus evaluated points, quantile-fitted
  Gumbel), not from a heuristic around the best posterior mean.
- Task feature uses integer encoding float(task_id) matching mtgp_build;
  fractional encodings collapse distinct tasks under BoTorch's integer cast.
- The target task rotates over tasks instead of being fixed to task 0.
- Acquisition candidates, g* samples (N=10) and Simpson resolution (61 points
  over mean +/- 4 std, paper Appendix A) use paper-scaled defaults.
- task_cost only weights the acquisition ratio; the per-task evaluation budget
  max_nfes keeps the platform's count-based semantics.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.07.17
Version: 3.0
"""
from tqdm import tqdm
import torch
import time
import numpy as np
from scipy.stats import norm
from scipy.integrate import simpson
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mtgp_build
import warnings

warnings.filterwarnings("ignore")


class MUMBO:
    """
    Multi-task Max-value Bayesian Optimization (MUMBO).

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements.
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, task_cost=None,
                 n_gstar_samples=10, n_candidates=None, n_quad=61,
                 n_grid=None, save_data=True, save_path='./Data', name='MUMBO',
                 disable_tqdm=True):
        """
        Initialize MUMBO.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance.
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50).
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100).
        task_cost : List[float] or None, optional
            Evaluation cost c(z) of each task. Defaults to equal costs
            [1, 1, ..., 1]; the next evaluation maximizes the information
            gain per unit cost alpha(x, z) / c(z) (paper Eq. 2). Costs only
            weight this selection ratio; budgets stay evaluation counts.
        n_gstar_samples : int, optional
            Number N of Monte Carlo g* samples per iteration (default: 10,
            the paper's robust MUMBO-10 setting).
        n_candidates : int, optional
            Acquisition candidates per task and iteration. Default None
            selects min(200 * d, 2000); 80% uniform random, 20% Gaussian
            perturbations around the task's best observed solutions.
        n_quad : int, optional
            Simpson integration points over mean +/- 4 std of the ESG
            (default: 61; forced odd).
        n_grid : int, optional
            Size of the random grid for mean-field g* sampling. Default None
            selects min(2000 * d, 20000) (paper uses 10000 * d; reduced for
            runtime, configurable). Evaluated target-task points are always
            appended to the grid.
        save_data : bool, optional
            Whether to save optimization data (default: True).
        save_path : str, optional
            Path to save results (default: './Data').
        name : str, optional
            Name for the experiment (default: 'MUMBO').
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True).
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.task_cost = task_cost
        self.n_gstar_samples = n_gstar_samples
        self.n_candidates = n_candidates
        self.n_quad = n_quad
        self.n_grid = n_grid
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute MUMBO.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives,
            and runtime.
        """
        data_type = torch.double
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Task evaluation costs c(z) (paper Eq. 2); default: equal costs
        if self.task_cost is not None:
            task_cost = np.asarray(self.task_cost, dtype=float).flatten()
            if task_cost.shape[0] != nt:
                raise ValueError(f"task_cost must have length {nt}, "
                                 f"got {task_cost.shape[0]}")
            if np.any(task_cost <= 0):
                raise ValueError("task_cost entries must be positive")
        else:
            task_cost = np.ones(nt, dtype=float)

        # Initialize samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        pbar = tqdm(total=sum(max_nfes_per_task),
                    initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        iteration = 0
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt)
                            if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Round-robin target task (the paper's z0) over unfinished tasks
            target_task = active_tasks[iteration % len(active_tasks)]
            iteration += 1

            # Build multi-task GP on per-task normalized objectives
            # (mtgp_build negates internally -> maximization model space)
            objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
            mtgp = mtgp_build(decs, objs_normalized, dims, data_type=data_type)

            # Sample N values of g* | D_n via the mean-field Gumbel method [2]
            g_samples = _sample_gstar(
                mtgp, decs, dims, target_task, data_type,
                n_gstar_samples=self.n_gstar_samples, n_grid=self.n_grid)

            # Select (task, x) maximizing alpha(x, z) / c(z)
            best_task, best_x = _select_next_point(
                mtgp, g_samples, target_task, active_tasks, decs, objs,
                dims, task_cost, data_type,
                n_candidates=self.n_candidates, n_quad=self.n_quad)

            # Evaluate on selected task
            candidate_np = best_x.reshape(1, -1)
            obj, _ = evaluation_single(problem, candidate_np, best_task)

            # Update data
            decs[best_task], objs[best_task] = vstack_groups(
                (decs[best_task], candidate_np),
                (objs[best_task], obj)
            )
            nfes_per_task[best_task] += 1
            pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data
        )

        return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _pad_to_max(x_np, max_dim):
    """Zero-pad decision matrix columns to max_dim (mtgp_build convention)."""
    if x_np.shape[1] < max_dim:
        pad = np.zeros((x_np.shape[0], max_dim - x_np.shape[1]))
        return np.hstack([x_np, pad])
    return x_np


def _with_task_column(x_np, task_id, data_type):
    """Append the integer-valued task feature column (mtgp_build convention)."""
    x_t = torch.tensor(x_np, dtype=data_type)
    task_col = torch.full((x_t.shape[0], 1), float(task_id), dtype=data_type)
    return torch.cat([x_t, task_col], dim=1)


def _posterior_marginals(mtgp, x_with_task, chunk=8192):
    """Posterior marginal mean and std (noiseless f) for a batch of inputs."""
    mtgp.eval()
    mus, sds = [], []
    with torch.no_grad():
        for start in range(0, x_with_task.shape[0], chunk):
            post = mtgp.posterior(x_with_task[start:start + chunk])
            mus.append(post.mean.squeeze(-1).cpu().numpy())
            sds.append(torch.sqrt(post.variance.clamp_min(1e-20))
                       .squeeze(-1).cpu().numpy())
    return np.concatenate(mus), np.concatenate(sds)


def _gumbel_fit_samples(mu, sigma, n_samples):
    """
    Sample the max-value g* via the mean-field Gumbel approximation of [2].

    Approximates P(g* <= y) = prod_i Phi((y - mu_i) / sigma_i) over the grid,
    locates the 0.25/0.5/0.75 quantiles by bisection, fits a Gumbel
    distribution to them, and samples from it.
    """
    sigma = np.maximum(sigma, 1e-10)

    def log_cdf_max(y):
        return norm.logcdf((y - mu) / sigma).sum()

    lo = float(np.min(mu - 6.0 * sigma))
    hi = float(np.max(mu + 6.0 * sigma))

    def quantile(q):
        lo_q, hi_q = lo, hi
        log_q = np.log(q)
        for _ in range(50):
            mid = 0.5 * (lo_q + hi_q)
            if log_cdf_max(mid) < log_q:
                lo_q = mid
            else:
                hi_q = mid
        return 0.5 * (lo_q + hi_q)

    y25, y50, y75 = quantile(0.25), quantile(0.5), quantile(0.75)

    # Gumbel CDF exp(-exp(-(y-a)/b)): y_q = a - b*log(-log q)
    c25, c50, c75 = [-np.log(-np.log(q)) for q in (0.25, 0.5, 0.75)]
    b = max((y75 - y25) / (c75 - c25), 1e-10)
    a = y50 - b * c50

    u = np.random.rand(n_samples)
    return a - b * np.log(-np.log(u))


def _sample_gstar(mtgp, decs, dims, target_task, data_type,
                  n_gstar_samples=10, n_grid=None):
    """
    Sample g* values for the target task (paper Appendix A).

    Uses a mean-field approximation over a random grid plus the target
    task's evaluated points, then Gumbel sampling via [2].
    """
    max_dim = max(dims)
    target_dim = dims[target_task]

    if n_grid is None:
        n_grid = min(2000 * target_dim, 20000)
    grid_np = np.random.rand(n_grid, max_dim)

    # Always include already evaluated target-task points (paper Appendix A)
    existing = _pad_to_max(decs[target_task].copy(), max_dim)
    grid_np = np.vstack([grid_np, existing])

    grid_with_task = _with_task_column(grid_np, target_task, data_type)
    mu, sigma = _posterior_marginals(mtgp, grid_with_task)

    return _gumbel_fit_samples(mu, sigma, n_gstar_samples)


def _bivariate_stats_batch(mtgp, candidates_np, task_id, target_task, dims,
                           data_type=torch.double):
    """
    Batched bivariate GP statistics between g = f(x, z0) and y = f(x, z) + eps.

    Uses a batched joint posterior of shape (n, 2) to extract, per candidate,
    the paper Appendix A quantities mu_g, sigma_g, sigma_f_noisy and the
    predictive correlation rho = Sigma / (sigma_g * sqrt(sigma_f^2 + sigma^2)).

    Returns
    -------
    mu_g, sigma_g, sigma_f_noisy, rho : np.ndarray, each shape (n,)
    """
    max_dim = max(dims)
    n = candidates_np.shape[0]
    x_padded = _pad_to_max(candidates_np, max_dim)
    x_t = torch.tensor(x_padded, dtype=data_type)

    x_target = torch.cat(
        [x_t, torch.full((n, 1), float(target_task), dtype=data_type)], dim=1)
    x_cand = torch.cat(
        [x_t, torch.full((n, 1), float(task_id), dtype=data_type)], dim=1)
    x_joint = torch.stack([x_target, x_cand], dim=1)  # (n, 2, max_dim+1)

    noise_tensor = mtgp.likelihood.noise.detach().cpu()
    noise_var = noise_tensor.mean().item()

    mtgp.eval()
    mus, covs = [], []
    with torch.no_grad():
        chunk = 1024
        for start in range(0, n, chunk):
            post = mtgp.posterior(x_joint[start:start + chunk])
            mus.append(post.mvn.mean.cpu().numpy())                  # (c, 2)
            covs.append(post.mvn.covariance_matrix.cpu().numpy())    # (c, 2, 2)
    mu_all = np.concatenate(mus, axis=0)
    cov_all = np.concatenate(covs, axis=0)

    mu_g = mu_all[:, 0]
    sigma_g = np.sqrt(np.maximum(cov_all[:, 0, 0], 1e-20))
    sigma_f_noisy = np.sqrt(np.maximum(cov_all[:, 1, 1] + noise_var, 1e-20))
    rho = np.clip(cov_all[:, 0, 1] / (sigma_g * sigma_f_noisy), -0.999, 0.999)

    return mu_g, sigma_g, sigma_f_noisy, rho


def _mumbo_acquisition_batch(mu_g, sigma_g, rho, g_samples, n_quad=61):
    """
    Vectorized MUMBO acquisition (paper Eq. 7 / Appendix A).

        alpha = (1/N) sum_{g* in G} [
            rho^2 * gamma * phi(gamma) / (2 * Phi(gamma))
            - log Phi(gamma)
            + E_{theta ~ ESG}[ log Phi((gamma - rho*theta) / sqrt(1-rho^2)) ]
        ]

    The ESG expectation is integrated with Simpson's rule over mean +/- 4 std
    of the ESG (paper Appendix A), using its analytic moments.

    Parameters
    ----------
    mu_g, sigma_g, rho : np.ndarray, shape (n,)
        Bivariate statistics per candidate (model space).
    g_samples : np.ndarray, shape (N,)
        Monte Carlo samples of g*.
    n_quad : int
        Simpson points (forced odd).

    Returns
    -------
    alpha : np.ndarray, shape (n,)
    """
    if n_quad % 2 == 0:
        n_quad += 1

    sg = np.maximum(sigma_g, 1e-12)
    gamma = (g_samples[None, :] - mu_g[:, None]) / sg[:, None]      # (n, N)
    r = np.clip(rho, -0.999, 0.999)[:, None]                        # (n, 1)
    sqrt_1mr2 = np.sqrt(np.maximum(1.0 - r ** 2, 1e-12))            # (n, 1)

    phi_g = norm.pdf(gamma)
    Phi_g = np.maximum(norm.cdf(gamma), 1e-30)
    mills = phi_g / Phi_g

    term1 = (r ** 2) * gamma * phi_g / (2.0 * Phi_g)
    term2 = -np.log(Phi_g)

    # ESG moments (paper Eq. 9)
    esg_mean = r * mills                                             # (n, N)
    esg_var = np.maximum(1.0 - (r ** 2) * mills * (gamma + mills), 1e-12)
    esg_std = np.sqrt(esg_var)

    # theta grid: mean +/- 4 std, shared standardized nodes u in [-4, 4]
    u = np.linspace(-4.0, 4.0, n_quad)                               # (Q,)
    theta = esg_mean[..., None] + esg_std[..., None] * u[None, None, :]
    inner = (gamma[..., None] - r[..., None] * theta) / sqrt_1mr2[..., None]
    Phi_inner = norm.cdf(inner)

    esg_density = norm.pdf(theta) * Phi_inner / Phi_g[..., None]
    integrand = esg_density * np.log(np.maximum(Phi_inner, 1e-30))
    term3 = simpson(integrand, x=u, axis=-1) * esg_std               # (n, N)

    alpha = np.mean(term1 + term2 + term3, axis=1)
    return np.maximum(alpha, 0.0)


def _make_candidates(task_dim, n_candidates, decs_task, objs_task):
    """
    Candidate pool for one task: 80% uniform random, 20% Gaussian
    perturbations (sigma 0.02 and 0.1) around the task's best observed
    solutions, clipped to [0, 1].
    """
    n_local = max(n_candidates // 5, 2)
    n_global = n_candidates - n_local

    cands = [np.random.rand(n_global, task_dim)]

    order = np.argsort(objs_task.flatten())
    top = decs_task[order[:3]]
    seeds = top[np.random.randint(0, top.shape[0], size=n_local)]
    scales = np.where(np.random.rand(n_local, 1) < 0.5, 0.02, 0.1)
    local = np.clip(seeds + scales * np.random.randn(n_local, task_dim), 0, 1)
    cands.append(local)

    return np.vstack(cands)


def _select_next_point(mtgp, g_samples, target_task, active_tasks, decs, objs,
                       dims, task_cost, data_type, n_candidates=None,
                       n_quad=61):
    """
    Select the next (task, x) pair by maximizing alpha(x, z) / c(z).

    For every candidate task z with remaining budget, a candidate pool is
    screened with the vectorized MUMBO acquisition computed against the
    current target task; the globally best cost-weighted candidate wins.

    Returns
    -------
    best_task : int
        Selected task index.
    best_x : np.ndarray
        Selected candidate point in the task's own dimension, shape (d,).
    """
    best_ratio = -np.inf
    best_task = active_tasks[0]
    best_x = np.random.rand(dims[active_tasks[0]])

    for task_id in active_tasks:
        dim = dims[task_id]
        n_cand = n_candidates if n_candidates is not None else min(200 * dim, 2000)
        candidates = _make_candidates(dim, n_cand, decs[task_id], objs[task_id])

        mu_g, sigma_g, _, rho = _bivariate_stats_batch(
            mtgp, candidates, task_id, target_task, dims, data_type)

        alpha = _mumbo_acquisition_batch(mu_g, sigma_g, rho, g_samples,
                                         n_quad=n_quad)
        ratio = alpha / task_cost[task_id]

        j = int(np.argmax(ratio))
        if ratio[j] > best_ratio:
            best_ratio = ratio[j]
            best_task = task_id
            best_x = candidates[j].copy()

    return best_task, best_x

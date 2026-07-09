"""
EMTO-OTL: Expensive Multi-Task Optimization via Optimal Transport Learning

This module implements EMTO-OTL for expensive multi-task single-objective optimization.

Key innovations:
1. Gaussian Optimal Transport-based knowledge transfer:
   - Wasserstein-2 distance for task similarity (symmetric, geometric)
   - Closed-form OT map for principled cross-task solution transfer
   - Multi-solution transfer with GP pre-screening for candidate selection

2. Competitive GP-OT selection via surrogate arbitration:
   - Multi-merit GP search: build GP once, try all exploration weights (g=0,2,4),
     pick best candidate by predicted mean
   - GP-screened OT transfer: transport top-K source solutions via OT map,
     pre-screen with local GP to select the most promising transfer candidate
   - Competitive selection: always generate BOTH GP and OT candidates, then
     use the GP surrogate to arbitrate between them, selecting the GP-predicted best

References
----------
    [1] G. Peyre and M. Cuturi, "Computational Optimal Transport: With Applications
        to Data Science," Found. Trends Mach. Learn., vol. 11, no. 5-6, pp. 355-607, 2019.
    [2] W. R. Thompson, "On the Likelihood that One Unknown Probability Exceeds
        Another in View of the Evidence of Two Samples," Biometrika, vol. 25,
        no. 3-4, pp. 285-294, 1933.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.27
Version: 4.0
"""
import time
import warnings
import numpy as np
import torch
from tqdm import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import select_local_data

warnings.filterwarnings("ignore")


# ============================================================================
# Gaussian OT Utilities
# ============================================================================

def _matrix_sqrt_sym(A):
    """Compute symmetric positive-definite matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _matrix_sqrt_inv_sym(A):
    """Compute inverse of symmetric positive-definite matrix square root."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


class GaussianOTModel:
    """
    Gaussian Optimal Transport model for cross-task knowledge transfer.

    Fits a Gaussian distribution to each task's elite solutions in unified space,
    computes pairwise Wasserstein-2 distances, and provides closed-form OT maps
    for transporting solutions between task distributions.

    Parameters
    ----------
    reg : float
        Regularization for covariance estimation (default: 1e-3)
    top_ratio : float
        Fraction of top solutions used for Gaussian fitting (default: 0.3)
    """

    def __init__(self, reg=1e-3, top_ratio=0.3):
        self.reg = reg
        self.top_ratio = top_ratio
        self.means = {}
        self.covs = {}
        self.w2_matrix = None
        self.nt = 0
        self.uni_dim = 0

    def fit(self, decs, objs, dims):
        """Fit Gaussian per task using fitness-weighted elite solutions in unified space."""
        self.nt = len(decs)
        self.uni_dim = max(dims)

        for i in range(self.nt):
            task_decs = decs[i].copy()
            task_objs = objs[i].flatten()

            # Pad to unified dimension with mid-point value
            if task_decs.shape[1] < self.uni_dim:
                pad = 0.5 * np.ones((len(task_decs), self.uni_dim - task_decs.shape[1]))
                task_decs = np.hstack([task_decs, pad])

            # Select elite solutions
            n_elite = max(int(len(task_objs) * self.top_ratio), min(len(task_objs), 5))
            elite_idx = np.argsort(task_objs)[:n_elite]
            elite_decs = task_decs[elite_idx]

            # Fitness-weighted statistics
            ranks = np.arange(1, n_elite + 1, dtype=float)
            w = 1.0 / ranks
            w = w / w.sum()

            mean = np.average(elite_decs, weights=w, axis=0)
            centered = elite_decs - mean
            cov = (centered * w[:, None]).T @ centered + self.reg * np.eye(self.uni_dim)

            self.means[i] = mean
            self.covs[i] = cov

        self._compute_w2_matrix()

    def _compute_w2_matrix(self):
        """Compute pairwise Wasserstein-2 distances between all task distributions."""
        self.w2_matrix = np.zeros((self.nt, self.nt))
        for i in range(self.nt):
            for j in range(i + 1, self.nt):
                w2 = self._w2_distance(self.means[i], self.covs[i],
                                       self.means[j], self.covs[j])
                self.w2_matrix[i, j] = w2
                self.w2_matrix[j, i] = w2

    def _w2_distance(self, mu1, cov1, mu2, cov2):
        """Compute Wasserstein-2 distance between two Gaussian distributions."""
        mean_term = np.sum((mu1 - mu2) ** 2)
        try:
            cov1_sqrt = _matrix_sqrt_sym(cov1)
            inner = cov1_sqrt @ cov2 @ cov1_sqrt
            inner_sqrt = _matrix_sqrt_sym(inner)
            cov_term = np.trace(cov1 + cov2 - 2 * inner_sqrt)
            return max(mean_term + cov_term, 0.0)
        except np.linalg.LinAlgError:
            return mean_term

    def select_source_task(self, target, active_tasks):
        """Select source task with smallest W2 distance to target."""
        candidates = [s for s in active_tasks if s != target]
        if not candidates:
            return None
        distances = [self.w2_matrix[target, s] for s in candidates]
        return candidates[np.argmin(distances)]

    def transport(self, x, from_task, to_task):
        """Transport solution via closed-form Gaussian OT map."""
        if from_task not in self.means or to_task not in self.means:
            return x
        try:
            mu_from = self.means[from_task]
            mu_to = self.means[to_task]
            cov_from = self.covs[from_task]
            cov_to = self.covs[to_task]

            cov_from_sqrt = _matrix_sqrt_sym(cov_from)
            cov_from_sqrt_inv = _matrix_sqrt_inv_sym(cov_from)
            M = cov_from_sqrt @ cov_to @ cov_from_sqrt
            M_sqrt = _matrix_sqrt_sym(M)
            A = cov_from_sqrt_inv @ M_sqrt @ cov_from_sqrt_inv

            x_transported = mu_to + (x - mu_from) @ A.T
            return np.clip(x_transported, 0, 1)
        except np.linalg.LinAlgError:
            return np.clip(x - self.means.get(from_task, 0) + self.means.get(to_task, 0), 0, 1)


# ============================================================================
# EMTO-OTL Algorithm
# ============================================================================

class EMTO_OTL:
    """
    Expensive Multi-Task Optimization via Optimal Transport Learning.

    The main loop builds a local GP for each active task, then generates
    candidates from both local GP search and cross-task OT transfer:
    - GP multi-merit: Optimize merit functions with multiple exploration
      weights (g=0,2,4), select best by predicted mean
    - OT transfer: Transport top-K source solutions via OT map,
      pre-screen with local GP to pick the most promising transfer candidate
    - Competitive selection: Compare GP and OT candidates via GP prediction,
      evaluate the GP-predicted better one (surrogate-arbitrated)

    Parameters
    ----------
    problem : MTOP
        Multi-task optimization problem
    n_initial : int or list, optional
        Initial samples per task (default: 50)
    max_nfes : int or list, optional
        Max function evaluations per task (default: 100)
    NC : int
        Number of nearest neighbor points for local GP (default: 60)
    NR : int
        Number of most recent points for local GP (default: 60)
    de_pop : int
        DE population size for merit function optimization (default: 15)
    de_merit_evals : int
        Total DE evaluations for merit optimization, split across g values (default: 2000)
    ot_reg : float
        Regularization for OT covariance estimation (default: 1e-3)
    top_ratio : float
        Fraction of elite solutions for OT Gaussian fitting (default: 0.3)
    gen_gap : int
        Refit OT model every gen_gap iterations (default: 3)
    g_values : list of float
        Exploration weights for GP merit functions (default: [0, 2, 4])
    ot_top_k : int
        Number of top source solutions to transport for OT pre-screening (default: 5)
    ablation_mode : str or None
        Ablation variant: None (full), 'no_ot', or 'no_gp_screen' (default: None)
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

    def __init__(self, problem, n_initial=None, max_nfes=None,
                 NC=60, NR=60,
                 de_pop=15, de_merit_evals=2000,
                 ot_reg=1e-3, top_ratio=0.3,
                 gen_gap=3,
                 g_values=None,
                 ot_top_k=5,
                 ablation_mode=None,
                 save_data=True, save_path='./Data',
                 name='EMTO-OTL', disable_tqdm=True):
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.NC = NC
        self.NR = NR
        self.de_pop = de_pop
        self.de_merit_evals = de_merit_evals
        self.ot_reg = ot_reg
        self.top_ratio = top_ratio
        self.gen_gap = gen_gap
        self.g_values = g_values if g_values is not None else [0, 2, 4]
        self.ot_top_k = ot_top_k
        self.ablation_mode = ablation_mode
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMTO-OTL algorithm.

        The main loop:
        1. Build local GP for each active task (shared by both strategies)
        2. Generate GP multi-merit candidate (local search)
        3. Generate OT transfer candidate (cross-task knowledge transfer)
        4. Use GP to compare both candidates, select the GP-predicted best
        5. Evaluate the winning candidate
        6. Periodically refit OT model

        Returns
        -------
        Results
            Optimization results with decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # ===== Initialization =====
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = [n_initial_per_task[i] for i in range(nt)]

        # Fit initial OT model
        ot = GaussianOTModel(reg=self.ot_reg, top_ratio=self.top_ratio)
        ot.fit(decs, objs, dims)

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        iteration = 0

        # ===== Main Loop =====
        while sum(nfes_per_task) < total_nfes:
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                if nfes_per_task[i] >= max_nfes_per_task[i]:
                    continue

                # Build local GP for this task (shared by both strategies)
                gp_info = self._build_local_gp(decs[i], objs[i], dims[i])

                candidate = None

                if self.ablation_mode == 'no_ot':
                    # Ablation: GP-only search, no cross-task transfer
                    if gp_info is not None:
                        cand = self._gp_multi_merit_with_info(gp_info, dims[i])
                        if cand is not None:
                            candidate = cand.reshape(1, -1)
                elif self.ablation_mode == 'no_gp_screen':
                    # Ablation: OT transfer without GP comparison
                    ot_cand = self._ot_transfer_candidate(
                        decs, objs, dims, ot, i, active_tasks, None)
                    if ot_cand is not None:
                        candidate = ot_cand
                    elif gp_info is not None:
                        cand = self._gp_multi_merit_with_info(gp_info, dims[i])
                        if cand is not None:
                            candidate = cand.reshape(1, -1)
                else:
                    # ===== Full algorithm: competitive GP-OT selection =====
                    gp_cand = None
                    ot_cand = None

                    # Generate GP multi-merit candidate
                    if gp_info is not None:
                        cand = self._gp_multi_merit_with_info(gp_info, dims[i])
                        if cand is not None:
                            gp_cand = cand.reshape(1, -1)

                    # Generate OT transfer candidate (with GP pre-screening)
                    ot_cand = self._ot_transfer_candidate(
                        decs, objs, dims, ot, i, active_tasks, gp_info)

                    # Compare candidates via GP prediction, pick the best
                    if gp_cand is not None and ot_cand is not None and gp_info is not None:
                        both = np.vstack([gp_cand, ot_cand])
                        pred = self._batch_merit(
                            gp_info['gp'], gp_info['y_mean'],
                            gp_info['y_std'], both, g=0)
                        candidate = both[np.argmin(pred)].reshape(1, -1)
                    elif gp_cand is not None:
                        candidate = gp_cand
                    elif ot_cand is not None:
                        candidate = ot_cand

                # Final fallback: random sample
                if candidate is None:
                    candidate = np.random.rand(1, dims[i])

                # Evaluate candidate
                obj_new, _ = evaluation_single(problem, candidate, i)
                decs[i] = np.vstack([decs[i], candidate])
                objs[i] = np.vstack([objs[i], obj_new])
                nfes_per_task[i] += 1
                pbar.update(1)

            # Periodically refit OT model
            iteration += 1
            if iteration % self.gen_gap == 0:
                ot.fit(decs, objs, dims)

        pbar.close()
        runtime = time.time() - start_time

        # Build staircase history for analysis
        all_decs, all_objs = build_staircase_history(decs, objs, k=1)

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    # ========================================================================
    # GP Building
    # ========================================================================

    def _build_local_gp(self, decs_t, objs_t, dim_t):
        """
        Build local GP model and compute search bounds for a task.

        Returns
        -------
        dict or None
            Dictionary with keys: gp, y_mean, y_std, lb, ub, dim
            Returns None if GP building fails
        """
        best_idx = np.argmin(objs_t.flatten())
        x_best = decs_t[best_idx]

        # Select local training data: NC nearest + NR most recent
        selected_idx, nearest_idx = select_local_data(decs_t, objs_t, self.NC, self.NR)

        train_x = decs_t[selected_idx]
        train_y = objs_t[selected_idx].flatten()

        # Define local search bounds based on NC nearest points range
        nc_decs = decs_t[nearest_idx]
        d = np.max(nc_decs, axis=0) - np.min(nc_decs, axis=0)
        lb = np.clip(x_best - d / 2, 0, 1)
        ub = np.clip(x_best + d / 2, 0, 1)

        too_small = (ub - lb) < 1e-6
        lb[too_small] = np.clip(x_best[too_small] - 0.05, 0, 1)
        ub[too_small] = np.clip(x_best[too_small] + 0.05, 0, 1)

        # Build GP model
        try:
            gp, y_mean, y_std = self._build_gp(train_x, train_y)
            return {'gp': gp, 'y_mean': y_mean, 'y_std': y_std,
                    'lb': lb, 'ub': ub, 'dim': dim_t}
        except Exception:
            return None

    # ========================================================================
    # GP-Based Infill Methods
    # ========================================================================

    def _gp_multi_merit_with_info(self, gp_info, dim_t):
        """
        Multi-merit GP search using pre-built GP model.

        Optimizes merit function for each g value via DE, then selects
        the candidate with the best predicted mean (g=0 evaluation).

        Returns
        -------
        np.ndarray or None
            Best candidate found, shape (dim,), or None if all DE fail
        """
        gp = gp_info['gp']
        y_mean = gp_info['y_mean']
        y_std = gp_info['y_std']
        lb = gp_info['lb']
        ub = gp_info['ub']

        candidates = []
        budget_per_g = self.de_merit_evals // max(len(self.g_values), 1)
        for g in self.g_values:
            try:
                cand = self._optimize_merit(gp, y_mean, y_std, lb, ub, dim_t, g, budget_per_g)
                candidates.append(cand)
            except Exception:
                pass

        if not candidates:
            return None

        candidates = np.array(candidates)
        pred_means = self._batch_merit(gp, y_mean, y_std, candidates, g=0)
        return candidates[np.argmin(pred_means)]

    def _build_gp(self, train_x, train_y):
        """Build and fit a local GP model with standardized Y."""
        train_x_t = torch.tensor(train_x, dtype=torch.double)
        train_y_t = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)

        y_mean = train_y_t.mean()
        y_std = train_y_t.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, dtype=torch.double)
        train_y_norm = (train_y_t - y_mean) / y_std

        gp = SingleTaskGP(train_x_t, train_y_norm)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        return gp, y_mean, y_std

    def _batch_merit(self, gp, y_mean, y_std, X, g):
        """Batch evaluate merit function f_M(x) = mean(x) - g*std(x)."""
        X_t = torch.tensor(X, dtype=torch.double)
        with torch.no_grad():
            posterior = gp.posterior(X_t)
            means = posterior.mean.squeeze(-1).numpy() * y_std.item() + y_mean.item()
            stds = posterior.variance.squeeze(-1).sqrt().numpy() * y_std.item()
        return means - g * stds

    def _optimize_merit(self, gp, y_mean, y_std, lb, ub, dim, g, budget=None):
        """Optimize merit function using DE within local bounds."""
        pop_size = self.de_pop
        if budget is None:
            budget = self.de_merit_evals
        max_gen = max(budget // pop_size, 1)
        F, CR = 0.5, 0.9

        pop = lb + (ub - lb) * np.random.rand(pop_size, dim)
        fit = self._batch_merit(gp, y_mean, y_std, pop, g)

        for _ in range(max_gen):
            trials = np.empty_like(pop)
            for j in range(pop_size):
                idxs = list(range(pop_size))
                idxs.remove(j)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trials[j] = np.where(mask, mutant, pop[j])

            trial_fit = self._batch_merit(gp, y_mean, y_std, trials, g)
            improved = trial_fit < fit
            pop[improved] = trials[improved]
            fit[improved] = trial_fit[improved]

        return pop[np.argmin(fit)]

    # ========================================================================
    # OT Transfer Methods
    # ========================================================================

    def _ot_transfer_candidate(self, decs, objs, dims, ot, target_task, active_tasks,
                               gp_info=None):
        """
        Generate a transfer candidate via OT mapping with GP pre-screening.

        Transports the top-K solutions from the best source task via OT map.
        If a GP model is available, pre-screens all transferred candidates
        by predicted mean and returns the best one.

        Parameters
        ----------
        decs, objs, dims : task data
        ot : GaussianOTModel
        target_task : int
        active_tasks : list of int
        gp_info : dict or None
            Pre-built GP info for pre-screening (from _build_local_gp)

        Returns
        -------
        np.ndarray or None
            Transfer candidate, shape (1, dims[target_task])
        """
        source = ot.select_source_task(target_task, active_tasks)
        if source is None:
            return None

        # Get top-K solutions from source task
        K = min(self.ot_top_k, len(objs[source]))
        top_idx = np.argsort(objs[source].flatten())[:K]

        candidates = []
        for idx in top_idx:
            x_source = decs[source][idx].copy()

            # Pad to unified dimension for OT
            if len(x_source) < ot.uni_dim:
                x_source = np.pad(x_source, (0, ot.uni_dim - len(x_source)),
                                  mode='constant', constant_values=0.5)

            # Transport via OT map
            x_transferred = ot.transport(x_source, source, target_task)

            # Truncate to target dimension + small perturbation
            candidate = x_transferred[:dims[target_task]]
            perturbation = np.random.randn(dims[target_task]) * 0.01
            candidate = np.clip(candidate + perturbation, 0, 1)
            candidates.append(candidate)

        if not candidates:
            return None

        # GP pre-screening: select best transferred candidate by predicted mean
        if gp_info is not None and len(candidates) > 1:
            gp = gp_info['gp']
            y_mean = gp_info['y_mean']
            y_std = gp_info['y_std']
            candidates_arr = np.array(candidates)
            pred_means = self._batch_merit(gp, y_mean, y_std, candidates_arr, g=0)
            best = candidates_arr[np.argmin(pred_means)]
            return best.reshape(1, -1)

        return candidates[0].reshape(1, -1)

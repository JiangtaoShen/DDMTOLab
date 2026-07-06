"""
Task-Adaptive Transfer Surrogate Bayesian Optimization (TATS-BO)

A novel expensive multi-task optimization algorithm featuring:
1. Dual-Surrogate Architecture: RBF for fast candidate screening + GP for precise
   acquisition, with adaptive switching based on optimization phase
2. Adaptive Wasserstein-Inspired Transfer: Task similarity via Spearman rank correlation
   in objective space with dimension overlap penalty, dynamically controlling transfer
3. Multi-Strategy Portfolio with Credit Assignment: DE-surrogate search, cross-task
   transfer, and BO acquisition compete via UCB bandit with online credit tracking

References
----------
    Novel algorithm proposed for expensive multi-task optimization.

Notes
-----
Author: Research Team
Date: 2026.02
Version: 1.0
"""

import numpy as np
from tqdm import tqdm
import torch
import time
import warnings

from scipy.interpolate import RBFInterpolator
from scipy.stats import spearmanr
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from ddmtolab.Methods.Algo_Methods.algo_utils import *

warnings.filterwarnings("ignore")


class TATS_BO:
    """
    Task-Adaptive Transfer Surrogate Bayesian Optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
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
                 alpha=0.3, beta=0.5,
                 disable_transfer=False, disable_bandit=False, disable_gp=False,
                 save_data=True, save_path='./Data',
                 name='TATS-BO', disable_tqdm=True):
        """
        Initialize TATS-BO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 20)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        alpha : float, optional
            Credit decay rate for temporal adaptation (default: 0.3)
        beta : float, optional
            Exploration parameter for UCB strategy selection (default: 0.5)
        disable_transfer : bool, optional
            Ablation: disable cross-task transfer strategy (default: False)
        disable_bandit : bool, optional
            Ablation: disable UCB bandit, use uniform random selection (default: False)
        disable_gp : bool, optional
            Ablation: disable GP surrogate, use only RBF (default: False)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'TATS-BO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 20
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.alpha = alpha
        self.beta = beta
        self.disable_transfer = disable_transfer
        self.disable_bandit = disable_bandit
        self.disable_gp = disable_gp
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm
        self.behavior_log = None  # Populated after optimize() if needed

    def optimize(self):
        """
        Execute the TATS-BO algorithm.

        Returns
        -------
        Results
            Optimization results
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        nfes_per_task = n_initial_per_task.copy()

        # --- Initialization ---
        decs = initialization(problem, self.n_initial, method='lhs', the_same=True)
        objs, _ = evaluation(problem, decs)

        # Strategy credit scores: [DE-surrogate, BO-acquisition, Transfer]
        # With ablation: disable_transfer removes strategy 2, disable_gp removes strategy 1
        n_strategies = 3
        available_strategies = [0, 1, 2]  # DE, BO, Transfer
        if self.disable_transfer:
            available_strategies = [s for s in available_strategies if s != 2]
        if self.disable_gp:
            available_strategies = [s for s in available_strategies if s != 1]
        strategy_credits = np.ones((nt, n_strategies)) / n_strategies
        strategy_counts = np.ones((nt, n_strategies))

        # Task similarity matrix
        sim_matrix = _compute_task_similarity(decs, objs, dims)

        # Track best objectives per task
        best_objs = [np.min(objs[i]) for i in range(nt)]

        # Cache GP models (expensive to build)
        gp_cache = [None] * nt
        gp_data_size = [0] * nt  # track when GP was last built

        # Behavior logging
        log_strategies = {i: [] for i in range(nt)}
        log_credits = {i: [] for i in range(nt)}
        log_similarities = []
        log_best_objs = {i: [] for i in range(nt)}

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        iteration = 0
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Update similarity periodically
            if iteration % 5 == 0 and iteration > 0:
                sim_matrix = _compute_task_similarity(decs, objs, dims)

            for i in active_tasks:
                remaining = max_nfes_per_task[i] - nfes_per_task[i]
                if remaining <= 0:
                    continue

                # === Build RBF (always, it's fast) ===
                rbf_model = _build_rbf(decs[i], objs[i])

                # === Build/update GP (only periodically - every 5 new evals) ===
                if not self.disable_gp:
                    data_growth = len(decs[i]) - gp_data_size[i]
                    if gp_cache[i] is None or data_growth >= 3:
                        gp_cache[i] = _build_gp(decs[i], objs[i])
                        gp_data_size[i] = len(decs[i])

                # === Strategy Selection via UCB Bandit ===
                if self.disable_bandit:
                    strategy_idx = available_strategies[
                        np.random.randint(len(available_strategies))
                    ]
                else:
                    strategy_idx = _select_strategy_constrained(
                        strategy_credits[i], strategy_counts[i],
                        self.beta, available_strategies
                    )

                # Log strategy selection
                log_strategies[i].append(strategy_idx)

                candidate = None
                if strategy_idx == 0:
                    candidate = _de_surrogate_search(decs[i], objs[i], rbf_model, dims[i])
                elif strategy_idx == 1:
                    candidate = _bo_acquisition(decs[i], objs[i], gp_cache[i], dims[i])
                else:
                    candidate = _cross_task_transfer(i, decs, objs, dims, sim_matrix)

                # Surrogate-guided filtering for transfer candidates
                if strategy_idx == 2 and candidate is not None and len(candidate) > 1 and rbf_model is not None:
                    try:
                        preds = rbf_model(candidate)
                        best_k = min(1, len(preds))
                        top_idx = np.argsort(preds)[:best_k]
                        candidate = candidate[top_idx]
                    except Exception:
                        candidate = candidate[:1]

                # Fallback
                if candidate is None or len(candidate) == 0:
                    candidate = _de_surrogate_search(decs[i], objs[i], rbf_model, dims[i])

                # Budget control
                n_eval = min(len(candidate), remaining)
                candidate = candidate[:n_eval]

                # Remove duplicates
                candidate = _remove_near_duplicates(candidate, decs[i])
                if len(candidate) == 0:
                    candidate = np.random.rand(1, dims[i])
                n_eval = min(len(candidate), remaining)
                candidate = candidate[:n_eval]

                # === Evaluate and Update ===
                new_objs, _ = evaluation_single(problem, candidate, i)
                decs[i], objs[i] = vstack_groups(
                    (decs[i], candidate), (objs[i], new_objs)
                )

                # Credit assignment
                new_best = np.min(new_objs)
                if new_best < best_objs[i]:
                    improvement = (best_objs[i] - new_best) / (abs(best_objs[i]) + 1e-10)
                    strategy_credits[i][strategy_idx] += improvement
                    best_objs[i] = new_best

                strategy_counts[i][strategy_idx] += 1

                # Decay credits
                strategy_credits[i] *= (1 - self.alpha * 0.05)
                strategy_credits[i] = np.maximum(strategy_credits[i], 0.01)

                nfes_per_task[i] += n_eval
                pbar.update(n_eval)

                # Log credits and best objectives
                log_credits[i].append(strategy_credits[i].copy())
                log_best_objs[i].append(best_objs[i])

            log_similarities.append(sim_matrix.copy())
            iteration += 1

        pbar.close()
        runtime = time.time() - start_time

        # Store behavior log
        self.behavior_log = {
            'strategies': log_strategies,
            'credits': log_credits,
            'similarities': log_similarities,
            'best_objs': log_best_objs,
        }

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data
        )
        return results


# =============================================================================
# Component Functions
# =============================================================================

def _build_rbf(decs, objs):
    """Build RBF surrogate for fast candidate screening."""
    try:
        return RBFInterpolator(decs, objs.flatten(), kernel='thin_plate_spline')
    except Exception:
        try:
            return RBFInterpolator(decs, objs.flatten(), kernel='linear')
        except Exception:
            return None


def _build_gp(decs, objs, data_type=torch.double):
    """Build GP model for precise BO acquisition."""
    try:
        train_X = torch.tensor(decs, dtype=data_type)
        train_Y = torch.tensor(-objs, dtype=data_type)
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp
    except Exception:
        return None


def _compute_task_similarity(decs, objs, dims):
    """Compute task similarity via Spearman correlation + dimension overlap."""
    nt = len(decs)
    sim = np.eye(nt)
    for i in range(nt):
        for j in range(i + 1, nt):
            n_common = min(len(objs[i]), len(objs[j]))
            if n_common >= 3:
                corr, _ = spearmanr(objs[i][:n_common].flatten(),
                                     objs[j][:n_common].flatten())
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            dim_ratio = min(dims[i], dims[j]) / max(dims[i], dims[j])
            similarity = 0.7 * abs(corr) + 0.3 * dim_ratio
            sim[i, j] = sim[j, i] = similarity
    return sim


def _select_strategy(credits, counts, beta):
    """UCB-like bandit strategy selection."""
    total = np.sum(counts)
    avg_reward = credits / counts
    exploration = beta * np.sqrt(np.log(total + 1) / counts)
    return int(np.argmax(avg_reward + exploration))


def _select_strategy_constrained(credits, counts, beta, available):
    """UCB-like bandit strategy selection, constrained to available strategies."""
    total = np.sum(counts)
    avg_reward = credits / counts
    exploration = beta * np.sqrt(np.log(total + 1) / counts)
    ucb_values = avg_reward + exploration
    # Only consider available strategies
    best_idx = available[0]
    best_val = ucb_values[best_idx]
    for idx in available[1:]:
        if ucb_values[idx] > best_val:
            best_val = ucb_values[idx]
            best_idx = idx
    return best_idx


def _de_surrogate_search(decs, objs, rbf_model, dim, n_pop=50, n_gens=20):
    """DE optimization on RBF surrogate to find promising candidate."""
    if rbf_model is None:
        return np.random.rand(1, dim)

    n = len(decs)
    best_indices = np.argsort(objs.flatten())[:min(10, n)]

    # Initialize population: mix of perturbations around best + random
    pop = np.random.rand(n_pop, dim)
    n_seed = min(len(best_indices), n_pop // 2)
    for k in range(n_seed):
        sigma = 0.1 * (1.0 + 0.5 * np.random.rand())
        pop[k] = decs[best_indices[k % len(best_indices)]] + sigma * np.random.randn(dim)
    pop = np.clip(pop, 0, 1)

    # Vectorized DE evolution on surrogate
    F, CR = 0.5, 0.9
    for _ in range(n_gens):
        # Vectorized mutation and crossover
        idx = np.array([np.random.choice(n_pop, 3, replace=False) for _ in range(n_pop)])
        mutants = pop[idx[:, 0]] + F * (pop[idx[:, 1]] - pop[idx[:, 2]])
        mutants = np.clip(mutants, 0, 1)

        mask = np.random.rand(n_pop, dim) < CR
        j_rand = np.random.randint(dim, size=n_pop)
        mask[np.arange(n_pop), j_rand] = True
        trials = np.where(mask, mutants, pop)

        # Batch surrogate evaluation
        try:
            pred_trials = rbf_model(trials)
            pred_pop = rbf_model(pop)
            improved = pred_trials < pred_pop
            pop[improved] = trials[improved]
        except Exception:
            pass

    # Return best predicted point
    try:
        pred = rbf_model(pop)
        return pop[[np.argmin(pred)]]
    except Exception:
        return pop[[0]]


def _bo_acquisition(decs, objs, gp_model, dim, data_type=torch.double):
    """GP-based BO with LogEI acquisition."""
    if gp_model is None:
        return np.random.rand(1, dim)
    try:
        train_Y = torch.tensor(-objs, dtype=data_type)
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
        best_f = train_Y.max()
        logEI = LogExpectedImprovement(model=gp_model, best_f=best_f)
        bounds = torch.stack([
            torch.zeros(dim, dtype=data_type),
            torch.ones(dim, dtype=data_type)
        ], dim=0)
        candidate, _ = optimize_acqf(
            logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=30
        )
        return candidate.detach().cpu().numpy()
    except Exception:
        return np.random.rand(1, dim)


def _cross_task_transfer(task_idx, decs, objs, dims, sim_matrix):
    """Transfer best solutions from similar tasks with dimension alignment."""
    nt = len(decs)
    candidates = []

    for j in range(nt):
        if j == task_idx:
            continue
        similarity = sim_matrix[task_idx, j]
        if similarity < 0.05 or np.random.rand() > similarity:
            continue

        # Transfer top-k solutions
        sorted_idx = np.argsort(objs[j].flatten())
        n_transfer = min(3, len(sorted_idx))

        for k in range(n_transfer):
            sol = decs[j][sorted_idx[k]].copy()
            sol = _align_dimensions(sol, dims[task_idx])
            candidates.append(sol)

            # Perturbed versions for diversity (multiple perturbation scales)
            if k == 0:
                for sigma in [0.03, 0.08]:
                    perturbed = sol + sigma * np.random.randn(dims[task_idx])
                    candidates.append(np.clip(perturbed, 0, 1))

    return np.array(candidates) if candidates else None


def _align_dimensions(sol, target_dim):
    """Align solution dimensions via padding or truncation."""
    if len(sol) < target_dim:
        padding = np.random.rand(target_dim - len(sol)) * 0.5 + 0.25
        return np.concatenate([sol, padding])
    elif len(sol) > target_dim:
        return sol[:target_dim]
    return sol


def _remove_near_duplicates(candidates, existing, tol=1e-4):
    """Remove candidates too close to existing solutions."""
    if len(candidates) == 0:
        return candidates
    unique = []
    for c in candidates:
        c_2d = c.reshape(1, -1)
        if len(existing) > 0:
            if np.min(np.linalg.norm(existing - c_2d, axis=1)) > tol:
                unique.append(c)
        else:
            unique.append(c)
    if not unique:
        return np.array([]).reshape(0, candidates.shape[1])
    return np.array(unique)

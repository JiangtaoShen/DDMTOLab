"""
BO-LCB-BCKT: Bayesian Optimization with Lower Confidence Bound and Bayesian Competitive Knowledge Transfer

This module implements BO-LCB-BCKT for expensive multi-task optimization problems.

References
----------
    [1] Lu, Yi, et al. "Multi-Task Surrogate-Assisted Search with Bayesian Competitive Knowledge Transfer for Expensive Optimization." arXiv preprint arXiv:2510.23407 (2025).

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.09
Version: 1.0
"""
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.bo_utils import bo_next_point_lcb
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings
import time
import numpy as np
import torch
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


class BO_LCB_BCKT:
    """
    BO-LCB-BCKT algorithm for expensive multi-task optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'heterogeneous',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, gen_gap=10,
                 sigma_I_sq=0.05 ** 2, save_data=True, save_path='./TestData',
                 name='BO_LCB_BCKT_test', disable_tqdm=True, padding='zero'):
        """
        Initialize BO-LCB-BCKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int, optional
            Number of initial samples per task (default: 50)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 100)
        gen_gap : int, optional
            Knowledge transfer trigger frequency (default: 10)
        sigma_I_sq : float, optional
            Base variance for prior and likelihood (default: 0.0025)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'BO_LCB_BCKT_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        padding : str, optional
            Padding strategy for unified space ('zero' or 'random', default: 'zero')
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.gen_gap = gen_gap
        self.sigma_I_sq = sigma_I_sq
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm
        self.padding = padding
        self.dims = problem.dims
        self.d_max = np.max(problem.dims)

    def optimize(self):
        """
        Execute the BO-LCB-BCKT algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives,
            runtime, and transfer actions
        """
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d_max = self.d_max
        n_initial = self.n_initial
        max_nfes = self.max_nfes
        gen_gap = self.gen_gap

        # ======================== Phase 1: Initialization ========================
        # Generate initial samples in real space (Latin Hypercube Sampling)
        decs_real = initialization(problem, n_initial, method='lhs')
        objs_real, _ = evaluation(problem, decs_real)

        # Transform to unified space for optimization
        decs_uni = space_transfer(problem, decs_real, type='uni', padding=self.padding)

        # Reorganize into task-specific history lists (unified space for optimization)
        all_decs_uni = reorganize_initial_data(decs_uni, nt, [n_initial] * nt)
        all_objs = reorganize_initial_data(objs_real, nt, [n_initial] * nt)

        # Also keep real space history for saving results
        all_decs_real = reorganize_initial_data(decs_real, nt, [n_initial] * nt)

        # Current working data (unified space)
        decs = [all_decs_uni[i][-1].copy() for i in range(nt)]  # Current unified decs
        objs = [all_objs[i][-1].copy() for i in range(nt)]  # Current objs

        # Initialize evaluation counters
        nfes_per_task = [n_initial] * nt
        total_nfes = sum(nfes_per_task)

        # Initialize Bayesian transferability tracking
        T_history = [[[] for _ in range(nt)] for _ in range(nt)]
        R_history = [[[] for _ in range(nt)] for _ in range(nt)]
        k_transfer = np.zeros((nt, nt), dtype=int)

        # Initialize transfer decision history
        transfer_actions = [[] for _ in range(nt)]

        # Cache for pending transfer info
        pending_transfer_info = [None] * nt

        # Initialize GP model cache
        gp_models = [None] * nt

        pbar = tqdm(total=max_nfes * nt, initial=total_nfes,
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # ==================== Phase 2: Main Optimization Loop =======================
        while total_nfes < max_nfes * nt:
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes]
            if not active_tasks:
                break

            # ---------- Step 1: Single-Task Optimization (BO-LCB) in unified space ----------
            solutions_in = []
            improvements_in = []

            for i in active_tasks:
                # Optimize using LCB in unified space (d_max dimensions)
                candidate_uni, gp_model = bo_next_point_lcb(
                    d_max, decs[i], objs[i], data_type=data_type
                )
                solutions_in.append(candidate_uni)

                # Cache GP model
                gp_models[i] = gp_model

                # Calculate internal improvement (Eq. 1)
                imp_in = improvement_internal(
                    gp_model, decs[i], objs[i], candidate_uni,
                    acquisition='lcb', data_type=data_type
                )
                improvements_in.append(imp_in)

            # ---------- Step 2: Bayesian Competitive Knowledge Transfer (unified space) ----------
            solutions_candidate = []

            for idx, target_idx in enumerate(active_tasks):
                if nfes_per_task[target_idx] % gen_gap == 0 and nfes_per_task[target_idx] > n_initial:
                    # Execute knowledge competition in unified space
                    (solution_ex, improvement_ex, source_idx,
                     delta_p, similarity) = knowledge_competition(
                        decs, objs, gp_models, target_idx,
                        T_history, R_history, k_transfer,
                        self.sigma_I_sq, data_type
                    )

                    if improvements_in[idx] >= improvement_ex:
                        solutions_candidate.append(solutions_in[idx])
                        transfer_actions[target_idx].append(0)
                        pending_transfer_info[target_idx] = None
                    else:
                        solutions_candidate.append(solution_ex)
                        transfer_actions[target_idx].append(source_idx + 1)
                        pending_transfer_info[target_idx] = {
                            'source_idx': source_idx,
                            'delta_p': delta_p,
                            'similarity': similarity,
                            'min_target_before': np.min(objs[target_idx])
                        }
                else:
                    solutions_candidate.append(solutions_in[idx])
                    transfer_actions[target_idx].append(0)
                    pending_transfer_info[target_idx] = None

            # ---------- Step 3: Evaluation in real space and Database Update ----------
            for idx, target_idx in enumerate(active_tasks):
                candidate_uni = solutions_candidate[idx]

                # Ensure uniqueness in unified space
                candidate_uni = ensure_unique(candidate_uni, decs[target_idx], epsilon=5e-3)

                # Transform to real space for evaluation
                candidate_real = candidate_uni[:, :dims[target_idx]]

                # real evaluation in real space
                obj, _ = evaluation_single(problem, candidate_real, target_idx)

                # Update unified space database
                decs[target_idx] = np.vstack([decs[target_idx], candidate_uni])
                objs[target_idx] = np.vstack([objs[target_idx], obj])

                # Store cumulative history in unified space (for optimization continuity)
                append_history(all_decs_uni[target_idx], decs[target_idx],
                               all_objs[target_idx], objs[target_idx])

                # Store cumulative history in real space (for result saving)
                # Get current real-space decs by transforming unified decs
                current_decs_real = decs[target_idx][:, :dims[target_idx]]
                all_decs_real[target_idx].append(current_decs_real.copy())

                # ---------- Step 4: Bayesian Update (if transfer occurred) ----------
                if pending_transfer_info[target_idx] is not None:
                    info = pending_transfer_info[target_idx]
                    source_idx = info['source_idx']
                    delta_p = info['delta_p']
                    similarity = info['similarity']
                    min_target_before = info['min_target_before']

                    new_obj_value = obj[0, 0] if obj.ndim > 1 else obj[0]

                    if delta_p != 0:
                        T_k = (min_target_before - new_obj_value) / delta_p
                    else:
                        T_k = 0.0

                    T_history[source_idx][target_idx].append(T_k)
                    R_history[source_idx][target_idx].append(similarity)
                    k_transfer[source_idx, target_idx] += 1

                nfes_per_task[target_idx] += 1
                total_nfes += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # ======================== Phase 3: Build Results (real space) ========================
        # Transform all_decs back to real space for saving
        final_decs_real = []
        for i in range(nt):
            task_decs_real = []
            for dec_uni in all_decs_uni[i]:
                dec_real = dec_uni[:, :dims[i]]
                task_decs_real.append(dec_real)
            final_decs_real.append(task_decs_real)

        # Build results with real space decision variables
        results = build_save_results(
            all_decs=final_decs_real,
            all_objs=all_objs,
            runtime=runtime,
            max_nfes=nfes_per_task,
            bounds=problem.bounds,  # real space bounds
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        results.transfer_states = transfer_actions

        return results


# ==================== Utility Functions ====================

def ensure_unique(candidate, decs, epsilon=5e-3, max_trials=50):
    """
    Ensure candidate solution is unique from database solutions.
    """
    for trial in range(max_trials):
        distances = np.max(np.abs(candidate - decs), axis=1)
        min_dist = np.min(distances)

        if min_dist >= epsilon:
            break

        scale = np.linspace(0.1, 1.0, max_trials)[trial]
        perturbation = scale * (np.random.rand(candidate.shape[1]) - 0.5)
        candidate = candidate + perturbation
        candidate = np.clip(candidate, 0, 1)

    return candidate


def knowledge_competition(decs, objs, gp_models, target_idx,
                          T_history, R_history, k_transfer,
                          sigma_I_sq, data_type):
    """
    Execute Bayesian Competitive Knowledge Transfer.

    All operations are performed in unified space.
    """
    nt = len(decs)
    Xt = decs[target_idx]
    Yt = objs[target_idx].flatten()

    improvements = np.full(nt, -np.inf)
    delta_ps = np.full(nt, 0.0)
    similarities = np.full(nt, 0.0)
    solutions_external = []

    for source_idx in range(nt):
        if source_idx == target_idx:
            solutions_external.append(None)
            continue

        Xs = decs[source_idx]
        Ys = objs[source_idx].flatten()

        # Calculate Task Similarity in unified space
        similarity, Yval = compute_similarity_ssrc(
            Xs, Ys, gp_models[source_idx], Xt, Yt, data_type
        )
        similarities[source_idx] = similarity

        # Calculate Δp^ij
        delta_p = compute_delta_p(Ys, Yval, Yt)
        delta_ps[source_idx] = delta_p

        # Bayesian Inference of Transferability
        k = k_transfer[source_idx, target_idx]

        if k == 0:
            tau_hat = similarity
            sigma_k_sq = sigma_I_sq
        else:
            tau_hat, sigma_k_sq = bayesian_update_tau(
                T_history[source_idx][target_idx],
                R_history[source_idx][target_idx],
                k, sigma_I_sq
            )

        transferability = np.random.normal(tau_hat, np.sqrt(sigma_k_sq))

        improvement = compute_external_improvement(
            delta_p, transferability, np.max(Yt)
        )
        improvements[source_idx] = improvement

        # Best solution from source task in unified space
        best_idx = np.argmin(Ys)
        solutions_external.append(Xs[best_idx:best_idx + 1])

    source_idx = np.argmax(improvements)
    improvement_ex = improvements[source_idx]
    solution_ex = solutions_external[source_idx]
    delta_p = delta_ps[source_idx]
    similarity = similarities[source_idx]

    return solution_ex, improvement_ex, source_idx, delta_p, similarity


def bayesian_update_tau(T_history_ij, R_history_ij, k, sigma_I_sq):
    """
    Bayesian update of transferability τ.
    """
    epsilon = sum(np.exp(l) for l in range(1, k + 1))
    omega = epsilon / (k + epsilon)

    weighted_T = sum(
        (np.exp(l) / epsilon) * T_history_ij[l - 1]
        for l in range(1, k + 1)
    )

    avg_R = sum(R_history_ij) / k
    tau_hat = omega * weighted_T + (1 - omega) * avg_R
    sigma_k_sq = sigma_I_sq / (k + epsilon)

    return tau_hat, sigma_k_sq


def compute_similarity_ssrc(Xs, Ys, gp_source, Xt, Yt, data_type):
    """
    Compute Surrogate-based Spearman Rank Correlation (SSRC).

    All inputs should be in unified space.
    """
    Xt_torch = torch.tensor(Xt, dtype=data_type)
    with torch.no_grad():
        posterior = gp_source.posterior(Xt_torch)
        objs_val = posterior.mean.cpu().numpy().flatten()

    ranks_target = np.argsort(np.argsort(Yt)) + 1
    ranks_val = np.argsort(np.argsort(objs_val)) + 1

    similarity, _ = spearmanr(ranks_target, ranks_val)

    if np.isnan(similarity):
        similarity = 0.0

    return similarity, objs_val


def compute_delta_p(objs_source, objs_val, objs_target):
    """
    Calculate Δp^ij.
    """
    min_val = np.min(objs_val)
    min_source = np.min(objs_source)
    max_target = np.max(objs_target)
    max_source = np.max(objs_source)

    if max_source != 0:
        delta_p = (min_val - min_source) * (max_target / max_source)
    else:
        delta_p = 0.0

    return delta_p


def compute_external_improvement(delta_p, transferability, max_target):
    """
    Calculate external improvement Δ_ex^ij.
    """
    improvement = transferability * delta_p
    return improvement


def improvement_internal(gp_model, decs, objs, candidate, acquisition='lcb',
                         data_type=torch.float):
    """
    Calculate internal improvement Δ_in^j.

    All inputs should be in unified space.
    """
    objs = objs.flatten()

    X_db_torch = torch.tensor(decs, dtype=data_type)
    with torch.no_grad():
        posterior_db = gp_model.posterior(X_db_torch)
        mu_db = posterior_db.mean.cpu().numpy().flatten()
        std_db = posterior_db.variance.sqrt().cpu().numpy().flatten()

    X_cand_torch = torch.tensor(candidate, dtype=data_type)
    with torch.no_grad():
        posterior_cand = gp_model.posterior(X_cand_torch)
        mu_cand = posterior_cand.mean.cpu().numpy().flatten()
        std_cand = posterior_cand.variance.sqrt().cpu().numpy().flatten()

    if acquisition == 'lcb':
        kappa = 2.0
        acq_db = mu_db - kappa * std_db
        acq_cand = mu_cand - kappa * std_cand
    elif acquisition == 'plain':
        acq_db = mu_db
        acq_cand = mu_cand
    else:
        raise ValueError(f"Unknown acquisition function: {acquisition}")

    improvement = np.min(acq_db) - acq_cand[0]

    return improvement
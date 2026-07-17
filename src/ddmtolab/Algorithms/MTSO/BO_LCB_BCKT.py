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
Version: 2.0
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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
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
                 sigma_I_sq=0.05 ** 2, save_data=True, save_path='./Data',
                 name='BO-LCB-BCKT', disable_tqdm=True, padding='zero'):
        """
        Initialize BO-LCB-BCKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance (all tasks optimized
            simultaneously, unlike the sequential-transfer CKT variant).
        n_initial : int, optional
            Number of initial samples per task (default: 50, matching the
            reference implementation)
        max_nfes : int, optional
            Maximum number of function evaluations per task, including the
            initial samples (default: 200, matching FEsMax = 200 x T in the
            reference code)
        gen_gap : int, optional
            Knowledge transfer trigger interval per task in true evaluations;
            the total-FE interval is ``n_tasks * gen_gap`` as in the reference
            code (default: 10)
        sigma_I_sq : float, optional
            Observation variance of the similarity measurements in the
            conjugate Gaussian model of transferability (default: 0.05**2)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'BO-LCB-BCKT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        padding : str, optional
            Padding mode for the unified search space (default: 'zero')
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 200
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
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d_max = self.d_max
        n_initial = self.n_initial
        max_nfes = self.max_nfes
        gen_gap = self.gen_gap
        # Similarity observation variance (configurable; 0.05^2 in the paper)
        sigma2_sim = self.sigma_I_sq

        # ======================== Phase 1: Initialization ========================
        decs_real = initialization(problem, n_initial, method='lhs')
        objs_real, _ = evaluation(problem, decs_real)

        # Transform to unified space
        decs_uni = space_transfer(problem, decs_real, type='uni', padding=self.padding)

        # Current working data (unified space)
        decs = [decs_uni[i].copy() for i in range(nt)]
        objs = [objs_real[i].copy() for i in range(nt)]

        # Evaluation counters
        nfes_per_task = [n_initial] * nt
        total_nfes = sum(nfes_per_task)

        # Bayesian transferability: MUs and SIGMA2s (MATLAB-style conjugate Gaussian)
        MUs = np.full((nt, nt), np.nan)
        SIGMA2s = np.full((nt, nt), np.nan)

        # Transfer decision history
        transfer_actions = [[] for _ in range(nt)]

        # GP model cache
        gp_models = [None] * nt

        # gen_gap in total FEs (MATLAB: gen_gap = no_tasks * paras.gen_gap)
        gen_gap_total = nt * gen_gap

        pbar = tqdm(total=max_nfes * nt, initial=total_nfes,
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # ==================== Phase 2: Main Optimization Loop =======================
        while total_nfes < max_nfes * nt:
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes]
            if not active_tasks:
                break

            # ---------- Step 1: Single-Task Optimization (BO-LCB) ----------
            solutions_in = {}
            improvements_in = {}

            for i in active_tasks:
                candidate_uni, gp_model = bo_next_point_lcb(
                    d_max, decs[i], objs[i], data_type=data_type
                )
                solutions_in[i] = candidate_uni
                gp_models[i] = gp_model

                imp_in = _improvement_internal(
                    gp_model, decs[i], objs[i], candidate_uni, data_type=data_type
                )
                improvements_in[i] = imp_in

            # ---------- Step 2: Bayesian Competitive Knowledge Transfer ----------
            solutions_candidate = {}
            trigger_transfer = (total_nfes % gen_gap_total == 0)

            for target_idx in active_tasks:
                if trigger_transfer:
                    # Save old MUs/SIGMA2s for rollback
                    MUs_old = MUs.copy()
                    SIGMA2s_old = SIGMA2s.copy()

                    # Execute knowledge competition
                    (solution_ex, improvement_ex, source_idx,
                     impn) = _knowledge_competition(
                        decs, objs, gp_models, target_idx,
                        MUs, SIGMA2s, sigma2_sim, data_type
                    )

                    if improvements_in[target_idx] >= improvement_ex:
                        # Internal wins: use internal solution, rollback Bayesian update
                        solutions_candidate[target_idx] = solutions_in[target_idx]
                        transfer_actions[target_idx].append(0)
                        MUs[source_idx, target_idx] = MUs_old[source_idx, target_idx]
                        SIGMA2s[source_idx, target_idx] = SIGMA2s_old[source_idx, target_idx]
                    else:
                        # External wins: use transferred solution
                        solutions_candidate[target_idx] = solution_ex
                        transfer_actions[target_idx].append(source_idx + 1)

                        # Hidden evaluation for Bayesian update (MATLAB behavior)
                        cand_real = solution_ex[:, :dims[target_idx]]
                        obj_hidden, _ = evaluation_single(problem, cand_real, target_idx)
                        obj_hidden_val = obj_hidden.flatten()[0]

                        # Compute actual transferability
                        min_target = np.min(objs[target_idx])
                        max_target = np.max(objs[target_idx])
                        if max_target != 0:
                            imp_actual = (min_target - obj_hidden_val) / max_target
                        else:
                            imp_actual = 0.0

                        if impn != 0:
                            mu_transfer = imp_actual / impn
                        else:
                            mu_transfer = 0.0

                        # Second Bayesian update with decaying variance
                        sigma2_transfer = sigma2_sim * np.exp(
                            -(total_nfes - n_initial * nt) / gen_gap_total
                        )
                        _bayesian_update_gaussian(
                            MUs, SIGMA2s, source_idx, target_idx,
                            mu_transfer, sigma2_transfer
                        )
                else:
                    solutions_candidate[target_idx] = solutions_in[target_idx]
                    transfer_actions[target_idx].append(0)

            # ---------- Step 3: Evaluation and Database Update ----------
            for target_idx in active_tasks:
                candidate_uni = solutions_candidate[target_idx]

                # Ensure uniqueness (Chebyshev distance, matching MATLAB)
                candidate_uni = _ensure_unique(candidate_uni, decs[target_idx])

                # Transform to real space for evaluation
                candidate_real = candidate_uni[:, :dims[target_idx]]
                obj, _ = evaluation_single(problem, candidate_real, target_idx)

                # Update database
                decs[target_idx] = np.vstack([decs[target_idx], candidate_uni])
                objs[target_idx] = np.vstack([objs[target_idx], obj])

                nfes_per_task[target_idx] += 1
                total_nfes += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # ======================== Phase 3: Build Results (real space) ========================
        all_decs_uni, all_objs = build_staircase_history(decs, objs, k=1)
        final_decs_real = []
        for i in range(nt):
            task_decs_real = []
            for dec_uni in all_decs_uni[i]:
                dec_real = dec_uni[:, :dims[i]]
                task_decs_real.append(dec_real)
            final_decs_real.append(task_decs_real)

        results = build_save_results(
            all_decs=final_decs_real,
            all_objs=all_objs,
            runtime=runtime,
            max_nfes=nfes_per_task,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        results.transfer_states = transfer_actions

        return results


# ==================== Utility Functions ====================

def _ensure_unique(candidate, decs, epsilon=5e-3, max_trials=50):
    """Ensure uniqueness using Chebyshev distance (matching MATLAB)."""
    scales = np.linspace(0.1, 1.0, max_trials)
    for trial in range(max_trials):
        distances = np.max(np.abs(candidate - decs), axis=1)
        min_dist = np.min(distances)
        if min_dist >= epsilon:
            break
        perturbation = scales[trial] * (np.random.rand(candidate.shape[1]) - 0.5)
        candidate = np.clip(candidate + perturbation, 0, 1)
    return candidate


def _bayesian_update_gaussian(MUs, SIGMA2s, source_idx, target_idx,
                              observation, sigma2_obs):
    """Standard conjugate Gaussian Bayesian update: MU/SIGMA2 in-place."""
    mu_old = MUs[source_idx, target_idx]
    s2_old = SIGMA2s[source_idx, target_idx]
    MUs[source_idx, target_idx] = (
        (mu_old * sigma2_obs + observation * s2_old) / (sigma2_obs + s2_old)
    )
    SIGMA2s[source_idx, target_idx] = (
        s2_old * sigma2_obs / (s2_old + sigma2_obs)
    )


def _knowledge_competition(decs, objs, gp_models, target_idx,
                           MUs, SIGMA2s, sigma2_sim, data_type):
    """
    Execute Bayesian Competitive Knowledge Transfer (matching MATLAB knowledge_competition.m).

    Returns (solution_ex, improvement_ex, source_idx, impn).
    MUs and SIGMA2s are updated IN-PLACE with similarity observations.
    """
    nt = len(decs)
    Xt = decs[target_idx]
    Yt = objs[target_idx].flatten()
    dim = Xt.shape[1]

    improvements = np.full(nt, -np.inf)
    impns = np.full(nt, -np.inf)
    solutions_external = [None] * nt

    for source_idx in range(nt):
        if source_idx == target_idx:
            continue

        Xs = decs[source_idx]
        Ys = objs[source_idx].flatten()

        # Compute similarity (SSRC) using source GP predictions on target data
        similarity, objs_val = _compute_similarity_ssrc(
            gp_models[source_idx], Xt, Yt, data_type
        )

        # Bayesian update with similarity as observation
        if np.isnan(MUs[source_idx, target_idx]):
            # First encounter: initialize
            MUs[source_idx, target_idx] = similarity
            SIGMA2s[source_idx, target_idx] = sigma2_sim
        else:
            # Subsequent: conjugate Gaussian update
            _bayesian_update_gaussian(
                MUs, SIGMA2s, source_idx, target_idx,
                similarity, sigma2_sim
            )

        # Sample transferability from posterior
        transferability = np.random.normal(
            MUs[source_idx, target_idx],
            np.sqrt(SIGMA2s[source_idx, target_idx])
        )

        # Compute external improvement (matching MATLAB improvement_external.m)
        imp, impn = _improvement_external(Ys, objs_val, Yt, transferability)
        improvements[source_idx] = imp
        impns[source_idx] = impn

        # Best solution from source task
        best_idx = np.argmin(Ys)
        solutions_external[source_idx] = Xs[best_idx:best_idx + 1]

    # Select best source
    best_source = np.argmax(improvements)
    return (solutions_external[best_source], improvements[best_source],
            best_source, impns[best_source])


def _compute_similarity_ssrc(gp_source, Xt, Yt, data_type):
    """
    Compute Surrogate-based Spearman Rank Correlation (SSRC).

    The GP from bo_next_point_lcb is trained on -objs, so predictions are negated back.
    Returns (similarity, objs_val) where objs_val is in ORIGINAL (non-negated) space.
    """
    Xt_torch = torch.tensor(Xt, dtype=data_type)
    with torch.no_grad():
        posterior = gp_source.posterior(Xt_torch)
        # GP predicts -obj, negate back to original space
        objs_val = -posterior.mean.cpu().numpy().flatten()

    # Spearman rank correlation
    if len(Yt) < 3:
        return 0.0, objs_val

    similarity, _ = spearmanr(Yt, objs_val)
    if np.isnan(similarity):
        similarity = 0.0

    return similarity, objs_val


def _improvement_external(objs_source, objs_val, objs_target, transferability):
    """
    Calculate external improvement (matching MATLAB improvement_external.m).

    Parameters
    ----------
    objs_source : source task true objectives (flattened)
    objs_val : source GP predictions on target data (original space)
    objs_target : target task true objectives (flattened)
    transferability : sampled Bayesian transferability

    Returns
    -------
    imp : external improvement
    impn_source : normalized source improvement (used for actual transferability calc)
    """
    combined = np.concatenate([objs_source, objs_val])
    max_combined = np.max(combined)

    if max_combined == 0:
        return -np.inf, 0.0

    impn_source = (np.min(objs_val) - np.min(objs_source)) / max_combined

    if impn_source < 0:
        return -np.inf, impn_source

    imp = transferability * impn_source * np.max(objs_target)
    return imp, impn_source


def _improvement_internal(gp_model, decs, objs, candidate,
                          data_type=torch.float, kappa=2.0):
    """
    Calculate internal improvement (LCB-based).

    The GP predicts -obj, so we negate back to compute LCB in original space:
    LCB_orig = -mu_gp - kappa * sigma_gp
    """
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

    # LCB in original space: obj ≈ -mu_gp, so LCB = -mu_gp - kappa * sigma
    lcb_db = -mu_db - kappa * std_db
    lcb_cand = -mu_cand - kappa * std_cand

    improvement = np.min(lcb_db) - lcb_cand[0]
    return improvement

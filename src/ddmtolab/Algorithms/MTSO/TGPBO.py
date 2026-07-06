"""
TGPBO: Transfer GP Bayesian Optimization

This module implements TGPBO for expensive multi-task single-objective optimization.
It combines GP surrogate modeling augmented with cross-task data with adaptive
competitive knowledge transfer for effective information sharing between tasks.

Strategy 1 - SSRC-Weighted Multi-Source Augmented BO:
    Builds GP surrogates augmented with source task data from correlated tasks.
    Task correlation is estimated via Surrogate-based Spearman Rank Correlation (SSRC),
    computed using individual GP cross-predictions. Source data is selected
    proportionally to SSRC similarity and normalized to target objective scale
    before augmentation. This enriches the GP training set, improving surrogate
    accuracy in data-scarce expensive optimization settings.

Strategy 2 - SSRC-Guided Competitive Knowledge Transfer:
    Uses SSRC-based similarity for adaptive transfer decisions. Each iteration,
    an internal EI-optimized candidate competes with transfer candidates derived
    from correlated source tasks' best solutions. Competition uses GP-predicted
    objectives on the augmented model. An adaptive threshold, updated based on
    transfer success rate, controls when transfer is attempted.

Notes
-----
Author: Algorithm Development
Date: 2026.02
Version: 2.0
"""
import time
import warnings
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict, bo_next_point

warnings.filterwarnings("ignore")


class TGPBO:
    """
    Transfer GP Bayesian Optimization for expensive multi-task optimization.

    Uses SSRC-weighted data augmentation to enrich GP training data with
    correlated source task information, and competitive knowledge transfer
    guided by surrogate-based similarity estimation.
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
                 ssrc_interval=8,
                 K_max=30,
                 ssrc_threshold=0.1,
                 transfer_threshold_init=0.2,
                 save_data=True, save_path='./Data',
                 name='TGPBO', disable_tqdm=True):
        """
        Initialize TGPBO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance.
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50).
        max_nfes : int or List[int], optional
            Maximum function evaluations per task (default: 100).
        NC : int
            Number of nearest neighbor points for local data (default: 60).
        NR : int
            Number of most recent points for local data (default: 60).
        ssrc_interval : int
            Recompute SSRC every this many iterations (default: 8).
        K_max : int
            Maximum source points for data augmentation (default: 30).
        ssrc_threshold : float
            Minimum SSRC for source data augmentation (default: 0.1).
        transfer_threshold_init : float
            Initial threshold for competitive transfer (default: 0.2).
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.NC = NC
        self.NR = NR
        self.ssrc_interval = ssrc_interval
        self.K_max = K_max
        self.ssrc_threshold = ssrc_threshold
        self.transfer_threshold = transfer_threshold_init
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """Execute the TGPBO algorithm."""
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # ===== Initialization =====
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes = [n_initial_per_task[i] for i in range(nt)]

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes),
                    desc=self.name, disable=self.disable_tqdm)

        # State variables
        ssrc_matrix = np.zeros((nt, nt))
        transfer_thresholds = np.full(nt, self.transfer_threshold)
        transfer_successes = np.zeros(nt)
        transfer_attempts = np.zeros(nt)

        iteration = 0
        while sum(nfes) < total_nfes:
            active_tasks = [i for i in range(nt)
                            if nfes[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # ===== Periodically recompute SSRC (Strategy 1) =====
            if iteration % self.ssrc_interval == 0:
                ssrc_matrix = self._compute_all_ssrc(decs, objs, dims, nt)

            # ===== Process each active task =====
            for i in active_tasks:
                if nfes[i] >= max_nfes_per_task[i]:
                    continue

                # Strategy 1: Build augmented data and GP
                aug_decs, aug_objs = self._build_augmented_data(
                    i, decs, objs, dims, ssrc_matrix, nt
                )

                try:
                    candidate_internal, gp_aug = self._build_gp_and_optimize_ei(
                        dims[i], aug_decs, aug_objs
                    )
                except Exception:
                    try:
                        candidate_internal = bo_next_point(
                            dims[i], decs[i], objs[i]
                        )
                        gp_aug = None
                    except Exception:
                        candidate_internal = np.random.rand(1, dims[i])
                        gp_aug = None

                # ===== Strategy 2: Competitive transfer =====
                candidate = candidate_internal
                is_transfer = False

                source_ssrcs = {j: ssrc_matrix[i, j]
                                for j in range(nt) if j != i}
                best_source = max(source_ssrcs, key=source_ssrcs.get)
                best_ssrc = source_ssrcs[best_source]

                if best_ssrc > transfer_thresholds[i] and gp_aug is not None:
                    candidate_transfer = self._get_transfer_candidate(
                        i, best_source, decs, objs, dims
                    )
                    if candidate_transfer is not None:
                        pred_int = self._gp_predict_mean(
                            gp_aug, candidate_internal
                        )
                        pred_trf = self._gp_predict_mean(
                            gp_aug, candidate_transfer
                        )
                        if pred_trf < pred_int:
                            candidate = candidate_transfer
                            is_transfer = True
                            transfer_attempts[i] += 1

                # Ensure uniqueness
                candidate = remove_duplicates(candidate, decs[i])
                if len(candidate) == 0:
                    candidate = np.clip(
                        candidate_internal
                        + 0.01 * np.random.randn(1, dims[i]),
                        0, 1
                    )

                # Real evaluation
                obj_new, _ = evaluation_single(problem, candidate, i)

                # Track transfer success
                if is_transfer and obj_new.flatten()[0] < np.min(objs[i]):
                    transfer_successes[i] += 1

                # Update database
                decs[i] = np.vstack([decs[i], candidate])
                objs[i] = np.vstack([objs[i], obj_new])
                nfes[i] += 1
                pbar.update(1)

                # Adapt transfer threshold
                if transfer_attempts[i] > 0:
                    success_rate = (transfer_successes[i]
                                    / transfer_attempts[i])
                    if success_rate > 0.3:
                        transfer_thresholds[i] *= 0.95
                    else:
                        transfer_thresholds[i] *= 1.05
                    transfer_thresholds[i] = np.clip(
                        transfer_thresholds[i], 0.05, 0.5
                    )

            iteration += 1

        pbar.close()
        runtime = time.time() - start_time

        # Build results
        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data
        )
        return results

    # ================================================================
    # Strategy 1: SSRC-Weighted Multi-Source Augmented BO
    # ================================================================

    def _compute_all_ssrc(self, decs, objs, dims, nt):
        """
        Compute pairwise SSRC matrix using individual GP cross-predictions.

        For each pair (i, j), builds a GP on task j's data, predicts at task
        i's evaluated points, and computes Spearman rank correlation between
        predicted and actual objectives.

        Parameters
        ----------
        decs : list of np.ndarray
            Decision variables per task.
        objs : list of np.ndarray
            Objective values per task.
        dims : list of int
            Dimensions per task.
        nt : int
            Number of tasks.

        Returns
        -------
        ssrc_matrix : np.ndarray, shape (nt, nt)
            Pairwise SSRC values.
        """
        ssrc_matrix = np.zeros((nt, nt))
        individual_gps = [None] * nt

        # Build individual GP for each task on all data
        for t in range(nt):
            try:
                individual_gps[t] = gp_build(decs[t], objs[t])
            except Exception:
                pass

        # Compute pairwise SSRC
        for i in range(nt):
            for j in range(nt):
                if i == j or individual_gps[j] is None:
                    continue
                try:
                    # Align target data to source GP's expected dimension
                    if dims[j] > dims[i]:
                        padding = np.full(
                            (len(decs[i]), dims[j] - dims[i]), 0.5
                        )
                        query_decs = np.hstack([decs[i], padding])
                    elif dims[j] < dims[i]:
                        query_decs = decs[i][:, :dims[j]]
                    else:
                        query_decs = decs[i]

                    # Source GP predicts on target's evaluated points
                    pred_objs, _ = gp_predict(individual_gps[j], query_decs)
                    pred_flat = pred_objs.flatten()
                    target_flat = objs[i].flatten()

                    if len(target_flat) >= 3:
                        corr, _ = spearmanr(target_flat, pred_flat)
                        if not np.isnan(corr):
                            ssrc_matrix[i, j] = max(corr, 0.0)
                except Exception:
                    pass

        return ssrc_matrix

    def _build_augmented_data(self, target, decs, objs, dims,
                              ssrc_matrix, nt):
        """
        Build augmented training data with SSRC-weighted source data.

        Combines local target data with selected source data from correlated
        tasks. Source objectives are normalized to target's distribution.

        Parameters
        ----------
        target : int
            Target task index.
        decs : list of np.ndarray
            Decision variables per task.
        objs : list of np.ndarray
            Objective values per task.
        dims : list of int
            Dimensions per task.
        ssrc_matrix : np.ndarray
            Pairwise SSRC values.
        nt : int
            Number of tasks.

        Returns
        -------
        aug_decs : np.ndarray, shape (n_aug, dims[target])
        aug_objs : np.ndarray, shape (n_aug, 1)
        """
        # Local target data
        local_t_decs, local_t_objs = self._select_local_data(
            decs[target], objs[target]
        )
        aug_decs = local_t_decs.copy()
        aug_objs = local_t_objs.copy()

        # Target distribution stats for normalization
        t_mean = np.mean(local_t_objs)
        t_std = np.std(local_t_objs) + 1e-10

        # Target's current best for distance computation
        best_idx = np.argmin(objs[target].flatten())
        x_best = decs[target][best_idx]

        for j in range(nt):
            if j == target:
                continue
            ssrc = ssrc_matrix[target, j]
            if ssrc < self.ssrc_threshold:
                continue

            # Source points proportional to SSRC
            K = max(3, int(ssrc * self.K_max))
            d_min = min(dims[target], dims[j])
            source_decs_j = decs[j][:, :d_min]

            # Select K nearest source points to target's best
            distances = cdist(
                x_best[:d_min].reshape(1, -1), source_decs_j
            )[0]
            k_actual = min(K, len(source_decs_j))
            nearest_idx = np.argsort(distances)[:k_actual]

            sel_decs = source_decs_j[nearest_idx]
            sel_objs = objs[j][nearest_idx].copy()

            # Normalize source objectives to target's distribution
            s_mean = np.mean(sel_objs)
            s_std = np.std(sel_objs) + 1e-10
            sel_objs_norm = (sel_objs - s_mean) / s_std * t_std + t_mean
            sel_objs_norm = np.clip(
                sel_objs_norm,
                t_mean - 3 * t_std,
                t_mean + 3 * t_std
            )

            # Pad source decs if target has more dimensions
            if dims[j] < dims[target]:
                pad = np.full(
                    (len(sel_decs), dims[target] - dims[j]), 0.5
                )
                sel_decs = np.hstack([sel_decs, pad])

            aug_decs = np.vstack([aug_decs, sel_decs])
            aug_objs = np.vstack([aug_objs, sel_objs_norm])

        return aug_decs, aug_objs

    def _select_local_data(self, decs_t, objs_t):
        """Select local data: NC nearest to best + NR most recent."""
        n = len(decs_t)
        best_idx = np.argmin(objs_t.flatten())
        x_best = decs_t[best_idx]

        nc = min(self.NC, n)
        distances = cdist(x_best.reshape(1, -1), decs_t)[0]
        nearest_idx = np.argsort(distances)[:nc]

        nr = min(self.NR, n)
        recent_idx = np.arange(max(0, n - nr), n)

        selected_idx = np.unique(np.concatenate([nearest_idx, recent_idx]))
        return decs_t[selected_idx], objs_t[selected_idx]

    def _build_gp_and_optimize_ei(self, dim, aug_decs, aug_objs):
        """
        Build GP on augmented data and optimize LogEI.

        Returns
        -------
        candidate : np.ndarray, shape (1, dim)
        gp : SingleTaskGP
            Fitted GP model (reusable for competition predictions).
        """
        bounds = torch.stack([
            torch.zeros(dim, dtype=torch.float),
            torch.ones(dim, dtype=torch.float)
        ], dim=0)

        train_X = torch.tensor(aug_decs, dtype=torch.float)
        train_Y = torch.tensor(-aug_objs, dtype=torch.float)
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = train_Y.max()
        logEI = LogExpectedImprovement(model=gp, best_f=best_f)
        candidate, _ = optimize_acqf(
            logEI, bounds=bounds, q=1,
            num_restarts=5, raw_samples=20
        )

        return candidate.detach().cpu().numpy(), gp

    def _gp_predict_mean(self, gp, X):
        """Predict mean objective using GP (original minimization scale)."""
        X_torch = torch.tensor(X, dtype=torch.float)
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(X_torch)
            pred_mean = -posterior.mean.cpu().numpy().flatten()[0]
        return pred_mean

    # ================================================================
    # Strategy 2: SSRC-Guided Competitive Knowledge Transfer
    # ================================================================

    def _get_transfer_candidate(self, target, source, decs, objs, dims):
        """
        Generate transfer candidate from source task's best solution.

        Dimension-aligns and adds small perturbation for exploration.
        """
        try:
            best_idx = np.argmin(objs[source].flatten())
            source_best = decs[source][best_idx].copy()

            if len(source_best) < dims[target]:
                source_best = np.pad(
                    source_best,
                    (0, dims[target] - len(source_best)),
                    constant_values=0.5
                )
            elif len(source_best) > dims[target]:
                source_best = source_best[:dims[target]]

            candidate = np.clip(source_best, 0, 1).reshape(1, -1)
            perturbation = 0.02 * (np.random.rand(1, dims[target]) - 0.5)
            candidate = np.clip(candidate + perturbation, 0, 1)
            return candidate
        except Exception:
            return None

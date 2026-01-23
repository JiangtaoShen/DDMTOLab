"""
Bayesian Optimization with Lower Confidence Bound and Competitive Knowledge Transfer (BO-LCB-CKT)

This module implements BO-LCB-CKT for expensive sequential transfer optimization problems.

Key Features:
- Task 0: Target task (to be optimized)
- Tasks 1:k: Source tasks (provide knowledge base)
- Only task 0 is actively optimized; source tasks are pre-optimized once

References
----------
    [1] Xue, Xiaoming, et al. "Surrogate-assisted search with competitive knowledge transfer for expensive optimization." IEEE Transactions on Evolutionary Computation (2024).

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.09
Version: 1.0
"""
from tqdm import tqdm
import torch
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import minimize
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict, bo_next_point_lcb
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings
import time

warnings.filterwarnings("ignore")


class BO_LCB_CKT:
    """
    Bayesian Optimization with Lower Confidence Bound and Competitive Knowledge Transfer.

    This algorithm optimizes task 0 (target) by leveraging knowledge from tasks 1:k (sources).
    Source tasks are pre-optimized once to build a knowledge base, then only the target task
    is optimized with competitive knowledge transfer.

    **Now supports tasks with different dimensions using space_transfer.**

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
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
                 gen_gap=10, ada_flag=False,
                 save_data=True, save_path='./TestData',
                 name='BO_LCB_CKT_test', disable_tqdm=False):
        """
        Initialize BO-LCB-CKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance.
            Task 0 is the target task, tasks 1:k are source tasks.
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: [500, 100, 100, ...])
            - First value: budget for target task (task 0)
            - Remaining values: budget for source tasks
        gen_gap : int, optional
            Knowledge transfer trigger frequency (default: 10)
        ada_flag : bool, optional
            Whether to enable task adaptation (default: False)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'BO_LCB_CKT_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: False)
        """
        self.problem = problem
        self.nt = problem.n_tasks
        self.dims = problem.dims
        self.dim_max = max(self.dims)  # Maximum dimension across all tasks
        self.dim_target = self.dims[0]  # Target task dimension
        self.n_initial = n_initial if n_initial is not None else 20

        # Default budget: target task gets more evaluations
        if max_nfes is None:
            self.max_nfes = [200] + [50] * (self.nt - 1)
        else:
            self.max_nfes = max_nfes

        self.gen_gap = gen_gap
        self.ada_flag = ada_flag
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm
        self.data_type = torch.float

    def optimize(self):
        """
        Execute the BO-LCB-CKT algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = self.nt

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # ===================================================================
        # Phase 1: Initialize all tasks in real space
        # ===================================================================
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Reorganize initial data into task-specific history lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        # Unified progress bar
        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        # ===================================================================
        # Phase 2: Build knowledge base from source tasks (tasks 1:k)
        # ===================================================================
        for i in range(1, nt):
            # Get task dimension
            dim_i = self.dims[i]

            # Optimize source task with BO-LCB (in real space)
            while nfes_per_task[i] < max_nfes_per_task[i]:
                candidate, _ = bo_next_point_lcb(
                    dim_i, decs[i], objs[i], data_type=self.data_type
                )

                # Evaluate (in real space)
                obj, _ = evaluation_single(problem, candidate, i)

                # Update dataset
                decs[i], objs[i] = vstack_groups((decs[i], candidate), (objs[i], obj))

                # Store cumulative history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

                nfes_per_task[i] += 1
                pbar.update(1)

        # ===================================================================
        # Phase 3: Convert to unified space and build GP models
        # ===================================================================
        # Convert all source tasks to unified space (pad to dim_max)
        decs_uni = space_transfer(problem, decs, type='uni', padding='zero')
        objs_uni = [objs[i].copy() for i in range(nt)]

        # Build knowledge base and surrogates in unified space
        knowledge_base = []
        surrogates_source = []

        for i in range(1, nt):
            knowledge_base.append({
                'task_id': i,
                'solutions_uni': decs_uni[i],
                'solutions_real': decs[i],
                'objs': objs[i],
                'best_solution_uni': decs_uni[i][np.argmin(objs[i])],
                'best_solution_real': decs[i][np.argmin(objs[i])],
                'best_obj': np.min(objs[i])
            })

            # Build GP in unified space
            gp = gp_build(decs_uni[i], objs[i], data_type=self.data_type)
            surrogates_source.append(gp)

        # ===================================================================
        # Phase 4: Task adaptation (optional) - in unified space
        # ===================================================================
        ada_vectors = np.zeros((len(knowledge_base), self.dim_max))
        if self.ada_flag:
            ada_vectors = self._solution_adaptation(
                surrogates_source, decs_uni[0], objs[0]
            )

        # ===================================================================
        # Phase 5: Optimize target task (task 0) with CKT
        # ===================================================================
        target_idx = 0
        transfer_states = []

        while nfes_per_task[target_idx] < max_nfes_per_task[target_idx]:
            # ============ Step 1: Internal Acquisition (BO-LCB) ============
            # Work in real space for target task
            solution_in, improvement_in = self._internal_acquisition(
                decs[target_idx], objs[target_idx]
            )

            # ============ Step 2: Knowledge Competition (periodic) ============
            if nfes_per_task[target_idx] % self.gen_gap == 0:
                # Work in unified space for knowledge transfer
                solution_ex_uni, improvement_ex, source_id = self._knowledge_competition(
                    decs_uni[target_idx], objs[target_idx], knowledge_base,
                    surrogates_source, ada_vectors
                )

                # Convert external solution from unified to real space
                solution_ex = solution_ex_uni[:self.dim_target]

                # Competitive selection
                if improvement_in >= improvement_ex:
                    solution_candidate = solution_in
                    transfer_states.append(0)  # Internal win
                else:
                    solution_candidate = solution_ex
                    transfer_states.append(source_id + 1)  # External win
            else:
                solution_candidate = solution_in
                transfer_states.append(0)

            # ============ Step 3: Ensure Diversity in real space ============
            solution_candidate = self._ensure_diversity(
                solution_candidate, decs[target_idx], epsilon=5e-3
            )

            # ============ Step 4: Evaluate in real space ============
            obj_candidate, _ = evaluation_single(
                problem, solution_candidate.reshape(1, -1), target_idx
            )

            # ============ Step 5: Update Database ============
            # Update real space data
            decs[target_idx], objs[target_idx] = vstack_groups(
                (decs[target_idx], solution_candidate), (objs[target_idx], obj_candidate)
            )

            # Update unified space data for target task
            solution_candidate_uni = np.pad(
                solution_candidate,
                (0, self.dim_max - self.dim_target),
                mode='constant',
                constant_values=0
            )
            decs_uni[target_idx] = np.vstack([decs_uni[target_idx], solution_candidate_uni])

            # Store cumulative history (real space)
            append_history(all_decs[target_idx], decs[target_idx],
                           all_objs[target_idx], objs[target_idx])

            nfes_per_task[target_idx] += 1
            pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time


        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        # Add transfer statistics
        results.transfer_states = transfer_states

        return results


    def _solution_adaptation(self, surrogates_source, solutions_target_uni, objs_target):
        """
        Perform task adaptation in unified space.

        Parameters
        ----------
        surrogates_source : List[SingleTaskGP]
            List of source task GP models (trained in unified space)
        solutions_target_uni : ndarray
            Target task decision variables (in unified space, dim_max)
        objs_target : ndarray
            Target task objectives

        Returns
        -------
        ada_vectors : ndarray
            Adaptation vectors for each source task (k x dim_max)
        """
        num_sources = len(surrogates_source)
        ada_vectors = np.zeros((num_sources, self.dim_max))

        for i in range(num_sources):
            def objective(theta):
                # Adapt target solutions in unified space
                X_adapted = solutions_target_uni - theta
                X_adapted = np.clip(X_adapted, 0, 1)

                try:
                    # Predict using source GP
                    objs_val, _ = gp_predict(
                        surrogates_source[i], X_adapted, data_type=self.data_type
                    )
                    objs_val = objs_val.flatten()
                except:
                    objs_val = np.full(len(X_adapted), np.mean(objs_target))

                # Compute ranks
                rank_target = self._compute_ranks(objs_target.flatten())
                rank_val = self._compute_ranks(objs_val)

                # Spearman correlation
                rho, _ = spearmanr(rank_target, rank_val)
                if np.isnan(rho):
                    rho = 0.0

                # Regularization term
                alpha_theta = 1.0 - np.max(np.abs(theta))

                # Adaptation-based similarity
                s_adapted = alpha_theta * rho

                return -s_adapted

            # Multi-start optimization
            no_local_opt = 10
            bounds = [(-1.0, 1.0)] * self.dim_max  # Use dim_max

            best_theta = None
            best_obj = np.inf

            # Try zero vector first
            try:
                result_zero = minimize(
                    objective, np.zeros(self.dim_max),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 100, 'disp': False}
                )
                if result_zero.fun < best_obj:
                    best_obj = result_zero.fun
                    best_theta = result_zero.x
            except:
                pass

            # Random starts
            for _ in range(no_local_opt - 1):
                x0 = np.random.uniform(-0.5, 0.5, self.dim_max)
                try:
                    result = minimize(
                        objective, x0,
                        method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 100, 'disp': False}
                    )
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_theta = result.x
                except:
                    continue

            if best_theta is not None:
                ada_vectors[i] = best_theta

        return ada_vectors

    def _internal_acquisition(self, decs, objs):
        """
        Internal acquisition using BO-LCB (in real space for target task).

        Parameters
        ----------
        decs : ndarray
            Current decision variables (real space, dim_target)
        objs : ndarray
            Current objectives

        Returns
        -------
        solution : ndarray
            Selected candidate solution (real space, dim_target)
        improvement : float
            Estimated internal improvement
        """
        # Select next point using BO-LCB in real space
        solution, gp = bo_next_point_lcb(
            self.dim_target, decs, objs, data_type=self.data_type
        )

        # Compute internal improvement
        improvement = self._improvement_internal(decs, objs, solution, gp)

        return solution.flatten(), improvement

    def _improvement_internal(self, decs, objs, solution_promising, gp):
        """
        Estimate internal improvement (same as before).
        """
        try:
            kappa = 2.0

            mu_db, std_db = gp_predict(gp, decs, data_type=self.data_type)
            lcb_db = mu_db.flatten() - kappa * std_db.flatten()

            mu_pr, std_pr = gp_predict(
                gp, solution_promising.reshape(1, -1), data_type=self.data_type
            )
            lcb_pr = mu_pr[0, 0] - kappa * std_pr[0, 0]

            improvement = np.min(lcb_db) - lcb_pr

            return improvement
        except:
            return 0.0

    def _knowledge_competition(self, decs_target_uni, objs_target, knowledge_base,
                               surrogates_source, ada_vectors):
        """
        Knowledge competition in unified space.

        Parameters
        ----------
        decs_target_uni : ndarray
            Target task decision variables (unified space, dim_max)
        objs_target : ndarray
            Target task objectives
        knowledge_base : List[dict]
            Source task data
        surrogates_source : List[SingleTaskGP]
            Source task GP models (trained in unified space)
        ada_vectors : ndarray
            Adaptation vectors (k x dim_max)

        Returns
        -------
        solution_external_uni : ndarray
            Selected external candidate (unified space, dim_max)
        improvement_max : float
            Maximum external improvement
        source_id : int
            ID of the winning source task
        """
        num_sources = len(knowledge_base)
        improvements = np.zeros(num_sources)
        solutions_uni = np.zeros((num_sources, self.dim_max))

        ranks_target = self._compute_ranks(objs_target.flatten())

        for i in range(num_sources):
            # Adapt target solutions in unified space
            X_adapted = decs_target_uni - ada_vectors[i]
            X_adapted = np.clip(X_adapted, 0, 1)

            try:
                objs_val, _ = gp_predict(
                    surrogates_source[i], X_adapted, data_type=self.data_type
                )
                objs_val = objs_val.flatten()
            except:
                objs_val = np.full(len(X_adapted), np.mean(objs_target))

            ranks_val = self._compute_ranks(objs_val)

            rho, _ = spearmanr(ranks_target, ranks_val)
            if np.isnan(rho):
                rho = 0.0

            if self.ada_flag and np.any(ada_vectors[i] != 0):
                alpha_theta = 1.0 - np.max(np.abs(ada_vectors[i]))
                similarity = alpha_theta * rho
            else:
                similarity = rho

            # Estimate improvement
            objs_source = knowledge_base[i]['objs'].flatten()
            improvement = self._improvement_external(
                objs_source, objs_val, objs_target.flatten(), similarity
            )
            improvements[i] = improvement

            # Prepare adapted solution (in unified space)
            x_source_best_uni = knowledge_base[i]['best_solution_uni']
            x_adapted = x_source_best_uni + ada_vectors[i]
            x_adapted = np.clip(x_adapted, 0, 1)
            solutions_uni[i] = x_adapted

        # Select best source
        source_id = np.argmax(improvements)
        solution_external_uni = solutions_uni[source_id]
        improvement_max = improvements[source_id]

        return solution_external_uni, improvement_max, source_id

    def _improvement_external(self, objs_source, objs_val, objs_target, similarity):
        """
        Estimate external improvement (same as before).
        """
        similarity = max(0, similarity)

        if similarity < 1e-6:
            return 0.0

        objs_source_sorted = np.sort(objs_source)[::-1]
        objs_target_sorted = np.sort(objs_target)[::-1]

        n_source = len(objs_source_sorted)
        n_target = len(objs_target_sorted)

        # Fit exponential decay for source
        gamma_o_source = np.min(objs_source_sorted)
        t_source = np.arange(1, n_source + 1)
        y_shifted_source = objs_source_sorted - gamma_o_source + 1e-10

        try:
            coeffs_source = np.polyfit(t_source, np.log(y_shifted_source), 1)
            lambda_source = -coeffs_source[0]
            gamma_i_source = np.exp(coeffs_source[1])

            if lambda_source <= 0:
                return 0.0
        except:
            return 0.0

        # Fit exponential decay for target
        gamma_o_target = np.min(objs_target_sorted)
        t_target = np.arange(1, n_target + 1)
        y_shifted_target = objs_target_sorted - gamma_o_target + 1e-10

        try:
            coeffs_target = np.polyfit(t_target, np.log(y_shifted_target), 1)
            lambda_target = -coeffs_target[0]
            gamma_i_target = np.exp(coeffs_target[1])

            if lambda_target <= 0:
                return 0.0
        except:
            return 0.0

        # Calculate time interval
        min_val = np.min(objs_val)

        try:
            tau_v = (1 / lambda_source) * np.log(
                gamma_i_source / (min_val - gamma_o_source + 1e-10)
            )
            tau_v = max(0, tau_v)
        except:
            tau_v = 0

        tau_max = n_source
        delta_tau = tau_max - tau_v

        # Analogize to target
        tau_current = n_target
        tau_analogized = similarity * (tau_current + delta_tau)

        predicted_obj = gamma_o_target + gamma_i_target * np.exp(-lambda_target * tau_analogized)
        improvement = similarity * (np.min(objs_target) - predicted_obj)

        return max(0, improvement)

    def _compute_ranks(self, values):
        """
        Compute ranks (same as before).
        """
        n = len(values)
        ranks = np.zeros(n)

        for i in range(n):
            ranks[i] = np.sum(values < values[i]) + 1

        return ranks

    def _ensure_diversity(self, solution, database, epsilon=5e-3, max_trials=50):
        """
        Ensure diversity (works in real space for target task).
        """
        scales = np.linspace(0.1, 1.0, max_trials)

        for trial in range(max_trials):
            distances = np.max(np.abs(database - solution), axis=1)
            min_dist = np.min(distances)

            if min_dist >= epsilon:
                break

            scale = scales[trial]
            perturbation = scale * (np.random.rand(self.dim_target) - 0.5)
            solution = solution + perturbation
            solution = np.clip(solution, 0, 1)

        return solution
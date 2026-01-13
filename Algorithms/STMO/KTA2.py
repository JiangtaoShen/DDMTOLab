"""
Kriging-Assisted Two-Archive Evolutionary Algorithm (KTA2)

This module implements KTA2 for expensive multi-objective/many-objective optimization problems.

References
----------
    [1] Z. Song, H. Wang, C. He, and Y. Jin. A Kriging-assisted two-archive evolutionary algorithm \
        for expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, \
        2021, 25(6): 1013-1027.

Notes
-----
Author: Based on Zhenshou Song's MATLAB implementation
Adapted for DDMTOLab
Date: 2025.01
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.stats import wilcoxon
from scipy.spatial.distance import pdist, squareform, cdist
from Methods.Algo_Methods.algo_utils import *
from Methods.Algo_Methods.bo_utils import gp_build, gp_predict
from Algorithms.STMO.TwoArch2 import TwoArch2
from Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class KTA2:
    """
    Kriging-Assisted Two-Archive Evolutionary Algorithm for expensive many-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, tau=0.75, phi=0.1, wmax=10, mu=5,
                 save_data=True, save_path='./TestData', name='KTA2_test', disable_tqdm=True):
        """
        Initialize KTA2 algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Archive size per task (default: 100)
        tau : float, optional
            Proportion of influential points in training data (default: 0.75)
        phi : float, optional
            Proportion of randomly selected individuals for uncertainty sampling (default: 0.1)
        wmax : int, optional
            Number of generations before updating CA and DA (default: 10)
        mu : int, optional
            Number of re-evaluated solutions at each generation (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'KTA2_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.n = n
        self.tau = tau
        self.phi = phi
        self.wmax = wmax
        self.mu = mu
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the KTA2 algorithm.

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
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Reorganize initial data into task-specific history lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task, self.mu)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task, self.mu)

        # Initialize archives for each task
        CAs = []  # Convergence Archive: (objs, decs)
        DAs = []  # Diversity Archive: (objs, decs)

        for i in range(nt):
            p_i = 1.0 / n_objs[i]
            CA_objs, CA_decs = self._update_CA(None, objs[i], decs[i], n_per_task[i])
            DA_objs, DA_decs = self._update_DA(None, objs[i], decs[i], n_per_task[i], p_i)
            CAs.append((CA_objs, CA_decs))
            DAs.append((DA_objs, DA_decs))

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                p_i = 1.0 / M
                n_i = n_initial_per_task[i]

                # Build influential point-insensitive Kriging models
                models_sensitive, models_insensitive, centers = self._build_kriging_models(
                    decs[i], objs[i], M, self.tau, data_type
                )

                # Store historical DA for diversity comparison
                DA_hist_objs = DAs[i][0].copy()

                # Create surrogate problem for TwoArch2 optimization
                model_container = {
                    'models_s': models_sensitive,
                    'models_i': models_insensitive,
                    'centers': centers,
                    'M': M,
                    'data_type': data_type,
                    'predictor': self._predict_with_models
                }

                def surrogate_func(x, mc=model_container):
                    return mc['predictor'](x, mc['models_s'], mc['models_i'], mc['centers'], mc['M'], mc['data_type'])

                surrogate_problem = MTOP()
                surrogate_problem.add_task(objective_func=surrogate_func, dim=dims[i])

                # Optimize surrogate using TwoArch2
                twoarch2 = TwoArch2(surrogate_problem, n=n_per_task[i], max_nfes=n_per_task[i] * self.wmax * 2, save_data=False)
                results = twoarch2.optimize()

                # Get CA and DA from TwoArch2 results
                CA_objs, CA_decs = results.CAs[0]
                DA_objs, DA_decs = results.DAs[0]

                # Predict MSE for DA solutions
                _, DA_mse = self._predict_with_mse(DA_decs, models_sensitive, models_insensitive, centers, M, data_type)

                # Adaptive sampling based on convergence/diversity state
                offspring_decs = self._adaptive_sampling(CA_objs, DA_objs, CA_decs, DA_decs, DA_mse, DA_hist_objs,
                                                         self.mu, p_i, self.phi)

                # Remove duplicates and already evaluated solutions
                offspring_decs = self._remove_duplicates(offspring_decs, decs[i])

                if offspring_decs.shape[0] > 0:
                    # Evaluate new solutions
                    off_objs, _ = evaluation_single(problem, offspring_decs, i)

                    # Update dataset with new samples
                    decs[i] = np.vstack([decs[i], offspring_decs])
                    objs[i] = np.vstack([objs[i], off_objs])

                    # Update archives with real evaluations
                    CAs[i] = self._update_CA(CAs[i], off_objs, offspring_decs, n_i)
                    DAs[i] = self._update_DA(DAs[i], off_objs, offspring_decs, n_i, p_i)

                    nfes_per_task[i] += offspring_decs.shape[0]
                    pbar.update(offspring_decs.shape[0])

                    # Store cumulative history
                    append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _build_kriging_models(self, decs, objs, M, tau, data_type):
        """
        Build influential point-insensitive Kriging models for each objective.

        Parameters
        ----------
        decs : np.ndarray
            Decision variables of shape (N, D)
        objs : np.ndarray
            Objective values of shape (N, M)
        M : int
            Number of objectives
        tau : float
            Proportion of influential points
        data_type : torch.dtype
            Data type for tensors

        Returns
        -------
        models_sensitive : list
            Sensitive GP models for each objective
        models_insensitive : list
            Insensitive GP models [M][2] for low/high regions
        centers : np.ndarray
            Center values for each objective, shape (M, 2)
        """
        N = decs.shape[0]
        num = int(np.ceil(N * tau))

        models_sensitive = []
        models_insensitive = [[None, None] for _ in range(M)]
        centers = np.zeros((M, 2))

        for j in range(M):
            # Build sensitive model on full dataset
            try:
                gp_sensitive = gp_build(decs, objs[:, j], data_type)
                models_sensitive.append(gp_sensitive)
            except Exception:
                models_sensitive.append(None)

            # Sort by objective j to identify low/high regions
            sorted_indices = np.argsort(objs[:, j])
            low_indices = sorted_indices[:num]
            high_indices = sorted_indices[-(num + 1):]

            # Calculate centers for model selection
            centers[j, 0] = np.mean(objs[low_indices, j])
            centers[j, 1] = np.mean(objs[high_indices, j])

            # Build insensitive models on low/high subsets
            for k, indices in enumerate([low_indices, high_indices]):
                try:
                    gp_insensitive = gp_build(decs[indices], objs[indices, j], data_type)
                    models_insensitive[j][k] = gp_insensitive
                except Exception:
                    models_insensitive[j][k] = None

        return models_sensitive, models_insensitive, centers

    def _predict_with_models(self, decs, models_sensitive, models_insensitive, centers, M, data_type):
        """
        Predict objectives using influential point-insensitive models.

        Parameters
        ----------
        decs : np.ndarray
            Decision variables of shape (N, D)
        models_sensitive : list
            Sensitive models for each objective
        models_insensitive : list
            Insensitive models [M][2] for each objective
        centers : np.ndarray
            Centers for each objective, shape (M, 2)
        M : int
            Number of objectives
        data_type : torch.dtype
            Data type for tensors

        Returns
        -------
        pop_objs : np.ndarray
            Predicted objectives of shape (N, M)
        """
        decs = np.atleast_2d(decs)
        N = decs.shape[0]
        pop_objs = np.zeros((N, M))

        for j in range(M):
            # Predict with sensitive model to determine region
            pred_sensitive, _ = gp_predict(models_sensitive[j], decs, data_type)
            pred_sensitive = pred_sensitive.flatten()

            # Select appropriate insensitive model based on proximity to centers
            for idx in range(N):
                dist_low = abs(pred_sensitive[idx] - centers[j, 0])
                dist_high = abs(pred_sensitive[idx] - centers[j, 1])

                if dist_low <= dist_high:
                    model = models_insensitive[j][0]
                else:
                    model = models_insensitive[j][1]

                pred, _ = gp_predict(model, decs[idx:idx + 1], data_type)
                pop_objs[idx, j] = pred.flatten()[0]

        return pop_objs

    def _predict_with_mse(self, decs, models_sensitive, models_insensitive, centers, M, data_type):
        """
        Predict objectives and MSE using influential point-insensitive models.

        Parameters
        ----------
        decs : np.ndarray
            Decision variables of shape (N, D)
        models_sensitive : list
            Sensitive models for each objective
        models_insensitive : list
            Insensitive models [M][2] for each objective
        centers : np.ndarray
            Centers for each objective, shape (M, 2)
        M : int
            Number of objectives
        data_type : torch.dtype
            Data type for tensors

        Returns
        -------
        pop_objs : np.ndarray
            Predicted objectives of shape (N, M)
        pop_mse : np.ndarray
            Predicted MSE (variance) of shape (N, M)
        """
        decs = np.atleast_2d(decs)
        N = decs.shape[0]
        pop_objs = np.zeros((N, M))
        pop_mse = np.zeros((N, M))

        for j in range(M):
            # Predict with sensitive model to determine region
            pred_sensitive, _ = gp_predict(models_sensitive[j], decs, data_type)
            pred_sensitive = pred_sensitive.flatten()

            # Select appropriate insensitive model and compute MSE
            for idx in range(N):
                dist_low = abs(pred_sensitive[idx] - centers[j, 0])
                dist_high = abs(pred_sensitive[idx] - centers[j, 1])

                if dist_low <= dist_high:
                    model = models_insensitive[j][0]
                else:
                    model = models_insensitive[j][1]

                pred, std = gp_predict(model, decs[idx:idx + 1], data_type)
                pop_objs[idx, j] = pred.flatten()[0]
                pop_mse[idx, j] = (std.flatten()[0]) ** 2

        return pop_objs, pop_mse

    def _update_CA(self, CA, new_objs, new_decs, max_size):
        """
        Update Convergence Archive using IBEA indicator-based selection.

        Parameters
        ----------
        CA : tuple or None
            Current CA (objs, decs) or None for initialization
        new_objs : np.ndarray
            New objective values
        new_decs : np.ndarray
            New decision variables
        max_size : int
            Maximum archive size

        Returns
        -------
        tuple
            Updated (CA_objs, CA_decs)
        """
        if CA is None:
            CA_objs = new_objs
            CA_decs = new_decs
        else:
            CA_objs, CA_decs = CA
            CA_objs = np.vstack([CA_objs, new_objs])
            CA_decs = np.vstack([CA_decs, new_decs])

        N = CA_objs.shape[0]
        if N <= max_size:
            return (CA_objs, CA_decs)

        # IBEA-style selection: normalize objectives
        min_vals = np.min(CA_objs, axis=0)
        max_vals = np.max(CA_objs, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        objs_norm = (CA_objs - min_vals) / range_vals

        # Calculate indicator matrix
        I = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                I[i, j] = np.max(objs_norm[i] - objs_norm[j])

        C = np.max(np.abs(I), axis=0)
        C[C == 0] = 1.0
        kappa = 0.05

        # Calculate fitness values
        C_matrix = np.tile(C, (N, 1))
        C_matrix = np.maximum(C_matrix, 1e-10)
        exponent = -I / C_matrix / kappa
        exponent = np.clip(exponent, -100, 100)
        F = np.sum(-np.exp(exponent), axis=0) + 1

        # Iteratively remove worst individuals
        choose = np.arange(N)
        while len(choose) > max_size:
            min_idx = np.argmin(F[choose])
            to_remove = choose[min_idx]

            if C[to_remove] > 1e-10:
                exp_update = -I[to_remove, :] / C[to_remove] / kappa
                exp_update = np.clip(exp_update, -100, 100)
                F = F + np.exp(exp_update)

            choose = np.delete(choose, min_idx)

        return (CA_objs[choose], CA_decs[choose])

    def _update_DA(self, DA, new_objs, new_decs, max_size, p):
        """
        Update Diversity Archive using non-dominated sorting and distance-based selection.

        Parameters
        ----------
        DA : tuple or None
            Current DA (objs, decs) or None for initialization
        new_objs : np.ndarray
            New objective values
        new_decs : np.ndarray
            New decision variables
        max_size : int
            Maximum archive size
        p : float
            Parameter for Minkowski distance calculation

        Returns
        -------
        tuple
            Updated (DA_objs, DA_decs)
        """
        if DA is None:
            DA_objs = new_objs
            DA_decs = new_decs
        else:
            DA_objs, DA_decs = DA
            DA_objs = np.vstack([DA_objs, new_objs])
            DA_decs = np.vstack([DA_decs, new_decs])

        # Keep only non-dominated solutions
        N = DA_objs.shape[0]
        front_no, _ = nd_sort(DA_objs, N)
        nd_mask = front_no == 1
        DA_objs = DA_objs[nd_mask]
        DA_decs = DA_decs[nd_mask]

        N = DA_objs.shape[0]
        if N <= max_size:
            return (DA_objs, DA_decs)

        # Select extreme solutions first
        choose = np.zeros(N, dtype=bool)
        M = DA_objs.shape[1]

        for m in range(M):
            choose[np.argmin(DA_objs[:, m])] = True
            choose[np.argmax(DA_objs[:, m])] = True

        if np.sum(choose) > max_size:
            chosen_indices = np.where(choose)[0]
            to_remove = np.random.choice(chosen_indices, size=np.sum(choose) - max_size, replace=False)
            choose[to_remove] = False
        elif np.sum(choose) < max_size:
            # Distance-based selection for remaining slots
            distance = np.full((N, N), np.inf)
            for i in range(N):
                diff = DA_objs - DA_objs[i]
                distance[i, :] = np.sum(np.abs(diff) ** p, axis=1) ** (1 / p)
            np.fill_diagonal(distance, np.inf)

            while np.sum(choose) < max_size:
                remaining = np.where(~choose)[0]
                chosen = np.where(choose)[0]
                min_distances = np.min(distance[np.ix_(remaining, chosen)], axis=1)
                max_idx = np.argmax(min_distances)
                choose[remaining[max_idx]] = True

        return (DA_objs[choose], DA_decs[choose])

    def _cal_convergence(self, CA_objs, DA_objs):
        """
        Calculate convergence flag using Wilcoxon signed-rank test.

        Parameters
        ----------
        CA_objs : np.ndarray
            Convergence archive objectives
        DA_objs : np.ndarray
            Diversity archive objectives

        Returns
        -------
        int
            1 if CA is significantly more converged than DA, 0 otherwise
        """
        N1 = CA_objs.shape[0]
        N2 = DA_objs.shape[0]

        if N1 != N2:
            return 0

        # Calculate ideal point
        combined = np.vstack([CA_objs, DA_objs])
        Zmin = np.min(combined, axis=0)

        # Normalize objectives
        PopObj = combined - Zmin
        max_vals = np.max(PopObj, axis=0)
        denom = max_vals - Zmin
        denom[denom == 0] = 1
        PopObj_norm = PopObj / denom

        # Calculate distances to ideal point
        dist1 = np.sqrt(np.sum(PopObj_norm[:N1], axis=1))
        dist2 = np.sqrt(np.sum(PopObj_norm[N1:], axis=1))

        try:
            from scipy.stats import rankdata

            # Remove zero differences
            diffxy = dist1 - dist2
            epsdiff = np.finfo(float).eps * 2
            nonzero_mask = np.abs(diffxy) >= epsdiff
            diffxy_nonzero = diffxy[nonzero_mask]

            n = len(diffxy_nonzero)
            if n == 0:
                return 0

            # Compute rank sums
            abs_diff = np.abs(diffxy_nonzero)
            tierank = rankdata(abs_diff, method='average')
            neg = diffxy_nonzero < 0
            r1 = np.sum(tierank[neg])
            r2 = n * (n + 1) / 2 - r1

            _, p_value = wilcoxon(dist1[nonzero_mask], dist2[nonzero_mask], alternative='two-sided')

            if p_value < 0.05 and r1 >= r2:
                return 1
            else:
                return 0

        except Exception:
            return 0

    def _pure_diversity(self, objs):
        """
        Calculate pure diversity score using iterative distance accumulation.

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (N, M)

        Returns
        -------
        float
            Pure diversity score
        """
        N = objs.shape[0]
        if N <= 1:
            return 0.0

        # Calculate pairwise distances using Minkowski distance with p=0.1
        D = squareform(pdist(objs, metric='minkowski', p=0.1))
        np.fill_diagonal(D, np.inf)

        # Connectivity matrix
        C = np.eye(N, dtype=bool)

        score = 0.0

        for k in range(N - 1):
            while True:
                # Find minimum distance for each point
                d = np.min(D, axis=1)
                J = np.argmin(D, axis=1)

                # Select point with maximum minimum distance
                i = np.argmax(d)

                # Update distances
                if D[J[i], i] != -np.inf:
                    D[J[i], i] = np.inf
                if D[i, J[i]] != -np.inf:
                    D[i, J[i]] = np.inf

                # Check connectivity
                P = np.any(C[i, :], axis=0) if C[i, :].ndim > 1 else C[i, :].copy()

                # Expand connectivity
                while not P[J[i]]:
                    newP = np.any(C[P, :], axis=0)
                    if np.array_equal(P, newP):
                        break
                    else:
                        P = newP

                if not P[J[i]]:
                    break

            # Update connectivity and accumulate score
            C[i, J[i]] = True
            C[J[i], i] = True
            D[i, :] = -np.inf
            score += d[i]

        return score

    def _adaptive_sampling(self, CA_objs, DA_objs, CA_decs, DA_decs, DA_mse, DA_hist_objs, mu, p, phi):
        """
        Adaptive sampling strategy based on convergence/diversity state.

        Parameters
        ----------
        CA_objs : np.ndarray
            Convergence archive objectives
        DA_objs : np.ndarray
            Diversity archive objectives
        CA_decs : np.ndarray
            Convergence archive decisions
        DA_decs : np.ndarray
            Diversity archive decisions
        DA_mse : np.ndarray
            Diversity archive MSE
        DA_hist_objs : np.ndarray
            Historical diversity archive objectives
        mu : int
            Number of solutions to select
        p : float
            Parameter for distance calculation
        phi : float
            Proportion for uncertainty sampling

        Returns
        -------
        offspring_decs : np.ndarray
            Selected decision variables for real evaluation
        """
        # Calculate convergence flag
        flag = self._cal_convergence(CA_objs, DA_objs)

        if flag == 1:
            # Convergence sampling when CA dominates DA
            offspring_decs = self._convergence_sampling(CA_objs, CA_decs, mu)
        else:
            # Compare pure diversity to decide between uncertainty and diversity sampling
            pd_current = self._pure_diversity(DA_objs)
            pd_hist = self._pure_diversity(DA_hist_objs)

            if pd_current < pd_hist:
                offspring_decs = self._uncertainty_sampling(DA_decs, DA_mse, mu, phi)
            else:
                offspring_decs = self._diversity_sampling(DA_objs, DA_decs, DA_hist_objs, mu, p)

        return offspring_decs

    def _convergence_sampling(self, CA_objs, CA_decs, mu):
        """
        Convergence sampling strategy using IBEA indicator.

        Parameters
        ----------
        CA_objs : np.ndarray
            Convergence archive objectives
        CA_decs : np.ndarray
            Convergence archive decisions
        mu : int
            Number of solutions to select

        Returns
        -------
        offspring_decs : np.ndarray
            Selected decision variables
        """
        N = CA_objs.shape[0]
        if N <= mu:
            return CA_decs.copy()

        # Normalize objectives
        min_vals = np.min(CA_objs, axis=0)
        max_vals = np.max(CA_objs, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        objs_norm = (CA_objs - min_vals) / range_vals

        # Calculate indicator matrix and fitness
        I = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                I[i, j] = np.max(objs_norm[i] - objs_norm[j])

        C = np.max(np.abs(I), axis=0)
        C[C == 0] = 1.0
        kappa = 0.05

        C_matrix = np.tile(C, (N, 1))
        C_matrix = np.maximum(C_matrix, 1e-10)
        exponent = -I / C_matrix / kappa
        exponent = np.clip(exponent, -100, 100)
        F = np.sum(-np.exp(exponent), axis=0) + 1

        # Select mu solutions with highest fitness
        choose = list(range(N))
        while len(choose) > mu:
            min_idx = np.argmin([F[c] for c in choose])
            to_remove = choose[min_idx]

            if C[to_remove] > 1e-10:
                exp_update = -I[to_remove, :] / C[to_remove] / kappa
                exp_update = np.clip(exp_update, -100, 100)
                F = F + np.exp(exp_update)

            choose.pop(min_idx)

        return CA_decs[choose]

    def _uncertainty_sampling(self, DA_decs, DA_mse, mu, phi):
        """
        Uncertainty sampling strategy based on prediction variance.

        Parameters
        ----------
        DA_decs : np.ndarray
            Diversity archive decisions
        DA_mse : np.ndarray
            Diversity archive MSE (variance)
        mu : int
            Number of solutions to select
        phi : float
            Proportion for random subset selection

        Returns
        -------
        offspring_decs : np.ndarray
            Selected decision variables
        """
        N = DA_decs.shape[0]
        if N <= mu:
            return DA_decs.copy()

        n_random = max(1, int(np.ceil(phi * N)))
        choose = []

        for _ in range(mu):
            # Randomly select subset and pick highest uncertainty
            random_indices = np.random.permutation(N)[:n_random]
            uncertainty = np.mean(DA_mse[random_indices], axis=1)
            best_idx = random_indices[np.argmax(uncertainty)]
            choose.append(best_idx)

        return DA_decs[choose]

    def _diversity_sampling(self, DA_objs, DA_decs, DA_hist_objs, mu, p):
        """
        Diversity sampling strategy based on distance to historical archive.

        Parameters
        ----------
        DA_objs : np.ndarray
            Current diversity archive objectives
        DA_decs : np.ndarray
            Current diversity archive decisions
        DA_hist_objs : np.ndarray
            Historical diversity archive objectives
        mu : int
            Number of solutions to select
        p : float
            Parameter for Minkowski distance calculation

        Returns
        -------
        offspring_decs : np.ndarray
            Selected decision variables
        """
        N_current = DA_objs.shape[0]
        if N_current <= mu:
            return DA_decs.copy()

        # Normalize using combined min/max
        combined = np.vstack([DA_objs, DA_hist_objs])
        min_vals = np.min(combined, axis=0)
        max_vals = np.max(combined, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1

        DA_hist_norm = (DA_hist_objs - min_vals) / range_vals
        DA_objs_norm = (DA_objs - min_vals) / range_vals

        N_hist = DA_hist_norm.shape[0]

        # Combine normalized objectives
        pop = np.vstack([DA_hist_norm, DA_objs_norm])
        NN = pop.shape[0]

        choose = np.zeros(NN, dtype=bool)
        choose[:N_hist] = True  # Mark historical solutions as chosen

        # Calculate distance matrix
        distance = np.full((NN, NN), np.inf)
        for i in range(NN):
            diff = pop - pop[i]
            distance[i, :] = np.sum(np.abs(diff) ** p, axis=1) ** (1 / p)
        np.fill_diagonal(distance, np.inf)

        # Select solutions maximizing distance to chosen set
        offspring_decs = []
        while len(offspring_decs) < mu:
            remaining = np.where(~choose)[0]
            chosen = np.where(choose)[0]

            if len(remaining) == 0:
                break

            if len(chosen) == 0:
                idx = np.random.choice(remaining)
            else:
                min_distances = np.min(distance[np.ix_(remaining, chosen)], axis=1)
                max_idx = np.argmax(min_distances)
                idx = remaining[max_idx]

            choose[idx] = True
            if idx >= N_hist:
                offspring_decs.append(DA_decs[idx - N_hist])

        if len(offspring_decs) == 0:
            # Fallback: return random samples from DA
            indices = np.random.choice(DA_decs.shape[0], size=min(mu, DA_decs.shape[0]), replace=False)
            return DA_decs[indices]

        return np.array(offspring_decs)

    def _remove_duplicates(self, offspring_decs, existing_decs, tol=1e-5):
        """
        Remove duplicate solutions from offspring.

        Parameters
        ----------
        offspring_decs : np.ndarray
            New decision variables
        existing_decs : np.ndarray
            Existing decision variables in dataset
        tol : float, optional
            Tolerance for duplicate detection (default: 1e-5)

        Returns
        -------
        unique_decs : np.ndarray
            Unique decision variables not in existing dataset
        """
        if offspring_decs.shape[0] == 0:
            return np.empty((0, existing_decs.shape[1]))

        # Remove duplicates within offspring
        unique_indices = []
        seen = set()

        for i, dec in enumerate(offspring_decs):
            dec_tuple = tuple(np.round(dec, 8))
            if dec_tuple not in seen:
                seen.add(dec_tuple)
                unique_indices.append(i)

        if len(unique_indices) == 0:
            return np.empty((0, offspring_decs.shape[1]))

        unique_offspring = offspring_decs[unique_indices]

        # Remove solutions already in existing_decs
        final_indices = []
        for i, dec in enumerate(unique_offspring):
            distances = np.min(cdist(dec.reshape(1, -1), existing_decs))
            if distances > tol:
                final_indices.append(i)

        if len(final_indices) == 0:
            return np.empty((0, offspring_decs.shape[1]))

        return unique_offspring[final_indices]
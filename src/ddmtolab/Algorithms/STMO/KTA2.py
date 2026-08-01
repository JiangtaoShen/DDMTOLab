"""
Kriging-Assisted Two-Archive Evolutionary Algorithm 2 (KTA2)

This module implements KTA2 for computationally expensive many-objective optimization.
It maintains two archives (convergence and diversity) and uses point-insensitive
Kriging models with adaptive sampling strategies.

References
----------
    [1] Z. Song, H. Wang, C. He, and Y. Jin. A Kriging-assisted two-archive evolutionary algorithm for expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(6): 1013-1027.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.16
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, rankdata
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
import warnings

warnings.filterwarnings("ignore")


class KTA2:
    """
    Kriging-Assisted Two-Archive Evolutionary Algorithm 2 for expensive
    many-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100,
                 tau=0.75, phi=0.1, wmax=10, mu=5,
                 save_data=True, save_path='./Data', name='KTA2', disable_tqdm=True):
        """
        Initialize KTA2 algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task. PlatEMO's KTA2 draws
            ``Problem.N`` Latin-hypercube samples, so the default is ``n``.
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population/archive size per task (default: 100)
        tau : float, optional
            Proportion of training data for insensitive models (default: 0.75)
        phi : float, optional
            Fraction of DA for uncertainty sampling (default: 0.1)
        wmax : int, optional
            Number of inner surrogate evolution generations (default: 10)
        mu : int, optional
            Number of re-evaluated solutions per generation (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'KTA2')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
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

        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # MATLAB: N = Problem.N; P = UniformPoint(N,Problem.D,'Latin')
        if self.n_initial is None:
            n_initial_per_task = list(n_per_task)
        else:
            n_initial_per_task = par_list(self.n_initial, nt)

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Initialize archives for each task
        CAs = []  # Convergence Archives: list of (objs, decs)
        DAs = []  # Diversity Archives: list of (objs, decs)

        for i in range(nt):
            CA_size_i = n_per_task[i]
            CA_objs, CA_decs = self._update_CA(None, objs[i], decs[i], CA_size_i)
            DA_objs, DA_decs = objs[i].copy(), decs[i].copy()
            CAs.append((CA_objs, CA_decs))
            DAs.append((DA_objs, DA_decs))

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break
            nfes_before_pass = sum(nfes_per_task)

            for i in active_tasks:
                M = n_objs[i]
                dim = dims[i]
                p_i = 1.0 / M
                CA_size_i = n_per_task[i]
                N = n_per_task[i]

                # ===== Build Sensitive Models (one per objective on full data) =====
                sensitive_models = []
                for j in range(M):
                    model = gp_build(decs[i], objs[i][:, j:j + 1], data_type)
                    sensitive_models.append(model)

                # ===== Build Insensitive Models (two per objective on subsets) =====
                num_samples = decs[i].shape[0]
                num_top = int(np.ceil(num_samples * self.tau))

                insensitive_models = [[None, None] for _ in range(M)]
                centers = np.zeros((M, 2))

                for j in range(M):
                    sorted_idx = np.argsort(objs[i][:, j])
                    # Group 0: best tau fraction, Group 1: worst tau fraction
                    group_indices = [sorted_idx[:num_top], sorted_idx[-(num_top + 1):]]

                    for g in range(2):
                        centers[j, g] = np.mean(objs[i][group_indices[g], j])
                        train_X = decs[i][group_indices[g]]
                        train_Y = objs[i][group_indices[g], j:j + 1]
                        insensitive_models[j][g] = gp_build(train_X, train_Y, data_type)

                # ===== Inner Loop: Surrogate-based Evolution =====
                CA_objs_pred = CAs[i][0].copy()
                CA_decs_pred = CAs[i][1].copy()
                DA_objs_pred = DAs[i][0].copy()
                DA_decs_pred = DAs[i][1].copy()
                DA_mse_pred = np.zeros_like(DA_objs_pred)

                for w in range(self.wmax):
                    # Mating selection
                    parentC_decs, parentM_decs = self._mating_selection(
                        CA_objs_pred, CA_decs_pred, DA_objs_pred, DA_decs_pred, N
                    )

                    # Generate offspring
                    # Crossover only (proC=1, disC=20, proM=0): matches MATLAB {1,20,0,0}
                    off_C = _crossover_only(parentC_decs, muc=20)
                    # Mutation only (proC=0, proM=1, disM=20): matches MATLAB {0,0,1,20}
                    off_M = _mutation_only(parentM_decs, mum=20)
                    off_decs_all = np.vstack([off_C, off_M])

                    # Combine populations: DA + CA + offspring
                    pop_decs = np.vstack([DA_decs_pred, CA_decs_pred, off_decs_all])

                    # Predict using insensitive models
                    pop_objs, pop_mse = self._predict_with_insensitive_models(
                        pop_decs, sensitive_models, insensitive_models, centers, M, data_type
                    )

                    # Update predicted CA and DA
                    CA_objs_pred, CA_decs_pred, _ = self._k_update_CA(
                        pop_objs, pop_decs, pop_mse, CA_size_i
                    )
                    DA_objs_pred, DA_decs_pred, DA_mse_pred = self._k_update_DA(
                        pop_objs, pop_decs, pop_mse, N, p_i
                    )

                # ===== Adaptive Sampling =====
                offspring_decs = self._adaptive_sampling(
                    CA_objs_pred, DA_objs_pred, CA_decs_pred, DA_decs_pred,
                    DA_mse_pred, DAs[i][0], DAs[i][1], self.mu, p_i, self.phi
                )

                # Remove duplicates against existing evaluated data
                # (MATLAB: unique(...,'rows') then the min-distance > 1e-5 filter)
                offspring_decs = remove_duplicates(offspring_decs, decs[i])

                # Never exceed the remaining budget of this task
                remaining = max_nfes_per_task[i] - nfes_per_task[i]
                if offspring_decs.shape[0] > remaining:
                    offspring_decs = offspring_decs[:remaining]

                if offspring_decs.shape[0] > 0:
                    # Evaluate with real function
                    off_objs, _ = evaluation_single(problem, offspring_decs, i)

                    # Update all evaluated data
                    decs[i] = np.vstack([decs[i], offspring_decs])
                    objs[i] = np.vstack([objs[i], off_objs])

                    # Update real CA and DA
                    CA_objs, CA_decs = self._update_CA(
                        CAs[i], off_objs, offspring_decs, CA_size_i
                    )
                    DA_objs, DA_decs = self._update_DA(
                        DAs[i], off_objs, offspring_decs, N, p_i
                    )
                    CAs[i] = (CA_objs, CA_decs)
                    DAs[i] = (DA_objs, DA_decs)

                    nfes_per_task[i] += offspring_decs.shape[0]
                    pbar.update(offspring_decs.shape[0])

            if sum(nfes_per_task) == nfes_before_pass:
                # No task could produce a new sample in a full pass; stop
                break

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.mu)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _mating_selection(CA_objs, CA_decs, DA_objs, DA_decs, N):
        """
        Select parents from CA and DA for offspring generation.

        Parameters
        ----------
        CA_objs : np.ndarray
            Convergence archive objectives
        CA_decs : np.ndarray
            Convergence archive decisions
        DA_objs : np.ndarray
            Diversity archive objectives
        DA_decs : np.ndarray
            Diversity archive decisions
        N : int
            Population size

        Returns
        -------
        parentC_decs : np.ndarray
            Parents for SBX crossover (convergence-oriented)
        parentM_decs : np.ndarray
            Parents for polynomial mutation (diversity-oriented)
        """
        CA_n = CA_objs.shape[0]
        DA_n = DA_objs.shape[0]
        half_N = int(np.ceil(N / 2))

        # Select from CA with dominance comparison
        idx1 = np.random.randint(0, CA_n, size=half_N)
        idx2 = np.random.randint(0, CA_n, size=half_N)

        any_less = np.any(CA_objs[idx1] < CA_objs[idx2], axis=1)
        any_greater = np.any(CA_objs[idx1] > CA_objs[idx2], axis=1)
        dominate = any_less.astype(int) - any_greater.astype(int)

        # Keep parent1 if it dominates, parent2 otherwise
        selected_CA = np.where(dominate == 1, idx1, idx2)

        # Random parents from DA
        selected_DA = np.random.randint(0, DA_n, size=half_N)

        parentC_decs = np.vstack([CA_decs[selected_CA], DA_decs[selected_DA]])

        # Mutation parents: random from CA
        parentM_decs = CA_decs[np.random.randint(0, CA_n, size=N)]

        return parentC_decs, parentM_decs

    @staticmethod
    def _update_CA(CA, new_objs, new_decs, max_size):
        """
        Update Convergence Archive using IBEA fitness (UpdateCA.m).

        Parameters
        ----------
        CA : tuple or None
            Current CA as (objs, decs) or None for initial
        new_objs : np.ndarray
            New objective values
        new_decs : np.ndarray
            New decision variables
        max_size : int
            Maximum archive size

        Returns
        -------
        CA_objs : np.ndarray
            Updated CA objectives
        CA_decs : np.ndarray
            Updated CA decisions
        """
        if CA is None:
            CA_objs = new_objs.copy()
            CA_decs = new_decs.copy()
        else:
            CA_objs = np.vstack([CA[0], new_objs])
            CA_decs = np.vstack([CA[1], new_decs])

        N = CA_objs.shape[0]
        if N <= max_size:
            return CA_objs, CA_decs

        # IBEA fitness-based selection
        fitness, I, C = ibea_fitness(CA_objs, kappa=0.05)

        choose = list(range(N))
        while len(choose) > max_size:
            fit_values = fitness[choose]
            min_idx = np.argmin(fit_values)
            to_remove = choose[min_idx]

            if C[to_remove] > 1e-10:
                fitness += np.exp(-I[to_remove, :] / C[to_remove] / 0.05)

            choose.pop(min_idx)

        return CA_objs[choose], CA_decs[choose]

    @staticmethod
    def _update_DA(DA, new_objs, new_decs, max_size, p):
        """
        Update Diversity Archive with non-dominated sorting and truncation (UpdateDA.m).

        Parameters
        ----------
        DA : tuple or None
            Current DA as (objs, decs) or None
        new_objs : np.ndarray
            New objective values
        new_decs : np.ndarray
            New decision variables
        max_size : int
            Maximum archive size
        p : float
            Parameter for fractional distance norm

        Returns
        -------
        DA_objs : np.ndarray
            Updated DA objectives
        DA_decs : np.ndarray
            Updated DA decisions
        """
        if DA is None:
            DA_objs = new_objs.copy()
            DA_decs = new_decs.copy()
        else:
            DA_objs = np.vstack([DA[0], new_objs])
            DA_decs = np.vstack([DA[1], new_decs])

        # Non-dominated sorting
        N = DA_objs.shape[0]
        front_no, _ = nd_sort(DA_objs, N)
        nd_mask = front_no == 1
        DA_objs = DA_objs[nd_mask]
        DA_decs = DA_decs[nd_mask]

        N = DA_objs.shape[0]
        if N <= max_size:
            return DA_objs, DA_decs

        # Select extreme solutions (min and max for each objective)
        choose = np.zeros(N, dtype=bool)
        for m in range(DA_objs.shape[1]):
            choose[np.argmin(DA_objs[:, m])] = True
            choose[np.argmax(DA_objs[:, m])] = True

        if np.sum(choose) > max_size:
            chosen_idx = np.where(choose)[0]
            to_remove = np.random.choice(chosen_idx, size=np.sum(choose) - max_size, replace=False)
            choose[to_remove] = False
        elif np.sum(choose) < max_size:
            # Truncation strategy: add solutions that maximize min distance to chosen set
            diff = DA_objs[:, np.newaxis, :] - DA_objs[np.newaxis, :, :]
            dist = np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
            np.fill_diagonal(dist, np.inf)

            while np.sum(choose) < max_size:
                remaining = np.where(~choose)[0]
                chosen = np.where(choose)[0]
                min_dists = np.min(dist[np.ix_(remaining, chosen)], axis=1)
                best = np.argmax(min_dists)
                choose[remaining[best]] = True

        return DA_objs[choose], DA_decs[choose]

    @staticmethod
    def _k_update_CA(pop_objs, pop_decs, pop_mse, max_size):
        """
        Update predicted CA using IBEA fitness (K_UpdateCA.m).

        Parameters
        ----------
        pop_objs : np.ndarray
            Predicted objective values
        pop_decs : np.ndarray
            Decision variables
        pop_mse : np.ndarray
            Predicted MSE values
        max_size : int
            Maximum archive size

        Returns
        -------
        CA_objs, CA_decs, CA_mse : np.ndarray
            Selected archive members
        """
        N = pop_objs.shape[0]
        if N <= max_size:
            return pop_objs.copy(), pop_decs.copy(), pop_mse.copy()

        fitness, I, C = ibea_fitness(pop_objs, kappa=0.05)

        choose = list(range(N))
        while len(choose) > max_size:
            fit_values = fitness[choose]
            min_idx = np.argmin(fit_values)
            to_remove = choose[min_idx]

            if C[to_remove] > 1e-10:
                fitness += np.exp(-I[to_remove, :] / C[to_remove] / 0.05)

            choose.pop(min_idx)

        return pop_objs[choose], pop_decs[choose], pop_mse[choose]

    @staticmethod
    def _k_update_DA(pop_objs, pop_decs, pop_mse, max_size, p):
        """
        Update predicted DA with ND sort + truncation on normalized objectives (K_UpdateDA.m).

        Parameters
        ----------
        pop_objs : np.ndarray
            Predicted objective values
        pop_decs : np.ndarray
            Decision variables
        pop_mse : np.ndarray
            Predicted MSE values
        max_size : int
            Maximum archive size
        p : float
            Parameter for fractional distance norm

        Returns
        -------
        DA_objs, DA_decs, DA_mse : np.ndarray
            Selected archive members
        """
        N = pop_objs.shape[0]

        # Normalize objectives BEFORE non-dominated sorting
        min_vals = np.min(pop_objs, axis=0)
        max_vals = np.max(pop_objs, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        pop_objs_norm = (pop_objs - min_vals) / range_vals

        # Non-dominated sorting on original objectives
        front_no, _ = nd_sort(pop_objs, N)
        nd_mask = front_no == 1

        pop_objs = pop_objs[nd_mask]
        pop_decs = pop_decs[nd_mask]
        pop_mse = pop_mse[nd_mask]
        pop_objs_norm = pop_objs_norm[nd_mask]

        N = pop_objs.shape[0]
        if N <= max_size:
            return pop_objs, pop_decs, pop_mse

        # Initial selection: random index from 1 to M (matching MATLAB)
        M = pop_objs_norm.shape[1]
        choose = np.zeros(N, dtype=bool)
        select = np.random.permutation(M)
        if select[0] < N:
            choose[select[0]] = True
        else:
            choose[0] = True

        if np.sum(choose) < max_size:
            # Truncation using normalized objectives
            diff = pop_objs_norm[:, np.newaxis, :] - pop_objs_norm[np.newaxis, :, :]
            dist = np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
            np.fill_diagonal(dist, np.inf)

            while np.sum(choose) < max_size:
                remaining = np.where(~choose)[0]
                chosen = np.where(choose)[0]
                if len(remaining) == 0:
                    break
                min_dists = np.min(dist[np.ix_(remaining, chosen)], axis=1)
                best = np.argmax(min_dists)
                choose[remaining[best]] = True

        return pop_objs[choose], pop_decs[choose], pop_mse[choose]

    @staticmethod
    def _predict_with_insensitive_models(pop_decs, sensitive_models, insensitive_models,
                                         centers, M, data_type):
        """
        Predict objectives and MSE using sensitive/insensitive model framework.

        For each objective: first predict with sensitive model to determine which
        insensitive model to use, then predict with the selected insensitive model.

        Parameters
        ----------
        pop_decs : np.ndarray
            Population decision variables, shape (N, D)
        sensitive_models : list
            List of M sensitive GP models
        insensitive_models : list
            List of M lists, each containing 2 insensitive GP models
        centers : np.ndarray
            Centers for model selection, shape (M, 2)
        M : int
            Number of objectives
        data_type : torch.dtype
            Data type for GP prediction

        Returns
        -------
        pop_objs : np.ndarray
            Predicted objectives, shape (N, M)
        pop_mse : np.ndarray
            Predicted MSE, shape (N, M)
        """
        pop_N = pop_decs.shape[0]
        pop_objs = np.zeros((pop_N, M))
        pop_mse = np.zeros((pop_N, M))

        for j in range(M):
            # Batch predict with sensitive model
            sens_pred, _ = gp_predict(sensitive_models[j], pop_decs, data_type)
            sens_pred = sens_pred.flatten()

            # Determine which insensitive model to use
            dist_to_center0 = np.abs(sens_pred - centers[j, 0])
            dist_to_center1 = np.abs(sens_pred - centers[j, 1])
            use_model0 = dist_to_center0 <= dist_to_center1

            # Batch predict with insensitive model 0
            idx0 = np.where(use_model0)[0]
            if len(idx0) > 0:
                pred0, std0 = gp_predict(insensitive_models[j][0], pop_decs[idx0], data_type)
                pop_objs[idx0, j] = pred0.flatten()
                pop_mse[idx0, j] = (std0.flatten()) ** 2

            # Batch predict with insensitive model 1
            idx1 = np.where(~use_model0)[0]
            if len(idx1) > 0:
                pred1, std1 = gp_predict(insensitive_models[j][1], pop_decs[idx1], data_type)
                pop_objs[idx1, j] = pred1.flatten()
                pop_mse[idx1, j] = (std1.flatten()) ** 2

        return pop_objs, pop_mse

    @staticmethod
    def _adaptive_sampling(CA_objs, DA_objs, CA_decs, DA_decs, DA_mse,
                           real_DA_objs, real_DA_decs, mu, p, phi):
        """
        Select solutions for expensive re-evaluation using adaptive strategy.

        Three strategies:
        1. Convergence sampling: IBEA-based selection from predicted CA
        2. Uncertainty sampling: highest MSE from predicted DA
        3. Diversity sampling: truncation-based from predicted DA

        Parameters
        ----------
        CA_objs, DA_objs : np.ndarray
            Predicted archive objectives
        CA_decs, DA_decs : np.ndarray
            Predicted archive decisions
        DA_mse : np.ndarray
            Predicted DA MSE values
        real_DA_objs, real_DA_decs : np.ndarray
            Real (evaluated) DA objectives and decisions
        mu : int
            Number of solutions to select
        p : float
            Distance norm parameter
        phi : float
            Fraction of DA for uncertainty sampling

        Returns
        -------
        offspring_decs : np.ndarray
            Selected decision variables for re-evaluation
        """
        # Compute ideal point
        combined_objs = np.vstack([CA_objs, DA_objs])
        ideal_point = np.min(combined_objs, axis=0)

        # Check convergence
        flag = KTA2._cal_convergence(CA_objs, DA_objs, ideal_point)

        if flag == 1:
            # Strategy 1: Convergence sampling from predicted CA
            N = CA_objs.shape[0]
            if N <= mu:
                return CA_decs.copy()

            fitness, I, C = ibea_fitness(CA_objs, kappa=0.05)

            choose = list(range(N))
            while len(choose) > mu:
                fit_values = fitness[choose]
                min_idx = np.argmin(fit_values)
                to_remove = choose[min_idx]

                if C[to_remove] > 1e-10:
                    fitness += np.exp(-I[to_remove, :] / C[to_remove] / 0.05)

                choose.pop(min_idx)

            return CA_decs[choose]
        else:
            pd_pred = KTA2._pure_diversity(DA_objs)
            pd_real = KTA2._pure_diversity(real_DA_objs)

            if pd_pred < pd_real:
                # Strategy 2: Uncertainty sampling from predicted DA
                DA_n = DA_mse.shape[0]
                subset_size = max(1, int(np.ceil(phi * DA_n)))
                chosen = []

                for _ in range(mu):
                    perm = np.random.permutation(DA_n)
                    subset_idx = perm[:subset_size]
                    uncertainty = np.mean(DA_mse[subset_idx], axis=1)
                    best = subset_idx[np.argmax(uncertainty)]
                    chosen.append(best)

                return DA_decs[chosen]
            else:
                # Strategy 3: Diversity sampling via truncation
                all_objs = np.vstack([DA_objs, real_DA_objs])
                min_vals = np.min(all_objs, axis=0)
                max_vals = np.max(all_objs, axis=0)
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1.0

                DA_Nor = (real_DA_objs - min_vals) / range_vals
                DA_Nor_pre = (DA_objs - min_vals) / range_vals

                N_real = DA_Nor.shape[0]
                Pop = np.vstack([DA_Nor, DA_Nor_pre])
                Pop_dec = np.vstack([real_DA_decs, DA_decs])
                NN = Pop.shape[0]

                # Start with all real DA chosen
                choose = np.zeros(NN, dtype=bool)
                choose[:N_real] = True
                target_size = N_real + mu

                # Compute pairwise distances
                diff = Pop[:, np.newaxis, :] - Pop[np.newaxis, :, :]
                dist_matrix = np.sum(np.abs(diff) ** p, axis=2) ** (1.0 / p)
                np.fill_diagonal(dist_matrix, np.inf)

                offspring = []
                while np.sum(choose) < target_size:
                    remaining = np.where(~choose)[0]
                    chosen_idx = np.where(choose)[0]
                    if len(remaining) == 0:
                        break
                    min_dists = np.min(dist_matrix[np.ix_(remaining, chosen_idx)], axis=1)
                    best = np.argmax(min_dists)
                    choose[remaining[best]] = True
                    offspring.append(Pop_dec[remaining[best]])

                if len(offspring) == 0:
                    idx = np.random.choice(DA_decs.shape[0], size=min(mu, DA_decs.shape[0]),
                                           replace=False)
                    return DA_decs[idx]

                return np.array(offspring)

    @staticmethod
    def _cal_convergence(pop_obj1, pop_obj2, z_min):
        """
        Check if predicted CA converges better than predicted DA using
        Wilcoxon signed rank test (Cal_Convergence.m).

        Parameters
        ----------
        pop_obj1 : np.ndarray
            Predicted CA objectives
        pop_obj2 : np.ndarray
            Predicted DA objectives
        z_min : np.ndarray
            Ideal point

        Returns
        -------
        flag : int
            1 if CA converges significantly better, 0 otherwise
        """
        N1 = pop_obj1.shape[0]
        N2 = pop_obj2.shape[0]

        if N1 != N2:
            return 0

        try:
            # Translate and normalize (matching MATLAB implementation)
            pop_obj = np.vstack([pop_obj1, pop_obj2]) - z_min
            denominator = np.max(pop_obj, axis=0) - z_min
            denominator[np.abs(denominator) < 1e-10] = 1.0
            pop_obj = pop_obj / denominator

            # Compute distance to origin: sqrt(sum of elements)
            distance1 = np.sqrt(np.clip(np.sum(pop_obj[:N1], axis=1), 0, None))
            distance2 = np.sqrt(np.clip(np.sum(pop_obj[N1:], axis=1), 0, None))

            # Remove pairs with negligible difference
            diff = distance1 - distance2
            abs_diff = np.abs(diff)
            eps_tol = np.finfo(float).eps * (np.abs(distance1) + np.abs(distance2))
            nonzero = abs_diff > eps_tol

            if np.sum(nonzero) < 2:
                return 0

            diff_nz = diff[nonzero]
            abs_diff_nz = abs_diff[nonzero]
            n = len(diff_nz)

            # Compute rank sums
            ranks = rankdata(abs_diff_nz)
            r1 = np.sum(ranks[diff_nz < 0])  # sum of ranks where CA closer (better)
            r2 = n * (n + 1) / 2 - r1

            # Wilcoxon signed rank test
            _, p_value = wilcoxon(distance1[nonzero], distance2[nonzero])

            flag = 1 if p_value <= 0.05 else 0

            # If significant but DA converges better, set flag to 0
            if flag == 1 and (r1 - r2) < 0:
                flag = 0

            return flag
        except Exception:
            return 0

    @staticmethod
    def _pure_diversity(pop_obj):
        """
        Compute pure diversity metric using spanning tree approach (PD function).

        Uses Minkowski distance with p=0.1 to build a maximum spanning tree
        and returns the sum of edge weights.

        Parameters
        ----------
        pop_obj : np.ndarray
            Objective values, shape (N, M)

        Returns
        -------
        score : float
            Pure diversity score
        """
        N = pop_obj.shape[0]
        if N <= 1:
            return 0.0

        # Connectivity matrix (each node connected to itself initially)
        C = np.eye(N, dtype=bool)

        # Minkowski distance with p=0.1
        D = cdist(pop_obj, pop_obj, metric='minkowski', p=0.1)
        np.fill_diagonal(D, np.inf)

        score = 0.0
        for k in range(N - 1):
            while True:
                d = np.min(D, axis=1)
                J = np.argmin(D, axis=1)
                i = np.argmax(d)
                j = J[i]

                # Mark edge as visited
                if D[j, i] != -np.inf:
                    D[j, i] = np.inf
                if D[i, j] != -np.inf:
                    D[i, j] = np.inf

                # Check if i and j are already connected via BFS
                P = C[i].copy()
                while not P[j]:
                    new_P = np.any(C[P], axis=0)
                    if np.array_equal(P, new_P):
                        break
                    P = new_P

                if not P[j]:
                    break

            C[i, j] = True
            C[j, i] = True
            D[i, :] = -np.inf
            score += d[i]

        return score


def _crossover_only(parents, muc=20):
    """
    Generate offspring using SBX crossover only (no mutation).

    Exact port of MATLAB ``OperatorGA(Problem,Parent,{1,20,0,0})``: the parent
    matrix is split into two halves ``P1 = Parent(1:floor(N/2))`` and
    ``P2 = Parent(floor(N/2)+1:2*floor(N/2))`` **without shuffling**, and each
    pair (P1_j, P2_j) produces two children. The pairing is load bearing in
    KTA2 because ParentCdec is ``[CA-selected ; DA-random]``, so the structured
    split crosses a CA parent with a DA parent.

    Parameters
    ----------
    parents : np.ndarray
        Parent population, shape (n, d)
    muc : float
        Distribution index for crossover

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (2*floor(n/2), d)
    """
    n, d = parents.shape
    num_pairs = n // 2
    if num_pairs == 0:
        return np.zeros((0, d))

    off1 = np.zeros((num_pairs, d))
    off2 = np.zeros((num_pairs, d))
    for j in range(num_pairs):
        c1, c2 = crossover(parents[j, :], parents[num_pairs + j, :], mu=muc)
        off1[j] = c1
        off2[j] = c2

    # MATLAB stacks all first children then all second children
    return np.vstack([off1, off2])


def _mutation_only(parents, mum=20):
    """
    Generate offspring using polynomial mutation only (no crossover).

    Exact port of MATLAB ``OperatorGA(Problem,Parent,{0,0,1,20})``. With
    ``proC = 0`` every SBX beta is forced to 1, so the crossover step returns the
    parents unchanged; PlatEMO then keeps only the first ``2*floor(N/2)`` parents
    and mutates them.

    Parameters
    ----------
    parents : np.ndarray
        Parent population, shape (n, d)
    mum : float
        Distribution index for mutation

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (2*floor(n/2), d)
    """
    n, d = parents.shape
    n_off = 2 * (n // 2)
    offdecs = np.zeros((n_off, d))
    for j in range(n_off):
        offdecs[j] = mutation(np.clip(parents[j, :], 0.0, 1.0), mu=mum)
    return offdecs

"""
Kriging-assisted Reference Vector Guided Evolutionary Algorithm (K-RVEA)

This module implements K-RVEA for computationally expensive multi/many-objective optimization.

References
----------
    [1] T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya. A surrogate-assisted reference vector guided evolutionary algorithm for computationally expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2018, 22(1): 129-142.

Notes
-----
Author: Jiangtao Shen
Date: 2025.01.11
Version: 1.1
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Algorithms.STMO.RVEA import RVEA
from ddmtolab.Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class KRVEA:
    """
    Kriging-assisted Reference Vector Guided Evolutionary Algorithm for expensive
    multi/many-objective optimization.

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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, alpha=2.0, wmax=20, mu=5, delta=0.05,
                 save_data=True, save_path='./TestData', name='KRVEA_test', disable_tqdm=True):
        """
        Initialize K-RVEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size per task (default: 100)
        alpha : float, optional
            Parameter controlling the rate of change of penalty (default: 2.0)
        wmax : int, optional
            Number of generations before updating Kriging models (default: 20)
        mu : int, optional
            Number of re-evaluated solutions at each generation (default: 5)
        delta : float, optional
            Threshold for switching between exploitation and exploration (default: 0.05)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'KRVEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.alpha = alpha
        self.wmax = wmax
        self.mu = mu
        self.delta = delta
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the K-RVEA algorithm.

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

        # Generate uniformly distributed reference vectors for each task
        V0 = []
        for i in range(nt):
            v_i, actual_n = uniform_point(n_per_task[i], n_objs[i])
            V0.append(v_i)
            n_per_task[i] = actual_n

        # Initialize adaptive reference vectors
        V = [v.copy() for v in V0]

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Reorganize initial data into task-specific history lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task, interval=self.mu)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task, interval=self.mu)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                m = n_objs[i]
                dim = dims[i]

                # Build GP models for each objective using mo_gp_build
                models = mo_gp_build(decs[i], objs[i], data_type)

                # Create surrogate problem wrapper using mo_gp_predict
                def surrogate_func(x, models=models, dtype=data_type):
                    return mo_gp_predict(models, x, dtype, mse=False)

                surrogate_problem = MTOP()
                surrogate_problem.add_task(objective_func=surrogate_func, dim=dim)

                # Optimize surrogates using RVEA
                rvea = RVEA(surrogate_problem, n=n_per_task[i], max_nfes=n_per_task[i] * self.wmax, save_data=False)
                results = rvea.optimize()

                # Get final population from RVEA
                pop_decs = results.best_decs[0]
                pop_objs = results.best_objs[0]

                # Predict MSE for infill selection using mo_gp_predict
                _, pop_mse = mo_gp_predict(models, pop_decs, data_type, mse=True)

                # Count inactive reference vectors for adaptive selection
                num_inactive_before = self._count_inactive_vectors(objs[i], V0[i])
                num_inactive_after = self._count_inactive_vectors(pop_objs, V0[i])

                # Select mu solutions for re-evaluation
                offspring_decs = self._infill_selection(
                    pop_decs, pop_objs, pop_mse, V[i],
                    num_inactive_before, num_inactive_after,
                    self.delta * n_per_task[i], self.mu, self.alpha
                )

                # Remove duplicates
                offspring_decs = remove_duplicates(offspring_decs, decs[i])

                if offspring_decs.shape[0] > 0:
                    # Evaluate new solutions
                    off_objs, _ = evaluation_single(problem, offspring_decs, i)

                    # Update dataset
                    decs[i] = np.vstack([decs[i], offspring_decs])
                    objs[i] = np.vstack([objs[i], off_objs])

                    # Adapt reference vectors
                    obj_range = objs[i].max(axis=0) - objs[i].min(axis=0)
                    obj_range = np.maximum(obj_range, 1e-6)
                    V[i] = V0[i] * obj_range

                    nfes_per_task[i] += offspring_decs.shape[0]
                    pbar.update(offspring_decs.shape[0])

                    append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    def _count_inactive_vectors(self, objs, V):
        """
        Count inactive reference vectors.

        Parameters
        ----------
        objs : np.ndarray
            Objective values, shape (N, M)
        V : np.ndarray
            Reference vectors, shape (NV, M)

        Returns
        -------
        num_inactive : int
            Number of inactive reference vectors
        """
        # Translate objectives
        objs_translated = objs - objs.min(axis=0)

        # Associate each solution to nearest reference vector
        angle = np.arccos(np.clip(1 - cdist(objs_translated, V, metric='cosine'), -1, 1))
        associate = np.argmin(angle, axis=1)

        # Count inactive vectors
        active = np.unique(associate)
        num_inactive = V.shape[0] - len(active)

        return num_inactive

    def _infill_selection(self, pop_decs, pop_objs, pop_mse, V, num_inactive_before,
                          num_inactive_after, delta, mu, alpha):
        """
        Select solutions for expensive re-evaluation.

        Switches between exploitation (APD-based) and exploration (uncertainty-based)
        depending on the change in inactive reference vectors.

        Parameters
        ----------
        pop_decs : np.ndarray
            Population decision variables
        pop_objs : np.ndarray
            Population predicted objectives
        pop_mse : np.ndarray
            Population predicted MSE
        V : np.ndarray
            Adaptive reference vectors
        num_inactive_before : int
            Number of inactive vectors before surrogate optimization
        num_inactive_after : int
            Number of inactive vectors after surrogate optimization
        delta : float
            Threshold for mode switching
        mu : int
            Number of solutions to select
        alpha : float
            APD penalty parameter

        Returns
        -------
        selected_decs : np.ndarray
            Selected decision variables for re-evaluation
        """
        N, M = pop_objs.shape

        # Get active reference vectors
        objs_translated = pop_objs - pop_objs.min(axis=0)
        angle = np.arccos(np.clip(1 - cdist(objs_translated, V, metric='cosine'), -1, 1))
        associate = np.argmin(angle, axis=1)
        active_vectors = np.unique(associate)

        # Cluster active vectors
        n_clusters = min(mu, len(active_vectors))
        if n_clusters == 0:
            # Fallback: random selection
            indices = np.random.choice(N, size=min(mu, N), replace=False)
            return pop_decs[indices]

        # K-means clustering on active reference vectors
        Va = V[active_vectors]
        cluster_labels = kmeans_clustering(Va, n_clusters)

        # Calculate APD for each solution
        cosine = 1 - cdist(Va, Va, metric='cosine')
        np.fill_diagonal(cosine, 0)
        gamma = np.min(np.arccos(np.clip(cosine, -1, 1)), axis=1)
        gamma = np.maximum(gamma, 1e-6)

        # Map solutions to active vector indices
        solution_to_active = {av: idx for idx, av in enumerate(active_vectors)}

        APD = np.full(N, np.inf)
        for idx, sol_associate in enumerate(associate):
            if sol_associate in solution_to_active:
                active_idx = solution_to_active[sol_associate]
                APD[idx] = (1 + M * alpha * angle[idx, sol_associate] / gamma[active_idx]) * \
                           np.sqrt(np.sum(objs_translated[idx] ** 2))

        # Map solutions to clusters
        solution_clusters = np.zeros(N, dtype=int)
        for idx, sol_associate in enumerate(associate):
            if sol_associate in solution_to_active:
                active_idx = solution_to_active[sol_associate]
                solution_clusters[idx] = cluster_labels[active_idx]

        # Select one solution per cluster
        selected_indices = []
        flag = num_inactive_after - num_inactive_before

        for c in range(n_clusters):
            cluster_mask = solution_clusters == c
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            if flag <= delta:
                # Exploitation: select best APD
                best_idx = cluster_indices[np.argmin(APD[cluster_indices])]
            else:
                # Exploration: select highest uncertainty
                uncertainty = np.mean(pop_mse[cluster_indices], axis=1)
                best_idx = cluster_indices[np.argmax(uncertainty)]

            selected_indices.append(best_idx)

        if len(selected_indices) == 0:
            indices = np.random.choice(N, size=min(mu, N), replace=False)
            return pop_decs[indices]

        return pop_decs[selected_indices]
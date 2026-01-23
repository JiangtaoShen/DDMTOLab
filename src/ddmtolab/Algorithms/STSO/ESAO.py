"""
Exploration-exploitation Switching Assisted Optimization (ESAO)

This module implements ESAO for expensive single-objective optimization problems.

References
----------
    [1] Wang, Xinjing, et al. "A novel evolutionary sampling assisted optimization method for high-dimensional expensive problems." IEEE Transactions on Evolutionary Computation 23.5 (2019): 815-827.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.13
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.STSO.DE import DE
from ddmtolab.Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class ESAO:
    """
    Exploration-exploitation Switching Assisted Optimization for expensive optimization problems.

    This algorithm adaptively switches between global and local search based on improvement:
    1. Global search: RBF model built on all data points
    2. Local search: RBF model built on top 2*dim points nearest to the best
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
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

    def __init__(self, problem, n_initial=None, max_nfes=None, save_data=True,
                 save_path='./TestData', name='ESAO_test', disable_tqdm=True):
        """
        Initialize ESAO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'ESAO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        # DE parameters for surrogate optimization
        self.global_popsize = 100
        self.global_max_gen = 5
        self.local_popsize = 100
        self.local_max_gen = 50

    def optimize(self):
        """
        Execute the ESAO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Initialize history: reorganize initial data into task-specific lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        # Current working dataset (accumulated samples)
        current_decs = [decs[i].copy() for i in range(nt)]
        current_objs = [objs[i].copy() for i in range(nt)]

        # Track search state for each task: 0 = global, 1 = local
        search_state = [0] * nt
        # Track best objective value before each iteration for improvement check
        best_before = [np.min(current_objs[i]) for i in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                X = current_decs[i]
                Y = current_objs[i]

                # Build surrogate models
                global_model = self._build_rbf_model(X, Y)
                local_model = self._build_local_rbf_model(X, Y, dim)

                # Execute search based on current state
                if search_state[i] == 0:
                    # Global search
                    candidate_np = self._global_search(global_model, dim)
                else:
                    # Local search
                    candidate_np = self._local_search(local_model, X, Y, dim)

                # Ensure uniqueness: avoid duplicate sampling
                candidate_np = self._ensure_uniqueness(candidate_np, X, dim)

                # Evaluate the candidate solution
                obj, _ = evaluation_single(problem, candidate_np, i)

                # Update current working dataset
                current_decs[i] = np.vstack([current_decs[i], candidate_np])
                current_objs[i] = np.vstack([current_objs[i], obj])

                # Append to history
                all_decs[i].append(current_decs[i].copy())
                all_objs[i].append(current_objs[i].copy())

                nfes_per_task[i] += 1
                pbar.update(1)

                # Check improvement and update search state
                best_after = np.min(current_objs[i])
                if best_after < best_before[i]:
                    # Improvement found, continue current search
                    pass
                else:
                    # No improvement, switch search state
                    search_state[i] = 1 - search_state[i]

                # Update best_before for next iteration
                best_before[i] = best_after

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    # Build RBF surrogate model using all data points
    def _build_rbf_model(self, X, Y):

        Y = Y.flatten()
        n_samples, dim = X.shape

        if n_samples > 1:
            # Compute pairwise distances
            dist_matrix = cdist(X, X, metric='euclidean')
            max_dist = dist_matrix.max()
            # Adaptive spread estimation (MATLAB newrbe formula)
            spread = max_dist / (dim * n_samples) ** (1.0 / dim)
        else:
            spread = 1.0

        # Use Gaussian RBF
        try:
            rbf_interpolator = RBFInterpolator(X, Y, kernel='gaussian', epsilon=1.0 / spread)
        except:
            # Fallback to thin_plate_spline if gaussian fails
            rbf_interpolator = RBFInterpolator(X, Y, kernel='thin_plate_spline')

        def rbf_model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            pred = rbf_interpolator(x)
            return pred.reshape(-1, 1)

        return rbf_model

    # Build local RBF surrogate model using top 2*dim points nearest to the best
    def _build_local_rbf_model(self, X, Y, dim):

        # Number of local samples: min(2*dim, available samples)
        n_local = min(2 * dim, len(X))
        n_local = max(n_local, 2)

        # Find the best point
        idx_best = np.argmin(Y)
        best_point = X[idx_best:idx_best + 1]

        # Compute distances to the best point
        distances = cdist(X, best_point, metric='euclidean').flatten()

        # Select the n_local nearest points
        idx_sorted = np.argsort(distances)[:n_local]

        X_local = X[idx_sorted]
        Y_local = Y[idx_sorted]

        # Build RBF model on local data
        return self._build_rbf_model(X_local, Y_local)

    # Global search using DE to optimize the global surrogate model
    def _global_search(self, global_model, dim):

        # Create surrogate problem wrapper
        def surrogate_objective(x):
            return global_model(x)

        surrogate_problem = MTOP()
        surrogate_problem.add_task(objective_func=surrogate_objective, dim=dim)

        # Run DE
        max_nfes = self.global_popsize * self.global_max_gen
        de = DE(surrogate_problem, n=self.global_popsize, max_nfes=max_nfes, disable_tqdm=True)
        results = de.optimize()

        # Get best solution
        best_solution = results.best_decs[0].reshape(1, -1)

        # Ensure solution is within [0, 1]
        best_solution = np.clip(best_solution, 0.0, 1.0)

        return best_solution

    # Local search using DE to optimize the local surrogate model
    def _local_search(self, local_model, X, Y, dim):

        # Define local search bounds based on local data region
        n_local = min(2 * dim, len(X))
        idx_best = np.argmin(Y)
        best_point = X[idx_best:idx_best + 1]
        distances = cdist(X, best_point, metric='euclidean').flatten()
        idx_sorted = np.argsort(distances)[:n_local]
        X_local = X[idx_sorted]

        # Get bounds of local region
        lb_local = np.min(X_local, axis=0)
        ub_local = np.max(X_local, axis=0)

        # Handle case where lb == ub (add small margin)
        range_mask = (ub_local - lb_local) < 1e-10
        if np.any(range_mask):
            lb_local[range_mask] = np.maximum(0.0, lb_local[range_mask] - 0.05)
            ub_local[range_mask] = np.minimum(1.0, ub_local[range_mask] + 0.05)

        # Create surrogate problem wrapper with local bounds
        def surrogate_objective(x):
            # x is in [0,1], denormalize to local bounds
            x_denorm = lb_local + x * (ub_local - lb_local)
            return local_model(x_denorm)

        surrogate_problem = MTOP()
        surrogate_problem.add_task(objective_func=surrogate_objective, dim=dim)

        # Run DE
        max_nfes = self.local_popsize * self.local_max_gen
        de = DE(surrogate_problem, n=self.local_popsize, max_nfes=max_nfes, disable_tqdm=True)
        results = de.optimize()

        # Get best solution in [0,1] space and denormalize
        best_x_normalized = results.best_decs[0].reshape(1, -1)
        best_solution = lb_local + best_x_normalized * (ub_local - lb_local)

        # Ensure solution is within [0, 1]
        best_solution = np.clip(best_solution, 0.0, 1.0)

        return best_solution

    # Ensure candidate is not too close to existing samples
    def _ensure_uniqueness(self, candidate, X, dim, epsilon=5e-3, max_trials=50):

        scales = np.linspace(0.1, 1.0, max_trials)
        trial_count = 0

        while trial_count < max_trials:
            # Compute minimum distance to existing samples
            dist = cdist(candidate, X, metric='euclidean')
            min_dist = np.min(dist)

            if min_dist >= epsilon:
                break  # Candidate is sufficiently unique

            # Apply perturbation
            scale = scales[trial_count % max_trials]
            perturbation = scale * (np.random.rand(1, dim) - 0.5)
            candidate = candidate + perturbation

            # Clip to [0, 1]
            candidate = np.clip(candidate, 0.0, 1.0)
            trial_count += 1

        return candidate
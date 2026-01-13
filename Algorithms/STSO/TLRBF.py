"""
Three-Level Radial Basis Function Method (TLRBF)

This module implements TLRBF for expensive single-objective optimization problems.

References
----------
    [1] Li, Genghui, et al. "A three-level radial basis function method for expensive optimization." IEEE Transactions on Cybernetics 52.7 (2021): 5720-5731.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.13
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.interpolate import RBFInterpolator
from Methods.Algo_Methods.algo_utils import *
from Algorithms.STSO.GA import GA
from Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class TLRBF:
    """
    Three-Level Radial Basis Function Method for expensive optimization problems.

    This algorithm uses three search strategies in rotation:
    1. Global search: Random sampling with distance filtering
    2. Subregion search: FCM clustering + local RBF models
    3. Local search: K-nearest neighbors + local RBF model
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
                 save_path='./TestData', name='TLRBF_test', disable_tqdm=True):
        """
        Initialize TLRBF algorithm.

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
            Name for the experiment (default: 'TLRBF_test')
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

    def optimize(self):
        """
        Execute the TLRBF algorithm.

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

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]

                # Determine search state: 0=global, 1=subregion, 2=local
                state = (nfes_per_task[i] - n_initial_per_task[i]) % 3

                if state == 0:
                    # ===== Global Search =====
                    candidate_np = self._global_search(current_decs[i], current_objs[i], dim)

                elif state == 1:
                    # ===== Subregion Search =====
                    candidate_np = self._subregion_search(current_decs[i], current_objs[i], dim)

                else:  # state == 2
                    # ===== Local Search =====
                    candidate_np = self._local_search(current_decs[i], current_objs[i], dim)

                # Evaluate the candidate solution
                obj, _ = evaluation_single(problem, candidate_np, i)

                # Update current working dataset
                current_decs[i] = np.vstack([current_decs[i], candidate_np])
                current_objs[i] = np.vstack([current_objs[i], obj])

                # Append to history (store the entire current dataset as a snapshot)
                all_decs[i].append(current_decs[i].copy())
                all_objs[i].append(current_objs[i].copy())

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    # Build RBF surrogate model
    def _build_rbf_model(self, X, Y):

        Y = Y.flatten()

        # Calculate spread parameter
        n_samples, dim = X.shape
        if n_samples > 1:
            # Compute pairwise distances
            dist_matrix = cdist(X, X, metric='euclidean')
            max_dist = dist_matrix.max()
            # Adaptive spread estimation
            spread = max_dist / (dim * n_samples) ** (1.0 / dim)
        else:
            spread = 1.0

        # Use Gaussian RBF
        try:
            rbf_model = RBFInterpolator(X, Y, kernel='gaussian', epsilon=1.0 / spread)
        except:
            # Fallback to thin_plate_spline if gaussian fails
            rbf_model = RBFInterpolator(X, Y, kernel='thin_plate_spline')

        return rbf_model

    # Global search with distance filtering
    def _global_search(self, decs_i, objs_i, dim):

        alpha = 0.4
        m = 200 * dim

        # Build RBF model
        rbf_model = self._build_rbf_model(decs_i, objs_i)

        # Generate random candidates in [0,1]^dim
        solutions_global = np.random.rand(m, dim)

        # Distance filtering
        dist = cdist(solutions_global, decs_i, metric='euclidean')
        mindist = np.min(dist, axis=1)
        deltag = alpha * np.max(mindist)

        # Remove candidates too close to existing points
        valid_mask = mindist > deltag
        solutions_global = solutions_global[valid_mask]

        if len(solutions_global) == 0:
            # If all filtered out, return a random point
            return np.random.rand(1, dim)

        # Predict and select best
        objs_pre = rbf_model(solutions_global)
        idx = np.argmin(objs_pre)
        candidate = solutions_global[idx:idx + 1, :]

        return candidate

    # Subregion search using FCM clustering
    def _subregion_search(self, decs_i, objs_i, dim):

        N = len(decs_i)
        L1 = 100
        L2 = 80

        if N <= L1:
            # Use all data
            X_subregion = decs_i
            Y_subregion = objs_i
            lb_subregion = np.zeros(dim)
            ub_subregion = np.ones(dim)
        else:
            # Compute number of clusters
            no_clusters = 1 + int(np.ceil((N - L1) / L2))

            # Normalize data for clustering
            X_min = decs_i.min(axis=0)
            X_max = decs_i.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range < 1e-10] = 1.0
            X_normalized = (decs_i - X_min) / X_range

            # Perform KMeans clustering (approximation of FCM)
            kmeans = KMeans(n_clusters=no_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_normalized)

            # Compute soft membership using inverse distance weighting
            distances = kmeans.transform(X_normalized)
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            U = 1.0 / distances
            U = U / U.sum(axis=1, keepdims=True)

            # Select top L1 points from each cluster based on membership
            X_clusters = []
            Y_clusters = []
            mean_objs = []

            for k in range(no_clusters):
                # Sort by membership in cluster k (descending)
                membership_k = U[:, k]
                idx_sorted = np.argsort(-membership_k)[:L1]

                X_clusters.append(decs_i[idx_sorted])
                Y_clusters.append(objs_i[idx_sorted])
                mean_objs.append(np.mean(objs_i[idx_sorted]))

            # Probabilistic cluster selection (prefer clusters with lower mean objective)
            mean_objs = np.array(mean_objs)
            # Rank clusters by mean objective (1 = best)
            ranks = np.argsort(np.argsort(mean_objs)) + 1
            # Probability proportional to rank (higher rank = worse cluster = lower prob)
            probs = ranks / no_clusters
            probs = probs / probs.sum()

            # Select cluster using rejection sampling
            selected_cluster = None
            while selected_cluster is None:
                sid = np.random.randint(0, no_clusters)
                if np.random.rand() <= probs[sid]:
                    selected_cluster = sid

            X_subregion = X_clusters[selected_cluster]
            Y_subregion = Y_clusters[selected_cluster]

            # Define subregion bounds
            lb_subregion = np.min(X_subregion, axis=0)
            ub_subregion = np.max(X_subregion, axis=0)

        # Build RBF model on subregion
        rbf_model = self._build_rbf_model(X_subregion, Y_subregion)

        # Optimize using GA on surrogate (50 generations)
        candidate = self._optimize_surrogate(rbf_model, lb_subregion, ub_subregion, dim, max_gen=50)

        return candidate

    # Local search using k-nearest neighbors
    def _local_search(self, decs_i, objs_i, dim):

        k = min(2 * dim, len(decs_i) - 1)
        k = max(k, 1)  # Ensure at least 1 neighbor

        # Find k nearest neighbors to the best point
        idx_min = np.argmin(objs_i)
        dist = cdist(decs_i, decs_i[idx_min:idx_min + 1], metric='euclidean').flatten()
        idx_sorted = np.argsort(dist)[:k + 1]  # +1 to include the best point itself

        X_local = decs_i[idx_sorted]
        Y_local = objs_i[idx_sorted]

        # Build RBF model on local region
        rbf_model = self._build_rbf_model(X_local, Y_local)

        # Define local bounds
        lb_local = np.min(X_local, axis=0)
        ub_local = np.max(X_local, axis=0)

        # Optimize using GA on surrogate (10 generations for local search)
        candidate = self._optimize_surrogate(rbf_model, lb_local, ub_local, dim, max_gen=10)

        return candidate

    # Optimize surrogate model using Genetic Algorithm
    def _optimize_surrogate(self, surrogate_model, lb, ub, dim, n_pop=50, max_gen=50):

        # Ensure bounds are valid
        lb = np.asarray(lb)
        ub = np.asarray(ub)

        # Handle case where lb == ub
        range_mask = (ub - lb) < 1e-10
        if np.any(range_mask):
            ub[range_mask] = lb[range_mask] + 1e-6

        # Create surrogate problem wrapper
        def surrogate_func(x):
            x_denorm = lb + x * (ub - lb)
            return surrogate_model(x_denorm).reshape(-1, 1)

        surrogate_problem = MTOP()
        surrogate_problem.add_task(objective_func=surrogate_func, dim=dim)

        # Run GA
        max_nfes = n_pop * max_gen
        ga = GA(surrogate_problem, n=n_pop, max_nfes=max_nfes, disable_tqdm=True)
        results = ga.optimize()

        # Get best solution in [0,1] space
        best_x_normalized = results.best_decs[0].reshape(1, -1)

        # Denormalize to [lb, ub] space
        best_solution = lb + best_x_normalized * (ub - lb)

        # Ensure solution is within [0, 1] (original problem bounds)
        best_solution = np.clip(best_solution, 0.0, 1.0)

        return best_solution
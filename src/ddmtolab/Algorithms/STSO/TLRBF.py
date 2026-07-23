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
Date: 2026.07.23
Version: 1.1
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.stats import qmc
from ddmtolab.Methods.Algo_Methods.algo_utils import *
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
        'n_tasks': '[1, K]',
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
                 save_path='./Data', name='TLRBF', disable_tqdm=True):
        """
        Initialize TLRBF algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'TLRBF')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
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

                # Ensure uniqueness before evaluation
                candidate_np = self._ensure_uniqueness(candidate_np, current_decs[i], dim)

                # Evaluate the candidate solution
                obj, _ = evaluation_single(problem, candidate_np, i)

                # Update current working dataset
                current_decs[i] = np.vstack([current_decs[i], candidate_np])
                current_objs[i] = np.vstack([current_objs[i], obj])

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [current_decs[i].copy() for i in range(nt)]
        db_objs = [current_objs[i].copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    # Global search with distance filtering
    def _global_search(self, decs_i, objs_i, dim):

        alpha = 0.4
        m = 200 * dim

        # Build RBF model on the full database
        rbf_model = newrbe_surrogate(decs_i, objs_i)

        # Generate random candidates in [0,1]^dim
        solutions_global = np.random.rand(m, dim)

        # Distance filtering: drop candidates too close to existing points
        dist = cdist(solutions_global, decs_i, metric='euclidean')
        mindist = np.min(dist, axis=1)
        deltag = alpha * np.max(mindist)
        solutions_global = solutions_global[mindist > deltag]

        if len(solutions_global) == 0:
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
        else:
            # Compute number of clusters
            no_clusters = 1 + int(np.ceil((N - L1) / L2))

            # Normalize each dimension to [0,1] for clustering
            X_min = decs_i.min(axis=0)
            X_max = decs_i.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range < 1e-12] = 1.0
            X_normalized = (decs_i - X_min) / X_range

            # Fuzzy c-means clustering; membership matrix (N, no_clusters)
            U = self._fcm(X_normalized, no_clusters).T

            # Select top L1 points from each cluster by membership
            X_clusters = []
            Y_clusters = []
            mean_objs = []
            for k in range(no_clusters):
                idx_sorted = np.argsort(-U[:, k], kind='stable')[:L1]
                X_clusters.append(decs_i[idx_sorted])
                Y_clusters.append(objs_i[idx_sorted])
                mean_objs.append(np.mean(objs_i[idx_sorted]))

            # Probabilistic cluster selection (MATLAB: [~,idx]=sort(mper,'descend'); pro=idx/K)
            mean_objs = np.array(mean_objs)
            idx_desc = np.argsort(-mean_objs, kind='stable')
            pro = (idx_desc + 1) / no_clusters

            # Rejection sampling (MATLAB: sid=randi(K); while rand>pro(sid): sid=randi(K))
            sid = np.random.randint(no_clusters)
            while np.random.rand() > pro[sid]:
                sid = np.random.randint(no_clusters)

            X_subregion = X_clusters[sid]
            Y_subregion = Y_clusters[sid]

        # Subregion bounds are the bounding box of the selected data
        lb_subregion = np.min(X_subregion, axis=0)
        ub_subregion = np.max(X_subregion, axis=0)

        # Build RBF model on subregion and search it with the template EA
        rbf_model = newrbe_surrogate(X_subregion, Y_subregion)
        candidate = self._template_ea_search(rbf_model, X_subregion, Y_subregion,
                                             lb_subregion, ub_subregion)

        return candidate

    # Local search using k-nearest neighbors
    def _local_search(self, decs_i, objs_i, dim):

        k = min(2 * dim, len(decs_i) - 1)
        k = max(k, 1)

        # Find k nearest neighbors to the best point (including the best itself)
        idx_min = np.argmin(objs_i)
        dist = cdist(decs_i, decs_i[idx_min:idx_min + 1], metric='euclidean').flatten()
        idx_sorted = np.argsort(dist, kind='stable')[:k]

        X_local = decs_i[idx_sorted]
        Y_local = objs_i[idx_sorted]

        # Local bounds are the bounding box of the selected data
        lb_local = np.min(X_local, axis=0)
        ub_local = np.max(X_local, axis=0)

        # Build RBF model on local region and search it with the template EA
        rbf_model = newrbe_surrogate(X_local, Y_local)
        candidate = self._template_ea_search(rbf_model, X_local, Y_local, lb_local, ub_local)

        return candidate

    # ==================== Surrogate Search (acquisition template) ====================

    def _template_ea_search(self, surrogate, X_data, Y_data, lb, ub,
                            popsize=50, n_queries=50):
        """
        Evolutionary search on a surrogate: 'elite+random' initialization,
        SBX + polynomial mutation + variable swap offspring, and roulette
        wheel selection, run for n_queries generations.
        """
        lb = np.asarray(lb, dtype=float).copy()
        ub = np.asarray(ub, dtype=float).copy()
        # Guard against zero-width dimensions for normalization
        degenerate = (ub - lb) < 1e-12
        ub[degenerate] = lb[degenerate] + 1e-6

        Y_flat = Y_data.flatten()
        n_half = popsize // 2
        if len(Y_flat) >= n_half:
            idx = np.argsort(Y_flat, kind='stable')[:n_half]
            pop = np.vstack([X_data[idx], self._lhs(popsize - n_half, lb, ub)])
        else:
            pop = np.vstack([X_data, self._lhs(popsize - len(Y_flat), lb, ub)])
        pop_acq = np.asarray(surrogate(pop)).flatten()

        for _ in range(n_queries):
            offspring = self._ea_offspring(pop, lb, ub)
            off_acq = np.asarray(surrogate(offspring)).flatten()
            pop, pop_acq = self._roulette_select(pop, pop_acq, offspring, off_acq, popsize)

        best_idx = np.argmin(pop_acq)
        return pop[best_idx:best_idx + 1].copy()

    def _ea_offspring(self, parents, lb, ub, muc=15, mum=15, probswap=0.5):
        """EA offspring: SBX crossover + polynomial mutation + variable swap."""
        popsize, dim = parents.shape
        range_vec = ub - lb

        # Normalize to [0,1] within bounds
        pop_norm = (parents - lb) / range_vec
        offspring = np.zeros((popsize, dim))
        ind_order = np.random.permutation(popsize)

        for i in range(popsize // 2):
            p1 = ind_order[i]
            p2 = ind_order[i + popsize // 2]

            # SBX crossover (shared cf vector for both children)
            u = np.random.rand(dim)
            cf = np.where(u <= 0.5,
                          (2 * u) ** (1 / (muc + 1)),
                          (2 * (1 - u)) ** (-1 / (muc + 1)))

            child1 = np.clip(0.5 * ((1 + cf) * pop_norm[p1] + (1 - cf) * pop_norm[p2]), 0, 1)
            child2 = np.clip(0.5 * ((1 + cf) * pop_norm[p2] + (1 - cf) * pop_norm[p1]), 0, 1)

            # Polynomial mutation
            for child in [child1, child2]:
                for j in range(dim):
                    if np.random.rand() < 1 / dim:
                        u_val = np.random.rand()
                        if u_val <= 0.5:
                            delta = (2 * u_val) ** (1 / (1 + mum)) - 1
                            child[j] += delta * child[j]
                        else:
                            delta = 1 - (2 * (1 - u_val)) ** (1 / (1 + mum))
                            child[j] += delta * (1 - child[j])
                child[:] = np.clip(child, 0, 1)

            # Variable swap
            swap = np.random.rand(dim) >= probswap
            temp = child2[swap].copy()
            child2[swap] = child1[swap]
            child1[swap] = temp

            # Denormalize back to bounds
            offspring[i] = lb + child1 * range_vec
            offspring[i + popsize // 2] = lb + child2 * range_vec

        return offspring

    def _roulette_select(self, pop, pop_objs, offspring, off_objs, popsize):
        """Roulette wheel selection over parents + offspring (inverse fitness)."""
        total_pop = np.vstack([pop, offspring])
        total_objs = np.concatenate([pop_objs, off_objs])

        shift = min(np.min(total_objs), 0)
        fit = 1.0 / (total_objs - shift + 1e-6)
        cum_fit = np.cumsum(fit)
        cum_fit /= cum_fit[-1]

        idx = np.searchsorted(cum_fit, np.random.rand(popsize))
        idx = np.clip(idx, 0, len(total_objs) - 1)

        return total_pop[idx].copy(), total_objs[idx].copy()

    def _lhs(self, n, lb, ub):
        """Latin hypercube samples scaled to [lb, ub]."""
        if n <= 0:
            return np.empty((0, len(lb)))
        sample = qmc.LatinHypercube(d=len(lb)).random(n)
        return lb + sample * (ub - lb)

    # ==================== FCM Clustering ====================

    def _fcm(self, data, n_clusters, expo=2.0, max_iter=100, min_impro=1e-5):
        """
        Fuzzy c-means clustering (MATLAB fcm equivalent).

        Returns the membership matrix U of shape (n_clusters, n_points).
        """
        n = data.shape[0]
        U = np.random.rand(n_clusters, n)
        U = U / U.sum(axis=0, keepdims=True)

        obj_prev = None
        for _ in range(max_iter):
            mf = U ** expo
            centers = (mf @ data) / mf.sum(axis=1, keepdims=True)
            dist = cdist(centers, data)
            obj = np.sum((dist ** 2) * mf)
            with np.errstate(divide='ignore', invalid='ignore'):
                tmp = dist ** (-2.0 / (expo - 1.0))
                U = tmp / tmp.sum(axis=0, keepdims=True)
            if obj_prev is not None and abs(obj - obj_prev) < min_impro:
                break
            obj_prev = obj

        return U

    # ==================== Utilities ====================

    def _ensure_uniqueness(self, candidate, X, dim, epsilon=5e-3, n_scales=50, max_trials=1000):
        """Perturb the candidate until it is not too close to existing samples."""
        scales = np.linspace(0.1, 1.0, n_scales)
        c = 0
        while cdist(candidate, X, metric='chebyshev').min() < epsilon and c < max_trials:
            perturbation = scales[c % n_scales] * (np.random.rand(1, dim) - 0.5)
            candidate = np.clip(candidate + perturbation, 0.0, 1.0)
            c += 1
        return candidate

"""
Lipschitz Surrogate-Assisted Differential Evolution (LSADE)

This module implements LSADE for expensive single-objective optimization problems.

References
----------
    [1] Kudela, J., & Matousek, R. (2023). Combining Lipschitz and RBF surrogate models for high-dimensional computationally expensive problems. Information Sciences, 619, 457-477.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.19
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class LSADE:
    """
    Lipschitz Surrogate-Assisted Differential Evolution for expensive optimization problems.

    Uses three surrogate strategies in deterministic rotation:
    1. RBF prescreening: DE/best/1 on Gaussian RBF model
    2. Lipschitz prescreening: DE/best/1 on Lipschitz lower-bound surrogate
    3. Local optimization: SQP on local RBF model around best solutions
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
                 save_path='./Data', name='LSADE', disable_tqdm=True):
        """
        Initialize LSADE algorithm.

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
            Name for the experiment (default: 'LSADE')
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
        Execute the LSADE algorithm.

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

        # Current working dataset
        current_decs = [decs[i].copy() for i in range(nt)]
        current_objs = [objs[i].copy() for i in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                X = current_decs[i]
                Y = current_objs[i]

                # Deterministic state rotation: 0=RBF, 1=Lipschitz, 2=Local
                state = (nfes_per_task[i] - n_initial_per_task[i]) % 3

                if state == 0:
                    candidate = self._state_rbf(X, Y, dim)
                elif state == 1:
                    candidate = self._state_lipschitz(X, Y, dim)
                else:
                    candidate = self._state_local(X, Y, dim)

                # Ensure uniqueness
                candidate = self._ensure_uniqueness(candidate, X, dim)

                # Evaluate
                obj, _ = evaluation_single(problem, candidate, i)

                # Update dataset
                current_decs[i] = np.vstack([current_decs[i], candidate])
                current_objs[i] = np.vstack([current_objs[i], obj])

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [current_decs[i].copy() for i in range(nt)]
        db_objs = [current_objs[i].copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data
        )

        return results

    # ==================== State Methods ====================

    def _state_rbf(self, X, Y, dim):
        """
        State 0: RBF prescreening with DE/best/1.

        Builds Gaussian RBF on all data, generates one set of DE/best/1
        offspring, and returns the best offspring by surrogate prediction.
        """
        model = self._build_rbf_model(X, Y)
        return self._prescreening(model, X, Y, dim)

    def _state_lipschitz(self, X, Y, dim):
        """
        State 1: Lipschitz prescreening with DE/best/1.

        Estimates the Lipschitz constant, builds a Lipschitz lower-bound
        surrogate, and performs DE/best/1 prescreening.
        """
        Y_flat = Y.flatten()
        k = self._k_est(Y_flat, X)

        def lip_model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return self._f_lip(Y_flat, X, k, x)

        return self._prescreening(lip_model, X, Y, dim)

    def _state_local(self, X, Y, dim):
        """
        State 2: Local optimization (SQP) on RBF.

        Selects top min(N, 3*dim) solutions by fitness, builds local RBF,
        and runs SQP optimization from a random starting point within
        local bounds.
        """
        N = len(X)
        n_local = min(N, 3 * dim)

        Y_flat = Y.flatten()
        idx = np.argsort(Y_flat)[:n_local]
        X_local = X[idx]
        Y_local = Y[idx]

        lb_local = np.min(X_local, axis=0)
        ub_local = np.max(X_local, axis=0)

        # Handle degenerate bounds
        mask = (ub_local - lb_local) < 1e-10
        lb_local[mask] = np.maximum(0.0, lb_local[mask] - 0.05)
        ub_local[mask] = np.minimum(1.0, ub_local[mask] + 0.05)

        model = self._build_rbf_model(X_local, Y_local)

        # Random starting point within local bounds (matching MATLAB behavior)
        x0 = lb_local + (ub_local - lb_local) * np.random.rand(dim)
        bounds = list(zip(lb_local, ub_local))

        def obj_func(x):
            return model(x.reshape(1, -1)).flatten()[0]

        try:
            result = minimize(obj_func, x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 20, 'disp': False})
            candidate = result.x.reshape(1, -1)
        except Exception:
            candidate = x0.reshape(1, -1)

        candidate = np.clip(candidate, 0.0, 1.0)
        return candidate

    # ==================== Surrogate Models ====================

    def _build_rbf_model(self, X, Y):
        """Build Gaussian RBF surrogate model (MATLAB newrbe equivalent)."""
        predict = newrbe_surrogate(X, Y)

        def model(x):
            return np.asarray(predict(x)).reshape(-1, 1)

        return model

    def _k_est(self, Y, X):
        """
        Estimate the Lipschitz constant from sampled data.

        Computes pairwise Lipschitz estimates k = |f(i)-f(j)| / ||x(i)-x(j)||,
        takes the maximum, and applies exponential scaling.

        Parameters
        ----------
        Y : np.ndarray, shape (n,)
            Function values at sampled points
        X : np.ndarray, shape (n, d)
            Decision variables at sampled points

        Returns
        -------
        float
            Estimated Lipschitz constant
        """
        alpha = 0.01

        # Pairwise distances and objective differences
        dist_matrix = cdist(X, X)
        np.fill_diagonal(dist_matrix, np.inf)
        Y_diff = np.abs(Y.reshape(-1, 1) - Y.reshape(1, -1))

        with np.errstate(divide='ignore', invalid='ignore'):
            k_matrix = Y_diff / dist_matrix
            k_matrix[~np.isfinite(k_matrix)] = 0.0

        k_max = np.max(k_matrix)

        if k_max > 0:
            i_t = int(np.ceil(np.log(k_max) / np.log(1 + alpha)))
            est = (1 + alpha) ** i_t
        else:
            est = 1.0
        return est

    def _f_lip(self, Y, X, k, X_query):
        """
        Evaluate the Lipschitz lower-bound surrogate at query points.

        For each query point, computes min_i { Y(i) + k * ||X(i) - x_query|| }
        which provides a lower bound on the true function value.

        Parameters
        ----------
        Y : np.ndarray, shape (n,)
            Observed function values
        X : np.ndarray, shape (n, d)
            Observed decision variables
        k : float
            Lipschitz constant
        X_query : np.ndarray, shape (nq, d)
            Query points

        Returns
        -------
        np.ndarray, shape (nq, 1)
            Predicted surrogate values
        """
        # dist: (nq, n) pairwise distances
        dist = cdist(X_query, X)
        # vals[i,j] = Y[j] + k * ||X_query[i] - X[j]||
        vals = Y.flatten() + k * dist
        # Take minimum over all observed points for each query
        result = np.min(vals, axis=1)
        return result.reshape(-1, 1)

    # ==================== DE Prescreening ====================

    def _prescreening(self, surrogate_func, X_full, Y_full, dim, popsize=50):
        """
        Single-generation DE/best/1 prescreening with best+randperm initialization.

        1. Initialize population: best individual + (popsize-1) random permutation from database
        2. Generate offspring via DE/best/1 (one generation)
        3. Evaluate offspring on surrogate
        4. Return the best offspring

        Parameters
        ----------
        surrogate_func : callable
            Surrogate model function, accepts (n, d) array, returns (n, 1) predictions
        X_full : np.ndarray
            Full database of decision variables
        Y_full : np.ndarray
            Full database of objectives
        dim : int
            Problem dimension
        popsize : int
            Population size for DE (default: 50)
        """
        Y_flat = Y_full.flatten()
        N = len(Y_flat)

        # best+randperm initialization: best individual first, then random permutation
        best_idx = np.argmin(Y_flat)
        seq = np.random.permutation(N)

        if N >= popsize:
            init_idx = np.concatenate([[best_idx], seq[:popsize - 1]])
        else:
            init_idx = np.concatenate([[best_idx], seq])

        pop = X_full[init_idx].copy()
        pop_objs = Y_flat[init_idx].copy()

        # Fill remaining slots with random solutions if database is small
        if len(pop) < popsize:
            n_extra = popsize - len(pop)
            extra = np.random.rand(n_extra, dim)
            pop = np.vstack([pop, extra])
            pop_objs = np.concatenate([pop_objs, surrogate_func(extra).flatten()])

        # Generate offspring via DE/best/1 (single generation)
        offspring = self._de_best1(pop, pop_objs, F=0.5, CR=0.8)

        # Evaluate offspring on surrogate
        off_objs = surrogate_func(offspring).flatten()

        # Return the best offspring
        best_off_idx = np.argmin(off_objs)
        return offspring[best_off_idx:best_off_idx + 1]

    def _de_best1(self, parents, objs, F=0.5, CR=0.8):
        """
        DE/best/1/bin offspring generation in [0,1] space.

        Parameters
        ----------
        parents : np.ndarray
            Parent population, shape (n, d), in [0,1]
        objs : np.ndarray
            Objective values, shape (n,)
        F : float
            Differential weight
        CR : float
            Crossover rate

        Returns
        -------
        offspring : np.ndarray
            Offspring population, shape (n, d), clipped to [0,1]
        """
        popsize, dim = parents.shape
        best_idx = np.argmin(objs)
        best = parents[best_idx]

        offspring = parents.copy()
        for i in range(popsize):
            idxs = np.delete(np.arange(popsize), i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            mutant = best + F * (parents[r1] - parents[r2])
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) <= CR) | (np.arange(dim) == j_rand)
            offspring[i, mask] = mutant[mask]

        return np.clip(offspring, 0, 1)

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

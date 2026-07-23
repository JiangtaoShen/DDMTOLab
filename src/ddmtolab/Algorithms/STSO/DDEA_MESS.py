"""
Data-Driven Evolutionary Algorithm with Multi-Evolutionary Sampling Strategy (DDEA-MESS)

This module implements DDEA-MESS for expensive single-objective optimization problems.

References
----------
    [1] Yu, F., Gong, W., & Zhen, H. (2022). A data-driven evolutionary algorithm with multi-evolutionary sampling strategy for expensive optimization. Knowledge-Based Systems, 242, 108436.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.07.23
Version: 1.1
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class DDEA_MESS:
    """
    Data-Driven Evolutionary Algorithm with Multi-Evolutionary Sampling Strategy.

    Dynamically selects from three search strategies based on evaluation budget usage:
    1. Global search: one-generation DE prescreening on RBF built from the first min(N, 300) samples
    2. Local search: DE/best/1 (10 generations) on RBF built from the top tau samples by fitness
    3. Interior-point search: local optimization on RBF around the best solution
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
                 save_path='./Data', name='DDEA-MESS', disable_tqdm=True):
        """
        Initialize DDEA-MESS algorithm.

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
            Name for the experiment (default: 'DDEA-MESS')
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
        Execute the DDEA-MESS algorithm.

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

                # Select strategy using MESS (MATLAB passes its FEsMax=500 here)
                strategy_id = self._mess(nfes_per_task[i], max_nfes_per_task[i])

                if strategy_id == 1:
                    candidate = self._strategy_global(X, Y, dim)
                elif strategy_id == 2:
                    candidate = self._strategy_local(X, Y, dim)
                else:
                    candidate = self._strategy_interior_point(X, Y, dim)

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

    def _mess(self, fes_used, fes_max):
        """
        Multi-Evolutionary Sampling Strategy selector.

        Returns
        -------
        int
            Strategy ID: 1 (global), 2 (local), or 3 (interior point)
        """
        ratio = fes_used / fes_max
        beta = (1 - ratio ** 3) ** 2
        alpha = abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))
        P3 = (1 - alpha) if alpha > 2 / 3 else 1 / 3
        P1 = (1 - P3) / 2
        r = np.random.rand()
        if r <= P1:
            return 1  # global search
        elif r <= 2 * P1:
            return 2  # local search
        else:
            return 3  # interior point search

    def _strategy_global(self, X, Y, dim):
        """
        Strategy 1: Global prescreening (one DE generation) on an RBF model
        built from the first min(N, 300) samples, searched over [0, 1]^dim.
        """
        N = len(X)
        m = min(N, 300)

        model = newrbe_surrogate(X[:m], Y[:m])

        candidate = self._de_search(
            model, X, Y,
            lb=np.zeros(dim), ub=np.ones(dim),
            dim=dim, popsize=50, max_gen=1, mode='rand'
        )

        return candidate

    def _strategy_local(self, X, Y, dim):
        """
        Strategy 2: Local search with DE/best/1 (10 generations) on an RBF model
        built from the top tau = min(dim+25, N) samples, searched inside their
        bounding box.
        """
        N = len(X)
        tau = min(dim + 25, N)

        Y_flat = Y.flatten()
        idx = np.argsort(Y_flat, kind='stable')[:tau]
        X_local = X[idx]
        Y_local = Y[idx]

        lb_local = np.min(X_local, axis=0)
        ub_local = np.max(X_local, axis=0)

        model = newrbe_surrogate(X_local, Y_local)

        candidate = self._de_search(
            model, X, Y,
            lb=lb_local, ub=ub_local,
            dim=dim, popsize=50, max_gen=10, mode='best'
        )

        return candidate

    def _strategy_interior_point(self, X, Y, dim):
        """
        Strategy 3: Interior-point local optimization on an RBF model built
        from the min(N, 5*dim) samples nearest to the current best, started
        from the best solution.
        """
        N = len(X)
        m = min(N, 5 * dim)

        Y_flat = Y.flatten()
        idx_min = np.argmin(Y_flat)

        # Select m nearest neighbors to the best point
        dist = cdist(X, X[idx_min:idx_min + 1]).flatten()
        idx = np.argsort(dist, kind='stable')[:m]
        X_trs = X[idx]
        Y_trs = Y[idx]

        model = newrbe_surrogate(X_trs, Y_trs)

        lb_trs = np.min(X_trs, axis=0)
        ub_trs = np.max(X_trs, axis=0)

        x0 = X[idx_min]
        bounds = list(zip(lb_trs, ub_trs))

        def obj_func(x):
            return float(model(x.reshape(1, -1))[0])

        try:
            result = minimize(obj_func, x0, method='trust-constr', bounds=bounds,
                              options={'maxiter': 20, 'disp': False})
            candidate = result.x.reshape(1, -1)
        except Exception:
            candidate = x0.reshape(1, -1)

        candidate = np.clip(candidate, 0.0, 1.0)
        return candidate

    def _de_search(self, surrogate_func, X_full, Y_full, lb, ub, dim,
                   popsize=50, max_gen=10, mode='rand'):
        """
        DE search on a surrogate with elite initialization from the full database.

        The initial parents carry their raw database objective values; offspring
        are scored on the surrogate and compete 1-to-1 (ties go to offspring).
        Parents from the full database may lie outside [lb, ub]; they are
        normalized without clipping, and only offspring are confined to the box.
        """
        Y_flat = Y_full.flatten()
        N = len(Y_flat)

        # Elite initialization: top popsize with raw objective values (sorted ascending)
        if N >= popsize:
            idx = np.argsort(Y_flat, kind='stable')[:popsize]
            pop = X_full[idx].copy()
            pop_objs = Y_flat[idx].copy()
        else:
            extra = lb + (ub - lb) * np.random.rand(popsize - N, dim)
            pop = np.vstack([X_full.copy(), extra])
            pop_objs = np.concatenate([Y_flat, np.asarray(surrogate_func(extra)).flatten()])

        range_vec = np.maximum(ub - lb, 1e-12)

        for gen in range(max_gen):
            # Normalize parents without clipping (offspring are clipped to [0,1])
            pop_norm = (pop - lb) / range_vec

            if mode == 'rand':
                off_norm = self._de_r1(pop_norm)
            else:
                off_norm = self._de_best1(pop_norm, pop_objs)

            offspring = lb + off_norm * range_vec

            off_objs = np.asarray(surrogate_func(offspring)).flatten()

            # 1-to-1 comparison selection (ties go to offspring)
            replace = off_objs <= pop_objs
            pop[replace] = offspring[replace]
            pop_objs[replace] = off_objs[replace]

        best_idx = np.argmin(pop_objs)
        return pop[best_idx:best_idx + 1]

    def _de_r1(self, parents, F=0.5, CR=0.8):
        """
        MATLAB 'de-r' offspring generation: the base vector is the first row of
        the (elite-sorted) population, i.e. the best individual at generation 1.
        """
        popsize, dim = parents.shape
        base = parents[0]

        offspring = parents.copy()
        for i in range(popsize):
            r1 = self._rand_index(popsize, {i})
            r2 = self._rand_index(popsize, {i, r1})
            r3 = self._rand_index(popsize, {i, r1, r2})
            mutant = base + F * (parents[r2] - parents[r3])
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) <= CR) | (np.arange(dim) == j_rand)
            offspring[i, mask] = mutant[mask]

        return np.clip(offspring, 0, 1)

    def _de_best1(self, parents, objs, F=0.5, CR=0.8):
        """DE/best/1/bin offspring generation in [0,1] normalized space."""
        popsize, dim = parents.shape
        best_idx = np.argmin(objs)
        best = parents[best_idx]

        offspring = parents.copy()
        for i in range(popsize):
            r1 = self._rand_index(popsize, {i})
            r2 = self._rand_index(popsize, {i, r1})
            mutant = best + F * (parents[r1] - parents[r2])
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) <= CR) | (np.arange(dim) == j_rand)
            offspring[i, mask] = mutant[mask]

        return np.clip(offspring, 0, 1)

    @staticmethod
    def _rand_index(n, exclude):
        """Random index in [0, n) avoiding the excluded set."""
        r = np.random.randint(n)
        while r in exclude:
            r = np.random.randint(n)
        return r

    def _ensure_uniqueness(self, candidate, X, dim, epsilon=5e-3, n_scales=50, max_trials=1000):
        """Perturb the candidate until it is not too close to existing samples."""
        scales = np.linspace(0.1, 1.0, n_scales)
        c = 0
        while cdist(candidate, X, metric='chebyshev').min() < epsilon and c < max_trials:
            perturbation = scales[c % n_scales] * (np.random.rand(1, dim) - 0.5)
            candidate = np.clip(candidate + perturbation, 0.0, 1.0)
            c += 1
        return candidate

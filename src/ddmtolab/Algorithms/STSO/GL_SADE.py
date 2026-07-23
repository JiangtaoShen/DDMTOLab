"""
Global-Local Surrogate-Assisted Differential Evolution (GL-SADE)

This module implements GL-SADE for expensive single-objective optimization problems.

References
----------
    [1] Wang, Weizhong, Hai-Lin Liu, and Kay Chen Tan. "A surrogate-assisted differential evolution algorithm for high-dimensional expensive optimization problems." IEEE Transactions on Cybernetics 53.4 (2022): 2685-2697.

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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class GL_SADE:
    """
    Global-Local Surrogate-Assisted Differential Evolution for expensive optimization problems.

    This algorithm adaptively switches between:
    1. Global search: RBF model with plain acquisition (DE/best/1, 50 generations)
    2. Local search: GPR model on recent samples with LCB-decay acquisition (prescreening)
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
                 save_path='./Data', name='GL-SADE', disable_tqdm=True):
        """
        Initialize GL-SADE algorithm.

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
            Name for the experiment (default: 'GL-SADE')
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
        Execute the GL-SADE algorithm.

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
                X = current_decs[i]
                Y = current_objs[i]

                # Local search if the last evaluation improved on all previous ones
                if len(Y) > 1 and Y[-1, 0] < np.min(Y[:-1, 0]):
                    candidate_np = self._local_search(X, Y, dim, max_nfes_per_task[i])
                else:
                    candidate_np = self._global_search(X, Y, dim)

                # Ensure uniqueness: avoid duplicate sampling
                candidate_np = self._ensure_uniqueness(candidate_np, X, dim)

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

    # Build GPR surrogate model
    def _build_gpr_model(self, X, Y):

        Y = Y.flatten()

        # Define kernel: Constant * RBF + WhiteKernel (noise)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-10, 1e-1))

        # Fit GPR with normalization
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        gpr.fit(X, Y)

        # Return a function that returns (mean, std)
        def gpr_model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            mean, std = gpr.predict(x, return_std=True)
            return mean.reshape(-1, 1), std.reshape(-1, 1)

        return gpr_model

    # Global search using RBF model with plain acquisition
    def _global_search(self, X, Y, dim):

        rbf_model = newrbe_surrogate(X, Y)

        def acquisition_func(x):
            return np.asarray(rbf_model(x)).reshape(-1, 1)

        # DE/best/1 with elite init from database, 50 generations
        candidate = self._de_best1_search(acquisition_func, X, Y, dim,
                                          popsize=50, max_gen=50)
        return candidate

    # Local search using GPR model with LCB-decay acquisition
    def _local_search(self, X, Y, dim, max_nfes):

        # Use only the most recent min(n, 2*dim) samples
        n_local = min(len(X), 2 * dim)
        X_local = X[-n_local:]
        Y_local = Y[-n_local:]

        # Build GPR model
        gpr_model = self._build_gpr_model(X_local, Y_local)

        # LCB weight with decay (MATLAB hardcodes the 500 denominator = its FEsMax)
        n_used = len(X)
        w = 2.0 - 2.0 / (1.0 + np.exp(5.0 - 20.0 * n_used / max_nfes))

        # Acquisition function (LCB: mean - w * std)
        def acquisition_func(x):
            mean, std = gpr_model(x)
            return mean - w * std

        # DE/best/1 with elite init, 1 generation = prescreening
        candidate = self._de_best1_search(acquisition_func, X, Y, dim,
                                          popsize=50, max_gen=1)
        return candidate

    def _de_best1_search(self, acquisition_func, X_db, Y_db, dim,
                         popsize=50, max_gen=50, F=0.5, CR=0.8):
        """
        DE/best/1/bin with elite initialization and 1-to-1 comparison selection.

        The initial parents carry their raw database objective values (MATLAB
        pop_ini 'elite'); only offspring are scored with the acquisition function.
        """
        N = len(X_db)
        Y_flat = Y_db.flatten()

        # Elite initialization: top-popsize solutions with raw objective values
        if N >= popsize:
            idx = np.argsort(Y_flat, kind='stable')[:popsize]
            pop = X_db[idx].copy()
            pop_acq = Y_flat[idx].copy()
        else:
            extra = np.random.rand(popsize - N, dim)
            pop = np.vstack([X_db.copy(), extra])
            pop_acq = np.concatenate([Y_flat, acquisition_func(extra).flatten()])

        n_pop = len(pop)
        for gen in range(max_gen):
            # Current best as base vector
            best_idx = np.argmin(pop_acq)
            x_best = pop[best_idx]

            offspring = np.empty_like(pop)
            for j in range(n_pop):
                # Mutation: DE/best/1
                candidates = [k for k in range(n_pop) if k != j]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = x_best + F * (pop[r1] - pop[r2])

                # Binomial crossover with forced index
                trial = pop[j].copy()
                j_rand = np.random.randint(dim)
                for d in range(dim):
                    if np.random.rand() <= CR or d == j_rand:
                        trial[d] = mutant[d]
                offspring[j] = np.clip(trial, 0.0, 1.0)

            # 1-to-1 comparison selection (ties go to offspring)
            offspring_acq = acquisition_func(offspring).flatten()
            replace = offspring_acq <= pop_acq
            pop[replace] = offspring[replace]
            pop_acq[replace] = offspring_acq[replace]

        # Return best solution
        best_idx = np.argmin(pop_acq)
        return pop[best_idx:best_idx + 1].copy()

    # Ensure candidate is not too close to existing samples
    def _ensure_uniqueness(self, candidate, X, dim, epsilon=5e-3, n_scales=50, max_trials=1000):

        scales = np.linspace(0.1, 1.0, n_scales)
        c = 0
        while cdist(candidate, X, metric='chebyshev').min() < epsilon and c < max_trials:
            perturbation = scales[c % n_scales] * (np.random.rand(1, dim) - 0.5)
            candidate = np.clip(candidate + perturbation, 0.0, 1.0)
            c += 1
        return candidate

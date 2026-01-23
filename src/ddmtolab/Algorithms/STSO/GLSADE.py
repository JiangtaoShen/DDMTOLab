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
Date: 2025.01.13
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.STSO.DE import DE
from ddmtolab.Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class GLSADE:
    """
    Global-Local Surrogate-Assisted Differential Evolution for expensive optimization problems.

    This algorithm adaptively switches between:
    1. Global search: RBF model with plain acquisition
    2. Local search: GPR model with LCB-decay acquisition
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
                 save_path='./TestData', name='GLSADE_test', disable_tqdm=True):
        """
        Initialize GL-SADE algorithm.

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
            Name for the experiment (default: 'GLSADE_test')
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
                X = current_decs[i]
                Y = current_objs[i]

                # Determine search state based on improvement
                if len(Y) > 1:
                    if Y[-1, 0] < np.min(Y[:-1, 0]):
                        state = 1  # Local search (last point improved)
                    else:
                        state = 0  # Global search (no improvement)
                else:
                    state = 0  # Default to global if only initial samples

                # Execute search based on state
                if state == 0:
                    candidate_np = self._global_search(X, Y, dim, nfes_per_task[i])
                else:
                    candidate_np = self._local_search(X, Y, dim, nfes_per_task[i])

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
            rbf_interpolator = RBFInterpolator(X, Y, kernel='gaussian', epsilon=1.0 / spread)
        except:
            # Fallback
            rbf_interpolator = RBFInterpolator(X, Y, kernel='thin_plate_spline')

        # Return a function that only returns predictions
        def rbf_model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            pred = rbf_interpolator(x)
            return pred.reshape(-1, 1)

        return rbf_model

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
    def _global_search(self, X, Y, dim, nfes_used):

        # Build RBF model
        rbf_model = self._build_rbf_model(X, Y)

        # Define acquisition function
        def acquisition_func(x):
            return rbf_model(x)

        # Optimize acquisition function using DE
        candidate = self._optimize_acquisition(
            acquisition_func,
            dim,
            popsize=50,
            max_gen=50,
            mode='plain'
        )

        return candidate

    # Local search using GPR model with LCB-decay acquisition
    def _local_search(self, X, Y, dim, nfes_used):

        # Use only recent samples (min(n, 2*dim))
        n_local = min(len(X), 2 * dim)
        X_local = X[-n_local:]
        Y_local = Y[-n_local:]

        # Build GPR model
        gpr_model = self._build_gpr_model(X_local, Y_local)

        # Calculate LCB weight with decay
        w = 2.0 - 2.0 / (1.0 + np.exp(5.0 - 20.0 * nfes_used / 500.0))

        # Define acquisition function (LCB: mean - w * std)
        def acquisition_func(x):
            mean, std = gpr_model(x)
            return mean - w * std

        # Optimize acquisition function using DE (only 1 generation for prescreening)
        candidate = self._optimize_acquisition(
            acquisition_func,
            dim,
            popsize=50,
            max_gen=1,
            mode='lcb'
        )

        return candidate

    # Optimize acquisition function using DE
    def _optimize_acquisition(self, acquisition_func, dim, popsize=50, max_gen=50, mode='plain'):

        # Create surrogate problem wrapper
        def surrogate_objective(x):
            return acquisition_func(x)

        surrogate_problem = MTOP()
        surrogate_problem.add_task(objective_func=surrogate_objective, dim=dim)

        # Run DE to optimize acquisition function
        max_nfes = popsize * max_gen
        de = DE(surrogate_problem, n=popsize, max_nfes=max_nfes, disable_tqdm=True)
        results = de.optimize()

        # Get best solution
        best_solution = results.best_decs[0].reshape(1, -1)

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
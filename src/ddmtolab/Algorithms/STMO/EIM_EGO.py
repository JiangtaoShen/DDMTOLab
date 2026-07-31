"""
Expected Improvement Matrix based Efficient Global Optimization (EIM-EGO)

This module implements EIM-EGO for computationally expensive multi-objective optimization.
It builds Kriging (Gaussian Process) models for each objective and uses the Expected Improvement
Matrix (EIM) criterion to select one promising candidate per iteration via GA optimization.

Three EIM infill criteria are supported:
    1. Euclidean distance-based EIM criterion (default)
    2. Maximin distance-based EIM criterion
    3. Hypervolume-based EIM criterion

References
----------
    [1] D. Zhan, Y. Cheng, and J. Liu. Expected improvement matrix-based infill criteria for expensive multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2017, 21(6): 956-975.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.stats import norm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
import warnings

warnings.filterwarnings("ignore")


class EIM_EGO:
    """
    Expected Improvement Matrix based Efficient Global Optimization
    for expensive multi-objective optimization.
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

    def __init__(self, problem, n_initial=None, max_nfes=None, eim_type=1,
                 ga_pop_size=100, ga_generations=100,
                 save_data=True, save_path='./Data', name='EIM-EGO', disable_tqdm=True):
        """
        Initialize EIM-EGO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        eim_type : int, optional
            EIM infill criterion type (default: 1)
            1 = Euclidean distance-based, 2 = Maximin distance-based, 3 = Hypervolume-based
        ga_pop_size : int, optional
            Population size for internal GA optimizer (default: 100)
        ga_generations : int, optional
            Number of generations for internal GA optimizer (default: 100)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EIM-EGO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.eim_type = eim_type
        self.ga_pop_size = ga_pop_size
        self.ga_generations = ga_generations
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EIM-EGO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        n_initial_per_task = par_list(
            self.n_initial if self.n_initial is not None else [11 * d - 1 for d in dims], nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                D = dims[i]

                # Scale objectives to [0, 1]
                obj_min = objs[i].min(axis=0)
                obj_max = objs[i].max(axis=0)
                obj_range = obj_max - obj_min
                obj_range[obj_range < 1e-10] = 1.0
                pop_obj_scaled = (objs[i] - obj_min) / obj_range

                # Build GP models for each objective on scaled objectives
                try:
                    models = mo_gp_build(decs[i], pop_obj_scaled)
                except Exception:
                    # If GP fitting fails, add a random point
                    new_dec = np.random.rand(1, D)
                    new_obj, _ = evaluation_single(problem, new_dec, i)
                    decs[i] = np.vstack([decs[i], new_dec])
                    objs[i] = np.vstack([objs[i], new_obj])
                    nfes_per_task[i] += 1
                    pbar.update(1)
                    continue

                # Select one candidate using EIM criterion via GA
                best_dec = _infill_sampling_eim(
                    models, pop_obj_scaled, D, M, self.eim_type,
                    self.ga_pop_size, self.ga_generations
                )

                # Remove duplicates
                new_dec = remove_duplicates(best_dec.reshape(1, -1), decs[i])
                if new_dec.shape[0] == 0:
                    # If duplicate, perturb slightly
                    new_dec = np.clip(best_dec + 1e-4 * np.random.randn(D), 0, 1).reshape(1, -1)

                # Evaluate the candidate
                new_obj, _ = evaluation_single(problem, new_dec, i)
                decs[i] = np.vstack([decs[i], new_dec])
                objs[i] = np.vstack([objs[i], new_obj])
                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [decs[i].copy() for i in range(nt)]
        db_objs = [objs[i].copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# EIM Infill Criterion
# =============================================================================

def _infill_sampling_eim(models, pop_obj_scaled, D, M, eim_type, ga_pop_size, ga_generations):
    """Select one candidate by maximizing the EIM criterion using GA.

    Parameters
    ----------
    models : list
        Trained GP models, one per objective
    pop_obj_scaled : np.ndarray
        Scaled objective values, shape (N, M)
    D : int
        Number of decision variables
    M : int
        Number of objectives
    eim_type : int
        EIM criterion type (1=Euclidean, 2=Maximin, 3=Hypervolume)
    ga_pop_size : int
        GA population size
    ga_generations : int
        Number of GA generations

    Returns
    -------
    best : np.ndarray
        Best candidate decision vector, shape (D,)
    """
    # Get non-dominated front from scaled objectives (NDSort(PopObjScaled,1))
    front_no, _ = nd_sort(pop_obj_scaled, 1)
    f = pop_obj_scaled[front_no == 1]  # (p, M)
    p = f.shape[0]

    # Track global best
    best = None
    eim_max = np.inf  # We negate EIM, so inf means worst

    # Initialize GA population with LHS
    offspring = initialization_single(D, ga_pop_size, method='lhs')

    for gen in range(ga_generations):
        # Predict mean and MSE from GP models
        pred_objs, pred_mse = mo_gp_predict(models, offspring, mse=True)
        u = pred_objs  # (ga_pop_size, M)
        s = np.sqrt(np.maximum(0, pred_mse))  # (ga_pop_size, M)

        # Compute EIM criterion
        eim = _compute_eim(f, u, s, M, p, eim_type)  # (ga_pop_size,), negated

        # Update global best
        best_idx = np.argmin(eim)
        if eim[best_idx] < eim_max:
            best = offspring[best_idx].copy()
            eim_max = eim[best_idx]

        # GA selection and evolution
        sorted_idx = np.argsort(eim)
        half = int(np.ceil(ga_pop_size / 2))
        parents = offspring[sorted_idx[:half]]
        parent_eim = eim[sorted_idx[:half]]

        # First half: tournament selection + SBX crossover + polynomial mutation
        tournament_idx = _tournament_selection(parent_eim, parents.shape[0])
        offspring1 = ga_generation(parents[tournament_idx], muc=20, mum=20)

        # Second half: mutation only on parents
        offspring2 = np.array([mutation(parents[j], mu=20) for j in range(parents.shape[0])])

        # Combine and clip
        offspring = np.vstack([offspring1, offspring2])[:ga_pop_size]
        offspring = np.clip(offspring, 0, 1)

    return best


def _compute_eim(f, u, s, M, p, eim_type):
    """Compute the EIM criterion for all candidates.

    Parameters
    ----------
    f : np.ndarray
        Non-dominated front (scaled), shape (p, M)
    u : np.ndarray
        Predicted means, shape (N, M)
    s : np.ndarray
        Predicted stds, shape (N, M)
    M : int
        Number of objectives
    p : int
        Number of non-dominated solutions
    eim_type : int
        EIM type (1=Euclidean, 2=Maximin, 3=Hypervolume)

    Returns
    -------
    eim : np.ndarray
        Negated EIM values, shape (N,), lower is better
    """
    N = u.shape[0]

    # Reshape for broadcasting: f (p, M, 1), u (1, M, N), s (1, M, N)
    f_3d = f[:, :, np.newaxis]  # (p, M, N_broadcast=1)
    u_3d = u.T[np.newaxis, :, :]  # (1, M, N)
    s_3d = s.T[np.newaxis, :, :]  # (1, M, N)

    # Avoid division by zero
    s_safe = np.maximum(s_3d, 1e-10)

    # EI matrix: EI = (f - u) * Phi((f-u)/s) + s * phi((f-u)/s)
    diff = f_3d - u_3d  # (p, M, N)
    z = diff / s_safe  # (p, M, N)
    ei_matrix = diff * norm.cdf(z) + s_3d * norm.pdf(z)  # (p, M, N)

    if eim_type == 1:
        # Euclidean distance-based: min over p of sqrt(sum over M of EI^2)
        dist = np.sqrt(np.sum(ei_matrix ** 2, axis=1))  # (p, N)
        eim = -np.min(dist, axis=0)  # (N,)
    elif eim_type == 2:
        # Maximin distance-based: min over p of max over M of EI
        maxmin = np.max(ei_matrix, axis=1)  # (p, N)
        eim = -np.min(maxmin, axis=0)  # (N,)
    elif eim_type == 3:
        # Hypervolume-based: min over p of (prod(1.1-f+EI) - prod(1.1-f))
        ref = 1.1 * np.ones(M)
        prod_with_ei = np.prod(ref[np.newaxis, :, np.newaxis] - f_3d + ei_matrix, axis=1)  # (p, N)
        prod_without_ei = np.prod(ref[np.newaxis, :] - f, axis=1)  # (p,)
        hv_contrib = prod_with_ei - prod_without_ei[:, np.newaxis]  # (p, N)
        eim = -np.min(hv_contrib, axis=0)  # (N,)
    else:
        raise ValueError(f"Unknown EIM type: {eim_type}. Must be 1, 2, or 3.")

    return eim


def _tournament_selection(fitness, n_select, n_tournament=2):
    """Binary tournament selection (lower fitness is better)."""
    N = len(fitness)
    selected = np.zeros(n_select, dtype=int)
    for k in range(n_select):
        candidates = np.random.randint(0, N, size=n_tournament)
        selected[k] = candidates[np.argmin(fitness[candidates])]
    return selected


def initialization_single(D, n, method='lhs'):
    """Initialize n solutions in [0,1]^D for a single task."""
    if method == 'lhs':
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=D)
        return sampler.random(n=n)
    else:
        return np.random.rand(n, D)

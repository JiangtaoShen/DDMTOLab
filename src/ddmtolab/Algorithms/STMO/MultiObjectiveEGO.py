"""
Multi-objective Efficient Global Optimization (MultiObjectiveEGO)

This module implements MultiObjectiveEGO for computationally expensive multi-objective optimization.
It uses reference direction-based Augmented Achievement Scalarizing Function (AASF) to decompose
the multi-objective problem into scalar subproblems, builds a single GP model per infill, and
maximizes Standard Expected Improvement to select new evaluation points.

References
----------
    [1] R. Hussein and K. Deb. A generative Kriging surrogate model for constrained and unconstrained multi-objective optimization. Proceedings of the Genetic and Evolutionary Computation Conference, 2016, 573-580.

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
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
import warnings

warnings.filterwarnings("ignore")


class MultiObjectiveEGO:
    """
    Multi-objective Efficient Global Optimization for expensive
    multi-objective optimization using reference direction-based
    scalarization and Expected Improvement.
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

    def __init__(self, problem, n_initial=None, max_nfes=None, alpha=0.7, num_k=5, H=21,
                 rho=1e-3, save_data=True, save_path='./Data', name='MultiObjectiveEGO',
                 disable_tqdm=True):
        """
        Initialize MultiObjectiveEGO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        alpha : float, optional
            Portion of samples for Kriging construction (default: 0.7)
        num_k : int, optional
            Number of infill points per reference direction (default: 5)
        H : int, optional
            Number of reference directions (default: 21)
        rho : float, optional
            Parameter for AASF scalarization (default: 1e-3)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MultiObjectiveEGO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.alpha = alpha
        self.num_k = num_k
        self.H = H
        self.rho = rho
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MultiObjectiveEGO algorithm.

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

        # Step 2: generate the reference direction set (once, as in MATLAB)
        R_per_task, N_R_per_task = [], []
        for i in range(nt):
            R_i, N_R_i = uniform_point(self.H, n_objs[i])
            R_i = R_i / np.sqrt(np.sum(R_i ** 2, axis=1, keepdims=True))
            R_per_task.append(R_i)
            N_R_per_task.append(N_R_i)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                D = dims[i]
                R, N_R = R_per_task[i], N_R_per_task[i]

                # Iterate over each reference direction
                for r_idx in range(N_R):
                    if nfes_per_task[i] >= max_nfes_per_task[i]:
                        break

                    for j in range(self.num_k):
                        if nfes_per_task[i] >= max_nfes_per_task[i]:
                            break

                        N_pop = decs[i].shape[0]
                        r_vec = R[r_idx]

                        # Points_Selector: select alpha*N_Pop closest points to this direction
                        norm_p = np.sqrt(np.sum(objs[i] ** 2, axis=1))
                        norm_p[norm_p < 1e-10] = 1e-10
                        cosine_p = np.sum(objs[i] * r_vec, axis=1) / norm_p
                        cosine_p = np.clip(cosine_p, -1, 1)
                        dist_b = norm_p * np.sqrt(np.maximum(1 - cosine_p ** 2, 0))

                        sorted_idx = np.argsort(dist_b)
                        N_sub = min(int(np.ceil(self.alpha * N_pop)), N_pop)
                        sub_idx = sorted_idx[:N_sub]

                        pop_dec = decs[i][sub_idx]
                        pop_obj = objs[i][sub_idx]

                        # Scale objectives to [0, 1]
                        obj_min = pop_obj.min(axis=0)
                        obj_max = pop_obj.max(axis=0)
                        obj_range = obj_max - obj_min
                        obj_range[obj_range < 1e-10] = 1.0
                        pop_obj_scaled = (pop_obj - obj_min) / obj_range

                        # Compute AASF scalarized value
                        # S = max(f_scaled / w) + rho * sum(f_scaled / w)
                        r_safe = np.maximum(r_vec, 1e-10)
                        weighted = pop_obj_scaled / r_safe
                        pop_smetric = np.max(weighted, axis=1) + self.rho * np.sum(weighted, axis=1)

                        # Build single GP model on scalarized values
                        try:
                            kriging_model = gp_build(pop_dec, pop_smetric.reshape(-1, 1))
                        except Exception:
                            # Guarantee progress: fall back to the space-filling
                            # infill criterion used by MATLAB for degenerate cases
                            kriging_model = None

                        if kriging_model is None:
                            best_x = _rga_max_distance(decs[i], D)
                        else:
                            f_min = pop_smetric.min()
                            # Maximize EI using GA
                            best_x = _rga_ei(kriging_model, f_min, D)

                        # If too close to existing points, maximize distance
                        if np.min(cdist(best_x.reshape(1, -1), decs[i])) < 1e-8:
                            best_x = _rga_max_distance(decs[i], D)

                        # Evaluate
                        new_dec = best_x.reshape(1, -1)
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
# GA operators (local ports of PlatEMO's OperatorGA / UniformPoint 'Latin')
# =============================================================================

def _lhs(n, d):
    """Latin hypercube sample in [0, 1]^d (UniformPoint(N,D,'Latin'))."""
    return qmc.LatinHypercube(d=d).random(n=n)


def _operator_ga(parent, proC=1.0, disC=20.0, proM=1.0, disM=20.0):
    """PlatEMO ``OperatorGA``/``GAreal`` for real variables in [0, 1]^D.

    A local port is used instead of the shared ``ga_generation`` helper because
    ``rGA.m`` relies on the explicit parameter tuple ``{0.9,2,1/D,20}`` for the
    second half of the offspring (per-variable mutation probability ``proM/D``)
    and on the un-shuffled first-half/second-half pairing of the sorted parents.

    Parameters
    ----------
    parent : np.ndarray
        Parent population, shape (N, D)
    proC : float, optional
        Crossover probability (default: 1.0)
    disC : float, optional
        Distribution index of simulated binary crossover (default: 20.0)
    proM : float, optional
        Expected number of mutated variables; the per-variable mutation
        probability is ``proM / D`` (default: 1.0)
    disM : float, optional
        Distribution index of polynomial mutation (default: 20.0)

    Returns
    -------
    offspring : np.ndarray
        Offspring population, shape (2*floor(N/2), D)
    """
    n_half = parent.shape[0] // 2
    parent1 = parent[:n_half]
    parent2 = parent[n_half:2 * n_half]
    N, D = parent1.shape

    # Simulated binary crossover
    mu = np.random.rand(N, D)
    beta = np.zeros((N, D))
    lo = mu <= 0.5
    beta[lo] = (2 * mu[lo]) ** (1 / (disC + 1))
    beta[~lo] = (2 - 2 * mu[~lo]) ** (-1 / (disC + 1))
    beta = beta * (-1.0) ** np.random.randint(0, 2, size=(N, D))
    beta[np.random.rand(N, D) < 0.5] = 1
    beta[np.tile(np.random.rand(N, 1) > proC, (1, D))] = 1
    offspring = np.vstack([
        (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
        (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    ])

    # Polynomial mutation (bounds are [0, 1] in DDMTOLab's unified space)
    site = np.random.rand(2 * N, D) < proM / D
    mu = np.random.rand(2 * N, D)
    offspring = np.clip(offspring, 0, 1)

    temp = site & (mu <= 0.5)
    offspring[temp] = offspring[temp] + (
        (2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - offspring[temp]) ** (disM + 1))
        ** (1 / (disM + 1)) - 1)
    temp = site & (mu > 0.5)
    offspring[temp] = offspring[temp] + (
        1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * offspring[temp] ** (disM + 1))
        ** (1 / (disM + 1)))

    return offspring


# =============================================================================
# GA-based Infill Optimization
# =============================================================================

def _rga_ei(kriging_model, f_min, D, ga_pop_size=None, ga_generations=100):
    """Maximize Expected Improvement using real-coded GA.

    Parameters
    ----------
    kriging_model : SingleTaskGP
        Trained GP model for the scalarized objective
    f_min : float
        Minimum observed scalarized value (best so far)
    D : int
        Number of decision variables
    ga_pop_size : int, optional
        GA population size (default: 10*D)
    ga_generations : int, optional
        Number of GA generations (default: 100)

    Returns
    -------
    best : np.ndarray, shape (D,)
        Best candidate decision vector
    """
    if ga_pop_size is None:
        ga_pop_size = 10 * D

    best = None
    obj_max = np.inf

    # The first GA generation, randomly generated (LHS)
    offspring = _lhs(ga_pop_size, D)

    for gen in range(ga_generations):
        # Predict and compute EI
        pred, std = gp_predict(kriging_model, offspring)
        pred = pred.flatten()
        std = std.flatten()
        s = np.maximum(std, 1e-10)

        # Standard EI: (f_min - y) * Phi((f_min-y)/s) + s * phi((f_min-y)/s)
        diff = f_min - pred
        z = diff / s
        ei = diff * norm.cdf(z) + s * norm.pdf(z)
        neg_ei = -ei  # the genetic algorithm tries to minimize the objective

        # Track best
        sorted_idx = np.argsort(neg_ei)
        if neg_ei[sorted_idx[0]] < obj_max:
            best = offspring[sorted_idx[0]].copy()
            obj_max = neg_ei[sorted_idx[0]]

        # Select top half as parents
        half = int(np.ceil(ga_pop_size / 2))
        parents = offspring[sorted_idx[:half]]
        parent_fitness = neg_ei[sorted_idx[:half]]

        # OperatorGA(Problem, Parent(TournamentSelection(2,...),:)) -> {1,20,1,20}
        t_idx = _tournament_selection(parent_fitness, parents.shape[0])
        offspring1 = _operator_ga(parents[t_idx], 1.0, 20.0, 1.0, 20.0)

        # OperatorGA(Problem, Parent, {0.9,2,1/D,20})
        offspring2 = _operator_ga(parents, 0.9, 2.0, 1.0 / D, 20.0)

        offspring = np.vstack([offspring1, offspring2])

    return best


def _rga_max_distance(existing_decs, D, ga_pop_size=None, ga_generations=100):
    """Maximize minimum distance to existing points using GA.

    Parameters
    ----------
    existing_decs : np.ndarray
        Existing evaluated decision variables, shape (N, D)
    D : int
        Number of decision variables

    Returns
    -------
    best : np.ndarray, shape (D,)
        Candidate maximizing distance to existing points
    """
    if ga_pop_size is None:
        ga_pop_size = 10 * D

    best = None
    obj_max = np.inf

    offspring = _lhs(ga_pop_size, D)

    for gen in range(ga_generations):
        # Infill_Maximal_Distance: -min(pdist2(x, sample_x), [], 2)
        dists = cdist(offspring, existing_decs)
        neg_min_dist = -np.min(dists, axis=1)

        sorted_idx = np.argsort(neg_min_dist)
        if neg_min_dist[sorted_idx[0]] < obj_max:
            best = offspring[sorted_idx[0]].copy()
            obj_max = neg_min_dist[sorted_idx[0]]

        half = int(np.ceil(ga_pop_size / 2))
        parents = offspring[sorted_idx[:half]]
        parent_fitness = neg_min_dist[sorted_idx[:half]]

        t_idx = _tournament_selection(parent_fitness, parents.shape[0])
        offspring1 = _operator_ga(parents[t_idx], 1.0, 20.0, 1.0, 20.0)
        offspring2 = _operator_ga(parents, 0.9, 2.0, 1.0 / D, 20.0)

        offspring = np.vstack([offspring1, offspring2])

    return best


def _tournament_selection(fitness, n_select, n_tournament=2):
    """Binary tournament selection (lower fitness is better)."""
    N = len(fitness)
    selected = np.zeros(n_select, dtype=int)
    for k in range(n_select):
        candidates = np.random.randint(0, N, size=n_tournament)
        selected[k] = candidates[np.argmin(fitness[candidates])]
    return selected

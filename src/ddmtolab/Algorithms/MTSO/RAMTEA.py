"""
Radial Basis Functions-Assisted MTEA (RAMTEA)

This module implements RAMTEA for expensive multi-task optimization with
surrogate-assisted adaptive knowledge transfer. Each task keeps an RBF surrogate
built from its evaluated database; a GA search seeded from that database refines
the surrogate optimum, and promising solutions are transferred between tasks with
a probability equal to their rank-correlation similarity.

References
----------
    [1] J. Shen, et al., "Surrogate-assisted adaptive knowledge transfer for
        expensive multitasking optimization," 2024 IEEE Congress on Evolutionary
        Computation (CEC). IEEE, 2024.

Notes
-----
Reconciled with the reference MATLAB implementation (MTO-Platform) and generalized
to K >= 3 tasks (v2.0):

- Task similarity is the Spearman rank correlation between the objective values a
  shared set of points attains on each task, clamped to >= 0 (the reference's
  ``S = max(S, 0)``). The previous version used Pearson correlation on the raw
  objectives and transferred on ``|corr|``, which incorrectly transferred between
  negatively correlated tasks.
- The surrogate search is seeded from each task's evaluated database and runs
  ``w_max`` GA generations, keeping the ``n_in`` best by RBF prediction each
  generation (the reference procedure), instead of a GA restarted from random
  points which can chase unreliable RBF extrapolations far from the data.
- Knowledge transfer generalizes the reference's coupled 2-task transfer to any
  number of tasks: task r always evaluates its own surrogate optimum and, for
  each other task s, additionally evaluates task s's optimum with probability
  ``S(r, s)``. For two tasks the per-direction transfer probability is identical
  to the reference.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.07.18
Version: 2.0
"""
import time
import warnings

import numpy as np
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator

from ddmtolab.Methods.Algo_Methods.sim_evaluation import sim_calculate
from ddmtolab.Methods.Algo_Methods.algo_utils import *

warnings.filterwarnings("ignore")


class RAMTEA:
    """
    Radial Basis Functions-Assisted Multi-Task Evolutionary Algorithm for expensive optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n_in=50, w_max=50,
                 muc=2, mum=5, save_data=True, save_path='./Data', name='RAMTEA',
                 disable_tqdm=True):
        """
        Initialize RAMTEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50; the reference uses
            100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        n_in : int, optional
            Number of solutions retained each generation of the surrogate search
            (default: 50)
        w_max : int, optional
            Number of generations of the surrogate search (default: 50)
        muc : float, optional
            Distribution index for SBX crossover in the surrogate search
            (default: 2)
        mum : float, optional
            Distribution index for polynomial mutation in the surrogate search
            (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'RAMTEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.n_in = n_in
        self.w_max = w_max
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the RAMTEA algorithm.

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
        nfes_per_task = n_initial_per_task.copy()

        # Shared initial design: the same points are evaluated on every task so
        # that the rank-correlation similarity is well defined.
        decs = initialization(problem, self.n_initial, method='lhs', the_same=True)
        objs, _ = evaluation(problem, decs)

        # Spearman rank-correlation similarity, clamped to >= 0 (reference S)
        sim = np.maximum(sim_calculate(objs, method='spearman'), 0.0)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Build an RBF surrogate per task and refine its optimum with a
            # database-seeded GA search.
            best_solutions = [None] * nt
            for i in active_tasks:
                best_solutions[i] = self._rbf_search(decs[i], objs[i], dims[i])

            # Similarity-based knowledge transfer + real evaluation per task
            for i in active_tasks:
                candidates = ramtea_knowledge_transfer(
                    task_idx=i, active_tasks=active_tasks, best_solutions=best_solutions,
                    dims=dims, sim=sim, nfes_per_task=nfes_per_task,
                    max_nfes_per_task=max_nfes_per_task)
                if candidates is None:
                    continue

                new_objs, _ = evaluation_single(problem, candidates, i)
                decs[i], objs[i] = vstack_groups((decs[i], candidates), (objs[i], new_objs))

                nfes_per_task[i] += len(candidates)
                pbar.update(len(candidates))

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results

    def _rbf_search(self, decs_t, objs_t, dim_t):
        """
        Refine the RBF optimum with a GA search seeded from the task's database.

        Builds an RBF surrogate from the task's evaluated data, seeds the search
        population with that data, and runs ``w_max`` GA generations keeping the
        ``n_in`` best individuals (by RBF prediction) each generation.

        Parameters
        ----------
        decs_t : np.ndarray
            Task decision database, shape (n, dim_t).
        objs_t : np.ndarray
            Task objective database, shape (n, 1).
        dim_t : int
            Task dimensionality.

        Returns
        -------
        np.ndarray
            Best decision vector found on the surrogate, shape (dim_t,).
        """
        rbf_model = _build_rbf(decs_t, objs_t.flatten())

        pop = decs_t.copy()
        for _ in range(self.w_max):
            offspring = ga_generation(pop, self.muc, self.mum)
            combined = np.vstack([pop, offspring])
            pred = rbf_model(combined)
            keep = min(self.n_in, len(combined))
            order = np.argsort(pred)[:keep]
            pop = combined[order]

        pred = rbf_model(pop)
        return pop[np.argmin(pred)]


def _build_rbf(decs, objs):
    """
    Build an RBF surrogate robustly against ill-conditioned interpolation systems.

    scipy's exact RBF interpolation can raise ``LinAlgError`` on degenerate or
    near-duplicate data (which arises here as transferred solutions accumulate).
    Escalating smoothing turns the interpolant into a regularized least-squares
    fit only as much as needed to obtain a solvable system.

    Parameters
    ----------
    decs : np.ndarray
        Training decisions, shape (n, dim).
    objs : np.ndarray
        Training objectives, shape (n,).

    Returns
    -------
    RBFInterpolator
        A fitted surrogate.
    """
    for smoothing in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        try:
            return RBFInterpolator(decs, objs, smoothing=smoothing)
        except np.linalg.LinAlgError:
            continue
    # Final fallback: heavy smoothing is always solvable
    return RBFInterpolator(decs, objs, smoothing=10.0)


def ramtea_knowledge_transfer(task_idx, active_tasks, best_solutions, dims, sim,
                              nfes_per_task, max_nfes_per_task):
    """
    Construct candidate solutions via similarity-based knowledge transfer.

    Task ``task_idx`` always evaluates its own surrogate optimum, and additionally
    evaluates each other active task's optimum with probability equal to the
    (rank-correlation, non-negative) similarity ``sim[task_idx, s]``.

    Parameters
    ----------
    task_idx : int
        Current task index.
    active_tasks : list[int]
        Active task indices.
    best_solutions : list[np.ndarray or None]
        Surrogate optima per task, length nt.
    dims : list[int]
        Dimensions of each task, length nt.
    sim : np.ndarray
        Non-negative similarity matrix, shape (nt, nt).
    nfes_per_task, max_nfes_per_task : list[int]
        Consumed and maximum evaluations per task.

    Returns
    -------
    candidates : np.ndarray or None
        Candidate solutions of shape (n_candidates, dims[task_idx]), or None if
        the task's budget is exhausted. Dimension mismatches are resolved by
        zero-padding or truncation.
    """
    if nfes_per_task[task_idx] >= max_nfes_per_task[task_idx]:
        return None

    # Always include the task's own surrogate optimum
    candidates = [np.asarray(best_solutions[task_idx]).flatten()]

    # Borrow other tasks' optima with probability = similarity
    for j in active_tasks:
        if task_idx == j or best_solutions[j] is None:
            continue
        if np.random.rand() < sim[task_idx, j]:
            sol_j = np.asarray(best_solutions[j]).flatten()
            if len(sol_j) < dims[task_idx]:
                sol_j = np.concatenate([sol_j, np.zeros(dims[task_idx] - len(sol_j))])
            elif len(sol_j) > dims[task_idx]:
                sol_j = sol_j[:dims[task_idx]]
            candidates.append(sol_j)

    candidates = np.clip(np.vstack(candidates), 0.0, 1.0)

    # Respect the remaining evaluation budget
    remaining_budget = max_nfes_per_task[task_idx] - nfes_per_task[task_idx]
    if len(candidates) > remaining_budget:
        candidates = candidates[:remaining_budget]

    return candidates

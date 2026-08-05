"""
Multi-Task Bayesian Optimization (MTBO)

This module implements MTBO for expensive multi-task optimization with knowledge transfer via multi-task Gaussian
processes.

Cost-sensitive evaluation (paper Sec. 2.4 / 3.3): each task carries an
evaluation cost c_k, and the budget is a shared amount of cost rather than a
per-task count of evaluations. The next evaluation maximizes the expected
improvement per unit cost, EI(x, k) / c_k, so a cheap task can absorb many
evaluations while an expensive one is entered only when it is worth its price.
Both are computed in log space as ``log EI - log c_k``, LogEI being what the
acquisition optimizer returns.

The default cost vector is [1, 1, ..., 1]. Under equal costs there is nothing to
trade off -- every task buys the same information per unit budget -- so the
algorithm keeps the round-robin schedule of one evaluation per task per
iteration, and the run is identical to the cost-unaware version, including the
total number of evaluations. A non-uniform cost vector switches the loop to the
paper's rule and lets the allocation across tasks be decided by the acquisition,
which means per-task evaluation counts are then an outcome rather than an input.

References
----------
    [1] Swersky, Kevin, Jasper Snoek, and Ryan P. Adams. "Multi-task bayesian optimization." Advances in neural information processing systems 26 (2013).

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.12
Version: 2.0
"""
from tqdm import tqdm
import torch
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mtgp_build, mtbo_next_point
import warnings

warnings.filterwarnings("ignore")


class MTBO:
    """
    Multi-Task Bayesian Optimization for expensive multi-task optimization problems.

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

    def __init__(self, problem, n_initial=None, max_nfes=None, task_cost=None, save_data=True, save_path='./Data',
                 name='MTBO', disable_tqdm=True):
        """
        Initialize Multi-Task Bayesian Optimization algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        task_cost : List[float] or None, optional
            Evaluation cost c_k of each task, e.g. the wall-clock duration of
            one evaluation. Defaults to equal costs [1, 1, ..., 1].

            The costs define a shared budget of ``sum(max_nfes_k * c_k)`` cost
            units, of which the initial samples already consume
            ``sum(n_initial_k * c_k)``. Every evaluation of task k spends c_k,
            and the task to evaluate next is the one maximizing EI per unit
            cost. Equal costs (of any magnitude) reproduce the cost-unaware
            schedule exactly; unequal costs let the algorithm buy more
            evaluations of the cheap tasks, so ``max_nfes`` then only sets the
            size of the budget and no longer fixes each task's own count.

            Note that ``max_nfes`` counts evaluations *at unit price*, so the
            budget scales with the costs: raising one task's cost enlarges the
            budget rather than shrinking the run. To hold the budget fixed while
            making a task pricier, lower ``max_nfes`` to match. Note also that
            the cost is one term of the trade-off and not an override -- a task
            whose expected improvement is high enough is still worth entering at
            several times the price.
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTBO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.task_cost = task_cost
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def _resolve_task_cost(self, nt):
        """
        Validate the cost vector and report whether the costs are uniform.

        Parameters
        ----------
        nt : int
            Number of tasks.

        Returns
        -------
        task_cost : np.ndarray
            Per-task evaluation costs, shape (nt,).
        equal_costs : bool
            True when every task costs the same, in which case the cost vector
            carries no information to act on.

        Raises
        ------
        ValueError
            If the vector has the wrong length or a non-positive entry.
        """
        if self.task_cost is None:
            return np.ones(nt, dtype=float), True

        task_cost = np.asarray(self.task_cost, dtype=float).flatten()
        if task_cost.shape[0] != nt:
            raise ValueError(f"task_cost must have length {nt}, got {task_cost.shape[0]}")
        if np.any(task_cost <= 0):
            raise ValueError("task_cost entries must be positive")

        return task_cost, bool(np.allclose(task_cost, task_cost[0]))

    def optimize(self):
        """
        Execute the Multi-Task Bayesian Optimization algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.double
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        task_cost, equal_costs = self._resolve_task_cost(nt)

        # Shared budget in cost units; the initial design already spends part of
        # it. With unit costs both reduce to plain evaluation counts.
        cost_budget = float(np.dot(max_nfes_per_task, task_cost))
        spent_cost = float(np.dot(n_initial_per_task, task_cost))

        # Initialize samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        pbar = tqdm(total=cost_budget, initial=spent_cost, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while spent_cost < cost_budget:
            # Only tasks whose next evaluation still fits in the budget
            affordable_tasks = [i for i in range(nt)
                                if task_cost[i] <= cost_budget - spent_cost + 1e-9]
            if not affordable_tasks:
                break

            if equal_costs:
                # No trade-off to make: keep the per-task budgets and the
                # round-robin schedule of one evaluation per task per iteration
                active_tasks = [i for i in affordable_tasks
                                if nfes_per_task[i] < max_nfes_per_task[i]]
                if not active_tasks:
                    break

            # Build multi-task Gaussian process surrogate model with normalized objectives
            objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
            mtgp = mtgp_build(decs, objs_normalized, dims, data_type=data_type)

            if equal_costs:
                for i in active_tasks:
                    # Select next sample point via acquisition function optimization
                    candidate_np = mtbo_next_point(mtgp=mtgp, task_id=i, objs=objs_normalized, dims=dims, nt=nt,
                                                   data_type=data_type)

                    obj, _ = evaluation_single(problem, candidate_np, i)

                    decs[i], objs[i] = vstack_groups((decs[i], candidate_np), (objs[i], obj))

                    nfes_per_task[i] += 1
                    spent_cost += task_cost[i]
                    pbar.update(task_cost[i])
            else:
                # Spend the budget where the expected improvement per unit cost
                # is largest (paper Sec. 3.3), comparing log EI - log c_k
                task_id, candidate_np = _select_cost_aware_point(
                    mtgp=mtgp, affordable_tasks=affordable_tasks, objs_normalized=objs_normalized,
                    dims=dims, nt=nt, task_cost=task_cost, nfes_per_task=nfes_per_task,
                    data_type=data_type)

                obj, _ = evaluation_single(problem, candidate_np, task_id)

                decs[task_id], objs[task_id] = vstack_groups((decs[task_id], candidate_np),
                                                             (objs[task_id], obj))

                nfes_per_task[task_id] += 1
                spent_cost += task_cost[task_id]
                pbar.update(task_cost[task_id])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _select_cost_aware_point(mtgp, affordable_tasks, objs_normalized, dims, nt, task_cost, nfes_per_task,
                             data_type=torch.double):
    """
    Pick the (task, point) pair with the largest expected improvement per unit cost.

    Every affordable task proposes its own best point, and the proposals are
    ranked by ``log EI - log c_k``, which orders them exactly as EI / c_k would
    while staying in the log space the acquisition optimizer works in.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Multi-task surrogate shared by all tasks.
    affordable_tasks : List[int]
        Tasks whose next evaluation still fits in the remaining budget.
    objs_normalized : List[np.ndarray]
        Per-task normalized objectives the surrogate was fitted on.
    dims : List[int]
        Decision space dimensionality of each task.
    nt : int
        Number of tasks.
    task_cost : np.ndarray
        Per-task evaluation costs.
    nfes_per_task : List[int]
        Evaluations spent on each task so far, used only to break a tie when no
        task reports a usable acquisition value.
    data_type : torch.dtype, optional
        Torch dtype for the acquisition optimization (default: torch.double).

    Returns
    -------
    task_id : int
        Task selected for the next evaluation.
    candidate_np : np.ndarray
        Point to evaluate on that task, shape (1, dims[task_id]).
    """
    candidates = {}
    scores = {}

    for i in affordable_tasks:
        candidate_np, log_ei = mtbo_next_point(mtgp=mtgp, task_id=i, objs=objs_normalized, dims=dims, nt=nt,
                                               data_type=data_type, return_acq_value=True)
        candidates[i] = candidate_np
        scores[i] = log_ei - np.log(task_cost[i])

    usable = [i for i in affordable_tasks if np.isfinite(scores[i])]
    if usable:
        task_id = max(usable, key=lambda i: scores[i])
    else:
        # Every acquisition underflowed to zero improvement; keep making
        # progress on the task that has been sampled least so far
        task_id = min(affordable_tasks, key=lambda i: nfes_per_task[i])

    return task_id, candidates[task_id]

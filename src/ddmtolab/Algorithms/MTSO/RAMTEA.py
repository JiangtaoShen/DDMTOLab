"""
Radial Basis Functions-Assisted MTEA (RAMTEA)

This module implements RAMTEA for expensive multi-task optimization with surrogate-assisted adaptive knowledge transfer.

References
----------
    [1] Shen, Jiangtao, et al. "Surrogate-assisted adaptive knowledge transfer for expensive \
        multitasking optimization." 2024 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2024.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.20
Version: 1.0
"""
from tqdm import tqdm
import time
from scipy.interpolate import RBFInterpolator
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Methods.Algo_Methods.sim_evaluation import sim_calculate
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class RAMTEA:
    """
    Radial Basis Functions-Assisted Multi-Task Evolutionary Algorithm for expensive optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
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

    def __init__(self, problem, n_initial=None, max_nfes=None, pop_size=50, w_max=50, save_data=True,
                 save_path='./TestData', name='RA-MTEA_test', disable_tqdm=True):
        """
        Initialize RAMTEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        pop_size : int, optional
            Population size for GA optimization on surrogate (default: 50)
        w_max : int, optional
            Maximum number of generations for GA optimization on surrogate (default: 50)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'RA-MTEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.pop_size = pop_size
        self.w_max = w_max
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

        # Initialize samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs', the_same=True)
        objs, _ = evaluation(problem, decs)

        # Compute task similarity matrix based on objective correlations
        sim = sim_calculate(objs)

        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            best_solutions = [None] * nt

            # Build RBF surrogate and optimize for each task
            for i in active_tasks:
                rbf_model = RBFInterpolator(decs[i], objs[i].flatten())

                def rbf(x):
                    return rbf_model(x)

                # Optimize surrogate using GA to find promising solutions
                surrogate_problem = MTOP()
                surrogate_problem.add_task(rbf, dim=dims[i])
                ga = GA(surrogate_problem, n=self.pop_size, max_nfes=self.pop_size * self.w_max, save_data=False)
                results = ga.optimize()
                best_solutions[i] = results.best_decs

            # Select candidates via similarity-based knowledge transfer
            for i in active_tasks:
                candidate = ramtea_knowledge_transfer(task_idx=i, active_tasks=active_tasks,
                                                      best_solutions=best_solutions,
                                                      dims=dims, sim=sim, nfes_per_task=nfes_per_task,
                                                      max_nfes_per_task=max_nfes_per_task)

                n_candidates = len(candidate)
                new_objs, _ = evaluation_single(problem, candidate, i)

                decs[i], objs[i] = vstack_groups((decs[i], candidate), (objs[i], new_objs))

                nfes_per_task[i] += n_candidates
                pbar.update(n_candidates)

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def ramtea_knowledge_transfer(task_idx, active_tasks, best_solutions, dims, sim, nfes_per_task, max_nfes_per_task):
    """
    Construct candidate solutions via similarity-based knowledge transfer.

    Parameters
    ----------
    task_idx : int
        Current task index
    active_tasks : list[int]
        List of active task indices
    best_solutions : list[np.ndarray or None]
        Best solutions for each task, length nt. Each element can be None or np.ndarray
    dims : list[int]
        Dimensions of each task, length nt
    sim : np.ndarray
        Task similarity matrix of shape (nt, nt)
    nfes_per_task : list[int]
        Number of function evaluations consumed for each task, length nt
    max_nfes_per_task : list[int]
        Maximum number of function evaluations for each task, length nt

    Returns
    -------
    candidates : np.ndarray or None
        Candidate solutions for current task of shape (n_candidates, dims[task_idx]).
        Returns None if current task has exhausted its evaluation budget

    Notes
    -----
    Knowledge transfer is performed with probability proportional to task similarity.
    The current task's best solution is always included, and solutions from other tasks
    are borrowed based on similarity values. Dimension alignment is performed via
    zero-padding or truncation as needed.
    """
    # Check if current task has exhausted its evaluation budget
    if nfes_per_task[task_idx] >= max_nfes_per_task[task_idx]:
        return None

    candidates = []

    # Add current task's best solution
    current_solution = np.asarray(best_solutions[task_idx]).flatten()
    candidates.append(current_solution)

    # Borrow solutions from other tasks based on similarity
    for j in active_tasks:
        if task_idx == j or best_solutions[j] is None:
            continue

        # Transfer knowledge with probability proportional to similarity
        if np.random.rand() < np.abs(sim[task_idx, j]):
            sol_j = np.asarray(best_solutions[j]).flatten()

            # Dimension alignment
            if len(sol_j) < dims[task_idx]:
                sol_j = np.concatenate([sol_j, np.zeros(dims[task_idx] - len(sol_j))])
            elif len(sol_j) > dims[task_idx]:
                sol_j = sol_j[:dims[task_idx]]

            candidates.append(sol_j)

    # Stack all candidates into a 2D array
    candidates = np.vstack(candidates)

    # Respect evaluation budget constraint
    remaining_budget = max_nfes_per_task[task_idx] - nfes_per_task[task_idx]
    if len(candidates) > remaining_budget:
        candidates = candidates[:remaining_budget]

    return candidates
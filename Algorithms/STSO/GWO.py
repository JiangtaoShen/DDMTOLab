"""
Grey Wolf Optimizer (GWO)

This module implements the Grey Wolf Optimizer for single-objective optimization problems.

References
----------
    [1] Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. \
        Advances in engineering software, 69, 46-61.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.08
Version: 1.0
"""
import time
from tqdm import tqdm
import numpy as np
from Methods.Algo_Methods.algo_utils import *


class GWO:
    """
    Grey Wolf Optimizer for single-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, save_data=True,
                 save_path='./TestData', name='GWO_test', disable_tqdm=True):
        """
        Initialize Grey Wolf Optimizer.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'GWO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Grey Wolf Optimizer algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population in [0,1] space and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize alpha, beta, delta wolves (top 3 solutions) for each task
        alpha_decs = [None] * nt
        alpha_objs = [None] * nt
        beta_decs = [None] * nt
        beta_objs = [None] * nt
        delta_decs = [None] * nt
        delta_objs = [None] * nt

        # Find initial alpha, beta, delta for each task
        for i in range(nt):
            # Sort by constraint violation first, then by objective
            cvs = np.sum(np.maximum(0, cons[i]), axis=1)
            sort_indices = np.lexsort((objs[i].flatten(), cvs))

            sorted_decs = decs[i][sort_indices]
            sorted_objs = objs[i][sort_indices]

            # Alpha: best solution
            alpha_decs[i] = sorted_decs[0:1, :]
            alpha_objs[i] = sorted_objs[0:1, :]

            # Beta: second best solution
            beta_decs[i] = sorted_decs[1:2, :] if n_per_task[i] > 1 else alpha_decs[i].copy()
            beta_objs[i] = sorted_objs[1:2, :] if n_per_task[i] > 1 else alpha_objs[i].copy()

            # Delta: third best solution
            delta_decs[i] = sorted_decs[2:3, :] if n_per_task[i] > 2 else beta_decs[i].copy()
            delta_objs[i] = sorted_objs[2:3, :] if n_per_task[i] > 2 else beta_objs[i].copy()

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Linearly decrease a from 2 to 0
                a = 2 - 2 * nfes_per_task[i] / max_nfes_per_task[i]

                # Update position of each search agent (grey wolf) in [0,1] space
                for j in range(n_per_task[i]):
                    # Calculate distance to alpha and update X1
                    r1 = np.random.rand(problem.dims[i])
                    r2 = np.random.rand(problem.dims[i])
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = np.abs(C1 * alpha_decs[i].flatten() - decs[i][j])
                    X1 = alpha_decs[i].flatten() - A1 * D_alpha

                    # Calculate distance to beta and update X2
                    r1 = np.random.rand(problem.dims[i])
                    r2 = np.random.rand(problem.dims[i])
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = np.abs(C2 * beta_decs[i].flatten() - decs[i][j])
                    X2 = beta_decs[i].flatten() - A2 * D_beta

                    # Calculate distance to delta and update X3
                    r1 = np.random.rand(problem.dims[i])
                    r2 = np.random.rand(problem.dims[i])
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = np.abs(C3 * delta_decs[i].flatten() - decs[i][j])
                    X3 = delta_decs[i].flatten() - A3 * D_delta

                    # Update position by averaging X1, X2, X3
                    decs[i][j] = (X1 + X2 + X3) / 3.0

                # Boundary constraint handling: clip to [0,1] space
                decs[i] = np.clip(decs[i], 0, 1)

                # Evaluate new positions (evaluation_single will transform to real space)
                objs[i], cons[i] = evaluation_single(problem, decs[i], i)

                # Update alpha, beta, delta
                # Sort by constraint violation first, then by objective
                cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                sort_indices = np.lexsort((objs[i].flatten(), cvs))

                sorted_decs = decs[i][sort_indices]
                sorted_objs = objs[i][sort_indices]

                # Update if better solutions found
                if sorted_objs[0] < alpha_objs[i][0]:
                    # Shift: alpha -> beta -> delta
                    delta_decs[i] = beta_decs[i].copy()
                    delta_objs[i] = beta_objs[i].copy()
                    beta_decs[i] = alpha_decs[i].copy()
                    beta_objs[i] = alpha_objs[i].copy()
                    alpha_decs[i] = sorted_decs[0:1, :]
                    alpha_objs[i] = sorted_objs[0:1, :]
                elif sorted_objs[0] < beta_objs[i][0]:
                    # Shift: beta -> delta
                    delta_decs[i] = beta_decs[i].copy()
                    delta_objs[i] = beta_objs[i].copy()
                    beta_decs[i] = sorted_decs[0:1, :]
                    beta_objs[i] = sorted_objs[0:1, :]
                elif sorted_objs[0] < delta_objs[i][0]:
                    delta_decs[i] = sorted_decs[0:1, :]
                    delta_objs[i] = sorted_objs[0:1, :]

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                # Append current population to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
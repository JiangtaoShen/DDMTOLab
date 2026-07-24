"""
Grey Wolf Optimizer (GWO)

This module implements the Grey Wolf Optimizer for single-objective optimization problems.

References
----------
    [1] Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69, 46-61.

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
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class GWO:
    """
    Grey Wolf Optimizer for single-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, save_data=True,
                 save_path='./Data', name='GWO', disable_tqdm=True):
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

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                n = n_per_task[i]
                d = problem.dims[i]

                # Rank current population (constraint violation first, then objective)
                # and take the top-3 wolves as alpha, beta, delta for this generation
                cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                order = np.lexsort((objs[i].flatten(), cvs))
                alpha = decs[i][order[0]]
                beta = decs[i][order[min(1, n - 1)]]
                delta = decs[i][order[min(2, n - 1)]]

                # Distances to leaders: D = |2*rand .* leader - X|
                d_alpha = np.abs(2 * np.random.rand(n, d) * alpha - decs[i])
                d_beta = np.abs(2 * np.random.rand(n, d) * beta - decs[i])
                d_delta = np.abs(2 * np.random.rand(n, d) * delta - decs[i])

                # Linearly decrease a from 2 to 0
                a = 2 * (1 - nfes_per_task[i] / max_nfes_per_task[i])

                # Leader-guided candidate positions: X_k = leader - A .* D, A in [-a, a]
                x1 = alpha - (2 * a * np.random.rand(n, d) - a) * d_alpha
                x2 = beta - (2 * a * np.random.rand(n, d) - a) * d_beta
                x3 = delta - (2 * a * np.random.rand(n, d) - a) * d_delta

                # New population: mean of the three leader-guided positions,
                # full replacement without elitism, clipped to [0,1] space
                decs[i] = np.clip((x1 + x2 + x3) / 3.0, 0, 1)

                # Evaluate new positions (evaluation_single will transform to real space)
                objs[i], cons[i] = evaluation_single(problem, decs[i], i)

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
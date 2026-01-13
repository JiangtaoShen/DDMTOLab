"""
Pareto Efficient Global Optimization (ParEGO)

This module implements ParEGO for expensive multi-objective optimization problems.

References
----------
    [1] Knowles, Joshua. "ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems." IEEE Transactions on Evolutionary Computation 10.1 (2006): 50-66.

Notes
-----
Author: Jiangtao Shen
Date: 2025.01.10
Version: 1.0
"""
from tqdm import tqdm
import torch
import numpy as np
from Methods.Algo_Methods.bo_utils import bo_next_point
from Methods.Algo_Methods.uniform_point import uniform_point
from Methods.Algo_Methods.algo_utils import *
import warnings
import time

warnings.filterwarnings("ignore")


class ParEGO:
    """
    Pareto Efficient Global Optimization algorithm for expensive multi-objective optimization.

    ParEGO uses scalarization with randomly selected weight vectors to convert
    multi-objective problems into single-objective problems, which are then
    solved using Bayesian Optimization with Expected Improvement.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'n_weights': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, n_weights=None, max_nfes=None, rho=0.05,
                 save_data=True, save_path='./TestData', name='ParEGO_test', disable_tqdm=True):
        """
        Initialize ParEGO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim - 1, following Knowles 2006)
        n_weights : int or List[int], optional
            Number of reference weight vectors per task (default: 10)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        rho : float, optional
            Augmentation coefficient for augmented Tchebycheff scalarization (default: 0.05)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'ParEGO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.n_weights = n_weights if n_weights is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.rho = rho
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the ParEGO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        no = problem.n_objs

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)

        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_weights_per_task = par_list(self.n_weights, nt)

        # Generate uniformly distributed weight vectors for each task
        W = []
        for i in range(nt):
            w_i, actual_n = uniform_point(n_weights_per_task[i], no[i])
            W.append(w_i)
            n_weights_per_task[i] = actual_n

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Reorganize initial data into task-specific history lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Randomly select a weight vector
                weight_idx = np.random.randint(0, n_weights_per_task[i])
                weight = W[i][weight_idx]

                # Scalarize multi-objective values to single objective using augmented Tchebycheff
                scalar_objs = self._scalarize(objs[i], weight)

                # Fit GP surrogate and select next candidate via BO with EI
                candidate_np = bo_next_point(dims[i], decs[i], scalar_objs, data_type=data_type)

                # Evaluate the candidate solution (get true multi-objective values)
                obj, _ = evaluation_single(problem, candidate_np, i)

                # Update dataset with new sample
                decs[i], objs[i] = vstack_groups((decs[i], candidate_np), (objs[i], obj))

                # Store cumulative history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    def _scalarize(self, objs, weight):
        """
        Scalarize multi-objective values using augmented Tchebycheff approach.

        The augmented Tchebycheff function is defined as:
            g(f|w) = max_j(w_j * f_j) + rho * sum_j(w_j * f_j)

        This scalarization is used in ParEGO to convert multi-objective problems
        to single-objective problems that can be optimized using standard BO.

        Parameters
        ----------
        objs : np.ndarray
            Multi-objective values of shape (N, M), where N is the number of
            samples and M is the number of objectives
        weight : np.ndarray
            Weight vector of shape (M,)

        Returns
        -------
        scalar_objs : np.ndarray
            Scalarized objective values of shape (N, 1)
        """
        # Normalize objectives to [0, 1] range for each objective dimension
        obj_min = np.min(objs, axis=0)
        obj_max = np.max(objs, axis=0)
        obj_range = obj_max - obj_min

        # Avoid division by zero
        obj_range = np.maximum(obj_range, 1e-10)
        normalized_objs = (objs - obj_min) / obj_range

        # Augmented Tchebycheff scalarization
        # g(f|w) = max_j(w_j * f_j) + rho * sum_j(w_j * f_j)
        weighted_objs = normalized_objs * weight

        # Max term (Tchebycheff)
        max_term = np.max(weighted_objs, axis=1)

        # Augmentation term
        aug_term = self.rho * np.sum(weighted_objs, axis=1)

        # Combined scalarized objective
        scalar_objs = max_term + aug_term

        return scalar_objs.reshape(-1, 1)
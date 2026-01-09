"""
Bayesian Optimization (BO)

This module implements Bayesian Optimization for expensive single-objective optimization problems.

References
----------
    [1] Jones, Donald R., Matthias Schonlau, and William J. Welch. \
        "Efficient global optimization of expensive black-box functions." \
        Journal of Global optimization 13.4 (1998): 455-492.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.11
Version: 1.0
"""
from tqdm import tqdm
import torch
from Methods.Algo_Methods.bo_utils import bo_next_point, bo_next_point_lcb
from Methods.Algo_Methods.algo_utils import *
import warnings
import time

warnings.filterwarnings("ignore")


class BO:
    """
    Bayesian Optimization algorithm for expensive optimization problems.

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

    def __init__(self, problem, n_initial=None, max_nfes=None, mode='ei', save_data=True,
                 save_path='./TestData', name='BO_test', disable_tqdm=True):
        """
        Initialize Bayesian Optimization algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        mode : str, optional
            Acquisition function mode: 'ei' for Expected Improvement or 'lcb' for Lower Confidence Bound
            (default: 'ei')
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'BO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.mode = mode.lower()
        if self.mode not in ['ei', 'lcb']:
            raise ValueError(f"mode must be 'ei' or 'lcb', got '{mode}'")
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Bayesian Optimization algorithm.

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
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Reorganize initial data into task-specific history lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Fit GP surrogate and select next candidate via acquisition function
                if self.mode == 'ei':
                    candidate_np = bo_next_point(dims[i], decs[i], objs[i], data_type=data_type)
                else:  # mode == 'lcb'
                    candidate_np, _ = bo_next_point_lcb(dims[i], decs[i], objs[i], data_type=data_type)

                # Evaluate the candidate solution
                obj, _ = evaluation_single(problem, candidate_np, i)

                # Update dataset with new sample
                decs[i], objs[i] = vstack_groups((decs[i], candidate_np), (objs[i], obj))

                # Store cumulative history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
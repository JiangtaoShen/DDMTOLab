"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

This module implements the CMA-ES algorithm for single-objective optimization problems.

References
----------
    [1] Hansen, N., & Ostermeier, A. (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2), 159-195. DOI: 10.1162/106365601750190398

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.12
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class CMA_ES:
    """
    CMA-ES for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, use_n=True,
                 save_data=True, save_path='./Data', name='CMA-ES', disable_tqdm=True):
        """
        Initialize CMA-ES Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: None, will use 4+3*log(D))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial step size (default: 0.3)
        use_n : bool, optional
            If True, use provided n; if False, use 4+3*log(D) (default: True)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'CMA-ES')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma0 = sigma0
        self.use_n = use_n
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the CMA-ES Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize parameters for each task
        params = []
        for t in range(nt):
            dim = problem.dims[t]

            # Determine population size
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dim))

            params.append(cmaes_init_params(dim, lam=lam, sigma0=self.sigma0))

        # Initialize tracking variables
        nfes_per_task = [0] * nt
        decs = [None] * nt
        objs = [None] * nt
        cons = [None] * nt
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                p = params[i]

                # Generate offspring using cmaes_sample
                sample_decs = cmaes_sample(p['m_dec'], p['sigma'], p['B'], p['D'], p['lam'])

                # Evaluate samples
                sample_objs, sample_cons = evaluation_single(problem, sample_decs, i)

                # Sort by constraint violation first, then by objective
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                sort_indices = constrained_sort(sample_objs, cvs)

                sample_decs = sample_decs[sort_indices]
                sample_objs = sample_objs[sort_indices]
                sample_cons = sample_cons[sort_indices]

                # Update current population
                decs[i] = sample_decs
                objs[i] = sample_objs
                cons[i] = sample_cons

                nfes_per_task[i] += p['lam']
                pbar.update(p['lam'])

                # Append to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

                # Update CMA-ES parameters (mean, paths, covariance, step size)
                cmaes_update(p, sample_decs, nfes_per_task[i])

        pbar.close()
        runtime = time.time() - start_time

        # Store CMA-ES state for external access (e.g., EEI-BO)
        self.cmaes_params = params

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

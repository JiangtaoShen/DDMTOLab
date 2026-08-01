"""
IPOP-CMA-ES (Increasing Population CMA-ES)

This module implements the IPOP-CMA-ES algorithm for single-objective optimization problems.

References
----------
    [1] Auger, A., & Hansen, N. (2005). A Restart CMA Evolution Strategy with Increasing Population Size. 2005 IEEE Congress on Evolutionary Computation, 2, 1769-1776. DOI: 10.1109/CEC.2005.1554902

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.27
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class IPOP_CMA_ES:
    """
    IPOP-CMA-ES for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, use_n=False,
                 save_data=True, save_path='./Data', name='IPOP-CMA-ES', disable_tqdm=True):
        """
        Initialize IPOP-CMA-ES Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Initial population size per task (default: None, will use 4+3*log(D))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial step size (default: 0.3)
        use_n : bool, optional
            If True, use provided n; if False, use 4+3*log(D) (default: False)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'IPOP-CMA-ES')
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
        Execute the IPOP-CMA-ES Algorithm.

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

            # Determine initial population size
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dim))

            p = self._initialize_task_params(dim, lam)
            params.append(p)

        # Initialize tracking variables
        nfes_per_task = [0] * nt
        decs = [None] * nt
        objs = [None] * nt
        cons = [None] * nt
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]
        obj_hist = [[] for _ in range(nt)]

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

                # Update objective history
                if len(obj_hist[i]) == 0:
                    obj_hist[i].append(sample_objs[0, 0])
                else:
                    obj_hist[i].append(min(obj_hist[i][-1], sample_objs[0, 0]))

                nfes_per_task[i] += p['lam']
                pbar.update(p['lam'])

                # Append to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

                # Update CMA-ES parameters (mean, paths, covariance, step size)
                cmaes_update(p, sample_decs, nfes_per_task[i])

                # Restart strategy IPOP
                # MATLAB: ObjHist = ObjHist(max(1, end - preGen):end) keeps preGen+1 entries
                pre_gen = 10 + (30 * int(p['dim'] / p['lam']))
                obj_hist[i] = obj_hist[i][max(0, len(obj_hist[i]) - pre_gen - 1):]
                obj_list = obj_hist[i] + sample_objs.flatten().tolist()

                # Check restart conditions
                if self._should_restart(p, obj_list):
                    # Double population size and reinitialize (MATLAB resets only
                    # mDec, sigma, and the lambda-dependent strategy parameters;
                    # evolution paths, covariance, and eigen state persist)
                    new_lam = p['lam'] * 2
                    new_p = self._initialize_task_params(p['dim'], new_lam)
                    for key in ('ps', 'pc', 'B', 'D', 'C', 'invsqrtC', 'eigenFE'):
                        new_p[key] = p[key]
                    p = new_p
                    params[i] = p
                    obj_hist[i] = []

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(
            all_decs=all_decs,
            all_objs=all_objs,
            runtime=runtime,
            max_nfes=nfes_per_task,
            all_cons=all_cons,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        return results

    def _initialize_task_params(self, dim, lam):
        """
        Initialize CMA-ES parameters for a task.

        Parameters
        ----------
        dim : int
            Problem dimension
        lam : int
            Population size

        Returns
        -------
        dict
            Dictionary containing all CMA-ES parameters
        """
        return cmaes_init_params(dim, lam=lam, sigma0=self.sigma0)

    def _should_restart(self, p, obj_list):
        """
        Check if restart conditions are met for IPOP strategy.

        Parameters
        ----------
        p : dict
            Current task parameters
        obj_list : list
            Combined objective history and current samples

        Returns
        -------
        bool
            True if restart is needed
        """
        if len(obj_list) == 0:
            return False

        # all(sigma{t} * (max(abs(pc{t}), sqrt(diag(C{t})))) < 1e-12 * Algo.sigma0)
        pc_c_max = np.maximum(np.abs(p['pc']), np.sqrt(np.diag(p['C'])))
        cond1 = np.all(p['sigma'] * pc_c_max < 1e-12 * self.sigma0)

        # any(sigma{t} * sqrt(diag(C{t})) > 1e8)
        cond2 = np.any(p['sigma'] * np.sqrt(np.diag(p['C'])) > 1e8)

        # sigma{t} * max(D{t}) == 0
        cond3 = p['sigma'] * np.max(p['D']) == 0

        # max(ObjList) - min(ObjList) < 1e-12
        cond4 = (np.max(obj_list) - np.min(obj_list)) < 1e-12

        return cond1 or cond2 or cond3 or cond4

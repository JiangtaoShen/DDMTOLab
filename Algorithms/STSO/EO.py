"""
Equilibrium Optimizer (EO)

This module implements the Equilibrium Optimizer for single-objective optimization problems.

References
----------
    [1] Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020).
        Equilibrium optimizer: A novel optimization algorithm.
        Knowledge-Based Systems, 191, 105190.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.12
Version: 1.0
"""
import time
from tqdm import tqdm
import numpy as np
from Methods.Algo_Methods.algo_utils import *


class EO:
    """
    Equilibrium Optimizer for single-objective optimization.

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
        """
        Get algorithm information.

        Parameters
        ----------
        print_info : bool, optional
            Whether to print information (default: True)

        Returns
        -------
        dict
            Algorithm information dictionary
        """
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, a1=2, a2=1, v=1, gp=0.5,
                 save_data=True, save_path='./TestData', name='EO_test', disable_tqdm=True):
        """
        Initialize Equilibrium Optimizer.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        a1 : float, optional
            Exploration constant (default: 2)
        a2 : float, optional
            Exploitation constant (default: 1)
        v : float, optional
            Volume coefficient (default: 1)
        gp : float, optional
            Generation probability (default: 0.5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.a1 = a1
        self.a2 = a2
        self.v = v
        self.gp = gp
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Equilibrium Optimizer algorithm.

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

        # Initialize equilibrium candidates (Ceq) for each task
        # Ceq contains 4 best solutions
        ceq_decs = [None] * nt
        ceq_objs = [None] * nt
        ceq_cons = [None] * nt

        for i in range(nt):
            dim = problem.dims[i]
            # Initialize 4 equilibrium candidates with worst values
            ceq_decs[i] = np.zeros((4, dim))
            ceq_objs[i] = np.full((4, 1), np.inf)
            ceq_cons[i] = np.full((4, 1), np.inf)

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = problem.dims[i]

                # Update equilibrium candidates (Ceq)
                cvs = np.sum(np.maximum(0, cons[i]), axis=1)

                for j in range(n_per_task[i]):
                    p_obj = objs[i][j, 0]
                    p_cv = cvs[j]

                    # Compare with each Ceq and update if better
                    for k in range(4):
                        c_obj = ceq_objs[i][k, 0]
                        c_cv = ceq_cons[i][k, 0]

                        if (p_cv < c_cv) or (p_cv == c_cv and p_obj < c_obj):
                            # Insert at position k and shift others
                            if k < 3:
                                ceq_decs[i][k + 1:] = ceq_decs[i][k:-1].copy()
                                ceq_objs[i][k + 1:] = ceq_objs[i][k:-1].copy()
                                ceq_cons[i][k + 1:] = ceq_cons[i][k:-1].copy()

                            ceq_decs[i][k] = decs[i][j].copy()
                            ceq_objs[i][k, 0] = p_obj
                            ceq_cons[i][k, 0] = p_cv
                            break

                # Calculate average equilibrium candidate (Ceq_ave)
                ceq_ave_dec = np.mean(ceq_decs[i], axis=0, keepdims=True)

                # Create equilibrium pool (4 Ceq + 1 Ceq_ave)
                c_pool_decs = np.vstack([ceq_decs[i], ceq_ave_dec])

                # Calculate ratio for exponential term
                ratio = (1 - nfes_per_task[i] / max_nfes_per_task[i]) ** \
                        (self.a2 * nfes_per_task[i] / max_nfes_per_task[i])

                # Generate offspring
                new_decs = np.zeros_like(decs[i])

                for j in range(n_per_task[i]):
                    # Random parameters
                    lam = np.random.rand(dim)
                    r = np.random.rand(dim)

                    # Randomly select from equilibrium pool
                    ceq_dec = c_pool_decs[np.random.randint(0, 5)]

                    # Exponential term (F)
                    F = self.a1 * np.sign(r - 0.5) * (np.exp(-lam * ratio) - 1)

                    # Generation control parameter (GCP)
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    GCP = 0.5 * r1 * np.ones(dim) * (r2 >= self.gp)

                    # Generation (G)
                    G0 = GCP * (ceq_dec - lam * decs[i][j])
                    G = G0 * F

                    # Update position
                    new_decs[j] = ceq_dec + (decs[i][j] - ceq_dec) * F + \
                                  (G / (lam + 1e-10) * self.v) * (1 - F)

                # Boundary constraint handling: clip to [0,1] space
                new_decs = np.clip(new_decs, 0, 1)

                # Evaluate new solutions
                new_objs, new_cons = evaluation_single(problem, new_decs, i)

                # Tournament selection: keep better solution
                new_cvs = np.sum(np.maximum(0, new_cons), axis=1)
                old_cvs = np.sum(np.maximum(0, cons[i]), axis=1)

                for j in range(n_per_task[i]):
                    # Compare by constraint violation first, then objective
                    if (new_cvs[j] < old_cvs[j]) or \
                            (new_cvs[j] == old_cvs[j] and new_objs[j] < objs[i][j]):
                        decs[i][j] = new_decs[j]
                        objs[i][j] = new_objs[j]
                        cons[i][j] = new_cons[j]

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
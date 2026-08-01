"""
Aquila Optimizer (AO)

This module implements the Aquila Optimizer for single-objective optimization problems.

References
----------
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A. A., Al-qaness, M. A., & Gandomi, A. H. (2021). Aquila Optimizer: A novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, 107250.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.10
Version: 1.0
"""
import time
from tqdm import tqdm
import numpy as np
from scipy.special import gamma
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class AO:
    """
    Aquila Optimizer for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, alpha=0.1, delta=0.1,
                 save_data=True, save_path='./Data', name='AO', disable_tqdm=True):
        """
        Initialize Aquila Optimizer.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        alpha : float, optional
            Exploitation adjustment parameter (default: 0.1)
        delta : float, optional
            Exploitation adjustment parameter (default: 0.1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'AO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.alpha = alpha
        self.delta = delta
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def levy_flight(self, d):
        """
        Generate Levy flight random walk.

        Parameters
        ----------
        d : int
            Dimension of the Levy flight

        Returns
        -------
        o : np.ndarray
            Levy flight step, shape (d,)
        """
        beta = 1.5
        sigma_num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        sigma_den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma = (sigma_num / sigma_den) ** (1 / beta)

        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        step = u / np.abs(v) ** (1 / beta)

        return step

    def optimize(self):
        """
        Execute the Aquila Optimizer algorithm.

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

        # Initialize best solution for each task (feasibility priority:
        # minimal constraint violation first, then minimal objective)
        best_decs = [None] * nt
        best_objs = [None] * nt
        best_cvs = [None] * nt

        for i in range(nt):
            cvs = np.sum(np.maximum(0, cons[i]), axis=1)
            sort_indices = np.lexsort((objs[i].flatten(), cvs))
            best_decs[i] = decs[i][sort_indices[0]].copy()
            best_objs[i] = objs[i][sort_indices[0], 0]
            best_cvs[i] = cvs[sort_indices[0]]

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
                n = n_per_task[i]
                max_nfes_i = max_nfes_per_task[i]

                # Per-generation dynamic parameters
                G1 = 2 * np.random.rand() - 1
                G2 = 2 * (1 - nfes_per_task[i] / max_nfes_i)

                # Spiral shape parameters
                to = np.arange(1, dim + 1)
                u = 0.0265
                r0 = 10
                r = r0 + u * to
                omega = 0.005
                phi0 = 3 * np.pi / 2
                phi = -omega * to + phi0
                x = r * np.sin(phi)
                y = r * np.cos(phi)

                # Quality function: QF = Gen^((2*rand-1)/(1-MaxGen)^2), where the
                # generation counter Gen = FE/N and MaxGen = maxFE/N
                gen = nfes_per_task[i] / n
                max_gen = max_nfes_i / n
                QF = gen ** ((2 * np.random.rand() - 1) / (1 - max_gen) ** 2)

                # Sequential per-individual generate -> evaluate -> select, so the
                # population mean, the random member, and the best solution all
                # reflect the partially updated population within the generation
                for j in range(n):
                    if nfes_per_task[i] <= 2 / 3 * max_nfes_i:
                        # Exploration phase
                        if np.random.rand() < 0.5:
                            # Method 1: Expanded exploration
                            new_dec = best_decs[i] * (
                                    1 - nfes_per_task[i] / max_nfes_i
                            ) + (np.mean(decs[i], axis=0) - best_decs[i]) * np.random.rand()
                        else:
                            # Method 2: Narrowed exploration
                            random_idx = np.random.randint(0, n)
                            new_dec = (best_decs[i] * self.levy_flight(dim) +
                                       decs[i][random_idx] +
                                       (y - x) * np.random.rand())
                    else:
                        # Exploitation phase
                        if np.random.rand() < 0.5:
                            # Method 1: Vertical stooping
                            new_dec = (
                                    (best_decs[i] - np.mean(decs[i], axis=0)) * self.alpha -
                                    np.random.rand() +
                                    np.random.rand() * self.delta
                            )
                        else:
                            # Method 2: Short glide attack
                            new_dec = (
                                    QF * best_decs[i] -
                                    (G1 * decs[i][j] * np.random.rand()) -
                                    G2 * self.levy_flight(dim) +
                                    np.random.rand() * G1
                            )

                    # Boundary constraint handling: clip to [0,1] space
                    new_dec = np.clip(new_dec, 0, 1)

                    # Evaluate offspring
                    new_obj, new_con = evaluation_single(problem, new_dec.reshape(1, -1), i)
                    new_obj = new_obj[0, 0]
                    new_con = new_con[0]
                    new_cv = np.sum(np.maximum(0, new_con))
                    nfes_per_task[i] += 1
                    pbar.update(1)

                    # Update global best (feasibility priority, ties keep the newer)
                    if (new_cv < best_cvs[i]) or \
                            (new_cv == best_cvs[i] and new_obj <= best_objs[i]):
                        best_decs[i] = new_dec.copy()
                        best_objs[i] = new_obj
                        best_cvs[i] = new_cv

                    # Tournament selection (MToP Selection_Tournament, Ep=0):
                    # replace if both infeasible and offspring CV is strictly lower,
                    # or both feasible and offspring objective is strictly lower
                    old_cv = np.sum(np.maximum(0, cons[i][j]))
                    old_obj = objs[i][j, 0]
                    if (old_cv > new_cv and old_cv > 0 and new_cv > 0) or \
                            (old_cv <= 0 and new_cv <= 0 and old_obj > new_obj):
                        decs[i][j] = new_dec
                        objs[i][j, 0] = new_obj
                        cons[i][j] = new_con

                # Append current population to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
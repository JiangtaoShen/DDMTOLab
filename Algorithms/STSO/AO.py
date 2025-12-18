"""
Aquila Optimizer (AO)

This module implements the Aquila Optimizer for single-objective optimization problems.

References
----------
    [1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A. A., Al-qaness, M. A., & \
        Gandomi, A. H. (2021). Aquila Optimizer: A novel meta-heuristic optimization \
        algorithm. Computers & Industrial Engineering, 157, 107250.

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
from Methods.Algo_Methods.algo_utils import *


class AO:
    """
    Aquila Optimizer for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, alpha=0.1, delta=0.1,
                 save_data=True, save_path='./TestData', name='AO_test', disable_tqdm=True):
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
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'AO_test')
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

        # Initialize best solution for each task
        best_decs = [None] * nt
        best_objs = [None] * nt

        for i in range(nt):
            # Sort by constraint violation first, then by objective
            cvs = np.sum(np.maximum(0, cons[i]), axis=1)
            sort_indices = np.lexsort((objs[i].flatten(), cvs))
            best_decs[i] = decs[i][sort_indices[0]:sort_indices[0] + 1, :]
            best_objs[i] = objs[i][sort_indices[0]:sort_indices[0] + 1, :]

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

                # Calculate dynamic parameters
                G1 = 2 * np.random.rand() - 1
                G2 = 2 * (1 - nfes_per_task[i] / max_nfes_per_task[i])

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

                # Quality function
                QF = nfes_per_task[i] ** (
                        (2 * np.random.rand() - 1) /
                        (1 - max_nfes_per_task[i]) ** 2
                )

                # Mean position
                mean_dec = np.mean(decs[i], axis=0)

                # Generate offspring
                new_decs = np.zeros_like(decs[i])

                for j in range(n_per_task[i]):
                    if nfes_per_task[i] <= 2 / 3 * max_nfes_per_task[i]:
                        # Exploration phase
                        if np.random.rand() < 0.5:
                            # Method 1: Expanded exploration
                            new_decs[j] = best_decs[i].flatten() * (
                                    1 - nfes_per_task[i] / max_nfes_per_task[i]
                            ) + (mean_dec - best_decs[i].flatten()) * np.random.rand()
                        else:
                            # Method 2: Narrowed exploration
                            random_idx = np.random.randint(0, n_per_task[i])
                            levy = self.levy_flight(dim)
                            new_decs[j] = (best_decs[i].flatten() * levy +
                                           decs[i][random_idx] +
                                           (y - x) * np.random.rand())
                    else:
                        # Exploitation phase
                        if np.random.rand() < 0.5:
                            # Method 1: Vertical stooping
                            new_decs[j] = (
                                    (best_decs[i].flatten() - mean_dec) * self.alpha -
                                    np.random.rand() +
                                    np.random.rand() * self.delta
                            )
                        else:
                            # Method 2: Short glide attack
                            levy = self.levy_flight(dim)
                            new_decs[j] = (
                                    QF * best_decs[i].flatten() -
                                    (G1 * decs[i][j] * np.random.rand()) -
                                    G2 * levy +
                                    np.random.rand() * G1
                            )

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

                        # Update best solution if improved
                        if (new_cvs[j] == 0 and new_objs[j] < best_objs[i][0]) or \
                                (new_cvs[j] < np.sum(np.maximum(0, cons[i][sort_indices[0]]))):
                            best_decs[i] = new_decs[j:j + 1, :]
                            best_objs[i] = new_objs[j:j + 1, :]

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
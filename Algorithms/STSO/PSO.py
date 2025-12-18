"""
Particle Swarm Optimization (PSO)

This module implements Particle Swarm Optimization for single-objective optimization problems.

References
----------
    [1] Kennedy, James, and Russell Eberhart. "Particle swarm optimization." Proceedings of \
        ICNN'95-international conference on neural networks. Vol. 4. IEEE, 1995.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.23
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from typing import List
from Methods.Algo_Methods.algo_utils import *


class PSO:
    """
    Particle Swarm Optimization algorithm for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, min_w=0.4, max_w=0.9, c1=0.2, c2=0.2, save_data=True,
                 save_path='./TestData', name='PSO_test', disable_tqdm=True):
        """
        Initialize Particle Swarm Optimization algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        min_w : float, optional
            Minimum inertia weight (default: 0.4)
        max_w : float, optional
            Maximum inertia weight (default: 0.9)
        c1 : float, optional
            Cognitive coefficient (self-learning factor) (default: 0.2)
        c2 : float, optional
            Social coefficient (swarm-learning factor) (default: 0.2)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'PSO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.min_w = min_w
        self.max_w = max_w
        self.c1 = c1
        self.c2 = c2
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Particle Swarm Optimization algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        dims = problem.dims
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population in [0,1] space and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize particle velocities to zero
        vel = [np.zeros_like(d) for d in decs]

        # Initialize personal best positions, objectives, and constraints
        pbest_decs = [d.copy() for d in decs]
        pbest_objs = [o.copy() for o in objs]
        pbest_cons = [c.copy() for c in cons]

        # Initialize global best for each task
        gbest_decs = []
        gbest_objs = []
        gbest_cons = []

        for i in range(nt):
            # Find best particle considering constraints
            cvs = np.sum(np.maximum(0, cons[i]), axis=1)
            sort_indices = np.lexsort((objs[i].flatten(), cvs))
            best_idx = sort_indices[0]

            gbest_decs.append(decs[i][best_idx:best_idx + 1, :])
            gbest_objs.append(objs[i][best_idx:best_idx + 1, :])
            gbest_cons.append(cons[i][best_idx:best_idx + 1, :])

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Linearly decrease inertia weight from max_w to min_w
                w = self.max_w - (self.max_w - self.min_w) * nfes_per_task[i] / max_nfes_per_task[i]

                # Generate random coefficients for cognitive and social components
                r1 = np.random.rand(n_per_task[i], dims[i])
                r2 = np.random.rand(n_per_task[i], dims[i])

                # Update velocity: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
                vel[i] = (w * vel[i] + self.c1 * r1 * (pbest_decs[i] - decs[i]) +
                          self.c2 * r2 * (gbest_decs[i] - decs[i]))

                # Update positions and enforce boundary constraints: clip to [0,1] space
                decs[i] = np.clip(decs[i] + vel[i], 0, 1)

                objs[i], cons[i] = evaluation_single(problem, decs[i], i)

                # Calculate constraint violations
                current_cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                pbest_cvs = np.sum(np.maximum(0, pbest_cons[i]), axis=1)

                # Update personal best if current position is better
                # Better means: lower constraint violation, or same violation but lower objective
                improved = (current_cvs < pbest_cvs) | \
                           ((current_cvs == pbest_cvs) & (objs[i].flatten() < pbest_objs[i].flatten()))

                pbest_decs[i][improved] = decs[i][improved]
                pbest_objs[i][improved] = objs[i][improved]
                pbest_cons[i][improved] = cons[i][improved]

                # Update global best if any personal best improves
                pbest_cvs = np.sum(np.maximum(0, pbest_cons[i]), axis=1)
                sort_indices = np.lexsort((pbest_objs[i].flatten(), pbest_cvs))
                best_idx = sort_indices[0]

                gbest_cv = np.sum(np.maximum(0, gbest_cons[i]))
                best_cv = pbest_cvs[best_idx]

                if (best_cv < gbest_cv) or \
                        (best_cv == gbest_cv and pbest_objs[i][best_idx] < gbest_objs[i][0]):
                    gbest_decs[i] = pbest_decs[i][best_idx:best_idx + 1, :]
                    gbest_objs[i] = pbest_objs[i][best_idx:best_idx + 1, :]
                    gbest_cons[i] = pbest_cons[i][best_idx:best_idx + 1, :]

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
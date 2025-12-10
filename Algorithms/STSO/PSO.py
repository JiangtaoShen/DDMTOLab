"""
Particle Swarm Optimization (PSO)

This module implements Particle Swarm Optimization for single-objective optimization problems.

References
----------
.. [1] Kennedy, James, and Russell Eberhart. "Particle swarm optimization." Proceedings of
   ICNN'95-international conference on neural networks. Vol. 4. IEEE, 1995.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.23
Version: 1.0
"""
import time
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
        'n_objs': '1',
        'n_cons': '0',
        'n': 'unequal',
        'max_nfes': 'unequal',
        'expensive': 'False',
        'knowledge_transfer': 'False'
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

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs = init_history(decs, objs)

        # Initialize particle velocities to zero
        vel = [np.zeros_like(d) for d in decs]

        # Initialize personal best positions and objectives
        pbest_decs = [d.copy() for d in decs]
        pbest_objs = [o.copy() for o in objs]

        # Initialize global best for each task
        gbest_decs = []
        gbest_objs = []
        for i in range(nt):
            min_idx = np.argmin(objs[i])
            gbest_decs.append(decs[i][min_idx:min_idx + 1, :])
            gbest_objs.append(objs[i][min_idx:min_idx + 1, :])

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
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

                # Update positions and enforce boundary constraints
                decs[i] = np.clip(decs[i] + vel[i], 0, 1)

                objs[i], _ = evaluation_single(problem, decs[i], i)

                # Update personal best if current position is better
                improved = (objs[i] < pbest_objs[i]).flatten()
                pbest_decs[i][improved] = decs[i][improved]
                pbest_objs[i][improved] = objs[i][improved]

                # Update global best if any personal best improves
                min_idx = np.argmin(pbest_objs[i])
                if pbest_objs[i][min_idx] < gbest_objs[i]:
                    gbest_decs[i] = pbest_decs[i][min_idx:min_idx + 1, :]
                    gbest_objs[i] = pbest_objs[i][min_idx:min_idx + 1, :]

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs, all_objs, runtime, nfes_per_task,
                                     save_path=self.save_path, filename=self.name, save_data=self.save_data)

        return results
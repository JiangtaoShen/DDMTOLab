"""
Social Learning Particle Swarm Optimization (SL-PSO)

This module implements the Social Learning PSO for single-objective optimization problems.

References
----------
    [1] Cheng, R., & Jin, Y. (2014). A social learning particle swarm optimization \
        algorithm for scalable optimization. Information Sciences, 291, 43-60.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.22
Version: 1.0
"""
import time
from tqdm import tqdm
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class SLPSO:
    """
    Social Learning Particle Swarm Optimization for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, save_data=True,
                 save_path='./TestData', name='SLPSO_test', disable_tqdm=True):
        """
        Initialize Social Learning PSO algorithm.

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
            Name for the experiment (default: 'SLPSO_test')
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
        Execute the Social Learning PSO algorithm.

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

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize velocities for each task
        vel = [np.zeros_like(d) for d in decs]

        # Calculate enhanced population size m for each task
        m_per_task = []
        for i in range(nt):
            M = n_per_task[i]
            m = M + int(np.floor(problem.dims[i] / 10))
            m_per_task.append(m)

            # Expand population to size m if necessary
            if decs[i].shape[0] < m:
                extra = m - decs[i].shape[0]
                extra_decs = np.random.rand(extra, problem.dims[i])
                extra_objs, extra_cons = evaluation_single(problem, extra_decs, i)
                decs[i] = np.vstack([decs[i], extra_decs])
                objs[i] = np.vstack([objs[i], extra_objs])
                cons[i] = np.vstack([cons[i], extra_cons])
                vel[i] = np.vstack([vel[i], np.zeros((extra, problem.dims[i]))])
                nfes_per_task[i] += extra

        # Calculate learning probability for each task
        PL = []
        for i in range(nt):
            dim = problem.dims[i]
            M = n_per_task[i]
            m = m_per_task[i]
            pl = np.zeros(m)
            for j in range(m):
                pl[j] = (1 - j / m) ** np.log(np.sqrt(np.ceil(dim / M)))
            PL.append(pl)

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = problem.dims[i]
                M = n_per_task[i]
                m = m_per_task[i]

                # Calculate social learning coefficient
                c3 = dim / M * 0.01

                # Sort population by fitness (descending order, worst first)
                cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                # For sorting: higher CV is worse, higher objective is worse (for minimization)
                sort_indices = np.lexsort((-objs[i].flatten(), -cvs))

                decs[i] = decs[i][sort_indices]
                objs[i] = objs[i][sort_indices]
                cons[i] = cons[i][sort_indices]
                vel[i] = vel[i][sort_indices]

                # Best individual is at the end (index m-1)
                best_dec = decs[i][-1:, :]
                best_obj = objs[i][-1:, :]

                # Calculate center position
                center = np.mean(decs[i], axis=0, keepdims=True)
                center = np.repeat(center, m, axis=0)

                # Random coefficients
                randco1 = np.random.rand(m, dim)
                randco2 = np.random.rand(m, dim)
                randco3 = np.random.rand(m, dim)

                # Social learning: select demonstrators
                # For each particle and each dimension, select a better (higher index) particle
                win_idx = np.zeros((m, dim), dtype=int)
                for j in range(m):
                    for k in range(dim):
                        # Select from particles with index >= j (better particles)
                        win_idx[j, k] = np.random.randint(j, m)

                # Get winner positions
                pwin = np.zeros((m, dim))
                for j in range(m):
                    for k in range(dim):
                        pwin[j, k] = decs[i][win_idx[j, k], k]

                # Social learning mask (learning probability)
                lpmask_1d = np.random.rand(m) < PL[i]
                lpmask_1d[-1] = False  # Best particle doesn't learn
                lpmask = np.repeat(lpmask_1d.reshape(-1, 1), dim, axis=1)

                # Update velocity and position
                v1 = randco1 * vel[i] + randco2 * (pwin - decs[i]) + c3 * randco3 * (center - decs[i])
                p1 = decs[i] + v1

                # Apply learning mask
                vel[i] = np.where(lpmask, v1, vel[i])
                decs[i] = np.where(lpmask, p1, decs[i])

                # Boundary constraint handling for learning particles (not the best)
                decs[i][:-1] = np.clip(decs[i][:-1], 0, 1)

                # Evaluate all particles except the best one
                new_objs, new_cons = evaluation_single(problem, decs[i][:-1], i)
                objs[i][:-1] = new_objs
                cons[i][:-1] = new_cons

                nfes_per_task[i] += (m - 1)
                pbar.update(m - 1)

                # Store only the base population (size M) for history
                # Take the M best particles (at the end after sorting)
                store_decs = decs[i][-M:]
                store_objs = objs[i][-M:]
                store_cons = cons[i][-M:]

                append_history(all_decs[i], store_decs, all_objs[i], store_objs,
                               all_cons[i], store_cons)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
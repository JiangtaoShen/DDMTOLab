"""
Social Learning Particle Swarm Optimization (SL-PSO)

This module implements the Social Learning PSO for single-objective optimization problems.

References
----------
    [1] Cheng, R., & Jin, Y. (2014). A social learning particle swarm optimization algorithm for scalable optimization. Information Sciences, 291, 43-60.

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


class SL_PSO:
    """
    Social Learning Particle Swarm Optimization for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, save_data=True,
                 save_path='./Data', name='SL-PSO', disable_tqdm=True):
        """
        Initialize Social Learning PSO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Base swarm size M per task (default: 100, the paper's setting).
            The actual swarm size per task is m = M + floor(dim / 10),
            following the reference formulation; M is also used in the
            learning-probability exponent and the social influence factor.
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
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Base swarm size M per task (paper: M = 100);
        # actual swarm size per task: m = M + floor(dim / 10)
        base_m_per_task = par_list(self.n, nt)
        m_per_task = [base_m_per_task[i] + int(np.floor(problem.dims[i] / 10))
                      for i in range(nt)]

        # Initialize swarm of size m in [0,1] space and evaluate for each task
        decs = initialization(problem, m_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = m_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize velocities for each task
        vel = [np.zeros_like(d) for d in decs]

        # Learning probability per task:
        # PL(i) = (1 - (i-1)/m)^(0.5*log(ceil(dim/M))), worst-to-best sorted swarm
        PL = []
        for i in range(nt):
            dim = problem.dims[i]
            M = base_m_per_task[i]
            m = m_per_task[i]
            pl = (1.0 - np.arange(m) / m) ** np.log(np.sqrt(np.ceil(dim / M)))
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
                M = base_m_per_task[i]
                m = m_per_task[i]

                # Social influence factor: epsilon = beta * dim / M, beta = 0.01
                c3 = dim / M * 0.01

                # Sort population by fitness (descending order, worst first, best last)
                cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                # For sorting: higher CV is worse, higher objective is worse (for minimization)
                sort_indices = np.lexsort((-objs[i].flatten(), -cvs))

                decs[i] = decs[i][sort_indices]
                objs[i] = objs[i][sort_indices]
                cons[i] = cons[i][sort_indices]
                vel[i] = vel[i][sort_indices]

                # Calculate center (mean) position of the swarm
                center = np.mean(decs[i], axis=0, keepdims=True)

                # Random coefficients
                randco1 = np.random.rand(m, dim)
                randco2 = np.random.rand(m, dim)
                randco3 = np.random.rand(m, dim)

                # Social learning: for particle j, each dimension learns from a
                # strictly better particle (demonstrator).
                # MATLAB: winidx = i + ceil(rand*(m-i)), i.e. uniform over {j+1, ..., m-1}
                idx = np.arange(m).reshape(-1, 1)
                offsets = np.ceil(np.random.rand(m, dim) * (m - 1 - idx)).astype(int)
                win_idx = np.clip(idx + offsets, 0, m - 1)

                # Get demonstrator positions (per-dimension indexing)
                pwin = np.take_along_axis(decs[i], win_idx, axis=0)

                # Social learning mask (learning probability)
                lpmask_1d = np.random.rand(m) < PL[i]
                lpmask_1d[-1] = False  # Best particle doesn't learn
                lpmask = lpmask_1d.reshape(-1, 1)

                # Update velocity and position
                v1 = randco1 * vel[i] + randco2 * (pwin - decs[i]) + c3 * randco3 * (center - decs[i])
                p1 = decs[i] + v1

                # Apply learning mask
                vel[i] = np.where(lpmask, v1, vel[i])
                decs[i] = np.where(lpmask, p1, decs[i])

                # Boundary constraint handling (best particle is unchanged and in-bounds)
                decs[i] = np.clip(decs[i], 0, 1)

                # Evaluate the whole swarm (canonical implementation: FES += m per generation)
                objs[i], cons[i] = evaluation_single(problem, decs[i], i)

                nfes_per_task[i] += m
                pbar.update(m)

                # Append current population to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i],
                               all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
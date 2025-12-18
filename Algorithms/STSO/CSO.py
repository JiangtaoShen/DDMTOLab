"""
Competitive Swarm Optimizer (CSO)

This module implements Competitive Swarm Optimizer for single-objective optimization problems.

References
----------
    [1] Cheng, Ran, and Yaochu Jin. "A competitive swarm optimizer for large scale optimization." \
        IEEE Transactions on Cybernetics 45.2 (2015): 191-204.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.31
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from Methods.Algo_Methods.algo_utils import *


class CSO:
    """
    Competitive Swarm Optimizer algorithm for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, phi=0.1, save_data=True, save_path='./TestData', name='CSO_test',
                 disable_tqdm=True):
        """
        Initialize Competitive Swarm Optimizer algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        phi : float, optional
            Social influence parameter for mean position learning (default: 0.1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'CSO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.phi = phi
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Competitive Swarm Optimizer algorithm.

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

        # Initialize particle velocities to zero
        vel = [np.zeros_like(d) for d in decs]

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Calculate constraint violations
                cvs = np.sum(np.maximum(0, cons[i]), axis=1)

                # Randomly pair particles for pairwise competition
                rnd_idx = np.random.permutation(n_per_task[i])
                loser_idx = rnd_idx[:n_per_task[i] // 2]
                winner_idx = rnd_idx[n_per_task[i] // 2:]

                # Determine actual winners and losers by comparing constraint violation first, then objectives
                loser_objs = objs[i][loser_idx]
                winner_objs = objs[i][winner_idx]
                loser_cvs = cvs[loser_idx]
                winner_cvs = cvs[winner_idx]

                # Swap indices if loser is better than winner
                # Better means: lower constraint violation, or same violation but lower objective
                swap_mask = (loser_cvs < winner_cvs) | \
                            ((loser_cvs == winner_cvs) & (loser_objs.flatten() < winner_objs.flatten()))

                temp_idx = loser_idx[swap_mask].copy()
                loser_idx[swap_mask] = winner_idx[swap_mask]
                winner_idx[swap_mask] = temp_idx

                # Calculate mean position of winners for social learning
                winner_mean = np.mean(decs[i][winner_idx], axis=0, keepdims=True)

                # Update each loser by learning from its paired winner and swarm mean
                for j, loser_j in enumerate(loser_idx):
                    winner_j = winner_idx[j]

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    r3 = np.random.rand()

                    # Velocity update: inertia + learn from winner + learn from swarm mean
                    vel[i][loser_j] = (r1 * vel[i][loser_j] +
                                       r2 * (decs[i][winner_j] - decs[i][loser_j]) +
                                       self.phi * r3 * (winner_mean - decs[i][loser_j]))

                    # Update position and enforce boundary constraints: clip to [0,1] space
                    decs[i][loser_j] = np.clip(decs[i][loser_j] + vel[i][loser_j], 0, 1)

                # Evaluate only updated losers (winners unchanged)
                objs[i][loser_idx], cons[i][loser_idx] = evaluation_single(problem, decs[i][loser_idx], i)

                nfes_per_task[i] += n_per_task[i] // 2
                pbar.update(n_per_task[i] // 2)

                # Append current population to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
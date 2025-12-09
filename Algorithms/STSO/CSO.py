"""
Competitive Swarm Optimizer (CSO)

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.31
Version: 1.0

References:
[1] Cheng, Ran, and Yaochu Jin. "A competitive swarm optimizer for large scale optimization."
    IEEE Transactions on Cybernetics 45.2 (2015): 191-204.
"""
import time
from tqdm import tqdm
from typing import List
from Methods.Algo_Methods.algo_utils import *


class CSO:

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
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, phi=0.1, save_data=True, save_path='./Data', name='cso_test',
                 disable_tqdm=True):
        """
        Competitive Swarm Optimizer (CSO)

        Args:
            problem: MTOP instance
            n (int or List[int]): Population size per task (default: 100)
            max_nfes (int or List[int]): Maximum number of function evaluations per task (default: 10000)
            phi (float): Social influence parameter for mean position learning (default: 0.1)
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

        start_time = time.time()
        problem = self.problem
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

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Randomly pair particles for pairwise competition
                rnd_idx = np.random.permutation(n_per_task[i])
                loser_idx = rnd_idx[:n_per_task[i] // 2]
                winner_idx = rnd_idx[n_per_task[i] // 2:]

                # Determine actual winners and losers by comparing objectives
                loser_objs = objs[i][loser_idx]
                winner_objs = objs[i][winner_idx]

                # Swap indices if loser has better (lower) objective value
                swap_mask = (loser_objs < winner_objs).flatten()
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

                    # Update position and enforce boundary constraints
                    decs[i][loser_j] = np.clip(decs[i][loser_j] + vel[i][loser_j], 0, 1)

                # Evaluate only updated losers (winners unchanged)
                objs[i][loser_idx], _ = evaluation_single(problem, decs[i][loser_idx], i)

                nfes_per_task[i] += n_per_task[i] // 2
                pbar.update(n_per_task[i] // 2)

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs, all_objs, runtime, nfes_per_task,
                                     save_path=self.save_path, filename=self.name, save_data=self.save_data)

        return results
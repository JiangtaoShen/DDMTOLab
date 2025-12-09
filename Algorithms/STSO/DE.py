"""
Differential Evolution (DE)

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.24
Version: 1.0

References:
[1] Storn, Rainer, and Kenneth Price. "Differential evolution–a simple and efficient heuristic for global optimization
    over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.
"""
import time
from tqdm import tqdm
from Methods.Algo_Methods.algo_utils import *


class DE:

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

    def __init__(self, problem, n=None, max_nfes=None, F=0.5, CR=0.9, save_data=True, save_path='./TestData',
                 name='DE_test', disable_tqdm=True):
        """
        Differential Evolution (DE)

        Args:
            problem: MTOP instance
            n (int or List[int]): Population size per task (default: 100)
            max_nfes (int or List[int]): Maximum number of function evaluations per task (default: 10000)
            F (float): Scaling factor (default: 0.5)
            CR (float): Crossover probability (default: 0.9)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.F = F
        self.CR = CR
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

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Generate trial vectors through DE mutation and crossover
                off_decs = de_generation(decs[i], self.F, self.CR)
                off_objs, _ = evaluation_single(problem, off_decs, i)

                # Greedy selection: replace parent if offspring is better
                improved = (off_objs < objs[i]).flatten()
                decs[i][improved] = off_decs[improved, :]
                objs[i][improved] = off_objs[improved, :]

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs, all_objs, runtime, nfes_per_task, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
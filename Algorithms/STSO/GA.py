"""
Genetic Algorithm (GA)

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.11
Version: 1.0

References:
[1] David E.Goldberg. Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley,Reading, MA, 1989.
[2] John H. Holland. Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology,
    Control, and Artificial Intelligence. University of Michigan Press, AnnArbor, Ml, 1st edition, 1975. Reprinted by MIT
    Press in 1992.
"""
from tqdm import tqdm
import time
from typing import List
from Methods.Algo_Methods.algo_utils import *


class GA:

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '0',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, muc=2.0, mum=5.0, save_data=True, save_path='./TestData',
                 name='GA_test', disable_tqdm=True):
        """
        Genetic Algorithm (GA)

        Args:
            problem: MTOP instance
            n (int or List[int]): Population size per task (default: 100)
            max_nfes (int or List[int]): Maximum number of function evaluations per task (default: 10000)
            muc (float): Distribution index for simulated binary crossover (SBX) (default: 2.0)
            mum (float): Distribution index for polynomial mutation (PM) (default: 5.0)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.muc = muc
        self.mum = mum
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
                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i], self.muc, self.mum)
                off_objs, _ = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs))

                # Elitist selection: keep top n individuals with minimum objective values
                index = selection_elit(objs[i], n_per_task[i])
                objs[i], decs[i] = select_by_index(index, objs[i], decs[i])

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs, all_objs, runtime, nfes_per_task, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
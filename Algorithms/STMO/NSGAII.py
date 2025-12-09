"""
Nondominated Sorting Genetic Algorithm II (NSGA-II)

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.23
Version: 1.0

References:
[1] Deb, Kalyanmoy, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary
    computation 6.2 (2002): 182-197.
"""
from tqdm import tqdm
import time
from Methods.Algo_Methods.algo_utils import *


class NSGAII:

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'n_objs': '2-M/unequal',
        'n_cons': '1-C/unequal',
        'n': 'unequal',
        'max_nfes': 'unequal',
        'expensive': 'False',
        'knowledge_transfer': 'False'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=15.0, save_data=True, save_path='./TestData',
                 name='NSGA-II_test', disable_tqdm=True):
        """
        Nondominated Sorting Genetic Algorithm II (NSGA-II)

        Args:
            problem: MTOP instance
            n (int or List[int]): Population size per task (default: 100)
            max_nfes (int or List[int]): Maximum number of function evaluations per task (default: 10000)
            muc (float): Distribution index for simulated binary crossover (SBX) (default: 2.0).
            mum (float): Distribution index for polynomial mutation (PM) (default: 5.0).
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
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Perform initial non-dominated sorting for each task
        rank = []
        for i in range(nt):
            rank_i, _, _ = nsga2_sort(objs[i], cons[i])
            rank.append(rank_i.copy())

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Parent selection via binary tournament based on rank
                matingpool = tournament_selection(2, n_per_task[i], rank[i])

                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i][matingpool, :], muc=self.muc, mum=self.mum)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs),
                                                          (cons[i], off_cons))

                # Environmental selection: sort and keep best n individuals
                rank[i], _, _ = nsga2_sort(objs[i], cons[i])
                index = np.argsort(rank[i])[:n_per_task[i]]
                objs[i], decs[i], cons[i], rank[i] = select_by_index(index, objs[i], decs[i], cons[i], rank[i])

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs, all_objs, runtime, nfes_per_task, all_cons, self.save_path, self.name,
                                     self.save_data)

        return results


def nsga2_sort(
    objs: np.ndarray,
    cons: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort solutions based on NSGA-II criteria: non-dominated sorting first,
    then crowding distance (larger is better).

    Parameters
    ----------
    objs : np.ndarray
        Objective value matrix, shape: (pop_size, n_obj)
    cons : np.ndarray or None, optional
        Constraint matrix, shape: (pop_size, n_con). If None, no constraints
        are considered (default: None)

    Returns
    -------
    rank : np.ndarray
        Ranking of each solution (0-based index after sorting), shape: (pop_size,).
        rank[i] indicates the position of solution i in the sorted order
    front_no : np.ndarray
        Non-dominated front number of each solution, shape: (pop_size,)
    crowd_dis : np.ndarray
        Crowding distance of each solution, shape: (pop_size,)
    """
    pop_size = objs.shape[0]

    # Perform non-dominated sorting
    if cons is not None:
        front_no, _ = nd_sort(objs, cons, pop_size)
    else:
        front_no, _ = nd_sort(objs, pop_size)

    # Calculate crowding distance for diversity preservation
    crowd_dis = crowding_distance(objs, front_no)

    # Sort by front number (ascending), then by crowding distance (descending)
    sorted_indices = np.lexsort((-crowd_dis, front_no))

    # Create rank array: rank[i] gives the sorted position of solution i
    rank = np.empty(pop_size, dtype=int)
    rank[sorted_indices] = np.arange(pop_size)

    return rank, front_no, crowd_dis
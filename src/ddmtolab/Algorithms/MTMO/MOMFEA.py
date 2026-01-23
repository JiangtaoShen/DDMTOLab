"""
Multiobjective Multifactorial Evolutionary Algorithm (MOMFEA)

This module implements MOMFEA for multi-objective multi-task optimization with knowledge transfer.

References
----------
    [1] Abhishek Gupta, Yew-Soon Ong, and Liang Feng. "Multifactorial Evolution: Toward \
        Evolutionary Multitasking." IEEE Transactions on Evolutionary Computation, 20(3): 343-357, 2015.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.27
Version: 1.0
"""
import time
from tqdm import tqdm
from ddmtolab.Algorithms.STMO.NSGAII import nsga2_sort
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MOMFEA:
    """
    Multiobjective Multifactorial Evolutionary Algorithm for multi-objective multi-task optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, save_data=True, save_path='./Data',
                 name='momfea_test', disable_tqdm=True):
        """
        Initialize MOMFEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.3)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'momfea_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MOMFEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize population and evaluate for each task
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:

            # Perform NSGA-II sorting to get dominance ranks for each task
            rank = []
            for i in range(nt):
                rank_i, _, _ = nsga2_sort(objs[i], cons[i])
                rank.append(rank_i.copy())

            # Select parents using binary tournament selection
            pop_decs = []
            pop_objs = []
            pop_cons = []
            for i in range(nt):
                matingpool_i = tournament_selection(2, n, rank[i])
                pop_decs.append(decs[i][matingpool_i, :])
                pop_objs.append(objs[i][matingpool_i, :])
                pop_cons.append(cons[i][matingpool_i, :])

            # Transform populations to unified search space for knowledge transfer
            pop_decs, pop_objs, pop_cons = space_transfer(problem, pop_decs, pop_objs, pop_cons, type='uni')

            # Merge populations from all tasks into single arrays
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_cons, pop_sfs)

            off_decs = np.zeros_like(pop_decs)
            off_objs = np.zeros_like(pop_objs)
            off_cons = np.zeros_like(pop_cons)
            off_sfs = np.zeros_like(pop_sfs)

            # Randomly pair individuals for assortative mating
            shuffled_index = np.random.permutation(pop_decs.shape[0])

            for i in range(0, len(shuffled_index), 2):
                p1 = shuffled_index[i]
                p2 = shuffled_index[i + 1]
                sf1 = pop_sfs[p1].item()
                sf2 = pop_sfs[p2].item()

                # Cross-task transfer: crossover if same task or rmp condition met
                if sf1 == sf2 or np.random.rand() < self.rmp:
                    off_dec1, off_dec2 = crossover(pop_decs[p1, :], pop_decs[p2, :], mu=2)
                    off_decs[i, :] = off_dec1
                    off_decs[i + 1, :] = off_dec2
                    off_sfs[i] = np.random.choice([sf1, sf2])
                    off_sfs[i + 1] = sf1 if off_sfs[i] == sf2 else sf2
                else:
                    # No transfer: mutate within own task
                    off_dec1 = mutation(pop_decs[p1, :], mu=5)
                    off_dec2 = mutation(pop_decs[p2, :], mu=5)
                    off_decs[i, :] = off_dec1
                    off_decs[i + 1, :] = off_dec2
                    off_sfs[i] = sf1
                    off_sfs[i + 1] = sf2

                # Trim to task dimensionality and evaluate offspring
                task_idx1 = off_sfs[i].item()
                task_idx2 = off_sfs[i + 1].item()

                off_dec1_trimmed = off_decs[i, :dims[task_idx1]]
                off_dec2_trimmed = off_decs[i + 1, :dims[task_idx2]]

                off_objs[i, :], off_cons[i, :] = (
                    x[0] for x in evaluation_single(problem, off_dec1_trimmed, task_idx1, unified=True, fill_value=0.)
                )
                off_objs[i + 1, :], off_cons[i + 1, :] = (
                    x[0] for x in evaluation_single(problem, off_dec2_trimmed, task_idx2, unified=True, fill_value=0.)
                )

            # Merge parents and offspring populations
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups((pop_decs, off_decs), (pop_objs, off_objs),
                                                                  (pop_cons, off_cons), (pop_sfs, off_sfs))

            # Environmental selection: keep best n individuals per task
            pop_decs, objs, cons, pop_sfs = momfea_selection(pop_decs, pop_objs, pop_cons, pop_sfs, n, nt)

            # Transform back to native search space
            decs, objs, cons = space_transfer(problem, pop_decs, objs, cons, type='real')

            nfes += n * nt
            pbar.update(n * nt)

            append_history(all_decs, decs, all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=max_nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def momfea_selection(all_decs, all_objs, all_cons, all_sfs, n, nt):
    """
    Environmental selection for MOMFEA using NSGA-II criteria.

    Parameters
    ----------
    all_decs : np.ndarray
        Decision variable matrix of the combined population of shape (n_total, d_max)
    all_objs : np.ndarray
        Objective value matrix corresponding to all_decs of shape (n_total, n_obj)
    all_cons : np.ndarray
        Constraint value matrix corresponding to all_decs of shape (n_total, n_con)
    all_sfs : np.ndarray
        Skill factor array indicating task assignment for each individual of shape (n_total, 1)
    n : int
        Number of individuals to select per task (population size per task)
    nt : int
        Number of tasks in the multi-task optimization problem

    Returns
    -------
    pop_decs : list[np.ndarray]
        Selected decision variable matrices for each task, length nt, each of shape (n, d_max)
    pop_objs : list[np.ndarray]
        Selected objective value matrices for each task, length nt, each of shape (n, n_obj)
    pop_cons : list[np.ndarray]
        Selected constraint matrices for each task, length nt, each of shape (n, n_con)
    pop_sfs : list[np.ndarray]
        Selected skill factor arrays for each task, length nt, each of shape (n, 1)

    Notes
    -----
    Selection is performed independently for each task using NSGA-II sorting based on
    non-dominated rank and crowding distance. The top-n individuals with smallest rank
    values are retained for each task.
    """
    pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []

    # Process each task separately
    for i in range(nt):
        # Extract all individuals belonging to task i
        indices = np.where(all_sfs.flatten() == i)[0]
        current_decs, current_objs, current_cons, current_sfs = select_by_index(
            indices, all_decs, all_objs, all_cons, all_sfs
        )

        # NSGA-II sorting: rank based on non-dominated sorting and crowding distance
        rank, _, _ = nsga2_sort(current_objs, current_cons)

        # Select top-n individuals with smallest rank values
        indices_select = np.argsort(rank)[:n]
        selected_decs, selected_objs, selected_cons, selected_sfs = select_by_index(
            indices_select, current_decs, current_objs, current_cons, current_sfs
        )

        # Store selected individuals for this task
        pop_decs, pop_objs, pop_cons, pop_sfs = append_history(
            pop_decs, selected_decs,
            pop_objs, selected_objs,
            pop_cons, selected_cons,
            pop_sfs, selected_sfs
        )

    return pop_decs, pop_objs, pop_cons, pop_sfs
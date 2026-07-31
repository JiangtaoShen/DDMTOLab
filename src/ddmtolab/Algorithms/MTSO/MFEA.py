"""
Multifactorial Evolutionary Algorithm (MFEA)

This module implements MFEA for multi-task optimization with knowledge transfer across tasks.

References
----------
    [1] Abhishek Gupta, Yew-Soon Ong, and Liang Feng. "Multifactorial Evolution: Toward Evolutionary Multitasking." IEEE Transactions on Evolutionary Computation, 20(3): 343-357, 2015.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.12
Version: 1.0
"""
import time
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MFEA:
    """
    Multifactorial Evolutionary Algorithm for multi-task optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=2.0, mum=5.0, save_data=True, save_path='./Data',
                 name='MFEA', disable_tqdm=True):
        """
        Initialize Multifactorial Evolutionary Algorithm.

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
        muc : float, optional
            Distribution index for SBX crossover (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (default: 5.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MFEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Multifactorial Evolutionary Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize population and evaluate for each task
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform populations to unified search space for knowledge transfer. The
        # reference initialises the whole unified vector with rand(1, max(D)), so the
        # padded dimensions must be uniform random (not a constant). Constraints are
        # padded separately with zeros so that padding never fabricates a violation.
        pop_decs = space_transfer(problem=problem, decs=decs, type='uni', padding='random')
        _, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni', padding='zero')
        pop_objs = objs

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:

            # Merge populations from all tasks into single arrays
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_cons, pop_sfs)

            n_pop = pop_decs.shape[0]
            half = n_pop // 2
            n_pairs = int(np.ceil(n_pop / 2))

            off_decs = np.zeros((2 * n_pairs, pop_decs.shape[1]))
            off_objs = np.zeros((2 * n_pairs, pop_objs.shape[1]))
            off_cons = np.zeros((2 * n_pairs, pop_cons.shape[1]))
            off_sfs = np.zeros((2 * n_pairs, 1), dtype=int)

            # Randomly pair individuals for assortative mating: the reference pairs
            # order[i] with order[i + floor(N / 2)] for i = 1..ceil(N / 2)
            ind_order = np.random.permutation(n_pop)

            count = 0
            for i in range(n_pairs):
                p1 = ind_order[i]
                p2 = ind_order[i + half]
                sf1 = pop_sfs[p1].item()
                sf2 = pop_sfs[p2].item()
                parent_sfs = (sf1, sf2)

                # Cross-task transfer: crossover if same task or rmp condition met
                if sf1 == sf2 or np.random.rand() < self.rmp:
                    off_dec1, off_dec2 = crossover(pop_decs[p1, :], pop_decs[p2, :], mu=self.muc)
                    # Vertical cultural transmission: each child independently imitates
                    # the skill factor of one uniformly chosen parent
                    off_sfs[count] = parent_sfs[np.random.randint(2)]
                    off_sfs[count + 1] = parent_sfs[np.random.randint(2)]
                else:
                    # No transfer: mutate within own task
                    off_dec1 = mutation(pop_decs[p1, :], mu=self.mum)
                    off_dec2 = mutation(pop_decs[p2, :], mu=self.mum)
                    off_sfs[count] = sf1
                    off_sfs[count + 1] = sf2

                off_decs[count, :] = off_dec1
                off_decs[count + 1, :] = off_dec2
                count += 2

            # Evaluate every offspring on the task named by its skill factor
            for t in range(nt):
                idx_t = np.where(off_sfs.flatten() == t)[0]
                if idx_t.size == 0:
                    continue
                objs_t, cons_t = evaluation_single(problem, off_decs[idx_t][:, :dims[t]], t, unified=True)
                off_objs[idx_t, :] = objs_t[:, :off_objs.shape[1]]
                if off_cons.shape[1] > 0:
                    off_cons[idx_t, :] = cons_t[:, :off_cons.shape[1]]
                nfes += idx_t.size
                pbar.update(idx_t.size)

            # Merge parents and offspring populations
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(
                (pop_decs, off_decs), (pop_objs, off_objs), (pop_cons, off_cons), (pop_sfs, off_sfs)
            )

            # Environmental selection: keep best n individuals per task
            pop_decs, pop_objs, pop_cons, pop_sfs = mfea_selection(pop_decs, pop_objs, pop_cons, pop_sfs, n, nt)

            # Transform back to native search space
            decs, cons = space_transfer(problem, decs=pop_decs, cons=pop_cons, type='real')

            append_history(all_decs, decs, all_objs, pop_objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=max_nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def mfea_selection(all_decs, all_objs, all_cons, all_sfs, n, nt):
    """
    Environmental selection for MFEA based on elitist strategy.

    Parameters
    ----------
    all_decs : np.ndarray
        Decision variable matrix of the combined population of shape (n_total, d_max).
        Contains solutions from all tasks in unified search space
    all_objs : np.ndarray
        Objective value matrix corresponding to all_decs of shape (n_total, 1).
        Each individual has been evaluated on its assigned task
    all_sfs : np.ndarray
        Skill factor array indicating task assignment for each individual of shape (n_total, 1).
        Values range from 0 to nt-1
    n : int
        Number of individuals to select per task (population size per task)
    nt : int
        Number of tasks in the multi-task optimization problem

    Returns
    -------
    pop_decs : list[np.ndarray]
        Selected decision variable matrices for each task, length nt, each of shape (n, d_max)
    pop_objs : list[np.ndarray]
        Selected objective value matrices for each task, length nt, each of shape (n, 1)
    pop_sfs : list[np.ndarray]
        Selected skill factor arrays for each task, length nt, each of shape (n, 1)

    Notes
    -----
    Selection is performed independently for each task by selecting the top-n individuals
    with minimum objective values among those assigned to that task.
    """
    pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []

    # Process each task separately
    for i in range(nt):
        # Extract all individuals belonging to task i
        indices = np.where(all_sfs == i)[0]
        current_decs, current_objs, current_cons, current_sfs = select_by_index(indices, all_decs, all_objs,
                                                                                all_cons, all_sfs)

        # Select top-n individuals with minimum objective values
        indices_select = selection_elit(objs=current_objs, n=n, cons=current_cons)
        selected_decs, selected_objs, selected_cons, selected_sfs = select_by_index(indices_select, current_decs,
                                                                                    current_objs, current_cons, current_sfs)

        # Store selected individuals for this task
        pop_decs, pop_objs, pop_cons, pop_sfs = append_history(
            pop_decs, selected_decs,
            pop_objs, selected_objs,
            pop_cons, selected_cons,
            pop_sfs, selected_sfs
        )

    return pop_decs, pop_objs, pop_cons, pop_sfs
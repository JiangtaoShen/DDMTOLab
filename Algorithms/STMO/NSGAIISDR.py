"""
Nondominated Sorting Genetic Algorithm II with Strengthened Dominance Relation (NSGA-II-SDR)

This module implements NSGA-II-SDR for multi-objective optimization problems.

References
----------
    [1] Y. Tian, R. Cheng, X. Zhang, Y. Su, and Y. Jin. A strengthened dominance \
        relation considering convergence and diversity for evolutionary many- \
        objective optimization. IEEE Transactions on Evolutionary Computation, \
        2019, 23(2): 331-345.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.14
Version: 1.0
"""
from tqdm import tqdm
import time
from scipy.spatial.distance import cdist
from Methods.Algo_Methods.algo_utils import *


class NSGAIISDR:
    """
    Nondominated Sorting Genetic Algorithm II with Strengthened Dominance Relation for multi-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
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

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=15.0, save_data=True, save_path='./TestData',
                 name='NSGA-II-SDR_test', disable_tqdm=True):
        """
        Initialize NSGA-II-SDR algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'NSGA-II-SDR_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
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
        """
        Execute the NSGA-II-SDR algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
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

        # Perform initial non-dominated sorting for each task using SDR
        rank = []
        for i in range(nt):
            rank_i, _, _ = nsga2sdr_sort(objs[i])
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
                off_objs, _ = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs))

                # Environmental selection: sort and keep best n individuals using SDR
                rank[i], _, _ = nsga2sdr_sort(objs[i])
                index = np.argsort(rank[i])[:n_per_task[i]]
                objs[i], decs[i], rank[i] = select_by_index(index, objs[i], decs[i], rank[i])

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def nsga2sdr_sort(objs):
    """
    Sort solutions based on NSGA-II-SDR criteria using strengthened dominance relation and crowding distance.

    Parameters
    ----------
    objs : np.ndarray
        Objective value matrix of shape (pop_size, n_obj)

    Returns
    -------
    rank : np.ndarray
        Ranking of each solution (0-based index after sorting) of shape (pop_size,).
        rank[i] indicates the position of solution i in the sorted order
    front_no : np.ndarray
        Non-dominated front number of each solution of shape (pop_size,)
    crowd_dis : np.ndarray
        Crowding distance of each solution of shape (pop_size,)

    Notes
    -----
    Solutions are sorted first by front number (ascending), then by crowding distance (descending).
    Larger crowding distance values indicate better diversity preservation.
    The strengthened dominance relation (SDR) is used for non-dominated sorting to improve
    performance on many-objective optimization problems.
    """
    pop_size = objs.shape[0]

    # Perform non-dominated sorting using strengthened dominance relation
    front_no, _ = nd_sort_sdr(objs, pop_size)

    # Calculate crowding distance for diversity preservation
    crowd_dis = crowding_distance(objs, front_no)

    # Sort by front number (ascending), then by crowding distance (descending)
    sorted_indices = np.lexsort((-crowd_dis, front_no))

    # Create rank array: rank[i] gives the sorted position of solution i
    rank = np.empty(pop_size, dtype=int)
    rank[sorted_indices] = np.arange(pop_size)

    return rank, front_no, crowd_dis


def nd_sort_sdr(pop_obj: np.ndarray, n_sort: int) -> Tuple[np.ndarray, int]:
    """
    Do non-dominated sorting by strengthened dominance relation (SDR).

    The SDR enhances the traditional Pareto dominance relation by considering both
    convergence and diversity, making it more effective for many-objective optimization.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective value matrix, shape (N, M)
    n_sort : int
        Number of solutions to sort

    Returns
    -------
    front_no : np.ndarray
        Non-dominated front number for each solution, shape (N,)
    max_fno : int
        Maximum front number assigned

    References
    ----------
    Y. Tian, R. Cheng, X. Zhang, Y. Su, and Y. Jin. A strengthened dominance
    relation considering convergence and diversity for evolutionary many-
    objective optimization. IEEE Transactions on Evolutionary Computation,
    2019, 23(2): 331-345.
    """
    N = pop_obj.shape[0]

    # Min-Max Normalization
    obj_min = np.min(pop_obj, axis=0)
    obj_max = np.max(pop_obj, axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1
    pop_obj = (pop_obj - obj_min) / obj_range

    # Calculate L1-norm (sum) of each solution for convergence measure
    norm_p = np.sum(pop_obj, axis=1)

    # Calculate cosine similarity for diversity measure
    if N > 1:
        cosine = 1 - cdist(pop_obj, pop_obj, metric='cosine')
        np.fill_diagonal(cosine, 0)
    else:
        cosine = np.zeros((1, 1))

    # Calculate angle (in radians) between solution vectors
    angle = np.arccos(np.clip(cosine, -1, 1))

    # Find minimum angle threshold for strengthened dominance
    if N > 1:
        # Get minimum angle for each solution
        min_angle_per_sol = np.min(angle, axis=1)
        # Get unique sorted minimum angles
        unique_min_angles = np.sort(np.unique(min_angle_per_sol))
        # Select the middle value (or the ceil(N/2)-th smallest unique value)
        idx = min(np.ceil(N / 2).astype(int), len(unique_min_angles) - 1)
        minA = unique_min_angles[idx]
    else:
        minA = np.pi / 4  # Default for single solution

    # Calculate theta values for strengthened dominance relation
    Theta = np.maximum(1, (angle / minA) ** 1)

    # Build dominance matrix using strengthened dominance relation
    # Solution i strengthened-dominates solution j if: norm_p[i] * Theta[i,j] < norm_p[j]
    dominate = np.zeros((N, N), dtype=bool)

    for i in range(N - 1):
        for j in range(i + 1, N):
            if norm_p[i] * Theta[i, j] < norm_p[j]:
                dominate[i, j] = True
            elif norm_p[j] * Theta[j, i] < norm_p[i]:
                dominate[j, i] = True

    # Non-dominated sorting based on strengthened dominance
    front_no = np.full(N, np.inf)
    max_fno = 0

    while np.sum(front_no != np.inf) < min(n_sort, N):
        max_fno += 1
        # Find solutions not dominated by any others under SDR
        current = ~np.any(dominate, axis=0) & (front_no == np.inf)
        # Assign front number
        front_no[current] = max_fno
        # Remove current solutions from domination relationships
        dominate[current, :] = False

    return front_no, max_fno
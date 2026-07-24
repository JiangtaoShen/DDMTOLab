"""
Nondominated Sorting Genetic Algorithm II with Strengthened Dominance Relation (NSGA-II-SDR)

This module implements NSGA-II-SDR for multi-objective optimization problems.

References
----------
    [1] Y. Tian, R. Cheng, X. Zhang, Y. Su, and Y. Jin. A strengthened dominance relation considering convergence and diversity for evolutionary many-objective optimization. IEEE Transactions on Evolutionary Computation, 2019, 23(2): 331-345.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.14
Version: 1.1
"""
from tqdm import tqdm
import time
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class NSGA_II_SDR:
    """
    Nondominated Sorting Genetic Algorithm II with Strengthened Dominance Relation for multi-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=20.0, save_data=True, save_path='./Data',
                 name='NSGA-II-SDR', disable_tqdm=True):
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
            Distribution index for polynomial mutation (PM) (default: 20.0)
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

        # Per-task ideal/nadir point estimates and initial environmental selection
        zmin, zmax, front_no, crowd_dis = [], [], [], []
        for i in range(nt):
            zmin.append(np.min(objs[i], axis=0))
            zmax.append(np.max(objs[i], axis=0))
            objs[i], decs[i], front_i, crowd_i = sdr_environmental_selection(
                objs[i], decs[i], n_per_task[i], zmin[i], zmax[i])
            front_no.append(front_i)
            crowd_dis.append(crowd_i)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Parent selection via binary tournament on front number, then crowding distance
                # (PlatEMO: TournamentSelection(2,N,FrontNo,-CrowdDis))
                matingpool = platemo_tournament_selection(2, n_per_task[i], front_no[i], -crowd_dis[i])

                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i][matingpool, :], muc=self.muc, mum=self.mum)
                off_objs, _ = evaluation_single(problem, off_decs, i)

                # Update the ideal point with all offspring, and estimate the nadir
                # point from the first front of the current (pre-merge) population
                zmin[i] = np.minimum(zmin[i], np.min(off_objs, axis=0))
                zmax[i] = np.max(objs[i][front_no[i] == 1], axis=0)

                # Merge parent and offspring populations
                merged_objs, merged_decs = vstack_groups((objs[i], off_objs), (decs[i], off_decs))

                # Environmental selection with the strengthened dominance relation
                objs[i], decs[i], front_no[i], crowd_dis[i] = sdr_environmental_selection(
                    merged_objs, merged_decs, n_per_task[i], zmin[i], zmax[i])

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


def sdr_environmental_selection(objs, decs, N, zmin, zmax):
    """
    Environmental selection of NSGA-II-SDR (PlatEMO EnvironmentalSelection.m).

    The objectives are translated by the ideal point and, if the objective
    ranges are sufficiently balanced (0.05*max(range) < min(range)), scaled by
    the range. Near-duplicate solutions (equal after rounding the normalized
    objectives to 1e-6) are removed before SDR-based sorting, so fewer than N
    solutions may be returned.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (pop_size, M)
    decs : np.ndarray
        Decision variables, shape (pop_size, D)
    N : int
        Number of solutions to select (upper bound)
    zmin : np.ndarray
        Ideal point estimate, shape (M,)
    zmax : np.ndarray
        Nadir point estimate, shape (M,)

    Returns
    -------
    objs : np.ndarray
        Selected objective values
    decs : np.ndarray
        Selected decision variables
    front_no : np.ndarray
        SDR front number of each selected solution
    crowd_dis : np.ndarray
        Crowding distance (in normalized objective space) of each selected solution
    """
    # Normalization (conditional, as in PlatEMO)
    pop_obj = objs - zmin
    obj_range = zmax - zmin
    if 0.05 * np.max(obj_range) < np.min(obj_range):
        pop_obj = pop_obj / obj_range

    # Remove (near-)duplicate solutions; MATLAB's unique(...,'rows') keeps the
    # first occurrence of each duplicate and sorts the rows lexicographically
    _, keep = np.unique(np.round(pop_obj * 1e6) / 1e6, axis=0, return_index=True)
    pop_obj = pop_obj[keep]
    objs = objs[keep]
    decs = decs[keep]
    N = min(N, pop_obj.shape[0])

    # Non-dominated sorting with the strengthened dominance relation
    front_no, max_fno = sdr_sort_core(pop_obj, N)

    # Crowding distance in the normalized objective space
    crowd_dis = crowding_distance(pop_obj, front_no)

    # Select the solutions in the last front based on their crowding distances
    next_mask = front_no < max_fno
    last = np.where(front_no == max_fno)[0]
    order = np.argsort(-crowd_dis[last], kind='stable')
    n_needed = N - int(np.sum(next_mask))
    next_mask[last[order[:n_needed]]] = True

    return objs[next_mask], decs[next_mask], front_no[next_mask], crowd_dis[next_mask]


def sdr_sort_core(pop_obj: np.ndarray, n_sort: int) -> Tuple[np.ndarray, int]:
    """
    Non-dominated sorting by the strengthened dominance relation (SDR) on
    pre-normalized objectives (core of PlatEMO's NDSort_SDR.m).

    Parameters
    ----------
    pop_obj : np.ndarray
        (Normalized) objective value matrix, shape (N, M)
    n_sort : int
        Number of solutions to sort

    Returns
    -------
    front_no : np.ndarray
        Non-dominated front number for each solution, shape (N,)
    max_fno : int
        Maximum front number assigned
    """
    N = pop_obj.shape[0]

    # L1-norm (sum) of each solution as the convergence measure
    norm_p = np.sum(pop_obj, axis=1)

    # Pairwise angles between solution vectors (diagonal treated as pi/2)
    if N > 1:
        cosine = 1 - cdist(pop_obj, pop_obj, metric='cosine')
        np.fill_diagonal(cosine, 0)
    else:
        cosine = np.zeros((1, 1))
    angle = np.arccos(np.clip(cosine, -1, 1))

    # minA = temp(min(ceil(N/2),end)) in MATLAB (1-based indexing), with
    # temp = sort(unique(min(Angle,[],2))). The literal unique() is not
    # replicated here: solutions sharing a ray direction (common once the
    # population converges, e.g. identical position variables on DTLZ
    # problems) have an angle of exactly 0 in exact arithmetic, and
    # collapsing those duplicates would push the index to the largest
    # min-angle, making Theta=1 for most pairs and destroying diversity.
    # In MATLAB the same angles come out as distinct floating-point noise
    # (pdist2/acos round-off, complex acos for cosines slightly above 1),
    # so unique() barely collapses anything and minA stays at the
    # ceil(N/2)-th smallest value -- which the plain sort below reproduces.
    min_angles = np.sort(np.min(angle, axis=1))
    idx = min(int(np.ceil(N / 2)), min_angles.shape[0]) - 1
    minA = min_angles[idx]

    # Theta = max(1,(Angle./minA).^1); MATLAB's max(1,NaN) yields 1
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = angle / minA
    theta = np.where(np.isnan(ratio), 1.0, np.maximum(1.0, ratio))

    # Strengthened dominance: i dominates j if norm_p[i]*Theta[i,j] < norm_p[j]
    dominate = norm_p[:, None] * theta < norm_p[None, :]
    np.fill_diagonal(dominate, False)
    # Resolve mutual domination as in the MATLAB pairwise loop (i < j has priority)
    mutual = dominate & dominate.T
    dominate &= ~np.tril(mutual)

    # Front-by-front peeling
    front_no = np.full(N, np.inf)
    max_fno = 0
    while np.sum(front_no != np.inf) < min(n_sort, N):
        max_fno += 1
        current = ~np.any(dominate, axis=0) & (front_no == np.inf)
        front_no[current] = max_fno
        dominate[current, :] = False

    return front_no, max_fno


def nd_sort_sdr(pop_obj: np.ndarray, n_sort: int) -> Tuple[np.ndarray, int]:
    """
    SDR-based non-dominated sorting with internal min-max normalization.

    This matches the NDSort_SDR variant shipped with PlatEMO's PIEA, which
    normalizes the objectives by the population minimum and maximum before
    applying the strengthened dominance relation (NSGA-II-SDR itself
    normalizes in its environmental selection instead; see
    :func:`sdr_environmental_selection`).

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
    """
    obj_min = np.min(pop_obj, axis=0)
    obj_max = np.max(pop_obj, axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1
    normalized = (pop_obj - obj_min) / obj_range

    return sdr_sort_core(normalized, n_sort)


def platemo_tournament_selection(K, N, *fitness):
    """
    Exact port of PlatEMO's TournamentSelection.

    Candidates are compared lexicographically on the given fitness keys
    (lower values are better). Solutions with identical fitness values share
    the same rank, so a tournament among tied candidates is decided by the
    (random) draw order, i.e. uniformly at random. This differs from ranking
    with a composite total order, which would break ties deterministically.

    Parameters
    ----------
    K : int
        Tournament size
    N : int
        Number of parents to select
    *fitness : np.ndarray
        One or more fitness vectors of equal length (primary key first)

    Returns
    -------
    index : np.ndarray
        Indices of the selected parents, shape (N,)
    """
    fits = np.column_stack([np.asarray(f, dtype=float).ravel() for f in fitness])
    _, loc = np.unique(fits, axis=0, return_inverse=True)
    loc = loc.ravel()
    parents = np.random.randint(0, fits.shape[0], size=(K, N))
    best = np.argmin(loc[parents], axis=0)
    return parents[best, np.arange(N)]

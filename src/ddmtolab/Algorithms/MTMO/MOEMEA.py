"""
Multiobjective Evolutionary Multitasking via Explicit Autoencoding (MO-EMEA)

This module implements the MO_EMEA algorithm for multi-task multi-objective optimization problems with knowledge transfer.

References
----------
    [1] L. Feng, L. Zhou, J. Zhong, A. Gupta, Y. -S. Ong, K. -C. Tan, and A. K. Qin. "Evolutionary Multitasking via Explicit Autoencoding." IEEE Transactions on Cybernetics, 49(9): 3457-3470, 2019.

Notes
-----
The code is developed in accordance with the MATLAB-based MTO-platform framework.

Author: Jing Wang
Email:
Date: 2026.01.09
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MOEMEA:
    """
    Multi-task Multi-objective Evolutionary Multitasking via Explicit Autoencoding (MO_EMEA).

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
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, operator='SP/NS',
                 s_num=None, t_gap=None, mu_c=None, mu_m=None, save_data=True,
                 save_path='./Data', name='MO_EMEA_test', disable_tqdm=True):
        """
        Initialize MO-EMEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        operator : str, optional
            Selection operator(s) split with '/', e.g., 'SP/NS' (default: 'SP/NS')
            - 'SP': SPEA2 selection
            - 'NS': NSGA-II selection
        s_num : int, optional
            Number of solutions for knowledge transfer (default: 10)
        t_gap : int, optional
            Generation gap for knowledge transfer (default: 10)
        mu_c : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20)
        mu_m : float, optional
            Distribution index for polynomial mutation (PM) (default: 15)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MO_EMEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        # Common parameters
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        self.operator = operator
        self.s_num = s_num if s_num is not None else 10
        self.t_gap = t_gap if t_gap is not None else 10
        self.mu_c = mu_c if mu_c is not None else 20
        self.mu_m = mu_m if mu_m is not None else 15

        self.gen = 0
        self.operators = self.operator.split('/')
        self.nt = problem.n_tasks
        self.n_per_task = par_list(self.n, self.nt)
        self.max_nfes_per_task = par_list(self.max_nfes, self.nt)

    def optimize(self):
        """
        Execute MO_EMEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = self.nt
        n_per_task = self.n_per_task

        # 1. Initialize population
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = np.array(n_per_task).copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Save initial population (for knowledge transfer)
        init_pop_dec = []
        for t in range(nt):
            init_pop_dec.append(decs[t][:, :problem.dims[t]].copy())

        # 2. Initial selection (SPEA2)
        fitness = [None] * nt
        for t in range(nt):
            decs[t], fitness[t] = selection_spea2(decs[t], objs[t], cons[t], n_per_task[t])

        # 3. Progress bar settings
        total_nfes = sum(self.max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # 4. Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            self.gen += 1  # Update generation count
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < self.max_nfes_per_task[i]]
            if not active_tasks:
                break

            for t in active_tasks:
                # 4.1 Tournament Selection (generate mating pool)
                mating_pool_idx = tournament_selection(2, n_per_task[t], fitness[t])
                mating_pool_decs = decs[t][mating_pool_idx]

                # 4.2 Generate offspring via GA
                off_decs = ga_generation(mating_pool_decs, self.mu_c, self.mu_m)

                # 4.3 Knowledge Transfer
                if self.s_num > 0 and self.gen % self.t_gap == 0:
                    inject_decs = knowledge_transfer(
                        t, self.gen, self.s_num, self.t_gap, self.problem, init_pop_dec, decs
                    )
                    if inject_decs is not None and len(inject_decs) > 0:
                        # Randomly replace offspring with injected solutions
                        replace_idx = np.random.permutation(len(off_decs))[:len(inject_decs)]
                        off_decs[replace_idx] = inject_decs[:len(replace_idx)]

                # 4.4 Evaluate offspring
                off_objs, off_cons = evaluation_single(problem, off_decs, t)
                nfes_per_task[t] += len(off_decs)
                pbar.update(len(off_decs))

                # 4.5 Merge parents and offspring
                decs[t] = np.vstack([decs[t], off_decs])
                objs[t] = np.vstack([objs[t], off_objs])
                cons[t] = np.vstack([cons[t], off_cons]) if cons[t] is not None else off_cons

                # 4.6 Environmental Selection (Alternate between SP and NS)
                op_idx = (t - 1) % len(self.operators)
                op = self.operators[op_idx]

                if op == 'SP':  # SPEA2 selection
                    decs[t], fitness[t] = selection_spea2(decs[t], objs[t], cons[t], n_per_task[t])
                elif op == 'NS':  # NSGA-II selection
                    rank, front_no, crowd_dis = nsga2_sort(objs[t], cons[t])
                    sorted_indices = np.lexsort((-crowd_dis, front_no))
                    select_idx = sorted_indices[:n_per_task[t]]
                    decs[t] = decs[t][select_idx]
                    fitness[t] = np.arange(1, n_per_task[t] + 1)  # Simple assignment for compatibility

                # 4.7 Update history
                objs[t], cons[t] = evaluation_single(problem, decs[t], t)
                append_history(all_decs, decs, all_objs, objs, all_cons, cons)

        # 5. Process results
        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def knowledge_transfer(t, gen, s_num, t_gap, problem, init_pop_dec, pop_decs_all):
    """
    Knowledge transfer via marginal Denoising Autoencoder (mDA).

    Parameters
    ----------
    t : int
        Current task index (0-based)
    gen : int
        Current generation number
    s_num : int
        Number of solutions to transfer (total across tasks)
    t_gap : int
        Interval (generations) between transfer events
    problem : MTOP
        Multi-task optimization problem instance
    init_pop_dec : list of np.ndarray
        Initial population decision variables for all tasks
    pop_decs_all : list of np.ndarray
        Current population decision variables for all tasks

    Returns
    -------
    inject_decs : np.ndarray or None
        Injected decision variables for current task, or None if no transfer occurs
    """
    if s_num <= 0 or gen % t_gap != 0:
        return None

    inject_decs = []
    inject_num = np.round(s_num / (problem.n_tasks - 1)).astype(int)

    for k in range(problem.n_tasks):
        if k == t:
            continue  # Skip current task

        # Extract best solutions from source task k
        his_pop_dec = pop_decs_all[k]
        his_best_dec = his_pop_dec[:inject_num, :problem.dims[k]]  # Take top 'inject_num' best solutions

        # Transfer via marginal Denoising Autoencoder (mDA)
        inject = mda(
            init_pop_dec[t][:, :problem.dims[t]],
            init_pop_dec[k][:, :problem.dims[k]],
            his_best_dec
        )

        # Dimensionality completion
        n_inject, n_dims_inject = inject.shape
        if n_dims_inject < problem.dims[t]:
            # Randomly fill remaining dimensions
            rand_part = np.random.rand(n_inject, problem.dims[t] - n_dims_inject)
            inject = np.hstack([inject, rand_part])

        # Boundary handling
        inject = np.clip(inject, 0, 1)
        inject_decs.append(inject)

    # Merge all injected solutions
    if inject_decs:
        inject_decs = np.vstack(inject_decs)
        return inject_decs
    return None


def mda(curr_pop: np.ndarray,
        his_pop: np.ndarray,
        his_bestSolution: np.ndarray) -> np.ndarray:
    """
    Marginal Denoising Autoencoder (mDA) for knowledge transfer in evolutionary multitasking.

    Parameters
    ----------
    curr_pop : np.ndarray
        Current population from target domain (n_samples, d_curr)
        n_samples: number of individuals, d_curr: variable dimension of target domain
    his_pop : np.ndarray
        Population from source domain (n_samples, d_his)
        n_samples: number of individuals (same as curr_pop), d_his: variable dimension of source domain
    his_bestSolution : np.ndarray
        Best solutions from source domain (n_best, d_his)
        n_best: number of best solutions, d_his: variable dimension of source domain

    Returns
    -------
    inj_solution : np.ndarray
        Transformed solutions for target domain (n_best, d_curr)
    """
    # ========== Step 1: Get dimension information ==========
    # curr_pop: (n, d_curr), his_pop: (n, d_his)
    n_curr, curr_len = curr_pop.shape
    n_his, tmp_len = his_pop.shape

    # Ensure both populations have same sample size
    assert n_curr == n_his, f"curr_pop ({n_curr}) and his_pop ({n_his}) must have the same number of samples!"

    # ========== Step 2: Align dimensions by padding zeros ==========
    if curr_len < tmp_len:
        # Target dim < Source dim: Pad curr_pop with 0
        pad_width = ((0, 0), (0, tmp_len - curr_len))
        curr_pop = np.pad(curr_pop, pad_width, mode='constant', constant_values=0)
    elif curr_len > tmp_len:
        # Target dim > Source dim: Pad his_pop with 0
        pad_width = ((0, 0), (0, curr_len - tmp_len))
        his_pop = np.pad(his_pop, pad_width, mode='constant', constant_values=0)

    # ========== Step 3: Transpose and add bias term  ==========
    xx = curr_pop.T
    noise = his_pop.T

    d, n = xx.shape  # d: aligned dimension, n: number of samples
    # Add bias term (last row is all 1s)
    xxb = np.vstack([xx, np.ones((1, n))])
    noise_xb = np.vstack([noise, np.ones((1, n))])

    # ========== Step 4: Calculate weight matrix W ==========
    lambda_ = 1e-5  # Regularization parameter
    Q = noise_xb @ noise_xb.T
    P = xxb @ noise_xb.T

    # Regularization term: lambda * eye(d+1), set last element to 0
    reg = lambda_ * np.eye(d + 1)
    reg[-1, -1] = 0

    # W = P / (Q + reg) -> W = P * inv(Q + reg)
    W = P @ np.linalg.inv(Q + reg)

    # ========== Step 5: Remove bias term from weight matrix ==========
    W = W[:-1, :-1]

    # ========== Step 6: Transform best solutions to target domain ==========
    if curr_len <= tmp_len:
        # Target dim <= Source dim: Transform then crop to target dimension
        tmp_solution = (W @ his_bestSolution.T).T
        inj_solution = tmp_solution[:, :curr_len]
    elif curr_len > tmp_len:
        # Target dim > Source dim: Pad best solutions, then transform
        pad_width = ((0, 0), (0, curr_len - tmp_len))
        his_bestSolution_padded = np.pad(his_bestSolution, pad_width, mode='constant', constant_values=0)
        inj_solution = (W @ his_bestSolution_padded.T).T

    return inj_solution


def nsga2_sort(objs, cons=None):
    """
    Sort solutions based on NSGA-II criteria using non-dominated sorting and crowding distance.

    Parameters
    ----------
    objs : np.ndarray
        Objective value matrix of shape (pop_size, n_obj)
    cons : np.ndarray, optional
        Constraint matrix of shape (pop_size, n_con). If None, no constraints are considered (default: None)

    Returns
    -------
    rank : np.ndarray
        Ranking of each solution (0-based index after sorting) of shape (pop_size,).
        rank[i] indicates the position of solution i in the sorted order
    front_no : np.ndarray
        Non-dominated front number of each solution of shape (pop_size,)
    crowd_dis : np.ndarray
        Crowding distance of each solution of shape (pop_size,)
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


def selection_spea2(pop_decs: np.ndarray, pop_objs: np.ndarray, pop_cons: np.ndarray,
                    n: int, epsilon: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Environmental selection of SPEA2.

    Parameters
    ----------
    pop_decs : np.ndarray
        Population decision variables (n_individuals, n_dims)
    pop_objs : np.ndarray
        Population objective values (n_individuals, n_objs)
    pop_cons : np.ndarray
        Population constraint violation values (n_individuals, n_cons)
        If no constraints, pass an empty array or all zeros
    n : int
        Target population size for next generation
    epsilon : float, optional
        Epsilon constraint tolerance (default: 0.0)

    Returns
    -------
    selected_decs : np.ndarray
        Selected population decision variables (n, n_dims)
    fitness_values : np.ndarray
        Fitness values of selected population (n,)
    """
    # Check if critical inputs are empty/invalid
    if pop_decs is None or pop_objs is None or len(pop_objs) == 0 or len(pop_decs) == 0:
        # Return empty arrays to avoid None type errors
        return np.array([]), np.array([])
    # Ensure n is a valid value (non-negative, not exceeding population size)
    n = max(0, min(n, len(pop_objs)))
    if n == 0:
        return np.array([]), np.array([])

    # ========== Step 1: Calculate constraint violation (CV) ==========
    # Handle empty constraints (no constraints)
    if pop_cons.size == 0:
        cvs = np.zeros(len(pop_objs))
    else:
        cvs = np.sum(np.maximum(0, pop_cons), axis=1)
    cvs[cvs < epsilon] = 0

    # ========== Step 2: Calculate SPEA2 fitness ==========
    fitness = cal_fitness(pop_objs, cvs)

    # ========== Step 3: Environmental selection ==========
    # Initial selection: fitness < 1
    next_mask = fitness < 1  # Boolean mask (True/False)
    n_selected = np.sum(next_mask)

    # Case 1: Not enough solutions (select top N by fitness)
    if n_selected < n:
        # Sort fitness and select top N
        sorted_indices = np.argsort(fitness)
        next_mask = np.zeros_like(next_mask, dtype=bool)
        next_mask[sorted_indices[:n]] = True

    # Case 2: Too many solutions (truncation to N)
    elif n_selected > n:
        # Get indices of initially selected solutions
        selected_indices = np.where(next_mask)[0]
        # Get their objective values
        selected_objs = pop_objs[selected_indices]
        # Calculate truncation indices (to delete)
        del_indices = truncation(selected_objs, n_selected - n)
        # Update next_mask: set del_indices to False
        next_mask[selected_indices[del_indices]] = False

    # ========== Step 4: Select population and sort by fitness ==========
    # Apply selection mask
    selected_decs = pop_decs[next_mask]
    fitness_values = fitness[next_mask]

    # Sort population by fitness (ascending)
    sorted_fitness_indices = np.argsort(fitness_values)
    fitness_values = fitness_values[sorted_fitness_indices]
    selected_decs = selected_decs[sorted_fitness_indices]

    return selected_decs, fitness_values


def truncation(pop_obj: np.ndarray, k: int) -> np.ndarray:
    """
    Truncation operator for SPEA2.
    Select K solutions to delete by density estimation.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values of the population to truncate (n_individuals, n_objs)
    k : int
        Number of solutions to delete

    Returns
    -------
    del_indices : np.ndarray
        Indices of solutions to delete (k,)
    """
    n = len(pop_obj)
    del_mask = np.zeros(n, dtype=bool)  # Initial mask (all False)

    # Calculate pairwise distance matrix
    distance = cdist(pop_obj, pop_obj)
    # Set diagonal to infinity (distance to self = inf)
    np.fill_diagonal(distance, np.inf)

    # Iteratively delete the most crowded solution until K are deleted
    while np.sum(del_mask) < k:
        # Get indices of remaining solutions
        remain_indices = np.where(~del_mask)[0]
        # Get distance matrix of remaining solutions
        remain_dist = distance[remain_indices][:, remain_indices]
        # Sort distances for each solution (ascending)
        sorted_dist = np.sort(remain_dist, axis=1)
        # Sort rows by the sorted distances
        # Use lexsort (sort by first column, then second, etc.)
        sorted_rows_indices = np.lexsort(np.rot90(sorted_dist))
        # Select the first solution (most crowded) to delete
        del_idx_in_remain = sorted_rows_indices[0]
        del_mask[remain_indices[del_idx_in_remain]] = True

    # Return indices of deleted solutions
    del_indices = np.where(del_mask)[0]
    return del_indices


def cal_fitness(pop_obj: np.ndarray, pop_cv: np.ndarray = None) -> np.ndarray:
    """
    Calculate SPEA2 fitness values.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values (n_individuals, n_objs)
    pop_cv : np.ndarray, optional
        Constraint violation values (n_individuals,) (default: None -> no constraints)

    Returns
    -------
    fitness : np.ndarray
        SPEA2 fitness values (n_individuals,)
    """
    n = len(pop_obj)
    if n == 0:
        return np.array([])
    # Handle no constraint case
    if pop_cv is None:
        cv = np.zeros(n)
    else:
        cv = pop_cv.copy()

    # ========== Step 1: Detect dominance relations ==========
    dominate = np.zeros((n, n), dtype=bool)  # Dominance matrix (dominate[i,j] = i dominates j)

    for i in range(n):
        for j in range(i + 1, n):
            # Compare constraints first
            if cv[i] < cv[j]:
                dominate[i, j] = True
            elif cv[i] > cv[j]:
                dominate[j, i] = True
            else:
                # No constraint violation: compare objectives (minimization)
                obj_i = pop_obj[i]
                obj_j = pop_obj[j]
                has_better = np.any(obj_i < obj_j)
                has_worse = np.any(obj_i > obj_j)
                k = 1 if has_better and not has_worse else (-1 if has_worse and not has_better else 0)

                if k == 1:
                    dominate[i, j] = True
                elif k == -1:
                    dominate[j, i] = True

    # ========== Step 2: Calculate strength S(i) ==========
    # S(i) = number of solutions dominated by i (sum over columns)
    s = np.sum(dominate, axis=1)

    # ========== Step 3: Calculate raw fitness R(i) ==========
    # R(i) = sum of S(j) for all j that dominate i (sum S where dominate[:,i] is True)
    r = np.zeros(n)
    for i in range(n):
        dominating_indices = np.where(dominate[:, i])[0]
        r[i] = np.sum(s[dominating_indices])

    # ========== Step 4: Calculate density D(i) ==========
    # Pairwise distance matrix
    distance = cdist(pop_obj, pop_obj)
    np.fill_diagonal(distance, np.inf)  # Distance to self = inf
    # Sort distances for each solution
    distance_sorted = np.sort(distance, axis=1)
    # k-th nearest neighbor (floor(sqrt(N)))
    k_neighbor = int(np.floor(np.sqrt(n)))
    # D(i) = 1 / (distance to k-th neighbor + 2)
    d = 1.0 / (distance_sorted[:, k_neighbor] + 2)

    # ========== Step 5: Total fitness = R + D ==========
    fitness = r + d
    return fitness
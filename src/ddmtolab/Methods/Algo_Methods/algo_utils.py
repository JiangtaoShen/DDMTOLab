"""
This script contains commonly used components for implementing algorithms.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.18
Version: 1.0
"""
import numpy as np
import pickle
import os
from dataclasses import asdict
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from dataclasses import dataclass
import copy
from typing import Any, List, Tuple, Union, Optional


@dataclass
class Results:
    """
    Container for optimization results.

    :no-index:

    Attributes
    ----------
    best_decs : List[np.ndarray]
        Best decision variables for each task
    best_objs : List[np.ndarray]
        Best objective values for each task
    all_decs : List[List[np.ndarray]]
        Decision variables history for all tasks across generations
    all_objs : List[List[np.ndarray]]
        Objective values history for all tasks across generations
    runtime : float
        Total runtime in seconds
    max_nfes : List[int]
        Maximum function evaluations per task
    best_cons : Optional[List[np.ndarray]]
        Best constraint values for each task (None if unconstrained)
    all_cons : Optional[List[List[np.ndarray]]]
        Constraint values history for all tasks (None if unconstrained)
    bounds : Optional[List[np.ndarray]]
        Bounds for each task, where each element is a 2D array with shape (2, dim)
    """
    best_decs: List[np.ndarray]
    best_objs: List[np.ndarray]
    all_decs: List[List[np.ndarray]]
    all_objs: List[List[np.ndarray]]
    runtime: float
    max_nfes: List[int]
    best_cons: Optional[List[np.ndarray]] = None
    all_cons: Optional[List[List[np.ndarray]]] = None
    bounds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


def build_save_results(
        all_decs: List[List[np.ndarray]],
        all_objs: List[List[np.ndarray]],
        runtime: float,
        max_nfes: List[int],
        all_cons: Optional[List[List[np.ndarray]]] = None,
        bounds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        save_path: Optional[str] = None,
        filename: Optional[str] = None,
        save_data: bool = True,
        **kwargs
) -> Results:
    """
    Extract best solutions, build results, and optionally save to file.

    Automatically detects single-objective vs multi-objective tasks:

    - Single-objective (n_objs=1): returns the best individual
    - Multi-objective (n_objs>1): returns the entire final population (Pareto front)

    Parameters
    ----------
    all_decs : List[List[np.ndarray]]
        Decision variables history for all tasks.
        all_decs[i][g] has shape (n_samples, dim) for task i at generation g.
    all_objs : List[List[np.ndarray]]
        Objective values history for all tasks.
        all_objs[i][g] has shape (n_samples, n_objs) for task i at generation g.
    runtime : float
        Total runtime in seconds
    max_nfes : List[int]
        Maximum function evaluations per task
    all_cons : List[List[np.ndarray]], optional
        Constraint values history for all tasks (default: None)
    bounds : List[Tuple[np.ndarray, np.ndarray]], optional
        Bounds (lower, upper) for each task (default: None)
    save_path : str, optional
        Directory path where the results will be saved (default: None)
    filename : str, optional
        Name of the output file without extension (default: None)
    save_data : bool, optional
        Whether to save the data to file (default: True)
    **kwargs : dict
        Additional data to include in the saved file

    Returns
    -------
    results : Results
        Results object containing best solutions and optimization history
    """
    nt = len(all_decs)

    best_decs = []
    best_objs = []
    best_cons = [] if all_cons is not None else None

    for i in range(nt):
        last_gen_objs = all_objs[i][-1]
        last_gen_decs = all_decs[i][-1]
        last_gen_cons = all_cons[i][-1] if all_cons is not None else None

        n_objs = last_gen_objs.shape[1]

        if n_objs == 1:
            best_idx = np.argmin(last_gen_objs[:, 0])
            best_objs.append(last_gen_objs[best_idx])
            best_decs.append(last_gen_decs[best_idx])
            if all_cons is not None:
                best_cons.append(last_gen_cons[best_idx])
        else:
            best_objs.append(last_gen_objs)
            best_decs.append(last_gen_decs)
            if all_cons is not None:
                best_cons.append(last_gen_cons)

    results = Results(
        best_decs=best_decs,
        best_objs=best_objs,
        all_decs=all_decs,
        all_objs=all_objs,
        runtime=runtime,
        max_nfes=max_nfes,
        best_cons=best_cons,
        all_cons=all_cons,
        bounds=bounds,
    )

    # Save results to file if path and filename are provided
    if save_data and save_path is not None and filename is not None:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{filename}.pkl")
        data_dict = asdict(results)
        data_dict.update(kwargs)
        with open(full_path, 'wb') as f:
            pickle.dump(data_dict, f)

    return results


def get_algorithm_information(algorithm_class: type, print_info: bool = True) -> dict:
    """
    Get algorithm information from any algorithm class.

    Parameters
    ----------
    algorithm_class : type
        Algorithm class with 'algorithm_information' attribute
    print_info : bool, optional
        Whether to print the information (default: True)

    Returns
    -------
    algo_info : dict
        Algorithm information dictionary
    """
    if not hasattr(algorithm_class, 'algorithm_information'):
        raise AttributeError(f"{algorithm_class.__name__} does not have 'algorithm_information' attribute")

    algo_info = algorithm_class.algorithm_information
    algo_name = algorithm_class.__name__

    information = '\n'.join([f"  - {k}: {v}" for k, v in algo_info.items()])
    info = f"🤖️ {algo_name} \nAlgorithm Information:\n{information}"

    if print_info:
        print(info)

    return algo_info


def init_history(
        decs: List[np.ndarray],
        objs: List[np.ndarray],
        cons: Optional[List[np.ndarray]] = None
) -> Union[tuple[List[List[np.ndarray]], List[List[np.ndarray]]],
tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]]:
    """
    Initialize history storage for populations across generations.

    Parameters
    ----------
    decs : List[np.ndarray]
        Initial decision variables for each task.
        decs[i] has shape (n_samples, dim) for task i.
    objs : List[np.ndarray]
        Initial objective values for each task.
        objs[i] has shape (n_samples, n_objs) for task i.
    cons : List[np.ndarray], optional
        Initial constraint values for each task (default: None).
        cons[i] has shape (n_samples, n_cons) for task i.

    Returns
    -------
    all_decs : List[List[np.ndarray]]
        History storage for decision variables
    all_objs : List[List[np.ndarray]]
        History storage for objective values
    all_cons : List[List[np.ndarray]], optional
        History storage for constraint values (only returned if cons is not None)
    """
    all_decs = [[d.copy()] for d in decs]
    all_objs = [[o.copy()] for o in objs]

    if cons is not None:
        all_cons = [[c.copy()] for c in cons]
        return all_decs, all_objs, all_cons

    return all_decs, all_objs


def vstack_groups(*args: Union[List[np.ndarray], tuple[np.ndarray, ...], None]) -> Union[
    np.ndarray, List[np.ndarray]]:
    """
    Stack population arrays vertically.

    Supports both single arrays (list of arrays) and tuples with variable number of arrays.

    Parameters
    ----------
    *args : Union[List[np.ndarray], tuple[np.ndarray, ...], None]
        Variable number of arguments, each can be:

        - List of arrays to stack vertically
        - Tuple of arrays (any number) to stack
        - None (will be skipped)

    Returns
    -------
    results : Union[np.ndarray, List[np.ndarray]]
        Stacked array if single input, or list of stacked arrays if multiple inputs
    """
    results = []
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, tuple):
            results.append(np.vstack(arg))
        else:
            results.append(np.vstack(arg))

    return results if len(results) > 1 else results[0]


def append_history(*pairs: Any) -> Tuple[list, ...]:
    """
    Append current generation data to history storage.

    Parameters
    ----------
    *pairs : tuple
        Alternating pairs of (history_list, current_data).

        - history_list: List to store historical data
        - current_data: Either a single np.ndarray (single task) or List[np.ndarray] (multi-task)

    Returns
    -------
    results : tuple
        All updated history lists (all_1, all_2, ...)
    """
    results = []

    for i in range(0, len(pairs), 2):
        all_list, data = pairs[i], pairs[i + 1]

        # Single task: input is a single array
        if isinstance(data, np.ndarray):
            all_list.append(data.copy())
        # Multi-task: input is a list of arrays
        else:
            for j in range(len(data)):
                all_list[j].append(data[j].copy())

        results.append(all_list)

    return tuple(results)


def select_by_index(index: np.ndarray, *arrays: Optional[np.ndarray]) -> Union[np.ndarray, List[Optional[np.ndarray]]]:
    """
    Select rows from arrays by index.

    Parameters
    ----------
    index : np.ndarray
        Indices to select, shape (n_selected,)
    *arrays : np.ndarray or None
        Variable number of arrays to select from.
        Each array has shape (n_samples,) or (n_samples, dim).
        None values are passed through unchanged.

    Returns
    -------
    results : Union[np.ndarray, List[Optional[np.ndarray]]]
        Selected array if single input, or list of selected arrays if multiple inputs.
        None inputs return None in the corresponding position.
    """
    results = []
    for arr in arrays:
        if arr is None:  # must be checked before accessing arr.ndim
            results.append(None)
        elif arr.ndim == 1:
            results.append(arr[index])
        else:
            results.append(arr[index, :])

    return results if len(results) > 1 else results[0]


def par_list(par: Union[int, List[int]], n_tasks: int) -> List[int]:
    """
    Convert a parameter to a list for multi-task scenarios.

    Parameters
    ----------
    par : Union[int, List[int]]
        Parameter value(s) - can be a single integer or a list
    n_tasks : int
        Number of tasks

    Returns
    -------
    par_per_task : List[int]
        List of parameter values, one for each task.

        - If par is int: returns [par, par, ..., par] (n_tasks times)
        - If par is list: returns the list as is
    """
    if isinstance(par, int):
        par_per_task = [par] * n_tasks
    else:
        par_per_task = list(par)
    return par_per_task


def build_staircase_history(
        db_decs: List[np.ndarray],
        db_objs: List[np.ndarray],
        k: int = 1,
        db_cons: Optional[List[np.ndarray]] = None
) -> Union[
    Tuple[List[List[np.ndarray]], List[List[np.ndarray]]],
    Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]
]:
    """
    Convert flat database lists into staircase history structure.

    For each task, creates an incrementally growing sequence of generations
    where generation g contains min(k * (g+1), total) solutions from the database.

    Parameters
    ----------
    db_decs : List[np.ndarray]
        Database of all decision variables per task. db_decs[i] has shape (n_i, dim_i).
    db_objs : List[np.ndarray]
        Database of all objective values per task. db_objs[i] has shape (n_i, n_objs).
    k : int, optional
        Step size controlling the staircase gradient (default: 1).
        Generation g contains min(k * (g+1), total) solutions.
    db_cons : List[np.ndarray], optional
        Database of all constraint values per task (default: None).

    Returns
    -------
    all_decs : List[List[np.ndarray]]
        Staircased decision variable history.
    all_objs : List[List[np.ndarray]]
        Staircased objective value history.
    all_cons : List[List[np.ndarray]], optional
        Staircased constraint history (only returned if db_cons is not None).
    """
    nt = len(db_decs)
    all_decs = []
    all_objs = []
    all_cons = [] if db_cons is not None else None

    for i in range(nt):
        n = db_decs[i].shape[0]
        task_decs = []
        task_objs = []
        task_cons = [] if db_cons is not None else None

        size = k
        while size < n:
            task_decs.append(db_decs[i][:size].copy())
            task_objs.append(db_objs[i][:size].copy())
            if db_cons is not None:
                task_cons.append(db_cons[i][:size].copy())
            size += k

        # Final generation with all points
        task_decs.append(db_decs[i][:n].copy())
        task_objs.append(db_objs[i][:n].copy())
        if db_cons is not None:
            task_cons.append(db_cons[i][:n].copy())

        all_decs.append(task_decs)
        all_objs.append(task_objs)
        if db_cons is not None:
            all_cons.append(task_cons)

    if db_cons is not None:
        return all_decs, all_objs, all_cons
    return all_decs, all_objs


def reorganize_initial_data(
        data: List[np.ndarray],
        nt: int,
        n_initial_per_task: List[int],
        interval: int = 1
) -> List[List[np.ndarray]]:
    """
    Reorganize initial data by task and number of initial points.

    Parameters
    ----------
    data : List[np.ndarray]
        Original data list, where data[i] is the data array for task i
    nt : int
        Number of tasks
    n_initial_per_task : List[int]
        List of number of initial points for each task
    interval : int, optional
        Interval for selecting points. Default is 1.
        - interval=1: 1, 2, 3, 4, ... points
        - interval=2: 2, 4, 6, 8, ... points
        - interval=k: k, 2k, 3k, 4k, ... (plus remaining points if not divisible)

    Returns
    -------
    all_data : List[List[np.ndarray]]
        Reorganized data
    """
    all_data = []
    for i in range(nt):
        task_data = []
        n = n_initial_per_task[i]
        # Store points at each interval
        for j in range(interval, n + 1, interval):
            task_data.append(data[i][:j].copy())
        # Add remaining points if not divisible
        if n % interval != 0:
            task_data.append(data[i][:n].copy())
        all_data.append(task_data)
    return all_data


def initialization(
        problem: 'MTOP',
        n: Union[int, List[int]],
        method: str = 'random',
        the_same: bool = False
) -> List[np.ndarray]:
    """
    Initialize decision variable matrices for multiple tasks.

    Parameters
    ----------
    problem : MTOP
        An instance of the MTOP class
    n : Union[int, List[int]]
        Number of samples per task.

        - If int: same number of samples for all tasks
        - If list: number of samples for each task, e.g., [30, 50]
    method : str, optional
        Sampling method: 'random' or 'lhs' (default: 'random')
    the_same : bool, optional
        If True, all tasks share the same sample points (default: False).
        For tasks with different dimensions, samples are generated in the
        maximum dimension and then truncated to each task's dimension.

    Returns
    -------
    decs : List[np.ndarray]
        List of decision variable matrices for each task.
        decs[i] has shape (n_i, d_i) for task i.
    """
    d = problem.dims
    nt = problem.n_tasks

    # Handle n: convert to array if it's an integer
    if isinstance(n, int):
        n_per_task = [n] * nt
    else:
        n_per_task = list(n)
        if len(n_per_task) != nt:
            raise ValueError(f"Length of n array ({len(n_per_task)}) must match number of tasks ({nt})")

    decs = []

    if the_same:
        # Generate samples in the maximum dimension and truncate to each task's dimension
        max_dim = max(d)
        max_n = max(n_per_task)

        if method.lower() == 'lhs':
            # Generate LHS samples in maximum dimension
            matrix_full = np.zeros((max_n, max_dim))
            for j in range(max_dim):
                intervals = np.linspace(0, 1, max_n + 1)
                samples = np.random.uniform(intervals[:-1], intervals[1:])
                np.random.shuffle(samples)
                matrix_full[:, j] = samples
        else:
            # Generate random samples in maximum dimension
            matrix_full = np.random.rand(max_n, max_dim)

        # Truncate to each task's sample size and dimension
        for i in range(nt):
            n_samples = n_per_task[i]
            dim = d[i]
            decs.append(matrix_full[:n_samples, :dim])

    else:
        # Generate independent samples for each task
        for i in range(nt):
            n_samples = n_per_task[i]

            if method.lower() == 'lhs':
                dim = d[i]
                matrix = np.zeros((n_samples, dim))
                for j in range(dim):
                    intervals = np.linspace(0, 1, n_samples + 1)
                    samples = np.random.uniform(intervals[:-1], intervals[1:])
                    np.random.shuffle(samples)
                    matrix[:, j] = samples
            else:
                matrix = np.random.rand(n_samples, d[i])

            decs.append(matrix)

    return decs


def evaluation(
        problem,
        decs: List[np.ndarray],
        unified: bool = False,
        fill_value: float = 0.0,
        eval_objectives: Union[bool, List[Union[bool, int, List[int]]]] = True,
        eval_constraints: Union[bool, List[Union[bool, int, List[int]]]] = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Evaluate a list of decision variable matrices on multiple tasks.

    Parameters
    ----------
    problem : MTOP
        An instance of the MTOP class.
    decs : list of ndarray
        List of decision variable matrices for each task, shape [n, d_i], scaled in [0,1].
    unified : bool, optional
        If True, pad objectives to m_max and constraints to c_max. Default False.
    fill_value : float, optional
        Value used for padding in unified mode. Default 0.0.
    eval_objectives : bool or list, optional
        - True: evaluate all objectives for all tasks (default)
        - False: skip objective evaluation for all tasks
        - List: per-task specification, each element can be:

            - True/False: evaluate all/none
            - int: evaluate only the i-th objective
            - List[int]: evaluate specified objectives
    eval_constraints : bool or list, optional
        - True: evaluate all constraints for all tasks (default)
        - False: skip constraint evaluation for all tasks
        - List: per-task specification, same format as eval_objectives

    Returns
    -------
    objs : list of ndarray
        List of objective value matrices for each task.

        - Normal mode: shape [n, m_i] or [n, len(selected)]
        - Unified mode: shape [n, m_max]
    cons : list of ndarray
        List of constraint value matrices for each task.

        - Normal mode: shape [n, c_i] or [n, 1] if no constraints
        - Unified mode: shape [n, c_max]
    """
    nt = problem.n_tasks
    bounds = problem.bounds

    # Expand eval_objectives/eval_constraints to per-task lists
    if not isinstance(eval_objectives, list) or len(eval_objectives) != nt:
        obj_modes = [eval_objectives] * nt
    else:
        obj_modes = eval_objectives

    if not isinstance(eval_constraints, list) or len(eval_constraints) != nt:
        con_modes = [eval_constraints] * nt
    else:
        con_modes = eval_constraints

    # Get max dimensions for unified mode
    m_max = problem.m_max if unified else None
    c_max = problem.c_max if unified else None

    objs = []
    cons = []

    for i in range(nt):
        # Scale decision variables from [0,1] to real bounds
        lb, ub = bounds[i]
        decs_real = decs[i] * (ub - lb) + lb

        # Evaluate task
        objectives, constraints = problem.evaluate_task(
            i, decs_real,
            eval_objectives=obj_modes[i],
            eval_constraints=con_modes[i]
        )

        n_samples = decs[i].shape[0]
        n_con = problem.get_n_constraints(i)

        # Handle no constraints case
        if con_modes[i] is not False and n_con == 0:
            constraints = np.zeros((n_samples, 1), dtype=np.float64)

        # Apply unified mode padding
        if unified:
            objectives = _pad_to_size(objectives, m_max, fill_value)
            constraints = _pad_to_size(constraints, c_max, fill_value)

        objs.append(objectives)
        cons.append(constraints)

    return objs, cons


def evaluation_single(
        problem,
        decs: np.ndarray,
        index: int,
        unified: bool = False,
        fill_value: float = 0.0,
        eval_objectives: Union[bool, int, List[int]] = True,
        eval_constraints: Union[bool, int, List[int]] = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate decision variables on a specific task.

    Parameters
    ----------
    problem : MTOP
        An instance of the MTOP class.
    decs : ndarray
        Decision variable matrix for the task, shape [n, d], scaled in [0,1].
    index : int
        Index of the task to evaluate.
    unified : bool, optional
        If True, pad objectives to m_max and constraints to c_max. Default False.
    fill_value : float, optional
        Value used for padding in unified mode. Default 0.0.
    eval_objectives : bool, int, or list of int, optional
        - True: evaluate all objectives (default)
        - False: skip objective evaluation
        - int: evaluate only the i-th objective
        - List[int]: evaluate specified objectives
    eval_constraints : bool, int, or list of int, optional
        - True: evaluate all constraints (default)
        - False: skip constraint evaluation
        - int: evaluate only the i-th constraint
        - List[int]: evaluate specified constraints

    Returns
    -------
    objs : ndarray
        Objective values for the task.

        - Normal mode: shape [n, m] or [n, len(selected)]
        - Unified mode: shape [n, m_max]
    cons : ndarray
        Constraint values for the task.

        - Normal mode: shape [n, c] or [n, 1] if no constraints
        - Unified mode: shape [n, c_max]
    """
    decs = np.atleast_2d(decs)
    n_samples = decs.shape[0]

    # Scale decision variables from [0,1] to real bounds
    bounds = problem.bounds
    lb, ub = bounds[index]
    decs_real = decs * (ub - lb) + lb

    # Evaluate task
    objs, cons = problem.evaluate_task(
        index, decs_real,
        eval_objectives=eval_objectives,
        eval_constraints=eval_constraints
    )

    n_con = problem.get_n_constraints(index)

    # Handle no constraints case
    if eval_constraints is not False and n_con == 0:
        cons = np.zeros((n_samples, 1), dtype=np.float64)

    # Apply unified mode padding
    if unified:
        m_max = problem.m_max
        c_max = problem.c_max
        objs = _pad_to_size(objs, m_max, fill_value)
        cons = _pad_to_size(cons, c_max, fill_value)

    return objs, cons


def _pad_to_size(arr: np.ndarray, target_size: int, fill_value: float) -> np.ndarray:
    """
    Pad array to target size along axis 1.

    Parameters
    ----------
    arr : np.ndarray
        Input array to pad, shape (n_samples, dim)
    target_size : int
        Target size for axis 1
    fill_value : float
        Value to use for padding

    Returns
    -------
    padded_arr : np.ndarray
        Padded array, shape (n_samples, target_size).
        Returns original array if already at or above target size.
    """
    if arr.shape[1] >= target_size:
        return arr
    pad_width = target_size - arr.shape[1]
    return np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=fill_value)


def crossover(
        par_dec1: np.ndarray,
        par_dec2: np.ndarray,
        mu: float = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Simulated Binary Crossover (SBX) on two decision vectors.

    Parameters
    ----------
    par_dec1 : np.ndarray
        First parent decision vector, values scaled in [0, 1], shape (dim,)
    par_dec2 : np.ndarray
        Second parent decision vector, values scaled in [0, 1], shape (dim,)
    mu : float, optional
        Distribution index for crossover (default: 2)

    Returns
    -------
    off_dec1 : np.ndarray
        First offspring decision vector, values clipped to [0, 1], shape (dim,)
    off_dec2 : np.ndarray
        Second offspring decision vector, values clipped to [0, 1], shape (dim,)
    """
    d = len(par_dec1)
    u = np.random.rand(d)
    beta = np.zeros(d)
    mask = u <= 0.5
    beta[mask] = (2 * u[mask]) ** (1 / (mu + 1))
    beta[~mask] = (2 * (1 - u[~mask])) ** (-1 / (mu + 1))
    beta *= (-1) ** np.random.randint(0, 2, size=d)
    beta[np.random.rand(d) < 0.5] = 1

    off_dec1 = 0.5 * ((1 + beta) * par_dec1 + (1 - beta) * par_dec2)
    off_dec2 = 0.5 * ((1 + beta) * par_dec2 + (1 - beta) * par_dec1)
    off_dec1 = np.clip(off_dec1, 0, 1)
    off_dec2 = np.clip(off_dec2, 0, 1)
    return off_dec1, off_dec2


def mutation(dec: np.ndarray, mu: float = 5) -> np.ndarray:
    """
    Perform polynomial mutation on a decision vector.

    Parameters
    ----------
    dec : np.ndarray
        Parent decision vector, values scaled in [0, 1], shape (dim,)
    mu : float, optional
        Distribution index for mutation (default: 5)

    Returns
    -------
    mutated_dec : np.ndarray
        Mutated decision vector, values clipped to [0, 1], shape (dim,)
    """
    d = len(dec)
    mutated_dec = dec.copy()
    prob_m = 1 / d
    for i in range(d):
        if np.random.rand() < prob_m:
            u = np.random.rand()
            if u <= 0.5:
                delta = ((2 * u + (1 - 2 * u) * (1 - mutated_dec[i]) ** (mu + 1))) ** (1 / (mu + 1)) - 1
                mutated_dec[i] += delta
            else:
                delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * mutated_dec[i] ** (mu + 1)) ** (1 / (mu + 1))
                mutated_dec[i] += delta
    mutated_dec = np.clip(mutated_dec, 0, 1)
    return mutated_dec


def ga_generation(parents: np.ndarray, muc: float, mum: float) -> np.ndarray:
    """
    Generate offspring population using genetic algorithm operators.

    Applies simulated binary crossover (SBX) and polynomial mutation
    to create offspring from parent population.

    Parameters
    ----------
    parents : np.ndarray
        Parent population, shape (n, d)
    muc : float
        Distribution index for crossover
    mum : float
        Distribution index for mutation

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (n, d)
    """
    n, d = parents.shape
    offdecs = np.zeros((0, d))
    # Copy before shuffling: shuffling the caller's population in place would
    # desynchronize decision rows from their objective/constraint rows
    parents = parents.copy()
    np.random.shuffle(parents)
    num_pairs = n // 2

    # Process pairs
    for j in range(num_pairs):
        offdec1, offdec2 = crossover(parents[j, :], parents[num_pairs + j, :], mu=muc)
        offdec1 = mutation(offdec1, mu=mum)
        offdec2 = mutation(offdec2, mu=mum)
        offdecs = np.vstack((offdecs, offdec1, offdec2))

    # Handle odd number of parents
    if n % 2 == 1:
        last_parent = parents[-1, :]
        random_idx = np.random.randint(0, n - 1)
        offdec1, _ = crossover(last_parent, parents[random_idx, :], mu=muc)
        offdec1 = mutation(offdec1, mu=mum)
        offdecs = np.vstack((offdecs, offdec1))

    return offdecs


def de_generation(parents: np.ndarray, F: float, CR: float) -> np.ndarray:
    """
    Generate offspring for a population using Differential Evolution (DE).

    Uses DE/rand/1/bin strategy: random base vector, one difference vector,
    and binomial crossover.

    Parameters
    ----------
    parents : np.ndarray
        Array of parent solutions, shape (n, d)
    F : float
        Differential weight (mutation scale factor)
    CR : float
        Crossover rate in [0, 1] for binomial crossover

    Returns
    -------
    offdecs : np.ndarray
        Offspring array, shape (n, d), clipped to [0, 1]
    """
    n, d = parents.shape
    offdecs = np.zeros((n, d), dtype=float)

    for j in range(n):
        # Choose 3 distinct indices != j
        indices = np.arange(n)
        indices = indices[indices != j]
        a, b, c = np.random.choice(indices, 3, replace=False)
        x_1, x_2, x_3 = parents[a], parents[b], parents[c]

        # DE mutation (rand/1)
        mutant = x_1 + F * (x_2 - x_3)

        # Binomial crossover (ensure at least one gene from mutant)
        trial = mutant.copy()
        replace_mask = np.random.rand(d) > CR
        replace_mask[np.random.randint(d)] = False
        trial[replace_mask] = parents[j][replace_mask]

        # Clip to bounds
        offdecs[j] = np.clip(trial, 0.0, 1.0)

    return offdecs


def space_transfer(
        problem: 'MTOP',
        decs: List[np.ndarray],
        objs: Optional[List[np.ndarray]] = None,
        cons: Optional[List[np.ndarray]] = None,
        type: str = 'real',
        padding: str = 'zero'
) -> Union[List[np.ndarray],
Tuple[List[np.ndarray], List[np.ndarray]],
Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    """
    Transfer decision variables, objectives, and constraints between unified and real spaces.

    Parameters
    ----------
    problem : MTOP
        An instance of the MTOP class containing task configuration
    decs : List[np.ndarray]
        List of decision variable matrices.

        - Real space: decs[i] has shape (n_i, d_i)
        - Unified space: decs[i] has shape (n_i, d_max)
    objs : List[np.ndarray], optional
        List of objective value matrices (default: None).

        - Real space: objs[i] has shape (n_i, m_i)
        - Unified space: objs[i] has shape (n_i, m_max)
    cons : List[np.ndarray], optional
        List of constraint value matrices (default: None).

        - Real space: cons[i] has shape (n_i, c_i)
        - Unified space: cons[i] has shape (n_i, c_max)
    type : str, optional
        Transfer type (default: 'real'):

        - 'uni': Pad matrices with zeros or random values to the maximum dimension (Unified Space)
        - 'real': Truncate matrices back to their original dimensions (Real Space)
    padding : str, optional
        Padding strategy when type='uni' (default: 'zero'):

        - 'zero': Pad with zeros
        - 'random': Pad with random values uniformly distributed in [0, 1]

    Returns
    -------
    new_decs : List[np.ndarray]
        Returned if objs and cons are None
    (new_decs, new_objs) : tuple[List[np.ndarray], List[np.ndarray]]
        Returned if objs is provided but cons is None
    (new_decs, new_objs, new_cons) : tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        Returned if both objs and cons are provided

    Notes
    -----
    The padding parameter only affects the 'uni' transfer type. When type='real',
    padding is ignored as truncation is performed instead.

    Fix (2025.12.22): Properly handles cases where some tasks have constraints and others don't.
    When ncons[i] = 0, ensures consistent shape (n, c_max) in unified space.
    """
    # Copy lists to avoid modifying the original input references
    new_decs = [d.copy() for d in decs]
    new_objs = [o.copy() for o in objs] if objs is not None else None
    new_cons = [c.copy() for c in cons] if cons is not None else None

    n_tasks = problem.n_tasks
    dims = problem.dims
    nobjs = problem.n_objs
    ncons = problem.n_cons

    if type == 'uni':
        # Handle Decision Variables (Padding)
        d_max = np.max(dims)
        for i in range(n_tasks):
            n_samples = new_decs[i].shape[0]
            dif = d_max - dims[i]
            if dif > 0:
                if padding == 'random':
                    pad_values = np.random.rand(n_samples, dif)
                elif padding == 'mid':
                    pad_values = np.full((n_samples, dif), 0.5)
                else:  # padding == 'zero'
                    pad_values = np.zeros((n_samples, dif))
                new_decs[i] = np.hstack([new_decs[i], pad_values])

        # Handle Objectives (Padding)
        if new_objs is not None:
            m_max = np.max(nobjs)
            for i in range(n_tasks):
                n_samples = new_objs[i].shape[0]
                dif = m_max - nobjs[i]
                if dif > 0:
                    if padding == 'random':
                        pad_values = np.random.rand(n_samples, dif)
                    elif padding == 'mid':
                        pad_values = np.full((n_samples, dif), 0.5)
                    else:  # padding == 'zero'
                        pad_values = np.zeros((n_samples, dif))
                    new_objs[i] = np.hstack([new_objs[i], pad_values])

        # Handle Constraints (Padding) - FIXED VERSION
        if new_cons is not None:
            c_max = np.max(ncons)
            if c_max > 0:  # Only process if at least one task has constraints
                for i in range(n_tasks):
                    n_samples = new_cons[i].shape[0]
                    current_ncons = ncons[i]

                    # Get current constraint shape
                    if current_ncons == 0:
                        # Task has no constraints: create matrix with c_max columns filled with zeros
                        new_cons[i] = np.zeros((n_samples, c_max))
                    else:
                        # Task has constraints: pad if necessary.
                        # Padded columns stand for constraints this task does not
                        # have, so they must contribute zero violation. `padding`
                        # is a decision-variable strategy and is deliberately not
                        # applied here: 'mid'/'random' would give every padded
                        # column a positive value, and since CV = sum(max(0, con))
                        # that fabricates infeasibility for the task with fewer
                        # constraints.
                        dif = c_max - current_ncons
                        if dif > 0:
                            pad_values = np.zeros((n_samples, dif))
                            new_cons[i] = np.hstack([new_cons[i], pad_values])
            else:
                # No task has constraints: ensure all constraint matrices are empty with same shape
                for i in range(n_tasks):
                    n_samples = new_cons[i].shape[0]
                    new_cons[i] = np.zeros((n_samples, 0))

    elif type == 'real':
        # Handle Decision Variables (Truncation)
        for i in range(n_tasks):
            new_decs[i] = new_decs[i][:, :dims[i]]

        # Handle Objectives (Truncation)
        if new_objs is not None:
            for i in range(n_tasks):
                new_objs[i] = new_objs[i][:, :nobjs[i]]

        # Handle Constraints (Truncation) - FIXED VERSION
        if new_cons is not None:
            for i in range(n_tasks):
                target_ncons = ncons[i]
                if target_ncons == 0:
                    # Task should have no constraints: return empty constraint matrix
                    n_samples = new_cons[i].shape[0]
                    new_cons[i] = np.zeros((n_samples, 0))
                else:
                    # Task has constraints: truncate to original size
                    new_cons[i] = new_cons[i][:, :target_ncons]

    # Construct return values based on provided arguments
    if objs is None and cons is None:
        return new_decs

    results = [new_decs]
    if objs is not None:
        results.append(new_objs)
    if cons is not None:
        results.append(new_cons)

    return tuple(results)


def nd_sort(objs: np.ndarray, *args) -> Tuple[np.ndarray, int]:
    """
    Perform non-dominated sorting on a population of objective values.

    Parameters
    ----------
    objs : np.ndarray
        Objective value matrix, shape (n, m)
    *args : tuple
        Optional arguments:

        - (n_sort,): Number of solutions to sort
        - (cons, n_sort): Constraint matrix and number of solutions to sort

    Returns
    -------
    front_no : np.ndarray
        Non-dominated front number for each solution, shape (n,)
    max_fno : int
        Maximum front number assigned
    """
    pop_obj = objs.copy()
    n, m = pop_obj.shape

    # Parse arguments
    if len(args) == 1:
        # nd_sort(objs, n_sort)
        n_sort = args[0]
    elif len(args) == 2:
        # nd_sort(objs, cons, n_sort)
        pop_con = args[0]
        n_sort = args[1]

        # Handle constraints using constrained domination
        if pop_con is not None:
            infeasible = np.any(pop_con > 0, axis=1)
            if np.any(infeasible):
                max_obj = np.max(pop_obj, axis=0)
                constraint_violation = np.sum(np.maximum(0, pop_con[infeasible, :]), axis=1)
                pop_obj[infeasible, :] = max_obj + constraint_violation[:, np.newaxis]
    else:
        raise ValueError("Invalid number of arguments. Use nd_sort(objs, n_sort) or nd_sort(objs, cons, n_sort)")

    # Find unique rows and their locations
    unique_obj, inverse_indices = np.unique(pop_obj, axis=0, return_inverse=True)

    # Count occurrences of each unique row
    table = np.bincount(inverse_indices, minlength=len(unique_obj))

    n_unique, m = unique_obj.shape
    front_no = np.full(n_unique, np.inf)
    max_fno = 0

    # Continue until enough solutions are sorted
    while np.sum(table[front_no < np.inf]) < min(n_sort, len(inverse_indices)):
        max_fno += 1

        for i in range(n_unique):
            if front_no[i] == np.inf:
                dominated = False

                # Check domination by solutions in current front
                for j in range(i - 1, -1, -1):
                    if front_no[j] == max_fno:
                        # Check if solution j dominates solution i
                        m_idx = 1
                        while m_idx < m and unique_obj[i, m_idx] >= unique_obj[j, m_idx]:
                            m_idx += 1

                        dominated = (m_idx == m)

                        if dominated or m == 2:
                            break

                if not dominated:
                    front_no[i] = max_fno

    # Map back to original indices
    front_no = front_no[inverse_indices]

    return front_no, max_fno


def crowding_distance(pop_obj: np.ndarray, front_no: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the crowding distance for a population of solutions.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective value matrix, shape (n, m), where n is the number of
        solutions and m is the number of objectives
    front_no : np.ndarray, optional
        Non-dominated front number for each solution, shape (n,).
        If not provided, all solutions are assumed to belong to the same front.

    Returns
    -------
    crowd_dis : np.ndarray
        Crowding distance for each solution, shape (n,).
        Boundary solutions are assigned infinite distance.
    """
    n, m = pop_obj.shape

    # If front_no is not provided, assume all solutions are in the same front
    if front_no is None:
        front_no = np.ones(n)

    crowd_dis = np.zeros(n)

    # Get all fronts except inf
    fronts = np.setdiff1d(np.unique(front_no), [np.inf])

    # Calculate crowding distance for each front
    for f in fronts:
        # Get indices of solutions in current front
        front = np.where(front_no == f)[0]

        # Skip if front has less than 2 solutions
        if len(front) < 2:
            crowd_dis[front] = np.inf
            continue

        # Get max and min values for each objective in this front
        fmax = np.max(pop_obj[front, :], axis=0)
        fmin = np.min(pop_obj[front, :], axis=0)

        # Calculate crowding distance for each objective
        for i in range(m):
            # Sort solutions by i-th objective
            rank = np.argsort(pop_obj[front, i])

            # Boundary solutions get infinite crowding distance
            crowd_dis[front[rank[0]]] = np.inf
            crowd_dis[front[rank[-1]]] = np.inf

            # Calculate crowding distance for intermediate solutions
            for j in range(1, len(front) - 1):
                if fmax[i] - fmin[i] > 0:
                    crowd_dis[front[rank[j]]] += (
                            (pop_obj[front[rank[j + 1]], i] - pop_obj[front[rank[j - 1]], i]) /
                            (fmax[i] - fmin[i])
                    )

    return crowd_dis


def tournament_selection(
        K: int,
        N: int,
        *fitness_arrays: np.ndarray,
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Perform tournament selection on a population.

    Parameters
    ----------
    K : int
        Tournament size. If K <= 1, selection is purely random with replacement.
    N : int
        Number of individuals to select
    *fitness_arrays : np.ndarray
        One or more arrays of fitness values, compared lexicographically
        (first array is the primary key). Lower values are considered better.
    rng : np.random.Generator, optional
        NumPy random number generator. If None, the legacy global stream
        (``np.random.*``) is used, so that ``np.random.seed`` reproduces the
        selection. Do NOT fall back to ``np.random.default_rng()``: an
        unseeded Generator draws from OS entropy and silently makes every
        run that relies on this helper irreproducible.

    Returns
    -------
    selected : np.ndarray
        Array of selected individual indices, shape (N,), dtype=int
    """
    # Convert inputs to 1D numpy arrays and validate lengths
    fitnesss = []
    for arr in fitness_arrays:
        a = np.asarray(arr).ravel()
        fitnesss.append(a)
    pop_size = fitnesss[0].shape[0]

    # Draw from the caller's generator when given, otherwise from the legacy
    # global stream that the rest of the platform (and np.random.seed) uses.
    def _integers(low, high, size):
        if rng is None:
            return np.random.randint(low, high, size=size)
        return rng.integers(low, high, size=size)

    if K <= 1:
        # Purely random selection with replacement
        selected = _integers(0, pop_size, N).astype(int)
    else:
        keys = tuple(fitnesss[::-1])
        order = np.lexsort(keys)
        ranks = np.empty(pop_size, dtype=int)
        ranks[order] = np.arange(pop_size)

        # Sample K contestants for each of N tournaments (with replacement)
        parents = _integers(0, pop_size, (K, N))
        parent_ranks = ranks[parents]
        winners_pos = np.argmin(parent_ranks, axis=0)
        selected = parents[winners_pos, np.arange(N)]

    return selected.astype(int)


def selection_elit(
        objs: np.ndarray,
        n: int,
        cons: Optional[np.ndarray] = None,
        epsilon: float = 0
) -> np.ndarray:
    """
    Elite selection for single-objective (optionally constrained) optimization.

    Parameters
    ----------
    objs : np.ndarray
        Objective values of the population (smaller is better), shape (N, 1)
    n : int
        Number of individuals to select
    cons : np.ndarray, optional
        Constraint violation values, each column is a constraint, shape (N, c).
        Default is None.
    epsilon : float, optional
        Threshold for constraint violations to be treated as feasible (default: 0)

    Returns
    -------
    indices : np.ndarray
        Selected individual indices (0-based), shape (n,)
    """
    N = objs.shape[0]

    if cons is not None:
        CVs = np.sum(np.maximum(0, cons), axis=1)
        CVs[CVs < epsilon] = 0
    else:
        CVs = np.zeros(N)

    rank = np.lexsort((objs.flatten(), CVs.flatten()))

    indices = rank[:n]

    return indices


def trim_excess_evaluations(
        all_decs: List[List[np.ndarray]],
        all_objs: List[List[np.ndarray]],
        nt: int,
        max_nfes_per_task: List[int],
        nfes_per_task: List[int],
        all_cons: Optional[List[List[np.ndarray]]] = None
) -> Union[
    Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[int]],
    Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[int], List[List[np.ndarray]]]
]:
    """
    Trim excess evaluations when nfes_per_task exceeds max_nfes_per_task

    Parameters
    ----------
    all_decs : List[List[np.ndarray]]
        Decision variables history for all tasks
    all_objs : List[List[np.ndarray]]
        Objective values history for all tasks
    nt : int
        Number of tasks
    max_nfes_per_task : List[int]
        Maximum evaluation budget per task
    nfes_per_task : List[int]
        Actual evaluations used per task
    all_cons : List[List[np.ndarray]], optional
        Constraint values history for all tasks

    Returns
    -------
    all_decs : List[List[np.ndarray]]
        Trimmed decision variables history
    all_objs : List[List[np.ndarray]]
        Trimmed objective values history
    nfes_per_task : List[int]
        Updated evaluation counts per task (after trimming)
    all_cons : List[List[np.ndarray]], optional
        Trimmed constraint values history (only if input all_cons is provided)
    """
    # Deep copy to avoid modifying original data
    all_decs_trimmed: List[List[np.ndarray]] = copy.deepcopy(all_decs)
    all_objs_trimmed: List[List[np.ndarray]] = copy.deepcopy(all_objs)
    nfes_per_task_updated: List[int] = copy.deepcopy(nfes_per_task)

    has_cons = all_cons is not None
    all_cons_trimmed: Optional[List[List[np.ndarray]]] = None
    if has_cons:
        all_cons_trimmed = copy.deepcopy(all_cons)

    # Process each task
    for i in range(nt):
        # Calculate excess evaluations
        excess = nfes_per_task_updated[i] - max_nfes_per_task[i]

        if excess > 0:
            # Check if there are generations to trim
            if len(all_decs_trimmed[i]) == 0:
                continue

            original_excess = excess

            # Get the last generation
            last_gen_decs = all_decs_trimmed[i][-1]
            last_gen_objs = all_objs_trimmed[i][-1]

            # Check if excess is greater than or equal to all points in last generation
            if excess >= len(last_gen_decs):
                # Remove entire last generation(s) if needed
                while excess > 0 and len(all_decs_trimmed[i]) > 0:
                    last_gen_size = len(all_decs_trimmed[i][-1])

                    if excess >= last_gen_size:
                        # Remove entire last generation
                        all_decs_trimmed[i].pop()
                        all_objs_trimmed[i].pop()
                        if has_cons and all_cons_trimmed is not None:
                            all_cons_trimmed[i].pop()
                        excess -= last_gen_size
                    else:
                        # Trim partial last generation
                        all_decs_trimmed[i][-1] = all_decs_trimmed[i][-1][:-excess]
                        all_objs_trimmed[i][-1] = all_objs_trimmed[i][-1][:-excess]
                        if has_cons and all_cons_trimmed is not None:
                            all_cons_trimmed[i][-1] = all_cons_trimmed[i][-1][:-excess]
                        excess = 0
            else:
                # Trim the last 'excess' rows from the last generation
                all_decs_trimmed[i][-1] = last_gen_decs[:-excess]
                all_objs_trimmed[i][-1] = last_gen_objs[:-excess]

                if has_cons and all_cons_trimmed is not None:
                    last_gen_cons = all_cons_trimmed[i][-1]
                    all_cons_trimmed[i][-1] = last_gen_cons[:-excess]

            # Update nfes_per_task: subtract the number of trimmed evaluations
            nfes_per_task_updated[i] -= original_excess

    # Return results based on whether constraints are present
    if has_cons and all_cons_trimmed is not None:
        return all_decs_trimmed, all_objs_trimmed, nfes_per_task_updated, all_cons_trimmed
    else:
        return all_decs_trimmed, all_objs_trimmed, nfes_per_task_updated


def normalize(
        data: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        axis: int = 0,
        method: str = 'minmax'
) -> Tuple[
    Union[np.ndarray, List[np.ndarray]],
    Union[np.ndarray, List[np.ndarray]],
    Union[np.ndarray, List[np.ndarray]]
]:
    """
    Normalize input data (arrays or matrices).

    Parameters
    ----------
    data : Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]
        Supports the following input formats:

        - Single 1D array: [1, 2, 3]
        - Single 2D matrix: [[1,2], [3,4]]
        - List of multiple arrays/matrices: [[1,2,3], [4,5,6]] or [matrix1, matrix2]
    axis : int, optional
        Axis along which to normalize (only applies to 2D+ data), default is 0.

        - 0: Column-wise normalization (default, recommended for multi-objective optimization)
        - 1: Row-wise normalization
        - For 1D arrays, global normalization is always used
    method : str, optional
        Normalization method, default is 'minmax'.

        - 'minmax': Min-max normalization, scales to [0, 1] (default)
        - 'zscore': Z-score normalization, mean=0, std=1

    Returns
    -------
    normalized : Union[np.ndarray, List[np.ndarray]]
        Normalized result (same format as input)
    stat1 : Union[np.ndarray, List[np.ndarray]]
        Min values (minmax) or mean values (zscore)
    stat2 : Union[np.ndarray, List[np.ndarray]]
        Max values (minmax) or std values (zscore)
    """
    if method not in ['minmax', 'zscore']:
        raise ValueError("method must be 'minmax' or 'zscore'")

    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    # Determine if input is a single array/matrix or a list of arrays
    def is_single_input(data):
        if isinstance(data, np.ndarray) and data.dtype != object:
            return True
        if isinstance(data, list) and len(data) > 0 and np.isscalar(data[0]):
            return True
        return False

    single_input = is_single_input(data)

    if single_input:
        data_list = [np.array(data)]
    else:
        data_list = [np.array(d) for d in data]

    normalized = []
    stat1_list = []  # min or mean
    stat2_list = []  # max or std

    for arr in data_list:
        arr = np.array(arr)
        arr_ndim = arr.ndim

        if arr_ndim == 1:
            # 1D array: global normalization
            if method == 'minmax':
                stat1 = np.min(arr)
                stat2 = np.max(arr)
                range_val = stat2 - stat1
                if range_val < 1e-10:
                    range_val = 1.0
                arr_norm = (arr - stat1) / range_val
            else:  # zscore
                stat1 = np.mean(arr)
                stat2 = np.std(arr)
                stat2_safe = stat2 if stat2 >= 1e-10 else 1.0
                arr_norm = (arr - stat1) / stat2_safe

            normalized.append(arr_norm)
            stat1_list.append(stat1)
            stat2_list.append(stat2)

        else:
            # 2D+ array: normalize along specified axis
            if method == 'minmax':
                stat1 = np.min(arr, axis=axis, keepdims=True)
                stat2 = np.max(arr, axis=axis, keepdims=True)

                range_val = stat2 - stat1
                range_val = np.where(range_val < 1e-10, 1.0, range_val)
                arr_norm = (arr - stat1) / range_val

            else:  # zscore
                stat1 = np.mean(arr, axis=axis, keepdims=True)
                stat2 = np.std(arr, axis=axis, keepdims=True)

                stat2_safe = np.where(stat2 < 1e-10, 1.0, stat2)
                arr_norm = (arr - stat1) / stat2_safe

            # Remove keepdims dimension
            stat1 = np.squeeze(stat1)
            stat2 = np.squeeze(stat2)

            normalized.append(arr_norm)
            stat1_list.append(stat1)
            stat2_list.append(stat2)

    # Return single result if input was single
    if single_input:
        return normalized[0], stat1_list[0], stat2_list[0]

    return normalized, stat1_list, stat2_list


def denormalize(
        data: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        stat1: Union[np.ndarray, List[np.ndarray]],
        stat2: Union[np.ndarray, List[np.ndarray]],
        axis: int = 0,
        method: str = 'minmax'
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Inverse normalization to restore original scale.

    Parameters
    ----------
    data : Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]
        Normalized data (same format as normalize() output).

        - Single 1D array: [0, 0.25, 0.5, 0.75, 1]
        - Single 2D matrix: [[0, 0], [0.5, 0.5], [1, 1]]
        - List of multiple arrays/matrices
    stat1 : Union[np.ndarray, List[np.ndarray]]
        Min values (minmax) or mean values (zscore) from normalize()
    stat2 : Union[np.ndarray, List[np.ndarray]]
        Max values (minmax) or std values (zscore) from normalize()
    axis : int, optional
        Must match the axis used in normalize(), default is 0.

        - 0: Column-wise normalization (default)
        - 1: Row-wise normalization
    method : str, optional
        Must match the method used in normalize(), default is 'minmax'.

        - 'minmax': Inverse min-max normalization (default)
        - 'zscore': Inverse z-score normalization

    Returns
    -------
    restored : Union[np.ndarray, List[np.ndarray]]
        Restored data in original scale (same format as input)
    """
    if method not in ['minmax', 'zscore']:
        raise ValueError("method must be 'minmax' or 'zscore'")

    if axis not in [0, 1]:
        raise ValueError("axis must be 0 or 1")

    # Determine if input is a single array/matrix or a list of arrays
    def is_single_input(data):
        if isinstance(data, np.ndarray) and data.dtype != object:
            return True
        if isinstance(data, list) and len(data) > 0 and np.isscalar(data[0]):
            return True
        return False

    single_input = is_single_input(data)

    if single_input:
        data_list = [np.array(data)]
        stat1_list = [stat1]
        stat2_list = [stat2]
    else:
        data_list = [np.array(d) for d in data]
        stat1_list = stat1
        stat2_list = stat2

    restored = []

    for arr, s1, s2 in zip(data_list, stat1_list, stat2_list):
        arr = np.array(arr)
        arr_ndim = arr.ndim

        # Ensure at least 2D for unified processing
        arr_2d = np.atleast_2d(arr)

        # Prepare stats for broadcasting
        s1 = np.atleast_2d(s1)
        s2 = np.atleast_2d(s2)
        if axis == 0:
            # For column-wise, stats should be (1, n_cols)
            if s1.shape[0] != 1:
                s1 = s1.reshape(1, -1)
                s2 = s2.reshape(1, -1)
        elif axis == 1:
            # For row-wise, stats should be (n_rows, 1)
            if s1.shape[1] != 1:
                s1 = s1.reshape(-1, 1)
                s2 = s2.reshape(-1, 1)

        if method == 'minmax':
            # Inverse min-max: x = norm * (max - min) + min
            range_val = s2 - s1
            range_val = np.where(np.abs(range_val) < 1e-10, 1.0, range_val)
            arr_restored = arr_2d * range_val + s1
        else:  # zscore
            # Inverse z-score: x = norm * std + mean
            s2_safe = np.where(np.abs(s2) < 1e-10, 1.0, s2)
            arr_restored = arr_2d * s2_safe + s1

        # Restore original dimensions
        if arr_ndim == 1:
            arr_restored = arr_restored.flatten()

        restored.append(arr_restored)

    # Return single result if input was single
    if single_input:
        return restored[0]

    return restored


def remove_duplicates(new_decs, existing_decs=None, tol=1e-6):
    """
    Remove duplicate solutions from new decision variables.

    Parameters
    ----------
    new_decs : np.ndarray
        New decision variables to be filtered, shape (N, D)
    existing_decs : np.ndarray, optional
        Existing decision variables to check against, shape (M, D)
        If None, only remove duplicates within new_decs
    tol : float, optional
        Tolerance for duplicate detection (default: 1e-6)

    Returns
    -------
    unique_decs : np.ndarray
        Unique decision variables, shape (K, D) where K <= N
    """
    if new_decs.shape[0] == 0:
        return np.empty((0, new_decs.shape[1]))

    # Step 1: Remove duplicates within new_decs
    unique_indices = []
    seen = set()

    for i, dec in enumerate(new_decs):
        dec_tuple = tuple(np.round(dec, 8))
        if dec_tuple not in seen:
            seen.add(dec_tuple)
            unique_indices.append(i)

    if len(unique_indices) == 0:
        return np.empty((0, new_decs.shape[1]))

    unique_decs = new_decs[unique_indices]

    # Step 2: Remove solutions already in existing_decs (if provided)
    if existing_decs is not None and existing_decs.shape[0] > 0:
        final_indices = []
        for i, dec in enumerate(unique_decs):
            distances = np.min(cdist(dec.reshape(1, -1), existing_decs))
            if distances > tol:
                final_indices.append(i)

        if len(final_indices) == 0:
            return np.empty((0, new_decs.shape[1]))

        unique_decs = unique_decs[final_indices]

    return unique_decs


def kmeans_clustering(data, k):
    """
    K-means clustering using sklearn.

    Parameters
    ----------
    data : np.ndarray
        Data points, shape (N, D)
    k : int
        Number of clusters

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each point, shape (N,)
        Labels are integers in range [0, k-1]
    """
    N = data.shape[0]

    # If k >= N, assign each point to its own cluster
    if k >= N:
        return np.arange(N)

    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=None)
    labels = kmeans.fit_predict(data)

    return labels


def ibea_fitness(objs, kappa):
    """
    Calculate fitness values for the population using IBEA indicator.

    Parameters
    ----------
    objs : ndarray
        Objective values with shape (N, M), where N is the number of
        individuals and M is the number of objectives.
    kappa : float
        Fitness scaling factor.

    Returns
    -------
    fitness : ndarray
        Fitness values for each individual with shape (N,).
    I : ndarray
        Indicator matrix with shape (N, N).
    C : ndarray
        Normalization constants with shape (N,).
    """
    # Normalize objective values to [0, 1]
    min_val = np.min(objs, axis=0)
    max_val = np.max(objs, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    objs_norm = (objs - min_val) / range_val

    # Calculate indicator matrix (vectorized)
    I = np.max(objs_norm[:, np.newaxis, :] - objs_norm[np.newaxis, :, :], axis=2)

    # Calculate normalization constants
    C = np.max(np.abs(I), axis=0)
    C[C == 0] = 1e-6  # Avoid division by zero

    # Calculate fitness
    fitness = np.sum(-np.exp(-I / C / kappa), axis=0) + 1

    return fitness, I, C


def is_duplicate(x, X, epsilon=1e-10):
    """
    Check if position(s) x are duplicates in X or within themselves.

    Parameters
    ----------
    x : np.ndarray
        Position(s) to check, shape (dim,) or (1, dim) or (n_points, dim)
    X : np.ndarray
        Existing positions, shape (n_samples, dim)
    epsilon : float
        Tolerance for duplicate detection

    Returns
    -------
    bool or list of bool
        - If x is single point: returns bool
        - If x contains multiple points: returns list of bool for each point
    """
    # Ensure x is at least 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape[0] == 1:
        # Single point case: return bool
        if len(X) == 0:
            return False

        dist = cdist(x, X, metric='euclidean')
        return bool(np.min(dist) < epsilon)
    else:
        # Multiple points case: return list of bool
        n_points = x.shape[0]
        results = [False] * n_points

        # Track which points in x are unique (not duplicate with any point checked so far)
        unique_points = []  # List of indices that are still considered unique

        # First pass: check against X
        if len(X) > 0:
            dist_to_X = cdist(x, X, metric='euclidean')
            for i in range(n_points):
                if np.any(dist_to_X[i] < epsilon):
                    results[i] = True
                else:
                    unique_points.append(i)
        else:
            unique_points = list(range(n_points))

        # Second pass: check duplicates among unique x points
        # We'll build up a list of truly unique points
        final_unique = []

        for i in unique_points:
            is_unique = True
            # Check against previously identified unique points
            for j in final_unique:
                dist_ij = np.linalg.norm(x[i] - x[j])
                if dist_ij < epsilon:
                    # x[i] is duplicate of x[j]
                    results[i] = True
                    is_unique = False
                    break

            if is_unique:
                final_unique.append(i)

        return results


# =============================================================================
# RBF Model (Multiquadric)
# =============================================================================

def rbf_build(X_train, Y_train, bf_c=1.0):
    """
    Build an RBF (Multiquadric) interpolation model.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs, shape (n, d)
    Y_train : np.ndarray
        Training outputs, shape (n,) or (n, 1)
    bf_c : float
        Multiquadric parameter (default: 1.0)

    Returns
    -------
    model : dict
        RBF model containing coefficients and parameters
    """
    Y_train = Y_train.flatten()
    n = X_train.shape[0]
    mean_y = np.mean(Y_train)

    # Compute MQ distance matrix
    dist_sq = cdist(X_train, X_train, metric='sqeuclidean')
    dist_matrix = np.sqrt(dist_sq + bf_c ** 2)

    # Solve for coefficients
    try:
        coefs = np.linalg.solve(dist_matrix, Y_train - mean_y)
    except np.linalg.LinAlgError:
        coefs = np.linalg.lstsq(dist_matrix, Y_train - mean_y, rcond=None)[0]

    return {
        'n': n,
        'mean_y': mean_y,
        'bf_c': bf_c,
        'coefs': coefs
    }


def rbf_predict(model, X_train, X_query):
    """
    Predict using RBF model.

    Parameters
    ----------
    model : dict
        RBF model from rbf_build
    X_train : np.ndarray
        Training inputs, shape (n, d)
    X_query : np.ndarray
        Query points, shape (nq, d)

    Returns
    -------
    Y_pred : np.ndarray
        Predicted values, shape (nq,)
    """
    bf_c = model['bf_c']

    # MQ distance from query to training points
    dist_sq = cdist(X_query, X_train, metric='sqeuclidean')
    dist_matrix = np.sqrt(dist_sq + bf_c ** 2)

    Y_pred = model['mean_y'] + dist_matrix @ model['coefs']
    return Y_pred


def newrbe_surrogate(X_train, Y_train, spread=None):
    """
    Build an exact Gaussian RBF interpolant equivalent to MATLAB's
    ``newrbe(X', y', spread)``. When ``spread`` is None it is estimated as
    ``max(pairwise distance) / (dim * n) ** (1 / dim)``.

    MATLAB's newrbe uses Gaussian basis functions ``exp(-(0.8326 * r / spread)^2)``
    (so that the response is 0.5 at distance ``spread``) plus an output bias,
    solved as an underdetermined linear system. Here the (n+1)-parameter system
    is solved with least squares (minimum-norm solution).

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs, shape (n, d)
    Y_train : np.ndarray
        Training outputs, shape (n,) or (n, 1)
    spread : float, optional
        Gaussian spread parameter; estimated from the data when None.

    Returns
    -------
    predict : callable
        ``predict(x)`` accepts an array of shape (nq, d) or (d,) and returns
        predictions of shape (nq,)
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float).flatten()
    n, dim = X_train.shape

    dist_matrix = cdist(X_train, X_train)
    if spread is None:
        max_dist = dist_matrix.max()
        if max_dist > 0:
            spread = max_dist / (dim * n) ** (1.0 / dim)
        else:
            spread = 1.0
    b = 0.8326 / spread

    A = np.exp(-(b * dist_matrix) ** 2)
    G = np.hstack([A, np.ones((n, 1))])
    w = np.linalg.lstsq(G, Y_train, rcond=None)[0]

    def predict(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        A_query = np.exp(-(b * cdist(x, X_train)) ** 2)
        return A_query @ w[:-1] + w[-1]

    return predict


def newrb_surrogate(X_train, Y_train, goal=0.1, spread=1.0, max_neurons=8):
    """
    Build an approximating Gaussian RBF network equivalent to MATLAB's
    ``newrb(X', y', goal, spread, max_neurons)``.

    Neurons (centered on training points) are added greedily: at each step the
    candidate input vector whose basis-function column best explains the
    current residual is added, and the output layer (weights + bias) is
    re-solved by least squares. Growth stops when the training MSE drops to
    ``goal`` or ``max_neurons`` neurons are reached.

    Parameters
    ----------
    X_train : np.ndarray
        Training inputs, shape (n, d)
    Y_train : np.ndarray
        Training outputs, shape (n,) or (n, 1)
    goal : float
        Target mean squared error (default: 0.1)
    spread : float
        Gaussian spread parameter (default: 1.0)
    max_neurons : int
        Maximum number of hidden neurons (default: 8)

    Returns
    -------
    predict : callable
        ``predict(x)`` accepts an array of shape (nq, d) or (d,) and returns
        predictions of shape (nq,)
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float).flatten()
    n = X_train.shape[0]
    b = 0.8326 / max(spread, 1e-30)

    A = np.exp(-(b * cdist(X_train, X_train)) ** 2)

    chosen = []
    err = Y_train.copy()
    w = np.zeros(1)
    for _ in range(min(max_neurons, n)):
        # Candidate whose column best reduces the residual (newrb's criterion)
        proj = A.T @ err
        norms = np.sum(A * A, axis=0)
        scores = proj ** 2 / np.maximum(norms, 1e-300)
        scores[chosen] = -np.inf
        j = int(np.argmax(scores))
        chosen.append(j)

        G = np.hstack([A[:, chosen], np.ones((n, 1))])
        w = np.linalg.lstsq(G, Y_train, rcond=None)[0]
        err = Y_train - G @ w
        if np.mean(err ** 2) <= goal:
            break

    centers = X_train[chosen]

    def predict(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        A_query = np.exp(-(b * cdist(x, centers)) ** 2)
        return A_query @ w[:-1] + w[-1]

    return predict


# =============================================================================
# dsmerge: Merge duplicate design sites
# =============================================================================

def dsmerge(S, Y, ds=1e-14):
    """
    Merge data for duplicate/near-duplicate design sites.

    Finds clusters of points within threshold distance in normalized space
    and merges them by averaging.

    Parameters
    ----------
    S : np.ndarray
        Design sites, shape (m, n)
    Y : np.ndarray
        Responses, shape (m,) or (m, 1)
    ds : float
        Threshold for near-duplicate detection (default: 1e-14)

    Returns
    -------
    mS : np.ndarray
        Merged design sites
    mY : np.ndarray
        Merged responses
    """
    S = S.copy()
    Y = Y.flatten().copy()

    more = True
    while more:
        m = S.shape[0]
        if m <= 1:
            break

        # Normalize sites
        mean_s = np.mean(S, axis=0)
        std_s = np.std(S, axis=0)
        std_s[std_s == 0] = 1.0
        sc_s = (S - mean_s) / std_s

        # Compute pairwise Euclidean distances
        D = cdist(sc_s, sc_s, metric='euclidean')
        np.fill_diagonal(D, np.inf)

        # Count near-duplicates for each point
        mult = np.sum(D < ds, axis=0)

        if np.max(mult) == 0:
            break

        ladr = []
        while np.max(mult) > 0:
            jj = np.argmax(mult)
            ladr.append(jj)

            # Find neighbors
            ngb = np.where(D[:, jj] < ds)[0]
            if len(ngb) == 0:
                mult[jj] = 0
                continue

            # Merge: average S and Y at the center point
            all_pts = np.concatenate([[jj], ngb])
            S[jj] = np.mean(S[all_pts], axis=0)
            Y[jj] = np.mean(Y[all_pts])

            mult[ngb] = 0
            mult[jj] = 0

        if len(ladr) == 0:
            break

        # Remove neighbors that were merged (not centers)
        keep = list(set(range(m)))
        for center in ladr:
            ngb = np.where(D[:, center] < ds)[0]
            for n_idx in ngb:
                if n_idx != center and n_idx in keep:
                    keep.remove(n_idx)

        keep = sorted(keep)
        S = S[keep]
        Y = Y[keep]

        # Check if any more duplicates
        if len(S) <= 1:
            break
        mean_s2 = np.mean(S, axis=0)
        std_s2 = np.std(S, axis=0)
        std_s2[std_s2 == 0] = 1.0
        sc_s2 = (S - mean_s2) / std_s2
        D2 = cdist(sc_s2, sc_s2, metric='euclidean')
        np.fill_diagonal(D2, np.inf)
        if np.min(D2) >= ds:
            more = False

    return S, Y


# =============================================================================
# Archive Update
# =============================================================================

def merge_archive(arc_decs, arc_objs, new_decs, new_objs):
    """
    Update archive by merging and removing duplicates.

    Parameters
    ----------
    arc_decs, arc_objs : np.ndarray
        Current archive
    new_decs, new_objs : np.ndarray
        New solutions

    Returns
    -------
    merged_decs, merged_objs : np.ndarray
        Updated archive
    """
    merged_decs = np.vstack([arc_decs, new_decs])
    merged_objs = np.vstack([arc_objs, new_objs])
    _, unique_idx = np.unique(merged_decs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return merged_decs[unique_idx], merged_objs[unique_idx]


# =============================================================================
# SPEA2 Fitness
# =============================================================================

def spea2_fitness(pop_obj, pop_con=None):
    """
    Calculate SPEA2 fitness with constrained dominance.

    Parameters
    ----------
    pop_obj : np.ndarray, shape (N, M)
    pop_con : np.ndarray, shape (N, C), optional
        Constraint violation values (positive = violation).

    Returns
    -------
    fitness : np.ndarray, shape (N,)
        SPEA2 fitness values. Lower is better; < 1 means non-dominated.
    """
    N = pop_obj.shape[0]
    if N == 0:
        return np.array([])

    if pop_con is not None:
        CV = np.sum(np.maximum(0, pop_con), axis=1) if pop_con.ndim == 2 else np.maximum(0, pop_con)
    else:
        CV = np.zeros(N)

    # Constrained dominance relation
    dominate = np.zeros((N, N), dtype=bool)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if CV[i] < CV[j]:
                dominate[i, j] = True
            elif CV[i] > CV[j]:
                dominate[j, i] = True
            else:
                better_i = np.any(pop_obj[i] < pop_obj[j])
                worse_i = np.any(pop_obj[i] > pop_obj[j])
                if better_i and not worse_i:
                    dominate[i, j] = True
                elif worse_i and not better_i:
                    dominate[j, i] = True

    # Strength S(i)
    S = np.sum(dominate, axis=1)

    # Raw fitness R(i)
    R = np.zeros(N, dtype=float)
    for i in range(N):
        dominators = np.where(dominate[:, i])[0]
        R[i] = np.sum(S[dominators])

    # Density D(i)
    dist = cdist(pop_obj, pop_obj)
    np.fill_diagonal(dist, np.inf)
    dist_sorted = np.sort(dist, axis=1)
    k = max(1, int(np.floor(np.sqrt(N))))
    D = 1.0 / (dist_sorted[:, min(k - 1, N - 1)] + 2.0)

    return R + D


# =============================================================================
# SPEA2 Truncation (Lexicographic)
# =============================================================================

def spea2_truncation(pop_obj, N):
    """
    Select N solutions by iteratively removing the most crowded.
    Uses SPEA2-style lexicographic nearest-neighbor comparison.

    Parameters
    ----------
    pop_obj : np.ndarray, shape (n, M)
        Objectives.
    N : int
        Number of solutions to keep.

    Returns
    -------
    selected : np.ndarray
        Indices of selected solutions.
    """
    n = pop_obj.shape[0]
    if n <= N:
        return np.arange(n)

    dist = cdist(pop_obj, pop_obj)
    np.fill_diagonal(dist, np.inf)

    deleted = np.zeros(n, dtype=bool)
    while np.sum(~deleted) > N:
        remaining = np.where(~deleted)[0]
        sub_dist = dist[np.ix_(remaining, remaining)]
        sorted_dist = np.sort(sub_dist, axis=1)
        rank = np.lexsort(sorted_dist[:, ::-1].T)
        deleted[remaining[rank[0]]] = True

    return np.where(~deleted)[0]


# =============================================================================
# SPEA2 Truncation (Fast, Min-Distance)
# =============================================================================

def spea2_truncation_fast(pop_obj, N):
    """
    Select N solutions by iteratively removing the one with smallest
    nearest-neighbor distance.

    Parameters
    ----------
    pop_obj : np.ndarray, shape (n, M)
        Objectives.
    N : int
        Number of solutions to keep.

    Returns
    -------
    selected : np.ndarray
        Indices of selected solutions.
    """
    n = pop_obj.shape[0]
    if n <= N:
        return np.arange(n)

    dist = cdist(pop_obj, pop_obj)
    np.fill_diagonal(dist, np.inf)

    deleted = np.zeros(n, dtype=bool)
    K = n - N

    for _ in range(K):
        remaining = np.where(~deleted)[0]
        sub_dist = dist[np.ix_(remaining, remaining)]
        min_dists = np.min(sub_dist, axis=1)
        worst = np.argmin(min_dists)
        deleted[remaining[worst]] = True

    return np.where(~deleted)[0]


# =============================================================================
# Dimension Alignment
# =============================================================================

def align_dimensions(dec, target_dim, fill='random'):
    """
    Align decision variable dimensions to target dimension.

    Supports both 1D vectors and 2D arrays. If the input has fewer dimensions
    than the target, padding is added. If it has more, it is truncated.

    Parameters
    ----------
    dec : np.ndarray
        Decision variable(s), shape (dim,) for 1D or (n, dim) for 2D
    target_dim : int
        Target dimension
    fill : str, optional
        Padding strategy: 'random' for U[0,1] values, 'zero' for zeros (default: 'random')

    Returns
    -------
    aligned_dec : np.ndarray
        Aligned decision variable(s), shape (target_dim,) or (n, target_dim)
    """
    if dec.ndim == 1:
        current_dim = len(dec)
        if current_dim == target_dim:
            return dec.copy()
        elif current_dim < target_dim:
            if fill == 'random':
                padding = np.random.rand(target_dim - current_dim)
            else:
                padding = np.zeros(target_dim - current_dim)
            return np.concatenate([dec, padding])
        else:
            return dec[:target_dim].copy()
    else:
        n, current_dim = dec.shape
        if current_dim == target_dim:
            return dec.copy()
        elif current_dim < target_dim:
            if fill == 'random':
                padding = np.random.rand(n, target_dim - current_dim)
            else:
                padding = np.zeros((n, target_dim - current_dim))
            return np.hstack([dec, padding])
        else:
            return dec[:, :target_dim].copy()


# =============================================================================
# Constrained Sort
# =============================================================================

def constrained_sort(objs, cvs):
    """
    Sort indices by constraint violation first, then by objective value (ascending).

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (n,) or (n, 1)
    cvs : np.ndarray
        Constraint violation values (total CV per individual), shape (n,) or (n, 1)

    Returns
    -------
    sorted_indices : np.ndarray
        Indices that would sort the population by (cvs, objs) ascending, shape (n,)
    """
    return np.lexsort((np.asarray(objs).flatten(), np.asarray(cvs).flatten()))


# =============================================================================
# CMA-ES Utilities
# =============================================================================

def cmaes_init_params(dim, lam=None, sigma0=0.3):
    """
    Initialize CMA-ES parameters for a given dimension.

    Parameters
    ----------
    dim : int
        Problem dimension
    lam : int, optional
        Population size (lambda). If None, uses ``4 + floor(3 * log(dim))``.
    sigma0 : float, optional
        Initial step size (default: 0.3)

    Returns
    -------
    params : dict
        CMA-ES state dictionary containing:

        - **Strategy parameters**: dim, lam, mu, weights, mueff
        - **Control parameters**: cs, damps, cc, c1, cmu, chiN
        - **State variables**: m_dec, ps, pc, B, D, C, invsqrtC, sigma, eigenFE
    """
    if lam is None:
        lam = int(4 + 3 * np.log(dim))
    mu = (lam + 1) // 2  # MATLAB: mu = round(lambda / 2), round-half-away-from-zero

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)

    cs = (mueff + 2) / (dim + mueff + 5)
    damps = 1 + cs + 2 * max(np.sqrt((mueff - 1) / (dim + 1)) - 1, 0)
    cc = (4 + mueff / dim) / (4 + dim + 2 * mueff / dim)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + 2 * mueff / 2))
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    return {
        'dim': dim, 'lam': lam, 'mu': mu, 'weights': weights, 'mueff': mueff,
        'cs': cs, 'damps': damps, 'cc': cc, 'c1': c1, 'cmu': cmu, 'chiN': chiN,
        'm_dec': np.random.rand(dim),
        'ps': np.zeros(dim), 'pc': np.zeros(dim),
        'B': np.eye(dim), 'D': np.ones(dim),
        'C': np.eye(dim), 'invsqrtC': np.eye(dim),
        'sigma': sigma0, 'eigenFE': 0
    }


def cmaes_sample(m_dec, sigma, B, D, lam, clip=True):
    """
    Generate offspring using CMA-ES sampling.

    Samples from ``N(m_dec, sigma^2 * C)`` where ``C = B * diag(D^2) * B^T``.

    Parameters
    ----------
    m_dec : np.ndarray
        Mean decision vector, shape (d,)
    sigma : float
        Step size (global scaling factor)
    B : np.ndarray
        Eigenvector matrix from covariance decomposition, shape (d, d)
    D : np.ndarray
        Square root of eigenvalues (standard deviations), shape (d,)
    lam : int
        Number of offspring to generate
    clip : bool, optional
        If True, clip offspring to [0, 1] (default: True)

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (lam, d)
    """
    d = len(m_dec)
    z = np.random.randn(lam, d)
    offdecs = m_dec + sigma * ((z * D) @ B.T)
    if clip:
        offdecs = np.clip(offdecs, 0, 1)
    return offdecs


def cmaes_update(params, sorted_decs, nfes):
    """
    Update CMA-ES parameters given fitness-sorted offspring.

    Updates mean, evolution paths, covariance matrix, step size, and
    performs eigendecomposition when needed. Modifies params dict in-place.

    Parameters
    ----------
    params : dict
        CMA-ES parameter dictionary from :func:`cmaes_init_params`
    sorted_decs : np.ndarray
        Offspring sorted by fitness (best first), shape (>=mu, dim)
    nfes : int
        Current number of function evaluations (for eigendecomposition timing)

    Returns
    -------
    restarted : bool
        True if eigendecomposition failed and covariance was reset to identity
    """
    p = params

    # Update mean
    old_dec = p['m_dec'].copy()
    p['m_dec'] = p['weights'] @ sorted_decs[:p['mu']]

    # Update evolution paths
    diff = (p['m_dec'] - old_dec) / p['sigma']
    p['ps'] = (1 - p['cs']) * p['ps'] + \
              np.sqrt(p['cs'] * (2 - p['cs']) * p['mueff']) * (p['invsqrtC'] @ diff)

    ps_norm = np.linalg.norm(p['ps'])
    factor = 1 - (1 - p['cs']) ** (2 * nfes / p['lam'])
    if factor > 0:
        hsig = float(ps_norm / np.sqrt(factor) / p['chiN'] < 1.4 + 2 / (p['dim'] + 1))
    else:
        hsig = 1.0

    p['pc'] = (1 - p['cc']) * p['pc'] + \
              hsig * np.sqrt(p['cc'] * (2 - p['cc']) * p['mueff']) * diff

    # Update covariance matrix
    artmp = (sorted_decs[:p['mu']] - old_dec) / p['sigma']
    delta = (1 - hsig) * p['cc'] * (2 - p['cc'])
    p['C'] = (1 - p['c1'] - p['cmu']) * p['C'] + \
             p['c1'] * (np.outer(p['pc'], p['pc']) + delta * p['C']) + \
             p['cmu'] * (artmp.T @ np.diag(p['weights']) @ artmp)

    # Update step size
    p['sigma'] *= np.exp(p['cs'] / p['damps'] * (ps_norm / p['chiN'] - 1))

    # Eigendecomposition (periodic)
    restarted = False
    if nfes - p['eigenFE'] > p['lam'] / (p['c1'] + p['cmu']) / p['dim'] / 10:
        p['eigenFE'] = nfes

        if not np.all(np.isfinite(p['C'])):
            restarted = True
        else:
            p['C'] = np.triu(p['C']) + np.triu(p['C'], 1).T
            try:
                eigvals, eigvecs = np.linalg.eigh(p['C'])
                if np.min(eigvals) <= 0:
                    restarted = True
                else:
                    p['B'] = eigvecs
                    p['D'] = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                restarted = True

        if restarted:
            p['B'] = np.eye(p['dim'])
            p['D'] = np.ones(p['dim'])
            p['C'] = np.eye(p['dim'])

        p['invsqrtC'] = p['B'] @ np.diag(1.0 / p['D']) @ p['B'].T

    return restarted
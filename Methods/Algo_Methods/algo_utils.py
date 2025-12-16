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
        if arr is None:  # 这一行必须在 arr.ndim 之前检查
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


def reorganize_initial_data(
        data: List[np.ndarray],
        nt: int,
        n_initial_per_task: List[int]
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

    Returns
    -------
    all_data : List[List[np.ndarray]]
        Reorganized data, where all_data[i][j] contains the first j+1
        initial points of task i
    """
    all_data = []
    for i in range(nt):
        task_data = []
        # Store each initial point as a separate generation
        for j in range(n_initial_per_task[i]):
            task_data.append(data[i][:j + 1].copy())
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
tuple[List[np.ndarray], List[np.ndarray]],
tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
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
    """
    # Copy lists to avoid modifying the original input references
    new_decs = [d for d in decs]
    new_objs = [o for o in objs] if objs is not None else None
    new_cons = [c for c in cons] if cons is not None else None

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
                    else:  # padding == 'zero'
                        pad_values = np.zeros((n_samples, dif))
                    new_objs[i] = np.hstack([new_objs[i], pad_values])

        # Handle Constraints (Padding)
        if new_cons is not None:
            c_max = np.max(ncons)
            if c_max > 0:
                for i in range(n_tasks):
                    n_samples = new_cons[i].shape[0]
                    dif = c_max - ncons[i]
                    if dif > 0:
                        if padding == 'random':
                            pad_values = np.random.rand(n_samples, dif)
                        else:  # padding == 'zero'
                            pad_values = np.zeros((n_samples, dif))
                        new_cons[i] = np.hstack([new_cons[i], pad_values])

    elif type == 'real':
        # Handle Decision Variables (Truncation)
        for i in range(n_tasks):
            new_decs[i] = new_decs[i][:, :dims[i]]

        # Handle Objectives (Truncation)
        if new_objs is not None:
            for i in range(n_tasks):
                new_objs[i] = new_objs[i][:, :nobjs[i]]

        # Handle Constraints (Truncation)
        if new_cons is not None:
            for i in range(n_tasks):
                new_cons[i] = new_cons[i][:, :ncons[i]]

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
        One or more arrays of fitness values. Higher fitness is considered better.
    rng : np.random.Generator, optional
        NumPy random number generator. If None, a new default RNG is created.

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

    if rng is None:
        rng = np.random.default_rng()

    if K <= 1:
        # Purely random selection with replacement
        selected = rng.integers(0, pop_size, size=N, dtype=int)
    else:
        keys = tuple(fitnesss[::-1])
        order = np.lexsort(keys)
        ranks = np.empty(pop_size, dtype=int)
        ranks[order] = np.arange(pop_size)

        # Sample K contestants for each of N tournaments (with replacement)
        parents = rng.integers(0, pop_size, size=(K, N))
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
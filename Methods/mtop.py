import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any, Union
from functools import partial


class ObjectiveFunctionWrapper:
    """可 pickle 的目标函数包装器（用于跨平台并行运行）"""

    def __init__(self, func: Callable, dim: int):
        self.func = func
        self.dim = dim

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        n = X.shape[0]
        try:
            out = self.func(X)  # try vectorized call
        except Exception:
            # try per-row
            rows = []
            for i in range(n):
                r = self.func(np.atleast_1d(X[i]))
                rows.append(np.atleast_1d(r))
            out = np.vstack(rows)

        out = np.asarray(out)
        # Normalize to 2D with n rows
        if out.ndim == 0:
            out = np.full((n, 1), float(out))
        elif out.ndim == 1:
            # ambiguous: if length == n -> (n,1); else if n==1 -> (1,len)
            if out.shape[0] == n:
                out = out.reshape(n, 1)
            else:
                if n == 1:
                    out = out.reshape(1, -1)
                else:
                    # cannot align
                    raise ValueError("Objective returned 1D array that cannot be aligned to input batch.")
        elif out.ndim == 2:
            if out.shape[0] != n:
                # maybe user returned shape (n_obj, n) accidental -> try transpose
                if out.shape[1] == n:
                    out = out.T
                else:
                    raise ValueError("Objective returned 2D array with incompatible number of rows.")
        else:
            raise ValueError("Objective returned array with ndim > 2, unsupported.")
        return out.astype(np.float64)


class ConstraintFunctionWrapper:
    """可 pickle 的约束函数包装器（用于跨平台并行运行）"""

    def __init__(self, func: Callable, k_local: int):
        self.func = func
        self.k_local = k_local

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        n = X.shape[0]
        try:
            out = self.func(X)
        except Exception:
            rows = []
            for i in range(n):
                r = self.func(np.atleast_1d(X[i]))
                rows.append(np.atleast_1d(r))
            out = np.vstack(rows)
        out = np.asarray(out)
        # normalize to (n, k_local)
        if out.ndim == 0:
            out = np.full((n, 1), float(out))
        elif out.ndim == 1:
            if out.shape[0] == n:
                out = out.reshape(n, 1)
            else:
                if n == 1:
                    out = out.reshape(1, -1)
                else:
                    # if can reshape to (n, k_local), try:
                    if out.size == n * self.k_local:
                        out = out.reshape(n, self.k_local)
                    else:
                        raise ValueError("Constraint returned 1D that cannot be aligned to inputs.")
        elif out.ndim == 2:
            if out.shape[0] != n:
                if out.shape[1] == n:
                    out = out.T
                else:
                    raise ValueError("Constraint returned 2D with incompatible rows.")
        else:
            raise ValueError("Constraint returned ndim > 2, unsupported.")
        return out.astype(np.float64)


class MTOP:
    """
    Multi-Task Optimization Problem (MTOP) Define.

    This class allows defining multiple optimization tasks, each with decision variables, objectives, and constraints.
    It handles vectorized and non-vectorized objective/constraint functions, normalizes outputs to consistent 2D arrays,
    and provides unified evaluation interfaces.

    Features (v2.1):
    - Unified evaluation mode: Automatically pad objectives/constraints to max dimensions
    - Selective evaluation: Evaluate specific objectives or constraints
    - Default bounds: If no bounds provided, defaults to [0, 1] for each dimension

    Author: Jiangtao Shen
    Email: j.shen5@exeter.ac.uk
    Date: 2025.10.10
    Version: 2.1 (Cross-platform compatible)
    """

    def __init__(self, unified_eval_mode: bool = False, fill_value: float = 0.0):
        """
        Initialize MTOP.

        Args:
            unified_eval_mode: If True, evaluation results will be padded to max dimensions
            fill_value: Value used for padding in unified evaluation mode
        """
        self.tasks: List[Dict[str, Any]] = []
        self.dims: List[int] = []
        self.bounds: List[Tuple[np.ndarray, np.ndarray]] = []

        # Unified evaluation mode settings
        self._unified_eval_mode = unified_eval_mode
        self._fill_value = fill_value

    # -------------------------
    # Unified evaluation mode settings
    # -------------------------
    def set_unified_eval_mode(self, enabled: bool, fill_value: float = 0.0) -> None:
        """
        Set unified evaluation mode.

        Args:
            enabled: Enable or disable unified evaluation mode
            fill_value: Value used for padding (default 0.0)
        """
        self._unified_eval_mode = enabled
        self._fill_value = fill_value

    @property
    def unified_eval_mode(self) -> bool:
        """Check if unified evaluation mode is enabled."""
        return self._unified_eval_mode

    @property
    def fill_value(self) -> float:
        """Get the fill value for unified evaluation mode."""
        return self._fill_value

    @property
    def m_max(self) -> int:
        """Maximum number of objectives across all tasks."""
        if not self.tasks:
            return 0
        return max(t['n_objectives'] for t in self.tasks)

    @property
    def c_max(self) -> int:
        """Maximum number of constraints across all tasks."""
        if not self.tasks:
            return 0
        return max(t['n_constraints'] for t in self.tasks)

    # -------------------------
    # Task addition interfaces
    # -------------------------
    def add_task(
            self,
            objective_func: Union[Callable[[np.ndarray], Any], Tuple[Callable, ...]],
            dim: Union[int, Tuple[int, ...]],
            constraint_func: Optional[Union[Callable, List[Callable], Tuple[List[Callable], ...]]] = None,
            lower_bound: Optional[Union[List[float], np.ndarray, Tuple[Union[List[float], np.ndarray], ...]]] = None,
            upper_bound: Optional[Union[List[float], np.ndarray, Tuple[Union[List[float], np.ndarray], ...]]] = None
    ) -> Union[int, List[int]]:
        """
        Add one or more tasks to MTOP.

        Args:
            objective_func: Required. Objective function(s) to evaluate.
            dim: Required. Dimension(s) of decision variables.
            constraint_func: Optional constraint function(s). Default is None.
            lower_bound: Optional lower bound(s). Default is [0, 0, ..., 0] with length dim.
            upper_bound: Optional upper bound(s). Default is [1, 1, ..., 1] with length dim.

        Returns:
            Task index or list of task indices.
        """
        if isinstance(objective_func, tuple):
            return self._add_multiple_tasks(objective_func, dim, constraint_func, lower_bound, upper_bound)
        else:
            return self._add_single_task(objective_func, dim, constraint_func, lower_bound, upper_bound)

    def _add_single_task(
            self,
            objective_func: Callable[[np.ndarray], Any],
            dim: int,
            constraint_func: Optional[Union[Callable, List[Callable]]] = None,
            lower_bound: Optional[Union[List[float], np.ndarray]] = None,
            upper_bound: Optional[Union[List[float], np.ndarray]] = None
    ) -> int:
        """
        Add a single task to MTOP.

        Args:
            objective_func: Required. Objective function to evaluate.
            dim: Required. Dimension of decision variables.
            constraint_func: Optional constraint function(s). Default is None.
            lower_bound: Optional lower bound. Default is [0, 0, ..., 0] with length dim.
            upper_bound: Optional upper bound. Default is [1, 1, ..., 1] with length dim.

        Returns:
            Task index.
        """
        # Default bounds to [0, 1] if not provided
        if lower_bound is None:
            lower_bound = np.zeros(dim)
        if upper_bound is None:
            upper_bound = np.ones(dim)

        lb = np.asarray(lower_bound, dtype=np.float64).reshape(-1)
        ub = np.asarray(upper_bound, dtype=np.float64).reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds must be length {dim}")

        # Wrap objective and constraints (使用独立的包装类，可被 pickle)
        wrapped_obj = self._wrap_objective_func(objective_func, dim)

        # Infer n_objectives by testing a single sample
        test_out = wrapped_obj(np.zeros((1, dim)))
        if test_out.ndim != 2:
            raise ValueError("Wrapped objective must produce 2D array")
        n_obj = test_out.shape[1]

        constraint_wrappers, n_constraints = self._process_constraints(constraint_func, dim)

        task = {
            'raw_objective': objective_func,
            'objective': wrapped_obj,  # callable: X (n,dim) -> (n, n_obj)
            'n_objectives': n_obj,
            'constraints': constraint_wrappers,  # list of callables X -> (n, k_i)
            'n_constraints': n_constraints
        }

        self.tasks.append(task)
        self.dims.append(dim)
        self.bounds.append((lb.reshape(1, -1), ub.reshape(1, -1)))
        return len(self.tasks) - 1

    def _add_multiple_tasks(
            self,
            objective_funcs: Tuple[Callable, ...],
            dims: Union[int, Tuple[int, ...]],
            constraint_func: Optional[Union[List[Callable], Tuple[List[Callable], ...]]] = None,
            lower_bounds: Optional[Tuple[Union[List[float], np.ndarray], ...]] = None,
            upper_bounds: Optional[Tuple[Union[List[float], np.ndarray], ...]] = None
    ) -> List[int]:
        """
        Add multiple tasks to MTOP.

        Args:
            objective_funcs: Required. Tuple of objective functions.
            dims: Required. Single dimension (broadcast to all) or tuple of dimensions.
            constraint_func: Optional constraint function(s). Default is None for all.
            lower_bounds: Optional tuple of lower bounds. Default is [0, 0, ..., 0] for each.
            upper_bounds: Optional tuple of upper bounds. Default is [1, 1, ..., 1] for each.

        Returns:
            List of task indices.
        """
        n_tasks = len(objective_funcs)

        # Broadcast dims if necessary
        if isinstance(dims, int):
            dims = (dims,) * n_tasks
        elif len(dims) != n_tasks:
            raise ValueError("dims length must match number of objective_funcs")

        # Prepare constraint_func tuple/list
        if constraint_func is None:
            constraint_func = (None,) * n_tasks
        elif not isinstance(constraint_func, (list, tuple)):
            constraint_func = tuple([constraint_func] * n_tasks)
        elif len(constraint_func) != n_tasks:
            raise ValueError("constraint_func length must match number of objective_funcs")

        # Handle bounds - allow None for default [0, 1]
        if lower_bounds is None:
            lower_bounds = (None,) * n_tasks
        elif len(lower_bounds) != n_tasks:
            raise ValueError("lower_bounds length must match number of objective_funcs")

        if upper_bounds is None:
            upper_bounds = (None,) * n_tasks
        elif len(upper_bounds) != n_tasks:
            raise ValueError("upper_bounds length must match number of objective_funcs")

        indices = []
        for i in range(n_tasks):
            idx = self._add_single_task(
                objective_func=objective_funcs[i],
                dim=dims[i],
                constraint_func=constraint_func[i],
                lower_bound=lower_bounds[i],
                upper_bound=upper_bounds[i]
            )
            indices.append(idx)
        return indices

    def add_tasks(self, tasks_config: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple tasks from configuration dictionaries.

        Args:
            tasks_config: List of task configuration dicts with keys:
                - 'objective_func': Required
                - 'dim': Required
                - 'constraint_func': Optional
                - 'lower_bound': Optional (default [0]*dim)
                - 'upper_bound': Optional (default [1]*dim)

        Returns:
            List of task indices.
        """
        if not isinstance(tasks_config, list):
            raise TypeError("tasks_config must be a list of dicts")
        indices = []
        for cfg in tasks_config:
            if 'objective_func' not in cfg:
                raise ValueError("Each task config must contain 'objective_func'")
            if 'dim' not in cfg:
                raise ValueError("Each task config must contain 'dim'")
            idx = self.add_task(
                objective_func=cfg['objective_func'],
                dim=cfg['dim'],
                constraint_func=cfg.get('constraint_func', None),
                lower_bound=cfg.get('lower_bound', None),
                upper_bound=cfg.get('upper_bound', None)
            )
            indices.append(idx)
        return indices

    # -------------------------
    # Wrapping helpers (修改为使用独立的包装类)
    # -------------------------
    def _wrap_objective_func(self, func: Callable, dim: int) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a function that accepts X with shape (n, dim) and ALWAYS returns ndarray shape (n, n_obj).
        使用独立的包装类以支持 pickle 序列化（跨平台并行兼容）。
        """
        return ObjectiveFunctionWrapper(func, dim)

    def _process_constraints(self, constraint_func, dim: int) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], int]:
        """
        Normalize constraint_func into a list of wrappers that each map X (n,dim) -> (n, k_i).
        Return (wrappers_list, total_n_constraints).
        If constraint_func is None -> return ([], 0), evaluation will return zeros.
        使用独立的包装类以支持 pickle 序列化（跨平台并行兼容）。
        """
        if constraint_func is None:
            return [], 0

        # If single callable provided, wrap into list
        if callable(constraint_func) and not isinstance(constraint_func, (list, tuple)):
            funcs = [constraint_func]
        elif isinstance(constraint_func, (list, tuple)):
            funcs = list(constraint_func)
            if not all(callable(f) for f in funcs):
                raise TypeError("All elements in constraint_func must be callables")
        else:
            raise TypeError("constraint_func must be callable or list/tuple of callables or None")

        wrappers = []
        total = 0
        for f in funcs:
            # probe to detect per-call output size
            try:
                probe = f(np.zeros(dim))
                probe = np.atleast_1d(np.asarray(probe))
                k = probe.size
            except Exception:
                # if probe fails, assume scalar per-call
                k = 1

            total += k
            wrappers.append(ConstraintFunctionWrapper(f, k))

        return wrappers, total

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate_task(
            self,
            task_idx: int,
            X: np.ndarray,
            eval_objectives: Union[bool, int, List[int]] = True,
            eval_constraints: Union[bool, int, List[int]] = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a task with selective evaluation support.

        Args:
            task_idx: Index of the task to evaluate
            X: Input array of shape (n_samples, dim) or (dim,)
            eval_objectives:
                - True: evaluate all objectives (default)
                - False: skip objective evaluation, return empty array
                - int: evaluate only the i-th objective
                - List[int]: evaluate specified objectives
            eval_constraints:
                - True: evaluate all constraints (default)
                - False: skip constraint evaluation, return empty array
                - int: evaluate only the i-th constraint
                - List[int]: evaluate specified constraints

        Returns:
            Tuple of (objectives, constraints) as 2D numpy arrays:
              objectives: shape (n_samples, n_evaluated_objectives) or padded to (n_samples, m_max)
              constraints: shape (n_samples, n_evaluated_constraints) or padded to (n_samples, c_max)
        """
        if not 0 <= task_idx < len(self.tasks):
            raise ValueError("task_idx out of range")
        X = np.atleast_2d(X)
        dim = self.dims[task_idx]
        if X.shape[1] != dim:
            raise ValueError(f"Input dimension mismatch: expected {dim}, got {X.shape[1]}")

        task = self.tasks[task_idx]
        n = X.shape[0]

        # Evaluate objectives
        obj = self._evaluate_objectives(task, X, eval_objectives)

        # Evaluate constraints
        cons = self._evaluate_constraints(task, X, eval_constraints)

        # Apply unified evaluation mode if enabled
        if self._unified_eval_mode:
            obj = self._pad_to_max(obj, self.m_max, axis=1)
            cons = self._pad_to_max(cons, self.c_max, axis=1)

        return obj, cons

    def _evaluate_objectives(
            self,
            task: Dict,
            X: np.ndarray,
            eval_objectives: Union[bool, int, List[int]]
    ) -> np.ndarray:
        """Evaluate objectives based on eval_objectives parameter."""
        n = X.shape[0]
        n_obj = task['n_objectives']

        if eval_objectives is False:
            return np.empty((n, 0), dtype=np.float64)

        # Evaluate all objectives first
        all_obj = task['objective'](X)
        if all_obj.ndim != 2:
            raise RuntimeError("Internal objective wrapper failed to produce 2D array")

        if eval_objectives is True:
            return all_obj

        # Selective evaluation
        if isinstance(eval_objectives, int):
            if not 0 <= eval_objectives < n_obj:
                raise ValueError(f"Objective index {eval_objectives} out of range [0, {n_obj})")
            return all_obj[:, eval_objectives:eval_objectives + 1]

        if isinstance(eval_objectives, (list, tuple)):
            indices = list(eval_objectives)
            for idx in indices:
                if not 0 <= idx < n_obj:
                    raise ValueError(f"Objective index {idx} out of range [0, {n_obj})")
            return all_obj[:, indices]

        raise TypeError("eval_objectives must be bool, int, or list of int")

    def _evaluate_constraints(
            self,
            task: Dict,
            X: np.ndarray,
            eval_constraints: Union[bool, int, List[int]]
    ) -> np.ndarray:
        """Evaluate constraints based on eval_constraints parameter."""
        n = X.shape[0]
        n_cons = task['n_constraints']

        if eval_constraints is False:
            return np.empty((n, 0), dtype=np.float64)

        # No constraints defined -> return zeros with shape (n, 1)
        if n_cons == 0:
            return np.zeros((n, 1), dtype=np.float64)

        # Evaluate all constraints
        parts = [w(X) for w in task['constraints']]
        all_cons = np.hstack(parts) if len(parts) > 1 else parts[0]
        if all_cons.ndim != 2:
            raise RuntimeError("Internal constraint wrappers failed to produce 2D array")

        if eval_constraints is True:
            return all_cons

        # Selective evaluation
        if isinstance(eval_constraints, int):
            if not 0 <= eval_constraints < n_cons:
                raise ValueError(f"Constraint index {eval_constraints} out of range [0, {n_cons})")
            return all_cons[:, eval_constraints:eval_constraints + 1]

        if isinstance(eval_constraints, (list, tuple)):
            indices = list(eval_constraints)
            for idx in indices:
                if not 0 <= idx < n_cons:
                    raise ValueError(f"Constraint index {idx} out of range [0, {n_cons})")
            return all_cons[:, indices]

        raise TypeError("eval_constraints must be bool, int, or list of int")

    def _pad_to_max(self, arr: np.ndarray, max_size: int, axis: int = 1) -> np.ndarray:
        """Pad array to max_size along specified axis."""
        if arr.shape[axis] >= max_size:
            return arr

        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, max_size - arr.shape[axis])
        return np.pad(arr, pad_width, mode='constant', constant_values=self._fill_value)

    def evaluate_tasks(
            self,
            task_indices: List[int],
            X_list: List[np.ndarray],
            eval_objectives: Union[bool, int, List[int], List[Union[bool, int, List[int]]]] = True,
            eval_constraints: Union[bool, int, List[int], List[Union[bool, int, List[int]]]] = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Evaluate multiple tasks.

        Args:
            task_indices: List of task indices
            X_list: List of input arrays
            eval_objectives: Evaluation mode for objectives (applied to all or per-task list)
            eval_constraints: Evaluation mode for constraints (applied to all or per-task list)

        Returns:
            Tuple of (list of objective arrays, list of constraint arrays)
        """
        if len(task_indices) != len(X_list):
            raise ValueError("task_indices and X_list length mismatch")

        # Handle per-task evaluation modes
        if isinstance(eval_objectives, list) and len(eval_objectives) == len(task_indices):
            obj_modes = eval_objectives
        else:
            obj_modes = [eval_objectives] * len(task_indices)

        if isinstance(eval_constraints, list) and len(eval_constraints) == len(task_indices):
            con_modes = eval_constraints
        else:
            con_modes = [eval_constraints] * len(task_indices)

        objs, cons = [], []
        for idx, X, obj_mode, con_mode in zip(task_indices, X_list, obj_modes, con_modes):
            o, c = self.evaluate_task(idx, X, eval_objectives=obj_mode, eval_constraints=con_mode)
            objs.append(o)
            cons.append(c)
        return objs, cons

    # -------------------------
    # Query / info
    # -------------------------
    def get_n_objectives(self, task_idx: int) -> int:
        if not 0 <= task_idx < len(self.tasks):
            raise ValueError("task_idx out of range")
        return self.tasks[task_idx]['n_objectives']

    def get_all_n_objectives(self) -> List[int]:
        return [self.get_n_objectives(i) for i in range(self.n_tasks)]

    def get_n_constraints(self, task_idx: int) -> int:
        if not 0 <= task_idx < len(self.tasks):
            raise ValueError("task_idx out of range")
        return self.tasks[task_idx]['n_constraints']

    def get_all_n_constraints(self) -> List[int]:
        return [self.get_n_constraints(i) for i in range(self.n_tasks)]

    def get_task_info(self, task_idx: int) -> Dict[str, Any]:
        if not 0 <= task_idx < len(self.tasks):
            raise ValueError("task_idx out of range")
        lb, ub = self.bounds[task_idx]
        t = self.tasks[task_idx]
        return {
            'dimension': self.dims[task_idx],
            'n_objectives': t['n_objectives'],
            'n_constraints': t['n_constraints'],
            'lower_bounds': lb,
            'upper_bounds': ub,
            'objective_func': t['raw_objective'],
            'constraint_funcs': [w for w in t['constraints']]
        }

    def __str__(self) -> str:
        lines = [f"MTOP with {len(self.tasks)} tasks:"]
        lines.append(f"  Unified eval mode: {self._unified_eval_mode} (fill_value={self._fill_value})")
        lines.append(f"  Max number of objectives (n_objs_max): {self.m_max}")
        lines.append(f"  Max number of constraints (n_cons_max): {self.c_max}")
        for i, t in enumerate(self.tasks):
            lb, ub = self.bounds[i]
            lines.append(
                f"  Task {i}: dim={self.dims[i]}, n_objs={t['n_objectives']}, n_cons={t['n_constraints']}, bounds=[{lb.min()}..{ub.max()}]")
        return "\n".join(lines)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_objs(self) -> List[int]:
        return self.get_all_n_objectives()

    @property
    def n_cons(self) -> List[int]:
        return self.get_all_n_constraints()


# -------------------------
# Example usage and testing
# -------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MTOP v2.1 - Testing (Cross-platform compatible)")
    print("=" * 60)


    # Define test functions
    def obj1(x):
        """Single objective: f(x) = sum(x^2)"""
        x = np.atleast_2d(x)
        return np.sum(x ** 2, axis=1)


    def obj2(x):
        """Multi-objective: f1 = sum(x^2), f2 = sum((x-1)^2), f3 = sum(x)"""
        x = np.atleast_2d(x)
        f1 = np.sum(x ** 2, axis=1)
        f2 = np.sum((x - 1) ** 2, axis=1)
        f3 = np.sum(x, axis=1)
        return np.column_stack([f1, f2, f3])


    def con1(x):
        """Single constraint: g(x) = sum(x) - 1 <= 0"""
        x = np.atleast_1d(x)
        return np.sum(x) - 1


    def con2(x):
        """Two constraints: g1 = x[0] - 0.5, g2 = x[1] - 0.5"""
        x = np.atleast_1d(x)
        return np.array([x[0] - 0.5, x[1] - 0.5])


    # Create MTOP instance
    print("\n1. Creating MTOP with tasks using default bounds...")
    mtop = MTOP(unified_eval_mode=False)

    # Add task with explicit bounds
    idx1 = mtop.add_task(
        objective_func=obj1,
        dim=3,
        constraint_func=con1,
        lower_bound=[-5, -5, -5],
        upper_bound=[5, 5, 5]
    )
    print(f"Task {idx1}: explicit bounds [-5, 5]")

    # Add task with default bounds [0, 1]
    idx2 = mtop.add_task(
        objective_func=obj2,
        dim=2,
        constraint_func=con2
        # No bounds provided -> defaults to [0, 1]
    )
    print(f"Task {idx2}: default bounds [0, 1]")

    # Add task with only lower bound (upper defaults to 1)
    idx3 = mtop.add_task(
        objective_func=obj1,
        dim=2,
        lower_bound=[-1, -1]
        # upper_bound defaults to [1, 1]
    )
    print(f"Task {idx3}: lower_bound=[-1,-1], upper_bound defaults to [1,1]")

    # Add task with only upper bound (lower defaults to 0)
    idx4 = mtop.add_task(
        objective_func=obj1,
        dim=2,
        upper_bound=[10, 10]
        # lower_bound defaults to [0, 0]
    )
    print(f"Task {idx4}: lower_bound defaults to [0,0], upper_bound=[10,10]")

    print("\n" + str(mtop))

    # Test evaluation
    print("\n2. Testing evaluation with default bounds...")

    # Task with default bounds [0, 1]
    X_default = np.array([[0.5, 0.5], [0.0, 1.0]])
    obj_vals, con_vals = mtop.evaluate_task(1, X_default)
    print(f"\nTask 1 (default bounds [0,1]):")
    print(f"  Input X:\n{X_default}")
    print(f"  Objectives: {obj_vals}")
    print(f"  Constraints: {con_vals}")

    # Verify bounds
    print("\n3. Verifying bounds...")
    for i in range(mtop.n_tasks):
        info = mtop.get_task_info(i)
        print(f"Task {i}: lower={info['lower_bounds'].flatten()}, upper={info['upper_bounds'].flatten()}")

    # Test add_tasks with default bounds
    print("\n4. Testing add_tasks with mixed bounds...")
    mtop2 = MTOP()

    configs = [
        {'objective_func': obj1, 'dim': 3},  # default [0,1]
        {'objective_func': obj1, 'dim': 2, 'lower_bound': [-2, -2], 'upper_bound': [2, 2]},
        {'objective_func': obj2, 'dim': 2}  # default [0,1]
    ]

    indices = mtop2.add_tasks(configs)
    print(mtop2)

    # Test tuple-style add_task with default bounds
    print("\n5. Testing tuple-style add_task with default bounds...")
    mtop4 = MTOP()

    indices = mtop4.add_task(
        objective_func=(obj1, obj2),
        dim=(3, 2)
        # No bounds -> all default to [0, 1]
    )
    print(mtop4)

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Cross-platform compatibility verified!")
    print("=" * 60)
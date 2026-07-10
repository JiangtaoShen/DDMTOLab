"""Algorithm-problem compatibility checks based on algorithm_information.

algorithm_information uses string descriptors like:
    'n_tasks': '[1, K]' / '[2, K]'      (allowed task-count range)
    'n_objs':  '1' / '[2, M]' / '[2, 3]' (allowed objectives per task)
    'dims':    'equal' / 'unequal'       (whether task dims must be equal)
"""
import re
from typing import List, Optional, Tuple


def _parse_range(value) -> Optional[Tuple[int, Optional[int]]]:
    """Parse '1', '[2, K]', '[2, 3]' into (min, max); max None means open."""
    if value is None:
        return None
    s = str(value).strip()
    m = re.match(r'^\[\s*(\d+)\s*,\s*(\w+)\s*\]$', s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2).isdigit() else None
        return lo, hi
    if s.isdigit():
        v = int(s)
        return v, v
    return None


def check_algorithm_compatibility(problem, info: dict) -> List[str]:
    """Return human-readable incompatibility reasons (empty list = compatible)."""
    issues = []
    if not info:
        return issues

    dims = list(getattr(problem, 'dims', []) or [])
    n_tasks = len(dims)

    # Task count
    rng = _parse_range(info.get('n_tasks'))
    if rng and n_tasks:
        lo, hi = rng
        if n_tasks < lo or (hi is not None and n_tasks > hi):
            issues.append(f"requires {info['n_tasks']} tasks, problem has {n_tasks}")

    # Objectives per task
    task_objs = []
    for t in getattr(problem, 'tasks', []) or []:
        if isinstance(t, dict) and t.get('n_objectives'):
            task_objs.append(int(t['n_objectives']))
    rng = _parse_range(info.get('n_objs'))
    if rng and task_objs:
        lo, hi = rng
        for i, m in enumerate(task_objs):
            if m < lo or (hi is not None and m > hi):
                issues.append(f"supports {info['n_objs']} objectives, task {i + 1} has {m}")
                break

    # Equal-dimension requirement
    if str(info.get('dims', '')).lower() == 'equal' and len(set(dims)) > 1:
        issues.append(f"requires equal task dimensions, problem dims are {dims}")

    return issues

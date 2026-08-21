"""Algorithm and problem registry for DDMTOLab UI.

This module uses auto-discovery to scan problem/algorithm packages and find classes.
Add new problems/algorithms by simply creating new files following the standard patterns.

Special handling for RWO category which combines algorithms from MTSO and MTMO.
"""

import importlib
from typing import Dict, Type, Any, List, Tuple, Optional

# Import scanners for auto-discovery
from utils.problem_scanner import (
    scan_all_problems, get_scanned_problem_suites, get_scanned_problem_methods,
    get_scanned_problem_params, get_problem_class, create_problem_from_scan,
    is_fixed_dimension_problem, map_method_kwargs,
    _filter_kwargs_for_callable,
    get_problem_module_path as get_scanned_problem_module_path,
)
from utils.algo_scanner import (
    discover_all_algorithms, get_discovered_algorithm_names,
    get_discovered_algorithm_class, get_discovered_algorithm_module_info,
)
from config.constants import CATEGORIES


def get_algorithm_names(category: str) -> List[str]:
    """Return list of algorithm display names for a category."""
    # RWO uses algorithms from both MTSO and MTMO
    if category == "RWO":
        mtso_algos = get_discovered_algorithm_names("MTSO")
        mtmo_algos = get_discovered_algorithm_names("MTMO")
        # Combine and deduplicate while preserving order
        seen = set()
        result = []
        for algo in mtso_algos + mtmo_algos:
            if algo not in seen:
                seen.add(algo)
                result.append(algo)
        return result

    return get_discovered_algorithm_names(category)


def get_algorithm_class(category: str, name: str) -> Type:
    """Lazily import and return an algorithm class."""
    # RWO: try MTSO first, then MTMO
    if category == "RWO":
        cls = get_discovered_algorithm_class("MTSO", name)
        if cls:
            return cls
        cls = get_discovered_algorithm_class("MTMO", name)
        if cls:
            return cls
        raise KeyError(f"Algorithm '{name}' not found in MTSO or MTMO")

    cls = get_discovered_algorithm_class(category, name)
    if cls:
        return cls
    raise KeyError(f"Algorithm '{name}' not found in category '{category}'")


def get_problem_suites(category: str) -> List[str]:
    """Return list of problem suite names for a category."""
    return get_scanned_problem_suites(category)


def get_problem_methods(category: str, suite: str) -> List[str]:
    """Return list of problem method names (P1, P2, ...) for a suite."""
    return get_scanned_problem_methods(category, suite)


def create_problem(category: str, suite: str, method: str, **kwargs):
    """Instantiate a problem suite class and call the specified method to get an MTOP."""
    return create_problem_from_scan(category, suite, method, **kwargs)


def get_problem_creator(category: str, suite: str, method: str, **kwargs) -> Tuple[callable, str, dict]:
    """Return (problem_creator_fn, display_name, creator_kwargs) for BatchExperiment.add_problem().

    creator_kwargs are the UI parameters mapped to the method's original names
    and filtered to what the method signature accepts, so they can be passed
    directly as ``add_problem(creator, name, **creator_kwargs)``.
    """
    cls = get_problem_class(category, suite)
    if cls is None:
        raise KeyError(f"Problem suite '{suite}' not found in category '{category}'")

    instance = cls()
    creator = getattr(instance, method)

    mapped = map_method_kwargs(category, suite, method, kwargs)
    creator_kwargs = _filter_kwargs_for_callable(creator, mapped)

    display_name = f"{suite}_{method}"
    return creator, display_name, creator_kwargs


def get_all_categories() -> List[str]:
    """Return all available categories."""
    return CATEGORIES


def is_multi_objective(category: str) -> bool:
    """Check if a category is multi-objective.

    RWO mixes single- and multiobjective suites, so this returns False for it.
    Use :func:`suite_is_multi_objective` when a suite is known.
    """
    return category in ["STMO", "MTMO"]


def suite_is_multi_objective(category: str, suite: str) -> bool:
    """Check if a specific suite is multi-objective.

    STMO and MTMO are multiobjective throughout. RWO holds both kinds, so decide
    from the suite itself: the multiobjective ones ship the reference data that
    IGD and HV need, under the module-level SETTINGS dict.
    """
    if is_multi_objective(category):
        return True
    return get_problem_settings(category, suite) is not None


def is_multi_task(category: str) -> bool:
    """Check if a category is multi-task."""
    return category in ["MTSO", "MTMO", "RWO"]


def get_problem_settings(category: str, suite: str) -> Optional[dict]:
    """
    Get SETTINGS dict from a problem module for metric calculation.

    Args:
        category: Problem category (STSO, STMO, MTSO, MTMO)
        suite: Problem suite name (e.g., 'DTLZ', 'ZDT')

    Returns:
        SETTINGS dict if available, None otherwise.
    """
    mod_path = get_scanned_problem_module_path(category, suite)
    if not mod_path:
        return None

    try:
        mod = importlib.import_module(mod_path)
        if hasattr(mod, 'SETTINGS'):
            return mod.SETTINGS.copy()
    except Exception:
        pass
    return None


def get_problem_module_path(category: str, suite: str) -> Optional[str]:
    """Get the module path for a problem suite."""
    return get_scanned_problem_module_path(category, suite)

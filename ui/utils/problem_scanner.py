"""Problem scanner - automatically discovers problem classes and their methods."""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

# Standard parameter definitions
# D = Decision variables, M = Objectives, K = Tasks
# Kp = Position parameters (WFG-specific, avoids conflict with K)
STANDARD_PARAMS = {
    'D': {'type': 'int', 'default': 50},
    'M': {'type': 'int', 'default': 3},
    'K': {'type': 'int', 'default': 10},
    'Kp': {'type': 'int', 'default': 4},
}

# Parameter aliases (legacy name -> standard name). Every suite shipped with
# the platform uses the standard names; this map only keeps user-authored
# problem classes working.
PARAM_ALIASES = {
    'dim': 'D',
    'task_num': 'K',
    'num_of_objective': 'M',
}

# Category to package mapping
CATEGORY_PACKAGES = {
    'STSO': 'ddmtolab.Problems.STSO',
    'STMO': 'ddmtolab.Problems.STMO',
    'MTSO': 'ddmtolab.Problems.MTSO',
    'MTMO': 'ddmtolab.Problems.MTMO',
    'RWO': 'ddmtolab.Problems.RWO',
}

# Classes to skip (not problem classes)
SKIP_CLASSES = {'MTOP', 'LZ09', 'TemplateProblems'}

# Modules to skip
SKIP_MODULES = {'TEMPLATE_PROBLEM', 'template_problem'}

# Every benchmark problem is a public method annotated ``-> MTOP``. Helper
# methods on a suite (plotting, data loading, ...) are recognised by the
# absence of that annotation and are never offered as problems.
PROBLEM_RETURN_ANNOTATION = 'MTOP'

# Cache for scanned results
_problem_cache: Dict[str, Dict] = {}
_scanned = False


def scan_problem_method(method: Callable) -> Dict[str, Dict]:
    """
    Extract D, M, K, Kp parameters from a problem method signature.

    Parameters
    ----------
    method : Callable
        The problem method to scan (e.g., ZDT().ZDT1)

    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping parameter names to their metadata

    Notes
    -----
    A ``None`` default means the suite derives that value from the other
    parameters (DTLZ and WFG derive D from M). There is no number to show for
    it, so the platform default from ``STANDARD_PARAMS`` is reported instead -
    the value the caller will actually get if the control is left alone.
    """
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return {}

    params = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue

        # Map aliases to standard names
        std_name = PARAM_ALIASES.get(name, name)

        if std_name in STANDARD_PARAMS:
            default = param.default
            if default is inspect.Parameter.empty or default is None:
                default = STANDARD_PARAMS[std_name]['default']
            params[std_name] = {
                'type': STANDARD_PARAMS[std_name]['type'],
                'default': default,
                'description': std_name,  # Use param name directly
                'original_name': name,  # Keep for calling the method
            }
    return params


def _is_problem_method(method_name: str, method_obj: Any) -> bool:
    """
    Check if a method is a problem creation method.

    A problem method is a public method whose return annotation is ``MTOP``.
    The annotation is compared by name so that a string annotation (from
    ``from __future__ import annotations``) is accepted too.
    """
    # Skip private/magic methods
    if method_name.startswith('_'):
        return False

    # Skip staticmethod
    if isinstance(method_obj, staticmethod):
        return False

    try:
        annotation = inspect.signature(method_obj).return_annotation
    except (ValueError, TypeError):
        return False

    if annotation is inspect.Signature.empty:
        return False

    name = getattr(annotation, '__name__', None) or str(annotation)
    return name.rsplit('.', 1)[-1] == PROBLEM_RETURN_ANNOTATION


def _scan_class(cls: type) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Scan a class for problem methods.

    Returns:
        Tuple of (method_names, {method: params})
    """
    methods = []
    all_params = {}

    for name in dir(cls):
        if name.startswith('_'):
            continue

        try:
            attr = getattr(cls, name)
        except Exception:
            continue

        # Check if it's a method
        if not callable(attr):
            continue

        # Skip static methods and class methods
        if isinstance(inspect.getattr_static(cls, name), (staticmethod, classmethod)):
            continue

        if not _is_problem_method(name, attr):
            continue

        methods.append(name)

        # Try to get params from instance method
        try:
            instance = cls()
            method = getattr(instance, name)
            params = scan_problem_method(method)
            if params:
                all_params[name] = params
        except Exception:
            pass

    # Sort methods naturally
    methods.sort(key=lambda x: (len(x), x))
    return methods, all_params


def _scan_module(module_path: str) -> Dict[str, Tuple[str, str, List[str], Dict]]:
    """
    Scan a module for problem classes.

    Returns:
        Dict mapping class_name to (module_path, class_name, [methods], {method_params})
    """
    results = {}

    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        logger.debug(f"Could not import {module_path}: {e}")
        return results

    # Find all classes in the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Skip if not defined in this module
        if obj.__module__ != module_path:
            continue

        # Skip utility classes
        if name in SKIP_CLASSES:
            continue

        # Only include classes with problem_information metadata
        if not hasattr(obj, 'problem_information'):
            continue

        # Scan class for methods
        methods, method_params = _scan_class(obj)

        if methods:
            results[name] = (module_path, name, methods, method_params)

    return results


def _scan_category(category: str) -> Dict[str, Tuple[str, str, List[str], Dict]]:
    """Scan all modules in a category package."""
    results = {}
    package_name = CATEGORY_PACKAGES.get(category)

    if not package_name:
        return results

    try:
        package = importlib.import_module(package_name)
    except Exception as e:
        logger.debug(f"Could not import package {package_name}: {e}")
        return results

    # Get package path
    if not hasattr(package, '__path__'):
        return results

    # Scan all modules in the package
    for importer, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg:
            continue
        if module_name.startswith('_'):
            continue
        if module_name in SKIP_MODULES:
            continue

        full_module_path = f"{package_name}.{module_name}"
        module_results = _scan_module(full_module_path)
        results.update(module_results)

    return results


def scan_all_problems(force: bool = False) -> Dict[str, Dict]:
    """
    Scan all problem categories and return discovered problems.

    Returns:
        Dict mapping category to {suite_name: (module_path, class_name, [methods], {params})}
    """
    global _problem_cache, _scanned

    if _scanned and not force:
        return _problem_cache

    _problem_cache = {}

    for category in CATEGORY_PACKAGES.keys():
        _problem_cache[category] = _scan_category(category)

    _scanned = True
    return _problem_cache


def get_scanned_problem_suites(category: str) -> List[str]:
    """Get list of problem suite names for a category from scanner."""
    if not _scanned:
        scan_all_problems()
    return list(_problem_cache.get(category, {}).keys())


def get_scanned_problem_methods(category: str, suite: str) -> List[str]:
    """Get list of problem method names for a suite from scanner."""
    if not _scanned:
        scan_all_problems()
    suite_info = _problem_cache.get(category, {}).get(suite)
    if suite_info:
        return suite_info[2]
    return []


def get_scanned_problem_params(category: str, suite: str) -> Dict[str, Dict]:
    """
    Get parameters for a problem suite (union of all method params).

    Returns:
        Dict mapping param_name to {type, default, description}
    """
    if not _scanned:
        scan_all_problems()

    suite_info = _problem_cache.get(category, {}).get(suite)
    if not suite_info:
        return {}

    # Merge params from all methods
    all_params = {}
    method_params = suite_info[3]

    for method, params in method_params.items():
        for param_name, param_info in params.items():
            if param_name not in all_params:
                all_params[param_name] = param_info.copy()

    return all_params


def get_problem_class(category: str, suite: str) -> Optional[type]:
    """Get the problem class for a suite."""
    if not _scanned:
        scan_all_problems()

    suite_info = _problem_cache.get(category, {}).get(suite)
    if not suite_info:
        return None

    module_path, class_name = suite_info[0], suite_info[1]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception:
        return None


def get_problem_module_path(category: str, suite: str) -> Optional[str]:
    """Get the module path for a problem suite."""
    if not _scanned:
        scan_all_problems()

    suite_info = _problem_cache.get(category, {}).get(suite)
    if suite_info:
        return suite_info[0]
    return None


def is_known_problem_suite(category: str, suite: str) -> bool:
    """
    Check whether the scanner found this suite.

    A suite can be known and still report no parameters at all (CEC17MTSO and
    friends are fixed by their benchmark definition), so callers must test this
    rather than treating an empty parameter dict as "scan failed".
    """
    if not _scanned:
        scan_all_problems()
    return suite in _problem_cache.get(category, {})


def is_fixed_dimension_problem(category: str, suite: str) -> bool:
    """Check if a problem suite has fixed dimensions (no D parameter)."""
    params = get_scanned_problem_params(category, suite)
    return 'D' not in params


def _filter_kwargs_for_callable(func: Callable, kwargs: dict) -> dict:
    """Keep only the kwargs that func's signature accepts (all if it takes **kwargs)."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return dict(kwargs)

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return dict(kwargs)

    accepted = {
        name for name, p in sig.parameters.items()
        if name != 'self' and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                         inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in accepted}


def map_method_kwargs(category: str, suite: str, method: str, kwargs: dict) -> dict:
    """Map standard param names (D/M/K/Kp) to the method's original parameter names."""
    if not _scanned:
        scan_all_problems()

    suite_info = _problem_cache.get(category, {}).get(suite)
    if not suite_info:
        return dict(kwargs)

    method_params = suite_info[3].get(method, {})
    mapped = {}
    for key, value in kwargs.items():
        original_name = method_params.get(key, {}).get('original_name', key)
        mapped[original_name] = value
    return mapped


def create_problem_from_scan(category: str, suite: str, method: str, **kwargs):
    """Create a problem instance using scanned info."""
    cls = get_problem_class(category, suite)
    if cls is None:
        raise ValueError(f"Problem suite not found: {category}/{suite}")

    # Map standard param names (D/M/K/Kp) back to the method's original names
    kwargs = map_method_kwargs(category, suite, method, kwargs)

    instance = cls()

    method_func = getattr(instance, method, None)
    if method_func is None:
        raise ValueError(f"Method not found: {suite}.{method}")

    # Only pass params the method actually accepts; unknown params for this
    # method (from the suite-wide union) are dropped individually instead of
    # discarding every parameter on a TypeError.
    call_kwargs = _filter_kwargs_for_callable(method_func, kwargs)
    return method_func(**call_kwargs)

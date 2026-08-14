"""Algorithm scanner - automatically discovers algorithm classes and extracts parameters."""

import ast
import os
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)

# Cache for scanned parameters
_PARAM_CACHE: Dict[str, Dict[str, Dict]] = {}
_SCANNED = False

# Parameters to exclude from UI (handled internally)
EXCLUDE_PARAMS = {'problem', 'save_data', 'save_path', 'name', 'disable_tqdm', 'self'}


def get_param_type(value: Any) -> str:
    """Infer parameter type from default value."""
    if value is None:
        return 'int'
    elif isinstance(value, bool):
        return 'bool'
    elif isinstance(value, int):
        return 'int'
    elif isinstance(value, float):
        return 'float'
    elif isinstance(value, str):
        return 'str'
    else:
        return 'any'


def _find_target_init(tree: ast.Module, class_name: Optional[str] = None) -> Optional[ast.FunctionDef]:
    """
    Find the __init__ of the algorithm class, not of helper classes.

    Priority: class named class_name -> class with algorithm_information ->
    first __init__ anywhere (legacy fallback).
    """
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    def init_of(cls_node):
        for item in cls_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                return item
        return None

    if class_name:
        for c in classes:
            if c.name == class_name:
                init = init_of(c)
                if init is not None:
                    return init

    for c in classes:
        has_info = any(
            isinstance(item, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'algorithm_information'
                for t in item.targets)
            for item in c.body
        )
        if has_info:
            init = init_of(c)
            if init is not None:
                return init

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            return node
    return None


def parse_init_from_source(source_code: str, class_name: Optional[str] = None) -> Dict[str, Dict]:
    """
    Parse __init__ method from source code using AST.

    class_name targets the algorithm's own class so helper classes defined
    earlier in the file don't shadow it.

    Returns dict of {param_name: {type, default, description}}.
    """
    params = {}

    try:
        tree = ast.parse(source_code)

        node = _find_target_init(tree, class_name)
        if node is not None:
            # First, extract defaults from function body
            # Pattern: self.x = x if x is not None else DEFAULT
            # Pattern: self.x = DEFAULT if x is None else x
            body_defaults = _extract_body_defaults(node.body)

            # Get arguments
            args = node.args

            # Get default values
            defaults = args.defaults
            num_defaults = len(defaults)
            num_args = len(args.args)

            # Map defaults to arguments (defaults apply to last N arguments)
            for i, arg in enumerate(args.args):
                param_name = arg.arg

                if param_name in EXCLUDE_PARAMS:
                    continue

                # Calculate default index
                default_idx = i - (num_args - num_defaults)

                if default_idx >= 0 and default_idx < len(defaults):
                    default_node = defaults[default_idx]
                    default_value = _eval_ast_node(default_node)
                else:
                    default_value = None

                # If signature default is None, try to get actual default from body
                if default_value is None and param_name in body_defaults:
                    default_value = body_defaults[param_name]

                params[param_name] = {
                    'type': get_param_type(default_value),
                    'default': default_value,
                    'description': param_name,  # Use param name directly
                }
    except Exception as e:
        print(f"Error parsing source: {e}")

    return params


def _extract_body_defaults(body: List) -> Dict[str, Any]:
    """
    Extract actual default values from function body.
    Looks for patterns like:
        self.x = x if x is not None else DEFAULT
        self.x = DEFAULT if x is None else x
    """
    defaults = {}

    for stmt in body:
        if isinstance(stmt, ast.Assign):
            # Check for self.x = ...
            if len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id == 'self':
                        attr_name = target.attr
                        value = stmt.value

                        # Check for ternary expression: x if x is not None else DEFAULT
                        if isinstance(value, ast.IfExp):
                            default_val = _extract_ternary_default(value, attr_name)
                            if default_val is not None:
                                defaults[attr_name] = default_val

    return defaults


def _extract_ternary_default(node: ast.IfExp, param_name: str) -> Any:
    """
    Extract default value from ternary expression.
    Pattern 1: param if param is not None else DEFAULT -> return DEFAULT
    Pattern 2: DEFAULT if param is None else param -> return DEFAULT
    """
    test = node.test
    body = node.body
    orelse = node.orelse

    # Pattern 1: param if param is not None else DEFAULT
    if isinstance(test, ast.Compare):
        if len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot):
            # x is not None
            if len(test.comparators) == 1:
                comp = test.comparators[0]
                if isinstance(comp, ast.Constant) and comp.value is None:
                    # The else part is the default
                    return _eval_ast_node(orelse)
                elif isinstance(comp, ast.NameConstant) and comp.value is None:
                    return _eval_ast_node(orelse)

        elif len(test.ops) == 1 and isinstance(test.ops[0], ast.Is):
            # x is None
            if len(test.comparators) == 1:
                comp = test.comparators[0]
                if isinstance(comp, ast.Constant) and comp.value is None:
                    # The body part is the default
                    return _eval_ast_node(body)
                elif isinstance(comp, ast.NameConstant) and comp.value is None:
                    return _eval_ast_node(body)

    return None


def _eval_ast_node(node) -> Any:
    """Safely evaluate an AST node to get its value."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Num):  # Python 3.7 compatibility
        return node.n
    elif isinstance(node, ast.Str):  # Python 3.7 compatibility
        return node.s
    elif isinstance(node, ast.NameConstant):  # Python 3.7 compatibility
        return node.value
    elif isinstance(node, ast.Name):
        if node.id == 'None':
            return None
        elif node.id == 'True':
            return True
        elif node.id == 'False':
            return False
        return None
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            operand = _eval_ast_node(node.operand)
            if operand is not None:
                return -operand
    return None


def scan_algorithm_file(filepath: str, class_name: Optional[str] = None) -> Dict[str, Dict]:
    """Scan a single algorithm file and extract parameters."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        return parse_init_from_source(source, class_name=class_name)
    except Exception as e:
        print(f"Error scanning {filepath}: {e}")
        return {}


def scan_all_algorithms(base_path: str = None) -> Dict[str, Dict[str, Dict]]:
    """
    Scan all algorithm files and extract parameters.

    Returns:
        Dict mapping "category/algorithm_name" to parameter dict.
    """
    global _PARAM_CACHE, _SCANNED

    if _SCANNED and _PARAM_CACHE:
        return _PARAM_CACHE

    if base_path is None:
        # Try to find the algorithms directory
        current = Path(__file__).resolve().parent
        # Go up to find src/ddmtolab/Algorithms
        for _ in range(5):
            current = current.parent
            algo_path = current / 'src' / 'ddmtolab' / 'Algorithms'
            if algo_path.exists():
                base_path = str(algo_path)
                break

    if base_path is None or not Path(base_path).exists():
        print(f"Algorithm path not found")
        return {}

    algo_base = Path(base_path)
    result = {}

    # Scan each category
    for category in ['STSO', 'STMO', 'MTSO', 'MTMO']:
        cat_path = algo_base / category
        if not cat_path.exists():
            continue

        for py_file in cat_path.glob('*.py'):
            if py_file.name.startswith('_'):
                continue

            # Get algorithm name from filename (algorithm class shares the name)
            algo_name = py_file.stem

            params = scan_algorithm_file(str(py_file), class_name=algo_name)
            if params:
                key = f"{category}/{algo_name}"
                result[key] = params

    _PARAM_CACHE = result
    _SCANNED = True
    return result


def get_algorithm_params_from_scan(category: str, algo_display_name: str) -> Dict[str, Dict]:
    """
    Get parameters for an algorithm using the scanner.

    Args:
        category: Algorithm category (STSO, STMO, MTSO, MTMO)
        algo_display_name: Display name of algorithm (e.g., "NSGA-II")

    Returns:
        Dict of parameters.
    """
    # Ensure algorithms are scanned
    all_params = scan_all_algorithms()

    file_name = DISPLAY_TO_FILE.get(algo_display_name, algo_display_name)
    key = f"{category}/{file_name}"

    if key in all_params:
        return all_params[key]

    # Fallback: try direct name match
    for k, v in all_params.items():
        if k.endswith(f"/{file_name}") or k.endswith(f"/{algo_display_name}"):
            return v

    # Default fallback
    return {
        'n': {'type': 'int', 'default': 100, 'description': 'n'},
        'max_nfes': {'type': 'int', 'default': 10000, 'description': 'max_nfes'},
    }


def get_common_params() -> Dict[str, Dict]:
    """Get common parameters shared by most algorithms."""
    return {
        'max_nfes': {'type': 'int', 'default': 10000, 'description': 'max_nfes'},
    }


def clear_cache():
    """Clear the parameter cache (useful for rescanning)."""
    global _PARAM_CACHE, _SCANNED
    _PARAM_CACHE = {}
    _SCANNED = False


# Cache for algorithm information
_INFO_CACHE: Dict[str, Dict] = {}


def get_algorithm_info(category: str, algo_display_name: str) -> Dict[str, str]:
    """
    Get algorithm_information dict for an algorithm.

    Args:
        category: Algorithm category (STSO, STMO, MTSO, MTMO)
        algo_display_name: Display name of algorithm (e.g., "GA", "NSGA-II")

    Returns:
        Dict with algorithm information or empty dict if not found.
    """
    cache_key = f"{category}/{algo_display_name}"
    if cache_key in _INFO_CACHE:
        return _INFO_CACHE[cache_key]

    file_name = DISPLAY_TO_FILE.get(algo_display_name, algo_display_name)

    # Find algo path
    current = Path(__file__).resolve().parent
    for _ in range(5):
        current = current.parent
        algo_path = current / 'src' / 'ddmtolab' / 'Algorithms' / category / f'{file_name}.py'
        if algo_path.exists():
            break
    else:
        _INFO_CACHE[cache_key] = {}
        return {}

    # Parse algorithm_information from source
    try:
        with open(algo_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == 'algorithm_information':
                                if isinstance(item.value, ast.Dict):
                                    info = {}
                                    for k, v in zip(item.value.keys, item.value.values):
                                        key = _eval_ast_node(k)
                                        val = _eval_ast_node(v)
                                        if key and val:
                                            info[key] = val
                                    _INFO_CACHE[cache_key] = info
                                    return info
    except Exception:
        pass

    _INFO_CACHE[cache_key] = {}
    return {}


def format_algorithm_info(info: Dict[str, str]) -> str:
    """Format algorithm_information dict as readable tooltip text."""
    if not info:
        return "No information available"

    lines = []
    for key, value in info.items():
        lines.append(f"{key}: {value}")

    return "\n".join(lines)


# =============================================================================
# Auto-discovery of algorithms
# =============================================================================

# Category to package mapping
ALGORITHM_PACKAGES = {
    'STSO': 'ddmtolab.Algorithms.STSO',
    'STMO': 'ddmtolab.Algorithms.STMO',
    'MTSO': 'ddmtolab.Algorithms.MTSO',
    'MTMO': 'ddmtolab.Algorithms.MTMO',
}

# Cache for discovered algorithms
_ALGO_REGISTRY_CACHE: Dict[str, Dict[str, Tuple[str, str]]] = {}
_ALGO_DISCOVERED = False

# Modules to skip (templates)
SKIP_ALGO_MODULES = {'TEMPLATE_ALGORITHM', 'template_algorithm'}

# File name to display name mapping (handles special characters)
FILE_TO_DISPLAY = {
    'CMA_ES': 'CMA-ES', 'MA_ES': 'MA-ES', 'IPOP_CMA_ES': 'IPOP-CMA-ES',
    'sep_CMA_ES': 'sep-CMA-ES', 'OpenAI_ES': 'OpenAI-ES', 'SA_COSO': 'SA-COSO',
    'GL_SADE': 'GL-SADE', 'KL_PSO': 'KL-PSO', 'SL_PSO': 'SL-PSO', 'EEI_BO': 'EEI-BO',
    'NSGA_II': 'NSGA-II', 'NSGA_III': 'NSGA-III', 'NSGA_II_SDR': 'NSGA-II-SDR',
    'MOEA_D': 'MOEA/D', 'MOEA_D_STM': 'MOEA/D-STM', 'MOEA_D_FRRMAB': 'MOEA/D-FRRMAB',
    'MOEA_DD': 'MOEA/DD', 'C_TAEA': 'C-TAEA', 'K_RVEA': 'K-RVEA',
    'DSAEA_PS': 'DSAEA-PS', 'MCEA_D': 'MCEA-D', 'CPS_MOEA': 'CPS-MOEA',
    'AS_SMEA': 'AS-SMEA',
    'MFEA_II': 'MFEA-II', 'G_MFEA': 'G-MFEA', 'MTEA_AD': 'MTEA-AD',
    'MTEA_SaO': 'MTEA-SaO', 'LCB_EMT': 'LCB-EMT', 'EEI_BO_plus': 'EEI-BO+',
    'MO_MFEA': 'MO-MFEA', 'MO_MFEA_II': 'MO-MFEA-II', 'MO_EMEA': 'MO-EMEA',
    'EMT_ET': 'EMT-ET', 'EMT_PD': 'EMT-PD', 'MTDE_MKTA': 'MTDE-MKTA',
    'MO_MTEA_SaO': 'MO-MTEA-SaO', 'ParEGO_KT': 'ParEGO-KT', 'MTEA_D_DN': 'MTEA-D-DN',
    'SAEA_AKT': 'SAEA-AKT',
    'TwoArch2': 'Two_Arch2', 'BO_LCB_BCKT': 'BO-LCB-BCKT', 'BO_LCB_CKT': 'BO-LCB-CKT',
}

# Reverse mapping: display name -> file name
DISPLAY_TO_FILE = {v: k for k, v in FILE_TO_DISPLAY.items()}


def _get_display_name(file_name: str, class_name: str) -> str:
    """Convert file/class name to display name."""
    # Try file name mapping first
    if file_name in FILE_TO_DISPLAY:
        return FILE_TO_DISPLAY[file_name]
    # Otherwise use file name (usually same as class name)
    return file_name


def _scan_algorithm_module(module_path: str, file_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Scan a module for algorithm class.

    Returns:
        Tuple of (display_name, module_path, class_name) or None
    """
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        logger.debug(f"Could not import {module_path}: {e}")
        return None

    # Find the main algorithm class (usually named same as file)
    class_name = file_name
    if hasattr(module, class_name):
        display_name = _get_display_name(file_name, class_name)
        return (display_name, module_path, class_name)

    # Fallback: find any class with algorithm_information
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module_path:
            continue
        if hasattr(obj, 'algorithm_information'):
            display_name = _get_display_name(file_name, name)
            return (display_name, module_path, name)

    return None


def _discover_algorithms_in_category(category: str) -> Dict[str, Tuple[str, str]]:
    """Discover all algorithms in a category package."""
    results = {}
    package_name = ALGORITHM_PACKAGES.get(category)

    if not package_name:
        return results

    try:
        package = importlib.import_module(package_name)
    except Exception as e:
        logger.debug(f"Could not import package {package_name}: {e}")
        return results

    if not hasattr(package, '__path__'):
        return results

    # Scan all modules in the package
    for importer, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg:
            continue
        if module_name.startswith('_'):
            continue
        if module_name in SKIP_ALGO_MODULES:
            continue

        full_module_path = f"{package_name}.{module_name}"
        result = _scan_algorithm_module(full_module_path, module_name)

        if result:
            display_name, mod_path, class_name = result
            results[display_name] = (mod_path, class_name)

    return results


def discover_all_algorithms(force: bool = False) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """
    Discover all algorithms in all categories.

    Returns:
        Dict mapping category to {display_name: (module_path, class_name)}
    """
    global _ALGO_REGISTRY_CACHE, _ALGO_DISCOVERED

    if _ALGO_DISCOVERED and not force:
        return _ALGO_REGISTRY_CACHE

    _ALGO_REGISTRY_CACHE = {}

    for category in ALGORITHM_PACKAGES.keys():
        _ALGO_REGISTRY_CACHE[category] = _discover_algorithms_in_category(category)

    _ALGO_DISCOVERED = True
    return _ALGO_REGISTRY_CACHE


def get_discovered_algorithm_names(category: str) -> List[str]:
    """Get list of algorithm display names for a category."""
    if not _ALGO_DISCOVERED:
        discover_all_algorithms()
    return list(_ALGO_REGISTRY_CACHE.get(category, {}).keys())


def get_discovered_algorithm_class(category: str, display_name: str) -> Optional[type]:
    """Get algorithm class by display name."""
    if not _ALGO_DISCOVERED:
        discover_all_algorithms()

    algo_info = _ALGO_REGISTRY_CACHE.get(category, {}).get(display_name)
    if not algo_info:
        return None

    module_path, class_name = algo_info
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception:
        return None


def get_discovered_algorithm_module_info(category: str, display_name: str) -> Optional[Tuple[str, str]]:
    """Get (module_path, class_name) for an algorithm."""
    if not _ALGO_DISCOVERED:
        discover_all_algorithms()
    return _ALGO_REGISTRY_CACHE.get(category, {}).get(display_name)

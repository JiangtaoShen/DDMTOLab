"""Audit: GUI-scanned algorithm parameters must match real __init__ signatures.

For every algorithm the GUI discovers, verify that every parameter the GUI
would offer is actually accepted by the algorithm constructor (a mismatch
means the GUI run would crash with TypeError), and report constructor
parameters the GUI does not expose.
"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ui'))

from utils.algo_scanner import (
    discover_all_algorithms, get_discovered_algorithm_class,
    get_algorithm_params_from_scan, EXCLUDE_PARAMS,
)


def main():
    registry = discover_all_algorithms()
    n_checked = 0
    invalid = []
    missing = []

    for category, algos in registry.items():
        for display_name in algos:
            cls = get_discovered_algorithm_class(category, display_name)
            if cls is None:
                invalid.append((category, display_name, 'class import failed'))
                continue
            try:
                sig = inspect.signature(cls.__init__)
            except (TypeError, ValueError):
                continue

            accepted = {p for p in sig.parameters if p != 'self'}
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values())

            scanned = set(get_algorithm_params_from_scan(category, display_name).keys())
            n_checked += 1

            if not has_var_kw:
                extra = scanned - accepted
                if extra:
                    invalid.append((category, display_name,
                                    f'GUI offers params not in __init__: {sorted(extra)}'))

            unexposed = accepted - scanned - EXCLUDE_PARAMS
            if unexposed:
                missing.append((category, display_name, sorted(unexposed)))

    print(f'Checked {n_checked} algorithms')
    print(f'\n[CRITICAL] GUI params that would crash the constructor: {len(invalid)}')
    for cat, name, msg in invalid:
        print(f'  - {cat}/{name}: {msg}')
    print(f'\n[INFO] Constructor params not exposed in GUI: {len(missing)}')
    for cat, name, params in missing:
        print(f'  - {cat}/{name}: {params}')

    return 1 if invalid else 0


if __name__ == '__main__':
    sys.exit(main())

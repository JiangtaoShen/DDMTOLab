"""Fast regression tests for the GUI support layer (scanners, registry, compat).

These cover the pure-Python plumbing the DearPyGui UI is built on, so they run
in CI without a display. Heavy end-to-end checks live in gui_ui_smoke.py
(real GUI, driven programmatically) and gui_smoke_harness.py (algorithm x
problem cross matrix); both are standalone scripts, not pytest collected.
"""
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ui_sys_path():
    for p in (str(ROOT / 'src'), str(ROOT / 'ui')):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Library-level fixes surfaced by GUI testing
# ---------------------------------------------------------------------------

def test_wfg_respects_custom_D():
    from ddmtolab.Problems.STMO.WFG import WFG
    for name in ['WFG1', 'WFG2', 'WFG3', 'WFG9']:
        p = getattr(WFG(), name)(M=3, Kp=4, D=12)
        assert p.dims == [12], f'{name} ignored D'


def test_wfg_default_D_unchanged():
    from ddmtolab.Problems.STMO.WFG import WFG
    p = WFG().WFG1(M=3)  # Kp defaults to M-1=2, D to Kp+10
    assert p.dims == [12]


def test_wfg_rejects_D_not_greater_than_Kp():
    from ddmtolab.Problems.STMO.WFG import WFG
    with pytest.raises(ValueError):
        WFG().WFG1(M=3, Kp=10, D=8)


def test_mtmo_dtlz_p1_constructs():
    from ddmtolab.Problems.MTMO.mtmo_dtlz import MTMO_DTLZ
    p = MTMO_DTLZ().P1(M=3, D=10)
    assert p.dims == [10, 10]


def test_mtgp_task_encoding_three_tasks():
    """MTGP task feature must stay integer-valued.

    Regression for the linspace(0, 1, nt) encoding: BoTorch casts the task
    column to long, so 0.5 collapsed to 0 and 3-task models silently merged
    tasks (and SELF crashed indexing a 2x2 correlation matrix).
    """
    torch = pytest.importorskip('torch')
    import numpy as np
    from ddmtolab.Methods.Algo_Methods.bo_utils import mtgp_build, mtgp_task_corr

    rng = np.random.default_rng(0)
    decs = [rng.random((12, 4)) for _ in range(3)]
    objs = [rng.random((12, 1)) for _ in range(3)]
    mtgp = mtgp_build(decs, objs, dims=[4, 4, 4], data_type=torch.double)
    corr = mtgp_task_corr(mtgp)
    assert corr.shape == (3, 3)


# ---------------------------------------------------------------------------
# GUI scanners and registry
# ---------------------------------------------------------------------------

def test_gui_discovers_all_categories():
    _ui_sys_path()
    from utils.registry import get_algorithm_names, get_problem_suites
    for cat in ['STSO', 'STMO', 'MTSO', 'MTMO']:
        assert len(get_algorithm_names(cat)) >= 10, f'too few algorithms in {cat}'
    for cat in ['STSO', 'STMO', 'MTSO', 'MTMO', 'RWO']:
        assert len(get_problem_suites(cat)) >= 3, f'too few suites in {cat}'
    # RWO algorithm list is the MTSO + MTMO union
    assert len(get_algorithm_names('RWO')) >= 20


def test_scanned_params_match_signatures():
    """Every parameter the GUI offers must be accepted by the constructor.

    Regression for the scanner picking up helper-class __init__ methods
    (e.g. DFMAB_MTO, EMTO_OTL), which made the GUI offer bogus parameters.
    """
    _ui_sys_path()
    from utils.algo_scanner import (
        discover_all_algorithms, get_discovered_algorithm_class,
        get_algorithm_params_from_scan,
    )
    registry = discover_all_algorithms()
    assert registry
    bad = []
    for category, algos in registry.items():
        for name in algos:
            cls = get_discovered_algorithm_class(category, name)
            if cls is None:
                bad.append((category, name, 'import failed'))
                continue
            sig = inspect.signature(cls.__init__)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in sig.parameters.values()):
                continue
            accepted = set(sig.parameters) - {'self'}
            scanned = set(get_algorithm_params_from_scan(category, name))
            extra = scanned - accepted
            if extra:
                bad.append((category, name, sorted(extra)))
    assert not bad, f'GUI offers params not accepted by __init__: {bad}'


def test_create_problem_param_mapping():
    _ui_sys_path()
    from utils.registry import create_problem, get_problem_creator
    p = create_problem('STMO', 'ZDT', 'ZDT1', D=7)
    assert p.dims == [7]
    # Params a method doesn't accept are dropped individually, not wholesale
    p = create_problem('STMO', 'ZDT', 'ZDT1', D=9, M=5)
    assert p.dims == [9]
    # Batch-mode creator returns kwargs that survive the round trip
    creator, _, kw = get_problem_creator('STMO', 'ZDT', 'ZDT2', D=11, M=4)
    assert kw == {'D': 11}
    assert creator(**kw).dims == [11]


def test_alias_mapping_rwo():
    _ui_sys_path()
    from utils.problem_scanner import map_method_kwargs
    # UI standard name K maps back to the method's own name task_num
    assert map_method_kwargs('RWO', 'MO_SCP', 'P1', {'K': 3}) == {'task_num': 3}


# ---------------------------------------------------------------------------
# Algorithm-problem compatibility checks
# ---------------------------------------------------------------------------

def test_compat_checker():
    _ui_sys_path()
    from utils.registry import create_problem
    from utils.compat import check_algorithm_compatibility
    from utils.algo_scanner import get_algorithm_info

    pepvm = create_problem('RWO', 'PEPVM', 'P1')     # K=3, unequal dims 5/7/5
    sopm = create_problem('RWO', 'SOPM', 'P1')       # K=3, 2 objectives

    # DTSKT requires equal task dims -> incompatible with PEPVM
    assert check_algorithm_compatibility(pepvm, get_algorithm_info('MTSO', 'DTSKT'))
    # MFEA supports unequal dims -> compatible
    assert not check_algorithm_compatibility(pepvm, get_algorithm_info('MTSO', 'MFEA'))
    # Single-objective MTSO algorithm on 2-objective SOPM -> incompatible
    assert check_algorithm_compatibility(sopm, get_algorithm_info('MTSO', 'MFEA'))
    # MO-MFEA supports [2, M] objectives -> compatible
    assert not check_algorithm_compatibility(sopm, get_algorithm_info('MTMO', 'MO-MFEA'))


def test_compat_range_parser():
    _ui_sys_path()
    from utils.compat import _parse_range
    assert _parse_range('1') == (1, 1)
    assert _parse_range('[2, K]') == (2, None)
    assert _parse_range('[2, 3]') == (2, 3)
    assert _parse_range('unknown') is None
    assert _parse_range(None) is None


# ---------------------------------------------------------------------------
# Batch-mode index bookkeeping (needs dearpygui importable only)
# ---------------------------------------------------------------------------

def test_batch_mode_index_helpers():
    pytest.importorskip('dearpygui')
    _ui_sys_path()
    from pages.batch_mode import _reindex_after_removal, _swap_index_entries

    # Removal shifts later entries down so names stay with their items
    assert _reindex_after_removal({0: 'a', 2: 'c'}, 1) == {0: 'a', 1: 'c'}
    assert _reindex_after_removal({0: 'a', 1: 'b'}, 0) == {0: 'b'}
    assert _reindex_after_removal({}, 0) == {}

    # Swapping never materializes None entries
    d = {0: 'x'}
    _swap_index_entries(d, 0, 1)
    assert d == {1: 'x'}
    d = {0: 'a', 1: 'b'}
    _swap_index_entries(d, 0, 1)
    assert d == {0: 'b', 1: 'a'}
    d = {}
    _swap_index_entries(d, 0, 1)
    assert d == {}

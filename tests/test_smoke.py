"""Smoke tests for ddmtolab.

Fast checks that the package imports, packaged benchmark data is present,
core operators behave correctly, and a tiny end-to-end optimization runs.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_core_imports():
    import ddmtolab  # noqa: F401
    from ddmtolab.Methods import mtop, metrics, batch_experiment, data_analysis  # noqa: F401
    from ddmtolab.Methods.Algo_Methods import algo_utils  # noqa: F401


def test_animation_generator_uses_fast_nd_sort():
    from ddmtolab.Methods import animation_generator
    # Would be False if the internal import path regressed
    assert animation_generator.ND_SORT_AVAILABLE


def test_new_mtso_algorithms_importable():
    from ddmtolab.Algorithms.MTSO.DFMAB_MTO import DFMAB_MTO  # noqa: F401
    from ddmtolab.Algorithms.MTSO.EMTO_OTL import EMTO_OTL  # noqa: F401
    from ddmtolab.Algorithms.MTSO.TATS_BO import TATS_BO  # noqa: F401
    from ddmtolab.Algorithms.MTSO.TGPBO import TGPBO  # noqa: F401


# ---------------------------------------------------------------------------
# Packaged benchmark data
# ---------------------------------------------------------------------------

def test_packaged_data_dirs_are_present_and_nonempty():
    from pathlib import Path
    import ddmtolab

    pkg_root = Path(ddmtolab.__file__).parent
    data_dirs = sorted(p for p in (pkg_root / 'Problems').rglob('data_*') if p.is_dir())
    assert len(data_dirs) >= 12, f"expected >=12 data dirs, found {len(data_dirs)}"
    for d in data_dirs:
        files = [f for f in d.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
        assert files, f"data directory {d} is empty"
        # data_pinn_hpo is a code package; proper data dirs must carry data files
        if d.name != 'data_pinn_hpo':
            non_py = [f for f in files if f.suffix != '.py']
            assert non_py, f"data directory {d} contains no data files"


def test_pkgutil_data_loading_samples():
    import pkgutil

    samples = [
        ('ddmtolab.Problems.MTMO', 'data_cec21mtmo/benchmark_1/matrix_1'),
        ('ddmtolab.Problems.MTMO', 'data_cec19matmo/M/M2/M2_1.txt'),
    ]
    for package, resource in samples:
        data = pkgutil.get_data(package, resource)
        assert data, f"pkgutil.get_data failed for {package}/{resource}"


# ---------------------------------------------------------------------------
# Core operators
# ---------------------------------------------------------------------------

def test_ga_generation_does_not_mutate_parents():
    from ddmtolab.Methods.Algo_Methods.algo_utils import ga_generation

    rng = np.random.default_rng(42)
    parents = rng.random((11, 5))  # odd count exercises the tail branch
    before = parents.copy()
    offspring = ga_generation(parents, 2.0, 5.0)
    # The caller's population must never be reordered in place: decision rows
    # would silently desynchronize from their objective rows
    np.testing.assert_array_equal(parents, before)
    assert offspring.shape == (11, 5)


def test_hv_handles_constant_objective_column():
    from ddmtolab.Methods.metrics import HV

    objs = np.array([[0.5, 1.0], [0.2, 1.0], [0.8, 1.0]])
    pf = np.array([[0.0, 1.0], [1.0, 1.0]])  # zero range in second objective
    value = HV().calculate(objs, pf=pf)
    assert np.isfinite(value) and value >= 0.0


def test_hv_reference_point_filters_outside_points():
    from ddmtolab.Methods.metrics import HV

    objs = np.array([[0.5, 0.5], [1.5, 1.5]])  # second point beyond reference
    value = HV().calculate(objs, reference=np.array([1.0, 1.0]))
    assert np.isfinite(value) and value > 0.0


def test_igdp_matches_reference_implementation():
    from ddmtolab.Methods.metrics import IGDp

    rng = np.random.default_rng(0)
    objs = rng.random((20, 3))
    pf = rng.random((50, 3))
    expected = float(np.mean([
        np.min(np.sqrt(np.sum(np.maximum(objs - p, 0.0) ** 2, axis=1))) for p in pf
    ]))
    assert IGDp().calculate(objs, pf) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# End-to-end optimization (tiny budgets)
# ---------------------------------------------------------------------------

def test_mtop_custom_problem_with_ga():
    from ddmtolab.Methods.mtop import MTOP
    from ddmtolab.Algorithms.STSO.GA import GA

    def sphere(x):
        return np.sum(x ** 2, axis=1, keepdims=True)

    problem = MTOP()
    problem.add_task(sphere, dim=5)
    algo = GA(problem, n=10, max_nfes=100, save_data=False, disable_tqdm=True)
    results = algo.optimize()
    best = np.asarray(results.best_objs[0]).ravel()[0]
    assert np.isfinite(best)
    assert best < 5.0  # better than the worst corner of [0, 1]^5


def test_cec17mtso_problem_loads_and_ga_runs():
    from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
    from ddmtolab.Algorithms.STSO.GA import GA

    problem = CEC17MTSO().P1()  # loads packaged .mat data
    algo = GA(problem, n=10, max_nfes=60, save_data=False, disable_tqdm=True)
    results = algo.optimize()
    assert len(results.best_objs) == 2  # P1 is a two-task benchmark
    for task_best in results.best_objs:
        assert np.all(np.isfinite(np.asarray(task_best, dtype=float)))

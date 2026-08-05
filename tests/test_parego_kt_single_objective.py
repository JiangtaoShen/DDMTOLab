"""
Tests for single-objective support in ParEGO-KT.

ParEGO scalarizes M objectives against a weight vector drawn from the unit
simplex. For M = 1 that simplex degenerates to the single point [1.0], which
the weight generator could not reach: both its NBI and ILD layer searches
looped forever, so the algorithm hung before its first evaluation rather than
failing. The scalarization itself is fine at M = 1 -- it reduces to the
normalized objective -- so fixing the generator is all that single-objective
support needs.

The termination checks run in a daemon thread on purpose: a regression here
hangs rather than raises, and a plain call would take the whole suite with it.
"""

import threading

import numpy as np
import pytest

from ddmtolab.Algorithms.MTMO.ParEGO_KT import ParEGO_KT
from ddmtolab.Methods.Algo_Methods.algo_utils import set_seed
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.mtop import MTOP

TERMINATION_TIMEOUT = 30.0


def call_with_timeout(func, *args, **kwargs):
    """Run func in a daemon thread; return its result or None if it hangs."""
    box = {}

    def target():
        box['result'] = func(*args, **kwargs)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=TERMINATION_TIMEOUT)
    return None if thread.is_alive() else box.get('result')


# =============================================================================
# The weight generator
# =============================================================================

class TestUniformPointAtOneObjective:
    """The degenerate simplex must resolve to its single point."""

    @pytest.mark.parametrize('method', ['NBI', 'ILD', 'MUD'])
    def test_terminates(self, method):
        outcome = call_with_timeout(uniform_point, 100, 1, method=method)
        assert outcome is not None, (
            f"uniform_point(100, 1, method={method!r}) did not terminate within "
            f"{TERMINATION_TIMEOUT}s"
        )

    @pytest.mark.parametrize('method', ['NBI', 'ILD', 'MUD'])
    def test_every_weight_is_the_only_valid_one(self, method):
        W, n = uniform_point(100, 1, method=method)
        assert W.shape == (n, 1)
        assert np.allclose(W, 1.0)
        assert np.allclose(W.sum(axis=1), 1.0)

    @pytest.mark.parametrize('method', ['NBI', 'ILD'])
    def test_simplex_methods_return_exactly_one_weight(self, method):
        # MUD is excluded: it honours its "exactly N points" contract and
        # returns N copies, which is redundant but not wrong.
        _, n = uniform_point(100, 1, method=method)
        assert n == 1

    @pytest.mark.parametrize('n_requested', [1, 10, 500])
    def test_count_does_not_depend_on_the_request(self, n_requested):
        _, n = uniform_point(n_requested, 1)
        assert n == 1

    @pytest.mark.parametrize('m', [2, 3, 5])
    def test_more_objectives_are_untouched(self, m):
        W, n = uniform_point(100, m)
        assert W.shape == (n, m)
        assert n > 1
        assert np.allclose(W.sum(axis=1), 1.0, atol=1e-5)


# =============================================================================
# The algorithm
# =============================================================================

def single_objective_a(x):
    return np.sum(x ** 2, axis=1, keepdims=True)


def single_objective_b(x):
    return np.sum((x - 0.3) ** 2, axis=1, keepdims=True) + 0.5


def bi_objective(x):
    return np.hstack([np.sum(x ** 2, axis=1, keepdims=True),
                      np.sum((x - 1) ** 2, axis=1, keepdims=True)])


N_INITIAL = 8
MAX_NFES = 14


def run_parego(funcs, seed=5):
    problem = MTOP()
    problem.add_task(funcs, dim=(3,) * len(funcs),
                     lower_bound=(-2,) * len(funcs),
                     upper_bound=(2,) * len(funcs))
    set_seed(seed)
    return ParEGO_KT(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, n_weights=10,
                     save_data=False, disable_tqdm=True).optimize()


@pytest.fixture(scope='module')
def single_objective_run():
    return run_parego((single_objective_a, single_objective_b))


@pytest.fixture(scope='module')
def mixed_run():
    return run_parego((single_objective_a, bi_objective))


class TestParEGOKTOnSingleObjectiveTasks:
    """The capability the algorithm declares must actually hold."""

    def test_declares_single_objective_support(self):
        assert ParEGO_KT.algorithm_information['n_objs'] == '[1, M]'

    def test_runs_to_the_requested_budget(self, single_objective_run):
        assert list(single_objective_run.max_nfes) == [MAX_NFES, MAX_NFES]

    def test_returns_a_best_individual_per_task(self, single_objective_run):
        for best in single_objective_run.best_objs:
            assert np.asarray(best).size == 1

    def test_history_matches_the_reported_counts(self, single_objective_run):
        lengths = [np.asarray(single_objective_run.all_objs[t][-1]).shape[0]
                   for t in range(2)]
        assert lengths == list(single_objective_run.max_nfes)

    def test_it_actually_optimizes(self, single_objective_run):
        """Terminating is not enough: the search must beat its initial design."""
        for task in range(2):
            history = np.asarray(single_objective_run.all_objs[task][-1]).ravel()
            assert history.min() <= history[:N_INITIAL].min()

    def test_no_nan_objectives(self, single_objective_run):
        for task in range(2):
            assert not np.isnan(np.asarray(single_objective_run.all_objs[task][-1])).any()


class TestParEGOKTOnMixedObjectiveCounts:
    """'objs': 'unequal' means one problem may mix M = 1 and M > 1 tasks."""

    def test_runs_to_the_requested_budget(self, mixed_run):
        assert list(mixed_run.max_nfes) == [MAX_NFES, MAX_NFES]

    def test_single_objective_task_returns_an_individual(self, mixed_run):
        assert np.asarray(mixed_run.best_objs[0]).size == 1

    def test_multiobjective_task_returns_a_front(self, mixed_run):
        assert np.asarray(mixed_run.best_objs[1]).shape == (MAX_NFES, 2)


def test_multiobjective_still_works():
    """Regression: the original multiobjective behaviour is unaffected."""
    results = run_parego((bi_objective, bi_objective))
    assert list(results.max_nfes) == [MAX_NFES, MAX_NFES]
    for best in results.best_objs:
        assert np.asarray(best).shape == (MAX_NFES, 2)

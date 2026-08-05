"""
Tests for MTBO's cost-sensitive evaluation.

Covers the cost vector's validation, the shared cost budget, the shift of
evaluations toward cheap tasks, and the guarantee that equal costs reproduce the
cost-unaware schedule.

The end-to-end cases fit real Gaussian processes, so they use deliberately tiny
budgets; they are still the only way to observe the allocation the acquisition
actually produces.
"""

import numpy as np
import pytest

from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Methods.Algo_Methods.algo_utils import set_seed
from ddmtolab.Methods.mtop import MTOP


# =============================================================================
# Helpers
# =============================================================================

def sphere(x):
    return np.sum(x ** 2, axis=1, keepdims=True)


def shifted_sphere(x):
    return np.sum((x - 0.3) ** 2, axis=1, keepdims=True) + 0.5


def make_problem(dim=2):
    """Two cheap, smooth, related single-objective tasks."""
    problem = MTOP()
    problem.add_task((sphere, shifted_sphere), dim=(dim, dim),
                     lower_bound=(-2, -2), upper_bound=(2, 2))
    return problem


def run_mtbo(seed=7, **kwargs):
    """Run MTBO from a fixed seed so allocations are reproducible."""
    set_seed(seed)
    return MTBO(make_problem(), save_data=False, **kwargs).optimize()


def fingerprint(results):
    """Every evaluated point and objective per task, for exact comparisons."""
    return [(np.asarray(results.all_decs[i][-1]).round(10).tolist(),
             np.asarray(results.all_objs[i][-1]).round(10).tolist())
            for i in range(len(results.all_objs))]


# =============================================================================
# Cost vector validation
# =============================================================================

class TestTaskCostValidation:
    """The cost vector is validated before any evaluation is spent."""

    def test_default_is_the_all_ones_vector(self):
        algorithm = MTBO(make_problem())
        task_cost, equal_costs = algorithm._resolve_task_cost(2)
        assert task_cost.tolist() == [1.0, 1.0]
        assert equal_costs is True

    def test_equal_costs_are_detected_at_any_magnitude(self):
        _, equal_costs = MTBO(make_problem(), task_cost=[3.0, 3.0])._resolve_task_cost(2)
        assert equal_costs is True

    def test_unequal_costs_are_detected(self):
        task_cost, equal_costs = MTBO(
            make_problem(), task_cost=[1.0, 4.0])._resolve_task_cost(2)
        assert task_cost.tolist() == [1.0, 4.0]
        assert equal_costs is False

    def test_accepts_tuples_and_arrays(self):
        for supplied in [(1.0, 4.0), np.array([1.0, 4.0]), [1, 4]]:
            task_cost, _ = MTBO(make_problem(), task_cost=supplied)._resolve_task_cost(2)
            assert task_cost.tolist() == [1.0, 4.0]

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match='must have length 2'):
            MTBO(make_problem(), task_cost=[1.0])._resolve_task_cost(2)

    @pytest.mark.parametrize('bad', [[1.0, 0.0], [1.0, -2.0], [0.0, 0.0]])
    def test_non_positive_cost_raises(self, bad):
        with pytest.raises(ValueError, match='must be positive'):
            MTBO(make_problem(), task_cost=bad)._resolve_task_cost(2)

    def test_validation_happens_before_any_evaluation(self):
        """A bad cost vector must fail fast, not after burning the initial design."""
        with pytest.raises(ValueError):
            MTBO(make_problem(), n_initial=4, max_nfes=6, task_cost=[1.0, -1.0],
                 save_data=False).optimize()


# =============================================================================
# Shared runs
#
# Every end-to-end case fits real GPs, so the runs are computed once and shared.
# N_INITIAL/MAX_NFES are kept small; the cost ratio is 8 because cost is only
# one term of the trade-off and a mild ratio can legitimately be outweighed by a
# large expected improvement on the pricier task.
# =============================================================================

N_INITIAL = 4
MAX_NFES = 6
RATIO = 8.0


@pytest.fixture(scope='module')
def default_run():
    return run_mtbo(n_initial=N_INITIAL, max_nfes=MAX_NFES)


@pytest.fixture(scope='module')
def ones_run():
    return run_mtbo(n_initial=N_INITIAL, max_nfes=MAX_NFES, task_cost=[1.0, 1.0])


@pytest.fixture(scope='module')
def uniform_run():
    return run_mtbo(n_initial=N_INITIAL, max_nfes=MAX_NFES, task_cost=[3.0, 3.0])


@pytest.fixture(scope='module')
def cheap_first_run():
    return run_mtbo(n_initial=N_INITIAL, max_nfes=MAX_NFES, task_cost=[1.0, RATIO])


@pytest.fixture(scope='module')
def cheap_second_run():
    return run_mtbo(n_initial=N_INITIAL, max_nfes=MAX_NFES, task_cost=[RATIO, 1.0])


# =============================================================================
# Equal costs reproduce the cost-unaware behaviour
# =============================================================================

class TestEqualCosts:
    """With nothing to trade off, the schedule and budget stay as they were."""

    def test_every_task_reaches_its_own_max_nfes(self, default_run):
        assert list(default_run.max_nfes) == [MAX_NFES, MAX_NFES]

    def test_explicit_ones_match_the_default(self, default_run, ones_run):
        assert fingerprint(ones_run) == fingerprint(default_run)

    def test_uniform_non_unit_cost_matches_the_default(self, default_run, uniform_run):
        """A uniform cost carries no information, so it must change nothing."""
        assert list(uniform_run.max_nfes) == list(default_run.max_nfes)
        assert fingerprint(uniform_run) == fingerprint(default_run)


# =============================================================================
# Unequal costs drive the allocation
# =============================================================================

class TestCostAwareAllocation:
    """Unequal costs buy more evaluations of the cheap task."""

    def test_cheap_task_gets_more_evaluations(self, cheap_first_run):
        extra = [count - N_INITIAL for count in cheap_first_run.max_nfes]
        assert extra[0] > extra[1], f"expected the cheap task to win, got {extra}"

    def test_flipping_the_prices_flips_the_allocation(self, cheap_first_run,
                                                      cheap_second_run):
        extra_first = [c - N_INITIAL for c in cheap_first_run.max_nfes]
        extra_second = [c - N_INITIAL for c in cheap_second_run.max_nfes]
        assert extra_first[0] > extra_first[1]
        assert extra_second[1] > extra_second[0]

    @pytest.mark.parametrize('fixture_name, task_cost', [
        ('cheap_first_run', [1.0, RATIO]),
        ('cheap_second_run', [RATIO, 1.0]),
    ])
    def test_shared_budget_is_respected_and_used(self, request, fixture_name, task_cost):
        results = request.getfixturevalue(fixture_name)
        cost = np.array(task_cost)

        budget = float(np.dot([MAX_NFES] * 2, cost))
        spent = float(np.dot(list(results.max_nfes), cost))

        assert spent <= budget + 1e-9, f"overspent: {spent} > {budget}"
        # Nothing affordable may be left over
        assert spent > budget - cost.max()

    def test_per_task_max_nfes_no_longer_pins_the_counts(self, cheap_first_run):
        """Under unequal costs max_nfes sizes the budget, not each task's count."""
        assert list(cheap_first_run.max_nfes) != [MAX_NFES, MAX_NFES]

    def test_budget_scales_with_the_cost_vector(self, default_run, cheap_first_run):
        """max_nfes counts evaluations at unit price, so a pricier task raises
        the total budget rather than shrinking the run."""
        unit_budget = float(np.dot([MAX_NFES] * 2, [1.0, 1.0]))
        scaled_budget = float(np.dot([MAX_NFES] * 2, [1.0, RATIO]))
        assert scaled_budget > unit_budget

        spent = float(np.dot(list(cheap_first_run.max_nfes), [1.0, RATIO]))
        assert spent == pytest.approx(scaled_budget)
        # A larger budget buys strictly more evaluations than the unit-cost run
        assert sum(cheap_first_run.max_nfes) > sum(default_run.max_nfes)

    def test_results_are_well_formed(self, cheap_first_run):
        assert len(cheap_first_run.best_decs) == 2
        assert len(cheap_first_run.best_objs) == 2
        assert not any(np.isnan(np.asarray(o)).any() for o in cheap_first_run.best_objs)
        # The recorded history matches the reported evaluation counts
        lengths = [np.asarray(cheap_first_run.all_objs[i][-1]).shape[0] for i in range(2)]
        assert lengths == list(cheap_first_run.max_nfes)

    def test_every_task_keeps_at_least_its_initial_design(self, cheap_first_run,
                                                          cheap_second_run):
        for results in (cheap_first_run, cheap_second_run):
            assert all(count >= N_INITIAL for count in results.max_nfes)


# =============================================================================
# The bo_utils hook stays backward compatible
# =============================================================================

class TestNextPointHook:
    """mtbo_next_point keeps its single return value unless asked otherwise."""

    def test_default_returns_only_the_candidate(self):
        import torch
        from ddmtolab.Methods.Algo_Methods.bo_utils import mtgp_build, mtbo_next_point
        from ddmtolab.Methods.Algo_Methods.algo_utils import (
            evaluation, initialization, normalize)

        set_seed(3)
        problem = make_problem()
        decs = initialization(problem, 5, method='lhs')
        objs, _ = evaluation(problem, decs)
        objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
        mtgp = mtgp_build(decs, objs_normalized, problem.dims, data_type=torch.double)

        candidate = mtbo_next_point(mtgp=mtgp, task_id=0, objs=objs_normalized,
                                    dims=problem.dims, nt=2, data_type=torch.double)
        assert isinstance(candidate, np.ndarray)
        assert candidate.shape == (1, problem.dims[0])

        candidate, acq_value = mtbo_next_point(
            mtgp=mtgp, task_id=0, objs=objs_normalized, dims=problem.dims, nt=2,
            data_type=torch.double, return_acq_value=True)
        assert isinstance(candidate, np.ndarray)
        assert isinstance(acq_value, float)

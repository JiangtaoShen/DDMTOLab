"""
Tests for the two high-dimensional surrogate-assisted multi-objective algorithms,
GSAEA and AS-SMEA.

Both spend a tiny budget of real evaluations and do all their work on surrogates,
so the failure modes worth guarding are the quiet ones: a budget that drifts away
from what was requested, a search that leaves the unit box, a variable grouping
or a local region that collapses, and an infill loop that burns evaluations
without ever beating the design it started from.
"""

import numpy as np
import pytest

from ddmtolab.Algorithms.STMO.AS_SMEA import AS_SMEA, _Region, _init_regions
from ddmtolab.Algorithms.STMO.GSAEA import GSAEA, _group_variables
from ddmtolab.Methods.Algo_Methods.algo_utils import (evaluation, initialization,
                                                      set_seed)
from ddmtolab.Methods.metrics import GD
from ddmtolab.Methods.mtop import MTOP

DIM = 6


def bi_sphere(x):
    """Smooth bi-objective problem whose front is known to be reachable."""
    x = np.atleast_2d(x)
    return np.hstack([np.sum(x ** 2, axis=1, keepdims=True),
                      np.sum((x - 1.0) ** 2, axis=1, keepdims=True)])


def make_problem(dim=DIM):
    problem = MTOP()
    problem.add_task(bi_sphere, dim=dim, lower_bound=0.0, upper_bound=1.0)
    return problem


def count_evaluations(problem):
    """Count real evaluations at MTOP's single evaluation choke point."""
    counts = [0] * problem.n_tasks
    original = problem.evaluate_task

    def counting_evaluate_task(task_idx, X, *args, **kwargs):
        counts[task_idx] += np.atleast_2d(X).shape[0]
        return original(task_idx, X, *args, **kwargs)

    problem.evaluate_task = counting_evaluate_task
    return counts


def run_gsaea(problem, n_initial=24, max_nfes=36):
    return GSAEA(problem, n_initial=n_initial, max_nfes=max_nfes, n=12, k=3,
                 wmax=5, n_sobol=32, save_data=False).optimize()


def run_as_smea(problem, n_initial=24, max_nfes=36):
    return AS_SMEA(problem, n_initial=n_initial, max_nfes=max_nfes, n_regions=2,
                   n_select=2, n=20, n_gen=4, n_screen=6, n_fantasy=2,
                   fantasy_size=4, n_hv_sample=512, n_restart_cand=200,
                   save_data=False).optimize()


RUNNERS = [('GSAEA', run_gsaea), ('AS-SMEA', run_as_smea)]


# =============================================================================
# Budget accounting
# =============================================================================

@pytest.mark.parametrize('name, runner', RUNNERS)
class TestBudget:
    """The reported count must be the number of real evaluations performed."""

    def test_reported_count_matches_the_evaluations_performed(self, name, runner):
        set_seed(3)
        problem = make_problem()
        observed = count_evaluations(problem)
        results = runner(problem)
        assert results.max_nfes == observed

    def test_the_requested_budget_is_spent_and_not_exceeded(self, name, runner):
        set_seed(3)
        problem = make_problem()
        observed = count_evaluations(problem)
        runner(problem, n_initial=24, max_nfes=36)
        assert observed[0] == 36

    def test_the_archive_holds_one_row_per_evaluation(self, name, runner):
        set_seed(3)
        problem = make_problem()
        results = runner(problem)
        assert results.best_objs[0].shape[0] == results.max_nfes[0]


# =============================================================================
# Result structure
# =============================================================================

@pytest.mark.parametrize('name, runner', RUNNERS)
class TestResults:
    """Whatever the surrogate does, the returned data must stay well formed."""

    def test_shapes_and_finiteness(self, name, runner):
        set_seed(5)
        problem = make_problem()
        results = runner(problem)

        assert results.best_objs[0].shape[1] == 2
        assert results.best_decs[0].shape[1] == DIM
        assert results.best_decs[0].shape[0] == results.best_objs[0].shape[0]
        assert np.all(np.isfinite(results.best_objs[0]))
        assert np.all(np.isfinite(results.best_decs[0]))

    def test_decisions_stay_inside_the_unit_box(self, name, runner):
        set_seed(5)
        problem = make_problem()
        results = runner(problem)
        assert results.best_decs[0].min() >= 0.0
        assert results.best_decs[0].max() <= 1.0

    def test_history_grows_monotonically(self, name, runner):
        set_seed(5)
        problem = make_problem()
        results = runner(problem)
        sizes = [generation.shape[0] for generation in results.all_objs[0]]
        assert len(sizes) > 1
        assert sizes == sorted(sizes)
        assert sizes[-1] == results.max_nfes[0]


# =============================================================================
# The infill has to earn its evaluations
# =============================================================================

@pytest.mark.parametrize('name, runner', RUNNERS)
def test_converges_closer_to_the_front_than_a_design_of_the_same_size(name, runner):
    """
    A surrogate-assisted run must place its budget closer to the front than
    spending all of it on the space-filling design it starts from -- that is what
    the infill is for.

    Convergence is measured with GD rather than IGD on purpose. At a budget this
    small the infill buys proximity and not coverage, so IGD, which also rewards
    spreading along the whole front, is dominated by the design's head start and
    says little about whether the surrogate is steering.
    """
    parameter = np.linspace(0, 1, 60)
    front = np.stack([parameter ** 2 * DIM, (1 - parameter) ** 2 * DIM], axis=1)
    gd = GD()

    for seed in (0, 1, 2):
        set_seed(seed)
        problem = make_problem()
        algorithm_score = gd.calculate(
            runner(problem, n_initial=20, max_nfes=40).best_objs[0], front)

        set_seed(seed)
        problem = make_problem()
        decs = initialization(problem, [40], method='lhs')
        objs, _ = evaluation(problem, decs)
        design_score = gd.calculate(objs[0], front)

        assert algorithm_score < design_score, (
            f"{name} did not improve on its own design at seed {seed}: "
            f"GD {algorithm_score:.4f} vs {design_score:.4f}"
        )


# =============================================================================
# GSAEA: the variable grouping
# =============================================================================

class TestVariableGrouping:
    """Eq. (11) must always yield a usable partition of the decision variables."""

    def _archive(self, dim=DIM, n=20):
        set_seed(7)
        decs = np.random.rand(n, dim)
        return decs, bi_sphere(decs)

    def test_partition_is_complete_and_disjoint(self):
        decs, objs = self._archive()
        cv, dv = _group_variables(decs, objs, n_sobol=32, data_type=None)
        assert np.array_equal(np.sort(np.concatenate([cv, dv])), np.arange(DIM))
        assert np.intersect1d(cv, dv).size == 0

    def test_both_groups_are_non_empty(self):
        """An empty group would leave one of the two searches with nothing to do."""
        decs, objs = self._archive()
        cv, dv = _group_variables(decs, objs, n_sobol=32, data_type=None)
        assert cv.size > 0 and dv.size > 0

    def test_a_flat_sensitivity_profile_still_splits(self):
        """
        Identical indices make the mean threshold of Eq. (10) select nothing;
        the fallback has to keep both groups usable.
        """
        set_seed(7)
        decs = np.random.rand(20, 4)
        constant = np.zeros((20, 2))
        cv, dv = _group_variables(decs, constant, n_sobol=16, data_type=None)
        assert cv.size > 0 and dv.size > 0
        assert np.array_equal(np.sort(np.concatenate([cv, dv])), np.arange(4))


# =============================================================================
# AS-SMEA: the local regions
# =============================================================================

class TestLocalRegion:
    """The hyper-ellipsoid has to stay a well-defined, non-degenerate container."""

    def _region(self, dim=DIM, n=15):
        set_seed(9)
        X = np.random.rand(n, dim) * 0.3 + 0.2
        return _Region(X.mean(axis=0), np.cov(X, rowvar=False), dim,
                       alpha=0.99735, shrinkage=0.1, n_arms=5)

    def test_repair_returns_points_inside_the_region_and_the_box(self):
        region = self._region()
        far = np.random.rand(30, DIM) * 6.0 - 3.0
        repaired = region.repair(far)
        assert repaired.min() >= 0.0 and repaired.max() <= 1.0
        # Clipping to the box can only shorten the step, so the cutoff still holds
        assert np.all(region.distance(repaired) <= region.radius + 1e-8)

    def test_samples_land_inside_the_unit_box(self):
        region = self._region()
        drawn = region.sample(50)
        assert drawn.shape == (50, DIM)
        assert drawn.min() >= 0.0 and drawn.max() <= 1.0

    def test_covariance_is_positive_definite_when_rank_deficient(self):
        """
        A region holds fewer points than there are variables, so its sample
        covariance is singular; the shrinkage must keep the ellipsoid usable.
        """
        region = self._region(dim=20, n=5)
        assert np.all(np.isfinite(region.C))
        assert np.min(np.linalg.eigvalsh(region.C)) > 0
        assert region.sigma > 0

    def test_members_tops_up_a_region_that_is_too_small_to_model(self):
        region = self._region()
        set_seed(9)
        decs = np.random.rand(40, DIM)
        idx = region.members(decs)
        assert idx.size >= min(40, max(5, DIM + 1))
        assert np.unique(idx).size == idx.size

    def test_update_keeps_the_distribution_finite(self):
        region = self._region()
        set_seed(9)
        decs = np.random.rand(12, DIM)
        region.update(decs, bi_sphere(decs))
        assert np.all(np.isfinite(region.m))
        assert np.all(np.isfinite(region.C))
        assert np.isfinite(region.sigma) and region.sigma > 0

    def test_initialization_produces_the_requested_number_of_regions(self):
        set_seed(9)
        decs = np.random.rand(30, DIM)
        regions = _init_regions(decs, bi_sphere(decs), n_regions=5, D=DIM,
                                alpha=0.99735, shrinkage=0.1, n_arms=5)
        assert len(regions) == 5
        assert all(np.all(np.isfinite(r.center)) for r in regions)

"""
Tests for :mod:`ddmtolab.Methods.metrics`.

Every metric is checked against its definition recomputed independently, plus
the invariants that make the values comparable across algorithms: a perfect
approximation scores 0 (or 1 for the maximized ones), the indicators are
monotone in dominance, and the hypervolume lands on the same [0, 1] scale
whether the reference point is derived from a front or given directly.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from ddmtolab.Methods.metrics import CV, FR, GD, HV, IGD, DeltaP, IGDp, Spacing, Spread


#: A convex two-objective front and an approximation that sits just behind it.
FRONT = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
APPROXIMATION = np.array([[0.0, 1.2], [0.6, 0.6], [1.2, 0.0]])


@pytest.fixture
def distances():
    """Pairwise distances from every front point to every obtained point."""
    return cdist(FRONT, APPROXIMATION, metric='euclidean')


# =============================================================================
# 1. Distance-based indicators
# =============================================================================

class TestDistanceIndicators:
    """IGD, GD, IGD+ and Delta_p against their definitions."""

    def test_igd_averages_the_nearest_obtained_point(self, distances):
        assert IGD()(APPROXIMATION, FRONT) == pytest.approx(np.mean(np.min(distances, axis=1)))

    def test_gd_follows_van_veldhuizen(self, distances):
        # ||d||_2 / n rather than the mean, which is the classical definition
        nearest = np.min(distances, axis=0)
        assert GD()(APPROXIMATION, FRONT) == pytest.approx(
            np.linalg.norm(nearest) / len(nearest))

    def test_igd_plus_only_counts_dominated_components(self):
        expected = np.mean([
            np.min(np.sqrt(np.sum(np.maximum(APPROXIMATION - point, 0.0) ** 2, axis=1)))
            for point in FRONT
        ])
        assert IGDp()(APPROXIMATION, FRONT) == pytest.approx(expected)

    def test_igd_plus_never_exceeds_igd(self):
        assert IGDp()(APPROXIMATION, FRONT) <= IGD()(APPROXIMATION, FRONT)

    def test_delta_p_is_the_worse_of_the_two_components(self, distances):
        expected = max(np.mean(np.min(distances, axis=1)), np.mean(np.min(distances, axis=0)))
        assert DeltaP()(APPROXIMATION, FRONT) == pytest.approx(expected)

    @pytest.mark.parametrize('metric', [IGD(), GD(), IGDp(), DeltaP()])
    def test_the_true_front_scores_zero(self, metric):
        assert metric(FRONT, FRONT) == pytest.approx(0.0)

    @pytest.mark.parametrize('metric', [IGD(), GD(), IGDp(), DeltaP()])
    def test_a_closer_approximation_scores_lower(self, metric):
        far = FRONT + 0.5
        near = FRONT + 0.1
        assert metric(near, FRONT) < metric(far, FRONT)

    def test_igd_plus_chunking_matches_the_unchunked_definition(self):
        # The implementation chunks over the reference set to bound memory
        reference = np.random.default_rng(0).random((5000, 3))
        obtained = np.random.default_rng(1).random((40, 3))
        expected = np.mean([
            np.min(np.sqrt(np.sum(np.maximum(obtained - point, 0.0) ** 2, axis=1)))
            for point in reference
        ])
        assert IGDp()(obtained, reference) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize('metric', [IGD(), GD(), IGDp(), DeltaP()])
    def test_mismatched_dimensions_give_nan(self, metric):
        assert np.isnan(metric(APPROXIMATION, np.zeros((3, 3))))

    @pytest.mark.parametrize('metric', [IGD(), GD(), IGDp(), DeltaP()])
    def test_empty_input_gives_nan(self, metric):
        assert np.isnan(metric(np.empty((0, 2)), FRONT))

    @pytest.mark.parametrize('metric, sign', [(IGD(), -1), (GD(), -1), (IGDp(), -1),
                                              (DeltaP(), -1), (HV(), 1), (FR(), 1),
                                              (CV(), -1), (Spacing(), -1), (Spread(), -1)])
    def test_the_sign_states_the_optimization_direction(self, metric, sign):
        assert metric.sign == sign


# =============================================================================
# 2. Distribution indicators
# =============================================================================

class TestDistributionIndicators:
    """Spacing and Spread describe how the points are laid out."""

    EVEN = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])

    def test_spacing_is_the_spread_of_nearest_neighbour_distances(self):
        neighbours = cdist(self.EVEN, self.EVEN, metric='cityblock')
        np.fill_diagonal(neighbours, np.inf)
        assert Spacing()(self.EVEN) == pytest.approx(np.std(neighbours.min(axis=1)))

    def test_an_evenly_spaced_set_has_zero_spacing(self):
        assert Spacing()(self.EVEN) == pytest.approx(0.0)

    def test_an_uneven_set_has_positive_spacing(self):
        uneven = np.array([[0.0, 3.0], [0.1, 2.9], [2.0, 1.0], [3.0, 0.0]])
        assert Spacing()(uneven) > 0.0

    def test_spacing_needs_two_points(self):
        assert np.isnan(Spacing()(self.EVEN[:1]))

    def test_spread_needs_at_least_as_many_points_as_objectives(self):
        assert np.isnan(Spread()(self.EVEN[:1], FRONT))

    def test_spread_is_finite_for_a_reasonable_set(self):
        value = Spread()(APPROXIMATION, FRONT)
        assert np.isfinite(value) and value >= 0.0


# =============================================================================
# 3. Constraint indicators
# =============================================================================

class TestConstraintIndicators:
    """Feasible rate and constraint violation."""

    CONSTRAINTS = np.array([[-1.0, -2.0], [0.5, -1.0], [0.0, 0.0], [3.0, 4.0]])

    def test_feasible_rate_counts_rows_that_satisfy_everything(self):
        # Rows 0 and 2 are feasible, since a constraint holds when it is <= 0
        assert FR()(self.CONSTRAINTS) == pytest.approx(0.5)

    def test_feasible_rate_of_a_fully_feasible_population(self):
        assert FR()(-np.ones((5, 3))) == pytest.approx(1.0)

    def test_constraint_violation_reports_the_best_row(self):
        assert CV()(self.CONSTRAINTS) == pytest.approx(0.0)

    def test_constraint_violation_sums_only_the_positive_parts(self):
        assert CV()(np.array([1.0, -2.0, 3.0])) == pytest.approx(4.0)

    def test_constraint_violation_of_an_infeasible_population(self):
        assert CV()(np.array([[1.0, 1.0], [2.0, 0.0]])) == pytest.approx(2.0)

    def test_a_single_constraint_column_is_accepted(self):
        assert FR()(np.array([-1.0, 1.0])) == pytest.approx(0.5)

    def test_empty_input_gives_nan(self):
        assert np.isnan(FR()(np.empty((0, 2))))
        assert np.isnan(CV()(np.empty((0, 2))))


# =============================================================================
# 4. Hypervolume
# =============================================================================

class TestHypervolume:
    """The exact algorithm, its Monte Carlo fallback and the normalization."""

    def test_a_single_point_covers_its_own_box(self):
        assert HV()._exact_hv(np.array([[0.5, 0.5]]), np.ones(2)) == pytest.approx(0.25)

    def test_a_two_dimensional_staircase(self):
        points = np.array([[0.1, 0.7], [0.4, 0.4], [0.7, 0.1]])
        expected = ((0.4 - 0.1) * (1 - 0.7) + (0.7 - 0.4) * (1 - 0.4)
                    + (1 - 0.7) * (1 - 0.1))
        assert HV()._exact_hv(points, np.ones(2)) == pytest.approx(expected)

    def test_a_single_point_in_three_dimensions(self):
        assert HV()._exact_hv(np.array([[0.5, 0.5, 0.5]]), np.ones(3)) == pytest.approx(0.125)

    @pytest.mark.parametrize('seed', [0, 1, 2])
    def test_monte_carlo_agrees_with_the_exact_algorithm(self, seed):
        # Monte Carlo takes over from four objectives on, where no exact value
        # is available, so pin it against the exact one while both still apply
        points = np.random.default_rng(seed + 7).random((6, 3)) * 0.8
        exact = HV()._exact_hv(points, np.ones(3))
        estimate = HV()._monte_carlo_hv(points, np.ones(3), n_samples=400_000, seed=seed)
        assert estimate == pytest.approx(exact, abs=3e-3)

    def test_monte_carlo_is_reproducible_and_leaves_the_global_rng_alone(self):
        points = np.random.default_rng(3).random((5, 4))
        np.random.seed(1234)
        before = np.random.random()
        np.random.seed(1234)
        first = HV()._monte_carlo_hv(points, np.ones(4), n_samples=10_000, seed=0)
        second = HV()._monte_carlo_hv(points, np.ones(4), n_samples=10_000, seed=0)
        after = np.random.random()
        assert first == second
        assert before == after

    def test_the_two_reference_modes_describe_the_same_box(self):
        # Passing a front places the reference 10% beyond its nadir; passing
        # that same point directly must give the identical value
        floor = np.minimum(APPROXIMATION.min(axis=0), 0.0)
        equivalent = floor + 1.1 * (FRONT.max(axis=0) - floor)
        assert HV().calculate(APPROXIMATION, pf=FRONT) == pytest.approx(
            HV().calculate(APPROXIMATION, reference=equivalent))

    def test_the_value_stays_within_the_unit_scale(self):
        for value in (HV().calculate(APPROXIMATION, pf=FRONT),
                      HV().calculate(APPROXIMATION, reference=np.array([1.5, 1.5]))):
            assert 0.0 <= value <= 1.0

    def test_a_dominating_set_scores_higher(self):
        assert (HV().calculate(APPROXIMATION * 0.5, pf=FRONT)
                > HV().calculate(APPROXIMATION, pf=FRONT))

    def test_points_beyond_the_reference_are_ignored(self):
        assert HV().calculate(np.array([[5.0, 5.0]]),
                              reference=np.array([1.0, 1.0])) == pytest.approx(0.0)

    def test_a_reference_point_reached_exactly_contributes_nothing(self):
        assert HV().calculate(np.array([[1.0, 1.0]]),
                              reference=np.array([1.0, 1.0])) == pytest.approx(0.0)

    def test_a_degenerate_objective_range_does_not_divide_by_zero(self):
        value = HV().calculate(np.array([[0.0, 0.5], [0.0, 0.2]]),
                               reference=np.array([0.0, 1.0]))
        assert np.isfinite(value)

    def test_neither_reference_raises(self):
        with pytest.raises(ValueError, match='Either pf or reference'):
            HV().calculate(APPROXIMATION)

    def test_an_empty_population_scores_zero(self):
        assert HV().calculate(np.empty((0, 2)), pf=FRONT) == pytest.approx(0.0)

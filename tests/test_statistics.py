"""
Tests for the statistical methods of :mod:`ddmtolab.Methods.data_analysis`.

Covers the Holm-Bonferroni correction, the Friedman test with its post-hoc
comparisons, Cliff's delta, the ``median[IQR]`` statistic, and the backward
compatibility of the table output when the new switches are left off.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')  # the figure tests only write files, they never display

from ddmtolab.Methods.data_analysis import (  # noqa: E402  (after the backend switch)
    ComparisonResult,
    OptimizationDirection,
    PlotConfig,
    PlotGenerator,
    StatisticType,
    StatisticsCalculator,
    TableConfig,
    TableFormat,
    TableGenerator,
)


# =============================================================================
# Helpers
# =============================================================================

def make_best_values(values):
    """
    Build the nested best-values structure the table generator consumes.

    Parameters
    ----------
    values : Dict[str, Dict[str, List[List[float]]]]
        values[algorithm][problem] = list of per-run lists, one entry per task.

    Returns
    -------
    Dict[str, Dict[str, Dict[int, List[float]]]]
        Structure keyed as best_values[algorithm][problem][run] = List[float].
    """
    return {
        algo: {
            prob: {run + 1: list(run_values) for run, run_values in enumerate(runs)}
            for prob, runs in problems.items()
        }
        for algo, problems in values.items()
    }


def cell_number(cell):
    """Parse the leading numeric value out of a rendered table cell."""
    return float(str(cell).split('(')[0].split('[')[0].replace('$', '').split()[0])


# =============================================================================
# 1. Holm-Bonferroni correction
# =============================================================================

class TestHolmBonferroni:
    """Holm-Bonferroni step-down correction of a family of p-values."""

    def test_matches_hand_worked_example(self):
        # m = 3: 3*0.01 = 0.03, 2*0.02 = 0.04, 1*0.03 = 0.03 -> monotone to 0.04
        assert StatisticsCalculator.holm_bonferroni([0.01, 0.02, 0.03]) == \
               pytest.approx([0.03, 0.04, 0.04])

    def test_preserves_input_order(self):
        # Same family as above, shuffled; each value keeps its own position
        assert StatisticsCalculator.holm_bonferroni([0.03, 0.01, 0.02]) == \
               pytest.approx([0.04, 0.03, 0.04])

    def test_single_element_is_unchanged(self):
        assert StatisticsCalculator.holm_bonferroni([0.02]) == pytest.approx([0.02])

    def test_empty_family(self):
        assert StatisticsCalculator.holm_bonferroni([]) == []

    def test_all_significant_stay_significant(self):
        raw = [1e-6, 2e-6, 3e-6, 4e-6]
        adjusted = StatisticsCalculator.holm_bonferroni(raw)
        assert all(p < 0.05 for p in adjusted)
        assert adjusted == pytest.approx([4e-6, 6e-6, 6e-6, 6e-6])

    def test_all_non_significant_stay_non_significant(self):
        adjusted = StatisticsCalculator.holm_bonferroni([0.4, 0.6, 0.8])
        assert all(p >= 0.05 for p in adjusted)

    def test_adjusted_never_below_raw(self):
        raw = [0.001, 0.01, 0.02, 0.04, 0.5]
        adjusted = StatisticsCalculator.holm_bonferroni(raw)
        assert all(a >= r for a, r in zip(adjusted, raw))

    def test_monotone_non_decreasing_in_rank(self):
        raw = [0.001, 0.01, 0.02, 0.04, 0.5]
        adjusted = StatisticsCalculator.holm_bonferroni(raw)
        ordered = [a for _, a in sorted(zip(raw, adjusted))]
        assert ordered == sorted(ordered)

    def test_capped_at_one(self):
        assert all(p <= 1.0 for p in StatisticsCalculator.holm_bonferroni([0.5, 0.6, 0.9]))

    def test_none_and_nan_are_excluded_from_the_family(self):
        # Only the two real p-values count, so m = 2 rather than 4
        adjusted = StatisticsCalculator.holm_bonferroni([0.01, None, 0.02, np.nan])
        assert adjusted[1] is None
        assert adjusted[3] is None
        assert [adjusted[0], adjusted[2]] == pytest.approx([0.02, 0.02])

    def test_all_none_family(self):
        assert StatisticsCalculator.holm_bonferroni([None, None]) == [None, None]

    def test_ties_share_the_same_adjusted_value(self):
        adjusted = StatisticsCalculator.holm_bonferroni([0.02, 0.02, 0.02])
        assert adjusted == pytest.approx([0.06, 0.06, 0.06])


# =============================================================================
# 2. Friedman test and post-hoc comparisons
# =============================================================================

class TestFriedmanTest:
    """Friedman omnibus test with Holm-corrected post-hoc comparisons."""

    # A on every instance beats B, which beats C, so the ranks are exactly 1/2/3
    SEPARATED = np.array([
        [1.0, 1.1, 0.9, 1.2, 1.05],
        [2.0, 2.1, 1.9, 2.2, 2.05],
        [3.0, 3.1, 2.9, 3.2, 3.05],
    ])
    NAMES = ['A', 'B', 'C']

    def test_average_ranks_of_a_fully_separated_example(self):
        result = StatisticsCalculator.perform_friedman_test(self.SEPARATED, self.NAMES)
        assert result.average_ranks == pytest.approx({'A': 1.0, 'B': 2.0, 'C': 3.0})

    def test_chi_square_of_a_fully_separated_example(self):
        # chi^2_F = 12N / (k(k+1)) * sum (R_i - (k+1)/2)^2 with k=3, N=5
        # = 12*5/12 * ((1-2)^2 + 0 + (3-2)^2) = 5 * 2 = 10
        result = StatisticsCalculator.perform_friedman_test(self.SEPARATED, self.NAMES)
        assert result.statistic == pytest.approx(10.0)
        assert result.p_value == pytest.approx(np.exp(-5.0))  # chi2 with 2 dof
        assert result.n_algorithms == 3
        assert result.n_instances == 5

    def test_maximize_reverses_the_ranks_but_not_the_statistic(self):
        minimize = StatisticsCalculator.perform_friedman_test(self.SEPARATED, self.NAMES)
        maximize = StatisticsCalculator.perform_friedman_test(
            self.SEPARATED, self.NAMES, direction=OptimizationDirection.MAXIMIZE
        )
        assert maximize.average_ranks == pytest.approx({'A': 3.0, 'B': 2.0, 'C': 1.0})
        assert maximize.statistic == pytest.approx(minimize.statistic)

    def test_ties_receive_the_average_rank(self):
        matrix = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ])
        result = StatisticsCalculator.perform_friedman_test(matrix, self.NAMES)
        # Two-way tie for the first two ranks -> both get (1+2)/2 = 1.5
        assert result.average_ranks == pytest.approx({'A': 1.5, 'B': 1.5, 'C': 3.0})

    def test_control_defaults_to_the_last_algorithm(self):
        result = StatisticsCalculator.perform_friedman_test(self.SEPARATED, self.NAMES)
        assert result.control == 'C'

    def test_post_hoc_z_scores_against_the_control(self):
        result = StatisticsCalculator.perform_friedman_test(
            self.SEPARATED, self.NAMES, control='C'
        )
        # se = sqrt(k(k+1) / (6N)) = sqrt(12/30) = 0.6325
        standard_error = np.sqrt(3 * 4 / (6 * 5))
        by_name = {entry.algorithm: entry for entry in result.post_hoc}
        assert by_name['A'].z_statistic == pytest.approx((1.0 - 3.0) / standard_error)
        assert by_name['B'].z_statistic == pytest.approx((2.0 - 3.0) / standard_error)
        assert by_name['C'].z_statistic == pytest.approx(0.0)

    def test_post_hoc_symbols_follow_the_ranks(self):
        result = StatisticsCalculator.perform_friedman_test(
            self.SEPARATED, self.NAMES, control='C', significance_level=0.05
        )
        by_name = {entry.algorithm: entry for entry in result.post_hoc}
        # A ranks best and clears Holm; B is better but not significantly so at N=5
        assert by_name['A'].symbol == '+'
        assert by_name['A'].significant
        assert by_name['B'].symbol in ('+', '=')
        assert by_name['C'].symbol == ''  # the control itself

    def test_post_hoc_p_values_are_holm_corrected(self):
        result = StatisticsCalculator.perform_friedman_test(self.SEPARATED, self.NAMES)
        compared = [e for e in result.post_hoc if e.algorithm != result.control]
        raw = [e.p_value for e in compared]
        expected = StatisticsCalculator.holm_bonferroni(raw)
        assert [e.p_adjusted for e in compared] == pytest.approx(expected)
        assert all(e.p_adjusted >= e.p_value for e in compared)

    def test_control_entry_carries_no_p_value(self):
        result = StatisticsCalculator.perform_friedman_test(self.SEPARATED, self.NAMES)
        control_entry = next(e for e in result.post_hoc if e.algorithm == result.control)
        assert control_entry.p_value is None
        assert control_entry.p_adjusted is None
        assert not control_entry.significant

    def test_explicit_control_selection(self):
        result = StatisticsCalculator.perform_friedman_test(
            self.SEPARATED, self.NAMES, control='A'
        )
        assert result.control == 'A'
        by_name = {entry.algorithm: entry for entry in result.post_hoc}
        assert by_name['C'].symbol == '-'  # worst rank against the best control

    def test_identical_algorithms_are_not_significant(self):
        """Fully tied blocks degenerate to 0/0 and must report no difference."""
        matrix = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (3, 1))
        result = StatisticsCalculator.perform_friedman_test(matrix, self.NAMES)
        assert result.average_ranks == pytest.approx({'A': 2.0, 'B': 2.0, 'C': 2.0})
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert all(e.symbol in ('', '=') for e in result.post_hoc)

    def test_constant_values_everywhere(self):
        """The all-identical-values boundary, with no variation at all."""
        result = StatisticsCalculator.perform_friedman_test(np.full((3, 5), 7.0), self.NAMES)
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert result.average_ranks == pytest.approx({'A': 2.0, 'B': 2.0, 'C': 2.0})
        assert all(not e.significant for e in result.post_hoc)

    def test_instances_with_nan_are_dropped(self):
        matrix = self.SEPARATED.copy()
        matrix[1, 2] = np.nan
        with pytest.warns(UserWarning, match='Dropped 1 instance'):
            result = StatisticsCalculator.perform_friedman_test(matrix, self.NAMES)
        assert result.n_instances == 4
        assert result.n_instances_dropped == 1

    def test_fewer_than_three_algorithms_raises(self):
        with pytest.raises(ValueError, match='at least 3 algorithms'):
            StatisticsCalculator.perform_friedman_test(
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), ['A', 'B']
            )

    def test_fewer_than_two_instances_raises(self):
        with pytest.raises(ValueError, match='at least 2 complete instances'):
            StatisticsCalculator.perform_friedman_test(
                np.array([[1.0], [2.0], [3.0]]), self.NAMES
            )

    def test_name_count_mismatch_raises(self):
        with pytest.raises(ValueError, match='must match'):
            StatisticsCalculator.perform_friedman_test(self.SEPARATED, ['A', 'B'])

    def test_unknown_control_raises(self):
        with pytest.raises(ValueError, match='not one of algorithm_names'):
            StatisticsCalculator.perform_friedman_test(
                self.SEPARATED, self.NAMES, control='Nope'
            )

    def test_non_2d_matrix_raises(self):
        with pytest.raises(ValueError, match='must be 2-D'):
            StatisticsCalculator.perform_friedman_test(
                np.array([1.0, 2.0, 3.0]), self.NAMES
            )

    # Ranks 1.25 / 2.00 / 2.75 over 4 instances, so chi^2_F = 4.5 stays below
    # its maximum N(k-1) = 8 and the Iman-Davenport ratio remains finite
    MIXED = np.array([
        [1.0, 2.0, 1.0, 1.0],
        [2.0, 1.0, 2.0, 3.0],
        [3.0, 3.0, 3.0, 2.0],
    ])

    def test_iman_davenport_follows_the_definition(self):
        # F_F = (N-1) chi2 / (N(k-1) - chi2) = 3 * 4.5 / (8 - 4.5)
        result = StatisticsCalculator.perform_friedman_test(self.MIXED, self.NAMES)
        assert result.statistic == pytest.approx(4.5)
        assert result.iman_davenport_statistic == pytest.approx(3.5 ** -1 * 13.5)

    def test_iman_davenport_p_value_is_below_chi_square_p_value(self):
        # The correction exists because chi^2_F is undesirably conservative
        result = StatisticsCalculator.perform_friedman_test(self.MIXED, self.NAMES)
        assert result.iman_davenport_p_value < result.p_value

    def test_iman_davenport_of_a_perfectly_separated_matrix(self):
        # chi^2_F attains its maximum N(k-1), which zeroes the denominator
        result = StatisticsCalculator.perform_friedman_test(self.SEPARATED, self.NAMES)
        assert result.statistic == pytest.approx(5 * 2.0)
        assert result.iman_davenport_statistic == np.inf
        assert result.iman_davenport_p_value == 0.0

    def test_iman_davenport_of_a_degenerate_all_tied_matrix(self):
        constant = np.ones((3, 4))
        result = StatisticsCalculator.perform_friedman_test(constant, self.NAMES)
        assert result.statistic == 0.0
        assert result.iman_davenport_statistic == 0.0
        assert result.iman_davenport_p_value == pytest.approx(1.0)


# =============================================================================
# 2b. Nemenyi test and its critical difference
# =============================================================================

class TestNemenyiTest:
    """Nemenyi all-pairs post-hoc test, checked against Demsar (2006)."""

    # Table 6 of the paper: AUC of four C4.5 variants on 14 UCI data sets
    PAPER_NAMES = ['C4.5', 'C4.5+m', 'C4.5+cf', 'C4.5+m+cf']
    PAPER_AUC = np.array([
        [0.763, 0.768, 0.771, 0.798], [0.599, 0.591, 0.590, 0.569],
        [0.954, 0.971, 0.968, 0.967], [0.628, 0.661, 0.654, 0.657],
        [0.882, 0.888, 0.886, 0.898], [0.936, 0.931, 0.916, 0.931],
        [0.661, 0.668, 0.609, 0.685], [0.583, 0.583, 0.563, 0.625],
        [0.775, 0.838, 0.866, 0.875], [1.000, 1.000, 1.000, 1.000],
        [0.940, 0.962, 0.965, 0.962], [0.619, 0.666, 0.614, 0.669],
        [0.972, 0.981, 0.975, 0.975], [0.957, 0.978, 0.946, 0.970],
    ]).T

    LADDER = np.array([
        [1.0, 1.1, 0.9, 1.2, 1.05, 1.0],
        [2.0, 2.1, 1.9, 2.2, 2.05, 2.0],
        [3.0, 3.1, 2.9, 3.2, 3.05, 3.0],
    ])
    NAMES = ['A', 'B', 'C']

    @pytest.mark.parametrize('k, alpha, expected', [
        (2, 0.05, 1.960), (3, 0.05, 2.343), (4, 0.05, 2.569), (10, 0.05, 3.164),
        (2, 0.10, 1.645), (4, 0.10, 2.291), (10, 0.10, 2.920),
    ])
    def test_critical_values_match_the_published_table(self, k, alpha, expected):
        # Table 5(a) of Demsar (2006), whose entries are truncated to 3 decimals
        assert StatisticsCalculator.nemenyi_critical_value(k, alpha) == \
               pytest.approx(expected, abs=1e-3)

    def test_critical_value_needs_two_algorithms(self):
        with pytest.raises(ValueError, match='at least 2 algorithms'):
            StatisticsCalculator.nemenyi_critical_value(1)

    def test_critical_value_rejects_an_alpha_outside_the_unit_interval(self):
        with pytest.raises(ValueError, match='must lie in'):
            StatisticsCalculator.nemenyi_critical_value(4, 1.5)

    def test_critical_difference_of_the_paper_example(self):
        # CD = 2.569 * sqrt(4*5 / (6*14)) = 1.25 at alpha = 0.05
        result = StatisticsCalculator.perform_nemenyi_test(
            self.PAPER_AUC, self.PAPER_NAMES,
            direction=OptimizationDirection.MAXIMIZE, significance_level=0.05
        )
        assert result.q_alpha == pytest.approx(2.569, abs=1e-3)
        assert result.critical_difference == pytest.approx(1.25, abs=5e-3)
        assert result.n_algorithms == 4
        assert result.n_instances == 14

    def test_no_pair_is_significant_at_five_percent_in_the_paper_example(self):
        # "even the difference between the best and the worst is smaller than CD"
        result = StatisticsCalculator.perform_nemenyi_test(
            self.PAPER_AUC, self.PAPER_NAMES,
            direction=OptimizationDirection.MAXIMIZE, significance_level=0.05
        )
        assert not any(comparison.significant for comparison in result.comparisons)
        assert result.cliques == [['C4.5+m+cf', 'C4.5+m', 'C4.5+cf', 'C4.5']]

    def test_two_groups_emerge_at_ten_percent_in_the_paper_example(self):
        # The paper's Figure 1(a): C4.5 is worse than C4.5+m and C4.5+m+cf,
        # while C4.5+cf cannot be assigned to either group
        result = StatisticsCalculator.perform_nemenyi_test(
            self.PAPER_AUC, self.PAPER_NAMES,
            direction=OptimizationDirection.MAXIMIZE, significance_level=0.10
        )
        assert result.critical_difference == pytest.approx(1.12, abs=5e-3)

        significant = {(c.algorithm_a, c.algorithm_b)
                       for c in result.comparisons if c.significant}
        assert significant == {('C4.5+m', 'C4.5'), ('C4.5+m+cf', 'C4.5')}
        assert result.cliques == [['C4.5+m+cf', 'C4.5+m', 'C4.5+cf'],
                                  ['C4.5+cf', 'C4.5']]

    def test_every_unordered_pair_is_compared_once(self):
        result = StatisticsCalculator.perform_nemenyi_test(self.LADDER, self.NAMES)
        assert len(result.comparisons) == 3
        pairs = {frozenset((c.algorithm_a, c.algorithm_b)) for c in result.comparisons}
        assert len(pairs) == 3

    def test_comparisons_are_ordered_by_rank_difference(self):
        result = StatisticsCalculator.perform_nemenyi_test(self.LADDER, self.NAMES)
        differences = [c.rank_difference for c in result.comparisons]
        assert differences == sorted(differences)

    def test_the_better_algorithm_comes_first_in_a_pair(self):
        result = StatisticsCalculator.perform_nemenyi_test(self.LADDER, self.NAMES)
        for comparison in result.comparisons:
            assert (result.average_ranks[comparison.algorithm_a]
                    <= result.average_ranks[comparison.algorithm_b])

    def test_significance_agrees_with_the_critical_difference(self):
        result = StatisticsCalculator.perform_nemenyi_test(self.LADDER, self.NAMES)
        for comparison in result.comparisons:
            assert comparison.significant == \
                   (comparison.rank_difference > result.critical_difference)

    def test_significance_agrees_with_the_p_value(self):
        result = StatisticsCalculator.perform_nemenyi_test(self.LADDER, self.NAMES)
        for comparison in result.comparisons:
            assert comparison.significant == (comparison.p_value < 0.05)

    def test_direction_flips_the_ranks(self):
        minimize = StatisticsCalculator.perform_nemenyi_test(self.LADDER, self.NAMES)
        maximize = StatisticsCalculator.perform_nemenyi_test(
            self.LADDER, self.NAMES, direction=OptimizationDirection.MAXIMIZE)
        assert minimize.average_ranks['A'] == pytest.approx(1.0)
        assert maximize.average_ranks['A'] == pytest.approx(3.0)
        assert minimize.critical_difference == pytest.approx(maximize.critical_difference)

    def test_ranks_match_the_friedman_test_on_the_same_matrix(self):
        friedman = StatisticsCalculator.perform_friedman_test(self.LADDER, self.NAMES)
        nemenyi = StatisticsCalculator.perform_nemenyi_test(self.LADDER, self.NAMES)
        assert nemenyi.average_ranks == pytest.approx(friedman.average_ranks)

    def test_a_wider_alpha_gives_a_smaller_critical_difference(self):
        strict = StatisticsCalculator.perform_nemenyi_test(
            self.LADDER, self.NAMES, significance_level=0.01)
        loose = StatisticsCalculator.perform_nemenyi_test(
            self.LADDER, self.NAMES, significance_level=0.10)
        assert loose.critical_difference < strict.critical_difference

    def test_all_algorithms_tied_form_one_clique(self):
        result = StatisticsCalculator.perform_nemenyi_test(np.ones((3, 4)), self.NAMES)
        assert result.cliques == [['A', 'B', 'C']]
        assert not any(comparison.significant for comparison in result.comparisons)

    def test_fully_separated_algorithms_form_no_clique(self):
        # 40 instances make the critical difference small enough to split all three
        separated = np.repeat(np.array([[1.0], [2.0], [3.0]]), 40, axis=1)
        result = StatisticsCalculator.perform_nemenyi_test(separated, self.NAMES)
        assert result.cliques == []
        assert all(comparison.significant for comparison in result.comparisons)

    def test_cliques_are_maximal_and_not_nested(self):
        result = StatisticsCalculator.perform_nemenyi_test(
            self.PAPER_AUC, self.PAPER_NAMES,
            direction=OptimizationDirection.MAXIMIZE, significance_level=0.10
        )
        for outer in result.cliques:
            for inner in result.cliques:
                if outer is not inner:
                    assert not set(inner).issubset(set(outer))

    def test_instances_with_nan_are_dropped(self):
        matrix = self.LADDER.copy()
        matrix[1, 2] = np.nan
        with pytest.warns(UserWarning, match='Dropped 1 instance'):
            result = StatisticsCalculator.perform_nemenyi_test(matrix, self.NAMES)
        assert result.n_instances == 5
        assert result.n_instances_dropped == 1

    def test_fewer_than_three_algorithms_raises(self):
        with pytest.raises(ValueError, match='at least 3 algorithms'):
            StatisticsCalculator.perform_nemenyi_test(
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), ['A', 'B']
            )

    def test_fewer_than_two_instances_raises(self):
        with pytest.raises(ValueError, match='at least 2 complete instances'):
            StatisticsCalculator.perform_nemenyi_test(
                np.array([[1.0], [2.0], [3.0]]), self.NAMES
            )


# =============================================================================
# 3. Cliff's delta
# =============================================================================

class TestCliffsDelta:
    """Cliff's delta effect size and its magnitude thresholds."""

    def test_complete_separation_in_favour_of_the_algorithm(self):
        # Minimization: every algorithm value below every baseline value
        result = StatisticsCalculator.cliffs_delta([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
        assert result.delta == pytest.approx(1.0)
        assert result.magnitude == 'large'

    def test_complete_separation_against_the_algorithm(self):
        result = StatisticsCalculator.cliffs_delta([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])
        assert result.delta == pytest.approx(-1.0)
        assert result.magnitude == 'large'

    def test_direction_flips_the_sign(self):
        algo, base = [1.0, 2.0, 3.0], [10.0, 11.0, 12.0]
        minimize = StatisticsCalculator.cliffs_delta(algo, base)
        maximize = StatisticsCalculator.cliffs_delta(
            algo, base, direction=OptimizationDirection.MAXIMIZE
        )
        assert minimize.delta == pytest.approx(1.0)
        assert maximize.delta == pytest.approx(-1.0)

    def test_identical_samples_give_zero(self):
        sample = [1.0, 2.0, 3.0, 4.0]
        result = StatisticsCalculator.cliffs_delta(sample, list(sample))
        assert result.delta == pytest.approx(0.0)
        assert result.magnitude == 'negligible'

    def test_same_distribution_is_near_zero(self):
        rng = np.random.default_rng(12345)
        algo = rng.normal(size=400).tolist()
        base = rng.normal(size=400).tolist()
        result = StatisticsCalculator.cliffs_delta(algo, base)
        assert abs(result.delta) < 0.147
        assert result.magnitude == 'negligible'

    def test_all_identical_values_give_zero(self):
        result = StatisticsCalculator.cliffs_delta([5.0] * 6, [5.0] * 6)
        assert result.delta == pytest.approx(0.0)
        assert result.magnitude == 'negligible'

    def test_known_partial_overlap(self):
        # Of the 9 pairs, 6 have algo < base (a win when minimizing), 1 has
        # algo > base and 2 are ties -> (6 - 1) / 9
        result = StatisticsCalculator.cliffs_delta([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert result.delta == pytest.approx(5.0 / 9.0)

    def test_single_run_each(self):
        assert StatisticsCalculator.cliffs_delta([1.0], [2.0]).delta == pytest.approx(1.0)
        assert StatisticsCalculator.cliffs_delta([2.0], [2.0]).delta == pytest.approx(0.0)

    def test_nan_entries_are_dropped(self):
        with_nan = StatisticsCalculator.cliffs_delta([1.0, np.nan, 3.0], [10.0, 11.0])
        without_nan = StatisticsCalculator.cliffs_delta([1.0, 3.0], [10.0, 11.0])
        assert with_nan.delta == pytest.approx(without_nan.delta)

    def test_empty_sample_is_undefined(self):
        result = StatisticsCalculator.cliffs_delta([], [1.0, 2.0])
        assert np.isnan(result.delta)
        assert result.magnitude == 'undefined'

    def test_all_nan_sample_is_undefined(self):
        result = StatisticsCalculator.cliffs_delta([np.nan, np.nan], [1.0, 2.0])
        assert np.isnan(result.delta)
        assert result.magnitude == 'undefined'

    def test_delta_stays_within_bounds(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            algo = rng.normal(size=15).tolist()
            base = rng.normal(loc=0.5, size=15).tolist()
            assert -1.0 <= StatisticsCalculator.cliffs_delta(algo, base).delta <= 1.0

    @pytest.mark.parametrize('delta, expected', [
        (0.0, 'negligible'),
        (0.146, 'negligible'),
        (0.147, 'small'),
        (0.329, 'small'),
        (0.33, 'medium'),
        (0.473, 'medium'),
        (0.474, 'large'),
        (1.0, 'large'),
        (-0.5, 'large'),
        (np.nan, 'undefined'),
    ])
    def test_magnitude_thresholds(self, delta, expected):
        assert StatisticsCalculator.classify_cliffs_delta(delta) == expected

    def test_method_name_is_reported(self):
        assert StatisticsCalculator.cliffs_delta([1.0], [2.0]).method == 'cliffs_delta'

    def test_rank_sum_test_reports_the_effect_size_on_request(self):
        algo, base = [1.0, 2.0, 3.0], [10.0, 11.0, 12.0]
        without = StatisticsCalculator.perform_rank_sum_test(algo, base)
        assert without.effect_size is None
        assert without.effect_magnitude is None

        with_effect = StatisticsCalculator.perform_rank_sum_test(
            algo, base, compute_effect_size=True
        )
        assert with_effect.effect_size == pytest.approx(1.0)
        assert with_effect.effect_magnitude == 'large'
        # The significance verdict is untouched by asking for the effect size
        assert with_effect.symbol == without.symbol
        assert with_effect.p_value == pytest.approx(without.p_value)


# =============================================================================
# 4. median[IQR]
# =============================================================================

class TestMedianIQR:
    """The MEDIAN_IQR statistic and its rendering."""

    def test_known_quartiles(self):
        # 1..9: median 5, Q1 3, Q3 7 -> IQR 4
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        median, iqr = StatisticsCalculator.calculate_statistic(
            data, StatisticType.MEDIAN_IQR
        )
        assert median == pytest.approx(5.0)
        assert iqr == pytest.approx(4.0)

    def test_matches_numpy_percentiles(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=37).tolist()
        median, iqr = StatisticsCalculator.calculate_statistic(
            data, StatisticType.MEDIAN_IQR
        )
        assert median == pytest.approx(np.median(data))
        assert iqr == pytest.approx(np.percentile(data, 75) - np.percentile(data, 25))

    def test_even_sample_size(self):
        # 1..4: median 2.5, Q1 1.75, Q3 3.25 -> IQR 1.5
        median, iqr = StatisticsCalculator.calculate_statistic(
            [1.0, 2.0, 3.0, 4.0], StatisticType.MEDIAN_IQR
        )
        assert median == pytest.approx(2.5)
        assert iqr == pytest.approx(1.5)

    def test_identical_values_give_zero_iqr(self):
        median, iqr = StatisticsCalculator.calculate_statistic(
            [2.5] * 8, StatisticType.MEDIAN_IQR
        )
        assert median == pytest.approx(2.5)
        assert iqr == pytest.approx(0.0)

    def test_single_run_gives_zero_iqr(self):
        median, iqr = StatisticsCalculator.calculate_statistic(
            [4.2], StatisticType.MEDIAN_IQR
        )
        assert median == pytest.approx(4.2)
        assert iqr == pytest.approx(0.0)

    def test_empty_data(self):
        median, iqr = StatisticsCalculator.calculate_statistic([], StatisticType.MEDIAN_IQR)
        assert np.isnan(median)
        assert np.isnan(iqr)

    def test_median_matches_the_plain_median_statistic(self):
        data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        plain, none_value = StatisticsCalculator.calculate_statistic(
            data, StatisticType.MEDIAN
        )
        with_iqr, iqr = StatisticsCalculator.calculate_statistic(
            data, StatisticType.MEDIAN_IQR
        )
        assert none_value is None
        assert with_iqr == pytest.approx(plain)
        assert iqr is not None

    def test_rendered_as_median_bracket_iqr(self, tmp_path):
        best_values = make_best_values({
            'A': {'P1': [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]]},
            'B': {'P1': [[10.0]] * 9},
        })
        config = TableConfig(
            table_format=TableFormat.EXCEL,
            statistic_type=StatisticType.MEDIAN_IQR,
            rank_sum_test=False,
            save_path=tmp_path,
        )
        df = TableGenerator(config).generate(best_values, ['A', 'B'])
        assert df.loc[0, 'A'] == '5.000e+00[4.0e+00]'

    def test_latex_rendering(self, tmp_path):
        best_values = make_best_values({
            'A': {'P1': [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]]},
            'B': {'P1': [[10.0]] * 9},
        })
        config = TableConfig(
            table_format=TableFormat.LATEX,
            statistic_type=StatisticType.MEDIAN_IQR,
            rank_sum_test=False,
            save_path=tmp_path,
        )
        latex = TableGenerator(config).generate(best_values, ['A', 'B'])
        assert '$5.000e+00$[$4.0e+00$]' in latex

    def test_best_value_is_still_detected(self, tmp_path):
        """The median[IQR] cell must still parse for best-value highlighting."""
        best_values = make_best_values({
            'A': {'P1': [[1.0], [2.0], [3.0]]},
            'B': {'P1': [[10.0], [11.0], [12.0]]},
        })
        config = TableConfig(
            table_format=TableFormat.LATEX,
            statistic_type=StatisticType.MEDIAN_IQR,
            rank_sum_test=False,
            save_path=tmp_path,
        )
        generator = TableGenerator(config)
        latex = generator.generate(best_values, ['A', 'B'])
        # A is the better (smaller) algorithm, so its cell carries the highlight
        assert '\\best $2.000e+00$' in latex

    def test_representative_run_matches_median(self):
        best_values = make_best_values({'A': {'P1': [[1.0], [5.0], [9.0]]}})
        median_run = StatisticsCalculator.select_representative_run(
            best_values, 'A', 'P1', 0, StatisticType.MEDIAN
        )
        iqr_run = StatisticsCalculator.select_representative_run(
            best_values, 'A', 'P1', 0, StatisticType.MEDIAN_IQR
        )
        assert median_run == iqr_run == 2


# =============================================================================
# 5. Table integration of the new switches
# =============================================================================

@pytest.fixture
def three_algorithm_values():
    """Three algorithms over four instances, A clearly best and C the baseline.

    Also reused by the instance-matrix and critical difference diagram tests.
    """
    rng = np.random.default_rng(2024)
    values = {}
    for algo, offset in [('A', 0.0), ('B', 5.0), ('C', 10.0)]:
        values[algo] = {}
        for prob in ['P1', 'P2', 'P3', 'P4']:
            samples = offset + rng.normal(scale=0.05, size=10)
            values[algo][prob] = [[float(v)] for v in samples]
    return make_best_values(values)


class TestTableIntegration:
    """The new options as seen through the generated table."""

    def test_holm_can_only_relax_symbols(self, three_algorithm_values, tmp_path):
        raw_config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path)
        holm_config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                                  holm_correction=True)

        raw = TableGenerator(raw_config).generate(three_algorithm_values, ['A', 'B', 'C'])
        holm = TableGenerator(holm_config).generate(three_algorithm_values, ['A', 'B', 'C'])

        data_rows = len(raw) - 2  # +/-/= and Average Rank footers
        for row in range(data_rows):
            for algo in ['A', 'B']:
                raw_symbol = str(raw.loc[row, algo]).split()[-1]
                holm_symbol = str(holm.loc[row, algo]).split()[-1]
                assert holm_symbol in (raw_symbol, '=')

    def test_holm_records_adjusted_p_values(self, tmp_path):
        best_values = make_best_values({
            'A': {f'P{i}': [[1.0 + 0.01 * r] for r in range(8)] for i in range(1, 5)},
            'B': {f'P{i}': [[2.0 + 0.01 * r] for r in range(8)] for i in range(1, 5)},
        })
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             holm_correction=True)
        generator = TableGenerator(config)
        rows, _, _, _ = generator._generate_data_rows(
            best_values, ['A', 'B'], OptimizationDirection.MINIMIZE
        )
        assert len(rows) == 4

    def test_effect_size_is_a_separate_field(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             effect_size=True)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        cell = str(df.loc[0, 'A'])
        # The symbol and the effect size are distinct fields of the cell
        assert ' + [d=' in cell
        assert cell.endswith('large]')
        # ...and the leading value is still parseable for highlighting
        assert cell_number(cell) == pytest.approx(cell_number(str(df.loc[0, 'A'])))

    def test_effect_size_absent_for_the_baseline_column(self, three_algorithm_values,
                                                        tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             effect_size=True)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        assert '[d=' not in str(df.loc[0, 'C'])

    def test_friedman_rows_reach_the_table(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             friedman_test=True)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])

        labels = [str(v) for v in df['Problem']]
        rank_label = next(label for label in labels if label.startswith('Friedman Rank'))
        assert 'chi2=' in rank_label and 'p=' in rank_label
        assert any(label.startswith('Friedman p_Holm (vs C)') for label in labels)

        rank_row = df[df['Problem'].astype(str).str.startswith('Friedman Rank')].iloc[0]
        assert float(rank_row['A']) == pytest.approx(1.0)
        assert float(rank_row['C']) == pytest.approx(3.0)

        post_hoc_row = df[df['Problem'].astype(str).str.startswith('Friedman p_Holm')].iloc[0]
        assert post_hoc_row['C'] == 'Control'
        assert str(post_hoc_row['A']).endswith('+')

    def test_average_rank_stays_the_last_row(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             friedman_test=True)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        assert str(df.iloc[-1]['Problem']) == 'Average Rank'

    def test_tied_instances_share_the_average_rank(self, tmp_path):
        # Every algorithm solves P2 and P3 equally well, so no one may be handed
        # a better rank there just for coming first in the column order
        best_values = make_best_values({
            'A': {'P1': [[1.0]] * 3, 'P2': [[5.0]] * 3, 'P3': [[5.0]] * 3},
            'B': {'P1': [[2.0]] * 3, 'P2': [[5.0]] * 3, 'P3': [[5.0]] * 3},
            'C': {'P1': [[3.0]] * 3, 'P2': [[5.0]] * 3, 'P3': [[5.0]] * 3},
        })
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             rank_sum_test=False)
        df = TableGenerator(config).generate(best_values, ['A', 'B', 'C'])

        row = df[df['Problem'] == 'Average Rank'].iloc[0]
        # P1 ranks 1/2/3, P2 and P3 are three-way ties worth 2 each; the row is
        # rendered with two decimals
        assert float(row['A']) == pytest.approx((1 + 2 + 2) / 3, abs=5e-3)
        assert float(row['B']) == pytest.approx((2 + 2 + 2) / 3, abs=5e-3)
        assert float(row['C']) == pytest.approx((3 + 2 + 2) / 3, abs=5e-3)

    def test_average_rank_agrees_with_the_friedman_rank(self, tmp_path):
        # The two footer rows rank the same numbers, so they must not disagree
        best_values = make_best_values({
            'A': {f'P{i}': [[1.0]] * 3 for i in range(1, 5)},
            'B': {f'P{i}': [[1.0]] * 3 for i in range(1, 5)},
            'C': {f'P{i}': [[2.0 + i]] * 3 for i in range(1, 5)},
        })
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             friedman_test=True, rank_sum_test=False)
        df = TableGenerator(config).generate(best_values, ['A', 'B', 'C'])

        friedman = df[df['Problem'].astype(str).str.startswith('Friedman Rank')].iloc[0]
        average = df[df['Problem'] == 'Average Rank'].iloc[0]
        for algo in ('A', 'B', 'C'):
            assert float(average[algo]) == pytest.approx(float(friedman[algo]))

    def test_friedman_rows_reach_the_latex_table(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.LATEX, save_path=tmp_path,
                             friedman_test=True)
        latex = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        assert '\\chi^2_F=' in latex
        assert 'Friedman $p_{Holm}$ (vs C)' in latex
        assert 'Control' in latex

    def test_friedman_control_is_configurable(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             friedman_test=True, friedman_control='A')
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        post_hoc_row = df[df['Problem'].astype(str).str.startswith('Friedman p_Holm')].iloc[0]
        assert post_hoc_row['A'] == 'Control'
        assert str(post_hoc_row['C']).endswith('-')

    def test_friedman_with_two_algorithms_raises(self, tmp_path):
        best_values = make_best_values({
            'A': {f'P{i}': [[1.0], [1.1], [0.9]] for i in range(1, 4)},
            'B': {f'P{i}': [[2.0], [2.1], [1.9]] for i in range(1, 4)},
        })
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             friedman_test=True)
        with pytest.raises(ValueError, match='at least 3 algorithms'):
            TableGenerator(config).generate(best_values, ['A', 'B'])

    def test_all_options_together(self, three_algorithm_values, tmp_path):
        config = TableConfig(
            table_format=TableFormat.EXCEL,
            statistic_type=StatisticType.MEDIAN_IQR,
            save_path=tmp_path,
            holm_correction=True,
            effect_size=True,
            friedman_test=True,
        )
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        cell = str(df.loc[0, 'A'])
        assert '[' in cell and '[d=' in cell
        assert cell_number(cell) == pytest.approx(0.0, abs=0.5)
        assert (tmp_path / 'results_table_median_iqr.xlsx').exists()


# =============================================================================
# 5b. Instance matrix and critical difference diagram
# =============================================================================

class TestInstanceMatrix:
    """The bridge from per-run results to the algorithm-by-instance matrix."""

    def test_shape_and_labels_of_single_task_problems(self, three_algorithm_values):
        matrix, labels = StatisticsCalculator.build_instance_matrix(
            three_algorithm_values, ['A', 'B', 'C'])
        assert matrix.shape == (3, 4)
        assert labels == ['P1', 'P2', 'P3', 'P4']

    def test_multi_task_problems_are_labelled_per_task(self):
        best_values = make_best_values({
            algo: {'P1': [[1.0, 2.0], [1.1, 2.1]], 'P2': [[3.0, 4.0], [3.1, 4.1]]}
            for algo in ('A', 'B', 'C')
        })
        matrix, labels = StatisticsCalculator.build_instance_matrix(
            best_values, ['A', 'B', 'C'])
        assert matrix.shape == (3, 4)
        assert labels == ['P1-T1', 'P1-T2', 'P2-T1', 'P2-T2']

    def test_entries_are_the_displayed_statistic(self, three_algorithm_values):
        matrix, _ = StatisticsCalculator.build_instance_matrix(
            three_algorithm_values, ['A', 'B', 'C'], StatisticType.MEDIAN)
        expected = np.median(
            StatisticsCalculator.collect_task_data(three_algorithm_values, 'B', 'P2', 0))
        assert matrix[1, 1] == pytest.approx(expected)

    def test_rows_follow_the_requested_algorithm_order(self, three_algorithm_values):
        forward, _ = StatisticsCalculator.build_instance_matrix(
            three_algorithm_values, ['A', 'B', 'C'])
        reversed_, _ = StatisticsCalculator.build_instance_matrix(
            three_algorithm_values, ['C', 'B', 'A'])
        assert forward[0] == pytest.approx(reversed_[2])

    def test_instances_match_the_rows_of_the_table(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path,
                             rank_sum_test=False)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        _, labels = StatisticsCalculator.build_instance_matrix(
            three_algorithm_values, ['A', 'B', 'C'])
        # The table appends one Average Rank footer row to the data rows
        assert len(labels) == len(df) - 1


class TestCriticalDifferenceDiagram:
    """Rendering of the Nemenyi diagram of Demsar (2006, Figure 1a)."""

    NAMES = ['A', 'B', 'C', 'D']
    MATRIX = np.array([
        [1.0, 1.1, 0.9, 1.2, 1.05, 1.0, 0.95, 1.15],
        [2.0, 1.9, 2.1, 2.2, 1.95, 2.0, 2.05, 1.85],
        [2.5, 2.6, 2.4, 2.7, 2.55, 2.5, 2.45, 2.65],
        [4.0, 4.1, 3.9, 4.2, 4.05, 4.0, 3.95, 4.15],
    ])

    def result(self, alpha=0.05):
        return StatisticsCalculator.perform_nemenyi_test(
            self.MATRIX, self.NAMES, significance_level=alpha)

    def test_figure_is_written(self, tmp_path):
        config = PlotConfig(save_path=tmp_path, figure_format='png')
        output = PlotGenerator(config).plot_cd_diagram(self.result())
        assert output == tmp_path / 'cd_diagram.png'
        assert output.exists() and output.stat().st_size > 0

    def test_filename_and_format_are_configurable(self, tmp_path):
        config = PlotConfig(save_path=tmp_path, figure_format='svg')
        output = PlotGenerator(config).plot_cd_diagram(
            self.result(), metric_name='IGD', filename='nemenyi')
        assert output == tmp_path / 'nemenyi.svg'
        assert output.exists()

    def test_save_directory_is_created(self, tmp_path):
        target = tmp_path / 'nested' / 'results'
        config = PlotConfig(save_path=target, figure_format='png')
        assert PlotGenerator(config).plot_cd_diagram(self.result()).exists()

    def test_every_algorithm_and_the_critical_difference_are_labelled(self, tmp_path):
        config = PlotConfig(save_path=tmp_path, figure_format='svg')
        output = PlotGenerator(config).plot_cd_diagram(self.result(), metric_name='IGD')
        content = output.read_text(encoding='utf-8')
        for name in self.NAMES:
            assert f'>{name}<' in content or name in content
        assert 'CD' in content

    def test_a_diagram_without_cliques_still_renders(self, tmp_path):
        separated = np.repeat(np.array([[1.0], [2.0], [3.0]]), 40, axis=1)
        result = StatisticsCalculator.perform_nemenyi_test(separated, ['A', 'B', 'C'])
        assert result.cliques == []
        config = PlotConfig(save_path=tmp_path, figure_format='png')
        assert PlotGenerator(config).plot_cd_diagram(result).exists()

    def test_a_single_clique_of_all_algorithms_still_renders(self, tmp_path):
        result = StatisticsCalculator.perform_nemenyi_test(np.ones((3, 4)), ['A', 'B', 'C'])
        assert len(result.cliques) == 1
        config = PlotConfig(save_path=tmp_path, figure_format='png')
        assert PlotGenerator(config).plot_cd_diagram(result).exists()

    def test_a_critical_difference_wider_than_the_axis_still_renders(self, tmp_path):
        # Two instances make CD larger than the 1..3 the axis spans
        result = StatisticsCalculator.perform_nemenyi_test(
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), ['A', 'B', 'C'])
        assert result.critical_difference > 2.0
        config = PlotConfig(save_path=tmp_path, figure_format='png')
        assert PlotGenerator(config).plot_cd_diagram(result).exists()

    def test_many_algorithms_render_without_error(self, tmp_path):
        names = [f'ALG{i:02d}' for i in range(1, 13)]
        rng = np.random.default_rng(7)
        matrix = np.linspace(0.0, 2.0, 12)[:, None] + rng.normal(0, 0.7, size=(12, 25))
        result = StatisticsCalculator.perform_nemenyi_test(matrix, names)
        config = PlotConfig(save_path=tmp_path, figure_format='png')
        assert PlotGenerator(config).plot_cd_diagram(result).exists()


# =============================================================================
# 6. Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    """With every new switch off, the output must be identical to before."""

    def test_defaults_are_all_off(self):
        config = TableConfig()
        assert config.holm_correction is False
        assert config.effect_size is False
        assert config.friedman_test is False
        assert config.friedman_control is None
        # ...and the pre-existing defaults are untouched
        assert config.rank_sum_test is True
        assert config.significance_level == 0.05
        assert config.statistic_type == StatisticType.MEAN

    def test_comparison_result_new_fields_default_to_none(self):
        result = ComparisonResult(symbol='+', p_value=0.01)
        assert result.p_adjusted is None
        assert result.effect_size is None
        assert result.effect_magnitude is None

    def test_mean_cells_are_unchanged(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        cell = str(df.loc[0, 'A'])
        assert '[' not in cell  # no effect size annotation
        assert cell.count('(') == 1  # only the std part
        assert cell.split()[-1] in ('+', '-', '=')

    def test_median_cells_are_unchanged(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL,
                             statistic_type=StatisticType.MEDIAN, save_path=tmp_path)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        cell = str(df.loc[0, 'A'])
        assert '(' not in cell and '[' not in cell

    def test_no_friedman_rows_by_default(self, three_algorithm_values, tmp_path):
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])
        assert not any('Friedman' in str(v) for v in df['Problem'])
        assert list(df['Problem'])[-2:] == ['+/-/=', 'Average Rank']

    def test_rank_sum_symbols_unchanged_without_holm(self, three_algorithm_values,
                                                     tmp_path):
        """Symbols must follow the raw p-values when the correction is off."""
        config = TableConfig(table_format=TableFormat.EXCEL, save_path=tmp_path)
        df = TableGenerator(config).generate(three_algorithm_values, ['A', 'B', 'C'])

        expected = StatisticsCalculator.perform_rank_sum_test(
            [v[0] for v in three_algorithm_values['A']['P1'].values()],
            [v[0] for v in three_algorithm_values['C']['P1'].values()],
            0.05, OptimizationDirection.MINIMIZE
        )
        assert str(df.loc[0, 'A']).split()[-1] == expected.symbol

    def test_perform_rank_sum_test_positional_call_still_works(self):
        result = StatisticsCalculator.perform_rank_sum_test(
            [1.0, 2.0, 3.0], [10.0, 11.0, 12.0], 0.05, OptimizationDirection.MINIMIZE
        )
        assert result.symbol == '+'
        assert result.effect_size is None

    def test_existing_statistic_types_unchanged(self):
        data = [1.0, 2.0, 3.0, 4.0]
        mean, std = StatisticsCalculator.calculate_statistic(data, StatisticType.MEAN)
        assert mean == pytest.approx(2.5)
        assert std == pytest.approx(np.std(data, ddof=1))

        assert StatisticsCalculator.calculate_statistic(data, StatisticType.MEDIAN) == \
               (pytest.approx(2.5), None)
        assert StatisticsCalculator.calculate_statistic(data, StatisticType.MAX) == \
               (pytest.approx(4.0), None)
        assert StatisticsCalculator.calculate_statistic(data, StatisticType.MIN) == \
               (pytest.approx(1.0), None)

    def test_statistic_type_values_are_stable(self):
        assert StatisticType('mean') == StatisticType.MEAN
        assert StatisticType('median') == StatisticType.MEDIAN
        assert StatisticType('max') == StatisticType.MAX
        assert StatisticType('min') == StatisticType.MIN
        assert StatisticType('median_iqr') == StatisticType.MEDIAN_IQR

    def test_excel_output_is_byte_identical_with_switches_off(self, three_algorithm_values,
                                                              tmp_path):
        """A table built with the switches explicitly off equals the default one."""
        default_dir = tmp_path / 'default'
        explicit_dir = tmp_path / 'explicit'

        default_df = TableGenerator(TableConfig(
            table_format=TableFormat.EXCEL, save_path=default_dir
        )).generate(three_algorithm_values, ['A', 'B', 'C'])

        explicit_df = TableGenerator(TableConfig(
            table_format=TableFormat.EXCEL, save_path=explicit_dir,
            holm_correction=False, effect_size=False, friedman_test=False
        )).generate(three_algorithm_values, ['A', 'B', 'C'])

        assert default_df.equals(explicit_df)
        assert (default_dir / 'results_table_mean.xlsx').read_bytes() == \
               (explicit_dir / 'results_table_mean.xlsx').read_bytes()

    def test_latex_output_unchanged_with_switches_off(self, three_algorithm_values,
                                                      tmp_path):
        default_latex = TableGenerator(TableConfig(
            table_format=TableFormat.LATEX, save_path=tmp_path / 'a'
        )).generate(three_algorithm_values, ['A', 'B', 'C'])

        explicit_latex = TableGenerator(TableConfig(
            table_format=TableFormat.LATEX, save_path=tmp_path / 'b',
            holm_correction=False, effect_size=False, friedman_test=False
        )).generate(three_algorithm_values, ['A', 'B', 'C'])

        assert default_latex == explicit_latex
        assert 'Friedman' not in default_latex
        assert '[d=' not in default_latex

"""
Tests for :mod:`ddmtolab.Methods.statistical_tests`.

The reference values come from the worked examples of the two papers the module
implements, so the tests double as a check that the implementation agrees with
the published methodology:

- J. Demsar, JMLR 7 (2006) 1-30 -- Table 2 (Wilcoxon signed ranks).
- J. Derrac et al., Swarm and Evolutionary Computation 1 (2011) 3-18 --
  Tables 7-11 (rankings), 13-15 (post-hoc z values), 16 (control APVs) and
  20 (all-pairs APVs).

Two entries of those tables are internally inconsistent and are documented at
the assertions that skip them.
"""

import numpy as np
import pytest

from ddmtolab.Methods.statistical_tests import (
    ALL_PAIRS_PROCEDURES,
    CONTROL_PROCEDURES,
    OptimizationDirection,
    RankScheme,
    RankingResult,
    adjust_p_values,
    all_pairs_post_hoc,
    contrast_estimation,
    control_post_hoc,
    exhaustive_sets,
    friedman_aligned_test,
    friedman_test,
    omnibus_test,
    quade_test,
    shaffer_t_values,
    sign_test,
    wilcoxon_signed_rank_test,
)


# =============================================================================
# Fixtures from the papers
# =============================================================================

#: Derrac et al. Table 7: error rate of four algorithms on four problems.
TOY_NAMES = ['A', 'B', 'C', 'D']
TOY_ERRORS = np.array([
    [2.711, 3.147, 2.515, 2.612],
    [7.832, 9.828, 7.832, 7.921],
    [0.012, 0.532, 0.122, 0.005],
    [3.431, 4.111, 3.401, 3.401],
]).T

#: Derrac et al. Table 11: the nine algorithms of the main case of study.
NINE_ALGORITHMS = ['PSO', 'IPOP-CMA-ES', 'CHC', 'SSGA', 'SS-BLX', 'SS-Arit',
                   'DE-Bin', 'DE-Exp', 'SaDE']

#: Derrac et al. Table 16: unadjusted p-values against the control DE-Exp.
CONTROL_P_VALUES = [0.000006, 0.000332, 0.009823, 0.014171,
                    0.083642, 0.141093, 0.518605, 0.660706]

#: Derrac et al. Table 20: unadjusted p-values of all 36 pairwise hypotheses.
PAIRWISE_P_VALUES = [
    0.000006, 0.000045, 0.000108, 0.000332, 0.001633, 0.002313, 0.003246, 0.005294,
    0.009823, 0.014171, 0.032109, 0.03424, 0.038867, 0.044015, 0.052808, 0.052808,
    0.063023, 0.070701, 0.083642, 0.141093, 0.196706, 0.255925, 0.266889, 0.278172,
    0.3017, 0.313946, 0.326516, 0.352622, 0.394183, 0.40867, 0.469706, 0.518605,
    0.660706, 0.796253, 0.836354, 0.897279,
]

PAIRWISE_HYPOTHESES = [
    ('PSO', 'DE-Exp'), ('PSO', 'SaDE'), ('PSO', 'DE-Bin'), ('CHC', 'DE-Exp'),
    ('CHC', 'SaDE'), ('PSO', 'SS-BLX'), ('CHC', 'DE-Bin'), ('PSO', 'IPOP-CMA-ES'),
    ('SSGA', 'DE-Exp'), ('SS-Arit', 'DE-Exp'), ('SSGA', 'SaDE'), ('CHC', 'SS-BLX'),
    ('PSO', 'SS-Arit'), ('SS-Arit', 'SaDE'), ('SSGA', 'DE-Bin'), ('PSO', 'SSGA'),
    ('IPOP-CMA-ES', 'CHC'), ('SS-Arit', 'DE-Bin'), ('IPOP-CMA-ES', 'DE-Exp'),
    ('SS-BLX', 'DE-Exp'), ('IPOP-CMA-ES', 'SaDE'), ('CHC', 'SS-Arit'),
    ('SSGA', 'SS-BLX'), ('IPOP-CMA-ES', 'DE-Bin'), ('SS-BLX', 'SaDE'), ('CHC', 'SSGA'),
    ('SS-BLX', 'SS-Arit'), ('PSO', 'CHC'), ('IPOP-CMA-ES', 'SSGA'), ('SS-BLX', 'DE-Bin'),
    ('IPOP-CMA-ES', 'SS-Arit'), ('DE-Bin', 'DE-Exp'), ('DE-Exp', 'SaDE'),
    ('IPOP-CMA-ES', 'SS-BLX'), ('DE-Bin', 'SaDE'), ('SSGA', 'SS-Arit'),
]

PAIRWISE_PAIRS = [
    tuple(sorted((NINE_ALGORITHMS.index(first), NINE_ALGORITHMS.index(second))))
    for first, second in PAIRWISE_HYPOTHESES
]

#: Demsar Table 2: AUC of C4.5 and of C4.5 with m tuned, on 14 data sets.
C45 = [0.763, 0.599, 0.954, 0.628, 0.882, 0.936, 0.661,
       0.583, 0.775, 1.000, 0.940, 0.619, 0.972, 0.957]
C45_M = [0.768, 0.591, 0.971, 0.661, 0.888, 0.931, 0.668,
         0.583, 0.838, 1.000, 0.962, 0.666, 0.981, 0.978]

# A published p-value rounded to 6 decimals and multiplied by the family size m
# carries an error of up to m * 5e-7
CONTROL_TOLERANCE = 8 * 5e-7
PAIRWISE_TOLERANCE = 36 * 5e-7


def ladder(n_instances=6, n_algorithms=3):
    """Matrix in which algorithm i is uniformly worse than algorithm i-1."""
    base = np.arange(1, n_algorithms + 1, dtype=float)[:, None]
    jitter = np.linspace(-0.1, 0.1, n_instances)[None, :]
    return base + jitter


# =============================================================================
# 1. Ranking schemes
# =============================================================================

class TestRankingSchemes:
    """Friedman, Friedman Aligned and Quade rankings on the paper's example."""

    def test_friedman_per_problem_ranks(self):
        # Derrac et al. Table 8
        result = friedman_test(TOY_ERRORS, TOY_NAMES)
        assert result.ranks == pytest.approx(np.array([
            [3, 1.5, 2, 3],
            [4, 4, 4, 4],
            [1, 1.5, 3, 1.5],
            [2, 3, 1, 1.5],
        ]))

    def test_friedman_average_ranks(self):
        # Table 8 prints 1.250 for algorithm C, but the ranks it lists in the
        # same column sum to 7, so the average is 1.75; the other three match
        result = friedman_test(TOY_ERRORS, TOY_NAMES)
        assert result.average_ranks == pytest.approx(
            {'A': 2.375, 'B': 4.0, 'C': 1.75, 'D': 1.875})

    def test_average_ranks_sum_to_the_expected_total(self):
        result = friedman_test(TOY_ERRORS, TOY_NAMES)
        k = result.n_algorithms
        assert sum(result.average_ranks.values()) == pytest.approx(k * (k + 1) / 2)

    def test_aligned_ranks(self):
        # Derrac et al. Table 9: all k*n aligned observations ranked together
        result = friedman_aligned_test(TOY_ERRORS, TOY_NAMES)
        assert result.ranks == pytest.approx(np.array([
            [12, 1.5, 8, 9],
            [14, 16, 13, 15],
            [4, 1.5, 11, 5.5],
            [10, 3, 7, 5.5],
        ]))
        assert result.average_ranks == pytest.approx(
            {'A': 7.625, 'B': 14.5, 'C': 5.5, 'D': 6.375})

    def test_quade_weighted_ranks(self):
        # Derrac et al. Table 10, row T_j
        result = quade_test(TOY_ERRORS, TOY_NAMES)
        assert result.average_ranks == pytest.approx(
            {'A': 2.3, 'B': 4.0, 'C': 1.55, 'D': 2.15})

    def test_friedman_statistic_of_the_main_study(self):
        # Reproduces chi^2_F = 35.99733 of Table 11 from its published ranks
        ranks = np.array([7, 4.84, 6.28, 5.5, 4.64, 5.4, 4, 3.5, 3.84])
        k, n = 9, 25
        statistic = (12.0 * n / (k * (k + 1))) * (np.sum(ranks ** 2) - k * (k + 1) ** 2 / 4)
        assert statistic == pytest.approx(35.99733, abs=1e-3)
        assert (n - 1) * statistic / (n * (k - 1) - statistic) == pytest.approx(
            5.267817, abs=1e-4)

    def test_iman_davenport_accompanies_the_friedman_statistic(self):
        result = friedman_test(TOY_ERRORS, TOY_NAMES)
        k, n = result.n_algorithms, result.n_instances
        expected = (n - 1) * result.statistic / (n * (k - 1) - result.statistic)
        assert result.iman_davenport_statistic == pytest.approx(expected)
        assert result.iman_davenport_p_value < result.p_value

    def test_quade_statistic_of_the_main_study(self):
        # Reproduces F_Q = 6.63067 of Table 11 from its published T_j
        weighted = np.array([6.5415, 4.7415, 7.1785, 5.8769, 5.1108,
                             5.6123, 3.5538, 3.1123, 3.2723])
        k, n = 9, 25
        sums = (weighted - (k + 1) / 2) * (n * (n + 1) / 2)
        a_term = n * (n + 1) * (2 * n + 1) * k * (k + 1) * (k - 1) / 72
        b_term = float(np.sum(sums ** 2)) / n
        assert (n - 1) * b_term / (a_term - b_term) == pytest.approx(6.63067, abs=1e-3)

    def test_only_the_friedman_scheme_reports_iman_davenport(self):
        assert np.isnan(friedman_aligned_test(TOY_ERRORS, TOY_NAMES).iman_davenport_statistic)
        assert np.isnan(quade_test(TOY_ERRORS, TOY_NAMES).iman_davenport_statistic)

    @pytest.mark.parametrize('scheme', list(RankScheme))
    def test_maximization_mirrors_minimization(self, scheme):
        minimized = omnibus_test(TOY_ERRORS, TOY_NAMES, scheme=scheme)
        maximized = omnibus_test(-TOY_ERRORS, TOY_NAMES,
                                 direction=OptimizationDirection.MAXIMIZE, scheme=scheme)
        assert minimized.average_ranks == pytest.approx(maximized.average_ranks)
        assert minimized.statistic == pytest.approx(maximized.statistic)

    @pytest.mark.parametrize('scheme', list(RankScheme))
    def test_identical_algorithms_are_not_significant(self, scheme):
        matrix = np.tile(np.linspace(1.0, 2.0, 8), (3, 1))
        result = omnibus_test(matrix, ['A', 'B', 'C'], scheme=scheme)
        assert result.p_value == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize('scheme', list(RankScheme))
    def test_instances_with_nan_are_dropped(self, scheme):
        matrix = ladder()
        matrix[1, 2] = np.nan
        with pytest.warns(UserWarning, match='Dropped 1 instance'):
            result = omnibus_test(matrix, ['A', 'B', 'C'], scheme=scheme)
        assert result.n_instances == 5
        assert result.n_instances_dropped == 1

    @pytest.mark.parametrize('scheme', list(RankScheme))
    def test_too_few_instances_raises(self, scheme):
        with pytest.raises(ValueError, match='at least 2 complete instances'):
            omnibus_test(np.array([[1.0], [2.0], [3.0]]), ['A', 'B', 'C'], scheme=scheme)

    def test_name_count_mismatch_raises(self):
        with pytest.raises(ValueError, match='must match'):
            friedman_test(ladder(), ['A', 'B'])

    def test_non_2d_matrix_raises(self):
        with pytest.raises(ValueError, match='must be 2-D'):
            friedman_test(np.array([1.0, 2.0, 3.0]), ['A', 'B', 'C'])

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match='Unknown ranking scheme'):
            omnibus_test(ladder(), ['A', 'B', 'C'], scheme='friedman')


# =============================================================================
# 2. Post-hoc comparisons against a control
# =============================================================================

def ranking_of(scheme, names, ranks, standard_error, n_instances=25):
    """Build a RankingResult carrying published ranks, to test the z formulas."""
    average_ranks = dict(zip(names, ranks))
    return RankingResult(
        scheme=scheme, average_ranks=average_ranks,
        ranks=np.zeros((len(average_ranks), n_instances)),
        statistic=np.nan, p_value=np.nan, statistic_name='',
        standard_error=standard_error, n_algorithms=len(average_ranks),
        n_instances=n_instances
    )


class TestControlPostHoc:
    """The 1xN family: every algorithm against one control."""

    FOUR = ['IPOP-CMA-ES', 'CHC', 'SS-BLX', 'SaDE']
    K, N = 4, 25

    # Derrac et al. Tables 12-15: ranks, standard errors, z values and p-values
    CASES = [
        (RankScheme.FRIEDMAN, [2.48, 3.12, 2.44, 1.96],
         np.sqrt(K * (K + 1) / (6 * N)),
         [3.176791, 1.424079, 1.314534], [0.001489, 0.154424, 0.188667]),
        (RankScheme.ALIGNED, [51.96, 65.92, 48.52, 35.6],
         np.sqrt(K * (K * N + 1) / 6),
         [3.694997, 1.993739, 1.574517], [0.000220, 0.046181, 0.115368]),
        (RankScheme.QUADE, [2.3785, 3.4185, 2.48, 1.7231],
         np.sqrt(K * (K + 1) * (2 * N + 1) * (K - 1) / (18 * N * (N + 1))),
         [3.315129, 1.480076, 1.281529], [0.000916, 0.138853, 0.200008]),
    ]

    @pytest.mark.parametrize('scheme, ranks, error, z_values, p_values', CASES)
    def test_z_statistics_match_the_paper(self, scheme, ranks, error, z_values, p_values):
        ranking = ranking_of(scheme, self.FOUR, ranks, error)
        result = control_post_hoc(ranking, control='SaDE')
        assert [h.z_statistic for h in result.hypotheses] == pytest.approx(z_values, abs=1e-4)

    @pytest.mark.parametrize('scheme, ranks, error, z_values, p_values', CASES)
    def test_unadjusted_p_values_match_the_paper(self, scheme, ranks, error,
                                                 z_values, p_values):
        ranking = ranking_of(scheme, self.FOUR, ranks, error)
        result = control_post_hoc(ranking, control='SaDE')
        assert [h.p_value for h in result.hypotheses] == pytest.approx(p_values, abs=1e-4)

    def test_the_family_has_one_hypothesis_per_non_control_algorithm(self):
        result = control_post_hoc(friedman_test(ladder(), ['A', 'B', 'C']), control='A')
        assert len(result.hypotheses) == 2
        assert result.family == 'control'
        assert result.control == 'A'

    def test_control_defaults_to_the_best_ranked_algorithm(self):
        result = control_post_hoc(friedman_test(ladder(), ['A', 'B', 'C']))
        assert result.control == 'A'

    def test_hypotheses_are_ordered_by_p_value(self):
        result = control_post_hoc(friedman_test(ladder(n_algorithms=5), list('ABCDE')))
        p_values = [h.p_value for h in result.hypotheses]
        assert p_values == sorted(p_values)

    def test_every_requested_procedure_is_reported(self):
        result = control_post_hoc(friedman_test(ladder(), ['A', 'B', 'C']))
        for hypothesis in result.hypotheses:
            assert set(hypothesis.adjusted) == set(CONTROL_PROCEDURES)

    def test_adjusted_never_below_unadjusted(self):
        result = control_post_hoc(friedman_test(ladder(n_algorithms=5), list('ABCDE')))
        for hypothesis in result.hypotheses:
            for procedure, value in hypothesis.adjusted.items():
                assert value >= hypothesis.p_value - 1e-12, procedure

    def test_rejected_lists_the_significant_hypotheses(self):
        result = control_post_hoc(friedman_test(ladder(n_instances=30), ['A', 'B', 'C']))
        rejected = result.rejected('holm', 0.05)
        assert rejected == [h.label for h in result.hypotheses
                            if h.adjusted['holm'] < 0.05]

    def test_unknown_control_raises(self):
        with pytest.raises(ValueError, match='is not one of the algorithms'):
            control_post_hoc(friedman_test(ladder(), ['A', 'B', 'C']), control='Nope')


# =============================================================================
# 3. Adjusted p-values, control family
# =============================================================================

class TestControlAdjustedPValues:
    """Every procedure of Derrac et al. Table 16."""

    EXPECTED = {
        'bonferroni': [0.000050, 0.002656, 0.078586, 0.113371, 0.669139, 1.0, 1.0, 1.0],
        'holm': [0.000050, 0.002324, 0.058940, 0.070857, 0.334569, 0.423278, 1.0, 1.0],
        'hochberg': [0.000050, 0.002324, 0.058940, 0.070857, 0.334569, 0.423278,
                     0.660706, 0.660706],
        'hommel': [0.000050, 0.002324, 0.049116, 0.070857, 0.282186, 0.423278,
                   0.660706, 0.660706],
        'holland': [0.000050, 0.002322, 0.057511, 0.068877, 0.294885, 0.366366,
                    0.768259, 0.768259],
        'finner': [0.000050, 0.001327, 0.025981, 0.028142, 0.130431, 0.183552,
                   0.566345, 0.660706],
        'li': [0.000018, 0.000978, 0.028137, 0.040093, 0.197766, 0.293707,
               0.604506, 0.660706],
    }

    @pytest.mark.parametrize('procedure', list(EXPECTED))
    def test_matches_the_published_table(self, procedure):
        assert adjust_p_values(CONTROL_P_VALUES, procedure) == pytest.approx(
            self.EXPECTED[procedure], abs=CONTROL_TOLERANCE)

    def test_every_documented_procedure_is_covered_by_the_table(self):
        assert set(self.EXPECTED) == set(CONTROL_PROCEDURES)

    @pytest.mark.parametrize('procedure', CONTROL_PROCEDURES)
    def test_order_of_the_input_is_preserved(self, procedure):
        forward = adjust_p_values(CONTROL_P_VALUES, procedure)
        backward = adjust_p_values(CONTROL_P_VALUES[::-1], procedure)
        assert forward == pytest.approx(backward[::-1])

    @pytest.mark.parametrize('procedure', CONTROL_PROCEDURES)
    def test_adjusted_never_below_raw(self, procedure):
        adjusted = adjust_p_values(CONTROL_P_VALUES, procedure)
        assert all(a >= r - 1e-12 for a, r in zip(adjusted, CONTROL_P_VALUES))

    @pytest.mark.parametrize('procedure', CONTROL_PROCEDURES)
    def test_monotone_in_the_raw_p_value(self, procedure):
        adjusted = adjust_p_values(CONTROL_P_VALUES, procedure)
        assert adjusted == sorted(adjusted)

    @pytest.mark.parametrize('procedure', CONTROL_PROCEDURES)
    def test_capped_at_one(self, procedure):
        assert all(value <= 1.0 for value in adjust_p_values([0.4, 0.6, 0.9], procedure))

    @pytest.mark.parametrize('procedure', CONTROL_PROCEDURES)
    def test_none_and_nan_are_excluded_from_the_family(self, procedure):
        adjusted = adjust_p_values([0.01, None, 0.02, np.nan], procedure)
        assert adjusted[1] is None and adjusted[3] is None
        assert adjusted[0] is not None and adjusted[2] is not None

    def test_empty_family(self):
        assert adjust_p_values([], 'holm') == []

    def test_holm_matches_the_documented_example(self):
        assert adjust_p_values([0.01, 0.02, 0.03], 'holm') == pytest.approx([0.03, 0.04, 0.04])

    def test_hommel_is_at_least_as_powerful_as_hochberg(self):
        hommel = adjust_p_values(CONTROL_P_VALUES, 'hommel')
        hochberg = adjust_p_values(CONTROL_P_VALUES, 'hochberg')
        assert all(a <= b + 1e-12 for a, b in zip(hommel, hochberg))

    def test_hochberg_is_at_least_as_powerful_as_holm(self):
        hochberg = adjust_p_values(CONTROL_P_VALUES, 'hochberg')
        holm = adjust_p_values(CONTROL_P_VALUES, 'holm')
        assert all(a <= b + 1e-12 for a, b in zip(hochberg, holm))

    def test_holm_is_at_least_as_powerful_as_bonferroni(self):
        holm = adjust_p_values(CONTROL_P_VALUES, 'holm')
        bonferroni = adjust_p_values(CONTROL_P_VALUES, 'bonferroni')
        assert all(a <= b + 1e-12 for a, b in zip(holm, bonferroni))

    def test_unknown_procedure_raises(self):
        with pytest.raises(ValueError, match='Unknown procedure'):
            adjust_p_values([0.01, 0.02], 'bogus')


# =============================================================================
# 4. Adjusted p-values, all-pairwise family
# =============================================================================

class TestAllPairsAdjustedPValues:
    """Nemenyi, Holm, Shaffer and Bergmann-Hommel of Derrac et al. Table 20."""

    EXPECTED = {
        'nemenyi': [0.000224, 0.001624, 0.00387, 0.011952, 0.058772, 0.08328,
                    0.116841, 0.190602, 0.353638, 0.51017] + [1.0] * 26,
        'holm': [0.000224, 0.001579, 0.003655, 0.010956, 0.052242, 0.071713,
                 0.097367, 0.15354, 0.275052, 0.382627, 0.834835, 0.856006] + [1.0] * 24,
        'shaffer': [0.000224, 0.001263, 0.00301, 0.009296, 0.045712, 0.064773,
                    0.090876, 0.148246, 0.275052, 0.311771, 0.706398, 0.753286,
                    0.855076, 0.968322] + [1.0] * 22,
        'bergmann': [0.000224, 0.001263, 0.002365, 0.009296, 0.034284, 0.04164,
                     0.051929, 0.095301, 0.216112, 0.255085, 0.513744, 0.513744,
                     0.621874, 0.621874, 0.63369, 0.686498, 0.756271,
                     0.756271] + [1.0] * 18,
    }

    def adjusted(self, procedure):
        return adjust_p_values(PAIRWISE_P_VALUES, procedure,
                               n_algorithms=9, pairs=PAIRWISE_PAIRS)

    @pytest.mark.parametrize('procedure', list(EXPECTED))
    def test_matches_the_published_table(self, procedure):
        got, expected = self.adjusted(procedure), self.EXPECTED[procedure]
        if procedure == 'holm':
            # Row 13 of Table 20 prints 1.0, but the formula the table itself
            # follows gives 24 * 0.038867 = 0.932808, as row 12 confirms with
            # 25 * 0.03424 = 0.856006
            got, expected = got[:12] + got[13:], expected[:12] + expected[13:]
        assert got == pytest.approx(expected, abs=PAIRWISE_TOLERANCE)

    def test_every_documented_procedure_is_covered_by_the_table(self):
        assert set(self.EXPECTED) == set(ALL_PAIRS_PROCEDURES)

    def test_the_logically_aware_procedures_reject_more(self):
        # Shaffer exploits that the pairwise hypotheses cannot all be false
        # independently, and Bergmann-Hommel goes further still
        holm = self.adjusted('holm')
        shaffer = self.adjusted('shaffer')
        bergmann = self.adjusted('bergmann')
        assert all(s <= h + 1e-12 for s, h in zip(shaffer, holm))
        assert sum(value < 0.05 for value in bergmann) > sum(value < 0.05 for value in holm)

    def test_shaffer_t_values_follow_the_published_table(self):
        # Implied by dividing the Shaffer column of Table 20 by its p-values
        assert shaffer_t_values(9)[:14] == [36] + [28] * 8 + [22] * 5

    def test_shaffer_t_values_of_three_algorithms(self):
        # Of the three pairwise equalities among A, B, C either all three or at
        # most one can hold, so t = 3, 1, 1
        assert shaffer_t_values(3) == [3, 1, 1]

    def test_shaffer_t_values_are_non_increasing(self):
        values = shaffer_t_values(6)
        assert values == sorted(values, reverse=True)
        assert values[0] == 15

    def test_shaffer_needs_the_full_family(self):
        with pytest.raises(ValueError, match='full all-pairwise family'):
            adjust_p_values([0.01, 0.02], 'shaffer', n_algorithms=9)

    def test_shaffer_needs_the_algorithm_count(self):
        with pytest.raises(ValueError, match='needs n_algorithms'):
            adjust_p_values([0.01, 0.02, 0.03], 'shaffer')

    def test_bergmann_needs_the_pairs(self):
        with pytest.raises(ValueError, match='needs n_algorithms and pairs'):
            adjust_p_values([0.01, 0.02, 0.03], 'bergmann', n_algorithms=3)

    def test_exhaustive_sets_of_three_algorithms(self):
        # {AB}, {AC}, {BC} and the whole family are the exhaustive sets
        pairs = [(0, 1), (0, 2), (1, 2)]
        sets = exhaustive_sets(pairs, 3)
        assert sorted(sets) == [[0], [0, 1, 2], [1], [2]]

    def test_exhaustive_sets_are_refused_for_too_many_algorithms(self):
        with pytest.raises(ValueError, match='only tractable up to'):
            exhaustive_sets([(0, 1)], 11)


# =============================================================================
# 5. All-pairs post-hoc wiring
# =============================================================================

class TestAllPairsPostHoc:
    """The NxN family built from a ranking."""

    def test_the_family_has_one_hypothesis_per_pair(self):
        result = all_pairs_post_hoc(friedman_test(ladder(n_algorithms=5), list('ABCDE')))
        assert len(result.hypotheses) == 10
        assert result.family == 'all_pairs'
        assert result.control is None

    def test_hypotheses_are_ordered_by_p_value(self):
        result = all_pairs_post_hoc(friedman_test(ladder(n_algorithms=4), list('ABCD')))
        p_values = [h.p_value for h in result.hypotheses]
        assert p_values == sorted(p_values)

    def test_the_better_algorithm_comes_first_in_a_pair(self):
        ranking = friedman_test(ladder(n_algorithms=4), list('ABCD'))
        for hypothesis in all_pairs_post_hoc(ranking).hypotheses:
            better, worse = hypothesis.algorithms
            assert ranking.average_ranks[better] <= ranking.average_ranks[worse]

    def test_every_requested_procedure_is_reported(self):
        result = all_pairs_post_hoc(friedman_test(ladder(n_algorithms=4), list('ABCD')))
        for hypothesis in result.hypotheses:
            assert set(hypothesis.adjusted) == set(ALL_PAIRS_PROCEDURES)

    def test_a_subset_of_procedures_can_be_requested(self):
        result = all_pairs_post_hoc(friedman_test(ladder(), ['A', 'B', 'C']),
                                    procedures=('holm',))
        assert set(result.hypotheses[0].adjusted) == {'holm'}

    def test_works_from_every_ranking_scheme(self):
        for scheme in RankScheme:
            ranking = omnibus_test(TOY_ERRORS, TOY_NAMES, scheme=scheme)
            result = all_pairs_post_hoc(ranking)
            assert len(result.hypotheses) == 6
            assert result.scheme is scheme


# =============================================================================
# 6. Pairwise tests
# =============================================================================

class TestWilcoxonSignedRank:
    """Demsar Table 2, C4.5 against C4.5 with m tuned."""

    def test_rank_sums_match_the_paper(self):
        result = wilcoxon_signed_rank_test(C45_M, C45,
                                           direction=OptimizationDirection.MAXIMIZE)
        assert result.r_plus == pytest.approx(93.0)
        assert result.r_minus == pytest.approx(12.0)
        assert result.statistic == pytest.approx(12.0)

    def test_null_hypothesis_is_rejected(self):
        # The paper rejects at alpha = 0.05, where T <= 21 is the critical value
        result = wilcoxon_signed_rank_test(C45_M, C45,
                                           direction=OptimizationDirection.MAXIMIZE)
        assert result.p_value < 0.05

    def test_rank_sums_add_up_to_the_total_rank_mass(self):
        result = wilcoxon_signed_rank_test(C45_M, C45,
                                           direction=OptimizationDirection.MAXIMIZE)
        n = result.n_instances
        assert result.r_plus + result.r_minus == pytest.approx(n * (n + 1) / 2)

    def test_swapping_the_samples_swaps_the_rank_sums(self):
        forward = wilcoxon_signed_rank_test(C45_M, C45,
                                            direction=OptimizationDirection.MAXIMIZE)
        backward = wilcoxon_signed_rank_test(C45, C45_M,
                                             direction=OptimizationDirection.MAXIMIZE)
        assert forward.r_plus == pytest.approx(backward.r_minus)
        assert forward.p_value == pytest.approx(backward.p_value)

    def test_direction_flips_which_sample_is_favoured(self):
        maximize = wilcoxon_signed_rank_test(C45_M, C45,
                                             direction=OptimizationDirection.MAXIMIZE)
        minimize = wilcoxon_signed_rank_test(C45_M, C45,
                                             direction=OptimizationDirection.MINIMIZE)
        assert minimize.r_plus == pytest.approx(maximize.r_minus)

    def test_identical_samples_are_not_significant(self):
        result = wilcoxon_signed_rank_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result.p_value == 1.0
        assert result.ties == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match='same problems'):
            wilcoxon_signed_rank_test([1.0, 2.0], [1.0])


class TestSignTest:
    """Counts of wins, losses and ties, with the binomial p-value."""

    def test_wins_match_the_paper(self):
        # C4.5+m wins on 10 data sets and shares one of the two ties
        result = sign_test(C45_M, C45, direction=OptimizationDirection.MAXIMIZE)
        assert result.wins == pytest.approx(11.0)
        assert result.losses == pytest.approx(3.0)
        assert result.ties == 2

    def test_the_papers_critical_value_is_one_sided(self):
        # Both papers tabulate 11 wins out of 14 as significant at alpha = 0.05,
        # which holds for the one-sided p-value but not for the two-sided one
        result = sign_test(C45_M, C45, direction=OptimizationDirection.MAXIMIZE)
        assert result.p_value_one_sided < 0.05
        assert result.p_value > 0.05

    def test_ties_are_split_rather_than_discarded(self):
        result = sign_test([1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 3.0])
        assert result.ties == 3
        # Three ties share one each and the odd one is ignored
        assert result.wins == pytest.approx(2.0)
        assert result.losses == pytest.approx(1.0)

    def test_a_clean_sweep_is_significant(self):
        first = list(range(10))
        second = [value + 1 for value in first]
        assert sign_test(first, second).p_value < 0.01

    def test_an_even_split_is_not_significant(self):
        assert sign_test([1, 2, 1, 2], [2, 1, 2, 1]).p_value == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match='same problems'):
            sign_test([1.0, 2.0], [1.0])


# =============================================================================
# 7. Contrast estimation
# =============================================================================

class TestContrastEstimation:
    """Differences between algorithms on the scale of the measure itself."""

    MATRIX = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [4.0, 5.0, 6.0],
    ])

    def test_estimates_the_pairwise_offsets(self):
        result = contrast_estimation(self.MATRIX, ['A', 'B', 'C'])
        assert result.estimators[0, 1] == pytest.approx(-1.0)
        assert result.estimators[0, 2] == pytest.approx(-3.0)
        assert result.estimators[1, 2] == pytest.approx(-2.0)

    def test_the_estimator_matrix_is_antisymmetric(self):
        result = contrast_estimation(self.MATRIX, ['A', 'B', 'C'])
        assert result.estimators == pytest.approx(-result.estimators.T)
        assert np.diag(result.estimators) == pytest.approx(np.zeros(3))

    def test_estimators_are_differences_of_the_per_algorithm_terms(self):
        result = contrast_estimation(self.MATRIX, ['A', 'B', 'C'])
        performances = result.performances
        assert result.estimators[0, 2] == pytest.approx(
            performances['A'] - performances['C'])

    def test_a_constant_offset_is_recovered_exactly(self):
        rng = np.random.default_rng(4)
        base = rng.normal(size=(1, 12))
        matrix = np.vstack([base, base + 2.5, base - 1.0])
        result = contrast_estimation(matrix, ['A', 'B', 'C'])
        assert result.estimators[1, 0] == pytest.approx(2.5)
        assert result.estimators[2, 0] == pytest.approx(-1.0)

    def test_the_median_shrugs_off_an_outlying_problem(self):
        matrix = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 99.0],
        ])
        result = contrast_estimation(matrix, ['A', 'B'])
        assert result.estimators[0, 1] == pytest.approx(-1.0)


# =============================================================================
# 8. The report DataAnalyzer builds from all of the above
# =============================================================================

class TestStatisticalReport:
    """Sections of ``DataAnalyzer.generate_statistical_report``."""

    NAMES = ['A', 'B', 'C', 'D']

    @pytest.fixture
    def analyzer(self, tmp_path):
        from ddmtolab.Methods.data_analysis import DataAnalyzer
        return DataAnalyzer(data_path=tmp_path / 'Data', save_path=tmp_path / 'Results',
                            multi_problem_report=True, clear_results=False)

    @pytest.fixture
    def sections(self, analyzer):
        from ddmtolab.Methods.statistical_tests import (
            all_pairs_post_hoc, contrast_estimation as estimate, control_post_hoc,
            omnibus_test,
        )
        matrix = ladder(n_instances=8, n_algorithms=4)
        rankings = {scheme: omnibus_test(matrix, self.NAMES, scheme=scheme)
                    for scheme in RankScheme}
        return analyzer._build_report_tables(
            matrix, self.NAMES, OptimizationDirection.MINIMIZE, rankings,
            control_post_hoc(rankings[RankScheme.FRIEDMAN]),
            all_pairs_post_hoc(rankings[RankScheme.FRIEDMAN]),
            estimate(matrix, self.NAMES)
        )

    def test_every_section_is_present(self, sections):
        assert list(sections) == ['Rankings', 'Control (A)', 'All pairs',
                                  'Contrast estimation', 'Pairwise tests']

    def test_the_rankings_section_covers_all_three_schemes(self, sections):
        table = sections['Rankings']
        assert set(scheme.value for scheme in RankScheme).issubset(table.columns)
        labels = list(table['Algorithm'])
        assert labels[:4] == self.NAMES
        assert labels[4:] == ['Statistic', 'p-value',
                              'Iman-Davenport F_F', 'Iman-Davenport p-value']

    def test_the_control_section_reports_every_procedure(self, sections):
        table = sections['Control (A)']
        assert len(table) == 3
        assert set(CONTROL_PROCEDURES).issubset(table.columns)

    def test_the_all_pairs_section_reports_every_procedure(self, sections):
        table = sections['All pairs']
        assert len(table) == 6
        assert set(ALL_PAIRS_PROCEDURES).issubset(table.columns)

    def test_the_contrast_section_is_a_square_matrix(self, sections):
        table = sections['Contrast estimation']
        assert list(table['Algorithm']) == self.NAMES
        assert set(self.NAMES).issubset(table.columns)

    def test_the_pairwise_section_covers_both_plain_tests(self, sections):
        table = sections['Pairwise tests']
        assert len(table) == 6
        assert {'Wins', 'Losses', 'Ties', 'Sign p', 'R+', 'R-',
                'Wilcoxon p'}.issubset(table.columns)

    def test_excel_report_has_one_sheet_per_section(self, analyzer, sections, tmp_path):
        import pandas as pd
        output = analyzer._write_statistical_report(sections)
        assert output.name == 'statistical_report.xlsx'
        assert set(pd.read_excel(output, sheet_name=None)) == set(sections)

    def test_latex_report_is_written_as_one_file(self, analyzer, sections):
        from ddmtolab.Methods.data_analysis import TableFormat
        analyzer.table_config.table_format = TableFormat.LATEX
        output = analyzer._write_statistical_report(sections)
        assert output.name == 'statistical_report.tex'
        content = output.read_text(encoding='utf-8')
        assert content.count('tabular') == 2 * len(sections)
        for name in sections:
            assert f'% {name}' in content

    def test_the_report_scheme_is_parsed_into_an_enum(self, tmp_path):
        from ddmtolab.Methods.data_analysis import DataAnalyzer
        analyzer = DataAnalyzer(data_path=tmp_path, report_scheme='quade')
        assert analyzer.report_scheme is RankScheme.QUADE

    def test_an_unknown_report_scheme_raises(self, tmp_path):
        from ddmtolab.Methods.data_analysis import DataAnalyzer
        with pytest.raises(ValueError):
            DataAnalyzer(data_path=tmp_path, report_scheme='kruskal')

    def test_the_report_is_off_by_default(self, tmp_path):
        from ddmtolab.Methods.data_analysis import DataAnalyzer
        assert DataAnalyzer(data_path=tmp_path).multi_problem_report is False

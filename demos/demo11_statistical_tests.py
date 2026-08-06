"""
Demo 11: Statistical Comparison of Algorithms

This demo walks through every statistical method the platform offers, on a real
batch experiment. It follows the methodology of the two standard references:

    [D06] J. Demsar. Statistical Comparisons of Classifiers over Multiple Data
          Sets. JMLR 7 (2006) 1-30.
    [D11] J. Derrac, S. Garcia, D. Molina, F. Herrera. A practical tutorial on
          the use of nonparametric statistical tests as a methodology for
          comparing evolutionary and swarm intelligence algorithms. Swarm and
          Evolutionary Computation 1 (2011) 3-18.

The single most important idea is that there are two different questions, and
they need different tests:

    per-instance   "On THIS problem, do the runs of A differ from the runs of
                    B?" -- many samples per problem, answered by a rank-sum
                    test, reported as the +/-/= column of the results table.

    multi-problem  "Across the WHOLE suite, is A better than B?" -- one value
                    per algorithm-problem pair, answered by the tests in
                    ddmtolab.Methods.statistical_tests.

Reporting a +/-/= tally as if it answered the second question is the most
common mistake in the field. Part 3 and Part 4 below keep them apart.

Key concepts:
- Per-instance comparison: rank-sum test, Holm correction, Cliff's delta
- Two algorithms over the suite: sign test, Wilcoxon signed-rank test
- Many algorithms: Friedman (+ Iman-Davenport), Friedman aligned ranks, Quade
- Post-hoc against a control: seven procedures, reported as adjusted p-values
- Post-hoc among all pairs: Nemenyi, Holm, Shaffer, Bergmann-Hommel
- Critical difference diagram and contrast estimation
- Running all of it with two switches on DataAnalyzer
"""

import numpy as np
import pandas as pd

from ddmtolab.Algorithms.STSO.CMA_ES import CMA_ES
from ddmtolab.Algorithms.STSO.DE import DE
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.STSO.PSO import PSO
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import (
    DataAnalyzer, DataUtils, StatisticsCalculator, StatisticType,
)
from ddmtolab.Methods.statistical_tests import (
    ALL_PAIRS_PROCEDURES, CONTROL_PROCEDURES, OptimizationDirection, RankScheme,
    adjust_p_values, all_pairs_post_hoc, cliffs_delta, contrast_estimation,
    control_post_hoc, friedman_aligned_test, friedman_test, nemenyi_test,
    omnibus_test, quade_test, rank_sum_test, sign_test, wilcoxon_signed_rank_test,
)
from ddmtolab.Problems.STSO.classical_so import CLASSICALSO

ALGORITHMS = ['GA', 'DE', 'PSO', 'CMA-ES']
CONTROL = 'CMA-ES'          # the algorithm the study is about
BASELINE = 'CMA-ES'         # the last entry of algorithm_order, used by the table
ALPHA = 0.05

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)


def banner(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


if __name__ == '__main__':
    # =========================================================================
    # Part 1: Run the experiment
    # =========================================================================
    # Rule of thumb from [D11]: use at least twice as many problems as
    # algorithms (n >= 2k). With four algorithms that means eight problems or
    # more, so the nine classical functions are enough. Below that the omnibus
    # tests have almost no power; far above n = 8k they start flagging
    # differences too small to matter.

    problems = CLASSICALSO()

    batch = BatchExperiment(base_path='./Data', clear_folder=True)
    for index in range(1, 10):
        batch.add_problem(getattr(problems, f'P{index}'), f'P{index}', D=30)

    for algorithm, name in ((GA, 'GA'), (DE, 'DE'), (PSO, 'PSO'), (CMA_ES, 'CMA-ES')):
        batch.add_algorithm(algorithm, name, n=50, max_nfes=5000)

    # n_runs is the sample size of the PER-INSTANCE tests; 20 or more is usual
    batch.run(n_runs=10, max_workers=8, base_seed=2026)

    # =========================================================================
    # Part 2: The whole analysis with two switches
    # =========================================================================
    # Everything the rest of this demo does by hand is also available from
    # DataAnalyzer. Each switch is off by default.

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=None,                  # None for single-objective; see below for MO
        algorithm_order=ALGORITHMS,     # the LAST entry is the table's baseline
        save_path='./Results',
        statistic_type='median_iqr',    # 'mean' -> mean (std), 'median_iqr' -> median[IQR]
        significance_level=ALPHA,

        # --- per-instance layer, the +/-/= column of the table ---
        rank_sum_test=True,             # on by default
        holm_correction=True,           # correct over every comparison in the table
        effect_size=True,               # Cliff's delta next to each symbol

        # --- multi-problem layer ---
        friedman_test=True,             # omnibus + post-hoc rows in the table footer
        friedman_control=CONTROL,
        cd_diagram=True,                # critical difference diagram, [D06] Figure 1(a)
        cd_alpha=0.10,                  # the alpha Demsar's own diagrams use
        multi_problem_report=True,      # the complete [D11] analysis as one workbook
        report_scheme='friedman',       # or 'aligned' / 'quade'
        report_control=CONTROL,

        figure_format='png',
        merge_plots=True,
    )
    results = analyzer.run()      # MetricResults: per-run values, runtimes, metric name

    # For MULTI-OBJECTIVE experiments the only change is a settings dictionary
    # naming the metric and where the reference fronts come from:
    #
    #     from ddmtolab.Problems.STMO.ZDT import SETTINGS
    #     settings = {**SETTINGS, 'metric': 'IGD'}   # or HV, IGDp, GD, DeltaP,
    #                                                # Spacing, Spread, FR, CV
    #     DataAnalyzer(data_path='./Data', settings=settings, ...)
    #
    # Everything below works the same way, on whichever metric was chosen.

    # =========================================================================
    # Part 3: The per-instance layer
    # =========================================================================
    # One problem at a time, comparing the runs of two algorithms. This is what
    # fills the results table. It says nothing about the suite as a whole.

    banner('Part 3: one instance at a time (runs of A vs runs of B)')

    # results.best_values[algorithm][problem][run] holds one value per task
    runs_of = {name: StatisticsCalculator.collect_task_data(
        results.best_values, name, 'P4', 0) for name in ALGORITHMS}

    print('Problem P4 (Rastrigin), final objective value of each run:')
    for name in ALGORITHMS:
        sample = np.array(runs_of[name])
        print(f'  {name:7s} median = {np.median(sample):.4e}   n = {sample.size}')

    print(f'\nWilcoxon rank-sum test against {BASELINE}, with the effect size:')
    for name in ALGORITHMS:
        if name == BASELINE:
            continue
        test = rank_sum_test(runs_of[name], runs_of[BASELINE])
        effect = cliffs_delta(runs_of[name], runs_of[BASELINE])
        print(f'  {name:7s} p = {test.p_value:.4e}   '
              f'delta = {effect.delta:+.2f} ({effect.magnitude})')

    print('\nThe table renders exactly this, as a symbol plus a bracketed delta.')
    print('Significance says a difference is unlikely to be noise; the effect')
    print('size says how large it is. They answer different questions, so the')
    print('table keeps them in separate fields.')

    # =========================================================================
    # Part 4: The multi-problem layer
    # =========================================================================
    # Collapse the runs of every instance into one number, and compare the
    # algorithms across the suite. build_instance_matrix uses the same
    # statistic the table displays, so the two never disagree.

    matrix, labels = StatisticsCalculator.build_instance_matrix(
        results.best_values, ALGORITHMS, StatisticType.MEDIAN)
    sense = DataUtils.get_metric_direction(results.metric_name)

    banner('Part 4: the algorithm-by-instance matrix everything below consumes')
    print(pd.DataFrame(matrix, index=ALGORITHMS, columns=labels).T.to_string(
        float_format=lambda value: f'{value:.4e}'))
    print(f'\nshape = {matrix.shape}  (k = {matrix.shape[0]} algorithms, '
          f'N = {matrix.shape[1]} problems), direction = {sense.value}')

    # ------------------------------------------------------------------ 4a ---
    # Two algorithms over the whole suite.

    banner('Part 4a: two algorithms over the suite')

    first, second = ALGORITHMS.index(CONTROL), ALGORITHMS.index('PSO')
    signs = sign_test(matrix[first], matrix[second], sense)
    wilcoxon = wilcoxon_signed_rank_test(matrix[first], matrix[second], sense)

    print(f'{CONTROL} vs PSO')
    print(f'  sign test      wins {signs.wins:.1f} / losses {signs.losses:.1f} / '
          f'ties {signs.ties}   p = {signs.p_value:.4f}')
    print(f'  Wilcoxon       R+ = {wilcoxon.r_plus:.1f}, R- = {wilcoxon.r_minus:.1f}, '
          f'T = {wilcoxon.statistic:.1f}   p = {wilcoxon.p_value:.4f}')
    print('\nThe sign test only counts wins, so it is immune to the scale of the')
    print('objective but very weak. Wilcoxon weighs how large each difference')
    print('is and is the recommended pairwise test -- but because it ranks the')
    print('MAGNITUDES, the values must be commensurable across problems. With')
    print('raw objectives on wildly different scales, normalize first or use a')
    print('scale-free metric such as IGD or HV.')

    # ------------------------------------------------------------------ 4b ---
    # Do the k algorithms differ at all? Three ranking schemes are available.

    banner('Part 4b: omnibus tests, do the algorithms differ at all')

    rankings = {}
    rows = []
    for scheme in RankScheme:
        ranking = omnibus_test(matrix, ALGORITHMS, sense, scheme)
        rankings[scheme] = ranking
        rows.append({'scheme': scheme.value,
                     **{name: round(rank, 3) for name, rank in ranking.average_ranks.items()},
                     'statistic': f'{ranking.statistic:.4f}',
                     'p-value': f'{ranking.p_value:.4e}'})
    print(pd.DataFrame(rows).to_string(index=False))

    friedman = rankings[RankScheme.FRIEDMAN]
    print(f'\nIman-Davenport F_F = {friedman.iman_davenport_statistic:.4f}, '
          f'p = {friedman.iman_davenport_p_value:.4e}')
    print('\n  friedman  ranks within each problem only')
    print('  aligned   subtracts each problem\'s average first, so observations')
    print('            from different problems become comparable; more powerful')
    print('            when few algorithms are compared')
    print('  quade     weighs each problem by how widely the algorithms spread')
    print('            out on it, so easy problems count less')
    print('\nchi^2_F is known to be conservative, which is why Iman-Davenport is')
    print('reported next to it and is the statistic to prefer. Check this p-value')
    print('BEFORE reading any post-hoc row below.')

    # omnibus_test(..., scheme) is a shorthand; each test is also callable on
    # its own, with the same arguments and the same RankingResult back:
    print()
    for test in (friedman_test, friedman_aligned_test, quade_test):
        ranking = test(matrix, ALGORITHMS, sense)
        print(f'  {test.__name__:22s} -> {ranking.statistic_name} = '
              f'{ranking.statistic:.4f}, p = {ranking.p_value:.4e}')

    # ------------------------------------------------------------------ 4c ---
    # Which algorithms differ from MY algorithm? k-1 hypotheses.

    banner(f'Part 4c: post-hoc against the control {CONTROL} (1 x N, k-1 hypotheses)')

    family = control_post_hoc(friedman, CONTROL)          # all seven procedures
    print(pd.DataFrame([
        {'hypothesis': h.label, 'z': round(h.z_statistic, 4),
         'unadjusted': f'{h.p_value:.4e}',
         **{procedure: f'{value:.4e}' for procedure, value in h.adjusted.items()}}
        for h in family.hypotheses
    ]).to_string(index=False))

    print(f'\nRejected at alpha = {ALPHA}:')
    for procedure in CONTROL_PROCEDURES:
        print(f'  {procedure:11s} {family.rejected(procedure, ALPHA)}')

    print('\nAn adjusted p-value (APV) is the smallest alpha at which that')
    print('hypothesis would still be rejected given the whole family, so it can')
    print('be compared directly against any alpha. Power increases roughly')
    print('bonferroni < holm < hochberg < hommel, with holland and finner as')
    print('sharper step-down variants; holm is the safe default because it makes')
    print('no extra assumptions, finner is noticeably more powerful.')

    # ------------------------------------------------------------------ 4d ---
    # Which algorithms differ from each other? k(k-1)/2 hypotheses.

    banner('Part 4d: post-hoc among all pairs (N x N, k(k-1)/2 hypotheses)')

    pairs = all_pairs_post_hoc(friedman)
    print(pd.DataFrame([
        {'hypothesis': h.label, 'rank diff': round(h.rank_difference, 4),
         'unadjusted': f'{h.p_value:.4e}',
         **{procedure: f'{value:.4e}' for procedure, value in h.adjusted.items()}}
        for h in pairs.hypotheses
    ]).to_string(index=False))

    print(f'\nRejected at alpha = {ALPHA}:')
    for procedure in ALL_PAIRS_PROCEDURES:
        rejected = pairs.rejected(procedure, ALPHA)
        print(f'  {procedure:9s} {len(rejected)}/{len(pairs.hypotheses)}  {rejected}')

    print('\nThis family is larger, so every hypothesis is harder to reject: use')
    print('4c when the study is about one proposed algorithm, and only use 4d')
    print('when every algorithm really is compared against every other. Note how')
    print('the same difference that 4c called significant may not survive here.')
    print('\nShaffer and Bergmann-Hommel exploit that pairwise equalities cannot')
    print('all be false independently -- if A differs from B they cannot both')
    print('equal C -- so their columns above are never larger than Holm\'s and')
    print('usually smaller, which is what buys the extra power.')

    # ------------------------------------------------------------------ 4e ---
    # The same all-pairs question, drawn instead of tabulated.

    banner('Part 4e: critical difference diagram')

    nemenyi = nemenyi_test(matrix, ALGORITHMS, sense, significance_level=0.10)
    print(f'CD = {nemenyi.critical_difference:.4f} at alpha = '
          f'{nemenyi.significance_level} '
          f'(q = {nemenyi.q_alpha:.4f}, k = {nemenyi.n_algorithms}, '
          f'N = {nemenyi.n_instances})')
    print('Two algorithms differ significantly when their average ranks differ')
    print('by more than CD. Groups that do not are connected in the diagram:')
    for clique in nemenyi.cliques:
        print(f'  {clique}')
    print('\nSaved by the pipeline as ./Results/cd_diagram.png (see cd_diagram=True).')

    # ------------------------------------------------------------------ 4f ---
    # How large are the differences, on the scale of the measure itself?

    banner('Part 4f: contrast estimation, the size of the differences')

    contrast = contrast_estimation(matrix, ALGORITHMS)
    print(pd.DataFrame(contrast.estimators, index=ALGORITHMS,
                       columns=ALGORITHMS).to_string(
        float_format=lambda value: f'{value:+.4e}'))
    print('\nEntry [u, v] estimates the performance difference of u minus v in')
    print('the units of the measure itself, from the median of their per-problem')
    print('differences. No p-value comes with it: this answers "by how much",')
    print('not "is it real". It assumes the values are commensurable.')

    # =========================================================================
    # Part 5: The building blocks on their own
    # =========================================================================
    # Every helper works on plain numbers, independently of any experiment.

    banner('Part 5: standalone helpers')

    raw = [0.001, 0.012, 0.031, 0.049]
    print(f'raw p-values of one family: {raw}')
    for procedure in ('bonferroni', 'holm', 'finner', 'hommel'):
        adjusted = adjust_p_values(raw, procedure)
        print(f'  {procedure:11s} {[round(value, 4) for value in adjusted]}   '
              f'rejects {sum(value < ALPHA for value in adjusted)}/4')

    print('\nAll-pairs procedures additionally need to know the algorithms each')
    print('hypothesis compares, so they can reason about the logical structure:')
    print("  adjust_p_values(p, 'shaffer', n_algorithms=4)")
    print("  adjust_p_values(p, 'bergmann', n_algorithms=4, pairs=[(0, 1), ...])")

    print('\nEffect size of two samples, oriented so that positive means better:')
    better, worse = [1.0, 1.1, 0.9, 1.2], [2.0, 2.1, 1.9, 2.2]
    effect = cliffs_delta(better, worse, OptimizationDirection.MINIMIZE)
    print(f'  cliffs_delta -> {effect.delta:+.2f} ({effect.magnitude})')

    # =========================================================================
    # What was written
    # =========================================================================

    banner('Files produced')
    print('./Results/results_table_median_iqr.xlsx   per-instance table, +/-/= and deltas')
    print('./Results/statistical_report.xlsx         the whole [D11] analysis, 5 sheets:')
    print('    Rankings              average ranks under all three schemes')
    print('    Control (<name>)      the k-1 comparisons, seven APVs each')
    print('    All pairs             the k(k-1)/2 comparisons, four APVs each')
    print('    Contrast estimation   how large the differences are')
    print('    Pairwise tests        sign test and Wilcoxon per pair')
    print('./Results/cd_diagram.png                  critical difference diagram')
    print('./Results/convergence_merged.png          convergence curves')
    print('./Results/runtime_comparison.png          runtime bar chart')

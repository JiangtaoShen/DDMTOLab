"""
Nonparametric Statistical Tests for Comparing Algorithms over Multiple Problems

This module implements the methodology recommended by the two standard
references on comparing (evolutionary) algorithms across a benchmark suite:

- J. Demsar, *Statistical Comparisons of Classifiers over Multiple Data Sets*,
  JMLR 7 (2006) 1-30.
- J. Derrac, S. Garcia, D. Molina, F. Herrera, *A practical tutorial on the use
  of nonparametric statistical tests as a methodology for comparing evolutionary
  and swarm intelligence algorithms*, Swarm and Evolutionary Computation 1
  (2011) 3-18.

Every test here consumes one value per algorithm-problem pair -- typically the
mean or median over the independent runs of that pair -- arranged as a matrix of
shape ``(n_algorithms, n_instances)``. This is the *multi-problem* setting: it
answers "is algorithm A better than B across the suite", which is a different
question from the per-problem rank-sum test over runs in
:mod:`ddmtolab.Methods.data_analysis`.

Which test to use
-----------------
- Two algorithms: :func:`sign_test` (quick, weak) or
  :func:`wilcoxon_signed_rank_test` (recommended).
- Many algorithms, omnibus: :func:`friedman_test` (plus its Iman-Davenport
  correction), :func:`friedman_aligned_test` or :func:`quade_test`.
- Many algorithms against one control: :func:`control_post_hoc`, which reports
  an adjusted p-value per comparison under any of seven procedures.
- Every algorithm against every other: :func:`all_pairs_post_hoc`, with the
  Nemenyi, Holm, Shaffer and Bergmann-Hommel procedures.
- How large the difference is: :func:`contrast_estimation`.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.05
Version: 1.0
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from math import comb, factorial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

__all__ = [
    # Enums and constants
    'OptimizationDirection',
    'RankScheme',
    'CONTROL_PROCEDURES',
    'ALL_PAIRS_PROCEDURES',
    'CLIFFS_DELTA_THRESHOLDS',
    'BERGMANN_MAX_ALGORITHMS',
    # Results
    'PairwiseTestResult',
    'EffectSizeResult',
    'RankingResult',
    'Hypothesis',
    'PostHocResult',
    'NemenyiComparison',
    'NemenyiResult',
    'ContrastResult',
    # Two algorithms
    'rank_sum_test',
    'sign_test',
    'wilcoxon_signed_rank_test',
    'cliffs_delta',
    'classify_cliffs_delta',
    # Many algorithms, omnibus
    'friedman_test',
    'friedman_aligned_test',
    'quade_test',
    'omnibus_test',
    # Many algorithms, post-hoc
    'control_post_hoc',
    'all_pairs_post_hoc',
    'nemenyi_test',
    'nemenyi_critical_value',
    'adjust_p_values',
    'shaffer_t_values',
    'exhaustive_sets',
    # Magnitude of the differences
    'contrast_estimation',
]


# =============================================================================
# Enums and constants
# =============================================================================

class OptimizationDirection(Enum):
    """Optimization direction enumeration."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class RankScheme(Enum):
    """
    Ranking scheme underlying an omnibus test.

    FRIEDMAN
        Ranks within each problem, 1 for the best. Only intra-problem
        comparisons are meaningful.
    ALIGNED
        Each value is first aligned by subtracting the average performance on
        its problem; all k*n aligned observations are then ranked together, so
        inter-problem comparisons become meaningful too.
    QUADE
        Friedman ranks weighted by how discriminating each problem is, measured
        by the range of the values observed on it.
    """
    FRIEDMAN = "friedman"
    ALIGNED = "aligned"
    QUADE = "quade"


#: Post-hoc procedures for a family of k-1 comparisons against one control.
#: Ordered from least to most powerful within each class.
CONTROL_PROCEDURES: Tuple[str, ...] = (
    'bonferroni', 'holm', 'holland', 'finner', 'hochberg', 'hommel', 'li'
)

#: Post-hoc procedures for the family of all k(k-1)/2 pairwise comparisons.
ALL_PAIRS_PROCEDURES: Tuple[str, ...] = ('nemenyi', 'holm', 'shaffer', 'bergmann')

#: Largest k for which the Bergmann-Hommel procedure is enumerated. The number
#: of exhaustive sets is the Bell number of k, which explodes beyond this.
BERGMANN_MAX_ALGORITHMS: int = 10

#: Magnitude thresholds for |Cliff's delta| (Romano et al., 2006). A delta whose
#: absolute value falls below the first threshold is negligible, below the second
#: small, below the third medium, and large otherwise.
CLIFFS_DELTA_THRESHOLDS: Tuple[float, float, float] = (0.147, 0.33, 0.474)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PairwiseTestResult:
    """
    Outcome of a test comparing exactly two algorithms over many problems.

    :no-index:

    Attributes
    ----------
    name : str
        Name of the test.
    statistic : float
        Test statistic: the number of wins for the sign test, T = min(R+, R-)
        for the Wilcoxon signed-rank test.
    p_value : float
        Two-tailed p-value.
    p_value_one_sided : float
        One-sided p-value for "the first algorithm is better". The critical
        value tables printed in both papers (Demsar's Table 3, Derrac's Table 4)
        correspond to this one, not to ``p_value``.
    n_instances : int
        Number of problems compared, after dropping incomparable ones.
    wins : float
        Problems on which the first algorithm was better. Ties are split evenly
        between the two algorithms, so this can be a half-integer.
    losses : float
        Problems on which the first algorithm was worse, ties split evenly.
    ties : int
        Number of tied problems, before splitting.
    r_plus : float
        Sum of ranks favouring the first algorithm (Wilcoxon only).
    r_minus : float
        Sum of ranks favouring the second algorithm (Wilcoxon only).
    """
    name: str
    statistic: float
    p_value: float
    n_instances: int
    p_value_one_sided: float = np.nan
    wins: float = 0.0
    losses: float = 0.0
    ties: int = 0
    r_plus: float = np.nan
    r_minus: float = np.nan


@dataclass
class EffectSizeResult:
    """
    Effect size of a pairwise comparison, reported separately from significance.

    :no-index:

    Attributes
    ----------
    delta : float
        Cliff's delta in [-1, 1], oriented so that a positive value means the
        algorithm is better than the baseline under the given optimization
        direction. np.nan when it cannot be computed.
    magnitude : str
        Qualitative magnitude derived from ``|delta|``: 'negligible', 'small',
        'medium', 'large', or 'undefined' when delta is np.nan.
    method : str
        Name of the effect size measure, always 'cliffs_delta'.
    """
    delta: float
    magnitude: str
    method: str = 'cliffs_delta'


@dataclass
class RankingResult:
    """
    Ranks and omnibus statistic produced by one of the three ranking schemes.

    :no-index:

    Attributes
    ----------
    scheme : RankScheme
        Ranking scheme used.
    average_ranks : Dict[str, float]
        Average rank per algorithm, keyed by name. Lower is better for every
        scheme; the best possible value is 1 for FRIEDMAN and QUADE, while
        ALIGNED ranks live on the 1..k*n scale.
    ranks : np.ndarray
        Per-instance ranks of shape (n_algorithms, n_instances).
    statistic : float
        Omnibus statistic: chi^2_F for FRIEDMAN, F_AR for ALIGNED, F_Q for QUADE.
    p_value : float
        P-value of ``statistic``.
    statistic_name : str
        Name of the statistic, for reporting.
    standard_error : float
        Standard error of a difference of two average ranks under this scheme,
        used by the post-hoc z statistics.
    n_algorithms : int
        Number of algorithms (k).
    n_instances : int
        Number of instances retained (N).
    n_instances_dropped : int
        Instances dropped because at least one algorithm had no value there.
    iman_davenport_statistic : float
        Iman-Davenport F_F correction of chi^2_F. FRIEDMAN scheme only.
    iman_davenport_p_value : float
        P-value of F_F under F(k-1, (k-1)(N-1)). FRIEDMAN scheme only.
    """
    scheme: RankScheme
    average_ranks: Dict[str, float]
    ranks: np.ndarray
    statistic: float
    p_value: float
    statistic_name: str
    standard_error: float
    n_algorithms: int
    n_instances: int
    n_instances_dropped: int = 0
    iman_davenport_statistic: float = np.nan
    iman_davenport_p_value: float = np.nan


@dataclass
class Hypothesis:
    """
    One comparison of a post-hoc family.

    :no-index:

    Attributes
    ----------
    label : str
        Human-readable description, e.g. ``'PSO vs DE-Exp'``.
    algorithms : Tuple[str, str]
        The two algorithms compared, better-ranked one first.
    rank_difference : float
        Difference of their average ranks.
    z_statistic : float
        Standardized rank difference.
    p_value : float
        Unadjusted two-tailed p-value.
    adjusted : Dict[str, float]
        Adjusted p-value (APV) per post-hoc procedure, keyed by procedure name.
        An APV can be compared directly against any significance level.
    """
    label: str
    algorithms: Tuple[str, str]
    rank_difference: float
    z_statistic: float
    p_value: float
    adjusted: Dict[str, float] = field(default_factory=dict)


@dataclass
class PostHocResult:
    """
    A family of post-hoc comparisons with their adjusted p-values.

    :no-index:

    Attributes
    ----------
    family : str
        ``'control'`` for the k-1 comparisons against a control algorithm,
        ``'all_pairs'`` for the k(k-1)/2 comparisons among all algorithms.
    scheme : RankScheme
        Ranking scheme the comparisons were derived from.
    control : Optional[str]
        Control algorithm, for the ``'control'`` family only.
    hypotheses : List[Hypothesis]
        Comparisons ordered by increasing unadjusted p-value.
    procedures : Tuple[str, ...]
        Post-hoc procedures applied, in the order requested.
    """
    family: str
    scheme: RankScheme
    control: Optional[str]
    hypotheses: List[Hypothesis]
    procedures: Tuple[str, ...]

    def rejected(self, procedure: str, significance_level: float = 0.05) -> List[str]:
        """
        Labels of the hypotheses a procedure rejects at a significance level.

        Parameters
        ----------
        procedure : str
            Name of a procedure present in ``procedures``.
        significance_level : float, optional
            Threshold applied to the adjusted p-values (default: 0.05).

        Returns
        -------
        List[str]
            Labels of the rejected hypotheses, most significant first.
        """
        return [h.label for h in self.hypotheses
                if h.adjusted.get(procedure, 1.0) < significance_level]


@dataclass
class NemenyiComparison:
    """
    One all-pairs comparison of the Nemenyi post-hoc test.

    :no-index:

    Attributes
    ----------
    algorithm_a : str
        First algorithm of the pair, the better-ranked one.
    algorithm_b : str
        Second algorithm of the pair, the worse-ranked one.
    rank_difference : float
        Absolute difference of the two average ranks.
    q_statistic : float
        Studentized range statistic of the pair.
    p_value : float
        Two-tailed p-value from the studentized range distribution, already
        accounting for all k(k-1)/2 comparisons.
    significant : bool
        Whether the pair differs significantly, i.e. whether the rank difference
        exceeds the critical difference.
    """
    algorithm_a: str
    algorithm_b: str
    rank_difference: float
    q_statistic: float
    p_value: float
    significant: bool


@dataclass
class NemenyiResult:
    """
    Outcome of a Nemenyi all-pairs post-hoc test, in critical-difference form.

    This is the shape the critical difference diagram consumes. For adjusted
    p-values of the same family, see :func:`all_pairs_post_hoc`.

    :no-index:

    Attributes
    ----------
    critical_difference : float
        Critical difference CD; two algorithms differ significantly when their
        average ranks differ by at least this much.
    q_alpha : float
        Critical value of the studentized range statistic divided by sqrt(2),
        as tabulated by Demsar (2006, Table 5a).
    average_ranks : Dict[str, float]
        Average rank per algorithm (1 is best), keyed by algorithm name.
    n_algorithms : int
        Number of algorithms compared (k).
    n_instances : int
        Number of instances retained after dropping incomplete ones (N).
    n_instances_dropped : int
        Number of instances dropped because at least one algorithm had no value.
    significance_level : float
        Significance level the critical difference was computed for.
    comparisons : List[NemenyiComparison]
        One entry per unordered pair, ordered by increasing rank difference.
    cliques : List[List[str]]
        Groups of algorithms that are not significantly different from each
        other, each sorted by average rank and each with at least two members.
        An algorithm in no clique differs significantly from every other one.
        These are the groups the critical difference diagram connects.
    """
    critical_difference: float
    q_alpha: float
    average_ranks: Dict[str, float]
    n_algorithms: int
    n_instances: int
    n_instances_dropped: int = 0
    significance_level: float = 0.05
    comparisons: List[NemenyiComparison] = field(default_factory=list)
    cliques: List[List[str]] = field(default_factory=list)


@dataclass
class ContrastResult:
    """
    Estimated differences between algorithm performances, based on medians.

    :no-index:

    Attributes
    ----------
    algorithms : List[str]
        Algorithm names, in the order of the matrix rows.
    estimators : np.ndarray
        Matrix of shape (k, k) whose entry [u, v] estimates the performance
        difference ``M_u - M_v``. It is antisymmetric with a zero diagonal.
    performances : Dict[str, float]
        The per-algorithm term ``m_u`` the estimators are built from; the
        estimator of ``M_u - M_v`` is ``m_u - m_v``.
    """
    algorithms: List[str]
    estimators: np.ndarray
    performances: Dict[str, float]


# =============================================================================
# Input validation
# =============================================================================

def _prepare_matrix(
        data_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        algorithm_names: Sequence[str],
        test_name: str,
        minimum_algorithms: int = 2
) -> Tuple[np.ndarray, int]:
    """
    Validate an algorithm-by-instance matrix and drop incomplete instances.

    Parameters
    ----------
    data_matrix : Union[np.ndarray, Sequence[Sequence[float]]]
        Results of shape (n_algorithms, n_instances).
    algorithm_names : Sequence[str]
        One name per row.
    test_name : str
        Name of the calling test, used in the messages.
    minimum_algorithms : int, optional
        Smallest number of algorithms the test accepts (default: 2).

    Returns
    -------
    Tuple[np.ndarray, int]
        The matrix restricted to complete instances, and the number of
        instances dropped.

    Raises
    ------
    ValueError
        If the shape is wrong, the names do not match the rows, there are too
        few algorithms, or fewer than 2 complete instances remain.
    """
    matrix = np.asarray(data_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            f"data_matrix must be 2-D with shape (n_algorithms, n_instances), "
            f"got shape {matrix.shape}"
        )
    if matrix.shape[0] != len(algorithm_names):
        raise ValueError(
            f"algorithm_names has {len(algorithm_names)} entries but data_matrix "
            f"has {matrix.shape[0]} rows; they must match"
        )
    if matrix.shape[0] < minimum_algorithms:
        raise ValueError(
            f"The {test_name} needs at least {minimum_algorithms} algorithms, "
            f"got {matrix.shape[0]}."
        )

    complete = ~np.isnan(matrix).any(axis=0)
    n_dropped = int(np.sum(~complete))
    if n_dropped:
        warnings.warn(
            f"Dropped {n_dropped} instance(s) from the {test_name} because at "
            f"least one algorithm had no value there."
        )
    matrix = matrix[:, complete]

    if matrix.shape[1] < 2:
        raise ValueError(
            f"The {test_name} needs at least 2 complete instances, got "
            f"{matrix.shape[1]}."
        )

    return matrix, n_dropped


def _oriented(matrix: np.ndarray, direction: OptimizationDirection) -> np.ndarray:
    """Return the matrix oriented so that smaller is always better."""
    return matrix if direction == OptimizationDirection.MINIMIZE else -matrix


# =============================================================================
# Adjusted p-values (APVs)
# =============================================================================

def shaffer_t_values(n_algorithms: int) -> List[int]:
    """
    Maximum number of hypotheses that can still be true at each step of
    Shaffer's static procedure.

    In an all-pairwise family the hypotheses are logically interrelated: if A
    differs from B, then A and B cannot both equal C. Shaffer exploits this by
    replacing Holm's divisor ``m - i + 1`` with ``t_i``, the largest number of
    hypotheses that can be simultaneously true once any ``i - 1`` of them are
    false. The attainable counts follow the recursion

    .. math:: S(k) = \\bigcup_{j=1}^{k} \\{ \\binom{j}{2} + x : x \\in S(k-j) \\}

    with ``S(0) = S(1) = {0}``.

    Parameters
    ----------
    n_algorithms : int
        Number of algorithms compared (k >= 2).

    Returns
    -------
    List[int]
        ``t_i`` for i = 1 .. k(k-1)/2, non-increasing.

    Examples
    --------
    >>> shaffer_t_values(3)
    [3, 1, 1]
    """
    if n_algorithms < 2:
        raise ValueError(f"Shaffer's procedure needs at least 2 algorithms, "
                         f"got {n_algorithms}.")

    attainable = {0: {0}, 1: {0}}
    for size in range(2, n_algorithms + 1):
        counts = set()
        for block in range(1, size + 1):
            counts.update(comb(block, 2) + rest for rest in attainable[size - block])
        attainable[size] = counts

    possible = sorted(attainable[n_algorithms])
    n_hypotheses = comb(n_algorithms, 2)
    return [max(value for value in possible if value <= n_hypotheses - index)
            for index in range(n_hypotheses)]


def _set_partitions(elements: List[int]) -> List[List[List[int]]]:
    """
    Enumerate every partition of a list of elements into non-empty blocks.

    Parameters
    ----------
    elements : List[int]
        Elements to partition.

    Returns
    -------
    List[List[List[int]]]
        One entry per partition, each a list of blocks.
    """
    if not elements:
        return [[]]

    first, rest = elements[0], elements[1:]
    partitions = []
    for partition in _set_partitions(rest):
        for index in range(len(partition)):
            partitions.append(
                partition[:index] + [[first] + partition[index]] + partition[index + 1:]
            )
        partitions.append([[first]] + partition)
    return partitions


def exhaustive_sets(
        pairs: Sequence[Tuple[int, int]],
        n_algorithms: int
) -> List[List[int]]:
    """
    All exhaustive index sets of an all-pairwise family of hypotheses.

    A set of hypotheses is *exhaustive* when exactly those hypotheses could be
    true at the same time. For pairwise equalities that happens precisely when
    the set collects all pairs inside the blocks of some partition of the
    algorithms, so the partitions of the k algorithms enumerate them.

    Parameters
    ----------
    pairs : Sequence[Tuple[int, int]]
        Algorithm index pair of every hypothesis, in the order of the p-values.
    n_algorithms : int
        Number of algorithms (k).

    Returns
    -------
    List[List[int]]
        Each entry lists the hypothesis indices of one non-empty exhaustive set.

    Raises
    ------
    ValueError
        If k exceeds :data:`BERGMANN_MAX_ALGORITHMS`, since the number of
        partitions grows faster than exponentially.
    """
    if n_algorithms > BERGMANN_MAX_ALGORITHMS:
        raise ValueError(
            f"Enumerating exhaustive sets is only tractable up to "
            f"{BERGMANN_MAX_ALGORITHMS} algorithms, got {n_algorithms}. Use "
            f"Shaffer's static procedure instead, which needs no enumeration."
        )

    hypothesis_of_pair = {frozenset(pair): index for index, pair in enumerate(pairs)}

    sets = []
    for partition in _set_partitions(list(range(n_algorithms))):
        indices = [
            hypothesis_of_pair[frozenset(pair)]
            for block in partition if len(block) > 1
            for pair in combinations(block, 2)
        ]
        if indices:
            sets.append(sorted(indices))
    return sets


def _hommel_adjusted(sorted_p: List[float]) -> List[float]:
    """
    Hommel's adjusted p-values for p-values sorted in ascending order.

    Implements the algorithm of Wright (1992), reproduced as Fig. 1 of Derrac
    et al. (2011): it walks the family sizes downwards, and at each size raises
    every APV to the critical value that size implies.

    Parameters
    ----------
    sorted_p : List[float]
        P-values in ascending order.

    Returns
    -------
    List[float]
        Adjusted p-values in the same (ascending) order.
    """
    m = len(sorted_p)
    adjusted = list(sorted_p)

    for size in range(m, 1, -1):
        # 1-based positions of the hypotheses in the current sub-family
        upper = list(range(m - size + 1, m + 1))
        c_min = min(size * sorted_p[i - 1] / (size + i - m) for i in upper)

        for i in upper:
            adjusted[i - 1] = max(adjusted[i - 1], c_min)
        for i in range(1, m - size + 1):
            adjusted[i - 1] = max(adjusted[i - 1], min(c_min, size * sorted_p[i - 1]))

    return adjusted


def _adjust_sorted(
        sorted_p: List[float],
        method: str,
        n_algorithms: Optional[int],
        sorted_pairs: Optional[List[Tuple[int, int]]]
) -> List[float]:
    """
    Apply one multiple-comparison procedure to ascending-sorted p-values.

    Parameters
    ----------
    sorted_p : List[float]
        P-values in ascending order.
    method : str
        Procedure name, see :func:`adjust_p_values`.
    n_algorithms : Optional[int]
        Number of algorithms, required by 'shaffer' and 'bergmann'.
    sorted_pairs : Optional[List[Tuple[int, int]]]
        Algorithm index pairs matching ``sorted_p``, required by 'bergmann'.

    Returns
    -------
    List[float]
        Adjusted p-values in the same order, each capped at 1.
    """
    m = len(sorted_p)
    indices = range(1, m + 1)

    if method in ('bonferroni', 'nemenyi'):
        raw = [m * p for p in sorted_p]
    elif method == 'holm':
        raw = list(np.maximum.accumulate([(m - i + 1) * p for i, p in zip(indices, sorted_p)]))
    elif method == 'holland':
        raw = list(np.maximum.accumulate(
            [1.0 - (1.0 - p) ** (m - i + 1) for i, p in zip(indices, sorted_p)]))
    elif method == 'finner':
        raw = list(np.maximum.accumulate(
            [1.0 - (1.0 - p) ** (m / i) for i, p in zip(indices, sorted_p)]))
    elif method == 'hochberg':
        # Step-up: the running minimum is taken from the largest p-value down
        raw = list(np.minimum.accumulate(
            [(m - i + 1) * p for i, p in zip(indices, sorted_p)][::-1]))[::-1]
    elif method == 'hommel':
        raw = list(np.maximum.accumulate(_hommel_adjusted(sorted_p)))
    elif method == 'li':
        # Two-step procedure; the largest p-value anchors the second step
        largest = sorted_p[-1]
        raw = [p / (p + 1.0 - largest) if p + 1.0 - largest > 0 else 1.0
               for p in sorted_p]
    elif method == 'shaffer':
        if n_algorithms is None:
            raise ValueError("Shaffer's procedure needs n_algorithms.")
        t_values = shaffer_t_values(n_algorithms)
        if len(t_values) != m:
            raise ValueError(
                f"Shaffer's procedure expects the full all-pairwise family of "
                f"{len(t_values)} hypotheses for {n_algorithms} algorithms, got {m}."
            )
        raw = list(np.maximum.accumulate(
            [t * p for t, p in zip(t_values, sorted_p)]))
    elif method == 'bergmann':
        if n_algorithms is None or sorted_pairs is None:
            raise ValueError("Bergmann-Hommel's procedure needs n_algorithms and pairs.")
        raw = [0.0] * m
        for indices_of_set in exhaustive_sets(sorted_pairs, n_algorithms):
            candidate = len(indices_of_set) * min(sorted_p[i] for i in indices_of_set)
            for i in indices_of_set:
                raw[i] = max(raw[i], candidate)
        # The exhaustive sets do not by themselves order the APVs, but an
        # adjusted p-value may never fall below one of a more significant
        # hypothesis, so enforce it
        raw = list(np.maximum.accumulate(raw))
    else:
        raise ValueError(
            f"Unknown procedure '{method}'. Available: "
            f"{sorted(set(CONTROL_PROCEDURES) | set(ALL_PAIRS_PROCEDURES))}"
        )

    return [float(min(1.0, value)) for value in raw]


def adjust_p_values(
        p_values: Sequence[Optional[float]],
        method: str = 'holm',
        n_algorithms: Optional[int] = None,
        pairs: Optional[Sequence[Tuple[int, int]]] = None
) -> List[Optional[float]]:
    """
    Convert raw p-values of a family of comparisons into adjusted p-values.

    An adjusted p-value (APV) is the smallest significance level at which the
    procedure would still reject that hypothesis, given the whole family. It can
    therefore be compared directly against any alpha, which is why Derrac et al.
    (2011) recommend reporting APVs rather than a bare accept/reject verdict.

    Supported procedures, from least to most powerful within each class:

    - one-step: ``'bonferroni'`` (``'nemenyi'`` is the same computation applied
      to an all-pairwise family)
    - step-down: ``'holm'``, ``'holland'``, ``'finner'``
    - step-up: ``'hochberg'``, ``'hommel'``
    - two-step: ``'li'``
    - logically-aware, all-pairwise only: ``'shaffer'``, ``'bergmann'``

    Entries that are None or NaN are not part of the family: they neither
    contribute to its size nor receive an adjusted value.

    Parameters
    ----------
    p_values : Sequence[Optional[float]]
        Raw p-values, in any order.
    method : str, optional
        Procedure name (default: ``'holm'``).
    n_algorithms : Optional[int], optional
        Number of algorithms, required by ``'shaffer'`` and ``'bergmann'``.
    pairs : Optional[Sequence[Tuple[int, int]]], optional
        Algorithm index pair of every hypothesis, aligned with ``p_values``.
        Required by ``'bergmann'``.

    Returns
    -------
    List[Optional[float]]
        Adjusted p-values in the input order, None wherever the input was.

    Examples
    --------
    >>> adjust_p_values([0.01, 0.02, 0.03], 'holm')
    [0.03, 0.04, 0.04]
    """
    adjusted: List[Optional[float]] = [None] * len(p_values)
    valid = [(index, float(p)) for index, p in enumerate(p_values)
             if p is not None and not np.isnan(p)]
    if not valid:
        return adjusted

    order = sorted(valid, key=lambda item: item[1])
    sorted_p = [p for _, p in order]
    sorted_pairs = None
    if pairs is not None:
        sorted_pairs = [tuple(pairs[index]) for index, _ in order]

    for (index, _), value in zip(order, _adjust_sorted(sorted_p, method,
                                                       n_algorithms, sorted_pairs)):
        adjusted[index] = value

    return adjusted


# =============================================================================
# Pairwise tests (two algorithms over many problems)
# =============================================================================

def _win_loss_tie(
        first: np.ndarray,
        second: np.ndarray,
        direction: OptimizationDirection
) -> Tuple[float, float, int]:
    """
    Count wins, losses and ties of the first algorithm, splitting ties evenly.

    Both papers insist that ties support the null hypothesis and must therefore
    be shared rather than discarded; an odd tie is ignored.

    Returns
    -------
    Tuple[float, float, int]
        Wins, losses and the raw number of ties.
    """
    better = first < second if direction == OptimizationDirection.MINIMIZE else first > second
    worse = first > second if direction == OptimizationDirection.MINIMIZE else first < second

    wins, losses = float(np.sum(better)), float(np.sum(worse))
    ties = int(np.sum(~better & ~worse))

    shared = ties // 2
    return wins + shared, losses + shared, ties


def rank_sum_test(
        first: Sequence[float],
        second: Sequence[float],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE
) -> PairwiseTestResult:
    """
    Wilcoxon rank-sum test on two *unpaired* samples.

    This is the test for the other setting the platform supports: the samples
    are the independent runs of two algorithms on a single problem, which are
    not paired with each other, so the ranks of the pooled sample are compared
    rather than the ranks of per-problem differences. Use
    :func:`wilcoxon_signed_rank_test` instead when the samples are one value per
    problem and therefore paired.

    Parameters
    ----------
    first : Sequence[float]
        Runs of the first algorithm.
    second : Sequence[float]
        Runs of the second algorithm; it need not have the same length.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE). It only orients the win and
        loss counts; the two-tailed p-value is the same either way.

    Returns
    -------
    PairwiseTestResult
        The rank-sum statistic and its two-tailed p-value. ``wins`` and
        ``losses`` compare the medians rather than counting problems, since
        unpaired runs cannot be matched up one by one.

    Raises
    ------
    ValueError
        If either sample is empty after dropping NaNs.
    """
    first_array = np.asarray(first, dtype=float).ravel()
    second_array = np.asarray(second, dtype=float).ravel()
    first_array = first_array[~np.isnan(first_array)]
    second_array = second_array[~np.isnan(second_array)]

    if first_array.size == 0 or second_array.size == 0:
        raise ValueError("The rank-sum test needs a non-empty sample on both sides.")

    statistic, p_value = stats.ranksums(first_array, second_array)

    better = np.median(first_array) < np.median(second_array)
    if direction == OptimizationDirection.MAXIMIZE:
        better = np.median(first_array) > np.median(second_array)
    tied = np.median(first_array) == np.median(second_array)

    return PairwiseTestResult(
        name='Wilcoxon rank-sum test',
        statistic=float(statistic),
        p_value=float(p_value),
        p_value_one_sided=float(p_value / 2.0),
        n_instances=int(first_array.size + second_array.size),
        wins=float(better and not tied),
        losses=float(not better and not tied),
        ties=int(tied)
    )


def sign_test(
        first: Sequence[float],
        second: Sequence[float],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE
) -> PairwiseTestResult:
    """
    Two-tailed sign test on the counts of wins, losses and ties.

    The simplest pairwise test: under the null hypothesis each algorithm wins on
    about half of the problems, so the number of wins follows a binomial
    distribution. It assumes nothing about the scores beyond their order, which
    also makes it the weakest of the pairwise tests -- it rejects only when one
    algorithm wins nearly always.

    Parameters
    ----------
    first : Sequence[float]
        Performance of the first algorithm, one value per problem.
    second : Sequence[float]
        Performance of the second algorithm, aligned with ``first``.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE).

    Returns
    -------
    PairwiseTestResult
        Wins, losses, ties and the exact binomial p-value. Both papers tabulate
        the critical number of wins for one-sided testing, so compare their
        tables against ``p_value_one_sided``: 11 wins out of 14 is their
        threshold at alpha = 0.05, which is a one-sided p of 0.029 and a
        two-sided p of 0.057.

    Raises
    ------
    ValueError
        If the two samples have different lengths or are empty.
    """
    first_array = np.asarray(first, dtype=float).ravel()
    second_array = np.asarray(second, dtype=float).ravel()
    if first_array.size != second_array.size:
        raise ValueError(
            f"Both samples must cover the same problems, got {first_array.size} "
            f"and {second_array.size} values."
        )

    valid = ~(np.isnan(first_array) | np.isnan(second_array))
    first_array, second_array = first_array[valid], second_array[valid]
    if first_array.size == 0:
        raise ValueError("The sign test needs at least one comparable problem.")

    wins, losses, ties = _win_loss_tie(first_array, second_array, direction)

    # An odd tie is ignored, which is what shrinks the effective sample
    trials = int(round(wins + losses))
    if trials:
        test = stats.binomtest(int(round(wins)), trials, 0.5)
        p_value = float(test.pvalue)
        p_one_sided = float(stats.binomtest(int(round(wins)), trials, 0.5,
                                            alternative='greater').pvalue)
    else:
        p_value = p_one_sided = 1.0

    return PairwiseTestResult(
        name='Sign test',
        statistic=wins,
        p_value=p_value,
        p_value_one_sided=p_one_sided,
        n_instances=int(first_array.size),
        wins=wins,
        losses=losses,
        ties=ties
    )


def wilcoxon_signed_rank_test(
        first: Sequence[float],
        second: Sequence[float],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE
) -> PairwiseTestResult:
    """
    Wilcoxon signed-rank test, the recommended pairwise comparison.

    The per-problem differences are ranked by absolute value and the ranks of
    the positive and negative differences are compared. Unlike the sign test it
    weighs how large each difference is, and unlike the paired t-test it assumes
    neither normality nor commensurability of the scores -- only that a larger
    difference counts for more. Zero differences are split evenly between the
    two rank sums, as both papers prescribe.

    Parameters
    ----------
    first : Sequence[float]
        Performance of the first algorithm, one value per problem.
    second : Sequence[float]
        Performance of the second algorithm, aligned with ``first``.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE). It only orients ``r_plus``
        and ``r_minus``; the two-tailed p-value is the same either way.

    Returns
    -------
    PairwiseTestResult
        T = min(R+, R-), its p-value, and both rank sums. ``r_plus`` is the sum
        of ranks on the problems where the *first* algorithm is better.

    Raises
    ------
    ValueError
        If the two samples have different lengths or are empty.
    """
    first_array = np.asarray(first, dtype=float).ravel()
    second_array = np.asarray(second, dtype=float).ravel()
    if first_array.size != second_array.size:
        raise ValueError(
            f"Both samples must cover the same problems, got {first_array.size} "
            f"and {second_array.size} values."
        )

    valid = ~(np.isnan(first_array) | np.isnan(second_array))
    first_array, second_array = first_array[valid], second_array[valid]
    if first_array.size == 0:
        raise ValueError("The Wilcoxon test needs at least one comparable problem.")

    # Orient so that a positive difference always favours the first algorithm
    differences = second_array - first_array
    if direction == OptimizationDirection.MAXIMIZE:
        differences = -differences

    magnitudes = np.abs(differences)
    ranks = stats.rankdata(magnitudes)
    zero_share = float(np.sum(ranks[differences == 0])) / 2.0
    r_plus = float(np.sum(ranks[differences > 0])) + zero_share
    r_minus = float(np.sum(ranks[differences < 0])) + zero_share

    if np.all(differences == 0):
        p_value = p_one_sided = 1.0
    else:
        p_value = float(stats.wilcoxon(differences, zero_method='zsplit').pvalue)
        p_one_sided = float(stats.wilcoxon(differences, zero_method='zsplit',
                                           alternative='greater').pvalue)

    wins, losses, ties = _win_loss_tie(first_array, second_array, direction)

    return PairwiseTestResult(
        name='Wilcoxon signed-rank test',
        statistic=min(r_plus, r_minus),
        p_value=p_value,
        p_value_one_sided=p_one_sided,
        n_instances=int(first_array.size),
        wins=wins,
        losses=losses,
        ties=ties,
        r_plus=r_plus,
        r_minus=r_minus
    )


# =============================================================================
# Effect size
# =============================================================================

def classify_cliffs_delta(delta: float) -> str:
    """
    Map a Cliff's delta to its qualitative magnitude.

    Parameters
    ----------
    delta : float
        Cliff's delta value; only its absolute value matters.

    Returns
    -------
    str
        'negligible' (``|delta| < 0.147``), 'small' (< 0.33), 'medium'
        (< 0.474), 'large' otherwise, or 'undefined' for NaN.
    """
    if delta is None or np.isnan(delta):
        return 'undefined'

    magnitude = abs(delta)
    negligible, small, medium = CLIFFS_DELTA_THRESHOLDS
    if magnitude < negligible:
        return 'negligible'
    if magnitude < small:
        return 'small'
    if magnitude < medium:
        return 'medium'
    return 'large'


def cliffs_delta(
        first: Sequence[float],
        second: Sequence[float],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE
) -> EffectSizeResult:
    """
    Compute Cliff's delta, a non-parametric effect size for two samples.

    Cliff's delta is the difference between the probability that a value of one
    sample exceeds a value of the other and the probability of the reverse; for
    independent samples it is equivalent to the rank-biserial correlation
    derived from the rank-sum statistic. It is reported independently of the
    significance test: a large delta on few runs may still be non-significant,
    and a significant difference may be negligible in size.

    The sign is oriented by ``direction`` so that a positive delta always means
    the first sample is better.

    Parameters
    ----------
    first : Sequence[float]
        Sample of the algorithm being tested. NaN entries are dropped.
    second : Sequence[float]
        Sample of the baseline. NaN entries are dropped.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE).

    Returns
    -------
    EffectSizeResult
        Delta in [-1, 1] with its qualitative magnitude, or
        (np.nan, 'undefined') when either sample is empty.
    """
    first_array = np.asarray(first, dtype=float).ravel()
    second_array = np.asarray(second, dtype=float).ravel()
    first_array = first_array[~np.isnan(first_array)]
    second_array = second_array[~np.isnan(second_array)]

    if first_array.size == 0 or second_array.size == 0:
        return EffectSizeResult(delta=np.nan, magnitude='undefined')

    differences = first_array[:, None] - second_array[None, :]
    dominance = int(np.sum(differences > 0)) - int(np.sum(differences < 0))
    delta = dominance / (first_array.size * second_array.size)

    # Positive must mean "the first sample is better", so flip when smaller wins
    if direction == OptimizationDirection.MINIMIZE:
        delta = -delta

    return EffectSizeResult(delta=delta, magnitude=classify_cliffs_delta(delta))


# =============================================================================
# Omnibus tests (many algorithms over many problems)
# =============================================================================

def friedman_test(
        data_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        algorithm_names: Sequence[str],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE
) -> RankingResult:
    """
    Friedman test: are the average ranks of the algorithms all equal?

    Each problem is ranked separately, rank 1 going to the best algorithm and
    tied algorithms sharing the average rank. The statistic

    .. math:: \\chi^2_F = \\frac{12N}{k(k+1)} \\left[ \\sum_j R_j^2 -
              \\frac{k(k+1)^2}{4} \\right]

    follows a chi-squared distribution with k-1 degrees of freedom. Because it
    is known to be conservative, the Iman-Davenport correction

    .. math:: F_F = \\frac{(N-1)\\chi^2_F}{N(k-1) - \\chi^2_F}

    is reported alongside it and is the statistic to prefer.

    Parameters
    ----------
    data_matrix : Union[np.ndarray, Sequence[Sequence[float]]]
        Results of shape (n_algorithms, n_instances).
    algorithm_names : Sequence[str]
        One name per row.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE).

    Returns
    -------
    RankingResult
        Average ranks, chi^2_F with its p-value, and the Iman-Davenport
        correction.
    """
    matrix, n_dropped = _prepare_matrix(data_matrix, algorithm_names, 'Friedman test')
    k, n = matrix.shape

    ranks = np.apply_along_axis(stats.rankdata, 0, _oriented(matrix, direction))
    average_ranks = ranks.mean(axis=1)

    statistic = (12.0 * n / (k * (k + 1))) * (
        float(np.sum(average_ranks ** 2)) - k * (k + 1) ** 2 / 4.0
    )
    statistic = max(statistic, 0.0)
    p_value = float(stats.chi2.sf(statistic, k - 1))

    denominator = n * (k - 1) - statistic
    if denominator <= 0:
        f_statistic, f_p_value = float('inf'), 0.0
    else:
        f_statistic = (n - 1) * statistic / denominator
        f_p_value = float(stats.f.sf(f_statistic, k - 1, (k - 1) * (n - 1)))

    return RankingResult(
        scheme=RankScheme.FRIEDMAN,
        average_ranks={name: float(rank) for name, rank in zip(algorithm_names, average_ranks)},
        ranks=ranks,
        statistic=float(statistic),
        p_value=p_value,
        statistic_name='chi^2_F',
        standard_error=float(np.sqrt(k * (k + 1) / (6.0 * n))),
        n_algorithms=k,
        n_instances=n,
        n_instances_dropped=n_dropped,
        iman_davenport_statistic=float(f_statistic),
        iman_davenport_p_value=f_p_value
    )


def friedman_aligned_test(
        data_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        algorithm_names: Sequence[str],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE
) -> RankingResult:
    """
    Friedman test on aligned ranks, which also compares across problems.

    Friedman ranks only within a problem, so a value is never compared against
    values from another problem. Aligning removes that limitation: the average
    performance on a problem is subtracted from every value on it, and all k*N
    aligned observations are then ranked together from 1 to k*N. This tends to
    be more powerful than the plain Friedman test when few algorithms are
    compared.

    Parameters
    ----------
    data_matrix : Union[np.ndarray, Sequence[Sequence[float]]]
        Results of shape (n_algorithms, n_instances).
    algorithm_names : Sequence[str]
        One name per row.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE).

    Returns
    -------
    RankingResult
        Average aligned ranks on the 1..k*N scale, and the statistic F_AR
        compared against a chi-squared distribution with k-1 degrees of freedom.

    Notes
    -----
    The average ranks and the post-hoc z statistics reproduce Tables 9, 12 and
    14 of Derrac et al. (2011) exactly. The omnibus statistic follows their
    Eq. (4), i.e. the Hodges-Lehmann form; the value printed in their Table 11
    is not reproducible from that equation and the ranks they publish.
    """
    matrix, n_dropped = _prepare_matrix(data_matrix, algorithm_names,
                                        'Friedman Aligned Ranks test')
    k, n = matrix.shape

    oriented = _oriented(matrix, direction)
    aligned = oriented - oriented.mean(axis=0, keepdims=True)
    ranks = stats.rankdata(aligned).reshape(k, n)

    rank_total_algorithm = ranks.sum(axis=1)
    rank_total_instance = ranks.sum(axis=0)
    average_ranks = rank_total_algorithm / n

    total = k * n
    numerator = (k - 1) * (
        float(np.sum(rank_total_algorithm ** 2)) - (k * n ** 2 / 4.0) * (total + 1) ** 2
    )
    denominator = (total * (total + 1) * (2 * total + 1) / 6.0
                   - float(np.sum(rank_total_instance ** 2)) / k)

    if denominator <= 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = max(numerator / denominator, 0.0)
        p_value = float(stats.chi2.sf(statistic, k - 1))

    return RankingResult(
        scheme=RankScheme.ALIGNED,
        average_ranks={name: float(rank) for name, rank in zip(algorithm_names, average_ranks)},
        ranks=ranks,
        statistic=float(statistic),
        p_value=p_value,
        statistic_name='F_AR',
        standard_error=float(np.sqrt(k * (total + 1) / 6.0)),
        n_algorithms=k,
        n_instances=n,
        n_instances_dropped=n_dropped
    )


def quade_test(
        data_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        algorithm_names: Sequence[str],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE
) -> RankingResult:
    """
    Quade test, a Friedman test that weighs the problems by how much they
    discriminate.

    Friedman treats every problem as equally important. Quade instead ranks the
    problems themselves by their sample range -- the spread between the best and
    the worst value observed on them -- and scales each problem's contribution
    by that rank, so a problem on which the algorithms differ widely counts for
    more than one on which they all behave alike.

    Parameters
    ----------
    data_matrix : Union[np.ndarray, Sequence[Sequence[float]]]
        Results of shape (n_algorithms, n_instances).
    algorithm_names : Sequence[str]
        One name per row.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE).

    Returns
    -------
    RankingResult
        Average weighted ranks T_j, and the statistic F_Q compared against an
        F distribution with k-1 and (k-1)(N-1) degrees of freedom.
    """
    matrix, n_dropped = _prepare_matrix(data_matrix, algorithm_names, 'Quade test')
    k, n = matrix.shape

    oriented = _oriented(matrix, direction)
    ranks = np.apply_along_axis(stats.rankdata, 0, oriented)

    # Problems ranked by how widely the algorithms spread out on them
    ranges = oriented.max(axis=0) - oriented.min(axis=0)
    problem_ranks = stats.rankdata(ranges)

    weighted = problem_ranks * ranks                       # W_i^j
    adjusted = problem_ranks * (ranks - (k + 1) / 2.0)     # S_i^j

    sums = adjusted.sum(axis=1)                            # S_j
    average_ranks = weighted.sum(axis=1) / (n * (n + 1) / 2.0)   # T_j

    a_term = n * (n + 1) * (2 * n + 1) * k * (k + 1) * (k - 1) / 72.0
    b_term = float(np.sum(sums ** 2)) / n

    if np.isclose(a_term, b_term):
        # Every problem ordered the algorithms identically; the paper gives the
        # exact probability of that happening by chance
        statistic = float('inf')
        p_value = float((1.0 / factorial(k)) ** (n - 1))
    else:
        statistic = (n - 1) * b_term / (a_term - b_term)
        p_value = float(stats.f.sf(statistic, k - 1, (k - 1) * (n - 1)))

    return RankingResult(
        scheme=RankScheme.QUADE,
        average_ranks={name: float(rank) for name, rank in zip(algorithm_names, average_ranks)},
        ranks=ranks,
        statistic=float(statistic),
        p_value=p_value,
        statistic_name='F_Q',
        standard_error=float(np.sqrt(
            k * (k + 1) * (2 * n + 1) * (k - 1) / (18.0 * n * (n + 1))
        )),
        n_algorithms=k,
        n_instances=n,
        n_instances_dropped=n_dropped
    )


def omnibus_test(
        data_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        algorithm_names: Sequence[str],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
        scheme: RankScheme = RankScheme.FRIEDMAN
) -> RankingResult:
    """
    Run the omnibus test of the requested ranking scheme.

    Parameters
    ----------
    data_matrix : Union[np.ndarray, Sequence[Sequence[float]]]
        Results of shape (n_algorithms, n_instances).
    algorithm_names : Sequence[str]
        One name per row.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE).
    scheme : RankScheme, optional
        FRIEDMAN, ALIGNED or QUADE (default: FRIEDMAN).

    Returns
    -------
    RankingResult
        Result of the corresponding test.
    """
    tests = {
        RankScheme.FRIEDMAN: friedman_test,
        RankScheme.ALIGNED: friedman_aligned_test,
        RankScheme.QUADE: quade_test,
    }
    if scheme not in tests:
        raise ValueError(f"Unknown ranking scheme: {scheme}")
    return tests[scheme](data_matrix, algorithm_names, direction)


# =============================================================================
# Post-hoc procedures
# =============================================================================

def _z_and_p(difference: float, standard_error: float) -> Tuple[float, float]:
    """Standardized rank difference and its two-tailed normal p-value."""
    z = difference / standard_error
    return float(z), float(2.0 * stats.norm.sf(abs(z)))


def control_post_hoc(
        ranking: RankingResult,
        control: Optional[str] = None,
        procedures: Sequence[str] = CONTROL_PROCEDURES
) -> PostHocResult:
    """
    Compare every algorithm against one control, with adjusted p-values.

    This is the 1xN family: k-1 hypotheses, one per non-control algorithm. It is
    the right family when the study asks whether a newly proposed method beats
    the existing ones, and it is more powerful than the all-pairwise family
    because it tests far fewer hypotheses.

    Parameters
    ----------
    ranking : RankingResult
        Output of :func:`friedman_test`, :func:`friedman_aligned_test` or
        :func:`quade_test`; its scheme decides the standard error used.
    control : Optional[str], optional
        Control algorithm. Default: None, which picks the best-ranked one.
    procedures : Sequence[str], optional
        Procedures whose APVs to report (default: all of
        :data:`CONTROL_PROCEDURES`).

    Returns
    -------
    PostHocResult
        One hypothesis per non-control algorithm, ordered by p-value.

    Raises
    ------
    ValueError
        If ``control`` is not one of the ranked algorithms.
    """
    average_ranks = ranking.average_ranks
    if control is None:
        control = min(average_ranks, key=average_ranks.get)
    if control not in average_ranks:
        raise ValueError(
            f"control '{control}' is not one of the algorithms: {list(average_ranks)}"
        )

    control_rank = average_ranks[control]
    hypotheses = []
    for name, rank in average_ranks.items():
        if name == control:
            continue
        z, p_value = _z_and_p(rank - control_rank, ranking.standard_error)
        better, worse = ((control, name) if control_rank <= rank else (name, control))
        hypotheses.append(Hypothesis(
            label=f'{control} vs {name}',
            algorithms=(better, worse),
            rank_difference=float(abs(rank - control_rank)),
            z_statistic=z,
            p_value=p_value
        ))

    hypotheses.sort(key=lambda hypothesis: hypothesis.p_value)
    _attach_adjusted(hypotheses, procedures, ranking.n_algorithms, pairs=None)

    return PostHocResult(
        family='control',
        scheme=ranking.scheme,
        control=control,
        hypotheses=hypotheses,
        procedures=tuple(procedures)
    )


def all_pairs_post_hoc(
        ranking: RankingResult,
        procedures: Sequence[str] = ALL_PAIRS_PROCEDURES
) -> PostHocResult:
    """
    Compare every algorithm against every other, with adjusted p-values.

    This is the NxN family: k(k-1)/2 hypotheses. Because the hypotheses are
    logically interrelated -- if A differs from B they cannot both equal C --
    Shaffer's and Bergmann-Hommel's procedures can reject more than Holm while
    still controlling the family-wise error rate.

    Parameters
    ----------
    ranking : RankingResult
        Output of one of the omnibus tests.
    procedures : Sequence[str], optional
        Procedures whose APVs to report (default: all of
        :data:`ALL_PAIRS_PROCEDURES`).

    Returns
    -------
    PostHocResult
        One hypothesis per unordered pair, ordered by p-value.
    """
    names = list(ranking.average_ranks)
    index_of = {name: index for index, name in enumerate(names)}

    hypotheses = []
    for first, second in combinations(names, 2):
        difference = ranking.average_ranks[first] - ranking.average_ranks[second]
        z, p_value = _z_and_p(difference, ranking.standard_error)
        better, worse = ((first, second) if difference <= 0 else (second, first))
        hypotheses.append(Hypothesis(
            label=f'{first} vs {second}',
            algorithms=(better, worse),
            rank_difference=float(abs(difference)),
            z_statistic=z,
            p_value=p_value
        ))

    hypotheses.sort(key=lambda hypothesis: hypothesis.p_value)
    pairs = [tuple(sorted((index_of[h.algorithms[0]], index_of[h.algorithms[1]])))
             for h in hypotheses]
    _attach_adjusted(hypotheses, procedures, ranking.n_algorithms, pairs)

    return PostHocResult(
        family='all_pairs',
        scheme=ranking.scheme,
        control=None,
        hypotheses=hypotheses,
        procedures=tuple(procedures)
    )


def nemenyi_critical_value(n_algorithms: int, significance_level: float = 0.05) -> float:
    """
    Critical value q_alpha of the two-tailed Nemenyi test.

    It is the studentized range statistic for ``n_algorithms`` groups and
    infinite degrees of freedom, divided by sqrt(2). The returned values
    reproduce Table 5(a) of Demsar (2006): 2.569 for 4 algorithms at
    alpha = 0.05 and 2.291 at alpha = 0.10.

    Parameters
    ----------
    n_algorithms : int
        Number of algorithms compared (k >= 2).
    significance_level : float, optional
        Significance level alpha (default: 0.05).

    Returns
    -------
    float
        Critical value q_alpha.

    Raises
    ------
    ValueError
        If fewer than 2 algorithms are given or alpha is outside (0, 1).
    """
    if n_algorithms < 2:
        raise ValueError(
            f"The Nemenyi critical value needs at least 2 algorithms, got {n_algorithms}."
        )
    if not 0.0 < significance_level < 1.0:
        raise ValueError(
            f"significance_level must lie in (0, 1), got {significance_level}."
        )

    return float(
        stats.studentized_range.ppf(1.0 - significance_level, n_algorithms, np.inf)
        / np.sqrt(2.0)
    )


def nemenyi_test(
        data_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        algorithm_names: Sequence[str],
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
        significance_level: float = 0.05
) -> NemenyiResult:
    """
    Nemenyi all-pairs post-hoc test in critical-difference form.

    Two algorithms differ significantly when their average Friedman ranks differ
    by at least

    .. math:: CD = q_\\alpha \\sqrt{k(k+1) / (6N)}

    with :math:`q_\\alpha` the studentized range statistic divided by sqrt(2).
    Because the critical value already accounts for all :math:`k(k-1)/2`
    comparisons, no further multiplicity correction applies on top of it.

    This returns the single threshold the critical difference diagram draws.
    :func:`all_pairs_post_hoc` answers the same question as adjusted p-values
    and additionally offers the more powerful Shaffer and Bergmann-Hommel
    procedures; use this one when a diagram is what you want.

    Parameters
    ----------
    data_matrix : Union[np.ndarray, Sequence[Sequence[float]]]
        Results of shape (n_algorithms, n_instances).
    algorithm_names : Sequence[str]
        One name per row.
    direction : OptimizationDirection, optional
        Optimization direction (default: MINIMIZE).
    significance_level : float, optional
        Significance level of the critical difference (default: 0.05). Demsar's
        own diagrams use 0.10, since the all-pairs test is conservative.

    Returns
    -------
    NemenyiResult
        Critical difference, average ranks, every pairwise comparison, and the
        cliques of algorithms that are not significantly different.

    Raises
    ------
    ValueError
        If fewer than 3 algorithms are supplied, if the names do not match the
        rows, or if fewer than 2 complete instances remain.
    """
    if len(algorithm_names) < 3:
        raise ValueError(
            f"The Nemenyi test needs at least 3 algorithms, got "
            f"{len(algorithm_names)}. Use rank_sum_test or "
            f"wilcoxon_signed_rank_test for a pairwise comparison instead."
        )

    ranking = friedman_test(data_matrix, algorithm_names, direction)
    average_ranks = ranking.average_ranks
    k = ranking.n_algorithms

    q_alpha = nemenyi_critical_value(k, significance_level)
    critical_difference = float(q_alpha * ranking.standard_error)

    # Better (lower) average rank first, so every pair reads "a vs worse b"
    ordered = sorted(algorithm_names, key=lambda name: average_ranks[name])

    comparisons = []
    for index, name_a in enumerate(ordered):
        for name_b in ordered[index + 1:]:
            difference = abs(average_ranks[name_a] - average_ranks[name_b])
            q_statistic = difference / ranking.standard_error * np.sqrt(2.0)
            comparisons.append(NemenyiComparison(
                algorithm_a=name_a,
                algorithm_b=name_b,
                rank_difference=float(difference),
                q_statistic=float(q_statistic),
                p_value=float(stats.studentized_range.sf(q_statistic, k, np.inf)),
                significant=bool(difference > critical_difference)
            ))

    comparisons.sort(key=lambda comparison: comparison.rank_difference)

    return NemenyiResult(
        critical_difference=critical_difference,
        q_alpha=q_alpha,
        average_ranks=average_ranks,
        n_algorithms=k,
        n_instances=ranking.n_instances,
        n_instances_dropped=ranking.n_instances_dropped,
        significance_level=significance_level,
        comparisons=comparisons,
        cliques=_build_cliques(ordered, average_ranks, critical_difference)
    )


def _build_cliques(
        ordered_names: List[str],
        average_ranks: Dict[str, float],
        critical_difference: float
) -> List[List[str]]:
    """
    Group algorithms whose average ranks lie within one critical difference.

    Walking the rank-sorted algorithms, every maximal run whose first and last
    member are at most CD apart is a clique: all its members are mutually
    non-significant, because any inner pair spans an even smaller rank interval.
    Runs contained in a longer one are dropped, as are single-member runs, which
    need no connecting bar in the diagram.

    Parameters
    ----------
    ordered_names : List[str]
        Algorithm names sorted by average rank, best first.
    average_ranks : Dict[str, float]
        Average rank per algorithm.
    critical_difference : float
        Critical difference of the Nemenyi test.

    Returns
    -------
    List[List[str]]
        Maximal cliques of at least two members, ordered by their best rank.
    """
    cliques = []
    last_end = -1

    for start in range(len(ordered_names)):
        end = start
        while (end + 1 < len(ordered_names)
               and (average_ranks[ordered_names[end + 1]]
                    - average_ranks[ordered_names[start]]) <= critical_difference):
            end += 1

        # A run ending no later than the previous one is contained in it
        if end > start and end > last_end:
            cliques.append(ordered_names[start:end + 1])
            last_end = end

    return cliques


def _attach_adjusted(
        hypotheses: List[Hypothesis],
        procedures: Sequence[str],
        n_algorithms: int,
        pairs: Optional[Sequence[Tuple[int, int]]]
) -> None:
    """
    Fill in the ``adjusted`` dictionary of every hypothesis, in place.

    Parameters
    ----------
    hypotheses : List[Hypothesis]
        Family of comparisons.
    procedures : Sequence[str]
        Procedures to apply.
    n_algorithms : int
        Number of algorithms, needed by Shaffer and Bergmann-Hommel.
    pairs : Optional[Sequence[Tuple[int, int]]]
        Algorithm index pairs aligned with ``hypotheses``, for Bergmann-Hommel.

    Returns
    -------
    None
    """
    raw = [hypothesis.p_value for hypothesis in hypotheses]
    for procedure in procedures:
        for hypothesis, value in zip(
                hypotheses,
                adjust_p_values(raw, procedure, n_algorithms=n_algorithms, pairs=pairs)):
            hypothesis.adjusted[procedure] = value


# =============================================================================
# Contrast estimation
# =============================================================================

def contrast_estimation(
        data_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        algorithm_names: Sequence[str]
) -> ContrastResult:
    """
    Estimate by how much each algorithm outperforms each other, using medians.

    Significance tests say whether a difference is real, not how large it is.
    Contrast estimation answers the second question on the original scale of the
    performance measure: for every pair it takes the median over problems of
    their per-problem differences, and turns those into one number per algorithm

    .. math:: m_u = \\frac{1}{k} \\sum_{j} Z_{uj}, \\qquad
              \\widehat{M_u - M_v} = m_u - m_v

    where :math:`Z_{uv}` is the median difference of u and v. It assumes the
    expected difference between two algorithms is the same across problems, so
    the values must be commensurable -- do not mix scales.

    The estimator is direction-agnostic: it is expressed in the units of the
    input, so with an error measure a negative value means u errs less than v.

    Parameters
    ----------
    data_matrix : Union[np.ndarray, Sequence[Sequence[float]]]
        Results of shape (n_algorithms, n_instances).
    algorithm_names : Sequence[str]
        One name per row.

    Returns
    -------
    ContrastResult
        The (k, k) estimator matrix and the per-algorithm terms behind it.
    """
    matrix, _ = _prepare_matrix(data_matrix, algorithm_names, 'contrast estimation')
    k = matrix.shape[0]

    # Z[u, v] is the median over problems of the difference between u and v
    medians = np.zeros((k, k))
    for u, v in combinations(range(k), 2):
        median = float(np.median(matrix[u] - matrix[v]))
        medians[u, v] = median
        medians[v, u] = -median

    performances = medians.sum(axis=1) / k
    estimators = performances[:, None] - performances[None, :]

    return ContrastResult(
        algorithms=list(algorithm_names),
        estimators=estimators,
        performances={name: float(value)
                      for name, value in zip(algorithm_names, performances)}
    )

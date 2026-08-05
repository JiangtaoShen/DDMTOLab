"""
Tests that Results.max_nfes reports the evaluations actually consumed.

The field is named after the budget but must hold a measurement. Reporting the
declared budget instead is silent: totals usually look right, while the split
across tasks is wrong, and DataAnalyzer uses the per-task value to scale the
horizontal axis of every convergence curve.

Two layers:

- a static scan of every ``build_save_results`` call site, which is fast and
  covers all algorithms at once;
- a runtime check against ground truth, obtained by counting at
  ``MTOP.evaluate_task`` -- the single point every real evaluation passes.
"""

import importlib
import pathlib
import re

import numpy as np
import pytest

from ddmtolab.Methods.Algo_Methods.algo_utils import set_seed
from ddmtolab.Methods.mtop import MTOP

ALGORITHMS_ROOT = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'ddmtolab' / 'Algorithms'


# =============================================================================
# Static scan of every call site
# =============================================================================

def iter_max_nfes_arguments():
    """Yield (relative path, argument expression) for every build_save_results call."""
    for path in sorted(ALGORITHMS_ROOT.rglob('*.py')):
        source = path.read_text(encoding='utf-8', errors='replace')
        for match in re.finditer(r'build_save_results\s*\(', source):
            index, depth = match.end(), 1
            while index < len(source) and depth:
                if source[index] == '(':
                    depth += 1
                elif source[index] == ')':
                    depth -= 1
                index += 1
            call = source[match.end():index]

            keyword = re.search(r'max_nfes\s*=\s*([A-Za-z_][\w.\[\]]*)', call)
            if keyword:
                argument = keyword.group(1)
            else:
                # positional: all_decs, all_objs, runtime, max_nfes
                positional = [a.strip() for a in call.split(',')[:4]]
                argument = positional[3] if len(positional) == 4 else '<unparsed>'
            yield str(path.relative_to(ALGORITHMS_ROOT)).replace('\\', '/'), argument


class TestNoCallSiteReportsTheDeclaredBudget:
    """No algorithm may hand the requested budget to the measured-count field."""

    def test_call_sites_were_found(self):
        found = list(iter_max_nfes_arguments())
        assert len(found) > 100, f"expected the whole algorithm suite, found {len(found)}"

    def test_no_call_site_passes_a_declared_budget(self):
        offenders = [(path, argument) for path, argument in iter_max_nfes_arguments()
                     if 'max_nfes' in argument]
        assert offenders == [], (
            "these call sites report the declared budget instead of the "
            f"evaluations actually consumed: {offenders}"
        )

    def test_every_call_site_was_parsed(self):
        unparsed = [path for path, argument in iter_max_nfes_arguments()
                    if argument == '<unparsed>']
        assert unparsed == [], f"could not read the max_nfes argument of: {unparsed}"


# =============================================================================
# Runtime check against ground truth
# =============================================================================

def sphere(x):
    return np.sum(x ** 2, axis=1, keepdims=True)


def shifted_sphere(x):
    return np.sum((x - 0.3) ** 2, axis=1, keepdims=True) + 0.5


def bi_objective_a(x):
    return np.hstack([np.sum(x ** 2, axis=1, keepdims=True),
                      np.sum((x - 1) ** 2, axis=1, keepdims=True)])


def bi_objective_b(x):
    return np.hstack([np.sum((x - 0.2) ** 2, axis=1, keepdims=True),
                      np.sum((x - 0.8) ** 2, axis=1, keepdims=True)])


def make_problem(kind):
    problem = MTOP()
    functions = (bi_objective_a, bi_objective_b) if kind == 'MTMO' else (sphere, shifted_sphere)
    problem.add_task(functions, dim=(4, 4), lower_bound=(-2, -2), upper_bound=(2, 2))
    return problem


def count_evaluations(problem):
    """Count real per-task evaluations at MTOP's single evaluation choke point."""
    counts = [0] * problem.n_tasks
    original = problem.evaluate_task

    def counting_evaluate_task(task_idx, X, *args, **kwargs):
        counts[task_idx] += np.atleast_2d(X).shape[0]
        return original(task_idx, X, *args, **kwargs)

    problem.evaluate_task = counting_evaluate_task
    return counts


MAX_NFES = 24
POP = 6

# (module, class, kind, trims_history)
#
# The trimming algorithms deliberately overshoot and then cut the recorded
# history back to the budget, so their reported count is the capped one.
CASES = [
    ('MTSO.MFEA', 'MFEA', 'MTSO', False),
    ('MTSO.MFEA_II', 'MFEA_II', 'MTSO', False),
    ('MTSO.G_MFEA', 'G_MFEA', 'MTSO', False),
    ('MTSO.SREMTO', 'SREMTO', 'MTSO', False),
    ('MTSO.EMTO_AI', 'EMTO_AI', 'MTSO', False),
    ('MTSO.MKTDE', 'MKTDE', 'MTSO', False),
    ('MTMO.MO_MFEA', 'MO_MFEA', 'MTMO', False),
    ('MTMO.MO_MFEA_II', 'MO_MFEA_II', 'MTMO', False),
    ('MTSO.DTSKT', 'DTSKT', 'MTSO', True),
    ('MTSO.EMEA', 'EMEA', 'MTSO', False),  # control: was already correct
]


@pytest.mark.parametrize('module_path, class_name, kind, trims', CASES)
def test_reported_counts_match_the_evaluations_performed(module_path, class_name, kind, trims):
    algorithm = getattr(importlib.import_module(f'ddmtolab.Algorithms.{module_path}'),
                        class_name)

    set_seed(11)
    problem = make_problem(kind)
    observed = count_evaluations(problem)
    results = algorithm(problem, n=POP, max_nfes=MAX_NFES, save_data=False,
                        disable_tqdm=True).optimize()
    reported = list(results.max_nfes)

    if trims:
        expected = [min(count, MAX_NFES) for count in observed]
        assert reported == expected, (
            f"{class_name} trims its history to the budget, so it should report "
            f"{expected}, not {reported} (raw evaluations were {observed})"
        )
    else:
        assert reported == observed, (
            f"{class_name} reported {reported} but performed {observed} evaluations"
        )


def test_the_split_across_tasks_is_not_assumed_even():
    """At least one algorithm must actually exercise an uneven split.

    If every case happened to divide evenly the checks above would pass even
    with the old behaviour, so the suite would not be testing anything.
    """
    uneven = False
    for module_path, class_name, kind, _ in CASES:
        algorithm = getattr(importlib.import_module(f'ddmtolab.Algorithms.{module_path}'),
                            class_name)
        set_seed(11)
        problem = make_problem(kind)
        observed = count_evaluations(problem)
        algorithm(problem, n=POP, max_nfes=MAX_NFES, save_data=False,
                  disable_tqdm=True).optimize()
        if len(set(observed)) > 1:
            uneven = True
            break
    assert uneven, ("no algorithm produced an uneven per-task split, so these "
                    "tests cannot distinguish measured counts from the budget")

"""Semantics of the MTOP problem abstraction.

These tests pin down the behaviour that the platform documents but that is easy
to regress silently: what selective evaluation returns, when outputs are padded,
how a failing evaluator surfaces, and what a task may declare about itself.
"""
import numpy as np
import pytest

from ddmtolab.Methods.mtop import MTOP


def sphere(x):
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def three_objectives(x):
    x = np.atleast_2d(x)
    return np.column_stack([
        np.sum(x ** 2, axis=1),
        np.sum((x - 1) ** 2, axis=1),
        np.sum(x, axis=1),
    ])


def two_constraints(x):
    x = np.atleast_2d(x)
    return np.column_stack([np.sum(x, axis=1) - 1.0, 0.5 - np.sum(x, axis=1)])


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.random((6, 3))


# --------------------------------------------------------------------------
# Selective evaluation
# --------------------------------------------------------------------------

def test_full_evaluation_returns_every_objective(X):
    p = MTOP()
    p.add_task(three_objectives, dim=3)
    obj, _ = p.evaluate_task(0, X)
    assert obj.shape == (6, 3)


def test_unconstrained_task_reports_one_satisfied_constraint_column(X):
    """An unconstrained task yields zeros((n, 1)), i.e. a trivially satisfied
    constraint, whereas skipping constraints yields an empty (n, 0) block."""
    p = MTOP()
    p.add_task(three_objectives, dim=3)
    _, con = p.evaluate_task(0, X)
    assert con.shape == (6, 1)
    assert np.all(con == 0.0)
    _, skipped = p.evaluate_task(0, X, eval_constraints=False)
    assert skipped.shape == (6, 0)


def test_selected_objectives_are_returned_narrow_and_in_order(X):
    """Unevaluated objectives are absent, not filled: the array narrows."""
    p = MTOP()
    p.add_task(three_objectives, dim=3)
    full, _ = p.evaluate_task(0, X)
    subset, _ = p.evaluate_task(0, X, eval_objectives=[2, 0])
    assert subset.shape == (6, 2)
    assert np.allclose(subset[:, 0], full[:, 2])
    assert np.allclose(subset[:, 1], full[:, 0])


def test_single_objective_index_is_accepted(X):
    p = MTOP()
    p.add_task(three_objectives, dim=3)
    full, _ = p.evaluate_task(0, X)
    one, _ = p.evaluate_task(0, X, eval_objectives=1)
    assert one.shape == (6, 1)
    assert np.allclose(one[:, 0], full[:, 1])


def test_skipping_objectives_yields_an_empty_column_block(X):
    p = MTOP()
    p.add_task(three_objectives, dim=3, constraint_func=two_constraints)
    obj, con = p.evaluate_task(0, X, eval_objectives=False)
    assert obj.shape == (6, 0)
    assert con.shape == (6, 2)


def test_selective_constraint_evaluation(X):
    p = MTOP()
    p.add_task(sphere, dim=3, constraint_func=two_constraints)
    full_obj, full_con = p.evaluate_task(0, X)
    _, one_con = p.evaluate_task(0, X, eval_constraints=[1])
    assert full_con.shape == (6, 2)
    assert one_con.shape == (6, 1)
    assert np.allclose(one_con[:, 0], full_con[:, 1])


def test_evaluation_is_stateless_and_uncharged(X):
    """MTOP never counts evaluations: accounting belongs to the algorithm, so a
    selective call is not implicitly cheaper at the problem level and repeated
    calls leave the problem unchanged."""
    p = MTOP()
    p.add_task(three_objectives, dim=3, budget=100)
    first, _ = p.evaluate_task(0, X, eval_objectives=[0])
    for _ in range(4):
        p.evaluate_task(0, X, eval_objectives=[0])
    again, _ = p.evaluate_task(0, X, eval_objectives=[0])
    assert np.allclose(first, again)
    assert p.budgets == [100]
    counters = [n for n in vars(p) if any(t in n.lower() for t in ('nfe', 'n_eval', 'count'))]
    assert counters == []


# --------------------------------------------------------------------------
# Padding is opt-in
# --------------------------------------------------------------------------

def test_no_padding_by_default_for_heterogeneous_tasks(X):
    p = MTOP()
    p.add_task(sphere, dim=3)
    p.add_task(three_objectives, dim=3)
    obj0, _ = p.evaluate_task(0, X)
    obj1, _ = p.evaluate_task(1, X)
    assert obj0.shape == (6, 1)
    assert obj1.shape == (6, 3)


def test_unified_mode_pads_to_the_maxima_with_fill_value(X):
    p = MTOP(unified_eval_mode=True, fill_value=-1.0)
    p.add_task(sphere, dim=3, constraint_func=two_constraints)
    p.add_task(three_objectives, dim=3)
    assert (p.m_max, p.c_max) == (3, 2)
    obj0, con0 = p.evaluate_task(0, X)
    assert obj0.shape == (6, 3) and con0.shape == (6, 2)
    assert np.all(obj0[:, 1:] == -1.0)
    obj1, con1 = p.evaluate_task(1, X)
    assert obj1.shape == (6, 3)
    # Task 1 is unconstrained: its single satisfied column is kept, the rest padded
    assert con1.shape == (6, 2)
    assert np.all(con1[:, 0] == 0.0) and np.all(con1[:, 1:] == -1.0)


def test_unified_mode_can_be_toggled_after_construction(X):
    p = MTOP()
    p.add_task(sphere, dim=3)
    p.add_task(three_objectives, dim=3)
    assert p.evaluate_task(0, X)[0].shape == (6, 1)
    p.set_unified_eval_mode(True, fill_value=0.0)
    assert p.unified_eval_mode is True
    assert p.evaluate_task(0, X)[0].shape == (6, 3)


# --------------------------------------------------------------------------
# Failure semantics
# --------------------------------------------------------------------------

def test_failing_evaluator_raises_rather_than_returning_a_sentinel(X):
    def broken(x):
        raise RuntimeError("solver crashed")

    p = MTOP()
    p.add_task(sphere, dim=3)
    p.tasks[0]['objective'] = broken
    with pytest.raises(RuntimeError):
        p.evaluate_task(0, X)


def test_non_vectorised_evaluator_is_retried_per_row(X):
    """A function that only accepts one sample still works in batch form."""
    calls = {'n': 0}

    def per_sample_only(x):
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] > 1:
            raise ValueError("this evaluator handles one sample at a time")
        calls['n'] += 1
        return float(np.sum(np.atleast_2d(x) ** 2))

    p = MTOP()
    p.add_task(per_sample_only, dim=3)
    obj, _ = p.evaluate_task(0, X)
    assert obj.shape == (6, 1)
    assert calls['n'] >= 6


def test_out_of_range_task_index_is_rejected(X):
    p = MTOP()
    p.add_task(sphere, dim=3)
    with pytest.raises(ValueError):
        p.evaluate_task(5, X)


# --------------------------------------------------------------------------
# What a task may declare about itself
# --------------------------------------------------------------------------

def test_budget_and_metadata_default_to_undeclared():
    p = MTOP()
    p.add_task(sphere, dim=3)
    assert p.budgets == [None]
    assert p.metadata == [{}]


def test_declared_budget_and_metadata_are_queryable():
    p = MTOP()
    p.add_task(sphere, dim=4, budget=120, metadata={'source': 'CFD', 'fidelity': 'RANS'})
    info = p.get_task_info(0)
    assert info['budget'] == 120
    assert info['metadata']['fidelity'] == 'RANS'


def test_budget_and_metadata_broadcast_over_the_tuple_form():
    p = MTOP()
    p.add_task((sphere, three_objectives), dim=(3, 5), budget=(30, 40), metadata={'suite': 'demo'})
    assert p.budgets == [30, 40]
    assert [m['suite'] for m in p.metadata] == ['demo', 'demo']
    q = MTOP()
    q.add_task((sphere, three_objectives), dim=3, budget=25)
    assert q.budgets == [25, 25]


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_budget_is_rejected(bad):
    p = MTOP()
    with pytest.raises(ValueError):
        p.add_task(sphere, dim=3, budget=bad)


def test_non_dict_metadata_is_rejected():
    p = MTOP()
    with pytest.raises(ValueError):
        p.add_task(sphere, dim=3, metadata=['not', 'a', 'dict'])


def test_heterogeneous_tasks_coexist_in_one_problem():
    """The defining case: tasks differing in dimension, objectives and constraints."""
    p = MTOP()
    p.add_task(sphere, dim=3, budget=50)
    p.add_task(three_objectives, dim=7, constraint_func=two_constraints, budget=80)
    assert p.n_tasks == 2
    assert p.dims == [3, 7]
    assert p.n_objs == [1, 3]
    assert p.n_cons == [0, 2]
    assert p.budgets == [50, 80]

"""Pin the decision variable type declaration on MTOP.

A task can say what each of its variables is: 'F' float, 'I' integer, 'B'
binary, 'C' categorical, 'O' ordinal. The declaration is descriptive. Nothing
in MTOP rounds, clamps or repairs a value because of it, so adding one to an
existing problem cannot change what that problem evaluates to.
"""
import pickle

import numpy as np
import pytest

from ddmtolab.Methods.mtop import (
    MTOP,
    VAR_TYPE_CODES,
    VAR_TYPE_NAMES,
    normalize_var_type,
    normalize_var_values,
    validate_var_bounds,
)


def sphere(x):
    return np.sum(x ** 2, axis=1)


def box(dim, lower=0.0, upper=1.0):
    return np.full(dim, lower, dtype=float), np.full(dim, upper, dtype=float)


# ---------------------------------------------------------------- defaults

def test_a_task_that_declares_nothing_is_all_float():
    problem = MTOP()
    problem.add_task(sphere, dim=5)
    info = problem.get_task_info(0)
    assert list(info["var_type"]) == ["F"] * 5
    assert info["var_values"] == {}


def test_get_task_info_keeps_every_key_it_had_before():
    problem = MTOP()
    problem.add_task(sphere, dim=3)
    info = problem.get_task_info(0)
    for key in ("dimension", "n_objectives", "n_constraints", "lower_bounds",
                "upper_bounds", "objective_func", "constraint_funcs", "budget",
                "metadata"):
        assert key in info


def test_the_declaration_does_not_change_what_a_task_evaluates_to():
    X = np.array([[0.3, 0.7, 2.4], [0.9, 0.1, 4.6]])

    plain = MTOP()
    plain.add_task(sphere, dim=3, lower_bound=[0, 0, 1], upper_bound=[1, 1, 5])

    declared = MTOP()
    declared.add_task(sphere, dim=3, lower_bound=[0, 0, 1], upper_bound=[1, 1, 5],
                      var_type="FBO", var_values={2: list("abcde")})

    assert np.array_equal(plain.evaluate_task(0, X)[0], declared.evaluate_task(0, X)[0])
    assert np.array_equal(plain.evaluate_task(0, X)[1], declared.evaluate_task(0, X)[1])


# ------------------------------------------------------------ input forms

def test_every_spelling_of_the_same_declaration_agrees():
    lower = [0, 0, 0, 0, 0, 0, 0, 0]
    upper = [1, 1, 1, 10, 10, 10, 4, 4]
    want = list("FFFIIICC")
    forms = [
        "FFFIIICC",
        ["F", "F", "F", "integer", "integer", "I", "categorical", "C"],
        [("float", 3), ("I", 3), ("C", 2)],
        {"I": [3, 4, 5], "C": range(6, 8)},
    ]
    for form in forms:
        problem = MTOP()
        problem.add_task(sphere, dim=8, lower_bound=lower, upper_bound=upper,
                         var_type=form)
        assert list(problem.get_task_info(0)["var_type"]) == want, form


def test_a_single_type_broadcasts_to_the_whole_task():
    for spec, code in (("I", "I"), ("integer", "I"), ("binary", "B"), ("O", "O")):
        problem = MTOP()
        lower, upper = box(4, 0, 1)
        problem.add_task(sphere, dim=4, lower_bound=lower, upper_bound=upper,
                         var_type=spec)
        assert list(problem.get_task_info(0)["var_type"]) == [code] * 4


def test_a_name_is_read_as_a_name_even_when_its_length_matches_the_dimension():
    """'integer' is seven characters; a seven-variable task must still read it
    as one type rather than as the letters i, n, t, e, g, e, r."""
    problem = MTOP()
    lower, upper = box(7, 0, 3)
    problem.add_task(sphere, dim=7, lower_bound=lower, upper_bound=upper,
                     var_type="integer")
    assert list(problem.get_task_info(0)["var_type"]) == ["I"] * 7


@pytest.mark.parametrize("code", VAR_TYPE_CODES)
def test_each_code_has_a_name_and_survives_a_round_trip(code):
    assert code in VAR_TYPE_NAMES
    assert normalize_var_type(VAR_TYPE_NAMES[code], 2).tolist() == [code, code]


# --------------------------------------------------------------- var_values

def test_var_values_records_what_each_index_stands_for():
    problem = MTOP()
    problem.add_task(sphere, dim=2, lower_bound=[0, 1], upper_bound=[4, 3],
                     var_type="CO",
                     var_values={0: ["tanh", "relu", "sigmoid", "sin", "swish"],
                                 1: [0.5, 0.75, 1.0]})
    values = problem.get_task_info(0)["var_values"]
    assert values[0] == ("tanh", "relu", "sigmoid", "sin", "swish")
    assert values[1] == (0.5, 0.75, 1.0)


def test_var_values_may_start_from_any_lower_bound():
    """A catalogue numbered 1..37 spans 37 entries, not 38."""
    problem = MTOP()
    problem.add_task(sphere, dim=1, lower_bound=[1], upper_bound=[37],
                     var_type="O", var_values={0: [f"IPE{i}" for i in range(37)]})
    assert len(problem.get_task_info(0)["var_values"][0]) == 37


def test_var_values_is_returned_as_a_copy():
    problem = MTOP()
    problem.add_task(sphere, dim=1, lower_bound=[0], upper_bound=[1],
                     var_type="C", var_values={0: ["a", "b"]})
    got = problem.get_task_info(0)["var_values"]
    got[0] = ("something", "else")
    assert problem.get_task_info(0)["var_values"][0] == ("a", "b")


def test_var_type_is_returned_as_a_copy():
    problem = MTOP()
    problem.add_task(sphere, dim=2, var_type="FF")
    got = problem.get_task_info(0)["var_type"]
    got[0] = "I"
    assert problem.get_task_info(0)["var_type"][0] == "F"


# --------------------------------------------------------------- multi-task

def test_one_spec_per_task():
    problem = MTOP()
    problem.add_task((sphere, sphere), dim=(3, 3), var_type=("FFI", "BBB"))
    assert ["".join(v) for v in problem.var_types] == ["FFI", "BBB"]


def test_one_run_length_spec_shared_by_every_task():
    problem = MTOP()
    problem.add_task((sphere, sphere), dim=(4, 4), var_type=[("F", 2), ("B", 2)])
    assert ["".join(v) for v in problem.var_types] == ["FFBB", "FFBB"]


def test_a_tuple_of_runs_is_one_spec_not_one_per_task():
    """(('F', 2), ('B', 2)) is a single task's runs even when the task count
    happens to be two."""
    problem = MTOP()
    problem.add_task((sphere, sphere), dim=(4, 4), var_type=(("F", 2), ("B", 2)))
    assert ["".join(v) for v in problem.var_types] == ["FFBB", "FFBB"]


def test_var_values_broadcasts_across_tasks():
    problem = MTOP()
    problem.add_task((sphere, sphere), dim=(1, 1), lower_bound=([0], [0]),
                     upper_bound=([2], [2]), var_type="C",
                     var_values={0: ["x", "y", "z"]})
    for t in range(2):
        assert problem.get_task_info(t)["var_values"][0] == ("x", "y", "z")


def test_add_tasks_accepts_the_declaration_in_its_config():
    problem = MTOP()
    problem.add_tasks([
        {"objective_func": sphere, "dim": 2, "var_type": "FF"},
        {"objective_func": sphere, "dim": 2, "lower_bound": [0, 0],
         "upper_bound": [1, 1], "var_type": "BB"},
    ])
    assert ["".join(v) for v in problem.var_types] == ["FF", "BB"]


# --------------------------------------------------------------- validation

@pytest.mark.parametrize("kwargs, fragment", [
    (dict(dim=3, var_type="FXF"), "unknown variable type"),
    (dict(dim=3, var_type="FF"), "2 entries but the task has 3"),
    (dict(dim=3, var_type=["F", "F", "F", "F"]), "covers 4 variables"),
    (dict(dim=3, var_type={"I": [0, 7]}), "outside the task"),
    (dict(dim=3, var_type={"I": [0], "B": [0]}), "both"),
    (dict(dim=2, lower_bound=[0, 0], upper_bound=[1, 5], var_type="BB"),
     "binary variables run over"),
    (dict(dim=2, lower_bound=[0, 0.5], upper_bound=[1, 3], var_type="FI"),
     "not integral"),
    (dict(dim=2, var_type="FF", var_values={0: ["a", "b"]}),
     "only categorical and ordinal"),
    (dict(dim=1, lower_bound=[0], upper_bound=[4], var_type="C",
          var_values={0: ["a", "b"]}), "spans 5 values"),
    (dict(dim=2, lower_bound=[0, 0], upper_bound=[1, 1], var_type="CC",
          var_values={5: ["a", "b"]}), "outside the task"),
    (dict(dim=2, var_type=42), "must be None, a string"),
    (dict(dim=2, var_type="FF", var_values=["a"]), "must be a dict"),
    (dict(dim=2, var_type=[]), "is empty"),
])
def test_a_declaration_that_cannot_hold_is_refused(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        MTOP().add_task(sphere, **kwargs)


def test_a_per_task_tuple_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one spec per task"):
        MTOP().add_task((sphere, sphere), dim=(2, 2), var_type=("FF", "FF", "FF"))


def test_binary_fits_the_default_box():
    """The default bounds are [0, 1], so 'B' needs no bounds of its own."""
    problem = MTOP()
    problem.add_task(sphere, dim=4, var_type="B")
    assert list(problem.get_task_info(0)["var_type"]) == ["B"] * 4


# ------------------------------------------------------------------ helpers

def test_normalize_var_type_handles_a_zero_length_run():
    assert normalize_var_type([("F", 2), ("I", 0), ("B", 1)], 3).tolist() == ["F", "F", "B"]


def test_validate_var_bounds_accepts_integral_bounds():
    codes = np.array(list("IBCO"), dtype="<U1")
    validate_var_bounds(codes, np.array([0., 0., 0., 2.]), np.array([9., 1., 3., 7.]))


def test_normalize_var_values_rejects_a_negative_run():
    with pytest.raises(ValueError, match="must not be negative"):
        normalize_var_type([("F", -1)], 1)


def test_normalize_var_values_returns_an_empty_map_for_none():
    codes = np.array(["F", "F"], dtype="<U1")
    assert normalize_var_values(None, codes, np.zeros(2), np.ones(2)) == {}


# ------------------------------------------------------------- persistence

def test_the_declaration_survives_pickling():
    """Batch runs ship problems to worker processes."""
    problem = MTOP()
    problem.add_task(sphere, dim=3, lower_bound=[0, 0, 1], upper_bound=[1, 1, 5],
                     var_type="FBO", var_values={2: list("abcde")})
    revived = pickle.loads(pickle.dumps(problem))
    before, after = problem.get_task_info(0), revived.get_task_info(0)
    assert np.array_equal(before["var_type"], after["var_type"])
    assert before["var_values"] == after["var_values"]


def test_var_types_property_reports_every_task():
    problem = MTOP()
    problem.add_task(sphere, dim=2, var_type="FF")
    problem.add_task(sphere, dim=3, var_type="B")
    assert ["".join(v) for v in problem.var_types] == ["FF", "BBB"]

"""Pin the STSO landscapes to the MToP MATLAB reference.

Every expected value here was produced by running the reference implementations
in MATLAB R2020b, from the MToP checkout, on the inputs written below:

    MTO/Problems/Base/*.m                                  (the nine landscapes)
    MTO/Problems/Single-task/CEC10-CSO/CEC10_CSO_Func.m    (the 18 CEC 2010 problems)

They are stored so the agreement keeps being checked without MATLAB installed.
CLASSICALSO and STSOtest are the nine landscapes under a rotation and a shift,
so pinning the landscapes pins both suites.
"""
import numpy as np
import pytest

from ddmtolab.Problems.BasicFunctions.basic_functions import (
    Ackley, Elliptic, Griewank, Rastrigin, Rosenbrock,
    Schwefel, Schwefel2, Sphere, Weierstrass,
)
from ddmtolab.Problems.STSO.cec10_cso import CEC10_CSO

FUNCS = {
    "Ackley": Ackley, "Elliptic": Elliptic, "Griewank": Griewank,
    "Rastrigin": Rastrigin, "Rosenbrock": Rosenbrock, "Schwefel": Schwefel,
    "Schwefel2": Schwefel2, "Sphere": Sphere, "Weierstrass": Weierstrass,
}

BASE_X = np.array([
    [0.5, -1.25, 2.0, 0.75],
    [-3.5, 0.125, 1.5, -2.25],
    [1.0, 2.5, -0.5, 3.0],
])
BASE_M = np.array([
    [0.8, -0.6, 0.0, 0.0],
    [0.6, 0.8, 0.0, 0.0],
    [0.0, 0.0, 0.6, -0.8],
    [0.0, 0.0, 0.8, 0.6],
])
BASE_O = np.array([[0.25, -0.5, 1.0, -0.75]])

# MATLAB: [Obj, ~] = <name>(BASE_X, BASE_M, BASE_O, 0.0)
MATLAB_BASE = {
    "Ackley": [5.777333275606758, 8.960122052068307, 8.749471201342377],
    "Elliptic": [2893620.6724999994, 272817.6406249999, 1255413.6899999995],
    "Griewank": [0.5315850664896096, 1.2040704088072829, 0.921603985624609],
    "Rastrigin": [70.44375757337522, 84.02419281186548, 39.30624242662475],
    "Rosenbrock": [324.8712499999999, 18294.9462890625, 34735.693125000005],
    "Schwefel": [1674.5221146045571, 1679.7978494771337, 1676.8568565683247],
    "Schwefel2": [2.312499999999999, 67.8125, 10.664999999999996],
    "Sphere": [3.875, 16.953125, 25.875],
    "Weierstrass": [9.8311178745726, 12.4713990235205, 6.168874495996192],
}

CEC10_X = np.array([
    [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -1.0],
    [0.5, 1.5, -2.5, 3.5, -4.5, 5.5, -6.5, 7.5, -8.5, 2.5],
])

# MATLAB: [Obj, Con] = CEC10_CSO_Func(CEC10_X, k) at D = 10
MATLAB_CEC10_OBJ = {
    1: [-0.0759717845028249, -0.07692754841446071],
    2: [8.506564091247332, 7.1735050453498825],
    3: [9588399150390.262, 10214972731640.365],
    4: [12.246522827444975, 9.089298730330405],
    5: [-11.007673762322495, 0.0030139665317179265],
    6: [16.19086235816082, 6.237189363693078],
    7: [1252513760.4018168, 588355882.7809429],
    8: [1252513760.4018168, 588355882.7809429],
    9: [23570678732.360264, 16083858870.517582],
    10: [23570678732.360264, 16083858870.517582],
    11: [-0.869921312355682, 1.3711958021700585],
    12: [88.67449892644038, 30.734898423455306],
    13: [26.848485090930087, 32.05144883578904],
    14: [2220060360.0323386, 1865073735.6312602],
    15: [2220060360.0323386, 1865073735.6312602],
    16: [1.0611676629769489, 1.0749715149669046],
    17: [1197.4574483998747, 825.1391142421887],
    18: [1302.582511025563, 800.8563516974837],
}
MATLAB_CEC10_CON = {
    1: [349275.0385433533, 0.0, 0.0, 0.0, 0.0, 0.0],
    2: [0.0, 26.278011066508633, 16.672428609621843, 0.0, 17.493500799576992, 17.23133637520183],
    3: [0.0, 198618.94064278447, 0.0, 221188.85935684072],
    4: [0.0, 1.5957267557380042, 220.53195237337076, 173549.56776595788, 25.99325437866527, 0.0, 1.1440656483971832, 299.3055864309308, 57831.058879613964, 30.99325437866527],
    5: [0.0, 19.873071879326776, 25.199776879031194, 0.0, 23.75195826719363, 21.215636903877954],
    6: [0.0, 110.95842399318494, 203.9826951509235, 0.0, 113.39527557617755, 215.2352575322361],
    7: [0.08722559376711958, 0.0, 0.48897779433948907, 0.0],
    8: [0.0, 0.0, 0.0, 0.0],
    9: [0.0, 258.717524744153, 0.0, 90.89208940942203],
    10: [0.0, 81.23930477755614, 0.0, 90.23705048258446],
    11: [0.0, 1824898.8537365266, 0.0, 1145307.7074287757],
    12: [368.2913647883166, 89381126.7824852, 230.04389994173272, 109839148.2126605],
    13: [0.0, 7.739021922349689, 0.0, 0.0, 0.0, 16.54804256206263, 0.0, 0.0],
    14: [0.0, 229.59366251692347, 0.0, 0.0, 0.0, 238.34066711098131, 0.0, 0.0],
    15: [0.0, 269.19167907973986, 707.9634098808325, 0.0, 0.0, 2118.0680288152935, 0.0, 0.0],
    16: [414.47470385025105, 0.0, 0.9035327645248756, 0.9035327645248756, 726.8395093186282, 25613.92119244894, 3.4269175636500324, 3.4269175636500324],
    17: [0.0, 1.9417811803930847, 1.540759604496317, 149868.39655840458, 0.0, 3.1992282710531708],
    18: [0.010841356461313213, 0.010741356461313214, 0.0, 0.7710834021508555],
}

# Weierstrass folds b**20 = 3**20 into a cosine, so a last-bit difference in the
# rotated input moves the result by about 1e-11. Every other landscape is exact.
TOL = {name: 1e-13 for name in FUNCS}
TOL["Weierstrass"] = 1e-10


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_landscape_matches_the_matlab_reference(name):
    got = FUNCS[name](BASE_X, BASE_M, BASE_O, 0.0).ravel()
    want = np.array(MATLAB_BASE[name])
    assert np.allclose(got, want, rtol=TOL[name], atol=0.0)


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_a_scalar_rotation_and_shift_broadcast(name):
    """MToP calls its landscapes as f(x, 1, 0, 0); that has to keep working."""
    D = BASE_X.shape[1]
    scalar = FUNCS[name](BASE_X, 1, 0, 0.0)
    spelled_out = FUNCS[name](BASE_X, np.eye(D), np.zeros((1, D)), 0.0)
    assert np.array_equal(scalar, spelled_out)


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_an_oversized_rotation_and_shift_are_cropped(name):
    """MATLAB slices M(1:D, 1:D) and opt(1:D); anything wider is ignored."""
    D = BASE_X.shape[1]
    rng = np.random.default_rng(11)
    big_M = rng.normal(size=(D + 3, D + 3))
    big_o = rng.normal(size=D + 3)
    wide = FUNCS[name](BASE_X, big_M, big_o, 0.0)
    cropped = FUNCS[name](BASE_X, big_M[:D, :D], big_o[:D].reshape(1, D), 0.0)
    assert np.array_equal(wide, cropped)


@pytest.mark.parametrize("k", range(1, 19))
def test_cec10_objective_matches_the_matlab_reference(k):
    problem = getattr(CEC10_CSO(), f"P{k}")(D=10)
    obj, _ = problem.evaluate_task(0, CEC10_X)
    assert np.allclose(obj.ravel(), MATLAB_CEC10_OBJ[k], rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("k", range(1, 19))
def test_cec10_constraints_match_the_matlab_reference(k):
    problem = getattr(CEC10_CSO(), f"P{k}")(D=10)
    _, con = problem.evaluate_task(0, CEC10_X)
    want = np.array(MATLAB_CEC10_CON[k]).reshape(con.shape)
    assert np.allclose(con, want, rtol=1e-12, atol=0.0)


def test_cec10_bounds_match_the_mtop_table():
    """benchmark_CEC10_CSO.m carries one lower and upper bound per problem."""
    lb = [0, -5.12, -1000, -50, -600, -600, -140, -140, -500,
          -500, -100, -1000, -500, -1000, -1000, -10, -10, -50]
    ub = [10, 5.12, 1000, 50, 600, 600, 140, 140, 500,
          500, 100, 1000, 500, 1000, 1000, 10, 10, 50]
    for k in range(1, 19):
        info = getattr(CEC10_CSO(), f"P{k}")(D=10).get_task_info(0)
        assert np.all(np.ravel(info["lower_bounds"]) == lb[k - 1]), f"P{k} lower bound"
        assert np.all(np.ravel(info["upper_bounds"]) == ub[k - 1]), f"P{k} upper bound"

"""Pin the real-world suites against MToP, and against themselves where MToP has none.

MToP ships a counterpart for five of the eight suites; the values below came
from it in MATLAB R2020b, on a fixed lattice of fractions mapped onto each
task's own box, so both sides reach the same points without carrying a decision
matrix per problem.

SCP, MO_SCP and SOPM agree to machine precision. The other two need a word:

PEPVM  matches exactly wherever its numerical guards do not bind. The port
       clamps the diode exponential and the residual, which MToP does not, so
       the two part company only where the residual has already blown past 1e10
       and the RMSE is in the hundreds of thousands. The optimum sits near
       1e-3, so the guard never touches the region an optimizer works in, and
       it keeps Inf out of a platform that fits surrogates to these values.

PKACP  draws its per-task Amax and Lmax by k-means over random samples and
       caches the result, on both platforms independently, so the two hold
       different problem instances. Feeding MToP's parameters into this
       implementation reproduces MToP to 1e-16 across all twenty tasks, which
       is what the arm test below pins.

PINN_HPO, NN_Training and TSP have no MToP counterpart. TSP is checked against
a closed form; the others are covered by the suite-wide smoke tests.
"""
import importlib

import numpy as np
import pytest

# name -> (module, class, method, [(D, M, objectives, constraints) per task])
MTOP = {
    "PEPVM": ("pepvm", "PEPVM", "P1", [(5, 1, [0.6844949227062461, 0.6318432705052156, 0.5792387040026902], [0.0, 0.0, 0.0]), (7, 1, [0.6870946562085897, 0.6341033370472691, 7420.523096122684], [0.0, 0.0, 0.0]), (5, 1, [1062.0855516984955, 211.34136998822913, 55.98171685222052], [0.0, 0.0, 0.0])]),
    "SCP": ("scp", "SCP", "P1", [(75, 1, [868.1850947231374, 845.6598795036667, 877.7638703634818], [0.0, 0.0, 0.0]), (78, 1, [845.0201363056648, 878.08882054416, 867.8390543628441], [0.0, 0.0, 0.0]), (81, 1, [843.7731712987566, 860.8150460994791, 844.060917074078], [0.0, 0.0, 0.0])]),
    "MOSCP": ("mo_scp", "MO_SCP", "P1", [(84, 2, [83.32, 36.88553778297375, 81.04, 37.96711924752896, 83.08, 36.38493729408014], [0.0, 0.0, 0.0]), (87, 2, [80.92, 39.00247967371665, 83.88, 37.489678100754595, 82.88, 38.565456610691896], [0.0, 0.0, 0.0]), (90, 2, [80.60000000000001, 39.346634870868314, 82.2, 39.386267137846744, 80.72, 40.536926081411416], [0.0, 0.0, 0.0])]),
    "MOSCP2": ("mo_scp", "MO_SCP", "P2", [(75, 2, [83.88, 41.473094723137415, 81.76, 42.23587950366671, 84.64, 40.82787036348177], [0.0, 0.0, 0.0]), (84, 2, [81.16, 45.767400255074925, 84.16, 44.512862046976295, 83.08, 45.43656042087363], [0.0, 0.0, 0.0]), (93, 2, [80.36, 48.657271521947074, 81.96, 48.89395342225529, 80.47999999999999, 49.93600550005314], [0.0, 0.0, 0.0])]),
    "SOPM_MTMO1": ("sopm", "SOPM", "P1", [(25, 2, [0.6972436007298234, 0.02688838948087591, 0.6598570873070403, 0.009369054171007696, 0.789873632790269, 1.0950884230497893], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0]), (25, 2, [3.4504725680825383, 0.5753308425904329, 3.2462928139102667, 0.7705530472587527, 3.1352276932174097, 1.3289140313443988], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0]), (25, 2, [1.721062872910241, 5.220515516814607, 1.078606426378824, 10.143785492224858, 1.5069222076441322, 0.7736664677704669], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0])]),
    "SOPM_MTMO2": ("sopm", "SOPM", "P2", [(30, 2, [1.9249224803880025, 11.656583020452016, 3.4681883042025565, 1.367713778327499, 1.713987066806379, 8.020313747292715], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691]), (30, 2, [4.270532844485247, 6.777862885946438, 3.7793151974096113, 1.9559458725953256, 1.5254453670716295, 1.4326359318983126], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691]), (30, 2, [1.6513265438938107, 10.582764737608041, 3.364678555241659, 9.572638044068134, 3.082952109665546, 16.410565603904228], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.93814532989691, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 77.9381453298969, 0.0])]),
}

# PEPVM's guards bind on the first task's probe, so it is checked on its own terms.
EXACT = [n for n in MTOP if n != "PEPVM"]

# Amax, Lmax and the distances MToP's own fw_kinematics gives on the same angles
PKACP_ARM = (0.375, 0.625, [0.5812212828147011, 0.5737119647849321, 0.5414635227523986])


def probe(lower, upper, task_index):
    """The same three points MATLAB builds, in the task's own box."""
    D = len(lower)
    a = np.arange(3)[:, None]
    b = np.arange(D)[None, :]
    frac = ((a * 7 + b * 13 + task_index * 5) % 97 + 0.5) / 97
    return lower + frac * (upper - lower)


def evaluate(name, t):
    mod_name, cls_name, meth = MTOP[name][:3]
    mod = importlib.import_module(f"ddmtolab.Problems.RWO.{mod_name}")
    problem = getattr(getattr(mod, cls_name)(), meth)()
    info = problem.get_task_info(t)
    lower = np.ravel(info["lower_bounds"]).astype(float)
    upper = np.ravel(info["upper_bounds"]).astype(float)
    X = probe(lower, upper, t)
    obj, con = problem.evaluate_task(t, X)
    return info, X, obj, con


@pytest.mark.parametrize("name", sorted(EXACT))
def test_matches_mtop(name):
    for t, (D, M, want_obj, want_con) in enumerate(MTOP[name][3]):
        info, _, obj, con = evaluate(name, t)
        assert info["dimension"] == D, f"{name} task {t + 1} dimension"
        assert info["n_objectives"] == M, f"{name} task {t + 1} objective count"
        assert np.allclose(obj.ravel(), want_obj, rtol=1e-10, atol=1e-10), \
            f"{name} task {t + 1} objectives"
        assert np.allclose(con.ravel(), want_con, rtol=1e-10, atol=1e-10), \
            f"{name} task {t + 1} constraints"


def test_pepvm_matches_mtop_where_the_guards_do_not_bind():
    """Strip the clamps and the single-diode residual reproduces MToP exactly."""
    _, X, _, _ = evaluate("PEPVM", 0)
    _, _, want_obj, _ = MTOP["PEPVM"][3][0]

    q, k, T = 1.60217646e-19, 1.3806503e-23, 273.15 + 33.0
    V_t = k * T / q
    V_L = np.array([-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678,
                    0.2132, 0.2545, 0.2924, 0.3269, 0.3585, 0.3873, 0.4137,
                    0.4373, 0.4590, 0.4784, 0.4960, 0.5119, 0.5265, 0.5398,
                    0.5521, 0.5633, 0.5736, 0.5833, 0.5900])
    I_L = np.array([0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570,
                    0.7570, 0.7555, 0.7540, 0.7505, 0.7465, 0.7385, 0.7280,
                    0.7065, 0.6755, 0.6320, 0.5730, 0.4990, 0.4130, 0.3165,
                    0.2120, 0.1035, -0.0100, -0.1230, -0.2100])
    I_ph, I_sd, R_s, R_sh, a = (X[:, i:i + 1] for i in range(5))
    total = np.zeros((len(X), 1))
    for i in range(len(V_L)):
        y = (I_ph - I_sd * (np.exp((V_L[i] + I_L[i] * R_s) / (a * V_t)) - 1)
             - (V_L[i] + I_L[i] * R_s) / R_sh - I_L[i])
        total += y ** 2
    assert np.allclose(np.sqrt(total / len(V_L)).ravel(), want_obj, rtol=1e-9, atol=0.0)


def test_pepvm_guards_only_bite_far_from_the_optimum():
    """Wherever the shipped objective differs from MToP, MToP is already in the
    tens of thousands, orders away from the 1e-3 the optimum reaches."""
    _, _, obj, _ = evaluate("PEPVM", 0)
    want = np.array(MTOP["PEPVM"][3][0][2])
    differs = ~np.isclose(obj.ravel(), want, rtol=1e-9, atol=0.0)
    assert np.all(want[differs] > 1e4), "a guard changed a value an optimizer would visit"
    assert np.allclose(obj.ravel()[~differs], want[~differs], rtol=1e-10, atol=1e-10)


def test_pkacp_arm_matches_mtop_given_the_same_task_parameters():
    """The instances differ because both platforms draw their own task_para;
    the kinematics behind them do not."""
    from ddmtolab.Problems.RWO.pkacp import PKACP

    Amax, Lmax, want = PKACP_ARM
    D = 20
    a = np.arange(3)[:, None]
    b = np.arange(D)[None, :]
    angles = ((a * 7 + b * 13) % 97 + 0.5) / 97
    got = PKACP()._evaluate_pkacp(angles, Amax, Lmax, D)
    assert np.allclose(got.ravel(), want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("pid", range(1, 7))
def test_tsp_objective_is_the_closed_tour_length(pid):
    """Decode the random keys independently and walk the tour."""
    from ddmtolab.Problems.RWO.tsp import TSP

    suite = TSP()
    coords = suite._get_coords(pid)
    problem = getattr(suite, f"P{pid}")()
    n = coords.shape[0]
    rng = np.random.default_rng(pid)
    X = rng.random((4, problem.get_task_info(0)["dimension"]))
    obj, _ = problem.evaluate_task(0, X)
    for r in range(4):
        order = np.argsort(X[r, :n])
        assert sorted(order.tolist()) == list(range(n)), "not a permutation"
        pts = coords[order]
        legs = np.sqrt(((pts - np.roll(pts, -1, axis=0)) ** 2).sum(axis=1))
        assert np.isclose(obj[r, 0], legs.sum(), rtol=0.0, atol=1e-9)


def test_tsp_circle_instance_hits_the_closed_form():
    """P2 puts 30 cities on a circle, so the in-order tour is the perimeter of
    the inscribed regular polygon."""
    from ddmtolab.Problems.RWO.tsp import TSP

    suite = TSP()
    coords = suite._get_coords(2)
    n = coords.shape[0]
    radius = np.linalg.norm(coords[0] - coords.mean(axis=0))
    problem = getattr(suite, "P2")()
    obj, _ = problem.evaluate_task(0, (np.arange(n) / n).reshape(1, -1))
    assert np.isclose(obj[0, 0], 2 * n * radius * np.sin(np.pi / n), rtol=1e-12)

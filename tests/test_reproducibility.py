"""Reproducibility guarantees: seeding, budget resolution, and batch bookkeeping.

The platform claims that one seeding call fixes a whole run and that a task's
declared budget reaches the algorithm. Both are cheap to break and expensive to
notice, so they are pinned here.
"""
import numpy as np
import pytest

from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.algo_utils import resolve_budget, set_seed


def sphere(x):
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def shifted(x):
    x = np.atleast_2d(x)
    return np.sum((x - 0.3) ** 2, axis=1)


# Batch runs execute in child processes, so problem creators must be importable
# at module level rather than closures.
def make_sphere_problem():
    p = MTOP()
    p.add_task(sphere, dim=3, lower_bound=-5, upper_bound=5)
    return p


def make_broken_problem():
    raise RuntimeError("evaluator unavailable")


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def test_set_seed_makes_numpy_and_random_streams_repeat():
    import random

    set_seed(1234)
    a = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
    set_seed(1234)
    b = (np.random.rand(5).tolist(), [random.random() for _ in range(5)])
    assert a == b


def test_set_seed_returns_the_applied_seed_for_logging():
    assert set_seed(7) == 7


def test_different_seeds_give_different_streams():
    set_seed(1)
    a = np.random.rand(5).tolist()
    set_seed(2)
    b = np.random.rand(5).tolist()
    assert a != b


def test_set_seed_covers_torch_when_available():
    torch = pytest.importorskip("torch")
    set_seed(99)
    a = torch.randn(4).tolist()
    set_seed(99)
    b = torch.randn(4).tolist()
    assert a == b


# --------------------------------------------------------------------------
# Budget resolution
# --------------------------------------------------------------------------

@pytest.fixture
def declared():
    p = MTOP()
    p.add_task(sphere, dim=3, budget=60)
    p.add_task(shifted, dim=4, budget=90)
    return p


@pytest.fixture
def undeclared():
    p = MTOP()
    p.add_task(sphere, dim=3)
    p.add_task(shifted, dim=4)
    return p


def test_explicit_budget_overrides_the_declared_one(declared):
    assert resolve_budget(declared, 5, 2) == [5, 5]
    assert resolve_budget(declared, [7, 8], 2) == [7, 8]


def test_declared_budget_is_used_when_the_caller_omits_one(declared):
    assert resolve_budget(declared, None, 2) == [60, 90]


def test_algorithm_default_applies_when_nothing_is_declared(undeclared):
    assert resolve_budget(undeclared, None, 2, default=200) == [200, 200]


def test_none_is_returned_when_there_is_nothing_to_fall_back_on(undeclared):
    assert resolve_budget(undeclared, None, 2) is None


def test_partially_declared_budgets_are_not_used(undeclared):
    """A half-specified problem must not silently mix declared and default."""
    p = MTOP()
    p.add_task(sphere, dim=3, budget=60)
    p.add_task(shifted, dim=4)
    assert resolve_budget(p, None, 2, default=11) == [11, 11]


def test_algorithm_consumes_the_declared_budget_end_to_end():
    from ddmtolab.Algorithms.MTMO.ParEGO_KT import ParEGO_KT

    def bi_objective(x):
        x = np.atleast_2d(x)
        return np.column_stack([np.sum(x ** 2, axis=1), np.sum((x - 1) ** 2, axis=1)])

    p = MTOP()
    p.add_task(bi_objective, dim=3, lower_bound=-2, upper_bound=2, budget=14)
    p.add_task(bi_objective, dim=4, lower_bound=-2, upper_bound=2, budget=18)
    algo = ParEGO_KT(problem=p, n_initial=[8, 10], n_weights=5,
                     save_data=False, disable_tqdm=True)
    res = algo.optimize()
    assert [int(n) for n in res.max_nfes] == [14, 18]


# --------------------------------------------------------------------------
# Batch experiment bookkeeping
# --------------------------------------------------------------------------

def test_batch_run_records_the_seed_of_every_run(tmp_path):
    from ddmtolab.Methods.batch_experiment import BatchExperiment
    from ddmtolab.Algorithms.STSO.GA import GA

    exp = BatchExperiment(base_path=str(tmp_path / 'Data'), clear_folder=True)
    exp.add_problem(make_sphere_problem, 'P1')
    exp.add_algorithm(GA, 'GA', n=10, max_nfes=40)
    records = exp.run(n_runs=3, verbose=False, max_workers=1, base_seed=42)

    assert sorted(r['Seed'] for r in records) == [42, 43, 44]
    assert all(r['Status'] == 'Success' for r in records)


def test_batch_run_without_a_base_seed_stays_unseeded(tmp_path):
    from ddmtolab.Methods.batch_experiment import BatchExperiment
    from ddmtolab.Algorithms.STSO.GA import GA

    exp = BatchExperiment(base_path=str(tmp_path / 'Data'), clear_folder=True)
    exp.add_problem(make_sphere_problem, 'P1')
    exp.add_algorithm(GA, 'GA', n=10, max_nfes=40)
    records = exp.run(n_runs=2, verbose=False, max_workers=1)

    assert all(r['Seed'] == '' for r in records)


def read_config(data_path):
    """Load the experiment configuration a batch wrote into its data folder."""
    import yaml

    with open(data_path / 'experiment_config.yaml', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def make_batch(data_path):
    from ddmtolab.Methods.batch_experiment import BatchExperiment
    from ddmtolab.Algorithms.STSO.GA import GA

    exp = BatchExperiment(base_path=str(data_path), clear_folder=False)
    exp.add_problem(make_sphere_problem, 'P1')
    exp.add_algorithm(GA, 'GA', n=10, max_nfes=40)
    return exp


def test_the_configuration_records_the_settings_of_the_run(tmp_path):
    data_path = tmp_path / 'Data'
    make_batch(data_path).run(n_runs=2, verbose=False, max_workers=1, base_seed=100)

    config = read_config(data_path)
    assert config['run_settings']['base_seed'] == 100
    assert config['run_settings']['n_runs'] == 2
    assert len(config['run_history']) == 1


def test_re_running_a_folder_keeps_the_settings_of_the_earlier_calls(tmp_path):
    # Completed runs are skipped on a re-run, so the folder ends up holding
    # results from both calls; a configuration describing only the second would
    # not account for the seeds the first one used
    data_path = tmp_path / 'Data'
    make_batch(data_path).run(n_runs=2, verbose=False, max_workers=1, base_seed=100)
    make_batch(data_path).run(n_runs=3, verbose=False, max_workers=1, base_seed=999)

    config = read_config(data_path)
    assert [(entry['base_seed'], entry['n_runs']) for entry in config['run_history']] == \
           [(100, 2), (999, 3)]
    # The latest call stays where anything reading the file expects it
    assert config['run_settings'] == config['run_history'][-1]


def test_clearing_the_folder_starts_a_fresh_history(tmp_path):
    from ddmtolab.Methods.batch_experiment import BatchExperiment
    from ddmtolab.Algorithms.STSO.GA import GA

    data_path = tmp_path / 'Data'
    make_batch(data_path).run(n_runs=1, verbose=False, max_workers=1, base_seed=100)

    exp = BatchExperiment(base_path=str(data_path), clear_folder=True)
    exp.add_problem(make_sphere_problem, 'P1')
    exp.add_algorithm(GA, 'GA', n=10, max_nfes=40)
    exp.run(n_runs=1, verbose=False, max_workers=1, base_seed=200)

    history = read_config(data_path)['run_history']
    assert [entry['base_seed'] for entry in history] == [200]


def test_a_configuration_written_before_the_history_existed_is_not_dropped(tmp_path):
    data_path = tmp_path / 'Data'
    data_path.mkdir(parents=True)
    (data_path / 'experiment_config.yaml').write_text(
        'created_at: x\nbase_path: y\nclear_folder: false\n'
        'problems: []\nalgorithms: []\n'
        'run_settings:\n  n_runs: 5\n  max_workers: 2\n  base_seed: 7\n'
        '  start_time: x\n',
        encoding='utf-8'
    )
    make_batch(data_path).run(n_runs=1, verbose=False, max_workers=1, base_seed=8)

    history = read_config(data_path)['run_history']
    assert [entry['base_seed'] for entry in history] == [7, 8]


def test_an_unreadable_configuration_does_not_stop_the_batch(tmp_path):
    data_path = tmp_path / 'Data'
    data_path.mkdir(parents=True)
    (data_path / 'experiment_config.yaml').write_text('{[ not yaml', encoding='utf-8')

    records = make_batch(data_path).run(n_runs=1, verbose=False, max_workers=1,
                                        base_seed=3)

    assert all(r['Status'] == 'Success' for r in records)
    assert [entry['base_seed'] for entry in read_config(data_path)['run_history']] == [3]


def test_the_configuration_still_loads_with_from_config(tmp_path):
    from ddmtolab.Methods.batch_experiment import BatchExperiment

    data_path = tmp_path / 'Data'
    make_batch(data_path).run(n_runs=1, verbose=False, max_workers=1, base_seed=100)
    make_batch(data_path).run(n_runs=2, verbose=False, max_workers=1, base_seed=200)

    loaded = BatchExperiment.from_config(str(data_path / 'experiment_config.yaml'))
    assert loaded._loaded_run_settings['base_seed'] == 200
    assert loaded._loaded_run_settings['n_runs'] == 2


def test_a_failing_run_is_recorded_without_stopping_the_batch(tmp_path):
    from ddmtolab.Methods.batch_experiment import BatchExperiment
    from ddmtolab.Algorithms.STSO.GA import GA

    exp = BatchExperiment(base_path=str(tmp_path / 'Data'), clear_folder=True)
    exp.add_problem(make_sphere_problem, 'P_ok')
    exp.add_problem(make_broken_problem, 'P_bad')
    exp.add_algorithm(GA, 'GA', n=10, max_nfes=40)
    records = exp.run(n_runs=1, verbose=False, max_workers=1)

    by_problem = {r['Problem']: r for r in records}
    assert by_problem['P_ok']['Status'] == 'Success'
    assert by_problem['P_bad']['Status'] == 'Failed'
    assert 'evaluator unavailable' in by_problem['P_bad']['Error']

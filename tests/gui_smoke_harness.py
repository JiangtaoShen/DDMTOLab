"""GUI smoke-test harness for DDMTOLab.

Runs every algorithm discovered by the GUI scanners on multiple benchmark
problems, through the exact code path the GUI Test Mode uses
(ui/utils/registry + ui/utils/algo_scanner), with tiny budgets.

Usage:
    python tests/gui_smoke_harness.py --matrix STSO --out results_stso.jsonl
    python tests/gui_smoke_harness.py --matrix RWO-MTSO --out results_rwo_mtso.jsonl
"""
import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ui'))

import matplotlib
matplotlib.use('Agg')

import numpy as np

from utils.registry import create_problem, get_algorithm_class, get_algorithm_names
from utils.algo_scanner import get_algorithm_params_from_scan, get_algorithm_info
from utils.compat import check_algorithm_compatibility

# Problem matrix: matrix key -> (algo category, [(prob_cat, suite, method, kwargs), ...])
MATRIX = {
    'STSO': ('STSO', [
        ('STSO', 'CLASSICALSO', 'P1', {'D': 8}),
        ('STSO', 'CEC10_CSO', 'P1', {'D': 10}),
        ('STSO', 'STSOtest', 'P2', {'D': 8}),
    ]),
    'STMO': ('STMO', [
        ('STMO', 'ZDT', 'ZDT1', {'D': 8}),
        ('STMO', 'DTLZ', 'DTLZ2', {'M': 3, 'D': 8}),
        ('STMO', 'WFG', 'WFG4', {'M': 3, 'Kp': 4, 'D': 10}),
    ]),
    'MTSO': ('MTSO', [
        ('MTSO', 'CEC17MTSO', 'P1', {}),
        ('MTSO', 'CMT', 'CMT1', {'D': 10}),
        ('MTSO', 'CEC19MaTSO', 'P1', {'K': 3}),
    ]),
    'MTMO': ('MTMO', [
        ('MTMO', 'CEC17MTMO', 'P1', {}),
        ('MTMO', 'MTMO_DTLZ', 'P1', {'M': 3, 'D': 10}),
        ('MTMO', 'CEC19MTMO', 'P1', {}),
    ]),
    # The GUI's RWO problem category runs MTSO / MTMO algorithms
    'RWO-MTSO': ('MTSO', [
        ('RWO', 'PEPVM', 'P1', {}),   # K=3, unequal dims (5/7/5)
        ('RWO', 'SCP', 'P1', {}),     # K=11, dims 75-105
    ]),
    'RWO-MTMO': ('MTMO', [
        ('RWO', 'MO_SCP', 'P1', {'K': 2}),
        ('RWO', 'SOPM', 'P1', {}),    # 2-objective, K=3
    ]),
}

INIT_PARAM_NAMES = ('n_initial', 'n_init', 'initial_size', 'init_size', 'n_samples', 'n0')

# Neural/generative algorithms train models per generation; clamp their
# training knobs so a smoke run finishes in seconds instead of hours on CPU.
TRAIN_PARAM_CLAMPS = {
    'train_epochs': 2, 'distill_epochs': 2, 'epochs': 2, 'n_epochs': 2,
    'n_diffusion_steps': 10, 'batch_size': 64, 'base_ch': 8,
}


def budget_overrides(algo_cat: str, algo_name: str, params: dict, max_dim: int) -> dict:
    """Small budgets mimicking what a user would type for a quick run.

    Surrogate-based algorithms need at least ~D+1 initial samples, so the
    initial-sample size is pinned explicitly and max_nfes sits just above it.
    """
    info = get_algorithm_info(algo_cat, algo_name)
    expensive = str(info.get('expensive', 'False')).lower() == 'true'
    overrides = {}
    if 'max_nfes' in params:
        if expensive:
            init_param = next((p for p in INIT_PARAM_NAMES if p in params), None)
            default_init = 0
            if init_param:
                d = params[init_param].get('default')
                if isinstance(d, (int, float)) and d:
                    default_init = int(d)
            init = max(default_init, max_dim + 2, 20)
            if init_param:
                overrides[init_param] = init
            overrides['max_nfes'] = init + 5
        else:
            overrides['max_nfes'] = 500

    for pname, clamp in TRAIN_PARAM_CLAMPS.items():
        if pname in params:
            default = params[pname].get('default')
            overrides[pname] = min(default, clamp) if isinstance(default, (int, float)) else clamp

    return overrides


def default_param_values(params: dict) -> dict:
    """Default values as the GUI presents them (None values are not passed)."""
    values = {}
    for pname, pinfo in params.items():
        default = pinfo.get('default')
        if default is not None:
            values[pname] = default
    return values


def validate_result(result, problem) -> str:
    """Return '' if the Results object looks sane, else a description."""
    if result is None:
        return 'optimize() returned None'
    if isinstance(result, tuple):
        result = result[0]
    best_objs = getattr(result, 'best_objs', None)
    if best_objs is None:
        return 'Results has no best_objs'
    n_tasks = len(problem.dims)
    if len(best_objs) != n_tasks:
        return f'best_objs has {len(best_objs)} entries, expected {n_tasks}'
    for i, obj in enumerate(best_objs):
        if obj is None:
            return f'best_objs[{i}] is None'
        arr = np.asarray(obj, dtype=float)
        if arr.size == 0:
            return f'best_objs[{i}] is empty'
        if not np.all(np.isfinite(arr)):
            return f'best_objs[{i}] contains non-finite values'
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True, choices=sorted(MATRIX.keys()))
    parser.add_argument('--out', required=True)
    parser.add_argument('--only', default='', help='comma-separated algorithm names to include')
    parser.add_argument('--skip', default='', help='comma-separated algorithm names to skip')
    parser.add_argument('--skip-combos', default='',
                        help='comma-separated ALGO:SUITE pairs to skip (too slow for smoke)')
    parser.add_argument('--save-dir', default='', help='save_data dir (empty = save_data False)')
    args = parser.parse_args()

    algo_cat, problems = MATRIX[args.matrix]
    algos = get_algorithm_names(algo_cat)
    if args.only:
        wanted = {a.strip() for a in args.only.split(',') if a.strip()}
        algos = [a for a in algos if a in wanted]
    if args.skip:
        skipped = {a.strip() for a in args.skip.split(',') if a.strip()}
        algos = [a for a in algos if a not in skipped]
    skip_combos = set()
    for pair in args.skip_combos.split(','):
        if ':' in pair:
            a, s = pair.split(':', 1)
            skip_combos.add((a.strip(), s.strip()))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(algos) * len(problems)
    n_done = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for algo_name in algos:
            params = get_algorithm_params_from_scan(algo_cat, algo_name)
            base_values = default_param_values(params)

            for prob_cat, suite, method, prob_kwargs in problems:
                n_done += 1
                rec = {
                    'matrix': args.matrix, 'algo_cat': algo_cat, 'algo': algo_name,
                    'prob_cat': prob_cat, 'suite': suite, 'method': method,
                    'prob_kwargs': prob_kwargs,
                }
                if (algo_name, suite) in skip_combos:
                    rec['status'] = 'skipped-slow'
                    rec['elapsed'] = 0.0
                    out.write(json.dumps(rec) + '\n')
                    out.flush()
                    print(f'[{n_done}/{n_total}] {algo_name} on {suite}/{method} skipped (slow)',
                          flush=True)
                    continue
                print(f'[{n_done}/{n_total}] {algo_name} on {suite}/{method} ...',
                      flush=True)
                t0 = time.time()
                capture = io.StringIO()
                try:
                    problem = create_problem(prob_cat, suite, method, **prob_kwargs)
                    max_dim = int(max(problem.dims))
                    values = dict(base_values)
                    values.update(budget_overrides(algo_cat, algo_name, params, max_dim))
                    rec['algo_params'] = dict(values)

                    # Same fail-fast check the GUI applies before running
                    info = get_algorithm_info(algo_cat, algo_name)
                    issues = check_algorithm_compatibility(problem, info)
                    if issues:
                        rec['status'] = 'incompatible'
                        rec['error'] = '; '.join(issues)
                        rec['elapsed'] = round(time.time() - t0, 2)
                        out.write(json.dumps(rec) + '\n')
                        out.flush()
                        print(f'    -> incompatible: {rec["error"][:120]}', flush=True)
                        continue

                    algo_cls = get_algorithm_class(algo_cat, algo_name)
                    safe_name = algo_name.replace('/', '-')
                    algo_kwargs = dict(problem=problem, disable_tqdm=True, name=safe_name)
                    if args.save_dir:
                        algo_kwargs.update(save_data=True, save_path=args.save_dir)
                    else:
                        algo_kwargs.update(save_data=False)
                    algo_kwargs.update(values)

                    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                        algo = algo_cls(**algo_kwargs)
                        result = algo.optimize()

                    issue = validate_result(result, problem)
                    rec['status'] = 'ok' if not issue else 'invalid'
                    if issue:
                        rec['error'] = issue
                except Exception as e:
                    rec['status'] = 'error'
                    rec['error'] = f'{type(e).__name__}: {e}'
                    rec['traceback'] = traceback.format_exc()[-3000:]
                finally:
                    rec['elapsed'] = round(time.time() - t0, 2)
                    stdout_tail = capture.getvalue()[-500:]
                    if rec.get('status') != 'ok' and stdout_tail:
                        rec['stdout_tail'] = stdout_tail

                out.write(json.dumps(rec) + '\n')
                out.flush()
                print(f'    -> {rec["status"]} ({rec["elapsed"]}s)'
                      + (f' {rec.get("error", "")[:120]}' if rec.get('error') else ''),
                      flush=True)

    print('DONE', flush=True)


if __name__ == '__main__':
    main()

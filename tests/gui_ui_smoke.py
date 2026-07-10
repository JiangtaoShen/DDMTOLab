"""Launch the real DDMTOLab GUI and drive it programmatically.

Covers: window construction, category/suite sweeps on both tabs, algorithm
panel interactions (add / rename / reorder / remove), an end-to-end Test Mode
run (GA + DE), an end-to-end Batch Experiment run with analysis, config
save/load round trip, and animation generation.

Run:
    python tests/gui_ui_smoke.py
Exits 0 on success, 1 if any step failed.
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'ui'))
sys.path.insert(0, str(ROOT / 'src'))

ERRORS = []
DONE = {"stopped": False}
T0 = time.time()
TIME_CAP = 600  # seconds


def step(name, fn):
    try:
        fn()
        print(f'[ok] {name}', flush=True)
        return True
    except Exception as e:
        ERRORS.append({'step': name, 'error': f'{e}', 'traceback': traceback.format_exc()[-2000:]})
        print(f'[FAIL] {name}: {e}', flush=True)
        return False


def run_smoke():
    import dearpygui.dearpygui as dpg
    import main as ui_main
    from pages import test_mode, batch_mode
    from utils.registry import get_problem_suites, get_problem_methods, get_algorithm_names

    def schedule(fn, delay_frames=30):
        dpg.set_frame_callback(dpg.get_frame_count() + delay_frames, lambda *a: fn())

    def finish():
        if not DONE["stopped"]:
            DONE["stopped"] = True
            dpg.stop_dearpygui()

    def check_time():
        if time.time() - T0 > TIME_CAP:
            ERRORS.append({'step': 'overall', 'error': f'time cap {TIME_CAP}s exceeded'})
            finish()
            return True
        return False

    # ---------- phase steps ----------
    def sweep_test_tab():
        for cat in ["STSO", "STMO", "MTSO", "MTMO", "RWO"]:
            dpg.set_value("test_prob_cat_combo", cat)
            test_mode._on_prob_category_change(None, cat)
            for suite in get_problem_suites(cat):
                dpg.set_value("test_suite_combo", suite)
                test_mode._on_suite_change(None, suite)

    def sweep_test_algo_panel():
        for cat in ["STSO", "STMO", "MTSO", "MTMO"]:
            dpg.set_value("test_algo_cat_combo", cat)
            test_mode._on_algo_category_change(None, cat)
            algos = get_algorithm_names(cat)
            assert algos, f'no algorithms discovered for {cat}'
            for a in algos[:2]:
                test_mode._on_algo_click(None, None, a)
            sel = test_mode._state["selected_algos"]
            assert len(sel) == 2, f'expected 2 selected algos, got {len(sel)}'
            test_mode._move_algo_down(None, None, sel[0]["id"])
            test_mode._on_algo_name_change(sel[0]["id"], sel[0]["algo_name"] + "_renamed")
            test_mode._remove_algo(None, None, sel[-1]["id"])
            assert len(test_mode._state["selected_algos"]) == 1

    def sweep_batch_tab():
        for cat in ["STSO", "STMO", "MTSO", "MTMO", "RWO"]:
            dpg.set_value("batch_prob_cat_combo", cat)
            batch_mode._on_prob_category_change(None, cat)
            for suite in get_problem_suites(cat)[:2]:
                dpg.set_value("batch_suite_combo", suite)
                batch_mode._on_suite_change(None, suite)
        # problem panel interactions
        dpg.set_value("batch_prob_cat_combo", "STSO")
        batch_mode._on_prob_category_change(None, "STSO")
        dpg.set_value("batch_suite_combo", "CLASSICALSO")
        batch_mode._on_suite_change(None, "CLASSICALSO")
        methods = get_problem_methods("STSO", "CLASSICALSO")
        batch_mode._on_prob_click(None, None, ("CLASSICALSO", methods[0]))
        batch_mode._on_prob_click(None, None, ("CLASSICALSO", methods[1]))
        batch_mode._on_prob_name_change(0, "First")
        batch_mode._move_prob_down(None, None, 0)
        assert batch_mode._state["prob_names"].get(1) == "First"
        batch_mode._move_prob_up(None, None, 1)
        batch_mode._remove_prob(None, None, 0)
        assert len(batch_mode._state["selected_probs"]) == 1
        batch_mode._remove_prob(None, None, 0)
        # algorithm panel interactions
        batch_mode._on_algo_click(None, None, "GA")
        batch_mode._on_algo_click(None, None, "DE")
        batch_mode._on_algo_name_change(0, "GA_custom")
        batch_mode._move_algo_down(None, None, 0)
        assert batch_mode._state["algo_names"].get(1) == "GA_custom"
        batch_mode._remove_algo(None, None, 0)
        assert batch_mode._state["selected_algos"] == ["GA"]
        batch_mode._remove_algo(None, None, 0)
        batch_mode._state["algo_names"] = {}

    def start_test_run():
        # Start from a clean workspace so later phases (animation) only see
        # this run's data, independent of what previous sessions left behind
        test_mode._state["backup_manager"].clean_without_backup()
        dpg.set_value("test_prob_cat_combo", "STSO")
        test_mode._on_prob_category_change(None, "STSO")
        dpg.set_value("test_suite_combo", "CLASSICALSO")
        test_mode._on_suite_change(None, "CLASSICALSO")
        dpg.set_value("test_method_combo", "P1")
        dpg.set_value("test_D_input", 6)
        dpg.set_value("test_algo_cat_combo", "STSO")
        test_mode._on_algo_category_change(None, "STSO")
        test_mode._deselect_all_algos(None, None)
        for name in ("GA", "DE"):
            test_mode._on_algo_click(None, None, name)
        for entry in test_mode._state["selected_algos"]:
            for pname, val in (("n", 20), ("max_nfes", 300)):
                tag = f"test_param_{entry['id']}_{pname}"
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, val)
        test_mode._run_clicked(None, None)
        assert test_mode._state["running"] or test_mode._state.get("worker_done"), \
            'test run did not start'

    def verify_test_run():
        statuses = test_mode._state.get("statuses", {})
        assert statuses, 'no statuses recorded'
        for name, st in statuses.items():
            assert not st.error, f'{name} failed: {st.error}'
        assert test_mode._state.get("analysis_error") is None, \
            f'analysis failed: {test_mode._state.get("analysis_error")}'
        results_path = Path(test_mode._state["file_manager"].get_results_path_str())
        assert list(results_path.glob("*.xlsx")), 'no Excel table generated'
        assert list(results_path.glob("*.png")), 'no figures generated'

    def start_batch_run():
        dpg.set_value("batch_prob_cat_combo", "STSO")
        batch_mode._on_prob_category_change(None, "STSO")
        dpg.set_value("batch_suite_combo", "CLASSICALSO")
        batch_mode._on_suite_change(None, "CLASSICALSO")
        batch_mode._state["selected_probs"] = []
        batch_mode._state["prob_names"] = {}
        batch_mode._state["prob_params"] = {}
        batch_mode._on_prob_click(None, None, ("CLASSICALSO", "P1"))
        batch_mode._on_prob_click(None, None, ("CLASSICALSO", "P2"))
        # set problem D via widgets
        for i, (suite, method) in enumerate(batch_mode._state["selected_probs"]):
            tag = f"batch_prob_param_{suite}_{method}_{i}_D"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, 6)
                batch_mode._on_prob_param_change(i, "D", 6)
        batch_mode._state["selected_algos"] = []
        batch_mode._state["algo_names"] = {}
        batch_mode._on_algo_click(None, None, "GA")
        batch_mode._on_algo_click(None, None, "DE")
        for i, algo in enumerate(batch_mode._state["selected_algos"]):
            for pname, val in (("n", 20), ("max_nfes", 200)):
                tag = f"batch_algo_param_{algo}_{i}_{pname}"
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, val)
        dpg.set_value("batch_nruns_input", 2)
        dpg.set_value("batch_workers_input", 2)
        batch_mode._run_clicked(None, None)
        st = batch_mode._state.get("status")
        assert st is not None and (st.running or st.finished), 'batch run did not start'

    def verify_batch_run():
        st = batch_mode._state.get("status")
        assert st is not None and st.finished, 'batch status not finished'
        assert not st.error, f'batch failed: {st.error}'
        results_path = Path(batch_mode._state["file_manager"].get_results_path_str())
        assert list(results_path.glob("*.xlsx")), 'no batch Excel table'
        assert list(results_path.glob("*.png")), 'no batch figures'

    def config_round_trip():
        batch_mode._save_config(None, None)
        cfg = Path(batch_mode._state["file_manager"].base_path) / "experiment_config.yaml"
        assert cfg.exists(), 'config not saved'
        batch_mode._load_config_from_file(str(cfg))
        assert batch_mode._state["selected_algos"] == ["GA", "DE"], \
            f'config load mismatch: {batch_mode._state["selected_algos"]}'
        assert len(batch_mode._state["selected_probs"]) == 2

    def start_animation():
        dpg.set_value = dpg.set_value  # no-op guard
        test_mode._animation_clicked(None, None)
        assert dpg.does_item_exist("animation_modal"), 'animation modal missing'
        dpg.set_value("anim_nfes_input", 300)
        dpg.set_value("anim_format_combo", "gif")
        test_mode._run_animation_generation(None, None)
        assert test_mode._state.get("_anim_thread") is not None, 'animation thread not started'

    def verify_animation():
        err = test_mode._state.get("_anim_error")
        assert err is None, f'animation failed: {err}'
        results_path = Path(test_mode._state["file_manager"].get_results_path_str())
        gifs = list(results_path.glob("*.gif"))
        assert gifs, 'no gif generated'

    # ---------- phase chain ----------
    def phase_start():
        step('test-tab category/suite sweep', sweep_test_tab)
        step('test-tab algorithm panel', sweep_test_algo_panel)
        step('batch-tab sweeps and panels', sweep_batch_tab)
        if not step('start test-mode run (GA+DE)', start_test_run):
            schedule(phase_batch, 10)
            return
        schedule(wait_test_done, 30)

    def wait_test_done():
        if check_time():
            return
        if test_mode._state.get("results_displayed"):
            step('verify test-mode run + analysis', verify_test_run)
            schedule(phase_animation, 10)
        else:
            schedule(wait_test_done, 30)

    def phase_animation():
        if not step('start animation generation', start_animation):
            schedule(phase_batch, 10)
            return
        schedule(wait_animation_done, 30)

    def wait_animation_done():
        if check_time():
            return
        if test_mode._state.get("_anim_thread") is None:
            step('verify animation output', verify_animation)
            schedule(phase_batch, 10)
        else:
            schedule(wait_animation_done, 30)

    def phase_batch():
        if not step('start batch run (GA+DE x P1,P2 x2)', start_batch_run):
            schedule(phase_config, 10)
            return
        schedule(wait_batch_done, 60)

    def wait_batch_done():
        if check_time():
            return
        if batch_mode._state.get("displayed"):
            step('verify batch run + analysis', verify_batch_run)
            schedule(phase_config, 10)
        else:
            schedule(wait_batch_done, 30)

    def phase_config():
        step('config save/load round trip', config_round_trip)
        finish()

    def on_ready():
        schedule(phase_start, 5)

    ui_main.main(smoke_frames=10 ** 9, on_ready=on_ready)


def main():
    run_smoke()

    print('\n===== GUI SMOKE SUMMARY =====')
    if ERRORS:
        print(f'{len(ERRORS)} step(s) FAILED:')
        for e in ERRORS:
            print(f"  - {e['step']}: {e['error']}")
    else:
        print('All steps passed.')

    import os
    import tempfile
    out_dir = os.environ.get('DDMTOLAB_SMOKE_OUT', tempfile.gettempdir())
    out = Path(out_dir) / 'gui_ui_smoke_result.json'
    out.write_text(json.dumps({'errors': ERRORS, 'elapsed': round(time.time() - T0, 1)},
                              indent=2), encoding='utf-8')
    print(f'Result written to {out}')
    return 1 if ERRORS else 0


if __name__ == '__main__':
    sys.exit(main())

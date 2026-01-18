from Methods.batch_experiment import BatchExperiment
from Methods.data_analysis import DataAnalyzer
from Problems.STMO.DTLZ import DTLZ, SETTINGS
from Algorithms.STMO.KRVEA import KRVEA


if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = DTLZ()

    batch_exp.add_problem(prob.DTLZ2, 'DTLZ2', dim=10)

    batch_exp.add_algorithm(KRVEA, 'K-RVEA', n_initial=50, max_nfes=200, disable_tqdm=False)

    batch_exp.run(n_runs=1, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=SETTINGS,
        algorithm_order=None,
        save_path='./Results',
        table_format='excel',
        figure_format='png',
        statistic_type='mean',
        significance_level=0.05,
        rank_sum_test=True,
        log_scale=True,
        show_pf=True,
        show_nd=True,
        best_so_far=True,
        clear_results=True
    )

    results = analyzer.run()
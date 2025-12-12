from Methods.batch_experiment import BatchExperiment
from Algorithms.STMO.NSGAII import NSGAII
from Algorithms.STMO.RVEA import RVEA
from Algorithms.MTMO.MOMFEA import MOMFEA
from Problems.STMO.ZDT import ZDT, SETTINGS
from Methods.data_analysis import DataAnalyzer


if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    zdt = ZDT()

    batch_exp.add_problem(zdt.ZDT5, 'ZDT5')



    n = 100
    max_nfes = 10000
    batch_exp.add_algorithm(NSGAII, 'NSGA-II', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(RVEA, 'RVEA', n=n, max_nfes=max_nfes)
    # batch_exp.add_algorithm(MOMFEA, 'MO-MFEA', n=n, max_nfes=max_nfes)

    batch_exp.run(n_runs=2, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=SETTINGS,
        algorithm_order=['NSGA-II', 'RVEA'],
        save_path='./Results',
        table_format='latex',
        figure_format='pdf',
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
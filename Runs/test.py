from Methods.batch_experiment import BatchExperiment
from Methods.data_analysis import DataAnalyzer
from Algorithms.MTSO.MFEA import MFEA
from Algorithms.STSO.GA import GA
from Algorithms.STSO.CMAES import CMAES
from Problems.RWP.pkacp import PKACP

if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = PKACP()

    batch_exp.add_problem(prob.P1, 'PKACP', task_num=5)


    # batch_exp.add_algorithm(MFEA, 'MFEA', n=100, max_nfes=10000, disable_tqdm=False)
    batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=4000)
    # batch_exp.add_algorithm(CMAES, 'CMA-ES', n=100, max_nfes=10000)

    batch_exp.run(n_runs=5, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=None,
        algorithm_order=None,
        save_path='./Results',
        table_format='excel',
        figure_format='png',
        statistic_type='mean',
        significance_level=0.05,
        rank_sum_test=True,
        log_scale=False,
        show_pf=True,
        show_nd=True,
        best_so_far=True,
        clear_results=True
    )

    results = analyzer.run()
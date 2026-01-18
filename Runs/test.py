from Methods.batch_experiment import BatchExperiment
from Methods.data_analysis import DataAnalyzer
from Problems.MTSO.cec17_mtso import CEC17MTSO
from Algorithms.MTSO.GMFEA import GMFEA
from Algorithms.STSO.GA import GA


if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = CEC17MTSO()
    batch_exp.add_problem(prob.P1, 'P1')
    batch_exp.add_problem(prob.P2, 'P2')
    batch_exp.add_problem(prob.P3, 'P3')
    batch_exp.add_problem(prob.P4, 'P4')


    batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=30000)
    batch_exp.add_algorithm(GMFEA, 'G-MFEA', n=100, max_nfes=30000)

    batch_exp.run(n_runs=2, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        algorithm_order=None,
        save_path='./Results',
        table_format='excel',
        figure_format='png',
        statistic_type='mean',
        significance_level=0.05,
        rank_sum_test=True,
        log_scale=True,
        best_so_far=True,
        clear_results=True
    )

    results = analyzer.run()
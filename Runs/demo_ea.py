from Methods.batch_experiment import BatchExperiment
from Methods.data_analysis import DataAnalyzer
from Algorithms.STSO.GA import GA
from Algorithms.STSO.PSO import PSO
from Algorithms.STSO.DE import DE
from Algorithms.MTSO.EMEA import EMEA
from Algorithms.MTSO.MFEA import MFEA
from Problems.MTSO.cec17_mtso import CEC17MTSO

if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    cec17mtso = CEC17MTSO()

    batch_exp.add_problem(cec17mtso.P1, 'P1')
    batch_exp.add_problem(cec17mtso.P2, 'P2')
    batch_exp.add_problem(cec17mtso.P3, 'P3')
    # batch_exp.add_problem(cec17mtso.P4, 'P4')
    # batch_exp.add_problem(cec17mtso.P5, 'P5')
    # batch_exp.add_problem(cec17mtso.P6, 'P6')
    # batch_exp.add_problem(cec17mtso.P7, 'P7')
    # batch_exp.add_problem(cec17mtso.P8, 'P8')
    # batch_exp.add_problem(cec17mtso.P9, 'P9')

    batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=20000)
    batch_exp.add_algorithm(DE, 'DE', n=100, max_nfes=20000)
    batch_exp.add_algorithm(PSO, 'PSO', n=100, max_nfes=20000)
    batch_exp.add_algorithm(MFEA, 'MFEA', n=100, max_nfes=20000)
    batch_exp.add_algorithm(EMEA, 'EMEA', n=100, max_nfes=20000)

    batch_exp.run(n_runs=2, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=None,
        algorithm_order=['GA', 'DE', 'PSO', 'EMEA', 'MFEA'],
        save_path='./Results',
        table_format='latex',
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

# if __name__ == '__main__':
#     batch_exp = BatchExperiment.from_config('./Data/experiment_config.yaml')
#     batch_exp.run()
#
#     data_analyzer(path='./Data', algorithm_order=None, settings=None,
#                   best_so_far=True, table_type='latex', figure_type='pdf',
#                   statistic_type='mean', significance_level=0.05,
#                   rank_sum_test=True, log_scale=False, show_pf=True,
#                   save_path='./Results', clear_results=True)
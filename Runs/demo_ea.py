from Methods.batch_experiment import BatchExperiment
from Methods.data_analysis import DataAnalyzer
from Algorithms.STSO.GA import GA
from Algorithms.STSO.PSO import PSO
from Algorithms.STSO.CSO import CSO
from Algorithms.STSO.DE import DE
from Algorithms.MTSO.EMEA import EMEA
from Algorithms.MTSO.LCBEMT import LCBEMT
from Algorithms.MTSO.MFEA import MFEA
from Algorithms.MTSO.MFEAII import MFEAII
from Algorithms.MTSO.GMFEA import GMFEA
from Algorithms.MTSO.EBS import EBS
from Algorithms.MTSO.MTEAAD import MTEAAD
from Algorithms.STSO.CMAES import CMAES
from Algorithms.STSO.GWO import GWO
from Algorithms.STSO.EO import EO
from Algorithms.STSO.KLPSO import KLPSO
from Algorithms.STSO.SLPSO import SLPSO
from Problems.MTSO.cec17_mtso import CEC17MTSO
from Problems.MTSO.cec19_matso import CEC19MaTSO
from Problems.MTSO.stop import STOP

if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = STOP()

    batch_exp.add_problem(prob.STOP1, 'STOP1', task_num=5)
    batch_exp.add_problem(prob.STOP2, 'STOP2', task_num=5)



    batch_exp.add_algorithm(MFEA, 'MFEA', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(EMEA, 'EMEA', n=100, max_nfes=20000)
    batch_exp.add_algorithm(MTEAAD, 'MTEA-AD', n=100, max_nfes=20000)
    # batch_exp.add_algorithm(LCBEMT, 'LCB-EMT', n=100, max_nfes=10000, disable_tqdm=False)
    # batch_exp.add_algorithm(MFEAII, 'MFEA-II', n=100, max_nfes=10000)
    # batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=20000)
    # batch_exp.add_algorithm(GMFEA, 'G-MFEA', n=100, max_nfes=10000)
    # batch_exp.add_algorithm(CMAES, 'CMA-ES', n=20, max_nfes=10000)
    # batch_exp.add_algorithm(EBS, 'EBS', n=20, max_nfes=10000, gen_init = 0)

    batch_exp.run(n_runs=2, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=None,
        algorithm_order=['MFEA', 'MTEA-AD'],
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

# if __name__ == '__main__':
#     batch_exp = BatchExperiment.from_config('./Data/experiment_config.yaml')
#     batch_exp.run()
#
#     data_analyzer(path='./Data', algorithm_order=None, settings=None,
#                   best_so_far=True, table_type='latex', figure_type='pdf',
#                   statistic_type='mean', significance_level=0.05,
#                   rank_sum_test=True, log_scale=False, show_pf=True,
#                   save_path='./Results', clear_results=True)
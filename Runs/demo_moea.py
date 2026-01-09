from Methods.batch_experiment import BatchExperiment
from Algorithms.STMO.NSGAII import NSGAII
from Algorithms.STMO.TwoArch2 import TwoArch2
from Algorithms.STMO.IBEA import IBEA
from Algorithms.STMO.MOEAD import MOEAD
from Algorithms.STMO.MOEADD import MOEADD
from Algorithms.STMO.NSGAIISDR import NSGAIISDR
from Algorithms.MTMO.MOMFEAII import MOMFEAII
from Algorithms.STMO.CTAEA import CTAEA
from Algorithms.STMO.MOEADFRRMAB import MOEADFRRMAB
from  Algorithms.STMO.MOEADSTM import MOEADSTM
from Algorithms.STMO.MCEAD import MCEAD
from Methods.data_analysis import DataAnalyzer
from Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS


if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = CEC17MTMO()

    batch_exp.add_problem(prob.P1, 'P1')
    batch_exp.add_problem(prob.P2, 'P2')
    batch_exp.add_problem(prob.P3, 'P3')
    batch_exp.add_problem(prob.P4, 'P4')


    batch_exp.add_algorithm(NSGAII, 'NSGA-II', n=50, max_nfes=500, disable_tqdm=False)
    batch_exp.add_algorithm(MCEAD, 'MCEA-D', n=50, max_nfes=500, disable_tqdm=False)



    batch_exp.run(n_runs=2, verbose=True, max_workers=8)

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
        log_scale=False,
        show_pf=True,
        show_nd=True,
        best_so_far=True,
        clear_results=True
    )

    results = analyzer.run()
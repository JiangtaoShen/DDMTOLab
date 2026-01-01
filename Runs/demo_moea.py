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
from Algorithms.STMO.CCMO import CCMO
from Methods.data_analysis import DataAnalyzer
from Problems.STMO.UF import UF, SETTINGS


if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = UF()

    batch_exp.add_problem(prob.UF1, 'UF1')
    batch_exp.add_problem(prob.UF2, 'UF2')
    batch_exp.add_problem(prob.UF3, 'UF3')
    batch_exp.add_problem(prob.UF4, 'UF4')
    batch_exp.add_problem(prob.UF5, 'UF5')
    batch_exp.add_problem(prob.UF6, 'UF6')
    batch_exp.add_problem(prob.UF7, 'UF7')
    batch_exp.add_problem(prob.UF8, 'UF8')


    n = 100
    max_nfes = 50000
    batch_exp.add_algorithm(NSGAII, 'NSGA-II', n=n, max_nfes=max_nfes, disable_tqdm=False)
    batch_exp.add_algorithm(MOEAD, 'MOEAD', n=n, max_nfes=max_nfes, disable_tqdm=False)
    batch_exp.add_algorithm(MOEADFRRMAB, 'MOEAD-FRRMAB', n=n, max_nfes=max_nfes, disable_tqdm=False)
    batch_exp.add_algorithm(MOEADSTM, 'MOEAD-STM', n=n, max_nfes=max_nfes, disable_tqdm=False)
    # batch_exp.add_algorithm(CCMO, 'CCMO', n=n, max_nfes=max_nfes, disable_tqdm=False)


    batch_exp.run(n_runs=4, verbose=True, max_workers=8)

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
from Methods.batch_experiment import BatchExperiment
from Algorithms.STMO.NSGAII import NSGAII
from Algorithms.STMO.TwoArch2 import TwoArch2
from Algorithms.STMO.IBEA import IBEA
from Algorithms.STMO.MOEAD import MOEAD
from Algorithms.STMO.MOEADD import MOEADD
from Algorithms.STMO.NSGAIISDR import NSGAIISDR
from Algorithms.MTMO.MOMFEA import MOMFEA
from Algorithms.MTMO.MOMFEAII import MOMFEAII
from Problems.STMO.UF import UF, SETTINGS
from Methods.data_analysis import DataAnalyzer


if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = UF()

    batch_exp.add_problem(prob.UF1, 'UF1')
    batch_exp.add_problem(prob.UF2, 'UF2')
    batch_exp.add_problem(prob.UF3, 'UF3')

    n = 100
    max_nfes = 20000
    batch_exp.add_algorithm(NSGAII, 'NSGA-II', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(MOEADD, 'MOEADD', n=n, max_nfes=max_nfes)
    # batch_exp.add_algorithm(NSGAIISDR, 'NSGA-II-SDR', n=n, max_nfes=max_nfes)
    # batch_exp.add_algorithm(TwoArch2, 'TwoArch2', n=n, max_nfes=max_nfes)
    # batch_exp.add_algorithm(IBEA, 'IBEA', n=n, max_nfes=max_nfes)
    # batch_exp.add_algorithm(MOEAD, 'MOEAD', n=n, max_nfes=max_nfes)
    # batch_exp.add_algorithm(MOMFEA, 'MO-MFEA', n=n, max_nfes=max_nfes)
    # batch_exp.add_algorithm(MOMFEAII, 'MO-MFEA-II', n=n, max_nfes=max_nfes)

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
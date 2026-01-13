from Methods.batch_experiment import BatchExperiment
from Algorithms.STMO.NSGAII import NSGAII
from Algorithms.STMO.TwoArch2 import TwoArch2
from Algorithms.STMO.IBEA import IBEA
from Algorithms.STMO.MOEAD import MOEAD
from Algorithms.STMO.MOEADD import MOEADD
from Algorithms.STMO.NSGAIISDR import NSGAIISDR
from Algorithms.MTMO.MOMFEAII import MOMFEAII
from Algorithms.STMO.CTAEA import CTAEA
from Algorithms.STMO.CCMO import CCMO
from Algorithms.STMO.MOEADFRRMAB import MOEADFRRMAB
from  Algorithms.STMO.MOEADSTM import MOEADSTM
from Algorithms.STMO.MCEAD import MCEAD
from Algorithms.STMO.ParEGO import ParEGO
from Algorithms.STMO.KRVEA import KRVEA
from Algorithms.STMO.KTA2 import KTA2
from Algorithms.STMO.DSAEAPS import DSAEAPS
from Algorithms.MTMO.MOMFEA import MOMFEA
from Algorithms.MTMO.MaOMFEA import MaOMFEA
from Methods.data_analysis import DataAnalyzer
from Problems.STMO.CF import CF, SETTINGS
# from Problems.MTMO.mtmo_instance import MTMOInstances, SETTINGS
# from Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS
# from Problems.STMO.DTLZ import DTLZ, SETTINGS

if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = CF()

    batch_exp.add_problem(prob.CF4, 'CF4')
    batch_exp.add_problem(prob.CF5, 'CF5')
    batch_exp.add_problem(prob.CF6, 'CF6')

    batch_exp.add_algorithm(NSGAII, 'NSGA-II', n=100, max_nfes=20000, disable_tqdm=False)
    batch_exp.add_algorithm(CCMO, 'CCMO', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(CTAEA, 'C-TAEA', n=100, max_nfes=100000, disable_tqdm=False)
    # batch_exp.add_algorithm(MOMFEA, 'MO-MFEA', n=100, max_nfes=12500, disable_tqdm=False)
    # batch_exp.add_algorithm(MaOMFEA, 'MaO-MFEA', n=100, max_nfes=20000, disable_tqdm=False)

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
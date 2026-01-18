from Algorithms.MTMO.EMTET import EMTET
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
from Algorithms.MTMO.MOEMEA import MOEMEA
from Algorithms.MTMO.MTEADDN import MTEADDN
from Algorithms.MTMO.MTDEMKTA import MTDEMKTA
from Algorithms.MTMO.EMTET import EMTET
from Algorithms.MTMO.EMTPD import EMTPD
from Algorithms.MTMO.MOMTEASaO import MOMTEASaO
from Methods.data_analysis import DataAnalyzer
# from Problems.STMO.CF import CF, SETTINGS
# from Problems.MTMO.mtmo_instance import MTMOInstances, SETTINGS
from Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS
# from Problems.STMO.DTLZ import DTLZ, SETTINGS
# from Problems.STMO.WFG import WFG, SETTINGS

if __name__ == '__main__':
    # batch_exp = BatchExperiment(base_path='./Data', clear_folder=False)
    #
    # prob = CEC17MTMO()
    #
    # batch_exp.add_problem(prob.P1, 'P1')
    # batch_exp.add_problem(prob.P2, 'P2')
    # batch_exp.add_problem(prob.P3, 'P3')
    # batch_exp.add_problem(prob.P4, 'P4')

    # batch_exp.add_algorithm(NSGAII, 'NSGA-II', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(MOMFEA, 'MO-MFEA', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(MTEADDN, 'MTEA-D-DN', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(MTDEMKTA, 'MTDE-MKTA', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(EMTET, 'EMT-ET', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(EMTPD, 'EMT-PD', n=100, max_nfes=20000, disable_tqdm=False)
    # batch_exp.add_algorithm(MOMTEASaO, 'MO-MTEA-SaO', n=100, max_nfes=20000, disable_tqdm=False)

    # batch_exp.run(n_runs=2, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=SETTINGS,
        algorithm_order=['NSGA-II', 'MO-MFEA', 'MO-MTEA-SaO'],
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
from Algorithms.STSO.TLRBF import TLRBF
from Algorithms.STSO.GLSADE import GLSADE
from Algorithms.STSO.ESAO import ESAO
from Algorithms.STSO.SHPSO import SHPSO
from Algorithms.STSO.SACOSO import SACOSO
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

    prob = CEC17MTSO()

    batch_exp.add_problem(prob.P1, 'P1')
    batch_exp.add_problem(prob.P2, 'P2')
    batch_exp.add_problem(prob.P3, 'P3')
    batch_exp.add_problem(prob.P4, 'P4')
    batch_exp.add_problem(prob.P5, 'P5')
    batch_exp.add_problem(prob.P6, 'P6')
    batch_exp.add_problem(prob.P7, 'P7')
    batch_exp.add_problem(prob.P8, 'P8')
    batch_exp.add_problem(prob.P9, 'P9')

    batch_exp.add_algorithm(TLRBF, 'TLRBF', n_initial=100, max_nfes=1000, disable_tqdm=False)
    batch_exp.add_algorithm(GLSADE, 'GL-SADE', n_initial=100, max_nfes=1000, disable_tqdm=False)
    batch_exp.add_algorithm(ESAO, 'ESAO', n_initial=100, max_nfes=1000, disable_tqdm=False)
    batch_exp.add_algorithm(SHPSO, 'SHPSO', n_initial=100, max_nfes=1000, disable_tqdm=False)
    batch_exp.add_algorithm(SACOSO, 'SA-COSO', n_initial=100, max_nfes=1000, disable_tqdm=False)

    batch_exp.run(n_runs=2, verbose=True, max_workers=8)

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
        log_scale=True,
        show_pf=True,
        show_nd=True,
        best_so_far=True,
        clear_results=True
    )

    results = analyzer.run()
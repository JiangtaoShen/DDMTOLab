from Methods.batch_experiment import BatchExperiment
from Algorithms.STSO.BO import BO
from Algorithms.MTSO.RAMTEA import RAMTEA
from Algorithms.MTSO.MTBO import MTBO
from Algorithms.MTSO.LCBEMT import LCBEMT
from Algorithms.MTSO.MUMBO import MUMBO
from Algorithms.MTSO.SELF import SELF
from Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from Methods.data_analysis import DataAnalyzer


if __name__ == '__main__':
    # batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)
    #
    # cec17mtso = CEC17MTSO_10D()
    #
    # batch_exp.add_problem(cec17mtso.P1, 'P1')
    # batch_exp.add_problem(cec17mtso.P2, 'P2')
    # # batch_exp.add_problem(cec17mtso.P3, 'P3')
    # # batch_exp.add_problem(cec17mtso.P4, 'P4')
    # # batch_exp.add_problem(cec17mtso.P5, 'P5')
    # # batch_exp.add_problem(cec17mtso.P6, 'P6')
    #
    # # batch_exp.add_algorithm(BO, 'BO', n_initial=20, max_nfes=50, disable_tqdm=False)
    # # batch_exp.add_algorithm(BO2, 'BO2', n_initial=10, max_nfes=50, disable_tqdm=False)
    # # batch_exp.add_algorithm(LCBEMT, 'LCB-EMT', n=10, max_nfes=100, Nt=1, TGap=2, disable_tqdm=False)
    # # batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=10, max_nfes=50, disable_tqdm=False)
    # # batch_exp.add_algorithm(MUMBO, 'MUMBO', n_initial=10, max_nfes=50, disable_tqdm=False)
    # # batch_exp.add_algorithm(RAMTEA, 'RAMTEA', n_initial=20, max_nfes=50, disable_tqdm=False)
    # # batch_exp.add_algorithm(SELF, 'SELF', max_nfes=100, disable_tqdm=False)
    #
    # batch_exp.run(n_runs=2, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=None,
        algorithm_order=None,
        save_path='./Results',
        table_format='latex',
        figure_format='pdf',
        statistic_type='mean',
        significance_level=0.05,
        rank_sum_test=False,
        log_scale=True,
        show_pf=True,
        show_nd=True,
        best_so_far=True,
        clear_results=True
    )

    results = analyzer.run()
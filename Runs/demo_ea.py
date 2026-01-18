from Methods.batch_experiment import BatchExperiment
from Methods.data_analysis import DataAnalyzer
from Problems.MTSO.cec19_matso import CEC19MaTSO
from Algorithms.STSO.CMAES import CMAES
from Algorithms.MTSO.EBS import EBS


if __name__ == '__main__':
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = CEC19MaTSO()
    batch_exp.add_problem(prob.P2, 'P2', task_num=10)

    n = 100
    max_nfes = 100000
    batch_exp.add_algorithm(CMAES, 'CMA-ES', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(EBS, 'EBS', n=n, max_nfes=max_nfes)

    batch_exp.run(n_runs=2, verbose=True, max_workers=8)

    analyzer = DataAnalyzer(
        data_path='./Data',
        algorithm_order=None,
        save_path='./Results',
        table_format='latex',
        figure_format='pdf',
        statistic_type='mean',
        significance_level=0.05,
        rank_sum_test=True,
        log_scale=True,
        best_so_far=True,
        clear_results=True
    )

    results = analyzer.run()
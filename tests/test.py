import ddmtolab
from ddmtolab.Algorithms.STMO.NSGAII import NSGAII
from ddmtolab.Problems.MTMO.mtmo_instance import MTMOInstances, SETTINGS
from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Methods.batch_experiment import BatchExperiment

if __name__ == "__main__":
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    prob = MTMOInstances()
    batch_exp.add_problem(prob.P1, 'P1')
    batch_exp.add_problem(prob.P2, 'P2')

    batch_exp.add_algorithm(NSGAII, 'NSGAII', max_nfes=10000)

    batch_exp.run(n_runs=5)

    data_analyzer = DataAnalyzer(settings=SETTINGS, figure_format='png')
    data_analyzer.run()


.. _quickstart:

Quick Start
===========

Overview
--------

DDMTOLab is an open-source platform for solving multi-task optimization problems (MTOPs). It takes a multi-task optimization problem as input, applies optimization algorithms to solve it, and outputs the optimal solutions obtained.

**Directory Structure:**

- **Quick Testing:** Run individual algorithms directly. Results are saved in ``TestData`` (raw data) and ``TestResults`` (analysis results).
- **Batch Experiments:** Run multiple algorithms on multiple problems. Results are saved in ``Data`` (raw data) and ``Results`` (analysis results).

Demo 1: STSO Test
-----------------------------------------------------

We use Genetic Algorithm (GA) to solve a single-variable optimization problem:

.. code-block:: python

    import numpy as np
    from Methods.mtop import MTOP
    from Algorithms.STSO.GA import GA

    # Define objective function
    def t1(x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    # Create problem and add task
    problem = MTOP()
    problem.add_task(t1, dim=1)

    # Run optimization
    results = GA(problem).optimize()
    print(results.best_decs, results.best_objs)

    # Analyze and visualize results
    from Methods.test_data_analysis import TestDataAnalyzer
    TestDataAnalyzer().run()

Demo 2: MTSO Test
-----------------------------------

We solve a 2-task optimization problem with different dimensions and evaluation budgets.

**Task 1:** Sphere function (2D, simple convex function)

**Task 2:** Rastrigin function (3D, multiple local minima)

We compare Genetic Algorithm (GA) and Bayesian Optimization (BO). For GA, we set population size ``n=[10, 10]`` and maximum evaluations ``max_nfes=[100, 50]`` for both tasks. For BO, we use smaller initial samples ``n_initial=[20, 10]`` with the same evaluation budget.

.. code-block:: python

    import numpy as np
    from Methods.mtop import MTOP
    from Algorithms.STSO.GA import GA
    from Algorithms.STSO.BO import BO

    # Define Task 1: Sphere function (simple convex optimization)
    def t1(x):
        return np.sum(x**2)

    # Define Task 2: Rastrigin function (highly multimodal optimization)
    def t2(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    # Create multi-task optimization problem
    problem = MTOP()

    # Add Task 1: 2D Sphere function with bounds [-5, 5]
    problem.add_task(t1, dim=2, lower_bound=np.array([-5, -5]), upper_bound=np.array([5, 5]))

    # Add Task 2: 3D Rastrigin function with bounds [-5.12, 5.12]
    problem.add_task(t2, dim=3, lower_bound=np.array([-5.12]*3), upper_bound=np.array([5.12]*3))

    # Run GA with population size [10, 10] and evaluation budget [100, 50]
    GA(problem, n=[10, 10], max_nfes=[100, 50]).optimize()

    # Run BO with initial samples [20, 10] and evaluation budget [100, 50]
    BO(problem, n_initial=[20, 10], max_nfes=[100, 50]).optimize()

    # Analyze and visualize results
    from Methods.test_data_analysis import TestDataAnalyzer
    TestDataAnalyzer().run()


Demo 3: Batch Experiments
--------------------------

This example demonstrates how to run batch experiments comparing multiple algorithms on multiple benchmark problems.

We test 5 algorithms (GA, DE, PSO, MFEA, EMEA) on 3 CEC17 MTSO benchmark problems:

.. code-block:: python

   from Methods.batch_experiment import BatchExperiment
   from Methods.data_analysis import DataAnalyzer
   from Algorithms.STSO.GA import GA
   from Algorithms.STSO.PSO import PSO
   from Algorithms.STSO.DE import DE
   from Algorithms.MTSO.EMEA import EMEA
   from Algorithms.MTSO.MFEA import MFEA
   from Problems.MTSO.cec17_mtso import CEC17MTSO

   if __name__ == '__main__':
       # Step 1: Create batch experiment manager
       batch_exp = BatchExperiment(
           base_path='./Data',      # Data save path
           clear_folder=True        # Clear existing data
       )

       # Step 2: Add test problems
       cec17mtso = CEC17MTSO()
       batch_exp.add_problem(cec17mtso.P1, 'P1')
       batch_exp.add_problem(cec17mtso.P2, 'P2')
       batch_exp.add_problem(cec17mtso.P3, 'P3')

       # Step 3: Add algorithms with parameters
       batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=20000)
       batch_exp.add_algorithm(DE, 'DE', n=100, max_nfes=20000)
       batch_exp.add_algorithm(PSO, 'PSO', n=100, max_nfes=20000)
       batch_exp.add_algorithm(MFEA, 'MFEA', n=100, max_nfes=20000)
       batch_exp.add_algorithm(EMEA, 'EMEA', n=100, max_nfes=20000)

       # Step 4: Run batch experiments
       batch_exp.run(
           n_runs=20,          # Run each algorithm-problem combination 20 times
           verbose=True,       # Show progress information
           max_workers=8       # Use 8 parallel processes
       )

       # Step 5: Configure data analyzer
       analyzer = DataAnalyzer(
           data_path='./Data',                                      # Experiment data path
           settings=None,                                           # No SETTINGS needed (single-objective)
           algorithm_order=['GA', 'DE', 'PSO', 'EMEA', 'MFEA'],   # Algorithm display order
           save_path='./Results',                                   # Results save path
           table_format='latex',                                    # Table format
           figure_format='pdf',                                     # Figure format
           statistic_type='mean',                                   # Statistic type
           significance_level=0.05,                                 # Significance level
           rank_sum_test=True,                                      # Perform rank-sum test
           log_scale=True,                                          # Logarithmic scale
           show_pf=True,                                            # Show Pareto front
           show_nd=True,                                            # Show non-dominated solutions
           best_so_far=True,                                        # Use best-so-far values
           clear_results=True                                       # Clear old results
       )

       # Step 6: Run data analysis
       results = analyzer.run()
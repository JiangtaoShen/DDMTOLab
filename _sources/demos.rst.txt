.. _demos:

Demos
=====

This chapter provides comprehensive demonstrations of D²MTOLab's capabilities through 10 carefully designed examples. Each demo illustrates key concepts and workflows for different optimization scenarios.

Overview
--------

D²MTOLab supports four categories of optimization problems:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Category
     - Abbreviation
     - Description
   * - **STSO**
     - Single-Task Single-Objective
     - Traditional optimization with one task and one objective
   * - **STMO**
     - Single-Task Multiobjective
     - Pareto optimization with multiple conflicting objectives
   * - **MTSO**
     - Multitask Single-Objective
     - Multiple related tasks sharing knowledge
   * - **MTMO**
     - Multitask Multiobjective
     - Multiple tasks with multiple objectives each

**Demo Structure:**

- **Demo 1-3**: Multitask Single-Objective (MTSO) - from custom problems to batch experiments
- **Demo 4-5**: Expensive MTSO - surrogate-assisted optimization with limited budgets
- **Demo 6-7**: Multitask Multiobjective (MTMO) - with IGD metric calculation
- **Demo 8-9**: Expensive Single-Task Multiobjective (STMO) - surrogate-assisted MOEAs
- **Demo 10**: Using alternative metrics (Hypervolume)

**Output Directories:**

- ``./Data``: Raw optimization data (decision variables, objectives, history)
- ``./Results``: Analysis outputs (convergence plots, tables, animations)

----

Demo 1: Custom Multitask Problem
---------------------------------

This demo shows how to define custom objective functions and build a multitask optimization problem from scratch.

**Key Concepts:**

- Defining objective functions with normalized input [0,1]
- Building MTOP with tasks of different dimensions
- Comparing single-task GA vs multitask MFEA

**Objective Functions:**

.. code-block:: python

   import numpy as np
   from ddmtolab.Methods.mtop import MTOP
   from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
   from ddmtolab.Methods.animation_generator import AnimationGenerator
   from ddmtolab.Algorithms.STSO.GA import GA
   from ddmtolab.Algorithms.MTSO.MFEA import MFEA

   # Define objective functions
   # Note: Input x is normalized to [0,1], shape: (n_samples, dim)
   # Functions should return shape: (n_samples, 1)

   def sphere(x):
       """Sphere function: f(x) = sum(x^2), global minimum at origin."""
       x_scaled = x * 10 - 5  # Scale from [0,1] to [-5,5]
       return np.sum(x_scaled ** 2, axis=1, keepdims=True)

   def rastrigin(x):
       """Rastrigin function: highly multimodal, global minimum at origin."""
       x_scaled = x * 10 - 5
       D = x.shape[1]
       return (10 * D + np.sum(x_scaled ** 2 - 10 * np.cos(2 * np.pi * x_scaled),
                               axis=1, keepdims=True))

   def rosenbrock(x):
       """Rosenbrock function: valley-shaped, global minimum at (1,1,...,1)."""
       x_scaled = x * 4 - 2  # Scale from [0,1] to [-2,2]
       result = np.zeros((x.shape[0], 1))
       for i in range(x.shape[1] - 1):
           result += (100 * (x_scaled[:, i+1:i+2] - x_scaled[:, i:i+1] ** 2) ** 2
                      + (x_scaled[:, i:i+1] - 1) ** 2)
       return result

**Problem Definition and Optimization:**

.. code-block:: python

   # Build multitask problem with 3 tasks of different dimensions
   problem = MTOP()
   problem.add_task(sphere, dim=10)      # Task 1: 10D Sphere
   problem.add_task(rastrigin, dim=20)   # Task 2: 20D Rastrigin
   problem.add_task(rosenbrock, dim=30)  # Task 3: 30D Rosenbrock

   # Run optimization
   # GA: solves each task independently
   # MFEA: exploits inter-task similarities via implicit knowledge transfer
   GA(problem, n=100, max_nfes=10000).optimize()
   MFEA(problem, n=100, max_nfes=10000).optimize()

**Results Analysis:**

.. code-block:: python

   # Analyze convergence curves and runtime
   analyzer = TestDataAnalyzer(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['GA', 'MFEA'],
       figure_format='png',
       log_scale=True,           # Log scale for convergence visualization
       best_so_far=True,         # Plot best fitness found so far
       clear_results=True
   )
   results = analyzer.run()

   # Generate optimization animation
   # merge=1: decision space separated, objective space merged
   generator = AnimationGenerator(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['GA', 'MFEA'],
       title='GA vs MFEA on Custom MTSO',
       merge=1,
       max_nfes=10000,
       format='mp4',
       log_scale=True
   )
   generator.run()

----

Demo 2: MTSO Single-Run Test
----------------------------

This demo compares algorithms on the CEC17-MTSO benchmark suite, demonstrating the standard single-run experiment workflow.

**Key Concepts:**

- Loading benchmark problems from CEC17-MTSO
- Single-run experiment with analysis and animation
- Visualization options: Pareto front, non-dominated solutions

.. code-block:: python

   from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
   from ddmtolab.Methods.animation_generator import AnimationGenerator
   from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
   from ddmtolab.Algorithms.STSO.GA import GA
   from ddmtolab.Algorithms.MTSO.MFEA import MFEA

   # Load benchmark problem
   problem = CEC17MTSO().P1()

   # Run algorithms with same budget
   GA(problem, n=100, max_nfes=10000).optimize()     # Single-task baseline
   MFEA(problem, n=100, max_nfes=10000).optimize()   # Multitask algorithm

   # Analyze results
   analyzer = TestDataAnalyzer(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['GA', 'MFEA'],
       figure_format='png',
       log_scale=False,
       show_pf=True,             # Show Pareto front reference
       show_nd=True,             # Show non-dominated solutions
       best_so_far=True,
       clear_results=True
   )
   results = analyzer.run()

   # Generate animation
   # merge=2: both decision and objective spaces merged
   generator = AnimationGenerator(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['GA', 'MFEA'],
       title='GA vs MFEA on CEC17-MTSO-P1',
       merge=2,
       max_nfes=10000,
       format='mp4',
       log_scale=False
   )
   generator.run()

----

Demo 3: MTSO Batch Experiment
-----------------------------

This demo runs batch experiments with statistical analysis, generating publication-ready tables and plots.

**Key Concepts:**

- ``BatchExperiment`` for parallel multi-run experiments
- ``DataAnalyzer`` for statistical comparison
- LaTeX table generation with Wilcoxon rank-sum test

.. code-block:: python

   from ddmtolab.Methods.data_analysis import DataAnalyzer
   from ddmtolab.Methods.batch_experiment import BatchExperiment
   from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
   from ddmtolab.Algorithms.STSO.GA import GA
   from ddmtolab.Algorithms.MTSO.MFEA import MFEA
   from ddmtolab.Algorithms.MTSO.MTEA_AD import MTEA_AD

   if __name__ == '__main__':
       # Initialize batch experiment manager
       batch_exp = BatchExperiment(base_path='./Data', clear_folder=False)

       # Add benchmark problems
       problems = CEC17MTSO()
       batch_exp.add_problem(problems.P1, 'P1')
       batch_exp.add_problem(problems.P2, 'P2')
       batch_exp.add_problem(problems.P3, 'P3')

       # Add algorithms to compare
       # GA: single-task baseline
       # MFEA: multi-factorial EA
       # MTEA-AD: multitask EA with adaptive distribution
       batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=10000)
       batch_exp.add_algorithm(MFEA, 'MFEA', n=100, max_nfes=10000)
       batch_exp.add_algorithm(MTEA_AD, 'MTEA-AD', n=100, max_nfes=10000)

       # Run experiments
       # n_runs: independent runs for statistical significance
       # max_workers: parallel processes
       batch_exp.run(n_runs=5, verbose=True, max_workers=5)

       # Statistical analysis
       analyzer = DataAnalyzer(
           data_path='./Data',
           settings=None,                    # No reference PF for MTSO
           algorithm_order=['GA','MFEA','MTEA-AD'],
           save_path='./Results',
           table_format='latex',             # Publication-ready LaTeX
           figure_format='pdf',
           statistic_type='mean',
           rank_sum_test=True,               # Wilcoxon rank-sum test
           log_scale=True,
           show_pf=True,
           show_nd=True,
           merge_plots=True,                 # Combine all problems in one figure
           merge_columns=3,
           best_so_far=True,
           clear_results=True,
           convergence_k=10                  # Sample k points for curves
       )
       results = analyzer.run()

----

Demo 4: Expensive MTSO Single-Run
---------------------------------

This demo focuses on expensive optimization where function evaluations are costly, comparing surrogate-assisted algorithms.

**Key Concepts:**

- Expensive optimization with limited budget (50 evaluations)
- Bayesian Optimization (BO) and multitask extensions
- Knowledge transfer for sample efficiency

.. code-block:: python

   from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
   from ddmtolab.Methods.animation_generator import AnimationGenerator
   from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
   from ddmtolab.Algorithms.STSO.GA import GA
   from ddmtolab.Algorithms.STSO.BO import BO
   from ddmtolab.Algorithms.MTSO.BO_LCB_BCKT import BO_LCB_BCKT

   # Use 10D variant for expensive optimization
   problem = CEC17MTSO_10D().P1()

   # Compare with limited budget (50 evaluations)
   # GA: baseline
   # BO: single-task Bayesian optimization
   # BO-LCB-BCKT: multitask BO with knowledge transfer
   GA(problem, n=10, max_nfes=50).optimize()
   BO(problem, n_initial=10, max_nfes=50).optimize()
   BO_LCB_BCKT(problem, n_initial=10, max_nfes=50).optimize()

   # Analyze results
   analyzer = TestDataAnalyzer(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['GA', 'BO', 'BO-LCB-BCKT'],
       figure_format='png',
       log_scale=False,
       show_pf=True,
       show_nd=True,
       best_so_far=True,
       clear_results=True
   )
   results = analyzer.run()

   # Generate animation
   generator = AnimationGenerator(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['GA', 'BO', 'BO-LCB-BCKT'],
       title='GA, BO, and BO-LCB-BCKT on CEC17-MTSO-10D-P1',
       merge=2,
       max_nfes=50,
       format='mp4',
       log_scale=False
   )
   generator.run()

----

Demo 5: Expensive MTSO Batch Experiment
---------------------------------------

This demo runs batch experiments comparing surrogate-assisted algorithms on expensive problems.

**Key Concepts:**

- Batch experiments with expensive algorithms
- Comparing BO, MTBO, and RAMTEA
- Statistical analysis with limited evaluation budget

.. code-block:: python

   from ddmtolab.Methods.data_analysis import DataAnalyzer
   from ddmtolab.Methods.batch_experiment import BatchExperiment
   from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
   from ddmtolab.Algorithms.STSO.GA import GA
   from ddmtolab.Algorithms.STSO.BO import BO
   from ddmtolab.Algorithms.MTSO.MTBO import MTBO
   from ddmtolab.Algorithms.MTSO.RAMTEA import RAMTEA

   if __name__ == '__main__':
       batch_exp = BatchExperiment(base_path='./Data', clear_folder=False)

       # Add 10D problems for expensive optimization
       problems = CEC17MTSO_10D()
       batch_exp.add_problem(problems.P1, 'P1')
       batch_exp.add_problem(problems.P2, 'P2')
       batch_exp.add_problem(problems.P3, 'P3')

       # Add algorithms with limited budget
       # GA: baseline (small population)
       # BO: single-task Bayesian optimization
       # MTBO: multitask Bayesian optimization
       # RAMTEA: ranking-based adaptive multitask EA
       batch_exp.add_algorithm(GA, 'GA', n=10, max_nfes=100)
       batch_exp.add_algorithm(BO, 'BO', n_initial=20, max_nfes=100, disable_tqdm=False)
       batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=20, max_nfes=100, disable_tqdm=False)
       batch_exp.add_algorithm(RAMTEA, 'RAMTEA', n_initial=20, max_nfes=100, disable_tqdm=False)

       batch_exp.run(n_runs=3, verbose=True, max_workers=6)

       # Statistical analysis
       analyzer = DataAnalyzer(
           data_path='./Data',
           settings=None,
           algorithm_order=['GA','BO','MTBO', 'RAMTEA'],
           save_path='./Results',
           table_format='latex',
           figure_format='pdf',
           statistic_type='mean',
           rank_sum_test=True,
           log_scale=True,
           show_pf=True,
           show_nd=True,
           merge_plots=True,
           merge_columns=4,
           best_so_far=True,
           clear_results=True,
           convergence_k=10
       )
       results = analyzer.run()

----

Demo 6: MTMO Single-Run Test
----------------------------

This demo demonstrates multitask multiobjective optimization with IGD metric calculation using reference Pareto fronts.

**Key Concepts:**

- Multitask multiobjective optimization
- Setting up ``SETTINGS`` for IGD calculation
- Reference Pareto front configuration per task

.. code-block:: python

   from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
   from ddmtolab.Methods.animation_generator import AnimationGenerator
   from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS
   from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
   from ddmtolab.Algorithms.MTMO.MO_MFEA import MO_MFEA

   # Load MTMO benchmark problem
   problem = CEC17MTMO().P1()

   # NSGA-II: single-task MOEA (solves each task independently)
   # MO-MFEA: multitask MOEA with implicit knowledge transfer
   NSGA_II(problem, n=100, max_nfes=10000).optimize()
   MO_MFEA(problem, n=100, max_nfes=10000).optimize()

   # Configure settings for IGD calculation
   # 'problems' maps each task to its problem key in SETTINGS
   # P1 has 2 tasks: SETTINGS['P1']['T1'], SETTINGS['P1']['T2']
   settings = SETTINGS.copy()
   settings['problems'] = ['P1', 'P1']  # One entry per task

   # Analyze with IGD metric
   analyzer = TestDataAnalyzer(
       data_path='./Data',
       settings=settings,            # Settings with reference PF
       save_path='./Results',
       algorithm_order=['NSGA-II', 'MO-MFEA'],
       figure_format='png',
       log_scale=False,
       show_pf=True,                 # Show reference Pareto front
       show_nd=True,                 # Show non-dominated solutions
       best_so_far=True,
       clear_results=True
   )
   results = analyzer.run()

   # Generate animation
   # merge=3: objective space separated for multiobjective
   generator = AnimationGenerator(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['NSGA-II', 'MO-MFEA'],
       title='NSGA-II vs MO-MFEA on CEC17-MTMO-P1',
       merge=3,
       max_nfes=10000,
       format='mp4',
       log_scale=False
   )
   generator.run()

----

Demo 7: MTMO Batch Experiment
-----------------------------

This demo runs batch experiments for multitask multiobjective optimization with statistical analysis.

**Key Concepts:**

- Batch experiments for MTMO problems
- ``SETTINGS`` configuration for benchmark IGD calculation
- Comparing NSGA-II, RVEA, and MTEA-D-DN

.. code-block:: python

   from ddmtolab.Methods.data_analysis import DataAnalyzer
   from ddmtolab.Methods.batch_experiment import BatchExperiment
   from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS
   from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
   from ddmtolab.Algorithms.STMO.RVEA import RVEA
   from ddmtolab.Algorithms.MTMO.MTEA_D_DN import MTEA_D_DN

   if __name__ == '__main__':
       batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

       # Add MTMO benchmark problems
       problems = CEC17MTMO()
       batch_exp.add_problem(problems.P1, 'P1')
       batch_exp.add_problem(problems.P2, 'P2')
       batch_exp.add_problem(problems.P8, 'P8')

       # Add algorithms
       # NSGA-II: classic Pareto-based MOEA
       # RVEA: reference vector guided EA
       # MTEA-D-DN: multitask MOEA with decomposition
       batch_exp.add_algorithm(NSGA_II, 'NSGA-II', n=100, max_nfes=10000)
       batch_exp.add_algorithm(RVEA, 'RVEA', n=100, max_nfes=10000)
       batch_exp.add_algorithm(MTEA_D_DN, 'MTEA-D-DN', n=100, max_nfes=10000)

       batch_exp.run(n_runs=5, verbose=True, max_workers=5)

       # SETTINGS contains reference Pareto fronts for IGD
       analyzer = DataAnalyzer(
           data_path='./Data',
           settings=SETTINGS,
           algorithm_order=['NSGA-II','RVEA','MTEA-D-DN'],
           save_path='./Results',
           table_format='latex',
           figure_format='pdf',
           statistic_type='mean',
           rank_sum_test=True,
           log_scale=True,
           show_pf=False,
           show_nd=True,
           merge_plots=True,
           merge_columns=4,
           best_so_far=True,
           clear_results=True,
           convergence_k=10
       )
       results = analyzer.run()

----

Demo 8: Expensive STMO Single-Run
---------------------------------

This demo compares surrogate-assisted MOEAs on expensive multiobjective problems using DTLZ benchmark.

**Key Concepts:**

- Expensive multiobjective optimization
- Surrogate-assisted algorithms: ParEGO, K-RVEA
- DTLZ benchmark with configurable objectives

.. code-block:: python

   from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
   from ddmtolab.Methods.animation_generator import AnimationGenerator
   from ddmtolab.Problems.STMO.DTLZ import DTLZ, SETTINGS
   from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
   from ddmtolab.Algorithms.STMO.ParEGO import ParEGO
   from ddmtolab.Algorithms.STMO.K_RVEA import K_RVEA

   # DTLZ2: scalable multiobjective benchmark
   # M=4: 4 objectives, dim=10: 10 decision variables
   problem = DTLZ().DTLZ2(M=4, dim=10)

   # Compare with limited budget (200 evaluations)
   # NSGA-II: standard MOEA (baseline)
   # ParEGO: scalarization with GP surrogate
   # K-RVEA: Kriging-assisted RVEA
   NSGA_II(problem, n=20, max_nfes=200).optimize()
   ParEGO(problem, n_initial=50, max_nfes=200, disable_tqdm=False).optimize()
   K_RVEA(problem, n_initial=50, max_nfes=200, disable_tqdm=False).optimize()

   # Configure settings for IGD
   settings = SETTINGS.copy()
   settings['problems'] = ['DTLZ2']

   # Analyze results
   analyzer = TestDataAnalyzer(
       data_path='./Data',
       settings=settings,
       save_path='./Results',
       algorithm_order=['NSGA-II', 'ParEGO', 'K-RVEA'],
       figure_format='png',
       log_scale=False,
       show_pf=True,
       show_nd=True,
       best_so_far=True,
       clear_results=True
   )
   results = analyzer.run()

   # Generate animation
   generator = AnimationGenerator(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['NSGA-II', 'ParEGO', 'K-RVEA'],
       title='NSGA-II, ParEGO, and K-RVEA on DTLZ2',
       merge=3,
       max_nfes=200,
       format='mp4',
       log_scale=False
   )
   generator.run()

----

Demo 9: Expensive STMO Batch Experiment
---------------------------------------

This demo runs batch experiments comparing surrogate-assisted MOEAs on DTLZ benchmark problems.

**Key Concepts:**

- Batch experiments for expensive STMO
- Comparing ParEGO, K-RVEA, and DSAEA-PS
- Statistical analysis with IGD metric

.. code-block:: python

   from ddmtolab.Methods.data_analysis import DataAnalyzer
   from ddmtolab.Methods.batch_experiment import BatchExperiment
   from ddmtolab.Problems.STMO.DTLZ import DTLZ, SETTINGS
   from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
   from ddmtolab.Algorithms.STMO.ParEGO import ParEGO
   from ddmtolab.Algorithms.STMO.K_RVEA import K_RVEA
   from ddmtolab.Algorithms.STMO.DSAEA_PS import DSAEA_PS

   if __name__ == '__main__':
       batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

       # Add DTLZ problems with 3 objectives
       problems = DTLZ()
       batch_exp.add_problem(problems.DTLZ1, 'DTLZ1', M=3, dim=10)
       batch_exp.add_problem(problems.DTLZ2, 'DTLZ2', M=3, dim=10)

       # Add surrogate-assisted algorithms
       # NSGA-II: baseline MOEA
       # ParEGO: efficient global optimization for MOO
       # K-RVEA: Kriging-assisted reference vector EA
       # DSAEA-PS: data-driven surrogate EA with Pareto selection
       batch_exp.add_algorithm(NSGA_II, 'NSGA-II', n=10, max_nfes=200)
       batch_exp.add_algorithm(ParEGO, 'ParEGO', n_initial=100, max_nfes=200, disable_tqdm=False)
       batch_exp.add_algorithm(K_RVEA, 'K-RVEA', n_initial=100, max_nfes=200, disable_tqdm=False)
       batch_exp.add_algorithm(DSAEA_PS, 'DSAEA-PS', n_initial=100, max_nfes=200, disable_tqdm=False)

       batch_exp.run(n_runs=3, verbose=True, max_workers=6)

       # Analyze with DTLZ reference Pareto fronts
       analyzer = DataAnalyzer(
           data_path='./Data',
           settings=SETTINGS,
           algorithm_order=['NSGA-II','ParEGO','K-RVEA', 'DSAEA-PS'],
           save_path='./Results',
           table_format='latex',
           figure_format='pdf',
           statistic_type='mean',
           rank_sum_test=True,
           log_scale=True,
           show_pf=False,
           show_nd=True,
           merge_plots=True,
           merge_columns=2,
           best_so_far=True,
           clear_results=True,
           convergence_k=10
       )
       results = analyzer.run()

----

Demo 10: Using Hypervolume Metric
---------------------------------

This demo shows how to use Hypervolume (HV) instead of IGD as the performance metric. HV requires reference points rather than reference Pareto fronts.

**Key Concepts:**

- Configuring HV metric with reference points
- Custom settings dictionary structure
- Reference point specification per task

**HV vs IGD:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Requirement
     - Interpretation
   * - **IGD**
     - Reference Pareto front
     - Lower is better
   * - **HV**
     - Reference point (upper bound)
     - Higher is better

.. code-block:: python

   from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
   from ddmtolab.Methods.animation_generator import AnimationGenerator
   from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO
   from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
   from ddmtolab.Algorithms.MTMO.MTEA_D_DN import MTEA_D_DN

   # Load problem
   problem = CEC17MTMO().P1()

   # Run algorithms
   NSGA_II(problem, n=100, max_nfes=20000).optimize()
   MTEA_D_DN(problem, n=100, max_nfes=20000).optimize()

   # Configure HV metric settings
   # Reference points should dominate all possible solutions
   # Format: {problem_key: {task_key: [ref_point_per_objective]}}
   settings = {
       'metric': 'HV',                      # Use Hypervolume instead of IGD
       'problems': ['P1', 'P1'],            # Map each task to problem key
       'P1': {
           'T1': [3, 3],                    # Reference point for Task 1
           'T2': [4, 4.5]                   # Reference point for Task 2
       }
   }

   # Analyze with HV metric (higher is better)
   analyzer = TestDataAnalyzer(
       data_path='./Data',
       settings=settings,
       save_path='./Results',
       algorithm_order=['NSGA-II', 'MTEA-D-DN'],
       figure_format='png',
       log_scale=False,
       show_pf=False,                       # No PF for HV
       show_nd=True,
       best_so_far=True,
       clear_results=True
   )
   results = analyzer.run()

   # Generate animation
   generator = AnimationGenerator(
       data_path='./Data',
       save_path='./Results',
       algorithm_order=['NSGA-II', 'MTEA-D-DN'],
       title='NSGA-II and MTEA-D-DN on CEC17-MTMO-P1',
       merge=3,
       max_nfes=20000,
       format='mp4',
       log_scale=False
   )
   generator.run()

----

Summary
-------

.. list-table:: Demo Overview
   :header-rows: 1
   :widths: 10 25 35 30

   * - Demo
     - Category
     - Description
     - Key Algorithms
   * - 1
     - MTSO
     - Custom problem definition
     - GA, MFEA
   * - 2
     - MTSO
     - CEC17-MTSO single run
     - GA, MFEA
   * - 3
     - MTSO
     - Batch experiment with statistics
     - GA, MFEA, MTEA-AD
   * - 4
     - Expensive MTSO
     - Surrogate-assisted single run
     - GA, BO, BO-LCB-BCKT
   * - 5
     - Expensive MTSO
     - Surrogate-assisted batch
     - GA, BO, MTBO, RAMTEA
   * - 6
     - MTMO
     - Multiobjective single run
     - NSGA-II, MO-MFEA
   * - 7
     - MTMO
     - Multiobjective batch
     - NSGA-II, RVEA, MTEA-D-DN
   * - 8
     - Expensive STMO
     - Surrogate-assisted MOEA
     - NSGA-II, ParEGO, K-RVEA
   * - 9
     - Expensive STMO
     - Surrogate-assisted batch
     - NSGA-II, ParEGO, K-RVEA, DSAEA-PS
   * - 10
     - MTMO
     - Hypervolume metric
     - NSGA-II, MTEA-D-DN

See Also
--------

* :ref:`algorithms` - Algorithm construction and usage
* :ref:`methods` - Analysis tools and utilities
* :ref:`problems` - Benchmark problem suites
* :ref:`api` - Complete API documentation

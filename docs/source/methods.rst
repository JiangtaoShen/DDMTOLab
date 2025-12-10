.. _methods:

Methods
=======

This chapter introduces the utility modules provided by **DDMTOLab**, including batch experiments, data analysis, performance metrics, and algorithm components. These modules provide standardized testing workflows and rich algorithm building tools.

Batch Experiments
-----------------

.. code-block:: python

    from Methods.batch_experiment import BatchExperiment

The batch experiment module provides a complete framework for running multiple optimization algorithms on multiple benchmark problems, supporting parallel processing, automatic logging, and configuration management.

Module Features
~~~~~~~~~~~~~~~

The ``BatchExperiment`` class offers the following core features:

1. **Flexible Configuration**: Support for adding multiple test problems and algorithms with their parameter configurations
2. **Parallel Computing**: Utilize multi-core CPU for parallel execution to significantly improve efficiency
3. **Complete Experiment Recording**: Automatically record execution time, status, and error information
4. **Configuration Persistence**: Save experiment configurations as YAML files for reproducibility
5. **Time Statistics**: Generate CSV files with detailed timing information
6. **Optional Folder Cleanup**: Support for cleaning old data before experiments
7. **Progress Visualization**: Real-time display of experiment progress and completion status

Class Initialization
~~~~~~~~~~~~~~~~~~~~

Initialize the ``BatchExperiment`` class:

.. code-block:: python

    batch_exp = BatchExperiment(
        base_path='./Data',      # Data storage path
        clear_folder=False       # Whether to clear folder
    )

**Parameters:**

- ``base_path``: Storage path for experiment data, default: ``./Data``
- ``clear_folder``: If ``True``, clear all contents in the target folder before initialization

Adding Problems
~~~~~~~~~~~~~~~

Use the ``add_problem`` method to add optimization problems:

.. code-block:: python

    from Problems.MTSO.cec17_mtso import CEC17MTSO
    cec17mtso = CEC17MTSO()

    # Add problems to batch experiment
    batch_exp.add_problem(problem_creator=cec17mtso.P1, problem_name='P1')
    batch_exp.add_problem(problem_creator=cec17mtso.P2, problem_name='P2')

**Parameters:**

- ``problem_creator``: Problem creation function that generates problem instances
- ``problem_name``: Problem name for result file naming
- ``**problem_params``: Optional parameters passed to the problem creator (e.g., dimension)

Adding Algorithms
~~~~~~~~~~~~~~~~~

Use the ``add_algorithm`` method to add optimization algorithms:

.. code-block:: python

    from Algorithms.STSO.GA import GA
    from Algorithms.STSO.DE import DE
    from Algorithms.STSO.PSO import PSO

    # Add algorithms with parameters
    batch_exp.add_algorithm(algorithm_class=GA, algorithm_name='GA',
                           n=100, max_nfes=10000)
    batch_exp.add_algorithm(algorithm_class=DE, algorithm_name='DE',
                           n=100, max_nfes=10000)
    batch_exp.add_algorithm(algorithm_class=PSO, algorithm_name='PSO',
                           n=100, max_nfes=10000)

**Parameters:**

- ``algorithm_class``: Algorithm class (e.g., ``GA``, ``DE``, ``PSO``)
- ``algorithm_name``: Algorithm name for subfolder and file naming
- ``**algorithm_params``: Algorithm parameters (``problem``, ``save_path``, and ``name`` are set automatically)

Running Experiments
~~~~~~~~~~~~~~~~~~~

Execute the batch experiment using the ``run`` method:

.. code-block:: python

    batch_exp.run(n_runs=30, verbose=True, max_workers=8)

**Parameters:**

- ``n_runs``: Number of independent runs for each algorithm on each problem
- ``verbose``: Whether to print detailed progress information, default: ``True``
- ``max_workers``: Maximum number of parallel worker processes, default: CPU core count

**Example Output:**

.. code-block:: text

    Clearing existing data folder: ./Data
    Configuration saved to: ./Data/experiment_config.yaml

    ============================================================
    Starting Batch Experiment (Parallel Mode)!
    ============================================================

    Number of problems: 2
    Number of algorithms: 3
    Number of independent runs: 30
    Total experiments: 180
    Max workers: 8

    Progress: 18/180 (10.0%)
    Progress: 36/180 (20.0%)
    ...

    Total time: 1200.00 seconds (20.00 minutes)
    Parallel speedup: 10.76x
    Timing summary saved to: ./Data/time_summary_20251203_143022.csv

    ============================================================
     All Experiments Completed!
    ============================================================

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

Experiment configurations are automatically saved as YAML files (``experiment_config.yaml``) when running, including:

1. Creation time and base path
2. Detailed problem configurations
3. Algorithm parameters
4. Run settings (number of runs, workers, etc.)

**Loading from Configuration:**

.. code-block:: python

    # Load experiment from saved configuration
    batch_exp = BatchExperiment.from_config('./Data/experiment_config.yaml')
    batch_exp.run()  # Use settings from config file

    # Override settings
    batch_exp = BatchExperiment.from_config('./Data/experiment_config.yaml')
    batch_exp.run(n_runs=50, max_workers=16)

Output Structure
~~~~~~~~~~~~~~~~

Batch experiments generate three types of files:

1. **Configuration File**: ``experiment_config.yaml``
2. **Algorithm Results**: Organized in subfolders

   .. code-block:: text

       Data/
       ├── GA/
       │   ├── GA_P1_1.pkl
       │   ├── GA_P1_2.pkl
       │   └── ...
       ├── DE/
       │   └── ...
       └── PSO/
           └── ...

3. **Timing Statistics**: ``time_summary_[timestamp].csv``

   .. list-table::
      :header-rows: 1
      :widths: 15 15 10 20 15 15 20

      * - Algorithm
        - Problem
        - Run
        - Filename
        - Time(s)
        - Status
        - Error
      * - GA
        - P1
        - 1
        - GA_P1_1
        - 1.2345
        - Success
        -
      * - GA
        - P1
        - 2
        - GA_P1_2
        - 1.2198
        - Success
        -
      * - PSO
        - P2
        - 5
        - PSO_P2_5
        - 0.0000
        - Failed
        - Division by zero

Data Analysis
-------------

.. code-block:: python

    from Methods.data_analysis import DataAnalyzer

The data analysis module provides comprehensive analysis and visualization for multi-task optimization experiment results, including metric calculation, statistical comparison tables, convergence curves, runtime analysis, and Pareto front visualization.

Module Features
~~~~~~~~~~~~~~~

The ``DataAnalyzer`` class offers:

1. **Automatic Data Scanning**: Automatically identify algorithms, problems, and run counts
2. **Multiple Performance Metrics**: Support for objective values (SO), IGD, and HV (MO)
3. **Statistical Analysis**: Mean, median, max, min statistics with Wilcoxon rank-sum test
4. **Table Generation**: Excel or LaTeX format tables with significance annotations
5. **Convergence Curves**: Plot algorithm convergence on each task with log-scale support
6. **Runtime Analysis**: Generate runtime comparison bar charts
7. **Pareto Front Visualization**: Support 2D, 3D, and high-dimensional non-dominated solutions
8. **Flexible Configuration**: Customizable color schemes, marker styles, and statistics
9. **Complete Pipeline**: One-step analysis or step-by-step execution

Class Initialization
~~~~~~~~~~~~~~~~~~~~

Initialize the ``DataAnalyzer`` with rich configuration options:

.. code-block:: python

    analyzer = DataAnalyzer(
        data_path='./Data',              # Data directory path
        settings=None,                   # Problem settings (for complex metrics)
        algorithm_order=None,            # Algorithm display order
        save_path='./Results',           # Results save path
        table_format='excel',            # Table format: 'excel' or 'latex'
        figure_format='pdf',             # Figure format: 'pdf', 'png', 'svg'
        statistic_type='mean',           # Statistic: 'mean', 'median', 'max', 'min'
        significance_level=0.05,         # Significance level for tests
        rank_sum_test=True,              # Whether to perform rank-sum test
        log_scale=False,                 # Whether to use log scale
        show_pf=True,                    # Whether to show true Pareto front
        show_nd=True,                    # Whether to show only non-dominated
        best_so_far=True,                # Whether to use best-so-far values
        clear_results=True               # Whether to clear results folder
    )

Metric Configuration
~~~~~~~~~~~~~~~~~~~~

For problems requiring complex metrics (e.g., multi-objective optimization), provide a ``settings`` configuration dictionary:

.. code-block:: python

    SETTINGS = {
        'metric': 'IGD',                    # Performance metric: 'IGD' or 'HV'
        'ref_path': './MOReference',        # Reference file path
        'n_ref': 10000,                     # Number of reference points

        # Problem P1 reference definitions
        'P1': {
            'T1': 'P1_T1_ref.npy',         # Method 1: File path
            'T2': 'P1_T2_ref.csv',         # Supports .npy and .csv
        },

        # Problem P2 reference definitions
        'P2': {
            'T1': lambda n, m: generate_pf(n, m),  # Method 2: Callable function
            'T2': [[1.0, 0.0], [0.0, 1.0]],        # Method 3: Direct array
        },
    }

    # Use settings to create analyzer
    analyzer = DataAnalyzer(data_path='./Data', settings=SETTINGS)

Reference definitions support three methods:

1. **File Path**: String filename or full path, supports ``.npy`` and ``.csv``
2. **Callable Function**: Accepts ``(n_points, n_objectives)`` parameters, returns reference array
3. **Array Data**: Directly provide list, tuple, or NumPy array

Complete Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**One-Step Analysis:**

.. code-block:: python

    from Methods.data_analysis import DataAnalyzer

    # Create analyzer instance (settings optional for SO)
    analyzer = DataAnalyzer()

    # Execute complete analysis pipeline
    results = analyzer.run()

**Step-by-Step Execution:**

.. code-block:: python

    # Create analyzer
    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=SETTINGS,
        algorithm_order=['NSGA-II', 'MOEA/D', 'MyAlgo'],
        clear_results=False
    )

    # Step 1: Scan data directory
    scan_result = analyzer.scan_data()

    # Step 2: Calculate metrics
    metric_results = analyzer.calculate_metrics()

    # Step 3: Selective generation
    analyzer.generate_tables()              # Statistical tables
    analyzer.generate_convergence_plots()   # Convergence curves
    analyzer.generate_runtime_plots()       # Runtime plots
    analyzer.generate_nd_solution_plots()   # Pareto front plots

Accessing Raw Results
~~~~~~~~~~~~~~~~~~~~~

Access raw data through the returned ``MetricResults`` object:

.. code-block:: python

    # Run analysis
    results = analyzer.run()

    # Access metric values (per generation)
    algo1_p1_run1_task0 = results.metric_values['GA']['P1'][1][0]
    print(f"Convergence length: {len(algo1_p1_run1_task0)}")

    # Access best values
    best_vals = results.best_values['GA']['P1'][1]
    print(f"Best values per task: {best_vals}")

    # Access objective values (Pareto solutions)
    pareto_solutions = results.objective_values['GA']['P1'][1][0]
    print(f"Solution shape: {pareto_solutions.shape}")

    # Access runtime
    runtime_seconds = results.runtime['GA']['P1'][1]
    print(f"Runtime: {runtime_seconds:.2f}s")

    # Access max function evaluations
    max_nfes_list = results.max_nfes['GA']['P1']
    print(f"Max NFEs per task: {max_nfes_list}")

    # Access metric name
    print(f"Metric used: {results.metric_name}")

Output Structure
~~~~~~~~~~~~~~~~

Complete analysis generates the following output files:

.. code-block:: text

    ./Results/
    ├── results_table_mean.xlsx      # Statistical table (Excel)
    ├── results_table_mean.tex       # Statistical table (LaTeX)
    ├── P1.pdf                       # Convergence curve for P1
    ├── P2-Task1.pdf                 # Convergence curve for P2 Task1
    ├── P2-Task2.pdf                 # Convergence curve for P2 Task2
    ├── runtime_comparison.pdf       # Runtime comparison
    └── ND_Solutions/                # Non-dominated solutions
        ├── P1-GA.pdf
        ├── P1-DE.pdf
        ├── P2-Task1-GA.pdf
        └── ...

Test Data Analysis
------------------

.. code-block:: python

    from Methods.test_data_analysis import TestDataAnalyzer

The ``TestDataAnalyzer`` is a lightweight version of ``DataAnalyzer`` for quickly analyzing single test runs. It directly reads files with ``_test.pkl`` suffix without statistical tests or multi-run aggregation, suitable for algorithm development and debugging.

Module Features
~~~~~~~~~~~~~~~

1. **Simplified Data Structure**: Read test files directly without algorithm-classified subfolders
2. **Fast Analysis**: Skip statistical tests and multi-run aggregation
3. **Complete Visualization**: Convergence curves, runtime comparison, and Pareto fronts
4. **Table Generation**: LaTeX format result tables and convergence summaries
5. **Flexible Configuration**: Same configuration options as ``DataAnalyzer``

Class Initialization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    analyzer = TestDataAnalyzer(
        data_path='./TestData',          # Test data directory
        settings=None,                   # Problem settings (for MO)
        algorithm_order=None,            # Algorithm display order
        save_path='./TestResults',       # Results save path
        figure_format='pdf',             # Figure format
        log_scale=False,                 # Log scale
        show_pf=True,                    # Show true Pareto front
        show_nd=True,                    # Show only non-dominated
        best_so_far=True,                # Use best-so-far values
        clear_results=True,              # Clear results folder
        file_suffix='_test.pkl'          # Test file suffix
    )

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from Methods.test_data_analysis import TestDataAnalyzer

    # Create analyzer (settings optional for SO)
    analyzer = TestDataAnalyzer(data_path='./TestData',
                               save_path='./TestResults')

    # Execute complete analysis
    results = analyzer.run()

Output Structure
~~~~~~~~~~~~~~~~

.. code-block:: text

    ./TestResults/
    ├── test_results_table.tex           # Results comparison table
    ├── convergence_summary_table.tex    # Convergence summary table
    ├── Task1_convergence.pdf            # Task1 convergence
    ├── Task2_convergence.pdf            # Task2 convergence (if any)
    ├── runtime_comparison.pdf           # Runtime comparison
    └── ND_Solutions/                    # Non-dominated solutions
        ├── Task1-GA.pdf
        ├── Task1-DE.pdf
        └── ...

Comparison with DataAnalyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - TestDataAnalyzer
     - DataAnalyzer
   * - Data Source
     - Single test files (``*_test.pkl``)
     - Multiple repeated experiments
   * - File Structure
     - Direct test files in directory
     - Subfolders per algorithm
   * - Statistical Analysis
     - No statistical tests
     - Wilcoxon rank-sum test
   * - Table Format
     - LaTeX only
     - Excel and LaTeX
   * - Use Case
     - Development and quick validation
     - Formal experiment analysis

Performance Metrics
-------------------

.. code-block:: python

    from Methods.metrics import IGD, HV

The performance metrics module provides implementations of optimization algorithm evaluation metrics with a unified interface design for easy extension.

Module Features
~~~~~~~~~~~~~~~

The metric module follows these design principles:

1. **Unified Interface**: All metric classes follow the same interface specification
2. **Direction Indicator**: Each metric has a ``sign`` attribute (``-1`` for minimization, ``1`` for maximization)
3. **Callable Support**: Metric instances support functional calling (``__call__`` method)

Metric Interface
~~~~~~~~~~~~~~~~

All metric classes should follow this template:

.. code-block:: python

    class MetricTemplate:
        """Performance metric template"""

        def __init__(self):
            """Initialize metric"""
            self.name = "MetricName"    # Metric name
            self.sign = -1 or 1         # Direction: -1 minimize, 1 maximize

        def calculate(self, *args, **kwargs) -> float:
            """Calculate metric value"""
            # Implementation...
            pass

        def __call__(self, *args, **kwargs) -> float:
            """Support instance as function call"""
            return self.calculate(*args, **kwargs)

Usage Example: IGD
~~~~~~~~~~~~~~~~~~

IGD (Inverted Generational Distance) is a common metric for evaluating solution set quality in multi-objective optimization:

.. code-block:: python

    from Methods.metrics import IGD

    # Create metric instance
    igd = IGD()

    # Calculate metric value
    igd_value = igd.calculate(objs, pf)  # Method 1
    igd_value = igd(objs, pf)            # Method 2 (functional call)

    # Query metric properties
    print(f"Metric name: {igd.name}")    # Output: IGD
    print(f"Direction: {igd.sign}")      # Output: -1 (minimize)

Algorithm Components
--------------------

Algorithm Utilities
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from Methods.Algo_Methods.algo_utils import *

The algorithm utilities module provides a complete toolkit for building optimization algorithms, including population initialization, evaluation, selection, mutation, crossover, and auxiliary functions.

.. list-table:: Key Functions in algo_utils
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``initialization``
     - Initialize multi-task decision variable matrices with Random or LHS sampling
   * - ``evaluation``
     - Batch evaluate multiple tasks with selective objective/constraint evaluation
   * - ``evaluation_single``
     - Evaluate a single specified task
   * - ``crossover``
     - Simulated Binary Crossover (SBX) for two parent vectors
   * - ``mutation``
     - Polynomial mutation on decision vectors
   * - ``ga_generation``
     - Generate offspring using GA operators (SBX + mutation)
   * - ``de_generation``
     - Generate offspring using DE/rand/1/bin strategy
   * - ``tournament_selection``
     - Tournament selection with multi-criteria lexicographic ordering
   * - ``selection_elit``
     - Single-objective elite selection considering constraint violation
   * - ``nd_sort``
     - Fast non-dominated sorting algorithm
   * - ``crowding_distance``
     - Calculate crowding distance for diversity preservation
   * - ``init_history``
     - Initialize population history storage structure
   * - ``append_history``
     - Append current generation data to history
   * - ``build_save_results``
     - Extract best solutions, build Results object, and save to file
   * - ``trim_excess_evaluations``
     - Trim history exceeding max function evaluations
   * - ``space_transfer``
     - Transfer data between unified and real spaces
   * - ``normalize``
     - Data normalization (min-max or z-score)
   * - ``denormalize``
     - Inverse normalization to restore original scale
   * - ``vstack_groups``
     - Vertically stack multiple population arrays
   * - ``select_by_index``
     - Synchronously select rows from multiple arrays by index
   * - ``par_list``
     - Convert single parameter to multi-task parameter list
   * - ``get_algorithm_information``
     - Extract and print algorithm metadata

Bayesian Optimization Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from Methods.Algo_Methods.bo_utils import *

The BO utilities module provides core Bayesian optimization functionalities based on BoTorch and GPyTorch, including single-task and multi-task Gaussian process modeling.

.. list-table:: Key Functions in bo_utils
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``gp_build``
     - Build and train single-task Gaussian process model
   * - ``gp_predict``
     - Predict using trained single-task GP model
   * - ``bo_next_point``
     - Get next sampling point via single-task BO (LogEI acquisition)
   * - ``mtgp_build``
     - Build multi-task Gaussian process model
   * - ``mtgp_predict``
     - Predict for specified task using multi-task GP
   * - ``mtgp_task_corr``
     - Extract task correlation matrix from multi-task GP
   * - ``mtbo_next_point``
     - Get next sampling point via multi-task BO

Similarity Evaluation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from Methods.Algo_Methods.sim_evaluation import *

The similarity evaluation module computes inter-task similarity for knowledge transfer decisions.

.. list-table:: Key Functions in sim_evaluation
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``sim_calculate``
     - Calculate similarity matrix between tasks using Pearson correlation

Uniform Point Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from Methods.Algo_Methods.uniform_point import *

The uniform point generation module provides various methods for generating uniformly distributed points for multi-objective optimization and decision space sampling.

.. list-table:: Key Functions in uniform_point
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``uniform_point``
     - Unified interface for point generation (NBI/ILD/MUD/grid/Latin)
   * - ``nbi_method``
     - Normal-Boundary Intersection for reference points on unit simplex
   * - ``ild_method``
     - Incremental Lattice Design for adaptive reference points
   * - ``mud_method``
     - Mixture Uniform Design using good lattice points
   * - ``grid_method``
     - Grid sampling in unit hypercube
   * - ``latin_method``
     - Latin Hypercube Sampling for decision space exploration
   * - ``good_lattice_point``
     - Generate good lattice points for MUD method
   * - ``calc_cd2``
     - Calculate Centered Discrepancy (CD2) for uniformity evaluation

See Also
--------

* :ref:`problems` - Problem definition guide
* :ref:`algorithms` - Algorithm implementation guide
* :ref:`api_reference` - Complete API documentation
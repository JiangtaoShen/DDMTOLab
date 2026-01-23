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

The ``BatchExperiment`` class offers:

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
- ``**problem_params``: Optional parameters passed to the problem creator (e.g., maximum number of fitness evaluations)

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

The data analysis module provides comprehensive analysis and visualization for optimization results, including metric calculation, statistical comparison tables, convergence curves, runtime analysis, Pareto front visualization, etc.

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

Initialize the ``DataAnalyzer`` with configuration options:

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

Reference Data Loading
~~~~~~~~~~~~~~~~~~~~~~

The reference data loading system provides a flexible interface for loading Pareto fronts, reference points, or other reference data required for performance metric calculation and visualization.

**Supported Reference Types:**

The system supports three types of reference definitions:

1. **Callable Functions**: Dynamically generate reference data based on problem parameters
2. **File Paths**: Load pre-computed reference data from .npy or .csv files
3. **Array Data**: Directly use numpy arrays, lists, or tuples as reference data

**Core Interface:**

.. code-block:: python

   from Methods.data_utils import DataUtils

   reference = DataUtils.load_reference(
       settings=SETTINGS,
       problem='DTLZ1',
       task_identifier='T1',  # or task index: 0
       M=3,                   # Number of objectives (required)
       D=10,                  # Number of variables (optional)
       C=0                    # Number of constraints (optional)
   )

**Parameters:**

- ``settings``: Dictionary containing problem configurations
- ``problem``: Problem name (e.g., "DTLZ1", "DTLZ2")
- ``task_identifier``: Task name (str "T1") or index (int 0)
- ``M``: Number of objectives (required)
- ``D``: Number of decision variables (optional)
- ``C``: Number of constraints (optional, default: 0)

**Returns:** NumPy array with shape (n_points, M), or None if unavailable

**Example 1: Callable Reference Function**

Most common for benchmark problems:

.. code-block:: python

   from Methods.Algo_Methods.uniform_point import uniform_point

   # Define reference generation function
   def DTLZ1_PF(N, M):
       W, _ = uniform_point(N, M)
       return W / 2

   # Configure in settings
   SETTINGS = {
       'metric': 'IGD',
       'n_ref': 2000,
       'DTLZ1': {
           'T1': DTLZ1_PF,     # Function reference
           'T2': DTLZ1_PF,
       }
   }

   # Load reference (automatically calls DTLZ1_PF(2000, 3))
   reference = DataUtils.load_reference(SETTINGS, 'DTLZ1', 'T1', M=3)

**Function Signatures:**

Reference functions can have different signatures based on requirements:

.. code-block:: python

   # Signature 1: Basic (N, M)
   def basic_ref(N, M):
       return generate_reference(N, M)

   # Signature 2: With dimension (N, M, D)
   def dimension_ref(N, M, D):
       scale = np.sqrt(D)
       return generate_reference(N, M) * scale

   # Signature 3: Full parameters (N, M, D, C)
   def full_ref(N, M, D, C):
       ref = generate_reference(N, M)
       if C > 0:
           # Apply constraint-based filtering
           pass
       return ref

The system automatically detects the function signature and passes appropriate parameters.

**Example 2: File-Based Reference**

Load pre-computed reference from files:

.. code-block:: python

   SETTINGS = {
       'ref_path': './MOReference',
       'MyProblem': {
           'T1': 'myproblem_t1_pf.npy',        # Relative path
           'T2': '/abs/path/to/reference.csv',  # Absolute path
       }
   }

   reference = DataUtils.load_reference(SETTINGS, 'MyProblem', 'T1', M=3)

Supported file formats: ``.npy`` (NumPy binary) and ``.csv`` (comma-separated)

**Automatic File Search:**

If the specified file is not found, the system searches for:

1. ``{ref_path}/{problem}_{task}_ref.npy``
2. ``{ref_path}/{problem}_{task}_ref.csv``

**Example 3: Direct Array Reference**

Provide reference data directly:

.. code-block:: python

   # Predefined reference points
   predefined_pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

   SETTINGS = {
       'SimpleProblem': {
           'T1': predefined_pf,              # NumPy array
           'T2': [[0, 1], [1, 0]],           # List
           'T3': ([0, 1], [1, 0])            # Tuple
       }
   }

   reference = DataUtils.load_reference(SETTINGS, 'SimpleProblem', 'T1', M=2)

**Example 4: Shared Reference for All Tasks**

Use the same reference for all tasks:

.. code-block:: python

   SETTINGS = {
       'n_ref': 10000,
       'DTLZ2': {
           'all_tasks': DTLZ2_PF  # Applied to all tasks
       }
   }

   # All tasks automatically use the same reference
   ref_t1 = DataUtils.load_reference(SETTINGS, 'DTLZ2', 'T1', M=3)
   ref_t2 = DataUtils.load_reference(SETTINGS, 'DTLZ2', 'T2', M=3)

**Integration with DataAnalyzer**

The reference loading is automatically handled by ``DataAnalyzer`` when settings are provided:

.. code-block:: python

   # Define references in settings
   SETTINGS = {
       'metric': 'IGD',
       'n_ref': 5000,
       'DTLZ1': {'T1': DTLZ1_PF, 'T2': DTLZ1_PF},
       'DTLZ2': {'all_tasks': DTLZ2_PF}
   }

   # DataAnalyzer automatically uses references for:
   # - Metric calculation (IGD, HV, etc.)
   # - Pareto front visualization
   analyzer = DataAnalyzer(data_path='./Data', settings=SETTINGS)
   results = analyzer.run()

**Best Practices:**

1. **Organize reference files systematically:**

   .. code-block:: text

      MOReference/
      ├── DTLZ1_T1_ref.npy
      ├── DTLZ1_T2_ref.npy
      ├── DTLZ2_T1_ref.csv
      └── CustomProblem/
          ├── T1_ref.npy
          └── T2_ref.npy

2. **Set appropriate n_ref for metrics and visualization:**

   When calculating multi-objective metrics (e.g., IGD), it is recommended to set ``n_ref`` to 1000 (preferably not exceeding 2000). Using too many reference points can result in very large PDF files when visualizing Pareto fronts with the true PF overlay.

   .. code-block:: python

      SETTINGS = {
          'metric': 'IGD',
          'n_ref': 1000,  # Recommended: balance accuracy and file size
          'DTLZ1': {'T1': DTLZ1_PF}
      }

3. **Always provide M parameter** (number of objectives)

4. **Provide D and C** if your reference function requires them

5. **Use meaningful function signatures:**

   - ``(N, M)`` for simple problems
   - ``(N, M, D)`` when dimension matters
   - ``(N, M, D, C)`` for constrained problems

**Error Handling:**

The system provides informative warnings:

.. code-block:: python

   # Problem not found
   reference = DataUtils.load_reference(SETTINGS, 'NonexistentProblem', 'T1', M=3)
   # Warning: Problem 'NonexistentProblem' not found in settings
   # Returns: None

   # File not found
   # Warning: File not found: './MOReference/missing_file.npy'
   # Returns: None

   # Missing parameter D (when needed)
   # Warning: D not provided for Problem_T1, using 0

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

Animation Generator
-------------------

.. code-block:: python

    from Methods.optimization_animator import OptimizationAnimator, create_optimization_animation

The animation generator module provides comprehensive visualization tools for optimization processes, supporting both single-objective and multi-objective optimization with multiple comparison modes.

Module Features
~~~~~~~~~~~~~~~

The animation generator offers:

1. **Multiple Visualization Types**: Decision space evolution, convergence curves (SO), and Pareto front evolution (MO)
2. **Flexible Comparison Modes**: Support for individual animations or merged comparisons across algorithms
3. **NFEs-based Tracking**: Display convergence in terms of function evaluations for better comparability
4. **Batch Processing**: Automatically scan and generate animations for all result files
5. **Customizable Display**: Configure algorithm order, animation quality, frame rate, and format
6. **Multi-format Output**: Support for GIF and MP4 formats
7. **Task-specific Configuration**: Different NFEs settings for different optimization tasks

Visualization Components
~~~~~~~~~~~~~~~~~~~~~~~~

**For Single-Objective Optimization:**

- **Decision Space (Left)**: Parallel coordinate plot showing decision variable evolution
- **Convergence Curve (Right)**: Best objective value vs. NFEs (Number of Function Evaluations)

**For Multi-Objective Optimization:**

- **Decision Space (Left)**: Parallel coordinate plot showing decision variable evolution
- **Objective Space (Right)**: Pareto front evolution

  - 2D: Scatter plot (f1 vs. f2)
  - 3D: 3D scatter plot with rotation view
  - High-dimensional: Parallel coordinate plot with normalized objectives

Quick Start
~~~~~~~~~~~

**Single File Animation:**

.. code-block:: python

    from Methods.optimization_animator import create_optimization_animation

    # Generate animation for a single result file
    create_optimization_animation(
        pkl_path='./Data/GA/GA_P1_1.pkl',
        max_nfes=10000,
        format='gif'
    )

**Batch Generation:**

.. code-block:: python

    # Automatically scan and generate animations for all .pkl files
    create_optimization_animation(
        data_path='./Data',
        save_path='./Animations',
        max_nfes=10000,
        fps=10,
        dpi=100
    )

Comparison Modes
~~~~~~~~~~~~~~~~

The animation generator supports four merge modes for algorithm comparison:

**Mode 0: Individual Animations (No Merge)**

Generate separate animation for each algorithm:

.. code-block:: python

    create_optimization_animation(
        data_path='./Data',
        save_path='./Animations',
        merge=0,  # Default: individual animations
        max_nfes=10000
    )

Output structure:

.. code-block:: text

    Animations/
    ├── GA_P1_1_animation.gif
    ├── DE_P1_1_animation.gif
    └── PSO_P1_1_animation.gif

**Mode 1: Full Merge**

All algorithms in the same plots (side-by-side decision and objective spaces):

.. code-block:: python

    create_optimization_animation(
        pkl_path=['./Data/GA/GA_P1_1.pkl',
                  './Data/DE/DE_P1_1.pkl',
                  './Data/PSO/PSO_P1_1.pkl'],
        merge=1,
        title='Algorithm Comparison',
        algorithm_order=['GA', 'DE', 'PSO'],
        max_nfes=10000
    )

Layout: ``[Merged Decision Space | Merged Objective Space]``

**Mode 2: Decision Separated, Objective Merged**

Separate decision space for each algorithm, merged objective space:

.. code-block:: python

    create_optimization_animation(
        pkl_path=['./Data/GA/GA_P1_1.pkl',
                  './Data/DE/DE_P1_1.pkl',
                  './Data/PSO/PSO_P1_1.pkl'],
        merge=2,
        title='Comparison',
        algorithm_order=['GA', 'DE', 'PSO'],
        max_nfes=10000
    )

Layout: ``[GA Decision | DE Decision | PSO Decision | Merged Objective]``

**Mode 3: All Separated**

Both decision and objective spaces separated for each algorithm:

.. code-block:: python

    create_optimization_animation(
        pkl_path=['./Data/GA/GA_P1_1.pkl',
                  './Data/DE/DE_P1_1.pkl'],
        merge=3,
        algorithm_order=['GA', 'DE'],
        max_nfes=10000
    )

Layout: ``[GA Dec | DE Dec | GA Obj | DE Obj]``

Class Initialization
~~~~~~~~~~~~~~~~~~~~

For advanced usage, directly instantiate the ``OptimizationAnimator`` class:

.. code-block:: python

    from Methods.optimization_animator import OptimizationAnimator

    animator = OptimizationAnimator(
        pkl_path='./Data/GA/GA_P1_1.pkl',
        output_path='./Results/animation.gif',
        fps=10,
        dpi=100,
        merge=0,
        title='My Optimization',
        algorithm_order=None,
        max_nfes=10000
    )

    # Generate animation
    animator.create_animation(interval=100)

**Parameters:**

- ``pkl_path``: Path to .pkl file(s), string for single file or list for merge mode
- ``output_path``: Output file path (optional, auto-generated if None)
- ``fps``: Frames per second (default: 10)
- ``dpi``: Resolution, affects file size and quality (default: 100)
- ``merge``: Comparison mode (0-3, default: 0)
- ``title``: Custom title for the animation (optional)
- ``algorithm_order``: List of algorithm names specifying display order (merge mode only)
- ``max_nfes``: Maximum NFEs, scalar or list for multi-task problems (default: 100)

NFEs Configuration
~~~~~~~~~~~~~~~~~~

The ``max_nfes`` parameter controls the x-axis scale for convergence curves in single-objective optimization.

**Scalar NFEs (Same for All Tasks):**

.. code-block:: python

    # All tasks use the same NFEs
    create_optimization_animation(
        pkl_path='results.pkl',
        max_nfes=10000  # All tasks: 10000 NFEs
    )

**List NFEs (Different per Task):**

.. code-block:: python

    # Multi-task problem with different NFEs per task
    create_optimization_animation(
        pkl_path='multi_task_results.pkl',
        max_nfes=[5000, 10000, 15000]  # Task 1: 5000, Task 2: 10000, Task 3: 15000
    )

**Automatic Compatibility:**

The system automatically handles single-task and multi-task scenarios:

.. code-block:: python

    # Single-task optimization
    create_optimization_animation('single_task.pkl', max_nfes=1000)

    # Multi-task optimization
    create_optimization_animation('multi_task.pkl', max_nfes=[1000, 2000])

Algorithm Order
~~~~~~~~~~~~~~~

Control the display order of algorithms in merge modes:

.. code-block:: python

    # Specify custom order
    create_optimization_animation(
        pkl_path=['BO-LCB-BCKT.pkl', 'BO.pkl', 'MTBO.pkl', 'RAMTEA.pkl'],
        merge=2,
        algorithm_order=['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT'],
        max_nfes=10000
    )

**Behavior:**

- Algorithms are reordered according to ``algorithm_order``
- Missing algorithms in the list are excluded with a warning
- Extra files not in ``algorithm_order`` are ignored
- If ``algorithm_order=None``, uses the original order from ``pkl_path``

Output Formats
~~~~~~~~~~~~~~

**GIF Format (Default):**

.. code-block:: python

    # Explicit GIF
    create_optimization_animation('results.pkl', format='gif')

    # Or specify output file
    create_optimization_animation('results.pkl', output_path='animation.gif')

**MP4 Format (Requires FFmpeg):**

.. code-block:: python

    # Explicit MP4
    create_optimization_animation('results.pkl', format='mp4')

    # Or specify output file
    create_optimization_animation('results.pkl', output_path='animation.mp4')

**Note:** MP4 requires FFmpeg installation:

.. code-block:: bash

    pip install ffmpeg-python

If FFmpeg is unavailable, the system automatically falls back to GIF format.

Quality Settings
~~~~~~~~~~~~~~~~

Adjust animation quality through ``fps`` and ``dpi`` parameters:

.. code-block:: python

    # High quality, larger file
    create_optimization_animation(
        'results.pkl',
        fps=20,      # Smoother animation
        dpi=150,     # Higher resolution
        format='mp4'
    )

    # Fast generation, smaller file
    create_optimization_animation(
        'results.pkl',
        fps=8,       # Fewer frames
        dpi=70,      # Lower resolution
        format='gif'
    )

**Recommended Settings:**

- **Preview/Draft**: ``fps=8, dpi=70``
- **Standard**: ``fps=10, dpi=100`` (default)
- **Publication**: ``fps=15, dpi=150``

Batch Processing
~~~~~~~~~~~~~~~~

**Automatic Scanning:**

.. code-block:: python

    # Scan ./TestData and save to ./TestResults
    results = create_optimization_animation(
        data_path='./TestData',
        save_path='./TestResults',
        max_nfes=10000,
        fps=10,
        dpi=100
    )

    # Check results
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")

**Batch with Merge Mode:**

.. code-block:: python

    # Automatically scan and merge all files
    create_optimization_animation(
        data_path='./Data',
        save_path='./Animations',
        merge=1,
        title='All Algorithms Comparison',
        algorithm_order=['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT'],
        max_nfes=[5000, 10000],  # Two tasks
        format='mp4'
    )

**Custom File Pattern:**

.. code-block:: python

    # Only process specific files
    create_optimization_animation(
        data_path='./Data',
        pattern='GA_*.pkl',  # Only GA results
        save_path='./Animations',
        max_nfes=10000
    )

Complete Example
~~~~~~~~~~~~~~~~

**Single-Objective Optimization:**

.. code-block:: python

    from Methods.optimization_animator import create_optimization_animation

    # Individual animations
    create_optimization_animation(
        data_path='./Data',
        save_path='./Animations/Individual',
        merge=0,
        max_nfes=10000,
        fps=10,
        dpi=100,
        format='gif'
    )

    # Merged comparison
    create_optimization_animation(
        pkl_path=['./Data/GA/GA_P1_1.pkl',
                  './Data/DE/DE_P1_1.pkl',
                  './Data/PSO/PSO_P1_1.pkl'],
        output_path='./Animations/comparison.mp4',
        merge=2,
        title='SO Algorithm Comparison',
        algorithm_order=['GA', 'DE', 'PSO'],
        max_nfes=10000,
        fps=15,
        dpi=120,
        format='mp4'
    )

**Multi-Objective Multi-Task Optimization:**

.. code-block:: python

    # Multi-task with different NFEs
    create_optimization_animation(
        pkl_path=['./Data/NSGAII/NSGAII_DTLZ_1.pkl',
                  './Data/MOEAD/MOEAD_DTLZ_1.pkl',
                  './Data/MyAlgo/MyAlgo_DTLZ_1.pkl'],
        output_path='./Animations/MO_comparison.gif',
        merge=3,
        title='Multi-Objective Comparison',
        algorithm_order=['NSGA-II', 'MOEA/D', 'MyAlgo'],
        max_nfes=[5000, 8000, 10000],  # Different NFEs for 3 tasks
        fps=12,
        dpi=100
    )

Command Line Usage
~~~~~~~~~~~~~~~~~~

The animation generator can be used from the command line:

.. code-block:: bash

    # Single file
    python -m Methods.optimization_animator results.pkl output.gif 10 100

    # Auto-scan mode (no arguments)
    python -m Methods.optimization_animator

**Arguments:**

1. ``pkl_file``: Path to .pkl file
2. ``output_file``: Output path (optional)
3. ``fps``: Frames per second (optional, default: 10)
4. ``dpi``: Resolution (optional, default: 100)

Output Structure
~~~~~~~~~~~~~~~~

**Individual Mode (merge=0):**

.. code-block:: text

    Animations/
    ├── GA_P1_1_animation.gif
    ├── GA_P1_2_animation.gif
    ├── DE_P1_1_animation.gif
    └── PSO_P1_1_animation.gif

**Merge Mode (merge>0):**

.. code-block:: text

    Animations/
    └── test_animation.gif  # Default name if title not specified

**Custom Title:**

.. code-block:: text

    Animations/
    └── Algorithm_Comparison_animation.gif

Console Output Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ======================================================================
    Optimization Animation Generator
    ======================================================================
    Data path: ./TestData
    Save path: ./TestResults
    Found 4 result files
    Animation params: FPS=10, DPI=100, Interval=100ms, Format=GIF
    Max NFEs: [5000, 10000]
    Mode: MERGE (Decision Separated)
    Algorithm Order: ['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT']
    ======================================================================

    Creating merged comparison animation...
    Original algorithms: ['BO-LCB-BCKT', 'BO', 'MTBO', 'RAMTEA']
    Ordered algorithms: ['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT']
    Generating animation... (this may take a while)
    Animation saved to: ./TestResults/test_animation.gif
      ✓ Success

    ======================================================================
    Processing Complete!
    Merged animation: Success
    ======================================================================
    Animations saved to: ./TestResults
    ======================================================================

Best Practices
~~~~~~~~~~~~~~

1. **Use MP4 for publication quality:**

   MP4 files are typically smaller and higher quality than GIF for the same content.

2. **Adjust frame rate based on convergence speed:**

   - Slow convergence (>1000 generations): ``fps=8-10``
   - Medium convergence (100-1000 generations): ``fps=10-15``
   - Fast convergence (<100 generations): ``fps=15-20``

3. **Balance DPI and file size:**

   - For presentations: ``dpi=100-120``
   - For papers: ``dpi=120-150``
   - For web sharing: ``dpi=70-100``

4. **Use merge mode 2 for comparing many algorithms:**

   Mode 2 allows clear visualization of individual decision spaces while comparing objectives together.

5. **Specify max_nfes consistently:**

   Ensure ``max_nfes`` matches your actual experimental setup for accurate NFEs display.

6. **Use algorithm_order for clarity:**

   Order algorithms logically (e.g., baseline first, variants after) for easier comparison.

Integration with Batch Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine with ``BatchExperiment`` for complete workflow:

.. code-block:: python

    from Methods.batch_experiment import BatchExperiment
    from Methods.optimization_animator import create_optimization_animation

    # Step 1: Run batch experiments
    batch_exp = BatchExperiment(base_path='./Data')
    batch_exp.add_problem(problem_creator=problem.P1, problem_name='P1')
    batch_exp.add_algorithm(algorithm_class=GA, algorithm_name='GA', n=100, max_nfes=10000)
    batch_exp.add_algorithm(algorithm_class=DE, algorithm_name='DE', n=100, max_nfes=10000)
    batch_exp.run(n_runs=30)

    # Step 2: Generate animations for first run of each algorithm
    create_optimization_animation(
        pkl_path=['./Data/GA/GA_P1_1.pkl',
                  './Data/DE/DE_P1_1.pkl'],
        merge=2,
        title='GA vs DE on P1',
        algorithm_order=['GA', 'DE'],
        max_nfes=10000,
        format='mp4'
    )

Troubleshooting
~~~~~~~~~~~~~~~

**Issue: "FFMpeg not installed, falling back to GIF"**

Solution: Install FFmpeg:

.. code-block:: bash

    pip install ffmpeg-python

**Issue: Animation file is too large**

Solutions:

- Reduce ``dpi`` (e.g., from 100 to 70)
- Reduce ``fps`` (e.g., from 15 to 8)
- Use MP4 instead of GIF
- Reduce number of frames by using fewer generations in data

**Issue: "Incompatible data: file X has Y tasks, expected Z"**

Solution: Ensure all .pkl files have the same number of tasks when using merge mode.

**Issue: "Algorithm names not found in pkl_paths"**

Solution: Check that algorithm names in ``algorithm_order`` match the file stems (filenames without .pkl).

**Issue: Animation generation is slow**

Solutions:

- Reduce ``dpi`` for faster processing
- Use fewer data points (subsample generations)
- Process files in smaller batches
- Use fewer algorithms in merge mode

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
* :ref:`api` - Complete API documentation
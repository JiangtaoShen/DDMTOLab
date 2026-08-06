.. _methods:

Methods
=======

This chapter introduces the utility modules provided by **D²MTOLab**, including batch experiments, data analysis, performance metrics, and algorithm components. These modules provide standardized testing workflows and rich algorithm building tools.

Batch Experiments
-----------------

.. code-block:: python

    from ddmtolab.Methods.batch_experiment import BatchExperiment

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

    from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
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

    from ddmtolab.Algorithms.STSO.GA import GA
    from ddmtolab.Algorithms.STSO.DE import DE
    from ddmtolab.Algorithms.STSO.PSO import PSO

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

    batch_exp.run(n_runs=30, verbose=True, max_workers=8, base_seed=42)

**Parameters:**

- ``n_runs``: Number of independent runs for each algorithm on each problem
- ``verbose``: Whether to print detailed progress information, default: ``True``
- ``max_workers``: Maximum number of parallel worker processes, default: CPU core count
- ``base_seed``: Base seed for system-level random-seed control, default: ``None`` (unseeded)

Reproducible Runs
~~~~~~~~~~~~~~~~~

Passing ``base_seed`` executes run ``r`` (1-indexed) under seed
``base_seed + r - 1``. The seed is applied to the global ``random``, NumPy and
PyTorch (including CUDA) generators inside the worker process, *before* the
problem and the algorithm are constructed, so every individual run is
reproducible from its own seed:

.. code-block:: python

    batch_exp.run(n_runs=30, max_workers=8, base_seed=42)

Because each run is seeded independently rather than the batch as a whole,
changing ``max_workers`` does not change any result -- the degree of
parallelism and reproducibility are decoupled. Re-running the same experiment
with the same ``base_seed`` reproduces it run for run.

The per-run seed is recorded in the ``Seed`` column of the timing summary CSV,
and ``base_seed`` itself is written to ``experiment_config.yaml``, so a batch
loaded with ``BatchExperiment.from_config`` repeats under the same seeds. With
``base_seed=None`` no seeding is performed and the ``Seed`` column stays empty.

.. note::

   A fixed seed does not guarantee bit-identical results across different
   machines or library versions, and a few algorithms that train neural
   networks may still vary because of non-deterministic PyTorch kernels, which
   seeding alone does not control.

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

    from ddmtolab.Methods.data_analysis import DataAnalyzer

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
        statistic_type='mean',           # 'mean', 'median', 'max', 'min', 'median_iqr'
        significance_level=0.05,         # Significance level for tests
        rank_sum_test=True,              # Whether to perform rank-sum test
        holm_correction=False,           # Holm-Bonferroni correct the p-values
        effect_size=False,               # Report Cliff's delta per comparison
        friedman_test=False,             # Append Friedman rows to the table
        friedman_control=None,           # Control algorithm for the post-hoc
        cd_diagram=False,                # Draw the critical difference diagram
        cd_alpha=None,                   # Its alpha (defaults to significance_level)
        log_scale=False,                 # Whether to use log scale
        show_pf=True,                    # Whether to show true Pareto front
        show_nd=True,                    # Whether to show only non-dominated
        best_so_far=True,                # Whether to use best-so-far values
        clear_results=True               # Whether to clear results folder
    )

Metric Configuration
~~~~~~~~~~~~~~~~~~~~

For problems requiring complex metrics (e.g., multiobjective optimization), provide a ``settings`` configuration dictionary:

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

Statistical Comparison
~~~~~~~~~~~~~~~~~~~~~~

The platform implements the two standard methodologies for comparing
algorithms:

- **[D06]** J. Demsar, *Statistical Comparisons of Classifiers over Multiple
  Data Sets*, JMLR 7 (2006) 1-30.
- **[D11]** J. Derrac, S. Garcia, D. Molina, F. Herrera, *A practical tutorial
  on the use of nonparametric statistical tests as a methodology for comparing
  evolutionary and swarm intelligence algorithms*, Swarm and Evolutionary
  Computation 1 (2011) 3-18.

Everything below is **disabled by default**, so existing scripts keep producing
identical output.

Two questions, two families of tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two families answer different questions and must not be confused:

**Per-instance** -- "on *this* problem, do the runs of A differ from the runs of
B?" One sample per run, and the answer is a cell annotation in the results
table. This is the ``+``/``-``/``=`` column of ``results_table_*.xlsx``,
produced by a Wilcoxon **rank-sum** test against the baseline (the last entry of
``algorithm_order``).

**Multi-problem** -- "across the whole benchmark suite, is A better than B?"
One value per algorithm-problem instance, obtained by collapsing that instance's
runs into the displayed statistic. This is what [D06] and [D11] are about, and
what :mod:`ddmtolab.Methods.statistical_tests` implements.

============================================  ==================================================
Question                                      Method
============================================  ==================================================
Runs of two algorithms on one problem         ``rank_sum_test=True`` (on by default)
...corrected over many instances              ``holm_correction=True``
...and how large is the difference            ``effect_size=True`` (Cliff's delta)
Two algorithms over the whole suite           ``sign_test``, ``wilcoxon_signed_rank_test``
Do k algorithms differ at all                 ``friedman_test`` (+ Iman-Davenport),
                                              ``friedman_aligned_test``, ``quade_test``
Which algorithms differ from *my* method      ``control_post_hoc`` (7 procedures)
Which algorithms differ from each other       ``all_pairs_post_hoc`` (4 procedures)
...as a picture                               ``cd_diagram=True``
By how much do they differ                    ``contrast_estimation``
All of the above in one file                  ``multi_problem_report=True``
============================================  ==================================================

Per-instance options
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    analyzer = DataAnalyzer(
        data_path='./Data',
        statistic_type='median_iqr',   # report median[IQR] instead of mean (std)
        holm_correction=True,          # correct for testing many instances
        effect_size=True,              # how large the difference is
        friedman_test=True,            # rank all algorithms at once
        cd_diagram=True,               # draw the critical difference diagram
        cd_alpha=0.10,                 # the alpha Demsar's own diagrams use
        multi_problem_report=True,     # the full [D11] analysis in one workbook
    )
    analyzer.run()

**Holm-Bonferroni correction** (``holm_correction``)
    A table compares every algorithm against the baseline on every instance, so
    a handful of "significant" results is expected by chance alone. The
    correction treats all those comparisons as one family, sorts the p-values,
    scales the i-th smallest of m by ``m - i`` and enforces monotonicity. The
    ``+``/``-``/``=`` symbols then follow the corrected p-values, which is
    stricter: a comparison can only lose significance this way, never gain it.
    The raw p-value stays available in ``ComparisonResult.p_value`` and the
    corrected one in ``p_adjusted``.

**Cliff's delta** (``effect_size``)
    Significance says a difference is unlikely to be noise; it says nothing
    about how big it is. Cliff's delta measures exactly that, as the difference
    between the probability that one algorithm beats the other and the reverse.
    It is reported as a separate bracketed field in the cell, never folded into
    the symbol, with the usual thresholds: ``negligible`` (``|delta| < 0.147``),
    ``small`` (< 0.33), ``medium`` (< 0.474), ``large`` otherwise. The sign is
    oriented so that a positive value always means the algorithm is better.

**Friedman test** (``friedman_test``, ``friedman_control``)
    A single omnibus test over all algorithms and instances at once, rather
    than many pairwise ones. It appends two rows to the table: the average rank
    of each algorithm, labelled with the chi-squared statistic, the
    Iman-Davenport ``F_F`` correction of it (which is less conservative) and
    their p-values, and Holm-corrected post-hoc comparisons against a control
    algorithm (``friedman_control``, defaulting to the baseline). It needs at
    least three algorithms and two instances, and raises a ``ValueError``
    otherwise rather than reporting a meaningless number.

**Critical difference diagram** (``cd_diagram``, ``cd_alpha``)
    The all-pairs counterpart of the Friedman post-hoc rows, drawn as Figure
    1(a) of Demsar (2006) and saved as ``cd_diagram.<figure_format>``. Average
    ranks are plotted on an axis turned so the best rank is on the right, and
    groups of algorithms that the Nemenyi test cannot tell apart are connected
    by a bar. Two algorithms differ significantly when their ranks differ by
    more than the critical difference shown above the axis,

    .. math:: CD = q_\alpha \sqrt{k(k+1) / (6N)}

    for k algorithms over N instances. Use it when every algorithm is compared
    against every other one; when one algorithm is the control, the Friedman
    post-hoc rows make only k-1 comparisons and are more powerful. Because the
    all-pairs test is conservative, Demsar's own diagrams use ``cd_alpha=0.10``.

**median[IQR]** (``statistic_type='median_iqr'``)
    Reports the median with its interquartile range, rendered as
    ``median[IQR]``. Useful when the distribution over runs is skewed enough
    that ``mean (std)`` misrepresents it. The existing ``'mean'`` and
    ``'median'`` statistics are unchanged.

The methods are also usable directly, independently of table generation:

.. code-block:: python

    from ddmtolab.Methods.data_analysis import (
        OptimizationDirection, PlotConfig, PlotGenerator, StatisticsCalculator)

    # Holm-Bonferroni over a family of p-values, order preserved
    StatisticsCalculator.holm_bonferroni([0.01, 0.02, 0.03])
    # -> [0.03, 0.04, 0.04]

    # Effect size of one comparison
    effect = StatisticsCalculator.cliffs_delta(algo_values, baseline_values,
                                               direction=OptimizationDirection.MINIMIZE)
    print(effect.delta, effect.magnitude)

    # Friedman over an algorithms x instances matrix
    result = StatisticsCalculator.perform_friedman_test(
        data_matrix, ['A', 'B', 'C'], control='C')
    print(result.statistic, result.p_value, result.average_ranks)
    print(result.iman_davenport_statistic, result.iman_davenport_p_value)

    # Nemenyi all-pairs post-hoc test and its diagram
    nemenyi = StatisticsCalculator.perform_nemenyi_test(
        data_matrix, ['A', 'B', 'C'], significance_level=0.10)
    print(nemenyi.critical_difference, nemenyi.cliques)
    PlotGenerator(PlotConfig(save_path='./Results')).plot_cd_diagram(nemenyi)

The matrix these tests consume can be built from the analyzer's own results with
``StatisticsCalculator.build_instance_matrix(best_values, algorithm_order,
statistic_type)``, which collapses the runs of every problem-task instance into
the same statistic the table displays.

Multi-problem analysis
^^^^^^^^^^^^^^^^^^^^^^

``multi_problem_report=True`` runs the complete [D11] methodology on that same
matrix and writes ``statistical_report.xlsx`` (or ``.tex``), one section per
step of the tutorial:

============================  ====================================================
Sheet                         Contents
============================  ====================================================
``Rankings``                  Average ranks under all three schemes, each
                              omnibus statistic and p-value, plus Iman-Davenport
``Control (<name>)``          The k-1 comparisons against the control, with the
                              adjusted p-value of all seven procedures
``All pairs``                 The k(k-1)/2 comparisons, with Nemenyi, Holm,
                              Shaffer and Bergmann-Hommel
``Contrast estimation``       Median-based estimate of every pairwise difference
``Pairwise tests``            Sign test and Wilcoxon signed-rank per pair
============================  ====================================================

.. code-block:: python

    analyzer = DataAnalyzer(
        data_path='./Data',
        multi_problem_report=True,
        report_scheme='friedman',   # or 'aligned' / 'quade'
        report_control='MyAlgo',    # default: the best-ranked algorithm
    )
    report = analyzer.generate_statistical_report()
    print(report['control'].rejected('finner', 0.05))

Where each piece lives
^^^^^^^^^^^^^^^^^^^^^^

Every test is implemented **once**, in
:mod:`ddmtolab.Methods.statistical_tests`, and follows one convention: data
first, then ``direction``, then the options; a dataclass comes back, never a
bare tuple; and the name is ``<name>_test``.

``ddmtolab.Methods.data_analysis`` is the reporting layer on top. Its
``StatisticsCalculator`` still exposes ``perform_rank_sum_test``,
``perform_friedman_test``, ``perform_nemenyi_test``, ``holm_bonferroni`` and
``cliffs_delta`` for backward compatibility, but they are thin wrappers that
delegate to the module above; new code should call the module functions.

The same tests are importable on their own for any matrix of results:

.. code-block:: python

    import numpy as np
    from ddmtolab.Methods.statistical_tests import (
        adjust_p_values, all_pairs_post_hoc, contrast_estimation, control_post_hoc,
        friedman_test, quade_test, sign_test, wilcoxon_signed_rank_test)

    matrix = np.array(...)          # (n_algorithms, n_problems)
    names = ['A', 'B', 'C', 'MyAlgo']

    # 1x1: two algorithms over the suite
    print(wilcoxon_signed_rank_test(matrix[3], matrix[0]).p_value)
    print(sign_test(matrix[3], matrix[0]).wins)

    # omnibus: do they differ at all?
    ranking = friedman_test(matrix, names)
    print(ranking.average_ranks, ranking.iman_davenport_p_value)

    # 1xN: everyone against the control
    control = control_post_hoc(ranking, control='MyAlgo')
    for hypothesis in control.hypotheses:
        print(hypothesis.label, hypothesis.p_value, hypothesis.adjusted['finner'])

    # NxN: everyone against everyone
    print(all_pairs_post_hoc(ranking).rejected('bergmann', 0.05))

    # how large are the differences?
    print(contrast_estimation(matrix, names).estimators)

**Which ranking scheme** (``report_scheme``)
    ``'friedman'`` ranks within each problem only; ``'aligned'`` first subtracts
    each problem's average so that observations from different problems become
    comparable, which is more powerful when few algorithms are compared;
    ``'quade'`` weighs each problem by how widely the algorithms spread out on
    it, so easy problems count less. All three feed the same post-hoc machinery.

**Which post-hoc procedure**
    The seven control procedures are, from least to most powerful,
    ``bonferroni`` < ``holm`` < ``hochberg`` < ``hommel``, with ``holland`` and
    ``finner`` as sharper step-down variants and ``li`` as a two-step
    alternative. For the all-pairs family, ``shaffer`` and ``bergmann`` exploit
    that the pairwise hypotheses are logically interrelated and therefore reject
    more than ``holm``; Bergmann-Hommel is enumerated only up to 10 algorithms.
    All of them are reported as *adjusted p-values*, which can be compared
    directly against any significance level.

**Practical guidance from [D11]**
    Use at least twice as many problems as algorithms (``n >= 2k``); beyond
    roughly ``n = 8k`` the tests start flagging trivial differences. Wilcoxon's
    test is the safe default for two algorithms, but never use a series of
    pairwise tests in place of a post-hoc family -- that is exactly the
    family-wise error the adjusted p-values exist to control. Check the omnibus
    p-value before reading the post-hoc rows.

Two documented deviations from the papers, both verified against their own
tables: the Friedman Aligned omnibus statistic follows Eq. (4) of [D11] (its
Table 11 value cannot be reproduced from that equation and the ranks printed
next to it), and the sign test reports the exact binomial p-value, whose
one-sided form is the one the critical-value tables of both papers tabulate.
Rom's procedure is deliberately absent: its constants in [D11] are specific to
alpha = 0.05 and are not reproducible from the recursion given for them, and
Hochberg and Hommel cover the same step-up family.

Complete Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**One-Step Analysis:**

.. code-block:: python

    from ddmtolab.Methods.data_analysis import DataAnalyzer

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

   from ddmtolab.Methods.data_analysis import DataUtils

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

   from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point

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

   When calculating multiobjective metrics (e.g., IGD), it is recommended to set ``n_ref`` to 1000 (preferably not exceeding 2000). Using too many reference points can result in very large PDF files when visualizing Pareto fronts with the true PF overlay.

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

    from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer

The ``TestDataAnalyzer`` is a lightweight version of ``DataAnalyzer`` for quickly analyzing single test runs. It directly reads the ``.pkl`` files written by a single run, without statistical tests or multi-run aggregation, suitable for algorithm development and debugging.

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
        data_path='./Data',              # Test data directory
        settings=None,                   # Problem settings (for MO)
        algorithm_order=None,            # Algorithm display order
        save_path='./Results',           # Results save path
        figure_format='pdf',             # Figure format
        log_scale=False,                 # Log scale
        show_pf=True,                    # Show true Pareto front
        show_nd=True,                    # Show only non-dominated
        best_so_far=True,                # Use best-so-far values
        clear_results=True,              # Clear results folder
        file_suffix='.pkl'               # Result file suffix
    )

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer

    # Create analyzer (settings optional for SO)
    analyzer = TestDataAnalyzer(data_path='./Data',
                               save_path='./Results')

    # Execute complete analysis
    results = analyzer.run()

Output Structure
~~~~~~~~~~~~~~~~

.. code-block:: text

    ./Results/
    ├── test_results.xlsx                # Results comparison table
    ├── P1-Task1_convergence.pdf         # Task 1 convergence
    ├── P1-Task2_convergence.pdf         # Task 2 convergence (if any)
    ├── runtime_comparison.pdf           # Runtime comparison
    └── ND_Solutions/                    # Non-dominated solutions
        ├── P1-Task1-GA.pdf
        ├── P1-Task1-DE.pdf
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
     - Single-run files (``*.pkl``) in one directory
     - Multiple repeated experiments
   * - File Structure
     - Direct test files in directory
     - Subfolders per algorithm
   * - Statistical Analysis
     - No statistical tests
     - Wilcoxon rank-sum test
   * - Table Format
     - Excel (``test_results.xlsx``)
     - Excel and LaTeX
   * - Use Case
     - Development and quick validation
     - Formal experiment analysis

Problem Definition (MTOP)
-------------------------

.. code-block:: python

    from ddmtolab.Methods.mtop import MTOP

The ``MTOP`` (Multitask Optimization Problem) class provides a unified interface for defining single-task and multitask optimization problems with support for objectives, constraints, and variable bounds.

Module Features
~~~~~~~~~~~~~~~

The ``MTOP`` class offers:

1. **Flexible Task Definition**: Add single or multiple tasks with different dimensions and objectives
2. **Constraint Support**: Define constraint functions for constrained optimization
3. **Automatic Vectorization**: Handle both vectorized and non-vectorized objective functions
4. **Unified Evaluation Mode**: Optionally pad outputs to consistent dimensions across tasks
5. **Cross-Platform Compatibility**: Pickle-compatible function wrappers for parallel execution
6. **Selective Evaluation**: Evaluate specific objectives or constraints as needed

Class Initialization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    mtop = MTOP(
        unified_eval_mode=False,  # Pad outputs to max dimensions
        fill_value=0.0            # Fill value for padding
    )

Adding Tasks
~~~~~~~~~~~~

**Single Task with Default Bounds [0, 1]:**

.. code-block:: python

    def sphere(x):
        return np.sum(x**2, axis=1)

    mtop = MTOP()
    idx = mtop.add_task(sphere, dim=3)

**Single Task with Custom Bounds:**

.. code-block:: python

    # Array bounds
    idx = mtop.add_task(sphere, dim=3, lower_bound=[-5, -5, -5], upper_bound=[5, 5, 5])

    # Scalar bounds (broadcast to all dimensions)
    idx = mtop.add_task(sphere, dim=5, lower_bound=-5, upper_bound=5)

**Multiple Tasks at Once:**

.. code-block:: python

    def f1(x): return np.sum(x**2, axis=1)
    def f2(x): return np.sum((x-1)**2, axis=1)

    indices = mtop.add_task(
        objective_func=(f1, f2),
        dim=(3, 4),
        lower_bound=([-1]*3, [-2]*4),
        upper_bound=([1]*3, [2]*4)
    )

**Task with Constraints:**

.. code-block:: python

    def constraint(x):
        return x[:, 0] - 0.5  # g(x) <= 0

    idx = mtop.add_task(sphere, dim=3, constraint_func=constraint)

**Multiobjective Task:**

.. code-block:: python

    def multi_obj(x):
        f1 = np.sum(x**2, axis=1)
        f2 = np.sum((x-1)**2, axis=1)
        return np.column_stack([f1, f2])

    idx = mtop.add_task(multi_obj, dim=3)

Evaluating Tasks
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Evaluate a single task
    X = np.random.rand(10, 3)
    objs, cons = mtop.evaluate_task(0, X)

    # Selective evaluation (evaluate only specific objectives)
    objs, cons = mtop.evaluate_task(0, X, eval_objectives=[0, 2])

    # Skip constraint evaluation
    objs, cons = mtop.evaluate_task(0, X, eval_constraints=False)

    # Evaluate multiple tasks
    X_list = [np.random.rand(10, 3), np.random.rand(10, 4)]
    objs_list, cons_list = mtop.evaluate_tasks([0, 1], X_list)

Querying Task Information
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get number of tasks
    n_tasks = mtop.n_tasks

    # Get dimensions for all tasks
    dims = mtop.dims

    # Get number of objectives/constraints for a task
    n_obj = mtop.get_n_objectives(0)
    n_con = mtop.get_n_constraints(0)

    # Get detailed task info
    info = mtop.get_task_info(0)
    # Returns: {'dimension', 'n_objectives', 'n_constraints', 'lower_bounds', 'upper_bounds', ...}

    # Print MTOP summary
    print(mtop)

Animation Generator
-------------------

.. code-block:: python

    from ddmtolab.Methods.animation_generator import AnimationGenerator, create_optimization_animation

The animation generator module provides comprehensive visualization tools for optimization processes, supporting both single-objective and multiobjective optimization with multiple comparison modes.

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

**For Multiobjective Optimization:**

- **Decision Space (Left)**: Parallel coordinate plot showing decision variable evolution
- **Objective Space (Right)**: Pareto front evolution

  - 2D: Scatter plot (f1 vs. f2)
  - 3D: 3D scatter plot with rotation view
  - High-dimensional: Parallel coordinate plot with normalized objectives

Quick Start
~~~~~~~~~~~

**Using AnimationGenerator Class:**

.. code-block:: python

    from ddmtolab.Methods.animation_generator import AnimationGenerator

    # Create generator and run
    generator = AnimationGenerator(data_path='./Data', save_path='./Results')
    generator.run()

**Single File Animation (Convenience Function):**

.. code-block:: python

    from ddmtolab.Methods.animation_generator import create_optimization_animation

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

Instantiate the ``AnimationGenerator`` class for batch processing:

.. code-block:: python

    from ddmtolab.Methods.animation_generator import AnimationGenerator

    generator = AnimationGenerator(
        data_path='./Data',           # Directory containing .pkl files
        save_path='./Results',        # Output directory
        algorithm_order=['GA', 'DE'], # Optional: specify display order
        title='My Comparison',        # Optional: custom title
        merge=0,                      # Merge mode (0-3)
        max_nfes=10000,               # Max function evaluations
        fps=10,                       # Frames per second
        dpi=100,                      # Resolution
        interval=100,                 # Frame interval (ms)
        format='gif',                 # Output format
        log_scale=False,              # Log scale for SO
        file_suffix='.pkl'            # File pattern
    )

    # Execute the pipeline
    results = generator.run()

**Parameters:**

- ``pkl_path``: Path to .pkl file(s), string for single file or list for merge mode
- ``output_path``: Output file path (optional, auto-generated if None)
- ``fps``: Frames per second (default: 10)
- ``dpi``: Resolution, affects file size and quality (default: 100)
- ``merge``: Comparison mode (0-3, default: 0)
- ``title``: Custom title for the animation (optional)
- ``algorithm_order``: List of algorithm names specifying display order (merge mode only)
- ``max_nfes``: Maximum NFEs, scalar or list for multitask problems (default: 100)

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

    # Multitask problem with different NFEs per task
    create_optimization_animation(
        pkl_path='multi_task_results.pkl',
        max_nfes=[5000, 10000, 15000]  # Task 1: 5000, Task 2: 10000, Task 3: 15000
    )

**Automatic Compatibility:**

The system automatically handles single-task and multitask scenarios:

.. code-block:: python

    # Single-task optimization
    create_optimization_animation('single_task.pkl', max_nfes=1000)

    # Multitask optimization
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

    # Scan ./Data and save to ./Results
    results = create_optimization_animation(
        data_path='./Data',
        save_path='./Results',
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

    from ddmtolab.Methods.animation_generator import create_optimization_animation

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

**Multiobjective Multitask Optimization:**

.. code-block:: python

    # Multitask with different NFEs
    create_optimization_animation(
        pkl_path=['./Data/NSGAII/NSGAII_DTLZ_1.pkl',
                  './Data/MOEAD/MOEAD_DTLZ_1.pkl',
                  './Data/MyAlgo/MyAlgo_DTLZ_1.pkl'],
        output_path='./Animations/MO_comparison.gif',
        merge=3,
        title='Multiobjective Comparison',
        algorithm_order=['NSGA-II', 'MOEA/D', 'MyAlgo'],
        max_nfes=[5000, 8000, 10000],  # Different NFEs for 3 tasks
        fps=12,
        dpi=100
    )

Command Line Usage
~~~~~~~~~~~~~~~~~~

Running the module directly executes a demo that scans ``./Data`` and writes GIFs to
``./Results`` with the default settings. It takes no positional arguments:

.. code-block:: bash

    python -m ddmtolab.Methods.animation_generator

To control the data path, output format, frame rate or resolution, call
``create_optimization_animation`` (or construct ``AnimationGenerator``) from Python as
shown above.

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
    Data path: ./Data
    Save path: ./Results
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
    Animation saved to: ./Results/test_animation.gif
      ✓ Success

    ======================================================================
    Processing Complete!
    Merged animation: Success
    ======================================================================
    Animations saved to: ./Results
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

    from ddmtolab.Methods.batch_experiment import BatchExperiment
    from ddmtolab.Methods.animation_generator import create_optimization_animation

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

    from ddmtolab.Methods.metrics import IGD, HV, GD, IGDp, FR, CV, DeltaP, Spacing, Spread

The performance metrics module provides comprehensive implementations of optimization algorithm evaluation metrics with a unified interface design.

Module Features
~~~~~~~~~~~~~~~

The metric module follows these design principles:

1. **Unified Interface**: All metric classes follow the same interface specification
2. **Direction Indicator**: Each metric has a ``sign`` attribute (``-1`` for minimization, ``1`` for maximization)
3. **Callable Support**: Metric instances support functional calling (``__call__`` method)

Available Metrics
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 10 75

   * - Metric
     - Sign
     - Description
   * - ``IGD``
     - -1
     - Inverted Generational Distance. Measures convergence and diversity by averaging distances from Pareto front points to nearest obtained solutions.
   * - ``GD``
     - -1
     - Generational Distance. Measures convergence by averaging distances from obtained solutions to the nearest Pareto front points.
   * - ``IGDp``
     - -1
     - Inverted Generational Distance Plus. A modified IGD that only penalizes dominated portions, making it Pareto-compliant.
   * - ``HV``
     - +1
     - Hypervolume. Measures the volume of objective space dominated by the obtained solutions, normalized to [0, 1]. Supports 2D/3D exact calculation and Monte Carlo for higher dimensions.
   * - ``DeltaP``
     - -1
     - Averaged Hausdorff Distance. Maximum of GD and IGD, measuring both convergence and diversity.
   * - ``Spacing``
     - -1
     - Spacing metric. Measures the standard deviation of nearest neighbor distances, indicating solution uniformity.
   * - ``Spread``
     - -1
     - Spread metric. Measures distribution uniformity relative to extreme points of the Pareto front.
   * - ``FR``
     - +1
     - Feasible Rate. Proportion of feasible solutions in the population (for constrained optimization).
   * - ``CV``
     - -1
     - Constraint Violation. Sum of constraint violations for the best solution (for constrained optimization).

Metric Interface
~~~~~~~~~~~~~~~~

All metric classes follow this template:

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

Usage Examples
~~~~~~~~~~~~~~

**Multiobjective Metrics (IGD, GD, HV, etc.):**

.. code-block:: python

    from ddmtolab.Methods.metrics import IGD, HV, GD, IGDp, DeltaP, Spacing, Spread
    import numpy as np

    # Obtained solutions and true Pareto front
    objs = np.random.rand(100, 2)  # 100 solutions, 2 objectives
    pf = np.random.rand(1000, 2)   # True Pareto front

    # IGD (requires Pareto front)
    igd = IGD()
    igd_value = igd(objs, pf)

    # GD (requires Pareto front)
    gd = GD()
    gd_value = gd(objs, pf)

    # IGD+ (Pareto-compliant version)
    igdp = IGDp()
    igdp_value = igdp(objs, pf)

    # HV (requires Pareto front or reference point)
    hv = HV()
    hv_value = hv(objs, pf=pf)  # Reference placed 10% beyond the front's nadir
    hv_value = hv(objs, reference=np.array([2.0, 2.0]))  # Reference given directly

The objectives are normalized so that the reference point maps to 1 in every
dimension, which puts HV on the [0, 1] scale whichever way the reference is
supplied. The two calls above therefore agree exactly when the reference point
is the one the front implies:
``hv(objs, pf=pf) == hv(objs, reference=fmin + 1.1 * (pf.max(0) - fmin))``,
with ``fmin = minimum(objs.min(0), 0)``.

    # Averaged Hausdorff Distance
    deltap = DeltaP()
    deltap_value = deltap(objs, pf)

    # Spacing (only requires obtained solutions)
    spacing = Spacing()
    spacing_value = spacing(objs)

    # Spread (requires Pareto front)
    spread = Spread()
    spread_value = spread(objs, pf)

**Constrained Optimization Metrics (FR, CV):**

.. code-block:: python

    from ddmtolab.Methods.metrics import FR, CV
    import numpy as np

    # Constraint values (n_solutions x n_constraints)
    # Constraint satisfied when cons <= 0
    cons = np.array([
        [-0.1, -0.2],  # Feasible (all <= 0)
        [0.5, -0.1],   # Infeasible
        [-0.3, 0.2],   # Infeasible
        [-0.1, -0.5],  # Feasible
    ])

    # Feasible Rate (proportion of feasible solutions)
    fr = FR()
    fr_value = fr(cons)  # Returns 0.5 (2 out of 4 feasible)

    # Constraint Violation (minimum CV in population)
    cv = CV()
    cv_value = cv(cons)  # Returns 0.0 (best solution has CV=0)

Algorithm Components
--------------------

Algorithm Utilities
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ddmtolab.Methods.Algo_Methods.algo_utils import *

The algorithm utilities module provides a complete toolkit for building optimization algorithms, including population initialization, evaluation, selection, mutation, crossover, and auxiliary functions.

.. list-table:: Key Functions in algo_utils
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``initialization``
     - Initialize multitask decision variable matrices with Random or LHS sampling
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
     - Convert single parameter to multitask parameter list
   * - ``get_algorithm_information``
     - Extract and print algorithm metadata

Bayesian Optimization Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ddmtolab.Methods.Algo_Methods.bo_utils import *

The BO utilities module provides core Bayesian optimization functionalities based on BoTorch and GPyTorch, including single-task and multitask Gaussian process modeling.

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
     - Build multitask Gaussian process model
   * - ``mtgp_predict``
     - Predict for specified task using multitask GP
   * - ``mtgp_task_corr``
     - Extract task correlation matrix from multitask GP
   * - ``mtbo_next_point``
     - Get next sampling point via multitask BO

Similarity Evaluation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ddmtolab.Methods.Algo_Methods.sim_evaluation import *

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

    from ddmtolab.Methods.Algo_Methods.uniform_point import *

The uniform point generation module provides various methods for generating uniformly distributed points for multiobjective optimization and decision space sampling.

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
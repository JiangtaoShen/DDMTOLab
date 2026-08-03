.. _algorithms:

Algorithms
==========

This chapter introduces the algorithm design philosophy and construction rules in **D²MTOLab**, providing comprehensive guidance for implementing custom optimization algorithms.

Algorithm Construction
----------------------

Considering the complexity and diversity of data-driven multitask optimization, **D²MTOLab** adopts a **loosely-coupled algorithm design philosophy**. The platform does not mandate algorithms to inherit specific base classes or implement fixed interface methods, thereby avoiding restrictions on algorithm flexibility. This design approach offers the following advantages:

1. **Enhanced Platform Compatibility**: Traditional gradient-based methods, evolutionary algorithms, advanced data-driven multitask optimization algorithms, and hybrid innovative architectures can all be seamlessly integrated into the platform.

2. **Improved Development Convenience**: Users can quickly implement algorithms across the full spectrum—from inexpensive single-task single-objective unconstrained optimization to expensive multitask multiobjective constrained optimization—without understanding complex class inheritance hierarchies.

3. **Guaranteed Algorithm Freedom**: Users are free to design data structures, optimization workflows, and knowledge transfer strategies according to specific problem characteristics and algorithm mechanisms, without framework constraints.

To facilitate subsequent data processing and efficient coordination with the platform's experiment modules and data analysis modules, **D²MTOLab** imposes only **3 basic rules** on algorithm construction, ensuring normal platform functionality while maximizing algorithm development flexibility.

Rule 1: Algorithm Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms must be implemented as **classes** and include the following core components:

1. **Algorithm Metadata**: Class attribute ``algorithm_information`` dictionary declaring the algorithm's basic characteristics
2. **Metadata Access Method**: Class method ``get_algorithm_information`` for retrieving and displaying algorithm metadata
3. **Initialization Method**: ``__init__`` method that must accept a ``problem`` (MTOP instance) as the first parameter
4. **Optimization Method**: ``optimize`` method that executes the optimization process and returns a ``Results`` object

**Example Structure**:

.. code-block:: python

    class AlgorithmName:
        # Component 1: Algorithm metadata (required, exactly these ten keys in this order)
        algorithm_information = {
            'n_tasks': '[1, K]',            # Supported task number types
            'dims': 'unequal',              # Decision variable dimension constraint
            'objs': 'equal',                # Objective number constraint
            'n_objs': '1',                  # Objective quantity type
            'cons': 'unequal',              # Constraint number constraint
            'n_cons': '[0, C]',             # Constraint quantity type
            'expensive': 'False',           # Whether expensive optimization
            'knowledge_transfer': 'False',  # Whether knowledge transfer involved
            'n': 'unequal',                 # Population size constraint
            'max_nfes': 'unequal'           # Evaluation budget constraint
        }

        # Component 2: Metadata access method (required)
        @classmethod
        def get_algorithm_information(cls, print_info=True):
            return get_algorithm_information(cls, print_info)

        # Component 3: Initialization method (required)
        def __init__(self, problem, n=None, max_nfes=None, ...,
                     save_data=True, save_path='./Data', name='AlgorithmName',
                     disable_tqdm=True):
            self.problem = problem
            # Other parameter initialization

        # Component 4: Optimization method (required)
        def optimize(self):
            # Algorithm implementation
            return results

The ninth key is ``n`` for purely population-based algorithms and ``n_initial`` for
algorithms with a design-of-experiments phase; the constructor must expose a parameter
of that same name. ``__init__`` always ends with the four standard parameters
``save_data``, ``save_path``, ``name`` and ``disable_tqdm``, whose defaults are
``True``, ``'./Data'``, the algorithm's display name (file name with underscores
replaced by hyphens) and ``True``.

Rule 2: Algorithm Input
~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms must accept an ``MTOP`` instance as an input parameter. The ``MTOP`` instance encapsulates complete information about the optimization problem, through which the algorithm obtains all problem information. Other parameters can be freely designed according to algorithm requirements.

**Example**:

.. code-block:: python

    def __init__(self, problem, n=None, max_nfes=None, ...):
        """
        Args:
            problem: MTOP instance (required parameter)
            n: Population size per task (custom parameter)
            ...: Other algorithm-specific parameters
        """
        self.problem = problem  # Store problem instance
        # Other parameter initialization

Rule 3: Algorithm Output
~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms must return a result object conforming to the ``Results`` dataclass specification. The ``Results`` class encapsulates complete information about the optimization process:

**Results Dataclass Definition**:

.. code-block:: python

    @dataclass
    class Results:
        """Optimization results container"""
        best_decs: List[np.ndarray]      # Best decision variables for each task
        best_objs: List[np.ndarray]      # Best objective values for each task
        all_decs: List[List[np.ndarray]] # Decision variable evolution history
        all_objs: List[List[np.ndarray]] # Objective value evolution history
        runtime: float                    # Total runtime (seconds)
        max_nfes: List[int]              # Max function evaluations per task
        best_cons: Optional[List[np.ndarray]] = None  # Best constraint values
        all_cons: Optional[List[List[np.ndarray]]] = None  # Constraint history

**Results Fields Description**

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Field
     - Data Type
     - Description
   * - ``best_decs``
     - ``List[np.ndarray]``
     - **Best decision variables**. List length is the number of tasks K. ``best_decs[i]`` is the best decision variable for task i. For single-objective tasks this is the single best row, shape :math:`(D^i,)`; for multiobjective tasks it is the whole final population, shape :math:`(n, D^i)`
   * - ``best_objs``
     - ``List[np.ndarray]``
     - **Best objective values**. List length is K. ``best_objs[i]`` is the best objective value for task i. Shape :math:`(M^i,)` for single-objective tasks and :math:`(n, M^i)` for multiobjective tasks
   * - ``all_decs``
     - ``List[List[np.ndarray]]``
     - **Decision variable history**. ``all_decs[i][g]`` represents all decision variables of task i at generation g. Shape is :math:`(n, D^i)`
   * - ``all_objs``
     - ``List[List[np.ndarray]]``
     - **Objective value history**. ``all_objs[i][g]`` represents all objective values of task i at generation g. Shape is :math:`(n, M^i)`
   * - ``runtime``
     - ``float``
     - **Total runtime** (seconds). Records total time from start to end for performance evaluation
   * - ``max_nfes``
     - ``List[int]``
     - **Maximum function evaluations**. List length is K. ``max_nfes[i]`` is the maximum number of function evaluations for task i
   * - ``best_cons``
     - ``Optional[List[np.ndarray]]``
     - **Best constraint values** (optional). ``best_cons[i]`` is the constraint value corresponding to the best solution of task i, shape :math:`(n, C^i)`. It is ``None`` when the algorithm does not pass ``all_cons`` to ``build_save_results``; algorithms that declare ``n_cons: '[0, C]'`` must pass it
   * - ``all_cons``
     - ``Optional[List[List[np.ndarray]]]``
     - **Constraint evolution history** (optional). ``all_cons[i][g]`` represents all constraint values of task i at generation g. Shape is :math:`(n, C^i)`. ``None`` when constraints are not tracked

The input/output structure is straightforward: **input must include an MTOP instance, and output must follow the specified data structure**.

Algorithm Metadata
------------------

Algorithms must declare their basic characteristics through the ``algorithm_information`` class attribute dictionary to facilitate algorithm management, experiment matching, and performance analysis. The key fields are described below:

**Metadata Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Description
   * - ``n_tasks``
     - Supported task numbers. ``'[1, K]'`` means both single-task and multitask are supported, ``'[2, K]'`` means multitask only (K≥2)
   * - ``dims``
     - Decision variable dimension constraint. ``equal`` requires same dimensions across tasks, ``unequal`` supports unequal-dimension tasks
   * - ``objs``
     - Objective number constraint. ``equal`` requires same number of objectives across tasks, ``unequal`` supports unequal objective numbers
   * - ``n_objs``
     - Objective quantity type. ``'1'`` means single-objective only, ``'[2, M]'`` means multiobjective (M≥2), ``'[2, 3]'`` restricts to two or three objectives
   * - ``cons``
     - Constraint number constraint. ``equal`` requires same number of constraints across tasks, ``unequal`` supports unequal constraint numbers. Pairs with ``n_cons``: ``'0'`` implies ``equal``, ``'[0, C]'`` implies ``unequal``
   * - ``n_cons``
     - Constraint quantity type. ``'0'`` means unconstrained only, ``'[0, C]'`` means constraints are supported and threaded through selection and into the returned ``Results``
   * - ``expensive``
     - Whether expensive optimization (involving surrogate models). ``True`` uses surrogate models, ``False`` does not
   * - ``knowledge_transfer``
     - Whether inter-task knowledge transfer involved. ``True`` means the algorithm includes knowledge transfer mechanisms, ``False`` means tasks are optimized independently
   * - ``n`` / ``n_initial``
     - Sizing parameter constraint. Use ``n`` (population size) for population-based algorithms and ``n_initial`` (initial sample count) for algorithms with a design-of-experiments phase. ``equal`` requires the same value across tasks, ``unequal`` allows a per-task list
   * - ``max_nfes``
     - Evaluation budget constraint. ``equal`` requires a single shared budget, ``unequal`` allows a per-task list

.. note::

   The bracketed forms are parsed literally by the platform's compatibility checker
   (``ui/utils/compat.py``). Writing ``'1-K'`` or ``'0-C'`` instead of ``'[1, K]'`` or
   ``'[0, C]'`` makes the check silently skip that field.

**Example: GA Metadata Declaration**:

.. code-block:: python

    class GA:
        algorithm_information = {
            'n_tasks': '[1, K]',        # Supports single and multitask
            'dims': 'unequal',          # Supports unequal dimensions
            'objs': 'equal',            # Requires the same objective count
            'n_objs': '1',              # Single-objective only
            'cons': 'unequal',          # Supports unequal constraints
            'n_cons': '[0, C]',         # Constraints supported
            'expensive': 'False',       # Not expensive (no surrogate)
            'knowledge_transfer': 'False',  # No knowledge transfer
            'n': 'unequal',             # Different population sizes
            'max_nfes': 'unequal'       # Different max evaluations
        }

        @classmethod
        def get_algorithm_information(cls, print_info=True):
            """Get and print algorithm metadata"""
            return get_algorithm_information(cls, print_info)

Viewing Algorithm Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**D²MTOLab** provides the ``get_algorithm_information`` class method for each algorithm to retrieve and display metadata:

.. code-block:: python

    from ddmtolab.Algorithms.STSO.GA import GA

    # Call class method to view GA metadata
    GA.get_algorithm_information()

**Output**:

.. code-block:: none

    🤖️ GA
    Algorithm Information:
      - n_tasks: [1, K]
      - dims: unequal
      - objs: equal
      - n_objs: 1
      - cons: unequal
      - n_cons: [0, C]
      - expensive: False
      - knowledge_transfer: False
      - n: unequal
      - max_nfes: unequal

This method prints the algorithm name and all metadata fields in a structured format, helping users quickly understand the algorithm's scope and characteristic constraints. By viewing the metadata, users can determine whether an algorithm is suitable for their optimization problem.

The method also supports returning metadata as a dictionary for programmatic processing:

.. code-block:: python

    from ddmtolab.Algorithms.STSO.GA import GA
    info = GA.get_algorithm_information(print_info=False)
    print(info)

**Output**:

.. code-block:: python

    {'n_tasks': '[1, K]', 'dims': 'unequal', 'objs': 'equal', 'n_objs': '1',
     'cons': 'unequal', 'n_cons': '[0, C]', 'expensive': 'False',
     'knowledge_transfer': 'False', 'n': 'unequal', 'max_nfes': 'unequal'}

Using Algorithms
----------------

Basic Usage
~~~~~~~~~~~

**Example: Single-Task Optimization**:

.. code-block:: python

    from ddmtolab.Methods.mtop import MTOP
    from ddmtolab.Algorithms.STSO.GA import GA
    import numpy as np

    # Define objective function
    def sphere(x):
        return np.sum(x**2, axis=1)

    # Create problem instance using MTOP
    problem = MTOP()
    problem.add_task(sphere, dim=30)

    # Initialize algorithm
    algorithm = GA(
        problem=problem,
        n=100,             # Population size
        max_nfes=10000,    # Max function evaluations
        muc=2.0,           # SBX crossover distribution index
        mum=5.0            # Polynomial mutation distribution index
    )

    # Run optimization
    results = algorithm.optimize()

    # Access results
    print(f"Best objective: {results.best_objs[0]}")
    print(f"Runtime: {results.runtime:.2f}s")

**Example: Multitask Optimization**:

.. code-block:: python

    from ddmtolab.Methods.mtop import MTOP
    from ddmtolab.Algorithms.MTSO.MFEA import MFEA
    import numpy as np

    # Define objective functions
    def sphere(x):
        return np.sum(x**2, axis=1)

    def rosenbrock(x):
        return np.sum(100*(x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

    def rastrigin(x):
        return 10*x.shape[1] + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)

    # Create multitask problem using MTOP
    problem = MTOP()
    problem.add_task(sphere, dim=30)
    problem.add_task(rosenbrock, dim=30)
    problem.add_task(rastrigin, dim=30)

    # Initialize MFEA
    algorithm = MFEA(
        problem=problem,
        n=100,
        max_nfes=10000,
        rmp=0.3  # Random mating probability
    )

    # Run optimization
    results = algorithm.optimize()

    # Compare task performance
    for i in range(problem.n_tasks):
        print(f"Task {i+1} best: {results.best_objs[i]}")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

**Custom Parameter Settings**:

.. code-block:: python

    # Configure algorithm with custom parameters. Sizing parameters accept either a
    # scalar (shared by every task) or a per-task list when the metadata says 'unequal'.
    algorithm = GA(
        problem=problem,
        n=[200],              # Larger population, per task
        max_nfes=[50000],     # More evaluations, per task
        muc=15.0,             # Custom crossover distribution index
        mum=20.0,             # Custom mutation distribution index
        save_path='./Data',   # Where the .pkl result file is written
        name='GA-large'       # File name stem for the saved results
    )

**Accessing Optimization History**:

.. code-block:: python

    results = algorithm.optimize()

    # Get evolution trajectory for task 0
    obj_history = results.all_objs[0]

    # Plot convergence curve
    import matplotlib.pyplot as plt

    best_per_gen = [min(gen_objs) for gen_objs in obj_history]
    plt.plot(best_per_gen)
    plt.xlabel('Generation')
    plt.ylabel('Best Objective Value')
    plt.title('Convergence Curve')
    plt.show()

Implementing Custom Algorithms
-------------------------------

You can easily implement custom algorithms by following the three construction rules:

**Example: Simple Custom Algorithm**:

.. code-block:: python

    import numpy as np
    import time
    from ddmtolab.Methods.Algo_Methods.algo_utils import (
        Results, get_algorithm_information, par_list,
        initialization, evaluation,
        init_history, append_history, build_save_results
    )

    class MyCustomAlgorithm:
        # Rule 1: Algorithm metadata
        algorithm_information = {
            'n_tasks': '[1, K]',
            'dims': 'unequal',
            'objs': 'equal',
            'n_objs': '1',
            'cons': 'equal',
            'n_cons': '0',
            'expensive': 'False',
            'knowledge_transfer': 'False',
            'n': 'unequal',
            'max_nfes': 'unequal'
        }

        @classmethod
        def get_algorithm_information(cls, print_info=True):
            return get_algorithm_information(cls, print_info)

        # Rule 2: Accept MTOP instance
        def __init__(self, problem, n=None, max_nfes=None,
                     save_data=True, save_path='./Data', name='MyAlgo',
                     disable_tqdm=True):
            self.problem = problem
            self.n = n if n is not None else 100
            self.max_nfes = max_nfes if max_nfes is not None else 10000
            self.save_data = save_data
            self.save_path = save_path
            self.name = name
            self.disable_tqdm = disable_tqdm

        # Rule 3: Return Results object
        def optimize(self):
            start_time = time.time()

            # Initialize population using algo_utils
            decs = initialization(self.problem, self.n)
            objs, cons = evaluation(self.problem, decs)

            # Initialize history tracking
            all_decs, all_objs, all_cons = init_history(decs, objs, cons)

            # Main optimization loop
            nfes = self.n
            while nfes < self.max_nfes:
                # Your optimization logic here
                # Generate new solutions, evaluate, select...

                # Track history. append_history takes (history, new_value) PAIRS,
                # not all histories followed by all values.
                append_history(all_decs, decs, all_objs, objs, all_cons, cons)
                nfes += self.n

            runtime = time.time() - start_time

            # Build and save results using utility function. Note the parameter is
            # `filename`, not `name`: build_save_results accepts **kwargs, so a wrong
            # keyword is stored as extra payload instead of raising, and the file is
            # then silently not written.
            return build_save_results(
                all_decs=all_decs,
                all_objs=all_objs,
                all_cons=all_cons,
                runtime=runtime,
                max_nfes=par_list(self.max_nfes, self.problem.n_tasks),
                bounds=self.problem.bounds,
                save_path=self.save_path,
                filename=self.name,
                save_data=self.save_data
            )

Available Algorithms
--------------------

D²MTOLab provides 110+ optimization algorithms organized into four categories:

STSO (Single-Task Single-Objective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classical evolutionary algorithms and surrogate-assisted methods for single-objective optimization.

**Inexpensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``GA``
     - Genetic Algorithm
   * - ``DE``
     - Differential Evolution
   * - ``PSO``
     - Particle Swarm Optimization
   * - ``SL_PSO``
     - Social Learning PSO
   * - ``KLPSO``
     - Knowledge Learning PSO
   * - ``CSO``
     - Competitive Swarm Optimizer
   * - ``CMA_ES``
     - Covariance Matrix Adaptation Evolution Strategy
   * - ``IPOP_CMA_ES``
     - CMA-ES with Increasing Population
   * - ``sep_CMA_ES``
     - Separable CMA-ES
   * - ``MA_ES``
     - Matrix Adaptation Evolution Strategy
   * - ``xNES``
     - Exponential Natural Evolution Strategy
   * - ``OpenAI_ES``
     - OpenAI Evolution Strategy
   * - ``AO``
     - Aquila Optimizer
   * - ``GWO``
     - Grey Wolf Optimizer
   * - ``EO``
     - Equilibrium Optimizer

**Expensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``BO``
     - Bayesian Optimization
   * - ``EEI_BO``
     - Evolutionary Expected Improvement based Bayesian Optimization
   * - ``ESAO``
     - Exploration-exploitation Switching Assisted Optimization
   * - ``SHPSO``
     - Surrogate-assisted Hierarchical Particle Swarm Optimization
   * - ``SA_COSO``
     - Surrogate-Assisted Cooperative Swarm Optimization
   * - ``TLRBF``
     - Three-Level Radial Basis Function Method
   * - ``GL_SADE``
     - Global-Local Surrogate-Assisted Differential Evolution
   * - ``AutoSAEA``
     - Surrogate-Assisted EA with Auto-Configuration
   * - ``DDEA_MESS``
     - Data-Driven EA with Multi-Evolutionary Sampling Strategy
   * - ``LSADE``
     - Lipschitz Surrogate-Assisted Differential Evolution
   * - ``LAEA``
     - LLM-Assisted Evolutionary Algorithm

.. note::

   ``LAEA`` uses a large language model as its surrogate rather than a fitted
   regression model, so it needs the ``llm`` extra
   (``pip install ddmtolab[llm]``) and an API key in the environment variable
   named by ``llm_api_key_env``.

   It spends a second budget alongside ``max_nfes``: every generation issues
   ``2 * n_initial`` inference calls and consumes one real evaluation, so the
   default setting reaches ``max_nfes=300`` only after roughly 25,000 calls.
   Cap this with ``max_llm_calls``, which stops the run and returns partial
   results once it is hit.

   Every response is written to a JSONL cache under
   ``<save_path>/llm_cache/``. Because a hosted model is not bit-reproducible
   even at ``temperature=0``, that cache -- not the seed -- is what makes a run
   repeatable: re-running with ``llm_backend='replay'`` replays it offline and
   at no cost, and raises on any prompt it has not seen. Archive the cache file
   together with the ``.pkl`` when reporting results.

STMO (Single-Task Multiobjective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiobjective evolutionary algorithms and surrogate-assisted methods.

**Inexpensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``NSGA_II``
     - Non-dominated Sorting Genetic Algorithm II
   * - ``NSGA_III``
     - Non-dominated Sorting Genetic Algorithm III
   * - ``NSGA_II_SDR``
     - NSGA-II with Strengthened Dominance Relation
   * - ``SPEA2``
     - Strength Pareto Evolutionary Algorithm 2
   * - ``MOEA_D``
     - Multiobjective Evolutionary Algorithm based on Decomposition
   * - ``MOEA_DD``
     - MOEA Based on Decomposition and Dominance
   * - ``MOEA_D_FRRMAB``
     - MOEA/D with Fitness-Rate-Rank Multi-Armed Bandit
   * - ``MOEA_D_STM``
     - MOEA/D with Stable Matching
   * - ``RVEA``
     - Reference Vector Guided Evolutionary Algorithm
   * - ``IBEA``
     - Indicator-Based Evolutionary Algorithm
   * - ``TwoArch2``
     - Two-Archive Algorithm 2
   * - ``MSEA``
     - Multistage Evolutionary Algorithm
   * - ``C_TAEA``
     - Constrained Two-Archive Evolutionary Algorithm
   * - ``CCMO``
     - Coevolutionary Constrained Multiobjective Optimization

**Expensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``ParEGO``
     - Pareto Efficient Global Optimization
   * - ``K_RVEA``
     - Kriging-assisted Reference Vector Guided Evolutionary Algorithm
   * - ``DSAEA_PS``
     - Dual-Surrogate Assisted EA with Portfolio Strategy
   * - ``KTA2``
     - Kriging-Assisted Two-Archive Evolutionary Algorithm 2
   * - ``REMO``
     - Expensive Multiobjective Optimization by Relation Learning and Prediction
   * - ``ADSAPSO``
     - Adaptive Dropout Surrogate-Assisted PSO
   * - ``CSEA``
     - Classification-based Surrogate-assisted EA
   * - ``DISK``
     - Distribution-based Kriging-assisted EA
   * - ``DRLSAEA``
     - Deep Reinforcement Learning Surrogate-Assisted EA
   * - ``DirHV_EI``
     - Expected Direction-based Hypervolume Improvement
   * - ``EDN_ARMOEA``
     - Efficient Dropout Neural Network based AR-MOEA
   * - ``EIM_EGO``
     - Expected Improvement Matrix based EGO
   * - ``EM_SAEA``
     - Ensemble Model Surrogate-Assisted EA
   * - ``KTS``
     - Kriging-Assisted Two-Archive Search
   * - ``MGSAEA``
     - Multigranularity Surrogate-Assisted EA
   * - ``MMRAEA``
     - Multi-Model Ranking Aggregation EA
   * - ``MOEA_D_EGO``
     - MOEA/D with Efficient Global Optimization
   * - ``MultiObjectiveEGO``
     - Multiobjective Efficient Global Optimization
   * - ``PCSAEA``
     - Pairwise Comparison Surrogate-Assisted EA
   * - ``PEA``
     - Pareto-based Kriging-assisted Constrained MOEA
   * - ``PIEA``
     - Performance Indicator-based EA
   * - ``SAEA_DBLL``
     - Surrogate-Assisted EA with Decomposition-Based Local Learning
   * - ``SSDE``
     - Self-Organizing Surrogate-Assisted Non-Dominated Sorting DE
   * - ``TEA``
     - Two-phase Evolutionary Algorithm
   * - ``CPS_MOEA``
     - Classification and Pareto Domination Based MOEA
   * - ``MCEA_D``
     - Multiple Classifiers-assisted EA based on Decomposition

MTSO (Multitask Single-Objective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multitask evolutionary algorithms with knowledge transfer for single-objective optimization.

**Inexpensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``MFEA``
     - Multi-Factorial Evolutionary Algorithm
   * - ``MFEA_II``
     - Multi-Factorial Evolutionary Algorithm II
   * - ``EMEA``
     - Evolutionary Multitasking via Explicit Autoencoding
   * - ``EBS``
     - Evolutionary Biocoenosis-based Symbiosis
   * - ``G_MFEA``
     - Generalized MFEA
   * - ``MTEA_AD``
     - Multitask EA with Adaptive Knowledge Transfer via Anomaly Detection
   * - ``MKTDE``
     - Meta-Knowledge Transfer-based Differential Evolution
   * - ``MTEA_SaO``
     - Multitask EA with Self-adaptive Solvers
   * - ``SREMTO``
     - Self-Regulated Evolutionary Multitask Optimization
   * - ``LCB_EMT``
     - Lower Confidence Bound Evolutionary Multitasking
   * - ``BLKT_DE``
     - Block-Level Knowledge Transfer DE
   * - ``DTSKT``
     - Distribution Direction-Assisted Two-Stage Knowledge Transfer
   * - ``EMTO_AI``
     - Evolutionary Multitask Optimization with Adaptive Intensity
   * - ``MFEA_AKT``
     - Multifactorial EA with Adaptive Knowledge Transfer
   * - ``MFEA_DGD``
     - MFEA Based on Diffusion Gradient Descent
   * - ``MFEA_VC``
     - MFEA with Variational Crossover
   * - ``MTDE_ADKT``
     - Multitask DE with Adaptive Dual Knowledge Transfer
   * - ``MTEA_HKTS``
     - Multitask EA with Hierarchical Knowledge Transfer Strategy
   * - ``MTEA_PAE``
     - Multitask EA with Progressive Auto-Encoding
   * - ``MTES_KG``
     - Multitask Evolution Strategy with Knowledge-Guided External Sampling
   * - ``SSLT_DE``
     - Scenario-based Self-Learning Transfer DE
   * - ``TNG_SNES``
     - Transfer Task-averaged Natural Gradient Separable NES

**Expensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``MTBO``
     - Multitask Bayesian Optimization
   * - ``RAMTEA``
     - Radial Basis Function-Assisted Multitask Evolutionary Algorithm
   * - ``SELF``
     - Surrogate-Assisted Evolutionary Framework for Expensive Multitask Optimization
   * - ``EEI_BO_plus``
     - Evolutionary Expected Improvement based BO for MTOPs
   * - ``MUMBO``
     - Multitask Max-value Bayesian Optimization
   * - ``BO_LCB_CKT``
     - BO with LCB and Competitive Knowledge Transfer
   * - ``BO_LCB_BCKT``
     - BO with LCB and Bayesian Competitive Knowledge Transfer
   * - ``MFEA_SSG``
     - MFEA with Single-Step Generative Model
   * - ``SaEF_AKT``
     - Surrogate-Assisted Evolutionary Framework with Adaptive Knowledge Transfer

MTMO (Multitask Multiobjective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multitask multiobjective evolutionary algorithms with knowledge transfer.

**Inexpensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``MO_MFEA``
     - Multiobjective Multi-Factorial Evolutionary Algorithm
   * - ``MO_MFEA_II``
     - Multiobjective MFEA II
   * - ``MO_EMEA``
     - Multiobjective Evolutionary Multitasking via Explicit Autoencoding
   * - ``MO_MTEA_SaO``
     - Multiobjective Multitask EA with Self-adaptive Solvers
   * - ``MTDE_MKTA``
     - Multiobjective Multitask DE with Multiple Knowledge Types and Transfer Adaptation
   * - ``MTEA_D_DN``
     - Multitask EA/D with Dynamic Neighborhood
   * - ``EMT_ET``
     - Evolutionary Multitasking with Effective Transfer
   * - ``EMT_PD``
     - Evolutionary Multitasking with Population Distribution-based Transfer
   * - ``EMT_GS``
     - Evolutionary Multitasking with Generative Strategies
   * - ``MO_MTEA_PAE``
     - Multiobjective MTEA with Progressive Auto-Encoding
   * - ``MO_SBO``
     - Multiobjective Symbiosis-Based Optimization
   * - ``MTEA_D_TSD``
     - Multitask EA/D with Transfer of Search Directions
   * - ``MTEA_DCK``
     - Multitask EA via Diversity- and Convergence-Oriented Knowledge Transfer

**Expensive**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``ParEGO_KT``
     - ParEGO with Knowledge Transfer
   * - ``SAEA_AKT``
     - Surrogate-Assisted EA with Adaptive Knowledge Transfer

See Also
--------

* :ref:`api` - Complete API documentation
* :ref:`demos` - Diverse demonstrations
.. _algorithms:

Algorithms
==========

This chapter introduces the algorithm design philosophy and construction rules in **DDMTOLab**, providing comprehensive guidance for implementing custom optimization algorithms.

Algorithm Construction
----------------------

Considering the complexity and diversity of data-driven multitask optimization, **DDMTOLab** adopts a **loosely-coupled algorithm design philosophy**. The platform does not mandate algorithms to inherit specific base classes or implement fixed interface methods, thereby avoiding restrictions on algorithm flexibility. This design approach offers the following advantages:

1. **Enhanced Platform Compatibility**: Traditional gradient-based methods, evolutionary algorithms, advanced data-driven multitask optimization algorithms, and hybrid innovative architectures can all be seamlessly integrated into the platform.

2. **Improved Development Convenience**: Users can quickly implement algorithms across the full spectrum—from inexpensive single-task single-objective unconstrained optimization to expensive multi-task multi-objective constrained optimization—without understanding complex class inheritance hierarchies.

3. **Guaranteed Algorithm Freedom**: Users are free to design data structures, optimization workflows, and knowledge transfer strategies according to specific problem characteristics and algorithm mechanisms, without framework constraints.

To facilitate subsequent data processing and efficient coordination with the platform's experiment modules and data analysis modules, **DDMTOLab** imposes only **3 basic rules** on algorithm construction, ensuring normal platform functionality while maximizing algorithm development flexibility.

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
        # Component 1: Algorithm metadata (required)
        algorithm_information = {
            'n_tasks': '1-K',               # Supported task number types
            'dims': 'unequal',              # Decision variable dimension constraint
            'objs': 'unequal',              # Objective number constraint
            'n_objs': '1-M',                # Objective quantity type
            'cons': 'unequal',              # Constraint number constraint
            'n_cons': '0-C',                # Constraint quantity type
            'expensive': 'False',           # Whether expensive optimization
            'knowledge_transfer': 'False',  # Whether knowledge transfer involved
            'param': 'unequal'              # Algorithm parameter constraint
        }

        # Component 2: Metadata access method (required)
        @classmethod
        def get_algorithm_information(cls, print_info=True):
            return get_algorithm_information(cls, print_info)

        # Component 3: Initialization method (required)
        def __init__(self, problem, n=None, max_nfes=None, ...):
            self.problem = problem
            # Other parameter initialization

        # Component 4: Optimization method (required)
        def optimize(self):
            # Algorithm implementation
            return results

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
     - **Best decision variables**. List length is the number of tasks K. ``best_decs[i]`` is the best decision variable for task i. Shape is :math:`(n, D^i)`, where n is the number of optimal solutions (n=1 for single-objective; n≥2 for multi-objective)
   * - ``best_objs``
     - ``List[np.ndarray]``
     - **Best objective values**. List length is K. ``best_objs[i]`` is the best objective value for task i. Shape is :math:`(n, M^i)`
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
     - **Best constraint values** (optional). Used only in constrained optimization. ``best_cons[i]`` is the constraint value corresponding to the best solution of task i. Shape is :math:`(n, C^i)`. None for unconstrained problems
   * - ``all_cons``
     - ``Optional[List[List[np.ndarray]]]``
     - **Constraint evolution history** (optional). ``all_cons[i][g]`` represents all constraint values of task i at generation g. Shape is :math:`(n, C^i)`. None for unconstrained problems

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
     - Supported task numbers. ``1`` means single-task only, ``K`` means multi-task only (K≥2), ``1-K`` means both single and multi-task supported
   * - ``dims``
     - Decision variable dimension constraint. ``equal`` requires same dimensions across tasks, ``unequal`` supports unequal-dimension tasks
   * - ``objs``
     - Objective number constraint. ``equal`` requires same number of objectives across tasks, ``unequal`` supports unequal objective numbers
   * - ``n_objs``
     - Objective quantity type. ``1`` means single-objective only, ``M`` means multi-objective only (M≥2), ``1-M`` means both supported
   * - ``cons``
     - Constraint number constraint. ``equal`` requires same number of constraints across tasks, ``unequal`` supports unequal constraint numbers
   * - ``n_cons``
     - Constraint quantity type. ``0`` means unconstrained, ``C`` means constrained only (C≥1), ``0-C`` means both supported
   * - ``expensive``
     - Whether expensive optimization (involving surrogate models). ``True`` uses surrogate models, ``False`` does not
   * - ``knowledge_transfer``
     - Whether inter-task knowledge transfer involved. ``True`` means the algorithm includes knowledge transfer mechanisms, ``False`` means tasks are optimized independently
   * - ``param``
     - Algorithm parameter constraint. ``equal`` requires same parameters (e.g., population size, evaluation count) across tasks, ``unequal`` allows different parameters per task

**Example: GA Metadata Declaration**:

.. code-block:: python

    class GA:
        algorithm_information = {
            'n_tasks': '1-K',           # Supports single and multi-task
            'dims': 'unequal',          # Supports unequal dimensions
            'objs': 'unequal',          # Supports unequal objectives
            'n_objs': '1',              # Single-objective only
            'cons': 'unequal',          # Supports unequal constraints
            'n_cons': '0',              # Unconstrained only
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

**DDMTOLab** provides the ``get_algorithm_information`` class method for each algorithm to retrieve and display metadata:

.. code-block:: python

    from Algorithms.STSO.GA import GA

    # Call class method to view GA metadata
    GA.get_algorithm_information()

**Output**:

.. code-block:: none

    🤖️ GA
    Algorithm Information:
      - n_tasks: 1-K
      - dims: unequal
      - objs: unequal
      - n_objs: 1
      - cons: unequal
      - n_cons: 0
      - expensive: False
      - knowledge_transfer: False
      - param: unequal

This method prints the algorithm name and all metadata fields in a structured format, helping users quickly understand the algorithm's scope and characteristic constraints. By viewing the metadata, users can determine whether an algorithm is suitable for their optimization problem.

The method also supports returning metadata as a dictionary for programmatic processing:

.. code-block:: python

    from Algorithms.STSO.GA import GA
    info = GA.get_algorithm_information(print_info=False)
    print(info)

**Output**:

.. code-block:: python

    {'n_tasks': '1-K', 'dims': 'unequal', 'objs': 'unequal', 'n_objs': '1',
     'cons': 'unequal', 'n_cons': '0', 'expensive': 'False',
     'knowledge_transfer': 'False', 'n': 'unequal', 'max_nfes': 'unequal'}

Using Algorithms
----------------

Basic Usage
~~~~~~~~~~~

**Example: Single-Task Optimization**:

.. code-block:: python

    from DDMTOLab.problems import Sphere
    from DDMTOLab.algorithms.STSO.GA import GA

    # Create problem instance
    problem = Sphere(n_tasks=1, dims=[30])

    # Initialize algorithm
    algorithm = GA(
        problem=problem,
        n=[100],           # Population size
        max_nfes=[10000],  # Max function evaluations
        pc=0.9,            # Crossover probability
        pm=0.1             # Mutation probability
    )

    # Run optimization
    results = algorithm.optimize()

    # Access results
    print(f"Best objective: {results.best_objs[0]}")
    print(f"Runtime: {results.runtime:.2f}s")

**Example: Multi-Task Optimization**:

.. code-block:: python

    from DDMTOLab.problems import MTOP
    from DDMTOLab.algorithms.MTSO.MFEA import MFEA

    # Create multi-task problem
    problem = MTOP(
        problems=['Sphere', 'Rosenbrock', 'Rastrigin'],
        n_tasks=3,
        dims=[30, 30, 30]
    )

    # Initialize MFEA
    algorithm = MFEA(
        problem=problem,
        n=[100, 100, 100],
        max_nfes=[10000, 10000, 10000],
        rmp=0.3  # Random mating probability
    )

    # Run optimization
    results = algorithm.optimize()

    # Compare task performance
    for i in range(3):
        print(f"Task {i+1} best: {results.best_objs[i]}")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

**Custom Parameter Settings**:

.. code-block:: python

    # Configure algorithm with custom parameters
    algorithm = GA(
        problem=problem,
        n=[200],              # Larger population
        max_nfes=[50000],     # More evaluations
        pc=0.85,              # Custom crossover rate
        pm=0.15,              # Custom mutation rate
        selection='tournament',  # Selection method
        tournament_size=3     # Tournament size
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
    from DDMTOLab.utils import Results
    import time

    class MyCustomAlgorithm:
        # Rule 1: Algorithm metadata
        algorithm_information = {
            'n_tasks': '1',
            'dims': 'unequal',
            'objs': 'unequal',
            'n_objs': '1',
            'cons': 'unequal',
            'n_cons': '0',
            'expensive': 'False',
            'knowledge_transfer': 'False',
            'param': 'unequal'
        }

        @classmethod
        def get_algorithm_information(cls, print_info=True):
            return get_algorithm_information(cls, print_info)

        # Rule 2: Accept MTOP instance
        def __init__(self, problem, n=None, max_nfes=None):
            self.problem = problem
            self.n = n if n else [100]
            self.max_nfes = max_nfes if max_nfes else [10000]

        # Rule 3: Return Results object
        def optimize(self):
            start_time = time.time()

            # Initialize tracking
            all_decs = [[] for _ in range(self.problem.n_tasks)]
            all_objs = [[] for _ in range(self.problem.n_tasks)]

            # Optimization loop for each task
            for task_id in range(self.problem.n_tasks):
                # Your optimization logic here
                # ...

                # Store best solution
                best_dec = np.array([...])
                best_obj = np.array([...])

            runtime = time.time() - start_time

            # Return Results object
            return Results(
                best_decs=[best_dec],
                best_objs=[best_obj],
                all_decs=all_decs,
                all_objs=all_objs,
                runtime=runtime,
                max_nfes=self.max_nfes
            )

See Also
--------

* :ref:`api` - Complete API documentation
* :ref:`quickstart` - Getting started guide
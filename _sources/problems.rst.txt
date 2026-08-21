.. _problems:

Problems
========

This chapter introduces the **MTOP (Multitask Optimization Problem)** class, the core component of D²MTOLab for defining and managing multitask optimization problems.

MTOP Class Overview
-------------------

The MTOP class provides the following key features:

1. **Flexible Task Definition**: Supports single/multiobjective, constrained/unconstrained tasks with varying dimensions
2. **Automatic Vectorization**: Wraps non-vectorized functions for batch evaluation
3. **Unified Evaluation Mode**: Pads evaluation results to uniform dimensions for algorithm processing
4. **Selective Evaluation**: Evaluates only specified objectives or constraints for complex scenarios
5. **Default Boundaries**: Uses [0, 1] as default search range when bounds are not specified

Initialization
--------------

Create an MTOP instance with optional unified evaluation mode:

.. code-block:: python

    from ddmtolab.Methods.mtop import MTOP

    # Basic initialization
    problem = MTOP()

    # Enable unified evaluation mode
    problem = MTOP(unified_eval_mode=True, fill_value=0.0)

**Parameters:**

- ``unified_eval_mode`` (bool): Whether to pad all task results to maximum dimensions. Default: False
- ``fill_value`` (float): Value used for padding in unified mode. Default: 0.0

Adding Tasks
------------

Using add_task Method
~~~~~~~~~~~~~~~~~~~~~

The ``add_task`` method is the primary way to add optimization tasks:

.. code-block:: python

    def add_task(
        objective_func,      # Objective function (required)
        dim,                 # Decision variable dimension (required)
        constraint_func=None,  # Constraint function (optional)
        lower_bound=None,    # Lower bounds (optional, default 0)
        upper_bound=None,    # Upper bounds (optional, default 1)
        budget=None,         # Evaluation budget of the task (optional)
        metadata=None        # Free-form descriptor dict (optional)
    ) -> int:               # Returns task index

**Example: 2 Tasks with [1,3] Objectives and [2,0] Constraints**:

.. code-block:: python

    import numpy as np
    from ddmtolab.Methods.mtop import MTOP

    # Task 1: Single-objective with 2 constraints
    def sphere(x):
        x = np.atleast_2d(x)
        return np.sum(x**2, axis=1, keepdims=True)

    def constraints_t1(x):
        x = np.atleast_2d(x)
        g1 = np.sum(x, axis=1) - 1
        g2 = x[:, 0]**2 + x[:, 1]**2 - 0.5
        return np.column_stack([g1, g2])

    # Task 2: 3-objective without constraints
    def multi_obj(x):
        x = np.atleast_2d(x)
        f1 = np.sum(x**2, axis=1)
        f2 = np.sum((x - 1)**2, axis=1)
        f3 = np.sum(np.abs(x), axis=1)
        return np.column_stack([f1, f2, f3])

    problem = MTOP()
    problem.add_task(sphere, dim=3, constraint_func=constraints_t1,
                     lower_bound=[-5]*3, upper_bound=[5]*3)
    problem.add_task(multi_obj, dim=2, lower_bound=[-1]*2, upper_bound=[1]*2)

    print(problem)

**Output**:

.. code-block:: python

    MTOP with 2 tasks:
      Unified eval mode: False (fill_value=0.0)
      Max number of objectives (n_objs_max): 3
      Max number of constraints (n_cons_max): 2
      Task 0: dim=3, n_objs=1, n_cons=2, bounds=[-5.0..5.0]
      Task 1: dim=2, n_objs=3, n_cons=0, bounds=[-1.0..1.0]

Batch Adding with Tuples
~~~~~~~~~~~~~~~~~~~~~~~~~

Add multiple tasks at once using tuple arguments:

.. code-block:: python

    indices = problem.add_task(
        objective_func=(obj1, obj2, obj3),
        dim=(3, 4, 5),
        constraint_func=(con1, con2, con3),
        lower_bound=([-1]*3, [-2]*4, [-3]*5),
        upper_bound=([1]*3, [2]*4, [3]*5)
    )

Using add_tasks Method
~~~~~~~~~~~~~~~~~~~~~~

Add tasks from configuration dictionaries:

.. code-block:: python

    tasks_config = [
        {
            'objective_func': obj1,
            'dim': 3,
            'constraint_func': con1,
            'lower_bound': [-1]*3,
            'upper_bound': [1]*3
        },
        {
            'objective_func': obj2,
            'dim': 4,
            'constraint_func': con2,
            'lower_bound': [-2]*4,
            'upper_bound': [2]*4
        }
    ]

    problem.add_tasks(tasks_config)

Task Evaluation
---------------

Single Task Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Evaluate individual tasks with ``evaluate_task``:

.. code-block:: python

    # Evaluate task 0
    X0 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])  # 2 solutions
    objs, cons = problem.evaluate_task(0, X0)
    print(f"Objectives shape: {objs.shape}")  # (2, n_objs)
    print(f"Constraints shape: {cons.shape}")  # (2, n_cons)

Selective Evaluation
~~~~~~~~~~~~~~~~~~~~

Evaluate only specific objectives or constraints. The ``eval_objectives`` and ``eval_constraints`` parameters support:

1. ``True``: Evaluate all (default)
2. ``False``: Skip evaluation, return empty array
3. ``int``: Evaluate single objective/constraint at specified index
4. ``List[int]``: Evaluate multiple objectives/constraints at specified indices

**Example**:

.. code-block:: python

    # Evaluate only objectives
    objs, cons = problem.evaluate_task(0, X, eval_objectives=True,
                                      eval_constraints=False)

    # Evaluate specific objective indices
    objs, cons = problem.evaluate_task(0, X, eval_objectives=[0, 2],
                                      eval_constraints=False)

    # Evaluate single constraint
    objs, cons = problem.evaluate_task(0, X, eval_objectives=False,
                                      eval_constraints=0)

Multitask Evaluation
~~~~~~~~~~~~~~~~~~~~~

Evaluate multiple tasks simultaneously with ``evaluate_tasks``:

.. code-block:: python

    # Prepare input data for each task
    X0 = np.random.uniform(-1, 1, (4, 3))   # Task 0: 4 solutions
    X1 = np.random.uniform(-2, 2, (3, 4))   # Task 1: 3 solutions
    X2 = np.random.uniform(-3, 3, (5, 5))   # Task 2: 5 solutions

    # Evaluate all tasks
    objs_list, cons_list = problem.evaluate_tasks(
        task_indices=[0, 1, 2],
        X_list=[X0, X1, X2]
    )

    for i, (objs, cons) in enumerate(zip(objs_list, cons_list)):
        print(f"Task {i}: objs shape={objs.shape}, cons shape={cons.shape}")

Multitask Selective Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply different evaluation settings to each task:

.. code-block:: python

    # Unified mode: same settings for all tasks
    objs_list, cons_list = problem.evaluate_tasks(
        task_indices=[0, 1, 2],
        X_list=[X0, X1, X2],
        eval_objectives=True,    # All tasks evaluate objectives
        eval_constraints=False   # All tasks skip constraints
    )

    # Per-task mode: different settings per task
    objs_list, cons_list = problem.evaluate_tasks(
        task_indices=[0, 1, 2],
        X_list=[X0, X1, X2],
        eval_objectives=[True, [0], [0, 2]],      # Task 0: all, Task 1: first, Task 2: first and third
        eval_constraints=[False, True, 0]          # Task 0: skip, Task 1: all, Task 2: first
    )

Unified Evaluation Mode
~~~~~~~~~~~~~~~~~~~~~~~

When tasks have different numbers of objectives or constraints, unified evaluation mode pads all results to maximum dimensions:

.. code-block:: python

    # Check current dimensions
    print(f"Max objectives: {problem.n_objs}")
    print(f"Max constraints: {problem.n_cons}")

    # Enable unified evaluation mode
    problem.set_unified_eval_mode(enabled=True, fill_value=0.0)

    X0 = np.array([[0.1, 0.2, 0.3]])
    X1 = np.array([[0.1, 0.2, 0.3, 0.4]])
    X2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

    print("\nWith unified evaluation mode:")
    for i, X in enumerate([X0, X1, X2]):
        objs, cons = problem.evaluate_task(i, X)
        print(f"  Task {i}: objs shape={objs.shape}, cons shape={cons.shape}")

    # Disable unified evaluation mode
    problem.set_unified_eval_mode(enabled=False)

Automatic Vectorization
~~~~~~~~~~~~~~~~~~~~~~~

MTOP automatically handles batch evaluation of functions. If a function raises an exception with batch input, it switches to row-by-row evaluation:

.. code-block:: python

    # Non-vectorized function: only handles single row
    def sphere_single_row(x):
        if x.ndim > 1:
            raise ValueError("Only handles 1D input")
        return np.sum(x**2)

    problem = MTOP()
    idx = problem.add_task(sphere_single_row, dim=3)

    X_batch = np.array([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9]])

    obj, _ = problem.evaluate_task(idx, X_batch)
    print(f"Batch evaluation result:\n{obj}")

**Important**: Automatic vectorization relies on functions raising exceptions for incompatible inputs. **It is strongly recommended to write vectorized functions directly**:

.. code-block:: python

    # Correct vectorized function
    def sphere_vectorized(x):
        x = np.atleast_2d(x)
        return np.sum(x**2, axis=1, keepdims=True)

.. _declaring-budget-metadata:

Declaring a Budget and Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A task can carry its own evaluation budget :math:`N^k` and a free-form
descriptor, so the problem definition states how expensive it is instead of
every call site repeating it:

.. code-block:: python

    problem = MTOP()
    problem.add_task(
        expensive_simulation, dim=10,
        budget=200,                                  # this task affords 200 evaluations
        metadata={'source': 'CFD', 'fidelity': 'high'}
    )

Both broadcast across tasks exactly the way ``dim`` does -- a single value
applies to every task, a tuple gives one value per task:

.. code-block:: python

    problem.add_task((cheap_model, expensive_model), dim=(10, 10),
                     budget=(1000, 100))            # per-task budgets

An algorithm reads the declared budget through ``resolve_budget``, which is
already wired into the algorithms that support it. An explicitly passed
``max_nfes`` always wins, so existing calls behave exactly as before; the
declared budget is consulted only when the caller leaves ``max_nfes`` unset:

.. code-block:: python

    from ddmtolab.Algorithms.MTMO.ParEGO_KT import ParEGO_KT

    ParEGO_KT(problem).optimize()                    # uses the declared budget
    ParEGO_KT(problem, max_nfes=150).optimize()      # explicit argument wins

``metadata`` is never populated or interpreted by the platform; it is stored
per task and returned by ``get_task_info`` for the caller's own bookkeeping.
Budgets must be positive integers and metadata must be a dict, both validated
when the task is added.

Querying Task Information
--------------------------

Basic Properties
~~~~~~~~~~~~~~~~

Query basic problem properties:

.. code-block:: python

    # Number of tasks
    print(f"Number of tasks: {problem.n_tasks}")

    # Dimensions per task
    print(f"Task dimensions: {problem.dims}")

    # Objectives per task
    print(f"Objectives per task: {problem.n_objs}")

    # Constraints per task
    print(f"Constraints per task: {problem.n_cons}")

    # Unified evaluation mode status
    print(f"Unified eval mode: {problem.unified_eval_mode}")
    print(f"Fill value: {problem.fill_value}")

    # Declared evaluation budgets and descriptors (None / {} when undeclared)
    print(f"Task budgets: {problem.budgets}")
    print(f"Task metadata: {problem.metadata}")

Detailed Task Information
~~~~~~~~~~~~~~~~~~~~~~~~~~

Get complete information for a specific task:

.. code-block:: python

    info = problem.get_task_info(1)
    print(info)

**Output**:

.. code-block:: python

    {'dimension': 4, 'n_objectives': 2, 'n_constraints': 2,
     'lower_bounds': array([[-2., -2., -2., -2.]]),
     'upper_bounds': array([[2., 2., 2., 2.]]),
     'objective_func': <function obj2>,
     'constraint_funcs': [<function wrapper>],
     'budget': None,
     'metadata': {}}

``budget`` is the evaluation budget the task declared, or ``None`` when it
declared none, and ``metadata`` is the descriptor supplied when the task was
added, or ``{}``. See :ref:`declaring-budget-metadata`.

Problem Summary
~~~~~~~~~~~~~~~

Print a summary of the entire problem:

.. code-block:: python

    print(problem)

**Output**:

.. code-block:: python

    MTOP with 3 tasks:
      Unified eval mode: False (fill_value=0.0)
      Max number of objectives (n_objs_max): 3
      Max number of constraints (n_cons_max): 3
      Task 0: dim=3, n_objs=1, n_cons=3, bounds=[-1.0..1.0]
      Task 1: dim=4, n_objs=2, n_cons=2, bounds=[-2.0..2.0]
      Task 2: dim=5, n_objs=3, n_cons=1, bounds=[-3.0..3.0]

MTOP Method Summary
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Category
     - Method
     - Description
   * - Initialization
     - ``__init__()``
     - Initialize MTOP instance with optional unified evaluation mode
   * -
     - ``set_unified_eval_mode()``
     - Dynamically set unified evaluation mode
   * - Task Addition
     - ``add_task()``
     - Add single or multiple tasks (supports tuple form)
   * -
     - ``add_tasks()``
     - Add tasks from configuration dictionary list
   * - Task Evaluation
     - ``evaluate_task()``
     - Evaluate single task with selective evaluation support
   * -
     - ``evaluate_tasks()``
     - Batch evaluate multiple tasks with selective evaluation
   * - Information Query
     - ``get_task_info()``
     - Get detailed information dictionary for specified task
   * - Properties
     - ``n_tasks``
     - Total number of tasks
   * -
     - ``dims``
     - List of dimensions per task
   * -
     - ``bounds``
     - List of bounds per task
   * -
     - ``n_objs``
     - List of objective counts per task
   * -
     - ``n_cons``
     - List of constraint counts per task
   * -
     - ``unified_eval_mode``
     - Unified evaluation mode status
   * -
     - ``fill_value``
     - Fill value setting

Important Notes
---------------

When using the MTOP class, note the following:

1. **Function Design**: Both objective and constraint functions should:

   - Handle 2D array input with shape (n, D)
   - Return 2D arrays with shape (n, M) for objectives or (n, C) for constraints
   - Use ``np.atleast_2d(x)`` to ensure 2D input
   - Compute along ``axis=1`` to keep rows independent
   - Constraints follow the form g(x) ≤ 0 (feasible when ≤ 0)

2. **Boundary Settings**: If ``lower_bound`` or ``upper_bound`` are not specified, default values of 0 and 1 are used. Explicitly specify search boundaries based on your problem.

3. **Unified Evaluation Mode**: Enabling this mode pads all task results to the same dimensions, useful for multitask algorithms but increases memory overhead.

4. **Selective Evaluation**: In complex data-driven optimization scenarios where evaluation costs vary across tasks/objectives/constraints, use selective evaluation based on actual data availability.

Complete Problem Definition Example
------------------------------------

For maintainability and platform compatibility, optimization problems should be encapsulated as Python classes. Problem files should be stored in ``./Problems/`` subdirectories by type:

- ``STSO``: Single-task single-objective
- ``STMO``: Single-task multiobjective
- ``MTSO``: Multitask single-objective
- ``MTMO``: Multitask multiobjective
- ``RWO``: Real-world optimization problems

Problem Class Definition
~~~~~~~~~~~~~~~~~~~~~~~~~

Example: CEC17MTMO Problem 1:

.. code-block:: python

    import numpy as np
    from ddmtolab.Methods.mtop import MTOP

    class CEC17MTMO:
        """CEC2017 Multitask Multiobjective Test Suite"""

        def __init__(self, mat_dir='../Problems/MTMO/data_cec17mtmo'):
            """Initialize problem class with data file path"""
            self.mat_dir = mat_dir

        def P1(self):
            """Problem 1: 2 tasks with different objectives"""
            dim = 50

            def T1(x):
                # Task 1: 2 objectives
                x = np.atleast_2d(x)
                q = 1.0 + np.sum(x[:, 1:] ** 2, axis=1)
                x1 = x[:, 0]
                f1 = q * np.cos(np.pi * x1 / 2)
                f2 = q * np.sin(np.pi * x1 / 2)
                return np.vstack([f1, f2]).T

            def T2(x):
                # Task 2: 2 objectives
                x = np.atleast_2d(x)
                q = 1.0 + 9.0 / (dim - 1) * np.sum(np.abs(x[:, 1:]), axis=1)
                x1 = x[:, 0]
                f1 = x1
                f2 = q * (1.0 - (x1 / q) ** 2)
                return np.vstack([f1, f2]).T

            # Define search space: x1∈[0,1], other variables∈[-100,100]
            lb = np.array([0.0] + [-100.0] * (dim - 1))
            ub = np.array([1.0] + [100.0] * (dim - 1))

            # Create and configure multitask optimization problem
            problem = MTOP()
            problem.add_task(T1, dim=dim, lower_bound=lb, upper_bound=ub)
            problem.add_task(T2, dim=dim, lower_bound=lb, upper_bound=ub)
            return problem

Metric Configuration
~~~~~~~~~~~~~~~~~~~~

For problems requiring complex metrics (e.g., multiobjective optimization), provide configuration for metric calculation:

.. code-block:: python

    def P1_T1_PF(N, M):
        """Generate true Pareto front for Task 1"""
        theta = np.linspace(0, np.pi / 2, N)
        f1 = np.cos(theta)
        f2 = np.sin(theta)
        return np.column_stack([f1, f2])

    def P1_T2_PF(N, M):
        """Generate true Pareto front for Task 2"""
        f1 = np.linspace(0, 1, N)
        f2 = 1 - f1 ** 2
        return np.column_stack([f1, f2])

    # Metric configuration dictionary
    SETTINGS = {
        'metric': 'IGD',            # Performance metric type
        'n_ref': 10000,             # Number of reference points to sample
        'ref_path': './MOReference',  # Directory of pre-stored reference files (optional)
        'P1': {'T1': P1_T1_PF, 'T2': P1_T2_PF}  # PF generators per task
    }

Implementation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem Class Structure**

1. Class name should clearly reflect problem source (e.g., ``CEC17MTMO``)
2. Each problem instance method (e.g., ``P1``, ``P2``) is annotated ``-> MTOP`` and
   returns a configured MTOP object. The annotation is the contract problem
   discovery keys off: a public method without it is treated as a helper, not a
   benchmark problem
3. Configurable sizes always use the standard parameter names: ``D`` (decision
   variables), ``M`` (objectives), ``K`` (tasks), ``Kp`` (WFG position parameters)
4. Task functions support batch computation using ``np.atleast_2d``

**Configuration Dictionary**

1. ``metric``: Performance metric type (**required**), e.g., ``'IGD'``, ``'HV'``
2. ``n_ref``: Number of reference points to sample from a PF generator
   (optional, default 10000)
3. ``ref_path``: Directory holding pre-stored reference files, used when a
   reference is given as a file name (optional, default ``'./MOReference'``)
4. Problem-task mapping: Nested dictionary structure ``{'P#': {'T#': reference_info}}``

**File Organization**

1. Recommended: Write problem class, reference generators, and SETTINGS in the same file
2. Alternative: Create separate ``problem_settings.py`` for complex configurations
3. File naming: lowercase with underscores (e.g., ``cec17_mtmo.py``)
4. Class naming: UpperCamelCase (e.g., ``CEC17MTMO``)

Benchmark Problems
------------------

D²MTOLab ships 28 problem suites holding 211 benchmark problems. Every
problem is a method annotated ``-> MTOP`` that returns a fully configured MTOP
instance; ``D``, ``M``, ``K`` and ``Kp`` in the tables below are the parameter
names those methods accept.

Single-Task Single-Objective (STSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 8 10 14 10 42

   * - Suite
     - Problems
     - Tasks
     - Dim
     - Obj / Con
     - Description
   * - ``CLASSICALSO``
     - 9
     - 1
     - D
     - 1 / 0
     - Classical landscapes (Ackley, Elliptic, Griewank, Rastrigin, Rosenbrock, Schwefel, Schwefel 2.22, Sphere, Weierstrass), unrotated and unshifted
   * - ``STSOtest``
     - 9
     - 1
     - D
     - 1 / 0
     - The same classical landscapes, rotated and shifted so the optimum is not at the origin
   * - ``CEC10_CSO``
     - 18
     - 1
     - 10-30
     - 1 / 2-5
     - CEC 2010 constrained real-parameter optimization (P1-P18); D is capped at 30 by the shipped offset vectors

**Usage Example:**

.. code-block:: python

   from ddmtolab.Problems.STSO import CLASSICALSO

   problem = CLASSICALSO().P1(D=30)   # 30-dimensional Ackley

Single-Task Multi-Objective (STMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 8 10 14 10 42

   * - Suite
     - Problems
     - Tasks
     - Dim
     - Obj / Con
     - Description
   * - ``ZDT``
     - 6
     - 1
     - D
     - 2 / 0
     - ZDT suite (ZDT1-ZDT6): convex, non-convex, disconnected and multi-modal Pareto fronts
   * - ``DTLZ``
     - 9
     - 1
     - D
     - M / 0-M
     - DTLZ suite (DTLZ1-DTLZ9), scalable in M and D; DTLZ8-DTLZ9 are constrained. Default D = M + k - 1
   * - ``WFG``
     - 9
     - 1
     - D
     - M / 0
     - WFG suite (WFG1-WFG9), scalable in M, D and the position parameter Kp. Default D = Kp + 10
   * - ``UF``
     - 10
     - 1
     - D
     - 2-3 / 0
     - CEC 2009 unconstrained suite (UF1-UF10): complicated Pareto sets
   * - ``CF``
     - 10
     - 1
     - D
     - 2-3 / 1-2
     - CEC 2009 constrained suite (CF1-CF10); CF6 and CF7 carry two constraints
   * - ``MW``
     - 14
     - 1
     - D
     - 2-3 / 1-4
     - Ma-Wang constrained suite (MW1-MW14): challenging constraint boundaries

**Usage Example:**

.. code-block:: python

   from ddmtolab.Problems.STMO import ZDT, DTLZ

   problem = ZDT().ZDT1(D=30)         # 30 decision variables
   problem = DTLZ().DTLZ2(M=5)        # 5 objectives, D defaults to M + k - 1

   # the reference Pareto fronts live next to the suite
   from ddmtolab.Problems.STMO.DTLZ import SETTINGS

Multi-Task Single-Objective (MTSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 8 10 14 10 42

   * - Suite
     - Problems
     - Tasks
     - Dim
     - Obj / Con
     - Description
   * - ``CEC17MTSO``
     - 9
     - 2
     - 25-50
     - 1 / 0
     - CEC 2017 EMTO competition (P1-P9): complete/partial/no intersection at high/medium/low similarity. Task 2 of P6 is 25-dimensional
   * - ``CEC17MTSO_10D``
     - 9
     - 2
     - 10
     - 1 / 0
     - 10-dimensional variant of CEC17MTSO, for expensive / surrogate-assisted studies
   * - ``CEC19MaTSO``
     - 6
     - K
     - 50
     - 1 / 0
     - CEC 2019 many-task optimization: K tasks generated from one landscape by rotation and shift
   * - ``CMT``
     - 9
     - 2
     - D
     - 1 / 1-2
     - Constrained multi-task benchmark (CMT1-CMT9); task 2 of CMT6 and CMT8 carries two constraints
   * - ``ManyTask_10D``
     - 4
     - 3-5
     - 10
     - 1 / 0
     - Compact many-task problems (P1-P4) with 3 or 5 tasks and varying similarity
   * - ``STOP``
     - 12
     - K
     - 25-50
     - 1 / 0
     - Scalable test problem generator for sequential transfer optimization (STOP1-STOP12)

**Usage Example:**

.. code-block:: python

   from ddmtolab.Problems.MTSO import CEC17MTSO

   problem = CEC17MTSO().P1()         # CI-HS: complete intersection, high similarity
   print(problem.n_tasks, problem.dims)

Multi-Task Multi-Objective (MTMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 8 10 14 10 42

   * - Suite
     - Problems
     - Tasks
     - Dim
     - Obj / Con
     - Description
   * - ``CEC17MTMO``
     - 9
     - 2
     - 10-50
     - 2-3 / 0
     - CEC 2017 MTMO competition (P1-P9): bi-objective and tri-objective task pairs
   * - ``CEC19MTMO``
     - 10
     - 2
     - 10-30
     - 2-3 / 0
     - CEC 2019 CPLX benchmark (P1-P10), LZ09-based, with complicated Pareto sets
   * - ``CEC19_MaTMO``
     - 6
     - K
     - 10-50
     - 2 / 0
     - CEC 2019 many-task multi-objective benchmark (P1-P6); P3 is 10-dimensional
   * - ``CEC21MTMO``
     - 10
     - 2
     - 50
     - 2 / 0
     - CEC 2021 MTMO benchmark (P1-P10)
   * - ``MTMOInstances``
     - 2
     - 2
     - 10
     - 2 / 0-1
     - Compact ZDT4-based instances (P1-P2); P2 is constrained

**Usage Example:**

.. code-block:: python

   from ddmtolab.Problems.MTMO import CEC17MTMO
   from ddmtolab.Problems.MTMO.cec17_mtmo import SETTINGS

   problem = CEC17MTMO().P8()         # T1: 3-objective, T2: 2-objective
   print(problem.n_objs)

Real-World Optimization (RWO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 8 10 14 10 42

   * - Suite
     - Problems
     - Tasks
     - Dim
     - Obj / Con
     - Description
   * - ``PEPVM``
     - 1
     - 3
     - 5-7
     - 1 / 0
     - Parameter extraction of photovoltaic models: single-diode, double-diode and PV-module tasks
   * - ``SOPM``
     - 2
     - 3
     - 25-30
     - 2 / 24-29
     - Synchronous optimal pulse-width modulation: harmonic distortion vs switching loss, monotone-angle constraints
   * - ``NN_Training``
     - 6
     - 1
     - 101-1867
     - 1 / 0
     - Neural network weight optimization by neuroevolution on scikit-learn datasets
   * - ``TSP``
     - 6
     - 1
     - 20-200
     - 1 / 0
     - Traveling salesman problem via random-keys encoding (20-200 cities)
   * - ``SCP``
     - 1
     - 11
     - 75-105
     - 1 / 0
     - Sensor coverage problem: one task per sensor count, so tasks differ in dimension
   * - ``MO_SCP``
     - 2
     - 4-5
     - 75-102
     - 2 / 0
     - Multi-objective sensor coverage problem: coverage vs deployment cost
   * - ``PKACP``
     - 1
     - K
     - D
     - 1 / 0
     - Planar kinematic arm control: reach a target with a D-joint arm, K tasks with different arm geometries
   * - ``PINN_HPO``
     - 12
     - 2-4
     - 6
     - 1 / 0
     - Hyperparameter optimization of physics-informed neural networks (convection, reaction, wave, Helmholtz)

**Usage Example:**

.. code-block:: python

   from ddmtolab.Problems.RWO import PEPVM

   problem = PEPVM().P1()             # 3 photovoltaic model-fitting tasks
   print(problem.n_tasks, problem.dims)

See Also
--------

* :ref:`algorithms` - Algorithm implementation guide
* :ref:`api` - Complete API documentation
.. _api:

API
===

Algorithms
----------

Single-Task Single-Objective (STSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GA
^^

.. automodule:: Algorithms.STSO.GA
   :no-members:

.. autoclass:: Algorithms.STSO.GA.GA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

DE
^^

.. automodule:: Algorithms.STSO.DE
   :no-members:

.. autoclass:: Algorithms.STSO.DE.DE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

PSO
^^^

.. automodule:: Algorithms.STSO.PSO
   :no-members:

.. autoclass:: Algorithms.STSO.PSO.PSO
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

CSO
^^^

.. automodule:: Algorithms.STSO.CSO
   :no-members:

.. autoclass:: Algorithms.STSO.CSO.CSO
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

BO
^^

.. automodule:: Algorithms.STSO.BO
   :no-members:

.. autoclass:: Algorithms.STSO.BO.BO
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multi-Task Single-Objective (MTSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MFEA
^^^^

.. automodule:: Algorithms.MTSO.MFEA
   :no-members:

.. autoclass:: Algorithms.MTSO.MFEA.MFEA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autofunction:: Algorithms.MTSO.MFEA.mfea_selection

EMEA
^^^^

.. automodule:: Algorithms.MTSO.EMEA
   :no-members:

.. autoclass:: Algorithms.MTSO.EMEA.EMEA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autofunction:: Algorithms.MTSO.EMEA.mDA

G-MFEA
^^^^^^

.. automodule:: Algorithms.MTSO.GMFEA
   :no-members:

.. autoclass:: Algorithms.MTSO.GMFEA.GMFEA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autofunction:: Algorithms.MTSO.GMFEA.decs_translation
.. autofunction:: Algorithms.MTSO.GMFEA.dimension_shuffling
.. autofunction:: Algorithms.MTSO.GMFEA.convert_to_original_decision_space

MTBO
^^^^

.. automodule:: Algorithms.MTSO.MTBO
   :no-members:

.. autoclass:: Algorithms.MTSO.MTBO.MTBO
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

RAMTEA
^^^^^^^

.. automodule:: Algorithms.MTSO.RAMTEA
   :no-members:

.. autoclass:: Algorithms.MTSO.RAMTEA.RAMTEA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autofunction:: Algorithms.MTSO.RAMTEA.ramtea_knowledge_transfer

SELF
^^^^

.. automodule:: Algorithms.MTSO.SELF
   :no-members:

.. autoclass:: Algorithms.MTSO.SELF.SELF
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autofunction:: Algorithms.MTSO.SELF.de_generation_with_core
.. autofunction:: Algorithms.MTSO.SELF.bo_next_point_de


Single-Task Multi-Objective (STMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NSGA-II
^^^^^^^

.. automodule:: Algorithms.STMO.NSGAII
   :no-members:

.. autoclass:: Algorithms.STMO.NSGAII.NSGAII
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

RVEA
^^^^

.. automodule:: Algorithms.STMO.RVEA
   :no-members:

.. autoclass:: Algorithms.STMO.RVEA.RVEA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multi-Task Multi-Objective (MTMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MO-MFEA
^^^^^^^

.. automodule:: Algorithms.MTMO.MOMFEA
   :no-members:

.. autoclass:: Algorithms.MTMO.MOMFEA.MOMFEA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autofunction:: Algorithms.MTMO.MOMFEA.momfea_selection

Problems
--------

Single-Task Single-Objective Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classical Single-Objective Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: Problems.STSO.classical_so
   :members:
   :undoc-members:

Single-Task Multi-Objective Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DTLZ Test Suite
^^^^^^^^^^^^^^^

.. automodule:: Problems.STMO.DTLZ
   :members:
   :undoc-members:

Multi-Task Single-Objective Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CEC17 MTSO Benchmark
^^^^^^^^^^^^^^^^^^^^

.. automodule:: Problems.MTSO.cec17_mtso
   :members:
   :undoc-members:

CEC17 MTSO 10D Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: Problems.MTSO.cec17_mtso_10d
   :members:
   :undoc-members:

CEC19 MATSO Benchmark
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: Problems.MTSO.cec19_matso
   :members:
   :undoc-members:

Multi-Task Multi-Objective Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CEC17 MTMO Benchmark
^^^^^^^^^^^^^^^^^^^^

.. automodule:: Problems.MTMO.cec17_mtmo
   :members:
   :undoc-members:

Real-World Optimization Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PINN Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Problems.RWP.PINN_HPO.pinn_hpo.PINN_HPO
   :members: P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12
   :undoc-members:

Methods and Utilities
---------------------

Batch Experiment
~~~~~~~~~~~~~~~~

.. automodule:: Methods.batch_experiment
   :members:
   :undoc-members:
   :show-inheritance:

Data Analysis
~~~~~~~~~~~~~

.. automodule:: Methods.data_analysis
   :members:
   :undoc-members:
   :noindex:

Test Data Analysis
~~~~~~~~~~~~~~~~~~

.. automodule:: Methods.test_data_analysis
   :members:
   :undoc-members:
   :noindex:

Metrics
~~~~~~~

.. automodule:: Methods.metrics
   :members:
   :undoc-members:
   :noindex:

MTOP Base Class
~~~~~~~~~~~~~~~

The **MTOP (Multi-Task Optimization Problem)** class provides a flexible and powerful framework for defining and managing multiple optimization tasks with different characteristics.

**Key Features:**

* Multi-task Management: Define and manage multiple optimization tasks
* Flexible Function Handling: Supports vectorized and non-vectorized functions
* Cross-platform Compatibility: Pickle-compatible wrappers for parallel execution
* Unified Evaluation Mode: Optional padding of results to consistent dimensions
* Selective Evaluation: Evaluate specific objectives or constraints

**Quick Example:**

.. code-block:: python

    from Methods.mtop import MTOP
    import numpy as np

    def sphere(x):
        return np.sum(x**2, axis=1)

    mtop = MTOP()
    mtop.add_task(sphere, dim=3)
    X = np.random.rand(10, 3)
    objectives, constraints = mtop.evaluate_task(0, X)

MTOP Class
^^^^^^^^^^

.. autoclass:: Methods.mtop.MTOP
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ObjectiveFunctionWrapper
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Methods.mtop.ObjectiveFunctionWrapper
   :members:
   :undoc-members:
   :show-inheritance:

ConstraintFunctionWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Methods.mtop.ConstraintFunctionWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: Methods.Algo_Methods.algo_utils
   :members:
   :undoc-members:
   :noindex:

Bayesian Optimization Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: Methods.Algo_Methods.bo_utils
   :members:
   :undoc-members:

Similarity Evaluation
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: Methods.Algo_Methods.sim_evaluation
   :members:
   :undoc-members:

Uniform Point Generation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: Methods.Algo_Methods.uniform_point
   :members:
   :undoc-members:
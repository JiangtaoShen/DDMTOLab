.. _quickstart:

Quick Start Guide
=================

Basic Example
-------------

Here's a simple example using Bayesian Optimization:

.. code-block:: python

   from Algorithms.STSO.BO import BO
   from Problems.Synthetic_MTOPs.PI_MTO import PI_MTO

   # Create a 2-task problem with 10 dimensions each
   problem = PI_MTO(n_tasks=2, dims=[10, 10])

   # Initialize optimizer
   optimizer = BO(
       problem=problem,
       n_initial=10,      # Initial samples per task
       max_nfes=50,       # Max evaluations per task
       save_data=True,
       save_path='./Data'
   )

   # Run optimization
   results = optimizer.optimize()

   # Print results
   print(f"Best objectives: {results['best_objs']}")
   print(f"Runtime: {results['runtime']:.2f} seconds")

Multi-Task Example
------------------

Using Multi-Task Bayesian Optimization with knowledge transfer:

.. code-block:: python

   from Algorithms.MTSO.MTBO import MTBO
   from Problems.Real_World_MTOPs.PINN_HPO.pinn_hpo import PINN_HPO

   # Create real-world problem
   pinn = PINN_HPO()

   # Run MTBO on problem P3
   optimizer = MTBO(
       problem=pinn.P3,
       n_initial=10,
       max_nfes=50,
       save_data=True
   )

   results = optimizer.optimize()

Batch Experiments
-----------------

Run multiple algorithms on multiple problems:

.. code-block:: python

   from Methods.batch_experiment import BatchExperiment
   from Algorithms.STSO.BO import BO
   from Algorithms.MTSO.MTBO import MTBO
   from Problems.Real_World_MTOPs.PINN_HPO.pinn_hpo import PINN_HPO

   # Setup batch experiment
   batch_exp = BatchExperiment(
       base_path='./Results',
       clear_folder=True
   )

   # Add problems
   pinn = PINN_HPO()
   batch_exp.add_problem(pinn.P3, 'P3')
   batch_exp.add_problem(pinn.P4, 'P4')

   # Add algorithms
   batch_exp.add_algorithm(BO, 'BO', n_initial=10, max_nfes=50)
   batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=10, max_nfes=50)

   # Run experiments
   batch_exp.run()

Working with Results
--------------------

.. code-block:: python

   # Access results
   best_solutions = results['decs']      # Best solutions per task
   best_objectives = results['objs']     # Best objectives per task
   all_history = results['all_objs']     # Complete history
   runtime = results['runtime']          # Total runtime

Visualization
-------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Plot convergence
   for task_id, objs in enumerate(results['all_objs']):
       running_min = np.minimum.accumulate(objs)
       plt.plot(running_min, label=f'Task {task_id}')

   plt.xlabel('Function Evaluations')
   plt.ylabel('Best Objective Value')
   plt.legend()
   plt.grid(True)
   plt.show()

Next Steps
----------

* Explore the :ref:`api_reference` for detailed documentation
* Check available algorithms and problems
* Try custom problems and algorithms
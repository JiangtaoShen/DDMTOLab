"""Benchmark problem suites.

The problems are grouped into five categories, one subpackage each:

===============  ==============================================================
``STSO``         Single-Task Single-Objective
``STMO``         Single-Task Multi-Objective
``MTSO``         Multi-Task Single-Objective
``MTMO``         Multi-Task Multi-Objective
``RWO``          Real-World Optimization (single/multi task, single/multi objective)
===============  ==============================================================

``BasicFunctions`` holds the shared landscape primitives (Ackley, Griewank,
Rastrigin, ...) that the synthetic suites are built from.

Every suite follows the same contract:

- the suite is a class carrying a ``problem_information`` dict
  (``n_cases``, ``n_tasks``, ``n_dims``, ``n_objs``, ``n_cons``, ``type``);
- every benchmark problem is a public method annotated ``-> MTOP``;
- configurable sizes always use the same parameter names: ``D`` (decision
  variables), ``M`` (objectives), ``K`` (tasks), ``Kp`` (WFG position
  parameters);
- multi-objective suites additionally expose a module-level ``SETTINGS`` dict
  describing the metric and the reference data for each problem.

Examples
--------
>>> from ddmtolab.Problems.STMO import ZDT
>>> problem = ZDT().ZDT1(D=30)
>>> problem.n_tasks
1

The subpackages are not imported eagerly: importing a category only pulls in
that category's dependencies (``RWO`` needs PyTorch, the synthetic suites do
not).
"""

__all__ = ['STSO', 'STMO', 'MTSO', 'MTMO', 'RWO', 'BasicFunctions']

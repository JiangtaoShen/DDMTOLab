"""Real-World Optimization (RWO) problem suites.

=================  =========  =======  =================================================
Suite              Problems   Kind     Description
=================  =========  =======  =================================================
``PEPVM``          1          MTSO     Photovoltaic model parameter extraction
``SOPM``           2          MTMO     Synchronous optimal pulse-width modulation
``NN_Training``    6          STSO     Neural network weight optimization
``TSP``            6          STSO     Traveling salesman, random-keys encoding
``SCP``            1          MTSO     Sensor coverage problem
``MO_SCP``         2          MTMO     Multi-objective sensor coverage problem
``PKACP``          1          MTSO     Planar kinematic arm control
``PINN_HPO``       12         MTSO     Physics-informed neural network HPO
=================  =========  =======  =================================================

``MO_SCP`` and ``SOPM`` also define a module-level ``SETTINGS`` dict (metric
plus reference points) used by the analysis tools::

    from ddmtolab.Problems.RWO.sopm import SOPM, SETTINGS

Notes
-----
``NN_Training`` and ``PINN_HPO`` evaluate by training a network, so a single
objective evaluation is expensive; ``MTOP`` probes an objective once while the
problem is being built, which makes constructing a ``PINN_HPO`` problem
expensive too.

Examples
--------
>>> from ddmtolab.Problems.RWO import PEPVM
>>> problem = PEPVM().P1()
>>> problem.n_tasks
3
"""

from ddmtolab.Problems.RWO.pepvm import PEPVM
from ddmtolab.Problems.RWO.sopm import SOPM
from ddmtolab.Problems.RWO.nn_training import NN_Training
from ddmtolab.Problems.RWO.tsp import TSP
from ddmtolab.Problems.RWO.scp import SCP
from ddmtolab.Problems.RWO.mo_scp import MO_SCP
from ddmtolab.Problems.RWO.pkacp import PKACP
from ddmtolab.Problems.RWO.pinn_hpo import PINN_HPO

__all__ = ["PEPVM", "SOPM", "NN_Training", "TSP", "SCP", "MO_SCP", "PKACP", "PINN_HPO"]

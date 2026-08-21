"""Multi-Task Multi-Objective (MTMO) problem suites.

===================  =========  ==========================================
Suite                Problems   Description
===================  =========  ==========================================
``CEC17MTMO``        9          CEC 2017 MTMO benchmark (2 tasks)
``CEC19MTMO``        10         CEC 2019 CPLX benchmark, LZ09-based
``CEC19_MaTMO``      6          CEC 2019 many-task MO benchmark (K tasks)
``CEC21MTMO``        10         CEC 2021 MTMO benchmark
``MTMOInstances``    2          Compact ZDT4-based instances
===================  =========  ==========================================

Each suite module also defines a ``SETTINGS`` dict (metric plus the reference
Pareto front of every problem) used by the analysis tools; import it from the
module it belongs to::

    from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS

Examples
--------
>>> from ddmtolab.Problems.MTMO import CEC17MTMO
>>> problem = CEC17MTMO().P1()
"""

from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO
from ddmtolab.Problems.MTMO.cec19_matmo import CEC19_MaTMO
from ddmtolab.Problems.MTMO.cec19_mtmo import CEC19MTMO
from ddmtolab.Problems.MTMO.cec21_mtmo import CEC21MTMO
from ddmtolab.Problems.MTMO.mtmo_instance import MTMOInstances

__all__ = [
    "CEC17MTMO", "CEC19MTMO", "CEC19_MaTMO", "CEC21MTMO", "MTMOInstances",
]

"""Multi-Task Single-Objective (MTSO) problem suites.

===================  =========  ==========================================
Suite                Problems   Description
===================  =========  ==========================================
``CEC17MTSO``        9          CEC 2017 EMTO benchmark (2 tasks, D=50)
``CEC17MTSO_10D``    9          10-dimensional variant for expensive runs
``CEC19MaTSO``       6          CEC 2019 many-task benchmark (K tasks)
``CMT``              9          Constrained multi-task benchmark
``ManyTask_10D``     4          Compact 3-5 task, 10-dimensional problems
``STOP``             12         Scalable transfer optimization problems
===================  =========  ==========================================

Examples
--------
>>> from ddmtolab.Problems.MTSO import CEC17MTSO
>>> problem = CEC17MTSO().P1()
>>> problem.n_tasks
2
"""

from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Problems.MTSO.cec19_matso import CEC19MaTSO
from ddmtolab.Problems.MTSO.cmt import CMT
from ddmtolab.Problems.MTSO.many_task_10d import ManyTask_10D
from ddmtolab.Problems.MTSO.stop import STOP

__all__ = [
    "CEC17MTSO", "CEC17MTSO_10D", "CEC19MaTSO", "CMT", "ManyTask_10D", "STOP",
]

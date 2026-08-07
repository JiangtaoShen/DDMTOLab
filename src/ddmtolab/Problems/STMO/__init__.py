"""Single-Task Multi-Objective (STMO) problem suites.

=========  =========  ==================================================
Suite      Problems   Description
=========  =========  ==================================================
``ZDT``    6          Zitzler-Deb-Thiele bi-objective suite
``DTLZ``   9          Deb-Thiele-Laumanns-Zitzler scalable suite
``WFG``    9          Walking Fish Group scalable suite
``UF``     10         CEC 2009 unconstrained suite
``CF``     10         CEC 2009 constrained suite
``MW``     14         Ma-Wang constrained suite
=========  =========  ==================================================

Each suite module also defines a ``SETTINGS`` dict (metric plus the reference
Pareto front of every problem) used by the analysis tools. Because all six
suites use the same name, import it from the module it belongs to::

    from ddmtolab.Problems.STMO.DTLZ import DTLZ, SETTINGS

Examples
--------
>>> from ddmtolab.Problems.STMO import ZDT
>>> problem = ZDT().ZDT1(D=30)
"""

from ddmtolab.Problems.STMO.CF import CF
from ddmtolab.Problems.STMO.DTLZ import DTLZ
from ddmtolab.Problems.STMO.MW import MW
from ddmtolab.Problems.STMO.UF import UF
from ddmtolab.Problems.STMO.WFG import WFG
from ddmtolab.Problems.STMO.ZDT import ZDT

__all__ = ["CF", "DTLZ", "MW", "UF", "WFG", "ZDT"]

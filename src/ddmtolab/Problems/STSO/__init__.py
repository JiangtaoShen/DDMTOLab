"""Single-Task Single-Objective (STSO) problem suites.

===============  =========  =============================================
Suite            Problems   Description
===============  =========  =============================================
``CEC10_CSO``    18         CEC 2010 constrained real-parameter benchmark
``CLASSICALSO``  9          Classical unconstrained landscapes
``STSOtest``     9          Classical landscapes, rotated and shifted
===============  =========  =============================================

Examples
--------
>>> from ddmtolab.Problems.STSO import CLASSICALSO
>>> problem = CLASSICALSO().P1(D=50)
"""

from ddmtolab.Problems.STSO.cec10_cso import CEC10_CSO
from ddmtolab.Problems.STSO.classical_so import CLASSICALSO
from ddmtolab.Problems.STSO.stsotest import STSOtest

__all__ = ["CEC10_CSO", "CLASSICALSO", "STSOtest"]

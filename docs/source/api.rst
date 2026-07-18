.. _api:

API Reference
=============

This page provides API documentation for the main classes and methods in D²MTOLab.

Algorithms
----------

All algorithms follow a consistent interface:

.. code-block:: python

   from ddmtolab.Algorithms.STSO.DE import DE
   from ddmtolab.Methods.mtop import MTOP

   problem = MTOP()
   problem.add_task(objective_func, dim=10)

   optimizer = DE(problem, n=50, max_nfes=1000)
   results = optimizer.optimize()

   print(results.best_decs, results.best_objs)

Single-Task Single-Objective (STSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Algorithms.STSO.GA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.GA.GA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.DE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.DE.DE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.PSO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.PSO.PSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.CMA_ES
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.CMA_ES.CMA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.IPOP_CMA_ES
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.IPOP_CMA_ES.IPOP_CMA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.sep_CMA_ES
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.sep_CMA_ES.sep_CMA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.MA_ES
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.MA_ES.MA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.OpenAI_ES
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.OpenAI_ES.OpenAI_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.xNES
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.xNES.xNES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.CSO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.CSO.CSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.SL_PSO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.SL_PSO.SL_PSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.KLPSO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.KLPSO.KLPSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.SHPSO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.SHPSO.SHPSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.GWO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.GWO.GWO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.AO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.AO.AO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.EO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.EO.EO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.GL_SADE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.GL_SADE.GL_SADE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.BO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.BO.BO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.EEI_BO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.EEI_BO.EEI_BO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.ESAO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.ESAO.ESAO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.SA_COSO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.SA_COSO.SA_COSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.TLRBF
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.TLRBF.TLRBF
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.AutoSAEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.AutoSAEA.AutoSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.DDEA_MESS
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.DDEA_MESS.DDEA_MESS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STSO.LSADE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STSO.LSADE.LSADE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Single-Task Multiobjective (STMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Algorithms.STMO.NSGA_II
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.NSGA_II.NSGA_II
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.NSGA_III
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.NSGA_III.NSGA_III
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.NSGA_II_SDR
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.NSGA_II_SDR.NSGA_II_SDR
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MOEA_D
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D.MOEA_D
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MOEA_DD
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_DD.MOEA_DD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MOEA_D_STM
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D_STM.MOEA_D_STM
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MOEA_D_FRRMAB
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D_FRRMAB.MOEA_D_FRRMAB
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MCEA_D
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MCEA_D.MCEA_D
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.RVEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.RVEA.RVEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.K_RVEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.K_RVEA.K_RVEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.IBEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.IBEA.IBEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.SPEA2
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.SPEA2.SPEA2
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.TwoArch2
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.TwoArch2.TwoArch2
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.CCMO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.CCMO.CCMO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.C_TAEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.C_TAEA.C_TAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.CPS_MOEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.CPS_MOEA.CPS_MOEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.ParEGO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.ParEGO.ParEGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MSEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MSEA.MSEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.REMO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.REMO.REMO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.KTA2
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.KTA2.KTA2
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.DSAEA_PS
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.DSAEA_PS.DSAEA_PS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.ADSAPSO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.ADSAPSO.ADSAPSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.CSEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.CSEA.CSEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.DISK
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.DISK.DISK
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.DRLSAEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.DRLSAEA.DRLSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.DirHV_EI
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.DirHV_EI.DirHV_EI
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.EDN_ARMOEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.EDN_ARMOEA.EDN_ARMOEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.EIM_EGO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.EIM_EGO.EIM_EGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.EM_SAEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.EM_SAEA.EM_SAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.KTS
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.KTS.KTS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MGSAEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MGSAEA.MGSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MMRAEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MMRAEA.MMRAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MOEA_D_EGO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D_EGO.MOEA_D_EGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.MultiObjectiveEGO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.MultiObjectiveEGO.MultiObjectiveEGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.PCSAEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.PCSAEA.PCSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.PEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.PEA.PEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.PIEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.PIEA.PIEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.SAEA_DBLL
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.SAEA_DBLL.SAEA_DBLL
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.SSDE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.SSDE.SSDE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.STMO.TEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.STMO.TEA.TEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multitask Single-Objective (MTSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Algorithms.MTSO.MFEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA.MFEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MFEA_II
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA_II.MFEA_II
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.G_MFEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.G_MFEA.G_MFEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MTEA_AD
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MTEA_AD.MTEA_AD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MTEA_SaO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MTEA_SaO.MTEA_SaO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.EMEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.EMEA.EMEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MKTDE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MKTDE.MKTDE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.SREMTO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.SREMTO.SREMTO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.RAMTEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.RAMTEA.RAMTEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.SELF
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.SELF.SELF
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.EBS
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.EBS.EBS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MTBO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MTBO.MTBO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MUMBO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MUMBO.MUMBO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.LCB_EMT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.LCB_EMT.LCB_EMT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.BO_LCB_CKT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.BO_LCB_CKT.BO_LCB_CKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.BO_LCB_BCKT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.BO_LCB_BCKT.BO_LCB_BCKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.EEI_BO_plus
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.EEI_BO_plus.EEI_BO_plus
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.BLKT_DE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.BLKT_DE.BLKT_DE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.DTSKT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.DTSKT.DTSKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.EMTO_AI
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.EMTO_AI.EMTO_AI
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MFEA_AKT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA_AKT.MFEA_AKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MFEA_DGD
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA_DGD.MFEA_DGD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MFEA_SSG
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA_SSG.MFEA_SSG
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MFEA_VC
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA_VC.MFEA_VC
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MTDE_ADKT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MTDE_ADKT.MTDE_ADKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MTEA_HKTS
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MTEA_HKTS.MTEA_HKTS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MTEA_PAE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MTEA_PAE.MTEA_PAE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.MTES_KG
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.MTES_KG.MTES_KG
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.SaEF_AKT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.SaEF_AKT.SaEF_AKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.SSLT_DE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.SSLT_DE.SSLT_DE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTSO.TNG_SNES
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTSO.TNG_SNES.TNG_SNES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multitask Multiobjective (MTMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Algorithms.MTMO.MO_MFEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_MFEA.MO_MFEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MO_MFEA_II
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_MFEA_II.MO_MFEA_II
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MO_EMEA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_EMEA.MO_EMEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MO_MTEA_SaO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_MTEA_SaO.MO_MTEA_SaO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MTEA_D_DN
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MTEA_D_DN.MTEA_D_DN
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MTDE_MKTA
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MTDE_MKTA.MTDE_MKTA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.EMT_ET
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.EMT_ET.EMT_ET
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.EMT_PD
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.EMT_PD.EMT_PD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.ParEGO_KT
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.ParEGO_KT.ParEGO_KT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.EMT_GS
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.EMT_GS.EMT_GS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MO_MTEA_PAE
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_MTEA_PAE.MO_MTEA_PAE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MO_SBO
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_SBO.MO_SBO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MTEA_D_TSD
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MTEA_D_TSD.MTEA_D_TSD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. automodule:: ddmtolab.Algorithms.MTMO.MTEA_DCK
   :no-members:

.. autoclass:: ddmtolab.Algorithms.MTMO.MTEA_DCK.MTEA_DCK
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Problems
--------

MTOP Class
~~~~~~~~~~

The **MTOP (Multitask Optimization Problem)** class is the core component for defining optimization problems.

.. autoclass:: ddmtolab.Methods.mtop.MTOP
   :members: add_task, add_tasks, evaluate_task, evaluate_tasks, get_task_info, set_unified_eval_mode
   :undoc-members:
   :show-inheritance:

Benchmark Problem Suites
~~~~~~~~~~~~~~~~~~~~~~~~

**STSO Problems:**

.. autoclass:: ddmtolab.Problems.STSO.classical_so.CLASSICALSO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STSO.cec10_cso.CEC10_CSO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STSO.stsotest.STSOtest
   :members:
   :undoc-members:

**STMO Problems:**

.. autoclass:: ddmtolab.Problems.STMO.ZDT.ZDT
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.DTLZ.DTLZ
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.WFG.WFG
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.UF.UF
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.CF.CF
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.MW.MW
   :members:
   :undoc-members:

**MTSO Problems:**

.. autoclass:: ddmtolab.Problems.MTSO.cec17_mtso.CEC17MTSO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.cec17_mtso_10d.CEC17MTSO_10D
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.cec19_matso.CEC19MaTSO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.cmt.CMT
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.stop.STOP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.many_task_10d.ManyTask_10D
   :members:
   :undoc-members:

**MTMO Problems:**

.. autoclass:: ddmtolab.Problems.MTMO.cec17_mtmo.CEC17MTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.cec19_mtmo.CEC19MTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.cec19_matmo.CEC19_MaTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.cec21_mtmo.CEC21MTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.mtmo_dtlz.MTMO_DTLZ
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.mtmo_instance.MTMOInstances
   :members:
   :undoc-members:

**Real-World Optimization (RWO) Problems:**

.. autoclass:: ddmtolab.Problems.RWO.pepvm.PEPVM
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.sopm.SOPM
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.nn_training.NN_Training
   :members:
   :undoc-members:

.. automodule:: ddmtolab.Problems.RWO.tsp
   :no-members:

.. autoclass:: ddmtolab.Problems.RWO.tsp.TSP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.scp.SCP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.mo_scp.MO_SCP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.pkacp.PKACP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.pinn_hpo.PINN_HPO
   :members:
   :undoc-members:

Methods and Utilities
---------------------

Batch Experiment
~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Methods.batch_experiment.BatchExperiment
   :members: add_problem, add_algorithm, run
   :undoc-members:
   :show-inheritance:

Data Analysis
~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Methods.data_analysis.DataAnalyzer
   :members: run
   :undoc-members:

.. autoclass:: ddmtolab.Methods.test_data_analysis.TestDataAnalyzer
   :members: run
   :undoc-members:

Performance Metrics
~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.metrics
   :members: IGD, GD, IGDp, HV, DeltaP, Spacing, Spread, FR, CV
   :undoc-members:

Algorithm Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.Algo_Methods.algo_utils
   :members: Results, initialization, evaluation, nd_sort, crowding_distance, tournament_selection, ga_generation, de_generation, init_history, append_history, build_save_results, rbf_build, rbf_predict, dsmerge, merge_archive, spea2_fitness, spea2_truncation, spea2_truncation_fast, reorganize_initial_data
   :undoc-members:

Bayesian Optimization Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.Algo_Methods.bo_utils
   :members:
   :undoc-members:

Uniform Point Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ddmtolab.Methods.Algo_Methods.uniform_point.uniform_point

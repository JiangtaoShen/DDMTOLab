Data-Driven Multitask Optimization Laboratory
==============================================

.. image:: _static/logo_new.svg
   :alt: D²MTOLab Logo
   :width: 500
   :align: center

|

.. image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://jiangtaoshen.github.io/DDMTOLab/
   :alt: Documentation

.. image:: https://img.shields.io/pypi/v/ddmtolab.svg
   :target: https://pypi.org/project/ddmtolab/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/dm/ddmtolab.svg
   :target: https://pypi.org/project/ddmtolab/
   :alt: Downloads

.. image:: https://img.shields.io/github/stars/JiangtaoShen/DDMTOLab?style=social
   :target: https://github.com/JiangtaoShen/DDMTOLab/stargazers
   :alt: GitHub Stars

.. image:: https://img.shields.io/badge/Python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/JiangtaoShen/DDMTOLab/blob/main/LICENSE
   :alt: License

----

Overview
--------

**D²MTOLab (Data-Driven Multitask Optimization Laboratory)** is a comprehensive Python platform for optimization research, featuring **115 algorithms**, **212 benchmark problems**, and powerful tools for problem definition, algorithm development, and performance evaluation.

Whether you're working on expensive black-box optimization, multiobjective optimization, or complex multitask scenarios, D²MTOLab provides a flexible and extensible framework to accelerate your **research** and support real-world **applications**.

Features
--------

* **Comprehensive Algorithms** - 115 algorithms for expensive/inexpensive, single/multitask, single/multiobjective optimization
* **Rich Problem Suites** - 211 benchmark problems in 28 suites, synthetic and real-world
* **Data-Driven Optimization** - Surrogate modelling (GP, BO) for expensive optimization
* **Flexible Framework** - Simple API and intuitive workflow for rapid prototyping
* **Fully Extensible** - Easy to add custom algorithms and problems
* **Powerful Analysis Tools** - Built-in visualization and statistical analysis
* **Parallel Computing** - Multi-core support for batch experiments
* **Desktop GUI** - Optional point-and-click interface for running experiments (see :ref:`gui`)

Documentation
-------------

.. toctree::
   :maxdepth: 1

   installation
   demos
   gui
   algorithms
   methods
   problems
   api

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install ddmtolab

**Requirements:** Python 3.10+, PyTorch 2.5+, NumPy 2.0+

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ddmtolab.Methods.mtop import MTOP
   from ddmtolab.Algorithms.MTSO.MTBO import MTBO

   # Define objective function
   def forrester(x):
       return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

   # Create and solve optimization problem
   problem = MTOP()
   problem.add_task(forrester, dim=1)
   results = MTBO(problem).optimize()

   print(f"Best solution: {results.best_decs}")
   print(f"Best objective: {results.best_objs}")

Batch Experiments
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ddmtolab.Methods.batch_experiment import BatchExperiment
   from ddmtolab.Methods.data_analysis import DataAnalyzer
   from ddmtolab.Algorithms.STSO.BO import BO
   from ddmtolab.Algorithms.MTSO.MTBO import MTBO
   from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D

   if __name__ == '__main__':
       batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

       prob = CEC17MTSO_10D()
       batch_exp.add_problem(prob.P1, 'P1')
       batch_exp.add_problem(prob.P2, 'P2')

       batch_exp.add_algorithm(BO, 'BO', n_initial=20, max_nfes=100)
       batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=20, max_nfes=100)

       batch_exp.run(n_runs=20, max_workers=8)
       DataAnalyzer().run()

Optimization Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/animation.gif
   :alt: Optimization Animation
   :width: 100%
   :align: center

Key Components
--------------

Algorithms (115)
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 12 12 76

   * - Category
     - Type
     - Algorithms
   * - **STSO**
     - Inexpensive
     - GA, DE, PSO, SL_PSO, KLPSO, CSO, CMA_ES, IPOP_CMA_ES, sep_CMA_ES, MA_ES, xNES, OpenAI_ES, AO, GWO, EO
   * - **STSO**
     - Expensive
     - BO, EEI_BO, ESAO, SHPSO, SA_COSO, TLRBF, GL_SADE, AutoSAEA, DDEA_MESS, LSADE, LAEA
   * - **STMO**
     - Inexpensive
     - NSGA_II, NSGA_III, NSGA_II_SDR, SPEA2, MOEA_D, MOEA_DD, MOEA_D_FRRMAB, MOEA_D_STM, RVEA, IBEA, TwoArch2, MSEA, C_TAEA, CCMO
   * - **STMO**
     - Expensive
     - ParEGO, K_RVEA, DSAEA_PS, KTA2, REMO, ADSAPSO, CSEA, DISK, DRLSAEA, DirHV_EI, EDN_ARMOEA, EIM_EGO, EM_SAEA, KTS, MGSAEA, MMRAEA, MOEA_D_EGO, MultiObjectiveEGO, PCSAEA, PEA, PIEA, SAEA_DBLL, SSDE, TEA, CPS_MOEA, MCEA_D
   * - **MTSO**
     - Inexpensive
     - MFEA, MFEA_II, EMEA, EBS, G_MFEA, MTEA_AD, MKTDE, MTEA_SaO, SREMTO, LCB_EMT, BLKT_DE, DTSKT, EMTO_AI, MFEA_AKT, MFEA_DGD, MFEA_VC, MTDE_ADKT, MTEA_HKTS, MTEA_PAE, MTES_KG, SSLT_DE, TNG_SNES
   * - **MTSO**
     - Expensive
     - MTBO, RAMTEA, SELF, EEI_BO_plus, MUMBO, BO_LCB_CKT, BO_LCB_BCKT, MFEA_SSG, SaEF_AKT
   * - **MTMO**
     - Inexpensive
     - MO_MFEA, MO_MFEA_II, MO_EMEA, MO_MTEA_SaO, MTDE_MKTA, MTEA_D_DN, EMT_ET, EMT_PD, EMT_GS, MO_MTEA_PAE, MO_SBO, MTEA_D_TSD, MTEA_DCK
   * - **MTMO**
     - Expensive
     - ParEGO_KT, SAEA_AKT

Problems (28 suites, 211 problems)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Category
     - Problem Suites
   * - **STSO**
     - CLASSICALSO (9), STSOtest (9), CEC10_CSO (18)
   * - **STMO**
     - ZDT (6), DTLZ (9), WFG (9), UF (10), CF (10), MW (14)
   * - **MTSO**
     - CEC17MTSO (9), CEC17MTSO_10D (9), CEC19MaTSO (6), CMT (9), STOP (12), ManyTask_10D (4)
   * - **MTMO**
     - CEC17MTMO (9), CEC19MTMO (10), CEC19_MaTMO (6), CEC21MTMO (10), MTMOInstances (2)
   * - **RWO**
     - PEPVM (1), PINN_HPO (12), SOPM (2), SCP (1), MO_SCP (2), PKACP (1), NN_Training (6), TSP (6)

Citation
--------

If you use D²MTOLab in your research, please cite:

.. code-block:: bibtex

   @software{ddmtolab2025,
     author = {Jiangtao Shen},
     title = {D$^2$MTOLab: A Python Platform for Data-Driven Multitask Optimization},
     year = {2025},
     url = {https://github.com/JiangtaoShen/DDMTOLab}
   }

Contact
-------

* **Author**: Jiangtao Shen
* **Email**: j.shen5@exeter.ac.uk
* **GitHub**: `JiangtaoShen/DDMTOLab <https://github.com/JiangtaoShen/DDMTOLab>`_
* **Issues**: `GitHub Issues <https://github.com/JiangtaoShen/DDMTOLab/issues>`_

License
-------

This project is licensed under the MIT License.

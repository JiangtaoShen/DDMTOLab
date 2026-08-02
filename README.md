# Data-Driven Multitask Optimization Laboratory

<p align="center">
  <img src="https://raw.githubusercontent.com/JiangtaoShen/DDMTOLab/main/docs/source/_static/logo_new.svg" alt="D²MTOLab Logo" width="530">
</p>

<p align="center">
  <a href="https://jiangtaoshen.github.io/DDMTOLab/">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation">
  </a>
  <a href="https://pypi.org/project/ddmtolab/">
    <img src="https://img.shields.io/pypi/v/ddmtolab.svg" alt="PyPI version">
  </a>
  <a href="https://github.com/JiangtaoShen/DDMTOLab/stargazers">
    <img src="https://img.shields.io/github/stars/JiangtaoShen/DDMTOLab?style=social" alt="GitHub Stars">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  </a>
  <a href="https://github.com/JiangtaoShen/DDMTOLab/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
</p>

---

## 📰 News

<!-- Rolling list: keep only the 5 most recent items, newest first. When adding a new item, delete the oldest so the list never exceeds 5. -->

- **2026-08-03** — Added SAEA-AKT, the first surrogate-assisted algorithm for expensive multitask multiobjective optimization.
- **2026-07-23** — Aligned seven surrogate-assisted single-task algorithms (TLRBF, GL-SADE, AutoSAEA, DDEA-MESS, LSADE, SHPSO, SA-COSO) with their MATLAB reference implementations; added shared `newrbe`/`newrb` RBF surrogates.
- **2026-07-19** — Verified and validated several expensive multitask optimization algorithms (MFEA-SSG, BO-LCB-CKT, BO-LCB-BCKT, MUMBO, SELF, EEI-BO / EEI-BO+, SaEF-AKT, RAMTEA, LCB-EMT).

## 📖 Overview

**D<sup>2</sup>MTOLab (Data-Driven Multitask Optimization Laboratory)** is a comprehensive Python platform for optimization research, featuring **110+ algorithms**, **180+ benchmark problems**, and powerful experiment tools for problem definition, algorithm development, and performance evaluation.

Whether you're working on expensive black-box optimization, multiobjective optimization, or complex multitask scenarios, D<sup>2</sup>MTOLab provides a flexible and extensible framework to accelerate your **research** and support real-world **applications**.

## ✨ Features

- 🚀 **Comprehensive Algorithms** - 110+ algorithms for expensive/inexpensive, single/multitask, single/multiobjective, unconstrained/constrained optimization
- 📊 **Rich Problem Suite** - 180+ benchmark problems and real-world applications
- 🤖 **Data-Driven Optimization** - Surrogate modelling for expensive optimization
- 🔧 **Flexible Framework** - Simple API and intuitive workflow for rapid prototyping
- 🔌 **Fully Extensible** - Easy to add custom algorithms and problems
- ⚡ **Parallel Computing** - Multi-core support for batch experiments
- 📈 **Powerful Analysis Tools** - Built-in visualization and statistical analysis
- 📝 **Complete Documentation** - Comprehensive [documentation](https://jiangtaoshen.github.io/DDMTOLab/) and API reference

## 🚀 Quick Start

### Installation

```bash
pip install ddmtolab
```

**Requirements:**
- Python 3.10+
- PyTorch 2.5+ (supports CPU, GPU optional for acceleration)
- NumPy 2.0+, SciPy 1.15+, scikit-learn 1.7+
- Matplotlib 3.10+, Pandas 2.3+

For detailed installation instructions, see [Installation Guide](https://jiangtaoshen.github.io/DDMTOLab/installation.html).

### Basic Usage

```python
import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Algorithms.MTSO.MTBO import MTBO

# Step 1: Define objective function
def forrester(x):
    """Forrester function: (6x-2)^2 * sin(12x-4)"""
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

# Step 2: Create optimization problem
problem = MTOP()
problem.add_task(forrester, dim=1)

# Step 3: Run optimization
results = MTBO(problem).optimize()

# Step 4: Display results
print(f"Best solution: {results.best_decs}")
print(f"Best objective: {results.best_objs}")

# Step 5: Analyze and visualize
from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
TestDataAnalyzer().run()
```

### Batch Experiment

```python
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D

if __name__ == '__main__':
    # Step 1: Create batch experiment manager
    batch_exp = BatchExperiment(
        base_path='./Data',      # Data save path
        clear_folder=True        # Clear existing data
    )

    # Step 2: Add test problems
    prob = CEC17MTSO_10D()
    batch_exp.add_problem(prob.P1, 'P1')
    batch_exp.add_problem(prob.P2, 'P2')

    # Step 3: Add algorithms with parameters
    batch_exp.add_algorithm(BO, 'BO', n_initial=20, max_nfes=100)
    batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=20, max_nfes=100)

    # Step 4: Run batch experiments
    batch_exp.run(n_runs=20, verbose=True, max_workers=8)

    # Step 5: Run data analysis
    analyzer = DataAnalyzer()
    results = analyzer.run()
```

### Optimization Process Visualization

D<sup>2</sup>MTOLab provides built-in animation tools to visualize the optimization process:

```python
from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Methods.animation_generator import create_optimization_animation

# Define problem
problem = CEC17MTSO_10D().P1()

# Run algorithms
BO(problem, n_initial=20, max_nfes=100, name='BO').optimize()
MTBO(problem, n_initial=20, max_nfes=100, name='MTBO').optimize()

# Generate animation
animation = create_optimization_animation(max_nfes=100, merge=2, title='BO vs MTBO')
```

<p align="center">
  <img src="https://raw.githubusercontent.com/JiangtaoShen/DDMTOLab/main/docs/source/_static/animation.gif"
       alt="BO and MTBO on CEC17-MTSO-10D-P1 Animation"
       width="100%">
</p>

## 🎯 Key Components

### Algorithms (110+)

| Category | Type | Algorithms |
|----------|------|------------|
| **STSO** | Inexpensive | GA, DE, PSO, SL_PSO, KLPSO, CSO, CMA_ES, IPOP_CMA_ES, sep_CMA_ES, MA_ES, xNES, OpenAI_ES, AO, GWO, EO |
| **STSO** | Expensive | BO, EEI_BO, ESAO, SHPSO, SA_COSO, TLRBF, GL_SADE, AutoSAEA, DDEA_MESS, LSADE |
| **STMO** | Inexpensive | NSGA_II, NSGA_III, NSGA_II_SDR, SPEA2, MOEA_D, MOEA_DD, MOEA_D_FRRMAB, MOEA_D_STM, RVEA, IBEA, TwoArch2, MSEA, C_TAEA, CCMO |
| **STMO** | Expensive | ParEGO, K_RVEA, DSAEA_PS, KTA2, REMO, ADSAPSO, CSEA, DISK, DRLSAEA, DirHV_EI, EDN_ARMOEA, EIM_EGO, EM_SAEA, KTS, MGSAEA, MMRAEA, MOEA_D_EGO, MultiObjectiveEGO, PCSAEA, PEA, PIEA, SAEA_DBLL, SSDE, TEA, CPS_MOEA, MCEA_D |
| **MTSO** | Inexpensive | MFEA, MFEA_II, EMEA, EBS, G_MFEA, MTEA_AD, MKTDE, MTEA_SaO, SREMTO, LCB_EMT, BLKT_DE, DTSKT, EMTO_AI, MFEA_AKT, MFEA_DGD, MFEA_VC, MTDE_ADKT, MTEA_HKTS, MTEA_PAE, MTES_KG, SSLT_DE, TNG_SNES |
| **MTSO** | Expensive | MTBO, RAMTEA, SELF, EEI_BO_plus, MUMBO, BO_LCB_CKT, BO_LCB_BCKT, MFEA_SSG, SaEF_AKT |
| **MTMO** | Inexpensive | MO_MFEA, MO_MFEA_II, MO_EMEA, MO_MTEA_SaO, MTDE_MKTA, MTEA_D_DN, EMT_ET, EMT_PD, EMT_GS, MO_MTEA_PAE, MO_SBO, MTEA_D_TSD, MTEA_DCK |
| **MTMO** | Expensive | ParEGO_KT, SAEA_AKT |

### Problems (180+)

| Category | Problem Suites |
|----------|----------------|
| **STSO** | CLASSICALSO (9 functions), CEC10_CSO (20 functions) |
| **STMO** | ZDT (6), DTLZ (9), WFG (9), UF (10), CF (10), MW (14) |
| **MTSO** | CEC17MTSO (9), CEC17MTSO_10D (9), CEC19MaTSO (many-task), CMT (9), STOP (12), ManyTask_10D (4) |
| **MTMO** | CEC17MTMO (9), CEC19MTMO (10), CEC19_MaTMO (many-task), CEC21MTMO (10), MTMO_DTLZ, MTMOInstances |
| **Real-World** | PEPVM, PINN_HPO (12), SOPM, SCP, MO_SCP, PKACP, NN_Training, TSP (6) |

### Methods

- **MTOP Class**: Flexible problem definition supporting single/multitask and single/multiobjective
- **Batch Experiments**: Parallel execution framework for large-scale experiments
- **Data Analysis**: Statistical analysis (mean, std, ranking) and visualization tools
- **Performance Metrics**: IGD, GD, IGD+, HV, DeltaP, Spacing, Spread, FR, CV
- **Animation Generator**: Optimization process visualization
- **Algorithm Utilities**: Reusable components (initialization, selection, operators)

## 🖥️ GUI (Experimental)

An experimental desktop GUI built with [DearPyGui](https://github.com/hoffstadt/DearPyGui) lives in the [`ui/`](ui/) directory. It is shipped in this repository only (not part of the PyPI package):

```bash
pip install -r ui/requirements.txt
python ui/main.py
```

## 📚 Documentation

- [Installation Guide](https://jiangtaoshen.github.io/DDMTOLab/installation.html)
- [Algorithms Reference](https://jiangtaoshen.github.io/DDMTOLab/algorithms.html)
- [Problems Reference](https://jiangtaoshen.github.io/DDMTOLab/problems.html)
- [Methods Guide](https://jiangtaoshen.github.io/DDMTOLab/methods.html)
- [Demo Scripts](https://jiangtaoshen.github.io/DDMTOLab/demos.html)
- [API Reference](https://jiangtaoshen.github.io/DDMTOLab/api.html)

## 📄 Citation

If you use D<sup>2</sup>MTOLab in your research, please cite:

```bibtex
@software{ddmtolab2025,
  author = {Jiangtao Shen},
  title = {D$^2$MTOLab: A Python Platform for Data-Driven Multitask Optimization},
  year = {2025},
  url = {https://github.com/JiangtaoShen/DDMTOLab}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

- **Author**: Jiangtao Shen
- **Email**: j.shen5@exeter.ac.uk
- **Documentation**: [https://jiangtaoshen.github.io/DDMTOLab/](https://jiangtaoshen.github.io/DDMTOLab/)
- **Issues**: [GitHub Issues](https://github.com/JiangtaoShen/DDMTOLab/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<p align="center">
  Made with ❤️ by <a href="https://github.com/JiangtaoShen">Jiangtao Shen</a>
</p>

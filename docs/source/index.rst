# DDMTOLab

<p align="center">
  <img src="docs/source/_static/logo.png" alt="DDMTOLab Logo" width="250">
</p>

<p align="center">
  <strong>Data-Driven Multitask Optimization Laboratory</strong>
</p>

<p align="center">
  <a href="https://jiangtaoshen.github.io/DDMTOLab/">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation">
  </a>
  <a href="https://pypi.org/project/ddmtolab/">
    <img src="https://img.shields.io/pypi/v/ddmtolab.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/ddmtolab/">
    <img src="https://img.shields.io/pypi/dm/ddmtolab.svg" alt="Downloads">
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

## 📖 Overview

**DDMTOLab is a comprehensive Python platform designed for data-driven multitask optimization**, featuring **60+ algorithms**, **180+ benchmark problems**, and powerful experiment tools for problem definition, algorithm development, and performance evaluation.

Whether you're working on expensive black-box optimization, multiobjective optimization, or complex multitask scenarios, DDMTOLab provides a flexible and extensible framework to accelerate your **research** and support real-world **applications**.

## ✨ Features

- 🚀 **Comprehensive Algorithms** - Expensive/inexpensive, Single/multi-task, single/multi-objective optimization algorithms
- 📊 **Rich Problem Suite** - Extensive benchmark problem suites and real-world applications
- 🤖 **Data-Driven Optimization** - Surrogate modelling for expensive optimization
- 🔧 **Flexible Framework** - Simple API and intuitive workflow for rapid prototyping
- 🔌 **Fully Extensible** - Easy to add custom algorithms and problems
- 📈 **Powerful Analysis Tools** - Built-in visualization and statistical analysis
- ⚡ **Parallel Computing** - Multi-core support for batch experiments
- 📝 **Complete Documentation** - Comprehensive [Tutorial](https://jiangtaoshen.github.io/DDMTOLab/quickstart.html) and API reference

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
def t1(x):
    """Forrester function: (6x-2)^2 * sin(12x-4)"""
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

# Step 2: Create optimization problem
problem = MTOP()
problem.add_task(t1, dim=1)

# Step 3: Run optimization
results = MTBO(problem).optimize()

# Step 4: Display results
print(results.best_decs, results.best_objs)

# Step 5: Analyze and visualize
from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
TestDataAnalyzer().run()
```

### Batch Experiments

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
    batch_exp.add_problem(prob.P3, 'P3')

    # Step 3: Add algorithms with parameters
    batch_exp.add_algorithm(BO, 'BO', n_initial=20, max_nfes=100)
    batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=20, max_nfes=100)

    # Step 4: Run batch experiments
    batch_exp.run(n_runs=20, verbose=True, max_workers=8)

    # Step 5: Configure data analyzer
    analyzer = DataAnalyzer()

    # Step 6: Run data analysis
    results = analyzer.run()
```

## 🎞️ Optimization Process

This animation demonstrates the optimization process and is generated using the built-in animation components provided by DDMTOLab.

<p align="center">
  <img src="docs/images/BO%20and%20MTBO%20on%20CEC17-MTSO-10D-P1_animation.gif"
       alt="BO and MTBO on CEC17-MTSO-10D-P1 Animation"
       width="100%">
</p>

## 🎯 Key Components

### Algorithms

**60+ state-of-the-art optimization algorithms** across four categories:

| Category | Algorithms |
|----------|-----------|
| **STSO** | GA, DE, PSO, SL-PSO, KL-PSO, CSO, CMA-ES, AO, GWO, EO, BO, EEI-BO, ESAO, SHPSO, SA-COSO, TLRBF, GL-SADE |
| **STMO** | NSGA-II, NSGA-III, NSGA-II-SDR, SPEA2, MOEA/D, MOEA/DD, FRRMAB, MOEA/D-STM, RVEA, IBEA, Two_Arch2, MSEA, C-TAEA, CCMO, MCEA/D, ParEGO, K-RVEA, DSAEA-PS |
| **MTSO** | MFEA, MFEA-II, EMEA, EBS, G-MFEA, MTEA-AD, MKTDE, MTEA-SaO, SREMTO, LCB-EMT, MTBO, RAMTEA, SELF, EEI-BO+, MUMBO, BO-LCB-CKT, BO-LCB-BCKT |
| **MTMO** | MO-MFEA, MO-MFEA-II, MO-EMEA, MO-MTEA-SaO, MTDE-MKTA, MTEA/D-DN, EMT-ET, EMT-PD |

### Problems

**180+ benchmark problems** across five categories:

| Category | Problem Suites |
|----------|---------------|
| **STSO** | Classical Functions (9), CEC10-CSO (18) |
| **STMO** | ZDT (6), DTLZ (9), WFG (9), UF (10), CF (10) |
| **MTSO** | CEC17-MTSO (9), CEC17-MTSO-10D (9), CEC19-MaTSO (6), CMT (9), STOP (12) |
| **MTMO** | CEC17-MTMO (9), CEC19-MTMO (10), CEC19-MaTMO (6), CEC21-MTMO (10), MTMO-Instance (2) |
| **Real-World** | PEPVM (1), PINN-HPO (12), SOPM (2), SCP (1), MO-SCP (2), PKACP (1) |

### Methods

- **Batch Experiments**: Parallel execution framework for large-scale experiments
- **Data Analysis**: Statistical analysis and visualization tools
- **Performance Metrics**: IGD, HV, Spacing, Spread, FR, CV, and more
- **Algorithm Components**: Reusable building blocks for rapid development

## 📚 Documentation

For detailed documentation, tutorials, and API reference, visit:

👉 **[https://jiangtaoshen.github.io/DDMTOLab/](https://jiangtaoshen.github.io/DDMTOLab/)**

- [Installation Guide](https://jiangtaoshen.github.io/DDMTOLab/installation.html)
- [Quick Start Tutorial](https://jiangtaoshen.github.io/DDMTOLab/quickstart.html)
- [API Reference](https://jiangtaoshen.github.io/DDMTOLab/api.html)
- [Problem Suites](https://jiangtaoshen.github.io/DDMTOLab/problems.html)
- [Algorithms](https://jiangtaoshen.github.io/DDMTOLab/algorithms.html)

## 📄 Citation

If you use DDMTOLab in your research, please cite:

```bibtex
@software{ddmtolab2025,
  author = {Jiangtao Shen},
  title = {DDMTOLab: A Python Platform for Data-Driven Multitask Optimization},
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

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JiangtaoShen/DDMTOLab&type=Date)](https://star-history.com/#JiangtaoShen/DDMTOLab&Date)

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/JiangtaoShen">Jiangtao Shen</a>
</p>
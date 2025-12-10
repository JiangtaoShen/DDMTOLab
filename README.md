# DDMTOLab

<p align="center">
  <img src="docs/source/_static/logo.png" alt="DDMTOLab Logo" width="300">
</p>

<p align="center">
  <strong>A Comprehensive Python Platform for Data-Driven Multitask Optimization</strong>
</p>

<p align="center">
  <a href="https://jiangtaoshen.github.io/DDMTOLab/">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation">
  </a>
  <a href="https://github.com/JiangtaoShen/DDMTOLab/stargazers">
    <img src="https://img.shields.io/github/stars/JiangtaoShen/DDMTOLab?style=social" alt="GitHub Stars">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  </a>
  <a href="https://github.com/JiangtaoShen/DDMTOLab/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
</p>

---

## 📖 Overview

**DDMTOLab** is a comprehensive Python platform designed for data-driven multitask optimization, featuring **50+ algorithms**, **100+ benchmark problems**, and powerful experiment tools for algorithm development and performance evaluation.

Whether you're working on expensive black-box optimization, multi-objective optimization, or complex multi-task scenarios, DDMTOLab provides a flexible and extensible framework to accelerate your research.

👉 **[Get Started with Our Tutorial](https://jiangtaoshen.github.io/DDMTOLab/quickstart.html)**

## ✨ Features

- 🚀 **Comprehensive Algorithms** - Single/multi-task, single/multi-objective optimization algorithms
- 📊 **Rich Problem Suite** - Extensive benchmark functions and real-world applications
- 🤖 **Data-Driven Optimization** - Surrogate-based methods for expensive problems
- 🔧 **Flexible Framework** - Simple API and intuitive workflow for rapid prototyping
- 🔌 **Fully Extensible** - Easy to add custom algorithms, problems, and metrics
- 📈 **Powerful Analysis Tools** - Built-in visualization and statistical analysis
- ⚡ **Parallel Computing** - Multi-core support for batch experiments
- 📝 **Complete Documentation** - Comprehensive guides and API reference

## 🚀 Quick Start

### Installation
```bash
pip install ddmtolab
```

Or install from source:
```bash
git clone https://github.com/JiangtaoShen/DDMTOLab.git
cd DDMTOLab
pip install -e .
```

### Basic Usage
```python
from DDMTOLab.problems import Sphere
from DDMTOLab.algorithms import GA

# Create problem
problem = Sphere(n_tasks=1, dims=[30])

# Initialize algorithm
algorithm = GA(problem=problem, n=[100], max_nfes=[10000])

# Run optimization
results = algorithm.optimize()

# Access results
print(f"Best objective: {results.best_objs[0]}")
print(f"Runtime: {results.runtime:.2f}s")
```

### Batch Experiments
```python
from Methods.batch_experiment import BatchExperiment
from Problems.MTSO.cec17_mtso import CEC17MTSO
from Algorithms.STSO.GA import GA
from Algorithms.STSO.DE import DE

# Create batch experiment
batch_exp = BatchExperiment(base_path='./Data')

# Add problems
cec17 = CEC17MTSO()
batch_exp.add_problem(cec17.P1, 'P1')
batch_exp.add_problem(cec17.P2, 'P2')

# Add algorithms
batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=10000)
batch_exp.add_algorithm(DE, 'DE', n=100, max_nfes=10000)

# Run experiments
batch_exp.run(n_runs=30, max_workers=8)
```

### Data Analysis
```python
from Methods.data_analysis import DataAnalyzer

# Create analyzer
analyzer = DataAnalyzer(data_path='./Data', save_path='./Results')

# Run complete analysis pipeline
results = analyzer.run()
```

## 🎯 Key Components

### Problems
- **Single-Task Single-Objective (STSO)**: Classic benchmark functions
- **Single-Task Multi-Objective (STMO)**: ZDT, DTLZ, WFG test suites
- **Multi-Task Single-Objective (MTSO)**: CEC17 MTSO, CEC19 MaTSO
- **Multi-Task Multi-Objective (MTMO)**: CEC17 MTMO
- **Real-World Problems (WRP)**: Engineering and industrial applications

### Algorithms
- **Single-Task Single-Objective (STSO)**: GA, DE, PSO, CSO, BO, etc.
- **Single-Task Multi-Objective (STMO)**: NSGA-II, RVEA, etc.
- **Multi-Task Single-Objective (MTSO)**: MFEA, EMEA, G-MFEA, MTBO, RA-MTEA, SELF, etc.
- **Multi-Task Multi-Objective (MTMO)**: MO-MFEA, etc.

### Utilities
- **Batch Experiments**: Parallel execution framework
- **Data Analysis**: Statistical testing and visualization
- **Performance Metrics**: IGD, HV, etc.
- **Algorithm Components**: Reusable building blocks

## 🔬 Research Applications

DDMTOLab is designed for researchers working on:

- Expensive black-box optimization
- Multi-objective optimization
- Multi-task optimization and transfer learning
- Surrogate-assisted optimization
- Algorithm benchmarking and comparison
- Real-world engineering optimization

## 📊 Example Results

<p align="center">
  <img src="docs/images/convergence_example.png" alt="Convergence Curves" width="45%">
  <img src="docs/images/pareto_example.png" alt="Pareto Fronts" width="45%">
</p>

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

## 📧 Contact

- **Author**: Jiangtao Shen
- **Email**: j.shen5@exeter.ac.uk
- **Documentation**: [https://jiangtaoshen.github.io/DDMTOLab/](https://jiangtaoshen.github.io/DDMTOLab/)
- **Issues**: [GitHub Issues](https://github.com/JiangtaoShen/DDMTOLab/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/JiangtaoShen">Jiangtao Shen</a>
</p>

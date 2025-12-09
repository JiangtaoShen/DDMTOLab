"""
Test Data Analyzer Module for Single-Run Algorithm Testing

This module provides a lightweight analysis pipeline for quick algorithm testing,
where pickle files are stored directly in the data folder with '_test.pkl' suffix.

Classes:
    TestScanResult: Dataclass for storing test scan results
    TestMetricResults: Dataclass for storing test metric results
    TestDataAnalyzer: Main class for test data analysis

Usage:
    analyzer = TestDataAnalyzer(data_path='./TestData', settings=SETTINGS)
    results = analyzer.run()

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.10.10
Version: 1.0
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import from project modules
from Methods.metrics import IGD, HV
from Methods.Algo_Methods.algo_utils import nd_sort


# =============================================================================
# Enums and Constants
# =============================================================================

class OptimizationDirection(Enum):
    """Optimization direction enumeration."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class StatisticType(Enum):
    """Statistical measure type enumeration."""
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"


# Default color palette for plots
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Default markers for plots
DEFAULT_MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '<', '>']


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestScanResult:
    """
    Result of scanning a test data directory.

    Attributes:
        algorithms: List[str]
            Sorted list of algorithm names found in the directory.
        problems: List[str]
            List of problem names (extracted from data or default).
        data_path: Path
            Path to the scanned data directory.
        file_mapping: Dict[str, Path]
            Mapping from algorithm name to pickle file path.
    """
    algorithms: List[str]
    problems: List[str]
    data_path: Path
    file_mapping: Dict[str, Path]


@dataclass
class TestMetricResults:
    """
    Container for test metric calculation results.

    Attributes:
        metric_values: Dict[str, List[np.ndarray]]
            Metric values per generation per algorithm.
            Structure: metric_values[algorithm] = List[np.ndarray] (per task)

        best_values: Dict[str, List[float]]
            Final best metric values per algorithm.
            Structure: best_values[algorithm] = List[float] (per task)

        objective_values: Dict[str, List[np.ndarray]]
            Original objective values per algorithm.
            Structure: objective_values[algorithm] = List[np.ndarray]

        runtime: Dict[str, float]
            Runtime in seconds per algorithm.

        max_nfes: Dict[str, List[int]]
            Maximum number of function evaluations per algorithm.

        metric_name: Optional[str]
            Name of the metric used.

        problems: List[str]
            List of problem names.
    """
    metric_values: Dict[str, List[np.ndarray]]
    best_values: Dict[str, List[float]]
    objective_values: Dict[str, List[np.ndarray]]
    runtime: Dict[str, float]
    max_nfes: Dict[str, List[int]]
    metric_name: Optional[str]
    problems: List[str]


@dataclass
class PlotConfig:
    """
    Configuration for plot generation.

    Attributes:
        figure_format: str
            Output figure format (e.g., 'pdf', 'png', 'svg').
        log_scale: bool
            Whether to use logarithmic scale for y-axis.
        show_pf: bool
            Whether to show true Pareto front in ND solution plots.
        show_nd: bool
            Whether to filter and show only non-dominated solutions.
        save_path: Path
            Directory path to save output figures.
        colors: List[str]
            Color palette for plotting algorithms.
        markers: List[str]
            Marker styles for plotting algorithms.
    """
    figure_format: str = 'pdf'
    log_scale: bool = False
    show_pf: bool = True
    show_nd: bool = True
    save_path: Path = Path('./TestResults')
    colors: List[str] = field(default_factory=lambda: DEFAULT_COLORS.copy())
    markers: List[str] = field(default_factory=lambda: DEFAULT_MARKERS.copy())


# =============================================================================
# Utility Functions
# =============================================================================

class DataUtils:
    """Utility class for data loading and processing operations."""

    @staticmethod
    def load_pickle(file_path: Path) -> Dict[str, Any]:
        """Load and return a Python object from a pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_reference(
            settings: Dict[str, Any],
            problem: str,
            task_identifier: Union[str, int],
            n_objectives: int
    ) -> Optional[np.ndarray]:
        """Load reference data (Pareto Front or reference point) for a specific problem and task."""
        task_name = f"T{task_identifier + 1}" if isinstance(task_identifier, int) else task_identifier

        if problem not in settings:
            return None

        problem_settings = settings[problem]

        if task_name not in problem_settings:
            return None

        ref_definition = problem_settings[task_name]

        if callable(ref_definition):
            n_points = settings.get('n_ref', 10000)
            return ref_definition(n_points, n_objectives)
        elif isinstance(ref_definition, str):
            return DataUtils._load_reference_from_file(settings, ref_definition, problem, task_name)
        elif isinstance(ref_definition, (list, tuple, np.ndarray)):
            reference = np.array(ref_definition)
            if reference.ndim == 1:
                reference = reference.reshape(1, -1)
            return reference
        else:
            return None

    @staticmethod
    def _load_reference_from_file(
            settings: Dict[str, Any],
            ref_definition: str,
            problem: str,
            task_name: str
    ) -> Optional[np.ndarray]:
        """Load reference data from file."""
        ref_path = settings.get('ref_path', './MOReference')

        if not os.path.isabs(ref_definition):
            full_path = os.path.join(ref_path, ref_definition)
        else:
            full_path = ref_definition

        try:
            if full_path.endswith('.npy'):
                return np.load(full_path)
            elif full_path.endswith('.csv'):
                return np.loadtxt(full_path, delimiter=',')
            else:
                return None
        except FileNotFoundError:
            return None
        except Exception:
            return None

    @staticmethod
    def get_metric_direction(metric_name: Optional[str]) -> OptimizationDirection:
        """Determine optimization direction based on metric type."""
        if metric_name is None or metric_name == 'IGD':
            return OptimizationDirection.MINIMIZE
        elif metric_name == 'HV':
            return OptimizationDirection.MAXIMIZE
        else:
            return OptimizationDirection.MINIMIZE


# =============================================================================
# Test Plot Generator
# =============================================================================

class TestPlotGenerator:
    """Class for generating visualization plots for test data."""

    def __init__(self, config: PlotConfig):
        """Initialize TestPlotGenerator with configuration."""
        self.config = config

    def plot_convergence_curves(
            self,
            metric_values: Dict[str, List[np.ndarray]],
            max_nfes: Dict[str, List[int]],
            algorithm_order: List[str],
            problems: List[str],
            metric_name: Optional[str] = None
    ) -> None:
        """Generate and save convergence curve plots for all algorithms and tasks."""
        save_dir = Path(self.config.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Determine number of tasks from first algorithm
        first_algo = algorithm_order[0]
        num_tasks = len(metric_values[first_algo])

        for task_idx in range(num_tasks):
            fig = self._create_convergence_figure(
                metric_values, max_nfes, algorithm_order,
                task_idx, num_tasks, problems, metric_name
            )

            if num_tasks == 1:
                problem_name = problems[0] if problems else 'Test'
                output_file = save_dir / f'{problem_name}_convergence.{self.config.figure_format}'
            else:
                problem_name = problems[task_idx] if task_idx < len(problems) else f'Task{task_idx + 1}'
                output_file = save_dir / f'{problem_name}_convergence.{self.config.figure_format}'

            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"Convergence plots saved to: {save_dir}")

    def _create_convergence_figure(
            self,
            metric_values: Dict[str, List[np.ndarray]],
            max_nfes: Dict[str, List[int]],
            algorithm_order: List[str],
            task_idx: int,
            num_tasks: int,
            problems: List[str],
            metric_name: Optional[str]
    ) -> plt.Figure:
        """Create a single convergence curve figure."""
        fig, ax = plt.subplots(figsize=(6, 4))

        for idx, algo in enumerate(algorithm_order):
            curve = np.array(metric_values[algo][task_idx]).ravel()

            if len(curve) == 0:
                continue

            nfes = max_nfes[algo][task_idx] if task_idx < len(max_nfes[algo]) else len(curve)
            x = np.linspace(0, nfes, len(curve))
            marker_interval = max(1, len(curve) // 10)

            ax.plot(
                x, curve, label=algo,
                color=self.config.colors[idx % len(self.config.colors)],
                marker=self.config.markers[idx % len(self.config.markers)],
                markevery=marker_interval,
                markersize=5, linewidth=1.5, linestyle='-', alpha=0.8
            )

        if self.config.log_scale:
            ax.set_yscale('log')

        # Apply scientific notation for large values
        self._apply_scientific_notation(ax)

        y_label = metric_name if metric_name is not None else 'Objective Value'
        ax.set_xlabel('NFEs', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)

        if num_tasks == 1:
            title = problems[0] if problems else 'Test Problem'
        else:
            title = problems[task_idx] if task_idx < len(problems) else f'Task {task_idx + 1}'
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='-')

        fig.tight_layout()
        return fig

    def _apply_scientific_notation(self, ax: plt.Axes) -> None:
        """Apply scientific notation to axes if values exceed threshold."""
        threshold = 1000
        xmax = ax.get_xlim()[1]
        ymax = ax.get_ylim()[1]

        for axis_name, axis, lim in [('x', ax.xaxis, xmax), ('y', ax.yaxis, ymax)]:
            if lim > threshold:
                formatter = plt.matplotlib.ticker.ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((0, 0))
                axis.set_major_formatter(formatter)
                ax.ticklabel_format(style='sci', axis=axis_name, scilimits=(0, 0))

    def plot_runtime(
            self,
            runtime: Dict[str, float],
            algorithm_order: List[str]
    ) -> None:
        """Generate and save a bar plot showing runtime comparison."""
        save_dir = Path(self.config.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))

        x = np.arange(len(algorithm_order))
        runtimes = [runtime[algo] for algo in algorithm_order]
        colors = [self.config.colors[i % len(self.config.colors)] for i in range(len(algorithm_order))]

        bars = ax.bar(x, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, runtimes):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Runtime (s)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithm_order, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='-')

        fig.tight_layout()

        output_file = save_dir / f'runtime_comparison.{self.config.figure_format}'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Runtime plot saved to: {output_file}")

    def plot_nd_solutions(
            self,
            objective_values: Dict[str, List[np.ndarray]],
            algorithm_order: List[str],
            problems: List[str],
            settings: Optional[Dict[str, Any]] = None
    ) -> None:
        """Generate and save non-dominated solution plots."""
        nd_folder = Path(self.config.save_path) / 'ND_Solutions'
        nd_folder.mkdir(parents=True, exist_ok=True)

        first_algo = algorithm_order[0]
        n_tasks = len(objective_values[first_algo])

        for algo in algorithm_order:
            for task_idx in range(n_tasks):
                objectives = objective_values[algo][task_idx]

                if objectives is None or objectives.shape[0] == 0:
                    continue

                n_objectives = objectives.shape[1]

                if n_objectives <= 1:
                    continue

                # Filter non-dominated solutions if requested
                if self.config.show_nd:
                    front_no, _ = nd_sort(objectives, objectives.shape[0])
                    nd_solutions = objectives[front_no == 1]
                else:
                    nd_solutions = objectives

                # Load true Pareto front if requested
                true_pf = None
                if self.config.show_pf and settings is not None:
                    prob = problems[task_idx] if task_idx < len(problems) else f'P{task_idx + 1}'
                    true_pf = DataUtils.load_reference(settings, prob, task_idx, n_objectives)

                # Create plot
                fig = self._create_nd_plot(nd_solutions, true_pf, n_objectives, n_tasks,
                                           problems, task_idx, algo)

                # Save figure
                prob_name = problems[task_idx] if task_idx < len(problems) else f'Task{task_idx + 1}'
                filename = f'{prob_name}-{algo}.{self.config.figure_format}'
                fig.savefig(nd_folder / filename, dpi=300)
                plt.close(fig)

        print(f"ND solution plots saved to: {nd_folder}")

    def _create_nd_plot(
            self,
            nd_solutions: np.ndarray,
            true_pf: Optional[np.ndarray],
            n_objectives: int,
            n_tasks: int,
            problems: List[str],
            task_idx: int,
            algo: str
    ) -> plt.Figure:
        """Create a non-dominated solution plot."""
        fig = plt.figure(figsize=(5, 4))

        if n_objectives == 2:
            ax = fig.add_subplot(111)

            if true_pf is not None and true_pf.shape[1] == 2:
                sort_idx = np.argsort(true_pf[:, 0])
                sorted_pf = true_pf[sort_idx]
                ax.plot(sorted_pf[:, 0], sorted_pf[:, 1],
                        'k-', linewidth=2, label='True PF', alpha=0.3, zorder=1)

            ax.scatter(nd_solutions[:, 0], nd_solutions[:, 1],
                       c='dodgerblue', s=60, alpha=0.8, edgecolors='black',
                       linewidth=0.8, label='ND Solutions', zorder=2)

            ax.set_xlabel('$f_1$', fontsize=12)
            ax.set_ylabel('$f_2$', fontsize=12)
            ax.grid(True, alpha=0.2, linestyle='-')
            ax.legend(loc='best', fontsize=9)

        elif n_objectives == 3:
            ax = fig.add_subplot(111, projection='3d')

            if true_pf is not None and true_pf.shape[1] == 3:
                ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2],
                           c='gray', s=2, alpha=0.1, label='True PF', zorder=1)

            ax.scatter(nd_solutions[:, 0], nd_solutions[:, 1], nd_solutions[:, 2],
                       c='dodgerblue', s=60, alpha=0.8, edgecolors='black',
                       linewidth=0.8, label='ND Solutions', zorder=2)

            ax.set_xlabel('$f_1$', fontsize=10)
            ax.set_ylabel('$f_2$', fontsize=10)
            ax.set_zlabel('$f_3$', fontsize=10)
            ax.view_init(elev=20, azim=60)

        else:
            # Parallel coordinates for many-objective
            ax = fig.add_subplot(111)

            for i in range(nd_solutions.shape[0]):
                ax.plot(range(n_objectives), nd_solutions[i, :],
                        'b-', alpha=0.3, linewidth=0.8)

            ax.set_xlabel('Objective Index', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_xticks(range(n_objectives))
            ax.set_xticklabels([rf'$f_{{{i + 1}}}$' for i in range(n_objectives)])
            ax.grid(True, alpha=0.3, linestyle='--')

        prob_name = problems[task_idx] if task_idx < len(problems) else f'Task{task_idx + 1}'
        title = f'{prob_name} - {algo}'
        plt.title(title, fontsize=11)
        plt.tight_layout()

        return fig


# =============================================================================
# Test Table Generator
# =============================================================================

class TestTableGenerator:
    """Class for generating LaTeX tables for test data."""

    def __init__(self, save_path: Path = Path('./TestResults')):
        """Initialize TestTableGenerator."""
        self.save_path = save_path

    def generate_latex_table(
            self,
            best_values: Dict[str, List[float]],
            runtime: Dict[str, float],
            algorithm_order: List[str],
            problems: List[str],
            metric_name: Optional[str] = None
    ) -> str:
        """
        Generate a LaTeX table comparing algorithm performance.

        Parameters:
            best_values: Dict[str, List[float]]
                Final metric values per algorithm per task.
            runtime: Dict[str, float]
                Runtime per algorithm.
            algorithm_order: List[str]
                Algorithm display order.
            problems: List[str]
                Problem/task names.
            metric_name: Optional[str]
                Metric name for table caption.

        Returns:
            str: LaTeX table string.
        """
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        num_tasks = len(best_values[algorithm_order[0]])
        direction = DataUtils.get_metric_direction(metric_name)

        # Build table data
        rows = []
        for task_idx in range(num_tasks):
            prob_name = problems[task_idx] if task_idx < len(problems) else f'Task {task_idx + 1}'
            row = {'Problem': prob_name}

            values = {}
            for algo in algorithm_order:
                val = best_values[algo][task_idx]
                values[algo] = val
                row[algo] = f"{val:.4e}".replace('e-', 'e$-$') if not np.isnan(val) else 'N/A'

            # Find best value
            valid_values = {k: v for k, v in values.items() if not np.isnan(v)}
            if valid_values:
                if direction == OptimizationDirection.MINIMIZE:
                    best_algo = min(valid_values, key=valid_values.get)
                else:
                    best_algo = max(valid_values, key=valid_values.get)
                row['_best'] = best_algo
            else:
                row['_best'] = None

            rows.append(row)

        # Add runtime row
        runtime_row = {'Problem': 'Runtime (s)'}
        for algo in algorithm_order:
            runtime_row[algo] = f"{runtime[algo]:.2f}"
        runtime_row['_best'] = min(runtime, key=runtime.get)
        rows.append(runtime_row)

        # Build LaTeX string
        num_cols = len(algorithm_order) + 1
        col_format = '|'.join(['c'] * num_cols)
        col_format = '|' + col_format + '|'

        metric_str = metric_name if metric_name else 'Objective Value'
        latex_str = "\\begin{table}[htbp]\n"
        latex_str += "\\renewcommand{\\arraystretch}{1.2}\n"
        latex_str += "\\centering\n"
        latex_str += f"\\caption{{Test Results Comparison ({metric_str})}}\n"
        latex_str += "\\label{tab:test_results}\n"
        latex_str += f"\\begin{{tabular}}{{{col_format}}}\n"
        latex_str += "\\hline\n"

        # Header row
        header = "Problem & " + " & ".join(algorithm_order) + " \\\\\n"
        latex_str += header
        latex_str += "\\hline\n"

        # Data rows
        for row in rows:
            row_str = row['Problem']
            best_algo = row.get('_best')

            for algo in algorithm_order:
                cell = row[algo]
                if algo == best_algo:
                    cell = f"\\textbf{{{cell}}}"
                row_str += f" & {cell}"

            row_str += " \\\\\n"
            latex_str += row_str

        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\end{table}\n"

        # Save to file
        output_file = save_dir / 'test_results_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_file}")

        return latex_str

    def generate_summary_table(
            self,
            metric_values: Dict[str, List[np.ndarray]],
            algorithm_order: List[str],
            problems: List[str],
            metric_name: Optional[str] = None
    ) -> str:
        """
        Generate a summary LaTeX table with convergence statistics.

        Parameters:
            metric_values: Dict[str, List[np.ndarray]]
                Metric values per generation.
            algorithm_order: List[str]
                Algorithm display order.
            problems: List[str]
                Problem/task names.
            metric_name: Optional[str]
                Metric name.

        Returns:
            str: LaTeX table string.
        """
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        num_tasks = len(metric_values[algorithm_order[0]])

        latex_str = "\\begin{table}[htbp]\n"
        latex_str += "\\renewcommand{\\arraystretch}{1.2}\n"
        latex_str += "\\centering\n"
        latex_str += "\\caption{Convergence Summary}\n"
        latex_str += "\\label{tab:convergence_summary}\n"
        latex_str += "\\begin{tabular}{|c|c|c|c|c|}\n"
        latex_str += "\\hline\n"
        latex_str += "Algorithm & Problem & Initial & Final & Improvement \\\\\n"
        latex_str += "\\hline\n"

        for algo in algorithm_order:
            for task_idx in range(num_tasks):
                prob_name = problems[task_idx] if task_idx < len(problems) else f'Task {task_idx + 1}'
                curve = np.array(metric_values[algo][task_idx]).ravel()

                if len(curve) > 0:
                    initial = curve[0]
                    final = curve[-1]
                    improvement = ((initial - final) / initial * 100) if initial != 0 else 0

                    initial_str = f"{initial:.4e}".replace('e-', 'e$-$')
                    final_str = f"{final:.4e}".replace('e-', 'e$-$')
                    imp_str = f"{improvement:.1f}\\%"

                    latex_str += f"{algo} & {prob_name} & {initial_str} & {final_str} & {imp_str} \\\\\n"

            latex_str += "\\hline\n"

        latex_str += "\\end{tabular}\n"
        latex_str += "\\end{table}\n"

        output_file = save_dir / 'convergence_summary_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"Convergence summary table saved to: {output_file}")

        return latex_str


# =============================================================================
# Main Test Data Analyzer Class
# =============================================================================

class TestDataAnalyzer:
    """
    Main class for analyzing single-run test data.

    This class handles pickle files stored directly in the data folder
    with '_test.pkl' suffix (e.g., ga_test.pkl, de_test.pkl).

    Attributes:
        data_path: Path
            Path to the data directory containing test pickle files.
        settings: Optional[Dict[str, Any]]
            Problem settings including reference definitions.
        algorithm_order: Optional[List[str]]
            Custom ordering of algorithms for display.
        plot_config: PlotConfig
            Configuration for plot generation.
    """

    def __init__(
            self,
            data_path: Union[str, Path] = './TestData',
            settings: Optional[Dict[str, Any]] = None,
            algorithm_order: Optional[List[str]] = None,
            save_path: Union[str, Path] = './TestResults',
            figure_format: str = 'pdf',
            log_scale: bool = False,
            show_pf: bool = True,
            show_nd: bool = True,
            best_so_far: bool = True,
            clear_results: bool = True,
            file_suffix: str = '_test.pkl'
    ):
        """
        Initialize TestDataAnalyzer.

        Parameters:
            data_path: Union[str, Path]
                Path to data directory containing test pickle files.
                Default: './TestData'

            settings: Optional[Dict[str, Any]]
                Problem settings dictionary for multi-objective metrics.
                Default: None (single-objective mode)

            algorithm_order: Optional[List[str]]
                Custom ordering of algorithms for display.
                Default: None (alphabetical order)

            save_path: Union[str, Path]
                Directory path to save all output files.
                Default: './TestResults'

            figure_format: str
                Output figure format: 'pdf', 'png', 'svg', etc.
                Default: 'pdf'

            log_scale: bool
                Whether to use logarithmic scale for convergence plot y-axis.
                Default: False

            show_pf: bool
                Whether to show true Pareto front in ND solution plots.
                Default: True

            show_nd: bool
                Whether to filter and show only non-dominated solutions.
                Default: True

            best_so_far: bool
                Whether to use best-so-far metric values.
                Default: True

            clear_results: bool
                Whether to clear existing results folder before analysis.
                Default: True

            file_suffix: str
                Suffix pattern for test files.
                Default: '_test.pkl'
        """
        self.data_path = Path(data_path)
        self.settings = settings
        self.algorithm_order = algorithm_order
        self.best_so_far = best_so_far
        self.clear_results = clear_results
        self.file_suffix = file_suffix

        self.plot_config = PlotConfig(
            figure_format=figure_format,
            log_scale=log_scale,
            show_pf=show_pf,
            show_nd=show_nd,
            save_path=Path(save_path)
        )

        self.table_save_path = Path(save_path)

        # Internal state
        self._scan_result: Optional[TestScanResult] = None
        self._metric_results: Optional[TestMetricResults] = None

    def scan_data(self) -> TestScanResult:
        """
        Scan the data directory to detect test pickle files.

        Returns:
            TestScanResult
                Dataclass containing:
                    - algorithms: List[str] - Algorithm names extracted from filenames
                    - problems: List[str] - Problem names
                    - data_path: Path - Path to scanned directory
                    - file_mapping: Dict[str, Path] - Algorithm to file path mapping

        Raises:
            FileNotFoundError: If data_path does not exist.
            ValueError: If no test pickle files found.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        # Find all test pickle files
        file_mapping = {}
        for pkl_file in self.data_path.glob(f'*{self.file_suffix}'):
            # Extract algorithm name from filename
            # e.g., ga_test.pkl -> ga, nsga2_test.pkl -> nsga2
            algo_name = pkl_file.stem.replace(self.file_suffix.replace('.pkl', ''), '')
            file_mapping[algo_name] = pkl_file

        if not file_mapping:
            raise ValueError(f"No test pickle files found in {self.data_path} with suffix '{self.file_suffix}'")

        algorithms = sorted(file_mapping.keys())

        # Try to extract problem names from first file
        first_file = file_mapping[algorithms[0]]
        data = DataUtils.load_pickle(first_file)
        n_tasks = len(data['all_objs'])

        # Generate problem names
        if self.settings and 'problems' in self.settings:
            problems = self.settings['problems'][:n_tasks]
        else:
            problems = [f'Task{i + 1}' for i in range(n_tasks)]

        print(f"Found {len(algorithms)} algorithms: {algorithms}")
        print(f"Found {n_tasks} tasks/problems: {problems}")
        print(f"Files: {list(file_mapping.values())}")

        self._scan_result = TestScanResult(
            algorithms=algorithms,
            problems=problems,
            data_path=self.data_path,
            file_mapping=file_mapping
        )

        return self._scan_result

    def calculate_metrics(self) -> TestMetricResults:
        """
        Calculate metric values for all algorithms.

        Returns:
            TestMetricResults
                Dataclass containing all computed metrics.

        Raises:
            RuntimeError: If scan_data() has not been called.
        """
        if self._scan_result is None:
            self.scan_data()

        scan = self._scan_result
        algo_order = self.algorithm_order if self.algorithm_order else scan.algorithms
        metric_name = self.settings.get('metric') if self.settings else None

        # Initialize storage dictionaries
        all_values = {}
        all_best_values = {}
        all_objective_values = {}
        all_runtime = {}
        all_max_nfes = {}

        for algo in algo_order:
            pkl_path = scan.file_mapping[algo]
            data = DataUtils.load_pickle(pkl_path)

            metric_values, metric_values_best_so_far = self._calculate_single_algorithm_metrics(data)

            selected = metric_values_best_so_far if self.best_so_far else metric_values
            all_values[algo] = selected

            # Extract final values
            last_vals = [
                np.asarray(task_arr).ravel()[-1] if len(task_arr) > 0 else np.nan
                for task_arr in selected
            ]
            all_best_values[algo] = last_vals

            # Extract final objective values
            last_objs = [data['all_objs'][t][-1] for t in range(len(data['all_objs']))]
            all_objective_values[algo] = last_objs

            all_runtime[algo] = data['runtime']
            all_max_nfes[algo] = data['max_nfes']

            print(f"  {algo}: {len(data['all_objs'])} tasks, runtime={data['runtime']:.2f}s")

        self._metric_results = TestMetricResults(
            metric_values=all_values,
            best_values=all_best_values,
            objective_values=all_objective_values,
            runtime=all_runtime,
            max_nfes=all_max_nfes,
            metric_name=metric_name,
            problems=scan.problems
        )

        return self._metric_results

    def _calculate_single_algorithm_metrics(
            self,
            data: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculate metric values for a single algorithm's data.

        Parameters:
            data: Dict[str, Any]
                Loaded pickle data.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]
                Tuple of (metric_values, metric_values_best_so_far).
        """
        all_objs = data['all_objs']
        n_tasks = len(all_objs)

        metric_values = []
        metric_values_best_so_far = []

        for t in range(n_tasks):
            n_gens = len(all_objs[t])
            task_values = np.zeros(n_gens)
            task_best_so_far = np.zeros(n_gens)

            best_so_far = None

            for gen in range(n_gens):
                objs_gen = all_objs[t][gen]
                M = objs_gen.shape[1]

                if M == 1:
                    # Single-objective
                    metric_value = np.min(objs_gen[:, 0])
                    sign = -1
                else:
                    # Multi-objective
                    if self.settings is None:
                        # Default to simple dominated hypervolume approximation
                        metric_value = np.min(np.sum(objs_gen, axis=1))
                        sign = -1
                    else:
                        metric_name = self.settings.get('metric', 'IGD')
                        prob = self._scan_result.problems[t] if t < len(self._scan_result.problems) else f'P{t + 1}'
                        reference = DataUtils.load_reference(self.settings, prob, t, M)

                        if metric_name == 'IGD':
                            metric_instance = IGD()
                            metric_value = metric_instance.calculate(objs_gen, reference)
                            sign = metric_instance.sign
                        elif metric_name == 'HV':
                            metric_instance = HV()
                            metric_value = metric_instance.calculate(objs_gen, reference)
                            sign = metric_instance.sign
                        else:
                            metric_value = np.min(np.sum(objs_gen, axis=1))
                            sign = -1

                task_values[gen] = metric_value

                if best_so_far is None:
                    best_so_far = metric_value
                else:
                    if sign == -1:
                        best_so_far = min(best_so_far, metric_value)
                    else:
                        best_so_far = max(best_so_far, metric_value)

                task_best_so_far[gen] = best_so_far

            metric_values.append(task_values)
            metric_values_best_so_far.append(task_best_so_far)

        return metric_values, metric_values_best_so_far

    def generate_convergence_plots(self) -> None:
        """Generate and save convergence curve plots."""
        if self._metric_results is None:
            self.calculate_metrics()

        algo_order = self.algorithm_order if self.algorithm_order else self._scan_result.algorithms

        plot_gen = TestPlotGenerator(self.plot_config)
        plot_gen.plot_convergence_curves(
            self._metric_results.metric_values,
            self._metric_results.max_nfes,
            algo_order,
            self._metric_results.problems,
            self._metric_results.metric_name
        )

    def generate_runtime_plots(self) -> None:
        """Generate and save runtime comparison bar plots."""
        if self._metric_results is None:
            self.calculate_metrics()

        algo_order = self.algorithm_order if self.algorithm_order else self._scan_result.algorithms

        plot_gen = TestPlotGenerator(self.plot_config)
        plot_gen.plot_runtime(self._metric_results.runtime, algo_order)

    def generate_nd_solution_plots(self) -> None:
        """Generate and save non-dominated solution plots."""
        if self._metric_results is None:
            self.calculate_metrics()

        algo_order = self.algorithm_order if self.algorithm_order else self._scan_result.algorithms

        plot_gen = TestPlotGenerator(self.plot_config)
        plot_gen.plot_nd_solutions(
            self._metric_results.objective_values,
            algo_order,
            self._metric_results.problems,
            self.settings
        )

    def generate_latex_tables(self) -> str:
        """
        Generate LaTeX comparison tables.

        Returns:
            str: LaTeX table string.
        """
        if self._metric_results is None:
            self.calculate_metrics()

        algo_order = self.algorithm_order if self.algorithm_order else self._scan_result.algorithms

        table_gen = TestTableGenerator(self.table_save_path)

        # Generate main results table
        main_table = table_gen.generate_latex_table(
            self._metric_results.best_values,
            self._metric_results.runtime,
            algo_order,
            self._metric_results.problems,
            self._metric_results.metric_name
        )

        # Generate convergence summary table
        summary_table = table_gen.generate_summary_table(
            self._metric_results.metric_values,
            algo_order,
            self._metric_results.problems,
            self._metric_results.metric_name
        )

        return main_table

    def run(self) -> TestMetricResults:
        """
        Execute the complete test analysis pipeline.

        Returns:
            TestMetricResults
                Complete metric results from the analysis.
        """
        print("=" * 60)
        print('🚀 Starting Test Data Analysis Pipeline! 🚀')
        print("=" * 60)

        # Step 0: Clear results folder if requested
        if self.clear_results:
            results_path = self.plot_config.save_path
            if results_path.exists():
                print(f'\n♻️  Clearing existing results folder: {results_path}')
                shutil.rmtree(results_path)
            results_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Scan data
        print('\n🔍 Scanning test data directory...')
        self.scan_data()

        # Step 2: Calculate metrics
        print('\n📊 Calculating metric values...')
        self.calculate_metrics()

        # Step 3: Generate LaTeX tables
        print('\n📋 Generating LaTeX tables...')
        self.generate_latex_tables()

        # Step 4: Plot convergence curves
        print('\n📈 Plotting convergence curves...')
        self.generate_convergence_plots()

        # Step 5: Plot runtime
        print('\n⏱️  Plotting runtime comparison...')
        self.generate_runtime_plots()

        # Step 6: Plot ND solutions (if multi-objective)
        first_algo = self._scan_result.algorithms[0]
        first_objs = self._metric_results.objective_values[first_algo][0]
        if first_objs.shape[1] > 1:
            print('\n🎯 Plotting non-dominated solutions...')
            self.generate_nd_solution_plots()

        print("\n" + "=" * 60)
        print('🎉 Test Data Analysis Completed! 🎉')
        print("=" * 60)

        return self._metric_results


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == '__main__':
    """
    Usage Examples for TestDataAnalyzer
    ====================================

    Example 1: Quick Start
    ----------------------
    Analyze test files in ./TestData folder:

        from test_data_analyzer import TestDataAnalyzer

        analyzer = TestDataAnalyzer(data_path='./TestData')
        results = analyzer.run()


    Example 2: With Custom Settings
    -------------------------------
    Multi-objective analysis with IGD metric:

        SETTINGS = {
            'metric': 'IGD',
            'ref_path': './MOReference',
            'problems': ['ZDT1', 'ZDT2'],
            'ZDT1': {'T1': 'ZDT1_ref.npy'},
            'ZDT2': {'T1': 'ZDT2_ref.npy'}
        }

        analyzer = TestDataAnalyzer(
            data_path='./TestData',
            settings=SETTINGS,
            save_path='./TestResults',
            figure_format='png'
        )
        results = analyzer.run()


    Example 3: Custom Algorithm Order
    ---------------------------------
        analyzer = TestDataAnalyzer(
            data_path='./TestData',
            algorithm_order=['de', 'ga', 'ga1'],  # Custom display order
            figure_format='pdf'
        )
        results = analyzer.run()


    Example 4: Step-by-Step Analysis
    --------------------------------
        analyzer = TestDataAnalyzer(data_path='./TestData')

        # Scan files
        scan_result = analyzer.scan_data()
        print(f"Found: {scan_result.algorithms}")

        # Calculate metrics
        results = analyzer.calculate_metrics()

        # Generate specific outputs
        analyzer.generate_latex_tables()
        analyzer.generate_convergence_plots()


    Expected File Structure
    -----------------------
        ./TestData/
        ├── ga_test.pkl
        ├── ga1_test.pkl
        ├── de_test.pkl
        └── ...

    Each .pkl file should contain:
        - 'all_objs': List[List[np.ndarray]] - Objectives per task per generation
        - 'runtime': float - Total runtime in seconds
        - 'max_nfes': List[int] - Max function evaluations per task
    """

    # Demo run
    print("TestDataAnalyzer - Demo")
    print("=" * 50)

    analyzer = TestDataAnalyzer(
        data_path='./TestData',
        save_path='./TestResults',
        figure_format='pdf',
        clear_results=True
    )

    # Run complete analysis
    try:
        results = analyzer.run()
    except (FileNotFoundError, ValueError) as e:
        print(f"Demo skipped: {e}")
        print("Create test files in ./TestData/ with '_test.pkl' suffix to run analysis.")
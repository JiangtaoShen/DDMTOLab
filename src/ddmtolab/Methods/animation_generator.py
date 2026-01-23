"""
Optimization Animation Generator

This module generates animations for optimization processes from .pkl result files.
Supports both single-objective and multi-objective optimization visualization.
Added multiple merge modes for comparing algorithms.

Author: Assistant
Date: 2026-01-23
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import warnings
import glob

# Try to import nd_sort from external package
try:
    from Methods.Algo_Methods.algo_utils import nd_sort

    ND_SORT_AVAILABLE = True
except ImportError:
    ND_SORT_AVAILABLE = False


    # Fallback implementation if external nd_sort is not available
    def nd_sort(objs, *args):
        """Fallback non-dominated sorting implementation."""
        from typing import Tuple

        pop_obj = objs.copy()
        n, m = pop_obj.shape

        # Parse arguments
        if len(args) == 1:
            n_sort = args[0]
        elif len(args) == 2:
            pop_con = args[0]
            n_sort = args[1]

            if pop_con is not None:
                infeasible = np.any(pop_con > 0, axis=1)
                if np.any(infeasible):
                    max_obj = np.max(pop_obj, axis=0)
                    constraint_violation = np.sum(np.maximum(0, pop_con[infeasible, :]), axis=1)
                    pop_obj[infeasible, :] = max_obj + constraint_violation[:, np.newaxis]
        else:
            raise ValueError("Invalid number of arguments")

        unique_obj, inverse_indices = np.unique(pop_obj, axis=0, return_inverse=True)
        table = np.bincount(inverse_indices, minlength=len(unique_obj))

        n_unique, m = unique_obj.shape
        front_no = np.full(n_unique, np.inf)
        max_fno = 0

        while np.sum(table[front_no < np.inf]) < min(n_sort, len(inverse_indices)):
            max_fno += 1

            for i in range(n_unique):
                if front_no[i] == np.inf:
                    dominated = False

                    for j in range(i - 1, -1, -1):
                        if front_no[j] == max_fno:
                            m_idx = 1
                            while m_idx < m and unique_obj[i, m_idx] >= unique_obj[j, m_idx]:
                                m_idx += 1

                            dominated = (m_idx == m)

                            if dominated or m == 2:
                                break

                    if not dominated:
                        front_no[i] = max_fno

        front_no = front_no[inverse_indices]
        return front_no, max_fno

# Try to import FFMpegWriter
try:
    from matplotlib.animation import FFMpegWriter

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

warnings.filterwarnings('ignore')


class OptimizationAnimator:
    """
    Generate animations for optimization processes.

    For single-objective tasks:
        - Decision space convergence (scatter plot)
        - Convergence curve (best objective value over NFEs)

    For multi-objective tasks:
        - Decision space convergence (scatter plot)
        - Objective space convergence (Pareto front evolution)

    Supports 4 merge modes for comparing multiple algorithms:
        - merge=0: No merge, individual animations
        - merge=1: Full merge, all algorithms in same plots
        - merge=2: Decision space separated, objective space merged
        - merge=3: Both decision and objective spaces separated
    """

    def __init__(self, pkl_path, output_path=None, fps=10, dpi=100, merge=0, pkl_paths=None, title=None,
                 algorithm_order=None, max_nfes=100, logscale=False):
        """
        Initialize the animator.

        Parameters
        ----------
        pkl_path : str or list
            Path to the .pkl results file, or list of paths when merge>0
        output_path : str, optional
            Path for output animation file. If None, uses same directory as pkl_path
        fps : int, optional
            Frames per second for animation (default: 10)
        dpi : int, optional
            DPI for the output animation (default: 100)
        merge : int, optional
            Merge mode (default: 0)
            - 0: No merge, individual animations for each file
            - 1: Full merge, all algorithms in same plots (left & right)
            - 2: Partial merge, separate decision space (left), merged objective space (right)
            - 3: Separate plots, decision and objective spaces both separated
        pkl_paths : list, optional
            List of .pkl file paths when merge>0 (deprecated, use pkl_path as list)
        title : str, optional
            Custom title for the animation. If None, uses filename(s) or 'test' for merge mode
        algorithm_order : list, optional
            List of algorithm names (file stems) specifying the display order.
            If None, uses the order from pkl_path/pkl_paths.
            Example: ['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT']
        max_nfes : int or list, optional
            Maximum number of function evaluations (NFEs) for each task.
            Can be a scalar (same for all tasks) or a list (one per task).
            Default: 100
        logscale : bool, optional
            Use logarithmic scale for y-axis in single-objective convergence curves (default: False)
        """
        self.fps = fps
        self.dpi = dpi
        self.merge = merge
        self.title = title
        self.logscale = logscale

        if merge > 0:
            # Merge mode: load multiple files
            if pkl_paths is not None:
                self.pkl_paths = [Path(p) for p in pkl_paths]
            elif isinstance(pkl_path, list):
                self.pkl_paths = [Path(p) for p in pkl_path]
            else:
                raise ValueError("When merge>0, pkl_path must be a list or pkl_paths must be provided")

            # Reorder pkl_paths based on algorithm_order if provided
            if algorithm_order is not None:
                self.pkl_paths = self._reorder_paths(self.pkl_paths, algorithm_order)

            # Load all data
            self.all_data = []
            self.algorithm_names = []

            for pkl_file in self.pkl_paths:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                self.all_data.append(data)
                self.algorithm_names.append(pkl_file.stem)

            self.n_algorithms = len(self.all_data)

            # Verify compatibility
            self._verify_merge_compatibility()

            # Use first file's structure as reference
            self.data = self.all_data[0]
            self.all_decs = [data['all_decs'] for data in self.all_data]
            self.all_objs = [data['all_objs'] for data in self.all_data]
            self.bounds = self.data.get('bounds', None)

            # Determine output path
            if output_path is None:
                output_dir = self.pkl_paths[0].parent
                if self.title:
                    self.output_path = output_dir / f"{self.title}_animation.gif"
                else:
                    self.output_path = output_dir / "test_animation.gif"
            else:
                self.output_path = Path(output_path)

            # Analyze tasks (use first algorithm as reference)
            self.n_tasks = len(self.all_decs[0])
            self.n_objs_per_task = [self.all_objs[0][i][0].shape[1] for i in range(self.n_tasks)]
            self.dims_per_task = [self.all_decs[0][i][0].shape[1] for i in range(self.n_tasks)]

            # Get generations for each algorithm and task
            self.n_generations_per_algo = []
            for algo_decs in self.all_decs:
                self.n_generations_per_algo.append([len(algo_decs[i]) for i in range(self.n_tasks)])

            # Maximum generations across all algorithms
            self.max_gen = max([max(gens) for gens in self.n_generations_per_algo])

        else:
            # Single file mode (merge=0)
            self.pkl_path = Path(pkl_path) if not isinstance(pkl_path, list) else Path(pkl_path[0])

            # Load data
            with open(self.pkl_path, 'rb') as f:
                self.data = pickle.load(f)

            # Extract data
            self.all_decs = self.data['all_decs']
            self.all_objs = self.data['all_objs']
            self.bounds = self.data.get('bounds', None)

            # Determine output path
            if output_path is None:
                self.output_path = self.pkl_path.parent / f"{self.pkl_path.stem}_animation.gif"
            else:
                self.output_path = Path(output_path)

            # Analyze tasks
            self.n_tasks = len(self.all_decs)
            self.n_objs_per_task = [self.all_objs[i][0].shape[1] for i in range(self.n_tasks)]
            self.dims_per_task = [self.all_decs[i][0].shape[1] for i in range(self.n_tasks)]
            self.n_generations = [len(self.all_decs[i]) for i in range(self.n_tasks)]

            # Determine max generations for synchronization
            self.max_gen = max(self.n_generations)

        # Process max_nfes parameter
        if isinstance(max_nfes, (list, tuple, np.ndarray)):
            # List of NFEs per task
            if len(max_nfes) != self.n_tasks:
                raise ValueError(f"max_nfes list length ({len(max_nfes)}) must match number of tasks ({self.n_tasks})")
            self.max_nfes_per_task = list(max_nfes)
        else:
            # Scalar: same NFEs for all tasks
            self.max_nfes_per_task = [max_nfes] * self.n_tasks

    def _reorder_paths(self, pkl_paths, algorithm_order):
        """
        Reorder pkl_paths based on algorithm_order.

        Parameters
        ----------
        pkl_paths : list of Path
            Original list of paths
        algorithm_order : list of str
            Desired order of algorithm names (file stems)

        Returns
        -------
        list of Path
            Reordered paths
        """
        # Create a mapping from algorithm name to path
        name_to_path = {path.stem: path for path in pkl_paths}

        # Check if all names in algorithm_order exist
        missing_names = [name for name in algorithm_order if name not in name_to_path]
        if missing_names:
            raise ValueError(f"Algorithm names not found in pkl_paths: {missing_names}")

        # Check if there are extra files not in algorithm_order
        extra_names = [name for name in name_to_path.keys() if name not in algorithm_order]
        if extra_names:
            print(f"Warning: The following algorithms are not in algorithm_order and will be ignored: {extra_names}")

        # Reorder paths
        reordered_paths = [name_to_path[name] for name in algorithm_order if name in name_to_path]

        return reordered_paths

    def _verify_merge_compatibility(self):
        """Verify that all data files are compatible for merging."""
        if len(self.all_data) < 2:
            raise ValueError("Merge mode requires at least 2 data files")

        # Check number of tasks
        n_tasks_ref = len(self.all_data[0]['all_decs'])
        for i, data in enumerate(self.all_data[1:], 1):
            n_tasks = len(data['all_decs'])
            if n_tasks != n_tasks_ref:
                raise ValueError(f"Incompatible data: file {i} has {n_tasks} tasks, expected {n_tasks_ref}")

        # Check dimensions and objectives for each task
        for task_id in range(n_tasks_ref):
            dim_ref = self.all_data[0]['all_decs'][task_id][0].shape[1]
            n_obj_ref = self.all_data[0]['all_objs'][task_id][0].shape[1]

            for i, data in enumerate(self.all_data[1:], 1):
                dim = data['all_decs'][task_id][0].shape[1]
                n_obj = data['all_objs'][task_id][0].shape[1]

                if dim != dim_ref:
                    raise ValueError(
                        f"Incompatible data: task {task_id} in file {i} has dimension {dim}, expected {dim_ref}"
                    )
                if n_obj != n_obj_ref:
                    raise ValueError(
                        f"Incompatible data: task {task_id} in file {i} has {n_obj} objectives, expected {n_obj_ref}"
                    )

    def create_animation(self, interval=100):
        """
        Create the optimization animation.

        Parameters
        ----------
        interval : int, optional
            Delay between frames in milliseconds (default: 100)

        Returns
        -------
        str
            Path to the saved animation file
        """
        # Determine layout based on merge mode
        # Keep fixed column width (6 units per column) and scale total width
        column_width = 6

        if self.merge == 0:
            # Single file: 2 columns (decision left, objective right)
            n_cols = 2
            fig_width = column_width * n_cols
        elif self.merge == 1:
            # Full merge: 2 columns (both merged)
            n_cols = 2
            fig_width = column_width * n_cols
        elif self.merge == 2:
            # Partial merge: n_algorithms + 1 columns (decision separated, objective merged)
            n_cols = self.n_algorithms + 1
            fig_width = column_width * n_cols
        elif self.merge == 3:
            # All separated: 2 * n_algorithms columns
            n_cols = 2 * self.n_algorithms
            fig_width = column_width * n_cols

        n_rows = self.n_tasks

        fig = plt.figure(figsize=(fig_width, 4 * n_rows))

        # Set title only if specified
        if self.title:
            fig.suptitle(self.title, fontsize=16, fontweight='bold')
        elif self.merge == 0:
            # Use filename as title for single file mode
            fig.suptitle(self.pkl_path.stem, fontsize=16, fontweight='bold')

        # Adjust spacing between subplots
        # plt.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.05, right=0.95)

        # Create subplots based on merge mode
        axes = self._create_subplots(fig, n_rows, n_cols)

        # Initialize plots
        self._init_plots(axes)

        # Create animation
        if self.merge > 0:
            anim = FuncAnimation(
                fig,
                self._update_merge,
                frames=self.max_gen,
                fargs=(axes,),
                interval=interval,
                blit=False,
                repeat=True
            )
        else:
            anim = FuncAnimation(
                fig,
                self._update,
                frames=self.max_gen,
                fargs=(axes,),
                interval=interval,
                blit=False,
                repeat=True
            )

        # Save animation
        print(f"Generating animation... (this may take a while)")

        # Choose writer based on file extension
        file_ext = self.output_path.suffix.lower()
        if file_ext == '.mp4':
            if FFMPEG_AVAILABLE:
                try:
                    writer = FFMpegWriter(fps=self.fps, bitrate=1800)
                    anim.save(self.output_path, writer=writer, dpi=self.dpi)
                except Exception as e:
                    print(f"Warning: Failed to save MP4 ({e}), falling back to GIF")
                    self.output_path = self.output_path.with_suffix('.gif')
                    writer = PillowWriter(fps=self.fps)
                    anim.save(self.output_path, writer=writer, dpi=self.dpi)
            else:
                print(f"Warning: FFMpeg not installed, falling back to GIF")
                print(f"To use MP4: pip install ffmpeg-python")
                self.output_path = self.output_path.with_suffix('.gif')
                writer = PillowWriter(fps=self.fps)
                anim.save(self.output_path, writer=writer, dpi=self.dpi)
        else:
            # Default to GIF
            writer = PillowWriter(fps=self.fps)
            anim.save(self.output_path, writer=writer, dpi=self.dpi)

        print(f"Animation saved to: {self.output_path}")

        plt.close(fig)
        return str(self.output_path)

    def _create_subplots(self, fig, n_rows, n_cols):
        """Create subplot structure based on merge mode."""
        axes = []

        if self.merge == 0:
            # Single file: standard 2-column layout
            for i in range(n_rows):
                ax_left = plt.subplot(n_rows, n_cols, i * n_cols + 1)
                if self.n_objs_per_task[i] == 3:
                    ax_right = plt.subplot(n_rows, n_cols, i * n_cols + 2, projection='3d')
                else:
                    ax_right = plt.subplot(n_rows, n_cols, i * n_cols + 2)
                axes.append((ax_left, ax_right))

        elif self.merge == 1:
            # Full merge: 2 columns, all algorithms in same plots
            for i in range(n_rows):
                ax_left = plt.subplot(n_rows, n_cols, i * n_cols + 1)
                if self.n_objs_per_task[i] == 3:
                    ax_right = plt.subplot(n_rows, n_cols, i * n_cols + 2, projection='3d')
                else:
                    ax_right = plt.subplot(n_rows, n_cols, i * n_cols + 2)
                axes.append({'decision': ax_left, 'objective': ax_right})

        elif self.merge == 2:
            # Partial merge: n_algorithms decision plots + 1 merged objective plot
            for task_id in range(n_rows):
                task_axes = {'decision': [], 'objective': None}

                # Create decision space plots for each algorithm
                for algo_idx in range(self.n_algorithms):
                    ax_dec = plt.subplot(n_rows, n_cols, task_id * n_cols + algo_idx + 1)
                    task_axes['decision'].append(ax_dec)

                # Create merged objective space plot
                if self.n_objs_per_task[task_id] == 3:
                    ax_obj = plt.subplot(n_rows, n_cols, task_id * n_cols + self.n_algorithms + 1, projection='3d')
                else:
                    ax_obj = plt.subplot(n_rows, n_cols, task_id * n_cols + self.n_algorithms + 1)
                task_axes['objective'] = ax_obj

                axes.append(task_axes)

        elif self.merge == 3:
            # All separated: n_algorithms decision plots + n_algorithms objective plots
            for task_id in range(n_rows):
                task_axes = {'decision': [], 'objective': []}

                # Create decision space plots for each algorithm
                for algo_idx in range(self.n_algorithms):
                    ax_dec = plt.subplot(n_rows, n_cols, task_id * n_cols + algo_idx + 1)
                    task_axes['decision'].append(ax_dec)

                # Create objective space plots for each algorithm
                for algo_idx in range(self.n_algorithms):
                    if self.n_objs_per_task[task_id] == 3:
                        ax_obj = plt.subplot(n_rows, n_cols,
                                             task_id * n_cols + self.n_algorithms + algo_idx + 1,
                                             projection='3d')
                    else:
                        ax_obj = plt.subplot(n_rows, n_cols,
                                             task_id * n_cols + self.n_algorithms + algo_idx + 1)
                    task_axes['objective'].append(ax_obj)

                axes.append(task_axes)

        return axes

    def _init_plots(self, axes):
        """Initialize all subplots with labels and limits."""
        if self.merge == 0:
            # Single file mode
            for task_id in range(self.n_tasks):
                ax_left, ax_right = axes[task_id]
                n_objs = self.n_objs_per_task[task_id]
                dim = self.dims_per_task[task_id]

                # Left plot: Decision space
                ax_left.set_title(f'Task {task_id + 1}: Decision Space (dim={dim})')
                ax_left.set_xlabel('Decision Variables')
                ax_left.set_ylabel('Normalized Value')
                ax_left.set_ylim(-0.1, 1.1)
                ax_left.grid(True, alpha=0.3)

                # Right plot: Objective space or convergence curve
                if n_objs == 1:
                    ax_right.set_title(f'Task {task_id + 1}: Convergence Curve')
                    ax_right.set_xlabel('NFEs')
                    ylabel = 'Best Objective Value (log)' if self.logscale else 'Best Objective Value'
                    ax_right.set_ylabel(ylabel)
                    if self.logscale:
                        ax_right.set_yscale('log')
                    ax_right.grid(True, alpha=0.3)
                elif n_objs == 2:
                    ax_right.set_title(f'Task {task_id + 1}: Objective Space')
                    ax_right.set_xlabel('f1')
                    ax_right.set_ylabel('f2')
                    ax_right.grid(True, alpha=0.3)
                elif n_objs == 3:
                    ax_right.set_title(f'Task {task_id + 1}: Objective Space (3D)')
                    ax_right.set_xlabel('f1')
                    ax_right.set_ylabel('f2')
                    ax_right.set_zlabel('f3')
                    ax_right.grid(True, alpha=0.3)
                    ax_right.view_init(elev=30, azim=45)
                else:
                    ax_right.set_title(f'Task {task_id + 1}: Objective Space (n_objs={n_objs})')
                    ax_right.set_xlabel('Objectives')
                    ax_right.set_ylabel('Normalized Objective Value')
                    ax_right.set_ylim(-0.1, 1.1)
                    ax_right.grid(True, alpha=0.3)

    def _update(self, frame, axes):
        """Update function for animation (single file mode, merge=0)."""
        for task_id in range(self.n_tasks):
            ax_left, ax_right = axes[task_id]

            # Get current generation index for this task
            gen_idx = min(frame, self.n_generations[task_id] - 1)

            # Clear previous plots
            ax_left.clear()
            ax_right.clear()

            # Re-initialize labels
            self._init_task_plots(task_id, ax_left, ax_right, gen_idx)

            # Get current data
            decs = self.all_decs[task_id][gen_idx]
            objs = self.all_objs[task_id][gen_idx]

            n_objs = self.n_objs_per_task[task_id]
            dim = self.dims_per_task[task_id]

            # Plot decision space (left)
            self._plot_decision_space(ax_left, decs, objs, dim, n_objs)

            # Plot objective space or convergence (right)
            if n_objs == 1:
                self._plot_convergence_curve(ax_right, task_id, gen_idx)
            else:
                self._plot_objective_space(ax_right, objs, n_objs)

    def _update_merge(self, frame, axes):
        """Update function for animation (merge modes 1, 2, 3)."""
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

        for task_id in range(self.n_tasks):
            n_objs = self.n_objs_per_task[task_id]
            dim = self.dims_per_task[task_id]

            if self.merge == 1:
                # Full merge mode
                ax_dec = axes[task_id]['decision']
                ax_obj = axes[task_id]['objective']

                ax_dec.clear()
                ax_obj.clear()

                self._init_task_plots_merge(task_id, ax_dec, ax_obj, mode=1)

                # Plot all algorithms
                for algo_idx in range(self.n_algorithms):
                    color = colors[algo_idx % len(colors)]
                    algo_name = self.algorithm_names[algo_idx]

                    max_gen_algo = self.n_generations_per_algo[algo_idx][task_id]
                    gen_idx = int((frame / self.max_gen) * max_gen_algo)
                    gen_idx = min(gen_idx, max_gen_algo - 1)

                    decs = self.all_decs[algo_idx][task_id][gen_idx]
                    objs = self.all_objs[algo_idx][task_id][gen_idx]

                    # Plot decision space
                    self._plot_decision_space_merge(ax_dec, decs, dim, color, algo_name, algo_idx == 0)

                    # Plot objective space
                    if n_objs == 1:
                        self._plot_convergence_curve_merge(ax_obj, algo_idx, task_id, frame, color, algo_name)
                    else:
                        self._plot_objective_space_merge(ax_obj, objs, n_objs, color, algo_name, algo_idx == 0)

                ax_dec.legend(loc='upper right', fontsize=8)
                ax_obj.legend(loc='upper right', fontsize=8)

            elif self.merge == 2:
                # Partial merge: decision separated, objective merged
                axes_dec = axes[task_id]['decision']
                ax_obj = axes[task_id]['objective']

                # Clear all axes
                for ax in axes_dec:
                    ax.clear()
                ax_obj.clear()

                # Initialize objective plot
                self._init_task_plots_merge(task_id, None, ax_obj, mode=2)

                # Plot each algorithm
                for algo_idx in range(self.n_algorithms):
                    color = colors[algo_idx % len(colors)]
                    algo_name = self.algorithm_names[algo_idx]

                    max_gen_algo = self.n_generations_per_algo[algo_idx][task_id]
                    gen_idx = int((frame / self.max_gen) * max_gen_algo)
                    gen_idx = min(gen_idx, max_gen_algo - 1)

                    decs = self.all_decs[algo_idx][task_id][gen_idx]
                    objs = self.all_objs[algo_idx][task_id][gen_idx]

                    # Calculate current NFEs for this algorithm
                    max_nfes = self.max_nfes_per_task[task_id]
                    current_nfes = int((gen_idx / (max_gen_algo - 1)) * max_nfes) if max_gen_algo > 1 else max_nfes

                    # Plot decision space (individual subplot)
                    ax_dec = axes_dec[algo_idx]
                    ax_dec.set_title(f'Task {task_id + 1}: {algo_name} (NFEs={current_nfes})')
                    ax_dec.set_xlabel('Decision Variables')
                    ax_dec.set_ylabel('Normalized Value')
                    ax_dec.set_ylim(-0.1, 1.1)
                    ax_dec.grid(True, alpha=0.3)
                    self._plot_decision_space_single(ax_dec, decs, dim, color)

                    # Plot objective space (merged)
                    if n_objs == 1:
                        self._plot_convergence_curve_merge(ax_obj, algo_idx, task_id, frame, color, algo_name)
                    else:
                        self._plot_objective_space_merge(ax_obj, objs, n_objs, color, algo_name, algo_idx == 0)

                ax_obj.legend(loc='upper right', fontsize=8)

            elif self.merge == 3:
                # All separated
                axes_dec = axes[task_id]['decision']
                axes_obj = axes[task_id]['objective']

                # Clear all axes
                for ax in axes_dec:
                    ax.clear()
                for ax in axes_obj:
                    ax.clear()

                # Plot each algorithm
                for algo_idx in range(self.n_algorithms):
                    color = colors[algo_idx % len(colors)]
                    algo_name = self.algorithm_names[algo_idx]

                    max_gen_algo = self.n_generations_per_algo[algo_idx][task_id]
                    gen_idx = int((frame / self.max_gen) * max_gen_algo)
                    gen_idx = min(gen_idx, max_gen_algo - 1)

                    decs = self.all_decs[algo_idx][task_id][gen_idx]
                    objs = self.all_objs[algo_idx][task_id][gen_idx]

                    # Calculate current NFEs for this algorithm
                    max_nfes = self.max_nfes_per_task[task_id]
                    current_nfes = int((gen_idx / (max_gen_algo - 1)) * max_nfes) if max_gen_algo > 1 else max_nfes

                    # Plot decision space (individual subplot)
                    ax_dec = axes_dec[algo_idx]
                    ax_dec.set_title(f'Task {task_id + 1}: {algo_name} Decision (NFEs={current_nfes})')
                    ax_dec.set_xlabel('Decision Variables')
                    ax_dec.set_ylabel('Normalized Value')
                    ax_dec.set_ylim(-0.1, 1.1)
                    ax_dec.grid(True, alpha=0.3)
                    self._plot_decision_space_single(ax_dec, decs, dim, color)

                    # Plot objective space (individual subplot)
                    ax_obj = axes_obj[algo_idx]
                    if n_objs == 1:
                        ax_obj.set_title(f'Task {task_id + 1}: {algo_name} Convergence')
                        ax_obj.set_xlabel('NFEs')
                        ylabel = 'Best Objective Value (log)' if self.logscale else 'Best Objective Value'
                        ax_obj.set_ylabel(ylabel)
                        if self.logscale:
                            ax_obj.set_yscale('log')
                        ax_obj.grid(True, alpha=0.3)
                        self._plot_convergence_curve_single(ax_obj, algo_idx, task_id, frame, color)
                    else:
                        if n_objs == 2:
                            ax_obj.set_title(f'Task {task_id + 1}: {algo_name} Objective')
                            ax_obj.set_xlabel('f1')
                            ax_obj.set_ylabel('f2')
                        elif n_objs == 3:
                            ax_obj.set_title(f'Task {task_id + 1}: {algo_name} Objective')
                            ax_obj.set_xlabel('f1')
                            ax_obj.set_ylabel('f2')
                            ax_obj.set_zlabel('f3')
                            ax_obj.view_init(elev=30, azim=45)
                        else:
                            ax_obj.set_title(f'Task {task_id + 1}: {algo_name} Objective')
                            ax_obj.set_xlabel('Objectives')
                            ax_obj.set_ylabel('Normalized Value')
                            ax_obj.set_ylim(-0.1, 1.1)
                        ax_obj.grid(True, alpha=0.3)
                        self._plot_objective_space_single(ax_obj, objs, n_objs, color)

    def _init_task_plots(self, task_id, ax_left, ax_right, gen_idx):
        """Initialize plots for a specific task (merge=0)."""
        n_objs = self.n_objs_per_task[task_id]
        dim = self.dims_per_task[task_id]

        # Left plot
        ax_left.set_title(f'Task {task_id + 1}: Decision Space (Gen {gen_idx + 1})')
        ax_left.set_xlabel('Decision Variables')
        ax_left.set_ylabel('Normalized Value')
        ax_left.set_ylim(-0.1, 1.1)
        ax_left.grid(True, alpha=0.3)

        # Right plot
        if n_objs == 1:
            ax_right.set_title(f'Task {task_id + 1}: Convergence Curve')
            ax_right.set_xlabel('NFEs')
            ylabel = 'Best Objective Value (log)' if self.logscale else 'Best Objective Value'
            ax_right.set_ylabel(ylabel)
            if self.logscale:
                ax_right.set_yscale('log')
        elif n_objs == 2:
            ax_right.set_title(f'Task {task_id + 1}: Objective Space (Gen {gen_idx + 1})')
            ax_right.set_xlabel('f1')
            ax_right.set_ylabel('f2')
        elif n_objs == 3:
            ax_right.set_title(f'Task {task_id + 1}: Objective Space (Gen {gen_idx + 1})')
            ax_right.set_xlabel('f1')
            ax_right.set_ylabel('f2')
            ax_right.set_zlabel('f3')
        else:
            ax_right.set_title(f'Task {task_id + 1}: Objective Space (Gen {gen_idx + 1})')
            ax_right.set_xlabel('Objectives')
            ax_right.set_ylabel('Normalized Objective Value')
            ax_right.set_ylim(-0.1, 1.1)

        ax_right.grid(True, alpha=0.3)

    def _init_task_plots_merge(self, task_id, ax_dec, ax_obj, mode=1):
        """Initialize plots for merge modes."""
        n_objs = self.n_objs_per_task[task_id]
        dim = self.dims_per_task[task_id]

        # Decision plot (for mode 1)
        if ax_dec is not None:
            ax_dec.set_title(f'Task {task_id + 1}: Decision Space (dim={dim})')
            ax_dec.set_xlabel('Decision Variables')
            ax_dec.set_ylabel('Normalized Value')
            ax_dec.set_ylim(-0.1, 1.1)
            ax_dec.grid(True, alpha=0.3)

        # Objective plot
        if n_objs == 1:
            ax_obj.set_title(f'Task {task_id + 1}: Convergence Curve')
            ax_obj.set_xlabel('NFEs')
            ylabel = 'Best Objective Value (log)' if self.logscale else 'Best Objective Value'
            ax_obj.set_ylabel(ylabel)
            if self.logscale:
                ax_obj.set_yscale('log')
        elif n_objs == 2:
            ax_obj.set_title(f'Task {task_id + 1}: Objective Space')
            ax_obj.set_xlabel('f1')
            ax_obj.set_ylabel('f2')
        elif n_objs == 3:
            ax_obj.set_title(f'Task {task_id + 1}: Objective Space (3D)')
            ax_obj.set_xlabel('f1')
            ax_obj.set_ylabel('f2')
            ax_obj.set_zlabel('f3')
            ax_obj.view_init(elev=30, azim=45)
        else:
            ax_obj.set_title(f'Task {task_id + 1}: Objective Space (n_objs={n_objs})')
            ax_obj.set_xlabel('Objectives')
            ax_obj.set_ylabel('Normalized Objective Value')
            ax_obj.set_ylim(-0.1, 1.1)

        ax_obj.grid(True, alpha=0.3)

    def _plot_decision_space(self, ax, decs, objs, dim, n_objs):
        """Plot decision space using parallel coordinates (merge=0)."""
        n_samples = decs.shape[0]
        x_positions = np.arange(dim)

        for i in range(n_samples):
            ax.plot(x_positions, decs[i, :], alpha=0.3, linewidth=1, color='blue')

        if dim <= 10:
            tick_indices = x_positions
        else:
            step = max(1, dim // 10)
            tick_indices = x_positions[::step]
            if tick_indices[-1] != x_positions[-1]:
                tick_indices = np.append(tick_indices, x_positions[-1])

        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{i + 1}' for i in tick_indices], rotation=0)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Normalized Value')
        ax.set_xlabel('Decision Variables')

    def _plot_decision_space_merge(self, ax, decs, dim, color, label, show_ticks=True):
        """Plot decision space for merge mode 1 (merged)."""
        n_samples = decs.shape[0]
        x_positions = np.arange(dim)

        for i in range(n_samples):
            if i == 0:
                ax.plot(x_positions, decs[i, :], alpha=0.3, linewidth=1, color=color, label=label)
            else:
                ax.plot(x_positions, decs[i, :], alpha=0.3, linewidth=1, color=color)

        if show_ticks:
            if dim <= 10:
                tick_indices = x_positions
            else:
                step = max(1, dim // 10)
                tick_indices = x_positions[::step]
                if tick_indices[-1] != x_positions[-1]:
                    tick_indices = np.append(tick_indices, x_positions[-1])

            ax.set_xticks(tick_indices)
            ax.set_xticklabels([f'{i + 1}' for i in tick_indices], rotation=0)
            ax.set_ylim(-0.1, 1.1)

    def _plot_decision_space_single(self, ax, decs, dim, color):
        """Plot decision space for individual subplot (merge modes 2, 3)."""
        n_samples = decs.shape[0]
        x_positions = np.arange(dim)

        for i in range(n_samples):
            ax.plot(x_positions, decs[i, :], alpha=0.3, linewidth=1, color=color)

        if dim <= 10:
            tick_indices = x_positions
        else:
            step = max(1, dim // 10)
            tick_indices = x_positions[::step]
            if tick_indices[-1] != x_positions[-1]:
                tick_indices = np.append(tick_indices, x_positions[-1])

        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{i + 1}' for i in tick_indices], rotation=0)

    def _plot_convergence_curve(self, ax, task_id, current_gen):
        """Plot convergence curve for single-objective optimization (merge=0)."""
        best_objs = []
        nfes_list = []

        max_nfes = self.max_nfes_per_task[task_id]
        n_gens = self.n_generations[task_id]

        for gen in range(current_gen + 1):
            objs = self.all_objs[task_id][gen]
            best_obj = np.min(objs)
            best_objs.append(best_obj)
            # Calculate NFEs: uniformly distribute over max_nfes
            nfes = int((gen / (n_gens - 1)) * max_nfes) if n_gens > 1 else max_nfes
            nfes_list.append(nfes)

        ax.plot(nfes_list, best_objs, 'r-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        ax.set_xlim(0, max_nfes)

        if not self.logscale and len(best_objs) > 0:
            y_min, y_max = min(best_objs), max(best_objs)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    def _plot_convergence_curve_merge(self, ax, algo_idx, task_id, current_frame, color, label):
        """Plot convergence curve for merge modes 1, 2 (merged objective)."""
        max_gen_algo = self.n_generations_per_algo[algo_idx][task_id]
        current_gen = int((current_frame / self.max_gen) * max_gen_algo)
        current_gen = min(current_gen, max_gen_algo - 1)

        best_objs = []
        nfes_list = []

        max_nfes = self.max_nfes_per_task[task_id]

        for gen in range(current_gen + 1):
            objs = self.all_objs[algo_idx][task_id][gen]
            best_obj = np.min(objs)
            best_objs.append(best_obj)
            # Calculate NFEs: uniformly distribute over max_nfes
            nfes = int((gen / (max_gen_algo - 1)) * max_nfes) if max_gen_algo > 1 else max_nfes
            nfes_list.append(nfes)

        ax.plot(nfes_list, best_objs, '-', linewidth=2, marker='o', markersize=3,
                color=color, label=label, alpha=0.7)
        ax.set_xlim(0, max_nfes)

        if not self.logscale and len(best_objs) > 0:
            current_lines = ax.get_lines()
            all_y_data = []
            for line in current_lines:
                all_y_data.extend(line.get_ydata())

            if all_y_data:
                y_min, y_max = min(all_y_data), max(all_y_data)
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    def _plot_convergence_curve_single(self, ax, algo_idx, task_id, current_frame, color):
        """Plot convergence curve for merge mode 3 (individual subplot)."""
        max_gen_algo = self.n_generations_per_algo[algo_idx][task_id]
        current_gen = int((current_frame / self.max_gen) * max_gen_algo)
        current_gen = min(current_gen, max_gen_algo - 1)

        best_objs = []
        nfes_list = []

        max_nfes = self.max_nfes_per_task[task_id]

        for gen in range(current_gen + 1):
            objs = self.all_objs[algo_idx][task_id][gen]
            best_obj = np.min(objs)
            best_objs.append(best_obj)
            # Calculate NFEs: uniformly distribute over max_nfes
            nfes = int((gen / (max_gen_algo - 1)) * max_nfes) if max_gen_algo > 1 else max_nfes
            nfes_list.append(nfes)

        ax.plot(nfes_list, best_objs, '-', linewidth=2, marker='o', markersize=3, color=color, alpha=0.7)
        ax.set_xlim(0, max_nfes)

        if not self.logscale and len(best_objs) > 0:
            y_min, y_max = min(best_objs), max(best_objs)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    def _plot_objective_space(self, ax, objs, n_objs):
        """Plot objective space (merge=0)."""
        # Filter for non-dominated solutions only
        front_no, _ = nd_sort(objs, len(objs))
        pareto_mask = (front_no == 1)
        pareto_objs = objs[pareto_mask]
        n_samples = pareto_objs.shape[0]

        if n_objs == 2:
            ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], c='red', alpha=0.5, s=50, label='Pareto Front')
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.legend(loc='upper right')

        elif n_objs == 3:
            ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], pareto_objs[:, 2], c='red', alpha=0.5, s=50,
                       label='Pareto Front')
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.set_zlabel('f3')
            ax.legend(loc='upper right')

        else:
            x_positions = np.arange(n_objs)
            objs_normalized = pareto_objs.copy()
            for j in range(n_objs):
                obj_min, obj_max = pareto_objs[:, j].min(), pareto_objs[:, j].max()
                if obj_max > obj_min:
                    objs_normalized[:, j] = (pareto_objs[:, j] - obj_min) / (obj_max - obj_min)
                else:
                    objs_normalized[:, j] = 0.5

            for i in range(n_samples):
                ax.plot(x_positions, objs_normalized[i, :], alpha=0.4, linewidth=1.5, color='red')

            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'f{i + 1}' for i in range(n_objs)])
            ax.set_ylim(-0.1, 1.1)
            ax.set_ylabel('Normalized Objective Value')
            ax.set_xlabel('Objectives')
            ax.legend([f'Pareto Front ({n_samples} solutions)'], loc='upper right')

    def _plot_objective_space_merge(self, ax, objs, n_objs, color, label, show_setup=True):
        """Plot objective space for merge modes 1, 2 (merged)."""
        # Filter for non-dominated solutions only
        front_no, _ = nd_sort(objs, len(objs))
        pareto_mask = (front_no == 1)
        pareto_objs = objs[pareto_mask]
        n_samples = pareto_objs.shape[0]

        if n_objs == 2:
            ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], c=color, alpha=0.5, s=50, label=label)

        elif n_objs == 3:
            ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], pareto_objs[:, 2], c=color, alpha=0.5, s=50, label=label)

        else:
            x_positions = np.arange(n_objs)
            objs_normalized = pareto_objs.copy()
            for j in range(n_objs):
                obj_min, obj_max = pareto_objs[:, j].min(), pareto_objs[:, j].max()
                if obj_max > obj_min:
                    objs_normalized[:, j] = (pareto_objs[:, j] - obj_min) / (obj_max - obj_min)
                else:
                    objs_normalized[:, j] = 0.5

            for i in range(n_samples):
                if i == 0:
                    ax.plot(x_positions, objs_normalized[i, :], alpha=0.4, linewidth=1.5,
                            color=color, label=label)
                else:
                    ax.plot(x_positions, objs_normalized[i, :], alpha=0.4, linewidth=1.5, color=color)

            if show_setup:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([f'f{i + 1}' for i in range(n_objs)])
                ax.set_ylim(-0.1, 1.1)

    def _plot_objective_space_single(self, ax, objs, n_objs, color):
        """Plot objective space for merge mode 3 (individual subplot)."""
        # Filter for non-dominated solutions only
        front_no, _ = nd_sort(objs, len(objs))
        pareto_mask = (front_no == 1)
        pareto_objs = objs[pareto_mask]
        n_samples = pareto_objs.shape[0]

        if n_objs == 2:
            ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], c=color, alpha=0.5, s=50)

        elif n_objs == 3:
            ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], pareto_objs[:, 2], c=color, alpha=0.5, s=50)

        else:
            x_positions = np.arange(n_objs)
            objs_normalized = pareto_objs.copy()
            for j in range(n_objs):
                obj_min, obj_max = pareto_objs[:, j].min(), pareto_objs[:, j].max()
                if obj_max > obj_min:
                    objs_normalized[:, j] = (pareto_objs[:, j] - obj_min) / (obj_max - obj_min)
                else:
                    objs_normalized[:, j] = 0.5

            for i in range(n_samples):
                ax.plot(x_positions, objs_normalized[i, :], alpha=0.4, linewidth=1.5, color=color)

            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'f{i + 1}' for i in range(n_objs)])


def create_optimization_animation(pkl_path=None, output_path=None, fps=10, dpi=100, interval=100,
                                  data_path='./TestData', save_path='./TestResults', pattern='*.pkl',
                                  format='gif', merge=0, title=None, algorithm_order=None, max_nfes=100,
                                  logscale=False):
    """
    Convenience function to create optimization animation.

    If pkl_path is None, automatically scans data_path for all .pkl files.

    Parameters
    ----------
    pkl_path : str or list, optional
        Path to the .pkl results file, or list of paths for merge mode.
        If None, scans data_path for all .pkl files
    output_path : str, optional
        Path for output animation file
    fps : int, optional
        Frames per second (default: 10)
    dpi : int, optional
        DPI for output (default: 100)
    interval : int, optional
        Delay between frames in milliseconds (default: 100)
    data_path : str, optional
        Directory to scan for .pkl files when pkl_path is None (default: './TestData')
    save_path : str, optional
        Directory to save animations when pkl_path is None (default: './TestResults')
    pattern : str, optional
        File pattern to search for when pkl_path is None (default: '*.pkl')
    format : str, optional
        Output format: 'gif' or 'mp4' (default: 'gif')
    merge : int, optional
        Merge mode (default: 0)
        - 0: No merge, individual animations for each file
        - 1: Full merge, all algorithms in same plots (left & right)
        - 2: Partial merge, separate decision space (left), merged objective space (right)
        - 3: Separate plots, decision and objective spaces both separated
    title : str, optional
        Custom title for the animation (for merge mode, default is 'test')
    algorithm_order : list, optional
        List of algorithm names (file stems) specifying the display order.
        Only used when merge>0. If None, uses the order from pkl_path.
        Example: ['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT']
    max_nfes : int or list, optional
        Maximum number of function evaluations (NFEs) for each task.
        Can be a scalar (same for all tasks) or a list (one per task).
        Default: 100
        Examples: max_nfes=200 or max_nfes=[200, 300]
    logscale : bool, optional
        Use logarithmic scale for y-axis in single-objective convergence curves (default: False)

    Returns
    -------
    str or dict
        If pkl_path is provided: Path to the saved animation file
        If pkl_path is None: Dictionary with 'success' and 'failed' lists

    Examples
    --------
    >>> # Single file - GIF (default)
    >>> create_optimization_animation('results.pkl')
    >>>
    >>> # Single file with custom NFEs and log scale
    >>> create_optimization_animation('results.pkl', max_nfes=200, logscale=True)
    >>> create_optimization_animation('results.pkl', max_nfes=[200, 300])  # Multi-task
    >>>
    >>> # Single file - MP4 (smaller file, better quality)
    >>> create_optimization_animation('results.pkl', format='mp4')
    >>> create_optimization_animation('results.pkl', output_path='result.mp4')
    >>>
    >>> # Merge mode 1: Full merge with log scale
    >>> create_optimization_animation(['algo1.pkl', 'algo2.pkl', 'algo3.pkl'], merge=1,
    >>>                               max_nfes=200, logscale=True)
    >>>
    >>> # Merge mode 2 with custom order: Decision separated, objective merged
    >>> create_optimization_animation(
    >>>     ['algo1.pkl', 'algo2.pkl', 'algo3.pkl'],
    >>>     merge=2,
    >>>     title='Comparison',
    >>>     algorithm_order=['BO', 'MTBO', 'RAMTEA'],
    >>>     max_nfes=[200, 300],
    >>>     logscale=True
    >>> )
    >>>
    >>> # Merge mode 3: All separated with custom order
    >>> create_optimization_animation(
    >>>     ['algo1.pkl', 'algo2.pkl'],
    >>>     merge=3,
    >>>     algorithm_order=['RAMTEA', 'BO'],
    >>>     max_nfes=500
    >>> )
    >>>
    >>> # Auto-scan - generate all as MP4
    >>> create_optimization_animation(format='mp4', max_nfes=200)
    >>> create_optimization_animation(dpi=70, fps=8, format='mp4')
    >>>
    >>> # Auto-scan and merge all files with custom order
    >>> create_optimization_animation(
    >>>     merge=1,
    >>>     title='All Algorithms',
    >>>     algorithm_order=['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT'],
    >>>     max_nfes=1000,
    >>>     logscale=True
    >>> )
    """
    if pkl_path is None:
        # Auto-scan mode
        return generate_all_animations(data_path, save_path, fps, dpi, interval, pattern, format, merge, title,
                                       algorithm_order, max_nfes, logscale)
    else:
        # Single file or merge mode
        # If output_path not specified, use format parameter
        if output_path is None:
            if isinstance(pkl_path, list):
                base_path = Path(pkl_path[0]).parent
                if title:
                    output_path = str(base_path / f"{title}_animation.{format}")
                else:
                    output_path = str(base_path / f"test_animation.{format}")
            else:
                output_path = str(Path(pkl_path).parent / f"{Path(pkl_path).stem}_animation.{format}")

        animator = OptimizationAnimator(pkl_path, output_path, fps, dpi, merge=merge, title=title,
                                        algorithm_order=algorithm_order, max_nfes=max_nfes, logscale=logscale)
        return animator.create_animation(interval=interval)


def generate_all_animations(data_path='./TestData', save_path='./TestResults',
                            fps=10, dpi=100, interval=100, pattern='*.pkl', format='gif',
                            merge=0, title=None, algorithm_order=None, max_nfes=100, logscale=False):
    """
    Automatically scan for .pkl files and generate animations for all results.

    Parameters
    ----------
    data_path : str
        Directory containing .pkl result files (default: './TestData')
    save_path : str
        Directory to save animation files (default: './TestResults')
    fps : int
        Frames per second (default: 10)
    dpi : int
        Resolution (default: 100)
    interval : int
        Interval between frames in milliseconds (default: 100)
    pattern : str
        File pattern to search for (default: '*.pkl')
    format : str
        Output format: 'gif' or 'mp4' (default: 'gif')
    merge : int
        Merge mode (default: 0)
    title : str, optional
        Custom title for merged animation (default: 'test')
    algorithm_order : list, optional
        List of algorithm names specifying the display order when merge>0.
        Example: ['BO', 'MTBO', 'RAMTEA', 'BO-LCB-BCKT']
    max_nfes : int or list, optional
        Maximum number of function evaluations (NFEs) for each task.
        Can be a scalar (same for all tasks) or a list (one per task).
        Default: 100
    logscale : bool, optional
        Use logarithmic scale for y-axis in single-objective convergence curves (default: False)

    Returns
    -------
    dict
        Dictionary with 'success' and 'failed' lists of filenames
    """
    # Create save directory
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Scan for pkl files
    search_pattern = str(Path(data_path) / pattern)
    pkl_files = glob.glob(search_pattern)

    if not pkl_files:
        print(f"⚠️  No {pattern} files found in '{data_path}'")
        return {'success': [], 'failed': []}

    print("=" * 70)
    print(f"Optimization Animation Generator")
    print("=" * 70)
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")
    print(f"Found {len(pkl_files)} result files")
    print(f"Animation params: FPS={fps}, DPI={dpi}, Interval={interval}ms, Format={format.upper()}")
    print(f"Max NFEs: {max_nfes}")
    print(f"Log scale: {logscale}")

    merge_mode_names = {
        0: 'INDIVIDUAL',
        1: 'MERGE (Full)',
        2: 'MERGE (Decision Separated)',
        3: 'MERGE (All Separated)'
    }
    print(f"Mode: {merge_mode_names.get(merge, 'UNKNOWN')}")

    if merge > 0 and algorithm_order is not None:
        print(f"Algorithm Order: {algorithm_order}")

    print("=" * 70)
    print()

    # Track results
    success_list = []
    failed_list = []

    if merge > 0:
        # Merge mode: create one animation with all files
        output_name = title if title else 'test'
        output_path = save_dir / f"{output_name}_animation.{format}"

        print(f"Creating merged comparison animation...")

        # Display algorithm names (before reordering)
        algo_names = [Path(f).stem for f in pkl_files]
        if algorithm_order is not None:
            print(f"Original algorithms: {algo_names}")
            print(f"Ordered algorithms: {algorithm_order}")
        else:
            print(f"Algorithms: {algo_names}")

        try:
            create_optimization_animation(
                pkl_path=pkl_files,
                output_path=str(output_path),
                fps=fps,
                dpi=dpi,
                interval=interval,
                merge=merge,
                title=title,
                algorithm_order=algorithm_order,
                max_nfes=max_nfes,
                logscale=logscale
            )
            success_list = algo_names
            print(f"  ✓ Success\n")

        except Exception as e:
            failed_list = algo_names
            print(f"  ✗ Failed: {e}\n")

    else:
        # Individual mode: process all files separately
        for i, pkl_file in enumerate(pkl_files, 1):
            filename = Path(pkl_file).stem
            output_path = save_dir / f"{filename}_animation.{format}"

            print(f"[{i}/{len(pkl_files)}] Processing: {filename}")

            try:
                create_optimization_animation(
                    pkl_path=pkl_file,
                    output_path=str(output_path),
                    fps=fps,
                    dpi=dpi,
                    interval=interval,
                    max_nfes=max_nfes,
                    logscale=logscale
                )
                success_list.append(filename)
                print(f"  ✓ Success\n")

            except Exception as e:
                failed_list.append(filename)
                print(f"  ✗ Failed: {e}\n")

    # Summary
    print("=" * 70)
    print("Processing Complete!")
    if merge > 0:
        print(f"Merged animation: {'Success' if success_list else 'Failed'}")
    else:
        print(f"Success: {len(success_list)}/{len(pkl_files)}")
    if failed_list:
        print(f"Failed: {len(failed_list)}/{len(pkl_files) if merge == 0 else 1}")
        print("\nFailed files:")
        for fname in failed_list:
            print(f"  - {fname}")
    print("=" * 70)
    print(f"Animations saved to: {save_path}")
    print("=" * 70)

    return {'success': success_list, 'failed': failed_list}


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        # No arguments: auto-scan mode with default paths
        print("No arguments provided. Scanning default paths...\n")
        generate_all_animations(
            data_path='./TestData',
            save_path='./TestResults',
            fps=10,
            dpi=100,
            interval=100
        )
    else:
        # Single file mode
        pkl_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 100

        print(f"Processing single file: {pkl_file}")
        create_optimization_animation(pkl_file, output_file, fps, dpi)
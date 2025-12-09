import os
import time
import csv
import shutil
import yaml
from datetime import datetime
from typing import Type, Dict, Any, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib


class BatchExperiment:
    """
    Batch Experiment Module

    This class provides a framework to define and run batch experiments for multiple
    optimization algorithms on multiple benchmark problems. It supports:

    - Adding multiple problems via problem creator functions.
    - Adding multiple optimization algorithm classes with fixed parameters.
    - Running experiments in parallel using multiple CPU cores.
    - Logging execution time, status, and errors for each run.
    - Saving timing summaries to CSV files.
    - Printing experiment configuration summaries to console.
    - Optional folder clearing before experiments.
    - Saving and loading experiment configuration from YAML files.

    Author: Jiangtao Shen
    Email: j.shen5@exeter.ac.uk
    Date: 2025.11.25
    Version: 1.0
    """

    def __init__(self, base_path: str = './Data', clear_folder: bool = False):
        """
        Initialize batch experiment

        Args:
            base_path: Base path for data storage, defaults to './Data'
            clear_folder: If True, clear the base_path folder before initialization, defaults to False
        """
        self.base_path = base_path
        self.problems = []  # Store problems: [(problem_creator, problem_name, problem_params), ...]
        self.algorithms = []  # Store algorithms: [(algo_class, algo_name, params), ...]
        self.experiment_config: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'base_path': str(base_path),
            'clear_folder': clear_folder,
            'problems': [],
            'algorithms': []
        }

        # Clear folder if requested
        if clear_folder and os.path.exists(self.base_path):
            self._clear_folder(self.base_path)
            print(f"♻️ Clearing existing data folder: {self.base_path}")

        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            print(f"Created base path: {self.base_path}")

    def _clear_folder(self, folder_path: str):
        """
        Clear all contents in the specified folder

        Args:
            folder_path: Path to the folder to be cleared
        """
        try:
            if os.path.exists(folder_path):
                # Remove all contents in the folder
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
        except Exception as e:
            print(f"Warning: Failed to clear folder {folder_path}: {str(e)}")

    def add_problem(self, problem_creator, problem_name: str, **problem_params):
        """
        Add an experiment problem using a creator function

        Args:
            problem_creator: Function that creates the problem instance
            problem_name: Problem name (used for file naming)
            **problem_params: Parameters to pass to problem creator
        """
        self.problems.append((problem_creator, problem_name, problem_params))

        # Get the class name from the problem creator's __self__ attribute
        class_name = 'unknown'
        if hasattr(problem_creator, '__self__'):
            class_name = problem_creator.__self__.__class__.__name__

        # Save problem configuration
        self.experiment_config['problems'].append({
            'name': problem_name,
            'class': class_name,
            'creator_name': problem_creator.__name__ if hasattr(problem_creator, '__name__') else str(problem_creator),
            'module': problem_creator.__module__ if hasattr(problem_creator, '__module__') else 'unknown',
            'params': problem_params
        })

    def add_algorithm(self, algorithm_class: Type, algorithm_name: str, **params):
        """
        Add an optimization algorithm class

        Args:
            algorithm_class: Algorithm class (e.g., GA, DE, PSO, etc.)
            algorithm_name: Algorithm name (used for file naming and folder creation)
            **params: Fixed parameters for the algorithm (e.g., n, max_nfes, muc, mum, etc.)
                     Note: problem, save_path, and name will be set automatically
        """
        self.algorithms.append((algorithm_class, algorithm_name, params))

        # Create folder for this algorithm
        algo_folder = os.path.join(self.base_path, algorithm_name)
        if not os.path.exists(algo_folder):
            os.makedirs(algo_folder)

        # Save algorithm configuration
        self.experiment_config['algorithms'].append({
            'name': algorithm_name,
            'class': algorithm_class.__name__,
            'module': algorithm_class.__module__,
            'parameters': params
        })

    def save_config(self, n_runs: int, max_workers: int):
        """
        Save experiment configuration to YAML file with custom formatting

        Args:
            n_runs: Number of independent runs
            max_workers: Maximum number of worker processes
        """
        # Add run settings
        self.experiment_config['run_settings'] = {'n_runs': n_runs, 'max_workers': max_workers,
                                                  'start_time': datetime.now().isoformat()}

        # Save to YAML file
        config_path = os.path.join(self.base_path, 'experiment_config.yaml')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                # Write basic info
                f.write(f"created_at: {self.experiment_config['created_at']}\n")
                f.write(f"base_path: {self.experiment_config['base_path']}\n")
                f.write(f"clear_folder: {self.experiment_config['clear_folder']}\n\n")

                # Write problems with blank lines between each
                f.write("problems:\n")
                for i, prob in enumerate(self.experiment_config['problems']):
                    if i > 0:
                        f.write("\n")  # Add blank line before each problem (except first)
                    f.write(f"  - name: {prob['name']}\n")
                    f.write(f"    class: {prob['class']}\n")
                    f.write(f"    creator_name: {prob['creator_name']}\n")
                    f.write(f"    module: {prob['module']}\n")
                    f.write(f"    params: {prob['params']}\n")

                # Write algorithms with blank lines between each
                f.write("\nalgorithms:\n")
                for i, algo in enumerate(self.experiment_config['algorithms']):
                    if i > 0:
                        f.write("\n")  # Add blank line before each algorithm (except first)
                    f.write(f"  - name: {algo['name']}\n")
                    f.write(f"    class: {algo['class']}\n")
                    f.write(f"    module: {algo['module']}\n")
                    f.write(f"    parameters:\n")
                    for key, value in algo['parameters'].items():
                        f.write(f"      {key}: {value}\n")

                # Write run settings
                f.write("\nrun_settings:\n")
                f.write(f"  n_runs: {self.experiment_config['run_settings']['n_runs']}\n")
                f.write(f"  max_workers: {self.experiment_config['run_settings']['max_workers']}\n")
                f.write(f"  start_time: {self.experiment_config['run_settings']['start_time']}\n")

            print(f"💾 Configuration saved to: {config_path}\n")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save configuration: {str(e)}\n")

    @classmethod
    def from_config(cls, config_path: str):
        """
        Load experiment configuration from YAML file and create BatchExperiment instance

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            BatchExperiment: Configured experiment instance ready to run
        """
        print(f"📂 Loading configuration from: {config_path}")

        # Read configuration file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file: {str(e)}")

        print(f"📅 Original creation time: {config['created_at']}")

        # Create BatchExperiment instance
        batch_exp = cls(
            base_path=config['base_path'],
            clear_folder=config['clear_folder']
        )

        # Load problems
        for prob_config in config['problems']:
            try:
                # Import module
                module_name = prob_config['module']
                creator_name = prob_config['creator_name']
                class_name = prob_config.get('class', 'unknown')

                # Import the module
                module = importlib.import_module(module_name)

                # Try to get the problem class
                problem_class = None
                if class_name != 'unknown':
                    # If we have the class name, try to get it directly
                    if hasattr(module, class_name):
                        problem_class = getattr(module, class_name)

                # If we still don't have the class, try to find it by looking for the creator method
                if problem_class is None:
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if hasattr(attr, creator_name):
                            problem_class = attr
                            break

                # If not found, skip this problem with a warning
                if problem_class is None:
                    print(
                        f"  ⚠️ Warning: Could not find creator '{creator_name}' in module '{module_name}', skipping problem '{prob_config['name']}'")
                    continue

                # Create instance and get problem creator
                instance = problem_class()
                problem_creator = getattr(instance, creator_name)
                batch_exp.add_problem(
                    problem_creator,
                    prob_config['name'],
                    **prob_config['params']
                )

            except Exception as e:
                print(f"  ⚠️ Warning: Failed to load {prob_config['name']}: {str(e)}, skipping...")
                continue

        # Load algorithms
        for algo_config in config['algorithms']:
            try:
                # Import module and get algorithm class
                module = importlib.import_module(algo_config['module'])
                algorithm_class = getattr(module, algo_config['class'])

                batch_exp.add_algorithm(
                    algorithm_class,
                    algo_config['name'],
                    **algo_config['parameters']
                )

            except Exception as e:
                print(f"  ❌ Failed to load {algo_config['name']}: {str(e)}")
                raise

        # Store run settings for later use
        batch_exp._loaded_run_settings = config.get('run_settings', {})

        print("✅ Configuration loaded successfully!")
        return batch_exp

    def _run_single_experiment(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment task - recreate problem and algorithm in child process

        Args:
            task: Dictionary containing experiment parameters

        Returns:
            Dictionary with experiment results and timing information
        """
        algo_class = task['algo_class']
        algo_name = task['algo_name']
        problem_creator = task['problem_creator']
        problem_name = task['problem_name']
        problem_params = task['problem_params']
        run_id = task['run_id']
        algo_params = task['algo_params']
        save_path = task['save_path']
        file_name = task['file_name']

        # Record start time for performance measurement
        exp_start_time = time.time()
        status = "Success"
        error_msg = ""

        try:
            # Recreate problem instance in child process to avoid pickling issues
            problem_instance = problem_creator(**problem_params)

            # Create algorithm instance with the problem and parameters
            algorithm = algo_class(
                problem=problem_instance,
                save_path=save_path,
                name=file_name,
                **algo_params
            )

            # Execute the optimization
            algorithm.optimize()

        except Exception as e:
            status = "Failed"
            error_msg = str(e)

        # Calculate execution duration
        exp_end_time = time.time()
        exp_duration = exp_end_time - exp_start_time

        return {
            'Algorithm': algo_name,
            'Problem': problem_name,
            'Run': run_id,
            'Filename': file_name,
            'Time(s)': round(exp_duration, 4),
            'Status': status,
            'Error': error_msg
        }

    def run(self, n_runs: int = None, verbose: bool = True, max_workers: int = None):
        """
        Run all experiments using multi-core parallel processing

        Args:
            n_runs: Number of independent runs for each algorithm on each problem
                   If None and loaded from config, uses config value
            verbose: Whether to print detailed progress information
            max_workers: Maximum number of worker processes, defaults to CPU count if None
                        If None and loaded from config, uses config value
        """
        if not self.problems:
            print("Error: No problems added!")
            return

        if not self.algorithms:
            print("Error: No algorithms added!")
            return

        # Use loaded settings if available
        if hasattr(self, '_loaded_run_settings'):
            if n_runs is None:
                n_runs = self._loaded_run_settings.get('n_runs', 30)
            if max_workers is None:
                max_workers = self._loaded_run_settings.get('max_workers', mp.cpu_count())
        else:
            # Use default values
            if n_runs is None:
                n_runs = 30
            if max_workers is None:
                max_workers = mp.cpu_count()

        # Save configuration before running
        self.save_config(n_runs, max_workers)

        total_experiments = len(self.problems) * len(self.algorithms) * n_runs
        timing_records = []

        # Display experiment configuration
        print(f"=" * 60)
        print("🚀🚀🚀 Starting Batch Experiment (Parallel Mode)! 🚀🚀🚀")
        print(f"=" * 60)
        print(f"\n1️⃣ Number of problems: {len(self.problems)}")
        print(f"2️⃣ Number of algorithms: {len(self.algorithms)}")
        print(f"3️⃣ Number of independent runs: {n_runs}")
        print(f"🔢 Total experiments: {total_experiments}")
        print(f"⚙️ Max workers: {max_workers}\n")

        start_time = time.time()

        # Prepare all tasks for parallel execution
        tasks = []
        for problem_creator, problem_name, problem_params in self.problems:
            for algo_class, algo_name, algo_params in self.algorithms:
                for run_id in range(1, n_runs + 1):
                    # Generate unique filename for each experiment
                    file_name = f"{algo_name}_{problem_name}_{run_id}"
                    save_path = os.path.join(self.base_path, algo_name)

                    task = {
                        'algo_class': algo_class,
                        'algo_name': algo_name,
                        'problem_creator': problem_creator,
                        'problem_name': problem_name,
                        'problem_params': problem_params,
                        'run_id': run_id,
                        'algo_params': algo_params,
                        'save_path': save_path,
                        'file_name': file_name
                    }
                    tasks.append(task)

        # Execute experiments in parallel using process pool
        completed_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the process pool
            future_to_task = {
                executor.submit(self._run_single_experiment, task): task
                for task in tasks
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    timing_records.append(result)
                    completed_count += 1

                    # Display progress information if verbose mode is enabled
                    if verbose:
                        progress = (completed_count / total_experiments) * 100
                        if completed_count % max(1, total_experiments // 100) == 0:
                            print(f"⏳ Progress: {completed_count}/{total_experiments} ({progress:.1f}%)")

                except Exception as e:
                    # Handle task execution failures
                    print(f"Task failed with exception: {e}")
                    timing_records.append({
                        'Algorithm': task['algo_name'],
                        'Problem': task['problem_name'],
                        'Run': task['run_id'],
                        'Filename': task['file_name'],
                        'Time(s)': 0.0,
                        'Status': "Failed",
                        'Error': str(e)
                    })
                    completed_count += 1

        # Calculate total execution time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Generate and save timing summary CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"time_summary_{timestamp}.csv"
        csv_path = os.path.join(self.base_path, csv_filename)

        self._save_timing_summary(timing_records, csv_path)

        # Display final summary
        print(f"\n⏰ Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        print(f"💥 Parallel speedup: {total_experiments / max_workers / (elapsed_time / 60):.2f}x")
        print(f"📊 Timing summary saved to: {csv_path}\n")
        print(f"=" * 60)
        print(f"🎉🎉🎉 All Experiments Completed! 🎉🎉🎉")
        print(f"=" * 60)
        print("\n")

    def _save_timing_summary(self, timing_records: List[Dict], csv_path: str):
        """
        Save timing summary to CSV file

        Args:
            timing_records: List of timing records from all experiments
            csv_path: Path to save the CSV file
        """
        if not timing_records:
            print("Warning: No timing records to save.")
            return

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Algorithm', 'Problem', 'Run', 'Filename', 'Time(s)', 'Status', 'Error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(timing_records)

        except Exception as e:
            print(f"Error saving timing summary: {str(e)}")


# Usage example and demonstration
if __name__ == "__main__":
    from Algorithms.STSO.GA import GA
    from Algorithms.STSO.PSO import PSO
    from Algorithms.STSO.DE import DE
    from Problems.MTSO.cec17_mtso import CEC17MTSO

    # ========== Method 1: Create and run experiments normally (auto-save config) ==========

    # Create batch experiment instance with folder clearing enabled
    # batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    # Add benchmark problems using creator functions
    # cec17mtso = CEC17MTSO()
    # batch_exp.add_problem(cec17mtso.P1, 'P1')
    # batch_exp.add_problem(cec17mtso.P2, 'P2')

    # Configure algorithm parameters
    # n = 100
    # max_nfes = 10000
    # disable_tqdm = True

    # Add optimization algorithm classes with their parameters
    # batch_exp.add_algorithm(GA, 'GA', n=n, max_nfes=max_nfes, disable_tqdm=disable_tqdm)
    # batch_exp.add_algorithm(DE, 'DE', n=n, max_nfes=max_nfes, disable_tqdm=disable_tqdm)
    # batch_exp.add_algorithm(PSO, 'PSO', n=n, max_nfes=max_nfes, disable_tqdm=disable_tqdm)

    # Execute experiments with parallel processing
    # batch_exp.run(n_runs=5, verbose=True, max_workers=6)

    # ========== Method 2: Load configuration from file and run experiments ==========

    # Load configuration and run
    # batch_exp2 = BatchExperiment.from_config('./Data/experiment_config.yaml')
    # batch_exp2.run()

    # Or override specific settings
    # batch_exp2.run(n_runs=10, max_workers=8)
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
from Methods.mtop import MTOP


class MO_SCP:
    """
    Implementation of Multi-Objective Sensor Coverage Problem (MO_SCP) for
    Multi-Task Multi-Objective Optimization.

    This problem extends the single-objective SCP to bi-objective optimization,
    balancing coverage maximization against sensor deployment costs. Each task
    corresponds to a different number of sensors with variable-length dimensions.

    Objectives:
    - f1: Inverted coverage percentage (minimize uncovered area)
    - f2: Total sensor cost (number of sensors + sensing costs)

    References
    ----------
    [1] Y. Li et al., "Transfer Search Directions Among Decomposed Subtasks for
        Evolutionary Multitasking in Multiobjective Optimization," GECCO, 2024.
    [2] Y. Li et al., "Evolutionary Competitive Multiobjective Multitasking:
        One-Pass Optimization of Heterogeneous Pareto Solutions," IEEE TEVC, 2024.

    Parameters
    ----------
    mat_path : str, optional
        Path to the MAT file containing target points matrix A
        (default: 'Problems/RWP/MO_SCP/SCP_Adata2.mat')

    Attributes
    ----------
    A : ndarray
        Target points to be covered, shape (n_points, 2)
    """

    def __init__(self, mat_path='../Problems/RWP/MO_SCP/SCP_Adata2.mat'):
        """
        Initialize MO_SCP problem.

        Parameters
        ----------
        mat_path : str
            Path to SCP_Adata2.mat file
        """
        self.mat_path = mat_path

        # Load target points data
        data = scipy.io.loadmat(mat_path)
        self.A = data['A']  # Target points matrix

    def P1(self, Nmin=28, task_num=5, gap=1) -> MTOP:
        """
        Generates MO_SCP Problem 1: Multi-Objective Sensor Coverage with uniform gap.

        Creates multiple tasks with different numbers of sensors, where sensor
        counts increase uniformly by a fixed gap.

        Parameters
        ----------
        Nmin : int, optional
            Minimum number of sensors (default: 28)
        task_num : int, optional
            Number of tasks to create (default: 5)
        gap : int, optional
            Gap between consecutive tasks' sensor numbers (default: 1)

        Task Structure:
        - T_i: 2-objective, (Nmin + gap*(i-1))*3 dimensional
          * Decision variables: [x1, y1, r1, ..., xk, yk, rk]
          * x, y: Sensor positions in [-1, 1]
          * r: Sensing radii in [0.1, 0.25]
          * Objective 1 (f1): Inverted coverage percentage (0-100)
          * Objective 2 (f2): Total cost = sensor_count + 10*sum(r²)

        - Relationship: Tasks with increasing sensor numbers test transfer
          learning across different problem scales.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        problem = MTOP()

        # Calculate sample range for coverage tolerance
        rsample = 2.0 / (self.A.shape[0] - 1)

        for t in range(task_num):
            num_sensors = Nmin + gap * t
            dim = num_sensors * 3

            # Create bounds
            lb = -np.ones(dim)
            ub = np.ones(dim)

            # Radius bounds: every 3rd variable
            radius_indices = np.arange(2, dim, 3)
            lb[radius_indices] = 0.1
            ub[radius_indices] = 0.25

            # Create task function with closure
            def create_task_function(k, rs):
                def task_func(x):
                    return self._evaluate_moscp_p1(x, k, rs)

                return task_func

            task_function = create_task_function(num_sensors, rsample)

            problem.add_task(
                task_function,
                dim=dim,
                lower_bound=lb,
                upper_bound=ub
            )

        return problem

    def P2(self, Nmin=25, task_num=4, gap=3) -> MTOP:
        """
        Generates MO_SCP Problem 2: Multi-Objective Sensor Coverage with larger gap.

        Creates multiple tasks with different numbers of sensors, where sensor
        counts increase with a larger gap. The second objective includes an
        additional coupling term for increased task correlation.

        Parameters
        ----------
        Nmin : int, optional
            Minimum number of sensors (default: 25)
        task_num : int, optional
            Number of tasks to create (default: 4)
        gap : int, optional
            Gap between consecutive tasks' sensor numbers (default: 3)

        Task Structure:
        - T_i: 2-objective, (Nmin + gap*(i-1))*3 dimensional
          * Decision variables: [x1, y1, r1, ..., xk, yk, rk]
          * x, y: Sensor positions in [-1, 1]
          * r: Sensing radii in [0.1, 0.25]
          * Objective 1 (f1): Inverted coverage percentage (0-100)
          * Objective 2 (f2): Total cost + f1/10 (coupled objectives)

        - Relationship: Larger gaps between tasks with objective coupling
          to test transfer learning in more heterogeneous scenarios.

        Returns
        -------
        MTOP
            A Multi-Task Multi-Objective Optimization Problem instance.
        """
        problem = MTOP()

        # Calculate sample range for coverage tolerance
        rsample = 2.0 / (self.A.shape[0] - 1)

        for t in range(task_num):
            num_sensors = Nmin + gap * t
            dim = num_sensors * 3

            # Create bounds
            lb = -np.ones(dim)
            ub = np.ones(dim)

            # Radius bounds: every 3rd variable
            radius_indices = np.arange(2, dim, 3)
            lb[radius_indices] = 0.1
            ub[radius_indices] = 0.25

            # Create task function with closure
            def create_task_function(k, rs):
                def task_func(x):
                    return self._evaluate_moscp_p2(x, k, rs)

                return task_func

            task_function = create_task_function(num_sensors, rsample)

            problem.add_task(
                task_function,
                dim=dim,
                lower_bound=lb,
                upper_bound=ub
            )

        return problem

    def _evaluate_moscp_p1(self, var, num_sensors, rsample):
        """
        Evaluate MO_SCP Problem 1 objectives.

        Parameters
        ----------
        var : array-like, shape (n_samples, dim) or (dim,)
            Decision variables
        num_sensors : int
            Number of sensors
        rsample : float
            Sample range for coverage tolerance

        Returns
        -------
        objs : ndarray, shape (n_samples, 2)
            [f1: inverted coverage %, f2: total cost]
        """
        var = np.atleast_2d(var)
        n_samples = var.shape[0]
        dim = num_sensors * 3

        b = 10  # Sensing cost weight
        c0 = 1  # Sensor count weight

        objs = np.zeros((n_samples, 2))

        for i in range(n_samples):
            x = var[i, :dim]

            # Reshape to (num_sensors, 3)
            sensors = x.reshape(num_sensors, 3)
            positions = sensors[:, :2]
            radii = sensors[:, 2]

            # Calculate distances
            distances = cdist(self.A, positions)

            # Check coverage with tolerance
            is_covered = (distances + rsample) <= radii.reshape(1, -1)
            max_is_covered = np.max(is_covered, axis=1)

            # Calculate coverage ratio
            coverage_ratio = np.sum(max_is_covered) / len(self.A)

            # Objective 1: Inverted coverage percentage
            f1 = 100 * (1 - coverage_ratio)

            # Objective 2: Total cost
            f2 = c0 * num_sensors + b * np.sum(radii ** 2)

            objs[i] = [f1, f2]

        return objs

    def _evaluate_moscp_p2(self, var, num_sensors, rsample):
        """
        Evaluate MO_SCP Problem 2 objectives (with objective coupling).

        Parameters
        ----------
        var : array-like, shape (n_samples, dim) or (dim,)
            Decision variables
        num_sensors : int
            Number of sensors
        rsample : float
            Sample range for coverage tolerance

        Returns
        -------
        objs : ndarray, shape (n_samples, 2)
            [f1: inverted coverage %, f2: total cost + f1/10]
        """
        var = np.atleast_2d(var)
        n_samples = var.shape[0]
        dim = num_sensors * 3

        b = 10  # Sensing cost weight
        c0 = 1  # Sensor count weight

        objs = np.zeros((n_samples, 2))

        for i in range(n_samples):
            x = var[i, :dim]

            # Reshape to (num_sensors, 3)
            sensors = x.reshape(num_sensors, 3)
            positions = sensors[:, :2]
            radii = sensors[:, 2]

            # Calculate distances
            distances = cdist(self.A, positions)

            # Check coverage with tolerance
            is_covered = (distances + rsample) <= radii.reshape(1, -1)
            max_is_covered = np.max(is_covered, axis=1)

            # Calculate coverage ratio
            coverage_ratio = np.sum(max_is_covered) / len(self.A)

            # Objective 1: Inverted coverage percentage
            f1 = 100 * (1 - coverage_ratio)

            # Objective 2: Total cost with coupling term
            f2 = c0 * num_sensors + b * np.sum(radii ** 2) + f1 / 10

            objs[i] = [f1, f2]

        return objs


SETTINGS = {
    'metric': 'HV',
    'P1': {'all_tasks': [80, 40]},
    'P2': {'all_tasks': [80, 40]},
}
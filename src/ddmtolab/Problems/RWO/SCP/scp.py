import numpy as np
import scipy.io
import pkgutil
import io
from scipy.spatial.distance import cdist
from ddmtolab.Methods.mtop import MTOP


class SCP:
    """
    Implementation of the Sensor Coverage Problem (SCP) for Multi-Task Optimization.

    This problem involves optimizing sensor placements to maximize coverage of target
    points while minimizing the number of sensors and their sensing radii. Each task
    corresponds to a different number of sensors (variable-length optimization).

    The problem optimizes sensor positions (x, y) and sensing radii (r) to:
    - Maximize coverage of target points
    - Minimize number of sensors
    - Minimize sensing costs (proportional to r²)

    References
    ----------
    [1] M. L. Ryerkerk et al., "Solving Metameric Variable-length Optimization Problems Using Genetic Algorithms," Genetic Programming and Evolvable Machines, vol. 18, no. 2, pp. 247-277, 2017.
    [2] G. Li et al., "Evolutionary Competitive Multitasking Optimization," IEEE Trans. Evol. Comput., 2022.

    Attributes
    ----------
    Nmin : int
        Minimum number of sensors (default: 25)
    Nmax : int
        Maximum number of sensors (default: 35)
    A : ndarray
        Target points to be covered, shape (n_points, 2)
    data_dir : str
        The directory path for problem data files.
    """

    def __init__(self, Nmin=25, Nmax=35):
        """
        Initialize SCP problem.

        Parameters
        ----------
        Nmin : int
            Minimum number of sensors
        Nmax : int
            Maximum number of sensors
        """
        self.Nmin = Nmin
        self.Nmax = Nmax
        self.data_dir = 'data_scp'

        # Load target points data
        data_bytes = pkgutil.get_data('ddmtolab.Problems.RWO',
                                      f'{self.data_dir}/SCP_Adata.mat')
        mat_file = io.BytesIO(data_bytes)
        data = scipy.io.loadmat(mat_file)
        self.A = data['A']  # Target points matrix

    def P1(self) -> MTOP:
        """
        Generates SCP Problem 1: Multi-Task Sensor Coverage Optimization.

        Creates tasks for different numbers of sensors from Nmin to Nmax.
        Each task optimizes sensor placements and radii.

        Task Structure:
        - T_i (i sensors): 1-objective, (3*i)-dimensional
          * Decision variables: [x1, y1, r1, x2, y2, r2, ..., xi, yi, ri]
          * x, y: Sensor positions in [-1, 1]
          * r: Sensing radii in [0.1, 0.25]
          * Objective: Weighted sum of:
            - Coverage penalty: 1000 * (1 - coverage_ratio)
            - Sensor count penalty: 1 * number_of_sensors
            - Sensing cost: 10 * sum(r²)

        - Relationship: Variable-length tasks with increasing complexity.
          Tests transfer learning across different problem dimensions.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance with (Nmax - Nmin + 1) tasks.
        """
        problem = MTOP()

        # Create tasks for different numbers of sensors
        for k in range(self.Nmin, self.Nmax + 1):
            dim = k * 3  # Each sensor has 3 variables: x, y, r

            # Create bounds
            lb = -np.ones(dim)
            ub = np.ones(dim)

            # Radius bounds: every 3rd variable (indices 2, 5, 8, ...)
            radius_indices = np.arange(2, dim, 3)
            lb[radius_indices] = 0.1
            ub[radius_indices] = 0.25

            # Create task function with closure over k
            def create_task_function(num_sensors):
                def task_func(x):
                    return self._evaluate_scp(x, num_sensors)

                return task_func

            task_function = create_task_function(k)

            problem.add_task(
                task_function,
                dim=dim,
                lower_bound=lb,
                upper_bound=ub
            )

        return problem

    def _evaluate_scp(self, var, num_sensors):
        """
        Evaluate the Sensor Coverage Problem objective function.

        Parameters
        ----------
        var : array-like, shape (n_samples, dim) or (dim,)
            Decision variables representing sensor configurations
        num_sensors : int
            Number of sensors (k) in this task

        Returns
        -------
        objs : ndarray, shape (n_samples,)
            Objective values for each sample
        """
        var = np.atleast_2d(var)
        n_samples = var.shape[0]
        dim = num_sensors * 3

        # Problem parameters
        a = 1000  # Coverage penalty weight
        b = 10  # Sensing cost weight
        c0 = 1  # Sensor count weight

        objs = np.zeros(n_samples)

        for i in range(n_samples):
            x = var[i, :dim]  # Ensure correct dimension

            # Reshape to (num_sensors, 3): each row is [x_i, y_i, r_i]
            sensors = x.reshape(num_sensors, 3)

            # Extract positions and radii
            positions = sensors[:, :2]  # (num_sensors, 2)
            radii = sensors[:, 2]  # (num_sensors,)

            # Calculate distances from all target points to all sensors
            # distances: (n_targets, num_sensors)
            distances = cdist(self.A, positions)

            # Check coverage: is each target covered by each sensor?
            # is_covered: (n_targets, num_sensors)
            is_covered = distances <= radii.reshape(1, -1)

            # Check if each target is covered by at least one sensor
            max_is_covered = np.max(is_covered, axis=1)  # (n_targets,)

            # Calculate coverage ratio
            coverage_ratio = np.sum(max_is_covered) / len(self.A)

            # Calculate objective
            coverage_penalty = a * (1 - coverage_ratio)
            sensor_count_penalty = c0 * num_sensors
            sensing_cost = b * np.sum(radii ** 2)

            objs[i] = coverage_penalty + sensor_count_penalty + sensing_cost

        return objs
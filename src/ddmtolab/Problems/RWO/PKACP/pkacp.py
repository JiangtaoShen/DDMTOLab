import numpy as np
import os
import scipy.io
from scipy.cluster.vq import kmeans2
from ddmtolab.Methods.mtop import MTOP


class PKACP:
    """
    Implementation of Planar Kinematic Arm Control Problem (PKACP) for
    Multi-Task/Many-Task Single-Objective Optimization.

    This problem involves controlling a planar kinematic arm with multiple joints
    to reach a target position. Each task has different constraints on the maximum
    angular range and link lengths, creating diverse optimization landscapes.

    The objective is to minimize the Euclidean distance between the end effector
    position and the target position (0.5, 0.5).

    References
    ----------
    [1] Y. Jiang et al., "A Bi-Objective Knowledge Transfer Framework for
        Evolutionary Many-Task Optimization," IEEE TEVC, 2022.
    [2] H. Xu et al., "Evolutionary Multi-Task Optimization with Adaptive
        Knowledge Transfer," IEEE TEVC, 2021.

    Attributes
    ----------
    data_dir : str
        Directory for task parameter files (in user's home directory)
    """

    def __init__(self):
        """Initialize PKACP problem."""
        # Use user's home directory for storing generated parameters
        home_dir = os.path.expanduser('~')
        self.data_dir = os.path.join(home_dir, '.ddmtolab', 'PKACP_data')

        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def P1(self, task_num=20, dim=20) -> MTOP:
        """
        Generates PKACP Problem 1: Planar Kinematic Arm Control.

        Creates multiple tasks with different maximum angular ranges and link
        lengths. Each task optimizes joint angles to minimize the distance
        between the end effector and target position.

        Parameters
        ----------
        task_num : int, optional
            Number of tasks to create (default: 20)
        dim : int, optional
            Number of joints (dimensionality) for each task (default: 20)

        Task Structure:
        - T_i: 1-objective, dim-dimensional
          * Decision variables: Joint angles in [0, 1]
          * These are scaled to actual angular ranges based on task parameters
          * Objective: Euclidean distance to target (0.5, 0.5)

        Task Parameters:
        - Amax: Maximum angular range for the task
        - Lmax: Maximum total link length for the task
        - These parameters are generated using CVT (Centroidal Voronoi Tessellation)
          to create diverse but structured task variations

        - Relationship: Tasks differ in kinematic constraints, testing transfer
          learning across different arm configurations.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance with task_num tasks.
        """
        # Generate or load task parameters
        task_para = self._generate_task_parameters(task_num, dim)

        problem = MTOP()

        for t in range(task_num):
            # Extract task-specific parameters
            Amax = task_para[t, 0]  # Maximum angular range
            Lmax = task_para[t, 1]  # Maximum link length

            # All joints have angles in [0, 1]
            lb = np.zeros(dim)
            ub = np.ones(dim)

            # Create task function with closure
            def create_task_function(amax, lmax, d):
                def task_func(x):
                    return self._evaluate_pkacp(x, amax, lmax, d)

                return task_func

            task_function = create_task_function(Amax, Lmax, dim)

            problem.add_task(
                task_function,
                dim=dim,
                lower_bound=lb,
                upper_bound=ub
            )

        return problem

    def _generate_task_parameters(self, task_num, dim):
        """
        Generate or load task parameters using CVT.

        Parameters
        ----------
        task_num : int
            Number of tasks
        dim : int
            Dimensionality

        Returns
        -------
        task_para : ndarray, shape (task_num, 2)
            Task parameters [Amax, Lmax] for each task
        """
        # Check if parameters already exist
        file_name = os.path.join(
            self.data_dir,
            f'cvt_d{dim}_nt{task_num}.mat'
        )

        if os.path.exists(file_name):
            # Load existing parameters
            data = scipy.io.loadmat(file_name)
            task_para = data['task_para']
        else:
            # Generate new parameters using CVT (k-means approximation)
            samples = 50 * task_num
            x = np.random.rand(samples, 2)

            # Use k-means to approximate CVT
            task_para, _ = kmeans2(x, task_num, minit='points')

            # Save parameters
            scipy.io.savemat(file_name, {'task_para': task_para})

        return task_para

    def _evaluate_pkacp(self, angles_var, Amax, Lmax, num_joints):
        """
        Evaluate the kinematic arm control objective.

        Parameters
        ----------
        angles_var : array-like, shape (n_samples, dim) or (dim,)
            Joint angles in [0, 1]
        Amax : float
            Maximum angular range for this task
        Lmax : float
            Maximum total link length for this task
        num_joints : int
            Number of joints

        Returns
        -------
        objs : ndarray, shape (n_samples,)
            Distance to target for each sample
        """
        angles_var = np.atleast_2d(angles_var)
        n_samples = angles_var.shape[0]

        # Calculate angular range and link lengths
        angular_range = Amax / num_joints
        lengths = np.ones(num_joints) * Lmax / num_joints

        # Target position
        target = np.array([0.5, 0.5])

        objs = np.zeros(n_samples)

        for i in range(n_samples):
            angles = angles_var[i, :]

            # Convert to actual commands (centered around 0)
            command = (angles - 0.5) * angular_range * np.pi * 2

            # Forward kinematics
            ef = self._forward_kinematics(command, lengths)

            # Calculate distance to target
            fitness = np.sqrt(np.sum((ef - target) ** 2))
            objs[i] = fitness

        return objs

    def _forward_kinematics(self, joint_angles, lengths):
        """
        Calculate end effector position using forward kinematics.

        Parameters
        ----------
        joint_angles : ndarray, shape (n_joints,)
            Joint angles in radians
        lengths : ndarray, shape (n_joints,)
            Link lengths

        Returns
        -------
        joint_xy : ndarray, shape (2,)
            End effector position (x, y)
        """
        mat = np.eye(4)

        # Add zero angle at the end
        angles = np.append(joint_angles, 0)
        n_dofs = len(angles)

        # Prepend zero length
        lengths = np.insert(lengths, 0, 0)

        joint_xy = np.zeros(2)

        for i in range(n_dofs):
            # Create transformation matrix for this joint
            c = np.cos(angles[i])
            s = np.sin(angles[i])

            m = np.array([
                [c, -s, 0, lengths[i]],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            # Accumulate transformations
            mat = mat @ m

            # Get position of current joint
            v = mat @ np.array([0, 0, 0, 1])
            joint_xy = v[:2]

        return joint_xy
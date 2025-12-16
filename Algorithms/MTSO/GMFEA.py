"""
Generalized Multifactorial Evolutionary Algorithm (G-MFEA)

This module implements G-MFEA for expensive multi-task optimization with adaptive knowledge transfer.

References
----------
    [1] Ding, Jinliang, et al. "Generalized multitasking for evolutionary optimization of expensive
        problems." IEEE Transactions on Evolutionary Computation 23.1 (2017): 44-58.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.19
Version: 1.0
"""
import time
from tqdm import tqdm
from Algorithms.MTSO.MFEA import mfea_selection
from Methods.Algo_Methods.algo_utils import *


class GMFEA:
    """
    Generalized Multifactorial Evolutionary Algorithm for expensive multi-task optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'n_objs': '1',
        'n_cons': '0',
        'n': 'equal',
        'max_nfes': 'equal',
        'expensive': 'False',
        'knowledge_transfer': 'True'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        """
        Get algorithm information.

        Parameters
        ----------
        print_info : bool, optional
            Whether to print information (default: True)

        Returns
        -------
        dict
            Algorithm information dictionary
        """
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, mu=0.4, phi=0.1, theta=0.02, scale_factor=1.25,
                 save_data=True, save_path='./TestData', name='G-MFEA_test', disable_tqdm=True):
        """
        Initialize G-MFEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.3)
        mu : float, optional
            Percentage of best solutions to estimate current optimums (default: 0.4)
        phi : float, optional
            Threshold value to start decision variable translation strategy (default: 0.1)
        theta : float, optional
            Frequency of calculating translation direction (default: 0.02)
        scale_factor : float, optional
            Scale factor for translation of decision variables (default: 1.25)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'G-MFEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.mu = mu
        self.phi = phi
        self.theta = theta
        self.scale_factor = scale_factor
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the G-MFEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize population and evaluate for each task
        decs = initialization(problem, n)
        objs, _ = evaluation(problem, decs)
        nfes = n * nt
        gen = 1

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        all_decs, all_objs = init_history(decs, objs)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:
            # Translate decision variables toward population center
            trans_decs, trans_vectors = decs_translation(decs, objs, nfes, max_nfes, dims, n, gen, self.phi,
                                                         self.theta, self.mu, self.scale_factor)

            # Transform populations to unified search space
            pop_decs = space_transfer(problem, trans_decs, type='uni')
            pop_objs = objs

            # Merge populations from all tasks into single arrays
            pop_decs, pop_objs, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_sfs)

            off_decs = np.zeros_like(pop_decs)
            off_objs = np.zeros_like(pop_objs)
            off_sfs = np.zeros_like(pop_sfs)

            # Randomly pair individuals for assortative mating
            shuffled_index = np.random.permutation(pop_decs.shape[0])

            for i in range(0, len(shuffled_index), 2):
                p1 = shuffled_index[i]
                p2 = shuffled_index[i + 1]
                sf1 = pop_sfs[p1].item()
                sf2 = pop_sfs[p2].item()

                # Shuffle dimensions to align heterogeneous tasks
                p1_dec, p2_dec, l1, l2 = dimension_shuffling(p1, p2, sf1, sf2, dims, pop_decs, pop_sfs)

                # Cross-task transfer: crossover if same task or rmp condition met
                if sf1 == sf2 or np.random.rand() < self.rmp:
                    off_dec1, off_dec2 = crossover(p1_dec, p2_dec, mu=2.0)
                    off_sfs[i] = np.random.choice([sf1, sf2])
                    off_sfs[i + 1] = sf1 if off_sfs[i] == sf2 else sf2
                else:
                    # No transfer: mutate within own task
                    off_dec1 = mutation(p1_dec, mu=5.0)
                    off_dec2 = mutation(p2_dec, mu=5.0)
                    off_sfs[i] = sf1
                    off_sfs[i + 1] = sf2

                # Inverse transformation: restore dimensions and translate back
                off_sf1 = off_sfs[i]
                off_sf2 = off_sfs[i + 1]
                off_dec1, off_dec2 = convert_to_original_decision_space(off_dec1, off_dec2, off_sf1, off_sf2, sf1,
                                                                        sf2, dims, l1, l2, trans_vectors)

                off_decs[i, :] = off_dec1
                off_decs[i + 1, :] = off_dec2

                # Trim to task dimensionality and evaluate offspring
                task_idx1 = off_sf1.item()
                task_idx2 = off_sf2.item()
                off_dec1_trimmed = off_decs[i, :dims[task_idx1]]
                off_dec2_trimmed = off_decs[i + 1, :dims[task_idx2]]
                off_objs[i, :], _ = evaluation_single(problem, off_dec1_trimmed, task_idx1)
                off_objs[i + 1, :], _ = evaluation_single(problem, off_dec2_trimmed, task_idx2)

            # Merge parents and offspring populations
            pop_decs, pop_objs, pop_sfs = vstack_groups((pop_decs, off_decs), (pop_objs, off_objs),
                                                        (pop_sfs, off_sfs))

            # Environmental selection: keep best n individuals per task
            pop_decs, pop_objs, pop_sfs = gmfea_selection(pop_decs, pop_objs, pop_sfs, n, nt)

            # Transform back to native search space
            decs = space_transfer(problem, pop_decs, type='real')
            objs = pop_objs

            nfes += n * nt
            pbar.update(n * nt)
            gen += 1

            append_history(all_decs, decs, all_objs, pop_objs)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=max_nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def decs_translation(decs, objs, nfes, max_nfes, dims, n, gen, phi, theta, mu, scale_factor):
    """
    Translate decision variables toward the center of the decision space.

    Parameters
    ----------
    decs : list[np.ndarray]
        Decision variables for all tasks, length nt, each of shape (n, d_i)
    objs : list[np.ndarray]
        Objective values for all tasks, length nt, each of shape (n, n_obj)
    nfes : int
        Current number of function evaluations
    max_nfes : int
        Maximum number of function evaluations
    dims : list[int]
        Dimensions of each task, length nt
    n : int
        Population size
    gen : int
        Current generation
    phi : float
        Threshold ratio to activate translation
    theta : float
        Interval ratio for translation frequency
    mu : float
        Ratio of top individuals to use for calculating mean
    scale_factor : float
        Scaling factor for translation vector

    Returns
    -------
    trans_decs : list[np.ndarray]
        Translated decision variables, length nt, each of shape (n, d_i)
    trans_vectors : list[np.ndarray]
        Translation vectors used for each task, length nt, each of shape (d_i,)

    Notes
    -----
    Translation moves populations toward the center point (0.5) of the normalized decision space.
    The translation vector is computed as: d_i = scale_factor * α * (cp_i - m_i),
    where α = (nfes/max_nfes)² is an adaptive coefficient, cp_i is the center point,
    and m_i is the mean position of top μ% individuals.
    """
    nt = len(decs)

    # Calculate adaptive coefficient based on evolution progress
    alpha = (nfes / max_nfes) ** 2

    # Calculate translation interval (how often to apply translation)
    interval = max(1, round(theta * max_nfes / (n * nt)))

    # Number of top individuals to consider for calculating centroid
    num = round(n * mu)

    trans_decs, trans_vectors = [], []

    # Apply translation only after phi * max_nfes evaluations and at specified intervals
    if nfes >= phi * max_nfes and gen % interval == 0:
        for i in range(nt):
            # Select top mu% individuals based on objective values
            indices = np.argsort(objs[i][:, 0])[:num]
            top_mu_decs = decs[i][indices]

            # Calculate mean position (centroid) of top individuals
            m_i = np.mean(top_mu_decs, axis=0)

            # Center point of decision space (0.5 for normalized [0,1] space)
            cp_i = np.ones(dims[i]) * 0.5

            # Calculate translation vector: move from current centroid toward center
            d_i = scale_factor * alpha * (cp_i - m_i)
            trans_vectors.append(d_i)

            # Apply translation and clip to valid bounds [0, 1]
            trans_decs.append(np.clip(decs[i] + d_i, 0, 1))
    else:
        # No translation applied: return original decision variables
        trans_decs = copy.deepcopy(decs)
        trans_vectors = [np.zeros(dims[i]) for i in range(nt)]

    return trans_decs, trans_vectors


def dimension_shuffling(p1, p2, sf1, sf2, dims, pop_decs, pop_sfs):
    """
    Handle dimension mismatch between parents using random dimension shuffling.

    Parameters
    ----------
    p1 : int
        Index of first parent
    p2 : int
        Index of second parent
    sf1 : int
        Skill factor (task ID) of first parent
    sf2 : int
        Skill factor (task ID) of second parent
    dims : list[int]
        Dimensions of each task, length nt
    pop_decs : np.ndarray
        Population decision variables of shape (pop_size, d_max)
    pop_sfs : np.ndarray
        Population skill factors of shape (pop_size, 1)

    Returns
    -------
    p1_dec : np.ndarray
        Processed decision variables of first parent of shape (d_max,)
    p2_dec : np.ndarray
        Processed decision variables of second parent of shape (d_max,)
    l1 : np.ndarray or None
        Shuffling indices used for p1, None if no shuffling needed
    l2 : np.ndarray or None
        Shuffling indices used for p2, None if no shuffling needed

    Notes
    -----
    When parents have different dimensionalities, the lower-dimensional parent borrows
    genetic material from a random individual of the higher-dimensional task. The parent's
    genes are placed into randomly selected dimensions via shuffling indices.
    """
    # Get dimensions of both parent tasks
    dim_1 = dims[sf1]
    dim_2 = dims[sf2]
    dim_max = max(dim_1, dim_2)

    # Extract decision variables for each parent in their native dimensions
    p1_dec = pop_decs[p1, :dim_1].copy()
    p2_dec = pop_decs[p2, :dim_2].copy()

    l1, l2 = None, None

    # Case 1: Parent 1 has lower dimension - needs dimension completion
    if dim_1 < dim_max:
        # Generate random permutation for dimension shuffling
        l1 = np.random.permutation(dim_max)

        # Borrow genetic material from a random individual of the higher-dimensional task
        candidates = np.where(pop_sfs.flatten() == sf2)[0]
        random_idx = np.random.choice(candidates)
        p_dec = pop_decs[random_idx, :dim_max].copy()

        # Place p1's genes into randomly selected dimensions
        p_dec[l1[:dim_1]] = p1_dec
        p1_dec = p_dec

    # Case 2: Parent 2 has lower dimension - needs dimension completion
    elif dim_2 < dim_max:
        # Generate random permutation for dimension shuffling
        l2 = np.random.permutation(dim_max)

        # Borrow genetic material from a random individual of the higher-dimensional task
        candidates = np.where(pop_sfs.flatten() == sf1)[0]
        random_idx = np.random.choice(candidates)
        p_dec = pop_decs[random_idx, :dim_max].copy()

        # Place p2's genes into randomly selected dimensions
        p_dec[l2[:dim_2]] = p2_dec
        p2_dec = p_dec

    return p1_dec, p2_dec, l1, l2


def convert_to_original_decision_space(off_dec1, off_dec2, off_sf1, off_sf2, sf1, sf2, dims, l1, l2, trans_vectors):
    """
    Convert offspring back to original task space by reversing shuffling and translation.

    Parameters
    ----------
    off_dec1 : np.ndarray
        First offspring decision variables of shape (d,)
    off_dec2 : np.ndarray
        Second offspring decision variables of shape (d,)
    off_sf1 : int
        Skill factor (task ID) of first offspring
    off_sf2 : int
        Skill factor (task ID) of second offspring
    sf1 : int
        Skill factor of first parent
    sf2 : int
        Skill factor of second parent
    dims : list[int]
        Dimensions of each task, length nt
    l1 : np.ndarray or None
        Shuffling indices used for parent 1, None if no shuffling
    l2 : np.ndarray or None
        Shuffling indices used for parent 2, None if no shuffling
    trans_vectors : list[np.ndarray]
        Translation vectors applied to each task, length nt

    Returns
    -------
    off_dec1 : np.ndarray
        First offspring in original space, padded to max dimension of shape (d_max,)
    off_dec2 : np.ndarray
        Second offspring in original space, padded to max dimension of shape (d_max,)

    Notes
    -----
    This function reverses the dimension shuffling by extracting genes at shuffled positions,
    subtracts the translation vectors, clips to valid bounds, and pads to maximum dimension.
    """
    # Reverse dimension shuffling for parent 1's task if it was applied
    if l1 is not None:
        dim_1 = dims[sf1]
        if off_sf1 == sf1:
            off_dec1 = off_dec1[l1[:dim_1]] - trans_vectors[sf1]
        if off_sf2 == sf1:
            off_dec2 = off_dec2[l1[:dim_1]] - trans_vectors[sf1]

    # Reverse dimension shuffling for parent 2's task if it was applied
    if l2 is not None:
        dim_2 = dims[sf2]
        if off_sf1 == sf2:
            off_dec1 = off_dec1[l2[:dim_2]] - trans_vectors[sf2]
        if off_sf2 == sf2:
            off_dec2 = off_dec2[l2[:dim_2]] - trans_vectors[sf2]

    # Clip values to valid bounds [0, 1]
    off_dec1 = np.clip(off_dec1, 0, 1)
    off_dec2 = np.clip(off_dec2, 0, 1)

    # Pad offspring to maximum dimension with zeros if needed
    max_dim = max(dims)
    if len(off_dec1) < max_dim:
        off_dec1 = np.pad(off_dec1, (0, max_dim - len(off_dec1)), mode='constant', constant_values=0)
    if len(off_dec2) < max_dim:
        off_dec2 = np.pad(off_dec2, (0, max_dim - len(off_dec2)), mode='constant', constant_values=0)

    return off_dec1, off_dec2


def gmfea_selection(all_decs, all_objs, all_sfs, n, nt):
    """
    Environmental selection for G-MFEA based on elitist strategy.

    Parameters
    ----------
    all_decs : np.ndarray
        Decision variable matrix of the combined population of shape (n_total, d_max).
        Contains solutions from all tasks in unified search space
    all_objs : np.ndarray
        Objective value matrix corresponding to all_decs of shape (n_total, 1).
        Each individual has been evaluated on its assigned task
    all_sfs : np.ndarray
        Skill factor array indicating task assignment for each individual of shape (n_total,).
        Values range from 0 to nt-1
    n : int
        Number of individuals to select per task (population size per task)
    nt : int
        Number of tasks in the multi-task optimization problem

    Returns
    -------
    pop_decs : List[np.ndarray]
        Selected decision variable matrices for each task, length nt, each of shape (n, d_max)
    pop_objs : List[np.ndarray]
        Selected objective value matrices for each task, length nt, each of shape (n, 1)
    pop_sfs : List[np.ndarray]
        Selected skill factor arrays for each task, length nt, each of shape (n,)

    Notes
    -----
    Selection is performed independently for each task by selecting the top-n individuals
    with minimum objective values among those assigned to that task.
    """
    pop_decs, pop_objs, pop_sfs = [], [], []

    # Process each task separately
    for i in range(nt):
        # Extract all individuals belonging to task i
        indices = np.where(all_sfs == i)[0]
        current_decs, current_objs, current_sfs = select_by_index(indices, all_decs, all_objs, all_sfs)

        # Select top-n individuals with minimum objective values
        indices_select = selection_elit(objs=current_objs, n=n)
        selected_decs, selected_objs, selected_sfs = select_by_index(indices_select, current_decs,
                                                                      current_objs, current_sfs)

        # Store selected individuals for this task
        pop_decs, pop_objs, pop_sfs = append_history(
            pop_decs, selected_decs,
            pop_objs, selected_objs,
            pop_sfs, selected_sfs
        )

    return pop_decs, pop_objs, pop_sfs
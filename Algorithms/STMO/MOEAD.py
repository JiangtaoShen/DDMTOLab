"""
Multi-objective Evolutionary Algorithm Based on Decomposition (MOEA/D)

This module implements MOEA/D for multi-objective optimization problems.

References
----------
    [1] Zhang, Qingfu, and Hui Li. "MOEA/D: A multiobjective evolutionary algorithm based on decomposition." \
        IEEE Transactions on Evolutionary Computation 11.6 (2007): 712-731.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.03
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from Methods.Algo_Methods.uniform_point import uniform_point
from Methods.Algo_Methods.algo_utils import *


class MOEAD:
    """
    Multi-objective Evolutionary Algorithm Based on Decomposition.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, decomp_type=1, muc=20.0, mum=15.0, save_data=True,
                 save_path='./TestData', name='MOEAD_test', disable_tqdm=True):
        """
        Initialize MOEA/D algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        decomp_type : int, optional
            Decomposition approach type (default: 1)
            1: PBI (Penalty-based Boundary Intersection)
            2: Tchebycheff
            3: Tchebycheff with normalization
            4: Modified Tchebycheff
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MOEAD_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.decomp_type = decomp_type
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MOEA/D.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        no = problem.n_objs
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate uniformly distributed weight vectors for each task
        W = []
        T = []  # Neighborhood size
        B = []  # Neighbor indices
        for i in range(nt):
            w_i, n = uniform_point(n_per_task[i], no[i])
            W.append(w_i)
            n_per_task[i] = n

            # Set neighborhood size to 10% of population
            T.append(int(np.ceil(n / 10)))

            # Detect the neighbors of each weight vector based on Euclidean distance
            distances = squareform(pdist(w_i))
            neighbors = np.argsort(distances, axis=1)[:, :T[i]]
            B.append(neighbors)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize ideal point for each task
        Z = []
        for i in range(nt):
            Z.append(np.min(objs[i], axis=0))

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for task_id in active_tasks:
                # Process each subproblem
                for subproblem_id in range(n_per_task[task_id]):
                    # Select parents from neighborhood
                    P = B[task_id][subproblem_id]
                    np.random.shuffle(P)
                    parent_indices = P[:2]

                    # Generate offspring through crossover and mutation
                    parent_decs = decs[task_id][parent_indices, :]
                    off_decs = ga_generation(parent_decs, muc=self.muc, mum=self.mum)

                    # Evaluate offspring (only one offspring per iteration)
                    off_obj, off_con = evaluation_single(problem, off_decs[:1], task_id)

                    # Update ideal point
                    Z[task_id] = np.minimum(Z[task_id], off_obj[0])

                    # Calculate scalar fitness values for neighbors
                    g_old = self._calculate_fitness(objs[task_id][P], cons[task_id][P],
                                                    W[task_id][P], Z[task_id],
                                                    objs[task_id], self.decomp_type)
                    g_new = self._calculate_fitness(off_obj, off_con,
                                                    W[task_id][P], Z[task_id],
                                                    objs[task_id], self.decomp_type)

                    # Update neighbors if offspring is better
                    CV_old = np.sum(np.maximum(0, cons[task_id][P]), axis=1)
                    CV_new = np.sum(np.maximum(0, off_con[0]))

                    # Update solutions: offspring replaces parent if it's better
                    # Better means: same constraint violation and better fitness, or less constraint violation
                    update_mask = (g_new < g_old) & (CV_old == CV_new) | (CV_old > CV_new)
                    update_indices = P[update_mask]

                    for idx in update_indices:
                        decs[task_id][idx] = off_decs[0]
                        objs[task_id][idx] = off_obj[0]
                        cons[task_id][idx] = off_con[0]

                nfes_per_task[task_id] += n_per_task[task_id]
                pbar.update(n_per_task[task_id])

                append_history(all_decs[task_id], decs[task_id], all_objs[task_id],
                               objs[task_id], all_cons[task_id], cons[task_id])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _calculate_fitness(self, objs, cons, weights, ideal_point, population_objs, decomp_type):
        """
        Calculate fitness values using the specified decomposition approach.

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (N, M) or (M,)
        cons : np.ndarray
            Constraint values of shape (N, K) or (K,)
        weights : np.ndarray
            Weight vectors of shape (N, M)
        ideal_point : np.ndarray
            Ideal point of shape (M,)
        population_objs : np.ndarray
            Population objective values of shape (pop_size, M) for normalization
        decomp_type : int
            Decomposition approach type

        Returns
        -------
        fitness : np.ndarray
            Fitness values of shape (N,) or scalar
        """
        # Handle single solution case
        if objs.ndim == 1:
            objs = objs.reshape(1, -1)
            single_solution = True
        else:
            single_solution = False

        N, M = objs.shape

        # Translate objectives by ideal point
        translated_objs = objs - ideal_point

        if decomp_type == 1:
            # PBI (Penalty-based Boundary Intersection) approach
            norm_w = np.linalg.norm(weights, axis=1, keepdims=True)
            norm_obj = np.linalg.norm(translated_objs, axis=1, keepdims=True)

            # Avoid division by zero
            norm_w = np.maximum(norm_w, 1e-10)
            norm_obj = np.maximum(norm_obj, 1e-10)

            # Calculate cosine of angle
            cosine = np.sum(translated_objs * weights, axis=1, keepdims=True) / (norm_w * norm_obj)
            cosine = np.clip(cosine, -1, 1)

            # Calculate d1 (distance along weight vector) and d2 (perpendicular distance)
            d1 = norm_obj * cosine
            d2 = norm_obj * np.sqrt(1 - cosine ** 2)

            # Penalty parameter
            theta = 5.0
            fitness = d1.flatten() + theta * d2.flatten()

        elif decomp_type == 2:
            # Tchebycheff approach
            fitness = np.max(np.abs(translated_objs) * weights, axis=1)

        elif decomp_type == 3:
            # Tchebycheff approach with normalization
            max_objs = np.max(population_objs, axis=0)
            range_objs = max_objs - ideal_point
            range_objs = np.maximum(range_objs, 1e-10)  # Avoid division by zero

            normalized_objs = np.abs(translated_objs) / range_objs
            fitness = np.max(normalized_objs * weights, axis=1)

        elif decomp_type == 4:
            # Modified Tchebycheff approach
            weights_safe = np.maximum(weights, 1e-10)  # Avoid division by zero
            fitness = np.max(np.abs(translated_objs) / weights_safe, axis=1)

        else:
            raise ValueError(f"Invalid decomposition type: {decomp_type}. Must be 1, 2, 3, or 4.")

        if single_solution:
            # Return repeated fitness for each neighbor
            return np.full(weights.shape[0], fitness[0])

        return fitness
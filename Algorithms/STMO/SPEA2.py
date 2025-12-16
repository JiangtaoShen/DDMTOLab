"""
Strength Pareto Evolutionary Algorithm 2 (SPEA2)

This module implements SPEA2 for multi-objective optimization problems.

References
----------
    [1] Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the Strength Pareto
        Evolutionary Algorithm For Multiobjective Optimization. In Evolutionary Methods for Design,
        Optimization and Control with Applications to Industrial Problems. Proceedings of the EUROGEN'2001.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.13
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from Methods.Algo_Methods.algo_utils import *


class SPEA2:
    """
    Strength Pareto Evolutionary Algorithm 2 for multi-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=15.0, epsilon=0, save_data=True,
                 save_path='./TestData', name='SPEA2_test', disable_tqdm=True):
        """
        Initialize SPEA2 algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        epsilon : float, optional
            Constraint epsilon value for epsilon-constraint method (default: 0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'SPEA2_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.muc = muc
        self.mum = mum
        self.epsilon = epsilon
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SPEA2 algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Calculate initial fitness for each task
        fitness = []
        for i in range(nt):
            cv = self._calculate_constraint_violations(cons[i])
            fitness_i = self._cal_fitness(objs[i], cv)
            fitness.append(fitness_i.copy())

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Environmental selection to get current population
                selected_indices, selected_fitness = self._spea2_selection(objs[i], cons[i], n_per_task[i])
                objs[i] = objs[i][selected_indices]
                decs[i] = decs[i][selected_indices]
                cons[i] = cons[i][selected_indices] if cons[i] is not None else None
                fitness[i] = selected_fitness

                # Parent selection via binary tournament based on fitness
                matingpool = tournament_selection(2, n_per_task[i], fitness[i])

                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i][matingpool, :], muc=self.muc, mum=self.mum)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs),
                                                          (cons[i], off_cons))

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _calculate_constraint_violations(self, cons):
        """
        Calculate constraint violations.

        Parameters
        ----------
        cons : ndarray or None
            Constraint values with shape (pop_size, n_cons)

        Returns
        -------
        cv : ndarray
            Constraint violation values with shape (pop_size,)
        """
        if cons is not None:
            # Python: np.sum(np.maximum(0, cons), axis=1)
            cv = np.sum(np.maximum(0, cons), axis=1)
            # Epsilon constraint
            cv[cv < self.epsilon] = 0
        else:
            cv = np.zeros(cons.shape[0]) if cons is not None else np.array([])
        return cv

    def _spea2_selection(self, objs, cons, N):
        """
        Environmental selection for SPEA2 algorithm.

        Parameters
        ----------
        objs : ndarray
            Objective values with shape (pop_size, n_objs)
        cons : ndarray or None
            Constraint values with shape (pop_size, n_cons)
        N : int
            Number of individuals to select

        Returns
        -------
        selected_indices : ndarray
            Indices of selected individuals
        selected_fitness : ndarray
            Fitness values of selected individuals
        """
        pop_size = objs.shape[0]

        # Calculate constraint violations
        cv = self._calculate_constraint_violations(cons)

        # Calculate fitness for all individuals
        fitness = self._cal_fitness(objs, cv)

        # Environmental selection
        next_selected = fitness < 1

        if np.sum(next_selected) < N:
            # Need to add more individuals
            sorted_indices = np.argsort(fitness)
            next_selected[sorted_indices[:N]] = True
        elif np.sum(next_selected) > N:
            # Need to remove some individuals using truncation
            to_remove = self._truncation(objs[next_selected], np.sum(next_selected) - N)
            temp_indices = np.where(next_selected)[0]
            next_selected[temp_indices[to_remove]] = False

        # Get selected indices and fitness
        selected_indices = np.where(next_selected)[0]
        selected_fitness = fitness[selected_indices]

        # Sort selected individuals by fitness
        sort_order = np.argsort(selected_fitness)
        selected_indices = selected_indices[sort_order]
        selected_fitness = selected_fitness[sort_order]

        return selected_indices, selected_fitness

    def _cal_fitness(self, objs, cv):
        """
        Calculate SPEA2 fitness values.

        Parameters
        ----------
        objs : ndarray
            Objective values with shape (N, M)
        cv : ndarray
            Constraint violation values with shape (N,)

        Returns
        -------
        fitness : ndarray
            Fitness values with shape (N,)
        """
        N = objs.shape[0]

        # Detect dominance relations
        dominate = np.zeros((N, N), dtype=bool)

        # MATLAB中按行操作，这里需要仔细处理
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Compare constraint violations
                if cv[i] < cv[j]:
                    dominate[i, j] = True
                elif cv[i] > cv[j]:
                    dominate[j, i] = True
                else:
                    # Same constraint violation, compare objectives
                    # MATLAB: any(PopObj(i,:) < PopObj(j,:)) - any(PopObj(i,:) > PopObj(j,:))
                    less = np.any(objs[i, :] < objs[j, :])
                    greater = np.any(objs[i, :] > objs[j, :])

                    if less and not greater:
                        dominate[i, j] = True
                    elif greater and not less:
                        dominate[j, i] = True

        # Calculate S(i): number of solutions that i dominates
        # MATLAB: S = sum(Dominate, 2) -> 按行求和
        # Python: np.sum(dominate, axis=1) -> 按行求和
        S = np.sum(dominate, axis=1)

        # Calculate R(i): sum of S values of solutions that dominate i
        R = np.zeros(N)
        for i in range(N):
            # Find solutions that dominate i
            dominating_i = dominate[:, i]
            # MATLAB: R(i) = sum(S(Dominate(:, i)))
            # Python: R[i] = np.sum(S[dominating_i])
            if np.any(dominating_i):
                R[i] = np.sum(S[dominating_i])

        # Calculate D(i): density estimation
        # Calculate pairwise distances
        distances = cdist(objs, objs, metric='euclidean')

        # Set diagonal to infinity
        np.fill_diagonal(distances, np.inf)

        # Sort distances for each solution
        sorted_distances = np.sort(distances, axis=1)

        # MATLAB: floor(sqrt(N))
        k = int(np.floor(np.sqrt(N)))

        # MATLAB: D = 1 ./ (Distance(:, k) + 2)
        # Python: 注意MATLAB索引从1开始，所以k需要调整
        # 确保k不超过数组边界
        if k > 0:
            k_idx = min(k, sorted_distances.shape[1]) - 1  # -1 for 0-based indexing
            D = 1.0 / (sorted_distances[:, k_idx] + 2)
        else:
            D = np.zeros(N)

        # Calculate fitness: F = R + D
        fitness = R + D

        return fitness

    def _truncation(self, objs, K):
        """
        Truncation operator for SPEA2.

        Parameters
        ----------
        objs : ndarray
            Objective values of candidate solutions
        K : int
            Number of solutions to remove

        Returns
        -------
        to_remove : ndarray
            Boolean array indicating which solutions to remove
        """
        N = objs.shape[0]
        to_remove = np.zeros(N, dtype=bool)

        if K <= 0:
            return to_remove

        # Calculate pairwise distances
        distances = cdist(objs, objs, metric='euclidean')

        # Set diagonal to infinity
        np.fill_diagonal(distances, np.inf)

        while np.sum(to_remove) < K:
            # Find remaining solutions
            remaining = np.where(~to_remove)[0]

            if len(remaining) <= 1:
                # If only one or zero remaining, break
                break

            # Get distances between remaining solutions
            remaining_distances = distances[np.ix_(remaining, remaining)]

            # Sort distances for each remaining solution
            sorted_distances = np.sort(remaining_distances, axis=1)

            # Find solution with smallest distance to its nearest neighbor
            # Use lexicographic comparison as in MATLAB
            min_idx = 0
            for i in range(1, len(remaining)):
                # Compare rows element-wise
                for j in range(sorted_distances.shape[1]):
                    if sorted_distances[i, j] < sorted_distances[min_idx, j]:
                        min_idx = i
                        break
                    elif sorted_distances[i, j] > sorted_distances[min_idx, j]:
                        break
                # If all equal, keep the smaller index
                else:
                    if remaining[i] < remaining[min_idx]:
                        min_idx = i

            # Remove the selected solution
            to_remove[remaining[min_idx]] = True

        return to_remove
"""
Two-Archive Algorithm 2 (Two_Arch2)

This module implements Two_Arch2 for many-objective optimization problems.

References
----------
    [1] Wang, H., Jiao, L., & Yao, X. (2015). Two_Arch2: An improved two-archive algorithm for
        many-objective optimization. IEEE Transactions on Evolutionary Computation, 19(4), 524-541.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.13
Version: 1.0
"""
from tqdm import tqdm
import time
from Methods.Algo_Methods.algo_utils import *


class TwoArch2:
    """
    Two-Archive Algorithm 2 for many-objective optimization.

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
        'n_cons': '0',
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

    def __init__(self, problem, n=None, max_nfes=None, CA_size=None, p=None, save_data=True,
                 save_path='./TestData', name='Two_Arch2_test', disable_tqdm=True):
        """
        Initialize Two_Arch2 algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        CA_size : int or None, optional
            Convergence archive size (default: None, will be set to population size)
        p : float or None, optional
            Parameter for fractional distance (default: None, will be set to 1/M)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'Two_Arch2_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.CA_size = CA_size
        self.p = p
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Two_Arch2 algorithm.

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
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs = init_history(decs, objs)

        # Initialize archives for each task
        CAs = []  # Convergence Archive
        DAs = []  # Diversity Archive

        for i in range(nt):
            # Set CA size and p parameter for this task
            CA_size_i = self.CA_size if self.CA_size is not None else n_per_task[i]
            p_i = self.p if self.p is not None else 1.0 / objs[i].shape[1]

            # Initialize archives
            CA_i = self._update_CA([], objs[i], decs[i], CA_size_i)
            DA_i = self._update_DA([], [], objs[i], decs[i], n_per_task[i], p_i)

            CAs.append(CA_i)
            DAs.append(DA_i)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Get CA and DA for current task
                CA_objs_i, CA_decs_i = CAs[i]
                DA_objs_i, DA_decs_i = DAs[i]

                # Mating selection
                parentC_indices, parentM_indices = self._mating_selection(
                    CA_objs_i, CA_decs_i, DA_objs_i, DA_decs_i, n_per_task[i]
                )

                # Generate offspring through crossover and mutation
                # ParentC: for convergence (SBX + polynomial mutation)
                parentC_decs = np.vstack([CA_decs_i[parentC_indices[:len(parentC_indices) // 2]],
                                          DA_decs_i[parentC_indices[len(parentC_indices) // 2:]]])
                off_decs_C = ga_generation(parentC_decs, muc=20.0, mum=0)

                # ParentM: for diversity (only mutation)
                parentM_decs = CA_decs_i[parentM_indices]
                off_decs_M = ga_generation(parentM_decs, muc=0, mum=20.0)

                # Combine offspring
                off_decs = np.vstack([off_decs_C, off_decs_M])
                off_objs, _ = evaluation_single(problem, off_decs, i)

                # Update archives
                CA_objs_i, CA_decs_i = self._update_CA(
                    (CA_objs_i, CA_decs_i), off_objs, off_decs,
                    self.CA_size if self.CA_size is not None else n_per_task[i]
                )
                DA_objs_i, DA_decs_i = self._update_DA(
                    (DA_objs_i, DA_decs_i), (CA_objs_i, CA_decs_i),
                    off_objs, off_decs, n_per_task[i],
                    self.p if self.p is not None else 1.0 / objs[i].shape[1]
                )

                # Update archives
                CAs[i] = (CA_objs_i, CA_decs_i)
                DAs[i] = (DA_objs_i, DA_decs_i)

                # Update main population with DA for tracking
                objs[i] = DA_objs_i
                decs[i] = DA_decs_i

                nfes_per_task[i] += off_decs.shape[0]
                pbar.update(off_decs.shape[0])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results (using DA as final population)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _mating_selection(self, CA_objs, CA_decs, DA_objs, DA_decs, N):
        """
        Mating selection for Two_Arch2.

        Parameters
        ----------
        CA_objs : ndarray
            Convergence archive objectives
        CA_decs : ndarray
            Convergence archive decisions
        DA_objs : ndarray
            Diversity archive objectives
        DA_decs : ndarray
            Diversity archive decisions
        N : int
            Population size

        Returns
        -------
        parentC_indices : ndarray
            Indices for convergence parents
        parentM_indices : ndarray
            Indices for diversity parents
        """
        CA_size = CA_objs.shape[0]
        DA_size = DA_objs.shape[0]

        # Select parents from CA (for convergence)
        CA_parent1 = np.random.randint(0, CA_size, size=int(np.ceil(N / 2)))
        CA_parent2 = np.random.randint(0, CA_size, size=int(np.ceil(N / 2)))

        # Determine dominance between CA parents
        dominate = np.zeros(len(CA_parent1))
        for idx in range(len(CA_parent1)):
            p1 = CA_parent1[idx]
            p2 = CA_parent2[idx]

            # Check if p1 dominates p2
            less = np.any(CA_objs[p1] < CA_objs[p2])
            greater = np.any(CA_objs[p1] > CA_objs[p2])

            if less and not greater:
                dominate[idx] = 1
            elif greater and not less:
                dominate[idx] = -1

        # Select based on dominance
        parentC_from_CA = []
        for idx in range(len(dominate)):
            if dominate[idx] == 1:
                parentC_from_CA.append(CA_parent1[idx])
            else:
                parentC_from_CA.append(CA_parent2[idx])

        # Add random parents from DA
        parentC_from_DA = np.random.randint(0, DA_size, size=int(np.ceil(N / 2)))

        # Combine parents for convergence
        parentC_indices = np.concatenate([parentC_from_CA, parentC_from_DA])

        # Select parents from CA for mutation (diversity)
        parentM_indices = np.random.randint(0, CA_size, size=N)

        return parentC_indices, parentM_indices

    def _update_CA(self, CA, new_objs, new_decs, max_size):
        """
        Update Convergence Archive (CA).

        Parameters
        ----------
        CA : tuple or None
            Current CA (objs, decs) or None
        new_objs : ndarray
            New objectives to add
        new_decs : ndarray
            New decisions to add
        max_size : int
            Maximum size of CA

        Returns
        -------
        CA_objs : ndarray
            Updated CA objectives
        CA_decs : ndarray
            Updated CA decisions
        """
        if CA is None or len(CA) == 0:
            CA_objs = new_objs
            CA_decs = new_decs
        else:
            CA_objs, CA_decs = CA
            CA_objs = np.vstack([CA_objs, new_objs])
            CA_decs = np.vstack([CA_decs, new_decs])

        N = CA_objs.shape[0]
        if N <= max_size:
            return CA_objs, CA_decs

        # Normalize objectives
        min_vals = np.min(CA_objs, axis=0)
        max_vals = np.max(CA_objs, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero

        CA_objs_norm = (CA_objs - min_vals) / range_vals

        # Calculate indicator matrix I
        I = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                I[i, j] = np.max(CA_objs_norm[i] - CA_objs_norm[j])

        # Calculate normalization constants
        C = np.max(np.abs(I), axis=0)
        C[C == 0] = 1.0  # Avoid division by zero

        # Calculate fitness (IBEA fitness)
        kappa = 0.05
        C_matrix = np.tile(C, (N, 1))

        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        C_matrix = np.maximum(C_matrix, epsilon)

        # Calculate fitness with numerical stability
        exponent = -I / C_matrix / kappa
        # Clip exponent to avoid overflow/underflow
        exponent = np.clip(exponent, -100, 100)
        F = np.sum(-np.exp(exponent), axis=0) + 1

        # Delete solutions based on fitness
        choose = np.arange(N)
        while len(choose) > max_size:
            # Find solution with minimum fitness
            min_idx = np.argmin(F[choose])
            to_remove = choose[min_idx]

            # Update fitness values with numerical stability
            if C[to_remove] > epsilon:
                exponent_update = -I[to_remove, :] / C[to_remove] / kappa
                exponent_update = np.clip(exponent_update, -100, 100)
                F = F + np.exp(exponent_update)

            # Remove solution
            choose = np.delete(choose, min_idx)

        return CA_objs[choose], CA_decs[choose]

    def _update_DA(self, DA, CA, new_objs, new_decs, max_size, p):
        """
        Update Diversity Archive (DA).

        Parameters
        ----------
        DA : tuple or None
            Current DA (objs, decs) or None
        CA : tuple or None
            Current CA (objs, decs) or None
        new_objs : ndarray
            New objectives to add
        new_decs : ndarray
            New decisions to add
        max_size : int
            Maximum size of DA
        p : float
            Parameter for fractional distance

        Returns
        -------
        DA_objs : ndarray
            Updated DA objectives
        DA_decs : ndarray
            Updated DA decisions
        """
        # Combine current DA and new solutions
        if DA is None or len(DA) == 0:
            DA_objs = new_objs
            DA_decs = new_decs
        else:
            DA_objs, DA_decs = DA
            DA_objs = np.vstack([DA_objs, new_objs])
            DA_decs = np.vstack([DA_decs, new_decs])

        # Non-dominated sorting using existing nd_sort function
        # We need to get the first front (non-dominated solutions)
        N = DA_objs.shape[0]

        # Use nd_sort to get non-dominated front
        front_no, max_fno = nd_sort(DA_objs, N)

        # Get non-dominated solutions (front number 1)
        non_dominated_mask = front_no == 1

        DA_objs = DA_objs[non_dominated_mask]
        DA_decs = DA_decs[non_dominated_mask]

        N = DA_objs.shape[0]
        if N <= max_size:
            return DA_objs, DA_decs

        # Select extreme solutions first
        choose = np.zeros(N, dtype=bool)

        # Minimum values for each objective
        for m in range(DA_objs.shape[1]):
            min_idx = np.argmin(DA_objs[:, m])
            choose[min_idx] = True

        # Maximum values for each objective
        for m in range(DA_objs.shape[1]):
            max_idx = np.argmax(DA_objs[:, m])
            choose[max_idx] = True

        # Adjust selection size
        if np.sum(choose) > max_size:
            # Randomly delete some solutions
            chosen_indices = np.where(choose)[0]
            to_remove = np.random.choice(chosen_indices, size=np.sum(choose) - max_size, replace=False)
            choose[to_remove] = False
        elif np.sum(choose) < max_size:
            # Add solutions using truncation strategy
            # Calculate distance matrix with p-norm
            distance = np.full((N, N), np.inf)

            # More efficient calculation using vectorization
            for i in range(N):
                diff = DA_objs - DA_objs[i]
                # Calculate p-norm distance
                if p == 1:
                    # Manhattan distance
                    distance[i, :] = np.sum(np.abs(diff), axis=1)
                elif p == 2:
                    # Euclidean distance
                    distance[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))
                else:
                    # General p-norm
                    distance[i, :] = np.sum(np.abs(diff) ** p, axis=1) ** (1 / p)

            # Set diagonal to infinity
            np.fill_diagonal(distance, np.inf)

            # Add solutions until reaching max_size
            while np.sum(choose) < max_size:
                remaining = np.where(~choose)[0]
                chosen = np.where(choose)[0]

                if len(chosen) == 0:
                    # If no solutions chosen yet, choose randomly
                    choose[np.random.choice(remaining)] = True
                else:
                    # Calculate minimum distance to chosen solutions for each remaining solution
                    min_distances = np.min(distance[np.ix_(remaining, chosen)], axis=1)

                    # Select solution with maximum minimum distance
                    max_idx = np.argmax(min_distances)
                    choose[remaining[max_idx]] = True

        return DA_objs[choose], DA_decs[choose]
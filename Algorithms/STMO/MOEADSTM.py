"""
MOEA/D with Stable Matching (MOEA/D-STM)

This module implements MOEA/D-STM for multi-objective optimization problems.

References
----------
    [1] Li, Ke, et al. "Stable matching-based selection in evolutionary multiobjective \
        optimization." IEEE Transactions on Evolutionary Computation 18.6 (2014): 909-923.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.01
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from Methods.Algo_Methods.uniform_point import uniform_point
from Methods.Algo_Methods.algo_utils import *


class MOEADSTM:
    """
    MOEA/D with Stable Matching for multi-objective optimization.

    This algorithm uses a stable matching model to select solutions for
    subproblems, ensuring a stable assignment between solutions and weight vectors.

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
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, T=None,
                 save_data=True, save_path='./TestData',
                 name='MOEADSTM_test', disable_tqdm=True):
        """
        Initialize MOEA/D-STM algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        T : int or List[int], optional
            Size of neighborhood (default: ceil(n/10))
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MOEADSTM_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.T = T
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MOEA/D-STM algorithm.

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

        # Set neighborhood size for each task
        if self.T is None:
            T_per_task = [int(np.ceil(n / 10)) for n in n_per_task]
        else:
            T_per_task = par_list(self.T, nt)

        # Generate uniformly distributed weight vectors for each task
        Weight = []
        B = []  # Neighborhood for each task
        for i in range(nt):
            Weight_i, n = uniform_point(n_per_task[i], no[i])
            Weight.append(Weight_i)
            n_per_task[i] = n

            # Detect the neighbors of each solution
            distance = cdist(Weight_i, Weight_i)
            B_i = np.argsort(distance, axis=1)[:, :T_per_task[i]]
            B.append(B_i)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs = init_history(decs, objs)

        # Initialize algorithm-specific variables for each task
        z = []  # Ideal point
        Pi = []  # Utility for each subproblem
        oldObj = []  # Old Tchebycheff function value

        for i in range(nt):
            z_i = np.min(objs[i], axis=0)
            z.append(z_i)

            Pi_i = np.ones(n_per_task[i])
            Pi.append(Pi_i)

            # Calculate old Tchebycheff values
            oldObj_i = np.max(np.abs((objs[i] - z_i) / Weight[i]), axis=1)
            oldObj.append(oldObj_i)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Perform 5 sub-generations
                for subgen in range(5):
                    # Choose I: boundary solutions + tournament selection
                    # Boundary solutions are those with weight close to axes
                    boundary = np.where(np.sum(Weight[i] < 1e-3, axis=1) == no[i] - 1)[0]
                    n_select = int(np.floor(n_per_task[i] / 5)) - len(boundary)

                    if n_select > 0:
                        # Tournament selection based on negative utility
                        tournament_indices = self._tournament_selection(10, n_select, -Pi[i])
                        I = np.concatenate([boundary, tournament_indices])
                    else:
                        I = boundary

                    # Generate offspring for each solution in I
                    P = np.zeros((len(I), 3), dtype=int)
                    for idx, sol_idx in enumerate(I):
                        if nfes_per_task[i] >= max_nfes_per_task[i]:
                            break

                        # Choose the parents
                        if np.random.rand() < 0.9:
                            # From neighborhood
                            P[idx, :] = B[i][sol_idx, np.random.permutation(T_per_task[i])[:3]]
                        else:
                            # From entire population
                            P[idx, :] = np.random.permutation(n_per_task[i])[:3]

                    # Generate offspring using DE
                    if len(I) > 0 and nfes_per_task[i] < max_nfes_per_task[i]:
                        # Prepare parent arrays
                        parent1 = decs[i][P[:, 0]]
                        parent2 = decs[i][P[:, 1]]
                        parent3 = decs[i][P[:, 2]]

                        # DE operator
                        off_decs = self._operator_de(parent1, parent2, parent3)

                        # Evaluate offspring
                        off_objs, _ = evaluation_single(problem, off_decs, i)

                        # Update ideal point
                        z[i] = np.minimum(z[i], np.min(off_objs, axis=0))

                        # Combine population and offspring
                        combined_decs = np.vstack([decs[i], off_decs])
                        combined_objs = np.vstack([objs[i], off_objs])

                        # STM selection
                        decs[i], objs[i] = self._stm_selection(
                            combined_objs, combined_decs, Weight[i], z[i],
                            np.max(objs[i], axis=0)
                        )

                        nfes_per_task[i] += len(off_decs)
                        pbar.update(len(off_decs))

                # Update Pi every 10 generations
                current_gen = int(np.ceil(nfes_per_task[i] / n_per_task[i]))
                if current_gen % 10 == 0:
                    # Calculate new Tchebycheff values
                    newObj_i = np.max(np.abs((objs[i] - z[i]) / Weight[i]), axis=1)
                    DELTA = oldObj[i] - newObj_i

                    # Update utility
                    Temp = DELTA < 0.001
                    Pi[i][~Temp] = 1
                    Pi[i][Temp] = (0.95 + 0.05 * DELTA[Temp] / 0.001) * Pi[i][Temp]

                    oldObj[i] = newObj_i

                # Update history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name, save_data=self.save_data
        )

        return results

    def _tournament_selection(self, tournament_size, n_select, fitness):
        """
        Tournament selection based on fitness values.

        Parameters
        ----------
        tournament_size : int
            Number of individuals in each tournament
        n_select : int
            Number of individuals to select
        fitness : ndarray
            Fitness values (higher is better after negation)

        Returns
        -------
        ndarray
            Selected indices
        """
        pop_size = len(fitness)
        selected = []

        for _ in range(n_select):
            # Randomly select tournament_size individuals
            candidates = np.random.choice(pop_size, size=min(tournament_size, pop_size), replace=False)
            # Select the best one
            winner = candidates[np.argmax(fitness[candidates])]
            selected.append(winner)

        return np.array(selected)

    def _operator_de(self, parent1, parent2, parent3, CR=1.0, F=0.5):
        """
        Differential Evolution operator with polynomial mutation.

        Parameters
        ----------
        parent1 : ndarray
            First parent (N x D)
        parent2 : ndarray
            Second parent (N x D)
        parent3 : ndarray
            Third parent (N x D)
        CR : float, optional
            Crossover rate (default: 1.0)
        F : float, optional
            Scaling factor (default: 0.5)

        Returns
        -------
        ndarray
            Offspring solutions
        """
        N, D = parent1.shape

        # DE mutation: V = X1 + F * (X2 - X3)
        V = parent1 + F * (parent2 - parent3)

        # Crossover
        offspring = parent1.copy()
        mask = np.random.rand(N, D) < CR
        offspring[mask] = V[mask]

        # Boundary handling
        offspring = np.clip(offspring, 0, 1)

        # Polynomial mutation
        proM = 1.0
        disM = 20
        site = np.random.rand(N, D) < proM / D
        mu = np.random.rand(N, D)

        # Lower half
        temp = site & (mu <= 0.5)
        delta = (2 * mu + (1 - 2 * mu) * (1 - offspring) ** (disM + 1)) ** (1 / (disM + 1)) - 1
        offspring[temp] = offspring[temp] + delta[temp]

        # Upper half
        temp = site & (mu > 0.5)
        delta = 1 - (2 * (1 - mu) + 2 * (mu - 0.5) * (1 - (1 - offspring)) ** (disM + 1)) ** (1 / (disM + 1))
        offspring[temp] = offspring[temp] + delta[temp]

        # Final boundary handling
        offspring = np.clip(offspring, 0, 1)

        return offspring

    def _stm_selection(self, objs, decs, W, z, znad):
        """
        Stable Matching based selection.

        This implements the stable matching model where solutions are matched
        to subproblems in a stable manner.

        Parameters
        ----------
        objs : ndarray
            Objective values of all solutions (N x M)
        decs : ndarray
            Decision variables of all solutions (N x D)
        W : ndarray
            Weight vectors (NW x M)
        z : ndarray
            Ideal point (M,)
        znad : ndarray
            Nadir point (M,)

        Returns
        -------
        tuple
            Selected decision variables and objectives (NW solutions)
        """
        N = objs.shape[0]
        NW = W.shape[0]

        # Calculate the modified Tchebycheff value of each solution on each subproblem
        g = np.zeros((N, NW))
        for i in range(N):
            # For each solution, calculate its Tchebycheff value on all subproblems
            g[i, :] = np.max(np.abs(objs[i] - z) / W, axis=1)

        # Calculate the perpendicular distance of each solution on each subproblem
        # Normalize objectives
        PopObj = (objs - z) / (znad - z + 1e-10)

        # Calculate cosine similarity
        Cosine = 1 - cdist(PopObj, W, metric='cosine')

        # Calculate perpendicular distance
        norms = np.sqrt(np.sum(PopObj ** 2, axis=1, keepdims=True))
        Distance = norms * np.sqrt(1 - Cosine ** 2)

        # STM selection - Stable Matching
        Fp = np.zeros(NW, dtype=int)  # Subproblem -> Solution mapping
        FX = np.zeros(N, dtype=int)  # Solution -> Subproblem mapping
        Phi = np.zeros((NW, N), dtype=bool)  # Proposal history

        # Continue until all subproblems are matched
        while np.any(Fp == 0):
            # Find unmatched subproblems
            RemainW = np.where(Fp == 0)[0]

            # Randomly select one unmatched subproblem
            i = RemainW[np.random.randint(len(RemainW))]

            # Find solutions that haven't been proposed to by this subproblem
            RemainX = np.where(~Phi[i, :])[0]

            # Select the best solution for this subproblem (minimum Tchebycheff)
            best_idx = np.argmin(g[RemainX, i])
            j = RemainX[best_idx]

            # Mark this proposal
            Phi[i, j] = True

            if FX[j] == 0:
                # Solution j is not matched, accept immediately
                Fp[i] = j + 1  # +1 because 0 means unmatched
                FX[j] = i + 1
            elif Distance[j, i] < Distance[j, FX[j] - 1]:
                # Solution j prefers subproblem i over its current match
                old_match = FX[j] - 1
                Fp[i] = j + 1
                Fp[old_match] = 0  # Old match becomes unmatched
                FX[j] = i + 1

        # Extract selected solutions (Fp contains 1-indexed solution IDs)
        selected_indices = Fp - 1  # Convert back to 0-indexed

        return decs[selected_indices], objs[selected_indices]
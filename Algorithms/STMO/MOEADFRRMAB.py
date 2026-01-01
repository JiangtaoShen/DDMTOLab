"""
MOEA/D with Fitness-Rate-Rank-based Multi-Armed Bandit (MOEA/D-FRRMAB)

This module implements MOEA/D-FRRMAB for multi-objective optimization problems.

References
----------
    [1] Li, Ke, et al. "Adaptive operator selection with bandits for a multiobjective \
        evolutionary algorithm based on decomposition." IEEE Transactions on Evolutionary \
        Computation 18.1 (2014): 114-130.

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


class MOEADFRRMAB:
    """
    MOEA/D with Fitness-Rate-Rank-based Multi-Armed Bandit.

    This algorithm uses a multi-armed bandit approach to adaptively select
    differential evolution operators during optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, C=5, W=None, D=1,
                 T=20, nr=2, save_data=True, save_path='./TestData',
                 name='MOEADFRRMAB_test', disable_tqdm=True):
        """
        Initialize MOEA/D-FRRMAB algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        C : float, optional
            Scaling factor in bandit-based operator selection (default: 5)
        W : int or List[int], optional
            Size of sliding window (default: ceil(n/2))
        D : float, optional
            Decaying factor in calculating credit value (default: 1)
        T : int, optional
            Size of neighborhood (default: 20)
        nr : int, optional
            Maximum number of solutions replaced by each offspring (default: 2)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MOEADFRRMAB_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.C = C
        self.W = W
        self.D = D
        self.T = T
        self.nr = nr
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MOEA/D-FRRMAB algorithm.

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

        # Set sliding window size for each task
        if self.W is None:
            W_per_task = [int(np.ceil(n / 2)) for n in n_per_task]
        else:
            W_per_task = par_list(self.W, nt)

        # Generate uniformly distributed weight vectors for each task
        Weight = []
        B = []  # Neighborhood for each task
        for i in range(nt):
            Weight_i, n = uniform_point(n_per_task[i], no[i])
            Weight.append(Weight_i)
            n_per_task[i] = n

            # Detect the neighbors of each solution
            distance = cdist(Weight_i, Weight_i)
            B_i = np.argsort(distance, axis=1)[:, :self.T]
            B.append(B_i)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs = init_history(decs, objs)

        # Initialize algorithm-specific variables for each task
        Z = []  # Ideal point
        Pi = []  # Utility for each subproblem
        oldObj = []  # Old Tchebycheff function value
        FRR = []  # Credit value of each operator
        SW = []  # Sliding window

        for i in range(nt):
            Z_i = np.min(objs[i], axis=0)
            Z.append(Z_i)

            Pi_i = np.ones(n_per_task[i])
            Pi.append(Pi_i)

            # Calculate old Tchebycheff values
            oldObj_i = np.max(np.abs((objs[i] - Z_i) * Weight[i]), axis=1)
            oldObj.append(oldObj_i)

            # Initialize FRR (4 operators)
            FRR_i = np.zeros(4)
            FRR.append(FRR_i)

            # Initialize sliding window: [operator_indices; rewards]
            SW_i = np.zeros((2, W_per_task[i]))
            SW.append(SW_i)

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

                    # For each solution in I
                    for idx in I:
                        if nfes_per_task[i] >= max_nfes_per_task[i]:
                            break

                        # Bandit-based operator selection
                        op = self._frrmab(FRR[i], SW[i], self.C)

                        # Choose the parents
                        if np.random.rand() < 0.9:
                            # From neighborhood
                            P = B[i][idx, np.random.permutation(self.T)]
                        else:
                            # From entire population
                            P = np.random.permutation(n_per_task[i])

                        # Generate an offspring using the selected DE operator
                        off_dec = self._four_de(
                            op,
                            decs[i][idx],
                            decs[i][P[0]],
                            decs[i][P[1]],
                            decs[i][P[2]],
                            decs[i][P[3]],
                            decs[i][P[4]] if len(P) > 4 else decs[i][P[0]]
                        )

                        # Evaluate offspring
                        off_obj, _ = evaluation_single(problem, off_dec.reshape(1, -1), i)
                        off_obj = off_obj[0]

                        # Update the ideal point
                        Z[i] = np.minimum(Z[i], off_obj)

                        # Update the solutions in P by Tchebycheff approach
                        P_subset = P[:self.nr * 5]  # Consider more neighbors for replacement
                        g_old = np.max(np.abs((objs[i][P_subset] - Z[i]) * Weight[i][P_subset]), axis=1)
                        g_new = np.max(np.abs((off_obj - Z[i]) * Weight[i][P_subset]), axis=1)

                        # Find solutions that can be replaced
                        better_mask = g_old >= g_new
                        replace_indices = np.where(better_mask)[0][:self.nr]

                        if len(replace_indices) > 0:
                            # Replace solutions
                            actual_replace = P_subset[replace_indices]
                            for r_idx in actual_replace:
                                decs[i][r_idx] = off_dec
                                objs[i][r_idx] = off_obj

                            # Calculate Fitness Improvement Rate (FIR)
                            FIR = np.sum((g_old[replace_indices] - g_new[replace_indices]) / g_old[replace_indices])
                        else:
                            FIR = 0

                        # Update sliding window
                        SW[i] = np.column_stack([SW[i][:, 1:], [op, FIR]])

                        # Update FRR through credit assignment
                        FRR[i] = self._credit_assignment(SW[i], self.D)

                        nfes_per_task[i] += 1
                        pbar.update(1)

                # Update Pi every 10 generations
                current_gen = int(np.ceil(nfes_per_task[i] / n_per_task[i]))
                if current_gen % 10 == 0:
                    # Calculate new Tchebycheff values
                    newObj_i = np.max(np.abs((objs[i] - Z[i]) * Weight[i]), axis=1)
                    DELTA = (oldObj[i] - newObj_i) / oldObj[i]

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

    def _frrmab(self, FRR, SW, C):
        """
        Fitness-Rate-Rank-based Multi-Armed Bandit operator selection.

        Parameters
        ----------
        FRR : ndarray
            Credit value of each operator (4 operators)
        SW : ndarray
            Sliding window [operator_indices; rewards]
        C : float
            Scaling factor

        Returns
        -------
        int
            Selected operator index (0-3)
        """
        # If any operator has zero credit or hasn't been used
        if np.any(FRR == 0) or np.any(SW[0, :] == 0):
            # Random selection
            return np.random.randint(0, 4)
        else:
            # Count how many times each operator has been used
            n = np.zeros(4)
            for op_idx in range(4):
                n[op_idx] = np.sum(SW[0, :] == op_idx)

            # UCB (Upper Confidence Bound) selection
            ucb = FRR + C * np.sqrt(2 * np.log(np.sum(n)) / (n + 1e-10))
            return np.argmax(ucb)

    def _credit_assignment(self, SW, D):
        """
        Credit assignment for operators based on their performance.

        Parameters
        ----------
        SW : ndarray
            Sliding window [operator_indices; rewards]
        D : float
            Decaying factor

        Returns
        -------
        ndarray
            Credit values (FRR) for each operator
        """
        K = 4  # Number of operators
        Reward = np.zeros(K)

        # Calculate total reward for each operator
        for i in range(K):
            Reward[i] = np.sum(SW[1, SW[0, :] == i])

        # Rank operators by reward (descending)
        Rank = np.argsort(-Reward)  # Sort descending
        Rank_inverse = np.zeros(K, dtype=int)
        Rank_inverse[Rank] = np.arange(K)

        # Apply decaying factor based on rank
        Decay = (D ** Rank_inverse) * Reward

        # Normalize to get FRR
        total = np.sum(Decay)
        if total > 0:
            FRR = Decay / total
        else:
            FRR = np.ones(K) / K

        return FRR

    def _four_de(self, op, x, x1, x2, x3, x4, x5):
        """
        Four different DE operators with polynomial mutation.

        Parameters
        ----------
        op : int
            Operator index (0-3)
        x : ndarray
            Current solution
        x1, x2, x3, x4, x5 : ndarray
            Parent solutions

        Returns
        -------
        ndarray
            Offspring solution
        """
        # Parameters
        CR = 1.0
        F = 0.5
        proM = 1.0
        disM = 20
        K = 0.5
        D = len(x)

        # Differential evolution
        if op == 0:
            # DE/rand/1
            v = x + F * (x1 - x2)
        elif op == 1:
            # DE/rand/2
            v = x + F * (x1 - x2) + F * (x3 - x4)
        elif op == 2:
            # DE/current-to-rand/2
            v = x + K * (x - x1) + F * (x2 - x3) + F * (x4 - x5)
        else:  # op == 3
            # DE/current-to-rand/1
            v = x + K * (x - x1) + F * (x2 - x3)

        # Crossover
        offspring = x.copy()
        CR_adjust = CR + (1 if op > 1 else 0)  # Higher CR for current-to-rand operators
        site = np.random.rand(D) < CR_adjust
        offspring[site] = v[site]

        # Boundary handling (clip to [0, 1] since we're in normalized space)
        offspring = np.clip(offspring, 0, 1)

        # Polynomial mutation
        site = np.random.rand(D) < proM / D
        mu = np.random.rand(D)

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
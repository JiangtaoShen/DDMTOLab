"""
Multi-task Multi-objective Evolutionary Algorithm Based on Decomposition with Dynamic Neighborhood (MTEA-D-DN)

This module implements MTEA-D-DN for multi-task multi-objective optimization problems.

References
----------
    [1] Wang, Xianpeng, Zhiming Dong, Lixin Tang, and Qingfu Zhang. "Multiobjective Multitask Optimization - Neighborhood as a Bridge for Knowledge Transfer." IEEE Transactions on Evolutionary Computation 27.1 (2023): 155-169.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.18
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from Methods.Algo_Methods.uniform_point import uniform_point
from Methods.Algo_Methods.algo_utils import *


class MTEADDN:
    """
    Multi-task Multi-objective Evolutionary Algorithm Based on Decomposition with Dynamic Neighborhood.

    This algorithm uses neighborhood structure as a bridge for knowledge transfer between tasks.
    It maintains two types of neighborhoods:
    - B: Primary neighborhood within the same task (based on weight vector distance)
    - B2: Secondary neighborhood from other tasks (for knowledge transfer)

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, beta=0.2, F=0.5, CR=0.9, mum=20.0,
                 save_data=True, save_path='./TestData', name='MTEADDN_test', disable_tqdm=True):
        """
        Initialize MTEA-D-DN algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        beta : float, optional
            Probability of choosing parents locally (from neighborhood) (default: 0.2)
        F : float, optional
            Scaling factor for DE mutation (default: 0.5)
        CR : float, optional
            Crossover rate for DE (default: 0.9)
        mum : float, optional
            Distribution index for polynomial mutation (default: 20.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MTEADDN_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.beta = beta
        self.F = F
        self.CR = CR
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTEA-D-DN algorithm.

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
        N = []  # Actual population sizes (may differ from n_per_task due to uniform point generation)
        DT = []  # Neighborhood sizes
        B = []  # Primary neighbor indices (within same task)

        for t in range(nt):
            w_t, n_t = uniform_point(n_per_task[t], no[t])
            W.append(w_t)
            N.append(n_t)
            n_per_task[t] = n_t

            # Neighborhood size: ceil(N/20)
            dt = int(np.ceil(n_t / 20))
            DT.append(dt)

            # Detect neighbors based on Euclidean distance of weight vectors
            distances = squareform(pdist(w_t))
            neighbors = np.argsort(distances, axis=1)[:, :dt]
            B.append(neighbors)

        # Initialize population and evaluate for each task
        decs = initialization(problem, N)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = N.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize ideal points for each task
        Z = []
        for t in range(nt):
            Z.append(np.min(objs[t], axis=0))

        # Initialize secondary neighborhoods (B2) for knowledge transfer
        B2k = []  # Target task indices for secondary neighborhood
        B2 = []  # Secondary neighbor indices (from other tasks)

        for t in range(nt):
            tar_pool = [k for k in range(nt) if k != t]
            b2k_t = []
            b2_t = []
            for i in range(N[t]):
                # Randomly select a target task
                target_task = tar_pool[np.random.randint(len(tar_pool))]
                b2k_t.append(target_task)
                # Randomly select DT[t] neighbors from the target task
                b2_t.append(np.random.permutation(N[target_task])[:DT[t]])
            B2k.append(b2k_t)
            B2.append(b2_t)

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            active_tasks = [t for t in range(nt) if nfes_per_task[t] < max_nfes_per_task[t]]
            if not active_tasks:
                break

            for t in active_tasks:
                for i in range(N[t]):
                    if rand() < self.beta:
                        # Local mating: choose parents from neighborhoods (same + other task)
                        P1 = B[t][i].copy()  # Primary neighborhood indices
                        P2 = B2[t][i].copy()  # Secondary neighborhood indices
                        target_task = B2k[t][i]

                        # Create combined parent pool with task labels
                        tasks = np.concatenate([
                            np.full(len(P1), t, dtype=int),
                            np.full(len(P2), target_task, dtype=int)
                        ])
                        P = np.concatenate([P1, P2])

                        # Shuffle the combined pool
                        perm = np.random.permutation(len(tasks))
                        tasks = tasks[perm]
                        P = P[perm]

                        # Generate offspring using DE/rand/1/bin + polynomial mutation
                        parent_decs = np.array([decs[t][i], decs[tasks[0]][P[0]], decs[tasks[1]][P[1]]])
                        offspring_dec = self._generation(parent_decs)

                        if rand() < 0.5:
                            # Knowledge transfer: evaluate on target task and update
                            k = target_task
                            offspring_dec_padded = self._align_dimensions(offspring_dec, problem.dims[k])
                            off_obj, off_con = evaluation_single(problem, offspring_dec_padded.reshape(1, -1), k)
                            nfes_per_task[k] += 1
                            pbar.update(1)

                            # Update ideal point
                            Z[k] = np.minimum(Z[k], off_obj[0])

                            # Tchebycheff scalarization for neighbors in target task
                            g_old = self._tchebycheff(objs[k][P2], Z[k], W[k][P2])
                            g_new = self._tchebycheff(np.tile(off_obj, (len(P2), 1)), Z[k], W[k][P2])

                            # Update neighbors where offspring is better
                            update_mask = g_old >= g_new
                            for idx, p_idx in enumerate(P2):
                                if update_mask[idx]:
                                    decs[k][p_idx] = offspring_dec_padded
                                    objs[k][p_idx] = off_obj[0]
                                    if cons[k] is not None:
                                        cons[k][p_idx] = off_con[0]

                            # Update secondary neighborhood if no improvement
                            if not np.any(update_mask):
                                # Re-initialize secondary neighborhood for this subproblem
                                tar_pool = [j for j in range(nt) if j != t]
                                B2k[t][i] = tar_pool[np.random.randint(len(tar_pool))]
                                B2[t][i] = np.random.permutation(N[B2k[t][i]])[:DT[t]]
                            elif np.any(update_mask):
                                # Keep only successful neighbors
                                B2[t][i] = P2[update_mask]
                        else:
                            # Evaluate on current task
                            offspring_dec_padded = self._align_dimensions(offspring_dec, problem.dims[t])
                            off_obj, off_con = evaluation_single(problem, offspring_dec_padded.reshape(1, -1), t)
                            nfes_per_task[t] += 1
                            pbar.update(1)

                            # Update ideal point
                            Z[t] = np.minimum(Z[t], off_obj[0])

                            # Tchebycheff scalarization for primary neighbors
                            g_old = self._tchebycheff(objs[t][P1], Z[t], W[t][P1])
                            g_new = self._tchebycheff(np.tile(off_obj, (len(P1), 1)), Z[t], W[t][P1])

                            # Update neighbors where offspring is better
                            update_mask = g_old >= g_new
                            for idx, p_idx in enumerate(P1):
                                if update_mask[idx]:
                                    decs[t][p_idx] = offspring_dec_padded
                                    objs[t][p_idx] = off_obj[0]
                                    if cons[t] is not None:
                                        cons[t][p_idx] = off_con[0]
                    else:
                        # Global mating: choose parents from entire population
                        P = np.random.permutation(N[t])

                        # Generate offspring
                        parent_decs = np.array([decs[t][i], decs[t][P[0]], decs[t][P[1]]])
                        offspring_dec = self._generation(parent_decs)
                        offspring_dec_padded = self._align_dimensions(offspring_dec, problem.dims[t])

                        # Evaluate offspring
                        off_obj, off_con = evaluation_single(problem, offspring_dec_padded.reshape(1, -1), t)
                        nfes_per_task[t] += 1
                        pbar.update(1)

                        # Update ideal point
                        Z[t] = np.minimum(Z[t], off_obj[0])

                        # Tchebycheff scalarization
                        g_old = self._tchebycheff(objs[t][P], Z[t], W[t][P])
                        g_new = self._tchebycheff(np.tile(off_obj, (len(P), 1)), Z[t], W[t][P])

                        # Update neighbors where offspring is better
                        update_mask = g_old >= g_new
                        for idx, p_idx in enumerate(P):
                            if update_mask[idx]:
                                decs[t][p_idx] = offspring_dec_padded
                                objs[t][p_idx] = off_obj[0]
                                if cons[t] is not None:
                                    cons[t][p_idx] = off_con[0]

                # Append to history
                append_history(all_decs[t], decs[t], all_objs[t], objs[t], all_cons[t], cons[t])

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, parent_decs):
        """
        Generate offspring using DE/rand/1/bin mutation and polynomial mutation.

        Parameters
        ----------
        parent_decs : np.ndarray
            Parent decision variables of shape (3, dim)
            parent_decs[0]: target vector
            parent_decs[1], parent_decs[2]: difference vectors

        Returns
        -------
        offspring_dec : np.ndarray
            Offspring decision variable of shape (dim,)
        """
        # DE mutation: v = x1 + F * (x2 - x3)
        mutant = parent_decs[0] + self.F * (parent_decs[1] - parent_decs[2])

        # DE crossover (binomial)
        dim = len(mutant)
        j_rand = np.random.randint(dim)
        mask = np.random.rand(dim) < self.CR
        mask[j_rand] = True
        offspring_dec = np.where(mask, mutant, parent_decs[0])

        # Polynomial mutation
        offspring_dec = mutation(offspring_dec, mu=self.mum)

        # Boundary handling
        offspring_dec = np.clip(offspring_dec, 0, 1)

        return offspring_dec

    def _tchebycheff(self, objs, z, weights):
        """
        Calculate Tchebycheff scalarization values.

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (N, M)
        z : np.ndarray
            Ideal point of shape (M,)
        weights : np.ndarray
            Weight vectors of shape (N, M)

        Returns
        -------
        g : np.ndarray
            Tchebycheff values of shape (N,)
        """
        # Avoid division by zero
        weights_safe = np.maximum(weights, 1e-10)
        return np.max(np.abs(objs - z) * weights_safe, axis=1)

    def _align_dimensions(self, dec, target_dim):
        """
        Align decision variable dimensions to target dimension.

        Parameters
        ----------
        dec : np.ndarray
            Decision variable of shape (dim,)
        target_dim : int
            Target dimension

        Returns
        -------
        aligned_dec : np.ndarray
            Aligned decision variable of shape (target_dim,)
        """
        current_dim = len(dec)
        if current_dim == target_dim:
            return dec
        elif current_dim < target_dim:
            # Pad with random values
            padding = np.random.rand(target_dim - current_dim)
            return np.concatenate([dec, padding])
        else:
            # Truncate
            return dec[:target_dim]

def rand():
    """Generate a random number in [0, 1)."""
    return np.random.rand()
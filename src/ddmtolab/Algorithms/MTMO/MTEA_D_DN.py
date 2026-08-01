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
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MTEA_D_DN:
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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, beta=0.2, F=0.5, CR=0.9, mum=20.0,
                 save_data=True, save_path='./Data', name='MTEA-D-DN', disable_tqdm=True):
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
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTEA-D-DN')
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
        dims = problem.dims
        d_max = max(dims)
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        total_nfes = sum(max_nfes_per_task)

        # Generate uniformly distributed weight vectors for each task
        W = []
        N = []  # Actual population sizes (may differ from n_per_task due to uniform point generation)
        DT = []  # Neighborhood sizes
        B = []  # Primary neighbor indices (within same task)

        for t in range(nt):
            w_t, n_t = uniform_point(n_per_task[t], no[t])
            W.append(w_t)
            N.append(n_t)

            # Neighborhood size: ceil(N/20)
            dt = int(np.ceil(n_t / 20))
            DT.append(dt)

            # Detect neighbors based on Euclidean distance of weight vectors
            distances = squareform(pdist(w_t))
            neighbors = np.argsort(distances, axis=1, kind='stable')[:, :dt]
            B.append(neighbors)

        # Initialize the populations in the unified [0, 1]^d_max space; every
        # genotype keeps d_max genes and only the first dims[t] are evaluated,
        # so a solution can cross tasks without any padding/truncation.
        decs, objs, cons = [], [], []
        for t in range(nt):
            decs_t = np.random.rand(N[t], d_max)
            objs_t, cons_t = evaluation_single(problem, decs_t[:, :dims[t]], t)
            decs.append(decs_t)
            objs.append(objs_t)
            cons.append(cons_t)
        nfes_per_task = list(N)
        nfes = sum(N)

        all_decs, all_objs, all_cons = init_history(
            [decs[t][:, :dims[t]] for t in range(nt)], objs, cons)

        # Initialize ideal points for each task
        Z = [np.min(objs[t], axis=0) for t in range(nt)]

        # Task pools used when (re-)drawing a second neighborhood
        tar_pools = [np.array([k for k in range(nt) if k != t]) for t in range(nt)]

        # Initialize secondary neighborhoods (B2) for knowledge transfer
        B2k = []  # Target task indices for secondary neighborhood
        B2 = []  # Secondary neighbor indices (from other tasks)

        for t in range(nt):
            b2k_t = np.empty(N[t], dtype=int)
            b2_t = []
            for i in range(N[t]):
                # Randomly select a target task
                target_task = tar_pools[t][np.random.randint(nt - 1)]
                b2k_t[i] = target_task
                # Randomly select DT[t] distinct subproblems of the target task
                b2_t.append(np.random.permutation(N[target_task])[:DT[t]])
            B2k.append(b2k_t)
            B2.append(b2_t)

        # Progress bar
        pbar = tqdm(total=total_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop (single shared evaluation budget, as in MToP)
        while nfes < total_nfes:
            for t in range(nt):
                for i in range(N[t]):
                    if np.random.rand() < self.beta:
                        # Local mating: pool the primary neighborhood of task t
                        # with the secondary neighborhood in task k
                        P1 = B[t][i]
                        P2 = B2[t][i]
                        k = B2k[t][i]

                        tasks = np.concatenate([
                            np.full(len(P1), t, dtype=int),
                            np.full(len(P2), k, dtype=int)
                        ])
                        P = np.concatenate([P1, P2])

                        # Shuffle the combined pool
                        perm = np.random.permutation(len(tasks))
                        tasks = tasks[perm]
                        P = P[perm]

                        # Generate offspring using DE/rand/1/bin + polynomial mutation
                        off_dec = self._generation(decs[t][i],
                                                   decs[tasks[0]][P[0]],
                                                   decs[tasks[1]][P[1]],
                                                   d_max)

                        if np.random.rand() < 0.5:
                            # Knowledge transfer: evaluate the offspring on task k
                            off_obj, off_con = evaluation_single(
                                problem, off_dec[:dims[k]].reshape(1, -1), k)
                            nfes_per_task[k] += 1
                            nfes += 1
                            pbar.update(1)

                            Z[k] = np.minimum(Z[k], off_obj[0])

                            # Tchebycheff aggregation on the secondary neighborhood
                            g_old = np.max(np.abs(objs[k][P2] - Z[k]) * W[k][P2], axis=1)
                            g_new = np.max(np.abs(off_obj[0] - Z[k]) * W[k][P2], axis=1)

                            success = g_old >= g_new
                            if np.any(success):
                                rep = P2[success]
                                decs[k][rep] = off_dec
                                objs[k][rep] = off_obj[0]
                                cons[k][rep] = off_con[0]
                                # Shrink the secondary neighborhood to the winners
                                B2[t][i] = rep
                            else:
                                # No winner: draw a brand new secondary neighborhood
                                new_k = tar_pools[t][np.random.randint(nt - 1)]
                                B2k[t][i] = new_k
                                B2[t][i] = np.random.permutation(N[new_k])[:DT[t]]
                        else:
                            # Evaluate the offspring on the current task
                            off_obj, off_con = evaluation_single(
                                problem, off_dec[:dims[t]].reshape(1, -1), t)
                            nfes_per_task[t] += 1
                            nfes += 1
                            pbar.update(1)

                            Z[t] = np.minimum(Z[t], off_obj[0])

                            g_old = np.max(np.abs(objs[t][P1] - Z[t]) * W[t][P1], axis=1)
                            g_new = np.max(np.abs(off_obj[0] - Z[t]) * W[t][P1], axis=1)

                            rep = P1[g_old >= g_new]
                            decs[t][rep] = off_dec
                            objs[t][rep] = off_obj[0]
                            cons[t][rep] = off_con[0]
                    else:
                        # Global mating: the whole population is both the mating
                        # pool and the replacement pool
                        P = np.random.permutation(N[t])

                        off_dec = self._generation(decs[t][i],
                                                   decs[t][P[0]],
                                                   decs[t][P[1]],
                                                   d_max)

                        off_obj, off_con = evaluation_single(
                            problem, off_dec[:dims[t]].reshape(1, -1), t)
                        nfes_per_task[t] += 1
                        nfes += 1
                        pbar.update(1)

                        Z[t] = np.minimum(Z[t], off_obj[0])

                        g_old = np.max(np.abs(objs[t][P] - Z[t]) * W[t][P], axis=1)
                        g_new = np.max(np.abs(off_obj[0] - Z[t]) * W[t][P], axis=1)

                        rep = P[g_old >= g_new]
                        decs[t][rep] = off_dec
                        objs[t][rep] = off_obj[0]
                        cons[t][rep] = off_con[0]

            # Record the maintained populations of every task once per generation
            append_history(all_decs, [decs[t][:, :dims[t]] for t in range(nt)],
                           all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, x0, x1, x2, d_max):
        """
        Generate one offspring with DE/rand/1/bin followed by polynomial mutation.

        Parameters
        ----------
        x0 : np.ndarray
            Base (current) parent in the unified space, shape (d_max,)
        x1, x2 : np.ndarray
            Difference parents in the unified space, shape (d_max,)
        d_max : int
            Unified decision space dimension

        Returns
        -------
        off_dec : np.ndarray
            Offspring decision vector clipped to [0, 1], shape (d_max,)
        """
        # DE mutation: v = x0 + F * (x1 - x2)
        off_dec = x0 + self.F * (x1 - x2)

        # DE binomial crossover against the base parent (MToP DE_Crossover)
        replace = np.random.rand(d_max) > self.CR
        replace[np.random.randint(d_max)] = False
        off_dec = np.where(replace, x0, off_dec)

        # Polynomial mutation, then boundary handling
        off_dec = mutation(off_dec, mu=self.mum)

        return np.clip(off_dec, 0.0, 1.0)
"""
Transfer Task-averaged Natural Gradient - Separable NES (TNG-SNES)

This module implements TNG-SNES for many-task single-objective optimization
using separable Natural Evolution Strategy with task-averaged gradient transfer.

References
----------
    [1] Li, Yanchi, et al. "Transfer Task-averaged Natural Gradient for Efficient
        Many-task Optimization." IEEE Transactions on Evolutionary Computation,
        29(5): 1952-1965, 2025.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class TNG_SNES:
    """
    Transfer Task-averaged Natural Gradient for Many-Task Optimization (Separable NES).

    Uses separable NES with task-averaged natural gradient for knowledge transfer:
    - Each task maintains a Gaussian distribution N(x, diag(S^2))
    - Natural gradients are computed from fitness-ranked utility weights
    - Task-averaged gradient is transferred with adaptive utilization rate (rho)
    - Adaptive transfer control adjusts rho and alpha via virtual parameter comparison

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, rho0=0.1,
                 alpha0=0.7, adj_gap=100, save_data=True, save_path='./Data',
                 name='TNG-SNES', disable_tqdm=True):
        """
        Initialize TNG-SNES algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial standard deviation for all dimensions (default: 0.3)
        rho0 : float, optional
            Initial utilization factor for gradient transfer (default: 0.1)
        alpha0 : float, optional
            Initial transfer rate / probability (default: 0.7)
        adj_gap : int, optional
            Generation interval for adaptive transfer control (default: 100)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'TNG-SNES')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma0 = sigma0
        self.rho0 = rho0
        self.alpha0 = alpha0
        self.adj_gap = adj_gap
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the TNG-SNES algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        maxD = max(dims)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        total_max_nfes = sum(max_nfes_per_task)

        # --- NES utility function (fitness-shaped weights) ---
        shape = np.maximum(0.0, np.log(n / 2 + 1.0) - np.log(np.arange(1, n + 1)))
        shape = shape / np.sum(shape) - 1.0 / n

        # --- Initialize distribution parameters ---
        x = np.zeros((maxD, nt))    # distribution mean per task
        S = np.zeros((maxD, nt))    # distribution std per task
        Gx = np.ones((maxD, nt))    # natural gradient of x
        GS = np.ones((maxD, nt))    # natural gradient of S
        etax = np.ones(nt)          # learning rate for x
        etaS = np.zeros(nt)         # learning rate for S

        for t in range(nt):
            etaS[t] = (3 + np.log(dims[t])) / (5 * np.sqrt(dims[t]))
            x[:, t] = np.mean(np.random.rand(maxD, n), axis=1)
            S[:, t] = self.sigma0

        vx = x.copy()               # virtual x for adaptive transfer control
        vS = S.copy()               # virtual S for adaptive transfer control
        rho = np.full(nt, self.rho0)     # utilization factor per task
        alpha = np.full(nt, self.alpha0) # transfer rate per task

        # --- History tracking ---
        nfes_per_task = [0] * nt
        total_nfes = 0
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        # Current population storage (for history recording)
        cur_decs = [None] * nt
        cur_objs = [None] * nt
        cur_cons = [None] * nt

        pbar = tqdm(total=total_max_nfes, initial=0, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        gen = 1  # 1-indexed to match MATLAB's Gen counter
        while total_nfes < total_max_nfes:
            # === First loop: sampling, evaluation, gradient computation ===
            for t in range(nt):
                # Sampling: Z ~ N(0,1), X = x + S * Z
                Z = np.random.randn(maxD, n)
                X = x[:, t:t + 1] + S[:, t:t + 1] * Z  # (maxD, n)

                # Boundary handling: compute boundary CV from unclipped samples
                X_clipped = np.clip(X, 0, 1)
                bound_cvs = np.sum((X - X_clipped) ** 2, axis=0)  # (n,)

                # Evaluate the raw (unclipped) decisions, exactly as MATLAB
                # does; out-of-range samples are only penalised via bound_cvs
                eval_decs = X[:dims[t]].T  # (n, dims[t])
                task_objs, task_cons = evaluation_single(problem, eval_decs, t)
                nfes_per_task[t] += n
                total_nfes += n
                pbar.update(n)

                # Constraint violation
                if task_cons.shape[1] > 0:
                    cv = np.sum(np.maximum(0, task_cons), axis=1)
                else:
                    cv = np.zeros(n)

                # Add boundary penalty: boundary violators rank worse than
                # the worst constraint-violating but in-bounds individual
                max_cv = np.max(cv)
                bound_cvs[bound_cvs > 0] += max_cv
                total_cv = cv + bound_cvs

                # Sort by total CV then objective (ascending)
                rank_t = np.lexsort((task_objs[:, 0], total_cv))

                # Assign utility weights based on rank
                weights = np.zeros(n)
                weights[rank_t] = shape

                # Store current population for history
                cur_decs[t] = eval_decs
                cur_objs[t] = task_objs
                cur_cons[t] = task_cons

                # --- Adaptive transfer control (every adj_gap generations) ---
                if gen % self.adj_gap == 0:
                    # Generate virtual samples using same Z but virtual params
                    vX = vx[:, t:t + 1] + vS[:, t:t + 1] * Z
                    veval_decs = vX[:dims[t]].T
                    vobjs, vcons = evaluation_single(problem, veval_decs, t)
                    nfes_per_task[t] += n
                    total_nfes += n
                    pbar.update(n)

                    # The virtual probes are genuine evaluations: keep them in
                    # the recorded history so that the evaluation accounting
                    # stays consistent with the stored populations.
                    cur_decs[t] = np.vstack([cur_decs[t], veval_decs])
                    cur_objs[t] = np.vstack([cur_objs[t], vobjs])
                    cur_cons[t] = np.vstack([cur_cons[t], vcons])

                    # Compare mean fitness (raw CV, not boundary-penalized)
                    if vcons.shape[1] > 0:
                        vcv = np.sum(np.maximum(0, vcons), axis=1)
                    else:
                        vcv = np.zeros(n)

                    Fit = 1e8 * np.mean(cv) + np.mean(task_objs[:, 0])
                    vFit = 1e8 * np.mean(vcv) + np.mean(vobjs[:, 0])

                    if vFit > Fit:
                        # Virtual (transfer) worse → reduce transfer
                        rho[t] *= 2.0 / 3.0
                        alpha[t] *= 2.0 / 3.0
                    else:
                        # Virtual better → increase transfer
                        rho[t] = min(1.0, 1.5 * rho[t])
                        alpha[t] = min(1.0, 1.5 * alpha[t])

                # Compute natural gradients for this task
                Gx[:, t] = Z @ weights          # (maxD,)
                GS[:, t] = (Z ** 2 - 1) @ weights  # (maxD,)

            # === Task-averaged natural gradient ===
            TaGx = np.mean(Gx, axis=1)  # (maxD,)
            TaGS = np.mean(GS, axis=1)  # (maxD,)

            # === Second loop: transfer and distribution update ===
            for t in range(nt):
                tGx = Gx[:, t].copy()
                tGS = GS[:, t].copy()

                # Compute virtual parameters (prep for next adj_gap check)
                if (gen + 1) % self.adj_gap == 0:
                    vtGx = tGx + 1.5 * rho[t] * TaGx
                    vtGS = tGS + 1.5 * rho[t] * TaGS
                    vdx = etax[t] * S[:, t] * vtGx
                    vdS = 0.5 * etaS[t] * vtGS
                    vx[:, t] = x[:, t] + vdx
                    vS[:, t] = S[:, t] * np.exp(vdS)

                # Transfer task-averaged natural gradient
                if np.random.rand() < alpha[t] or (gen + 1) % self.adj_gap == 0:
                    tGx += rho[t] * TaGx
                    tGS += rho[t] * TaGS

                # Update distribution parameters
                dx = etax[t] * S[:, t] * tGx
                dS = 0.5 * etaS[t] * tGS
                x[:, t] += dx
                S[:, t] *= np.exp(dS)

            # Record history
            if cur_decs[0] is not None:
                for t in range(nt):
                    all_decs[t].append(cur_decs[t].copy())
                    all_objs[t].append(cur_objs[t].copy())
                    all_cons[t].append(cur_cons[t].copy())

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        # Trim excess evaluations
        all_decs, all_objs, nfes_per_task, all_cons = trim_excess_evaluations(
            all_decs, all_objs, nt, max_nfes_per_task, nfes_per_task, all_cons)

        # Build and save results
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

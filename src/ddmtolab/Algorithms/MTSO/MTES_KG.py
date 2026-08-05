"""
Multi-Task Evolution Strategy with Knowledge-Guided External Sampling (MTES-KG)

This module implements the MTES-KG algorithm for multi-task single-objective optimization.
The algorithm extends CMA-ES with two types of knowledge-guided external sampling across tasks:
DoS (Domain of Solution knowledge) and SaS (Shape of function knowledge), along with adaptive
negative transfer mitigation.

References
----------
    [1] Y. Li, W. Gong, and S. Li. "Multitask Evolution Strategy With Knowledge-Guided External
        Sampling." IEEE Transactions on Evolutionary Computation, 28(6): 1733-1745, 2024.

Notes
-----
The code is developed in accordance with the MATLAB-based MTO-platform framework.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.21
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MTES_KG:
    """
    Multi-Task Evolution Strategy with Knowledge-Guided External Sampling.

    Each task maintains an independent CMA-ES instance. Knowledge transfer between tasks
    is achieved through external samples generated via two strategies:

    - DoS (Domain of Solution knowledge): Samples from an auxiliary task's distribution,
      projected to within the current task's neighborhood
    - SaS (Shape of function knowledge): Transfers the search direction from an auxiliary
      task's successful solutions using CMA-ES coordinate system transformation

    An adaptive mechanism adjusts the number of external samples (tau) to mitigate
    negative transfer.

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

    def __init__(self, problem, n=None, max_nfes=None, tau0=2, alpha=0.5,
                 adj_gap=50, sigma0=0.3, save_data=True, save_path='./Data',
                 name='MTES-KG', disable_tqdm=True):
        """
        Initialize MTES-KG algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size (lambda) per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        tau0 : int, optional
            Initial external sample number per task (default: 2)
        alpha : float, optional
            Probability of using DoS vs SaS for external sampling (default: 0.5)
        adj_gap : int, optional
            Generation gap for adjusting tau (default: 50)
        sigma0 : float, optional
            Initial step size for CMA-ES (default: 0.3)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTES-KG')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.tau0 = tau0
        self.alpha = alpha
        self.adj_gap = adj_gap
        self.sigma0 = sigma0
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTES-KG algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives,
            constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_dim = max(dims)  # CMA-ES operates in max dimension space
        lam = self.n  # lambda: population size per task
        mu = (lam + 1) // 2  # number of effective parents (MATLAB round(lambda/2))
        max_nfes_per_task = par_list(self.max_nfes, nt)
        total_max_nfes = self.max_nfes * nt

        # Per-task CMA-ES state initialization
        cma_params = []
        for t in range(nt):
            params = cmaes_init_params(n_dim, lam=lam, sigma0=self.sigma0)
            # MTES_KG.m:66 starts from the mean of lambda uniform samples
            # (near the centre of the unified space), not from a single
            # rand(1, n) draw as the shared CMA-ES helper does.
            params['m_dec'] = np.mean(np.random.rand(lam, n_dim), axis=0)
            cma_params.append(params)

        # MTES_KG-specific per-task variables
        m_step = [0.0] * nt
        tau = [self.tau0] * nt
        num_ex_s = [[] for _ in range(nt)]
        suc_ex_s = [[] for _ in range(nt)]

        # MATLAB performs no separate initialization evaluation: the first
        # loop body already samples, evaluates and updates the distributions.
        nfes = 0
        nfes_per_task = [0] * nt
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        # Storage for previous generation data (needed for SaS)
        old_samples = [None] * nt
        old_rank = [None] * nt

        gen = 0
        pbar = tqdm(total=total_max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < total_max_nfes:
            gen += 1

            # === Step 1: Sample new internal solutions for each task ===
            internal_samples = [None] * nt
            for t in range(nt):
                internal_samples[t] = cmaes_sample(cma_params[t]['m_dec'], cma_params[t]['sigma'],
                                                   cma_params[t]['B'], cma_params[t]['D'], lam,
                                                   clip=False)
                # Compute mean step size (average distance from mean)
                diffs = internal_samples[t] - cma_params[t]['m_dec']
                m_step[t] = np.mean(np.sqrt(np.sum(diffs ** 2, axis=1)))

            # === Step 2: Sample external solutions (DoS/SaS) ===
            external_samples = [None] * nt
            for t in range(nt):
                if tau[t] == 0:
                    external_samples[t] = np.empty((0, n_dim))
                    continue

                # Select random auxiliary task
                other_tasks = list(range(nt))
                other_tasks.remove(t)
                k = other_tasks[np.random.randint(len(other_tasks))]

                ext = np.zeros((tau[t], n_dim))
                p_t = cma_params[t]
                p_k = cma_params[k]
                for i in range(tau[t]):
                    if gen < 2:
                        # First generation: random sample from own distribution
                        z_i = np.random.randn(n_dim)
                        ext[i] = p_t['m_dec'] + p_t['sigma'] * (p_t['B'] @ (p_t['D'] * z_i))
                    elif np.random.rand() < self.alpha:
                        # DoS: Domain knowledge-guided external sampling
                        z_i = np.random.randn(n_dim)
                        sample_k = p_k['m_dec'] + p_k['sigma'] * (p_k['B'] @ (p_k['D'] * z_i))
                        vec = sample_k - p_t['m_dec']
                        vec_norm = np.linalg.norm(vec)
                        if vec_norm < m_step[t]:
                            ext[i] = sample_k
                        else:
                            uni_vec = vec / (vec_norm + 1e-30)
                            ext[i] = p_t['m_dec'] + uni_vec * m_step[t]
                    else:
                        # SaS: Shape knowledge-guided external sampling
                        idx = list(range(mu))
                        idx.pop(np.random.randint(len(idx)))
                        top_decs_k = old_samples[k][old_rank[k][idx]]
                        vec_mean = np.mean(top_decs_k, axis=0)
                        vec = (vec_mean - p_k['m_dec']) / p_k['sigma']
                        # Transform from k's coordinate system to t's
                        transformed = p_t['B'] @ (p_t['D'] * (p_k['B'].T @ (vec / p_k['D'])))
                        ext[i] = p_t['m_dec'] + p_t['sigma'] * transformed

                external_samples[t] = ext

            # === Step 3: Evaluate, sort, update CMA-ES for each task ===
            cur_decs = [None] * nt
            cur_objs = [None] * nt
            cur_cons = [None] * nt

            for t in range(nt):
                # Concatenate internal and external samples
                if external_samples[t] is not None and len(external_samples[t]) > 0:
                    all_sample_decs = np.vstack([internal_samples[t], external_samples[t]])
                else:
                    all_sample_decs = internal_samples[t].copy()

                total_samples = len(all_sample_decs)

                # Boundary constraint violation (extended bounds [-0.05, 1.05])
                clipped = np.clip(all_sample_decs, -0.05, 1.05)
                bound_cv = np.sum((all_sample_decs - clipped) ** 2, axis=1)

                # Evaluate the raw (unclipped) decisions, exactly as MATLAB
                # does; out-of-range samples are only penalised via bound_cv
                eval_decs = all_sample_decs[:, :dims[t]]
                objs_t, cons_t = evaluation_single(problem, eval_decs, t)
                nfes += total_samples
                nfes_per_task[t] += total_samples
                pbar.update(total_samples)

                # Compute constraint violation
                if cons_t is not None and cons_t.size > 0:
                    cvs = np.sum(np.maximum(0, cons_t), axis=1)
                else:
                    cvs = np.zeros(total_samples)

                # Add boundary violation penalty
                if np.any(bound_cv > 0) and np.max(cvs) > 0:
                    bound_cv[bound_cv > 0] += np.max(cvs)
                total_cv = cvs + bound_cv

                # Sort by [total_cv, obj] lexicographically (ascending)
                rank_t = constrained_sort(objs_t, total_cv)

                # Track external sample success (how many external samples are in top mu)
                num_ex_s[t].append(tau[t])
                suc_count = int(np.sum(rank_t[:mu] >= lam))
                suc_ex_s[t].append(suc_count)

                # Negative transfer mitigation: adjust tau periodically
                if gen % self.adj_gap == 0 and gen > 0:
                    start_idx = max(0, len(num_ex_s[t]) - self.adj_gap)
                    num_all = sum(num_ex_s[t][start_idx:])
                    suc_all = sum(suc_ex_s[t][start_idx:])
                    if (num_all > 0 and suc_all / num_all > 0.5) or num_all == 0:
                        tau[t] = min(self.tau0, tau[t] + 1)
                    else:
                        tau[t] = max(0, tau[t] - 1)

                # Save current samples and rank for next generation's SaS
                old_samples[t] = all_sample_decs.copy()
                old_rank[t] = rank_t.copy()

                # ===== CMA-ES parameter update =====
                # MTES_KG.m:154 uses ceil((FE - lambda*(t-1)) / (lambda*T)) + 1
                # as the generation counter driving hsig; with the shared helper
                # (which divides nfes by lambda) that is exactly (Gen + 1)*lambda.
                restarted = cmaes_update(cma_params[t], all_sample_decs[rank_t],
                                         (gen + 1) * lam)
                if restarted:
                    cma_params[t]['ps'] = np.zeros(n_dim)
                    cma_params[t]['pc'] = np.zeros(n_dim)
                    cma_params[t]['sigma'] = min(max(2 * cma_params[t]['sigma'], 0.01), 0.3)

                # Store every evaluated sample (best first) so that the recorded
                # history stays consistent with the consumed evaluations
                cur_decs[t] = all_sample_decs[rank_t][:, :dims[t]]
                cur_objs[t] = objs_t[rank_t]
                if cons_t is not None and cons_t.size > 0:
                    cur_cons[t] = cons_t[rank_t]
                else:
                    cur_cons[t] = np.zeros((total_samples, 0))

            append_history(all_decs, cur_decs, all_objs, cur_objs, all_cons, cur_cons)

        pbar.close()
        runtime = time.time() - start_time

        # Samples are stored best-first, so trimming the surplus tail keeps the
        # elite of the final generation intact
        all_decs, all_objs, nfes_per_task, all_cons = trim_excess_evaluations(
            all_decs, all_objs, nt, max_nfes_per_task, nfes_per_task, all_cons)

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data
        )

        return results

"""
Evolutionary Expected Improvement based Bayesian Optimization for MTOPs (EEI-BO+)

Multi-task extension of EEI-BO. Each task keeps its own GP surrogate and its own
persistent CMA-ES search distribution, which advances one generation per BO
iteration. Knowledge is transferred by periodically letting one task's EEI be
guided by another task's CMA-ES distribution: every ``switch_interval`` no-transfer
iterations, a single transfer iteration replaces each task's own search
distribution with the (dimension-adapted) distribution of another task, following
the reference's global 6-no-transfer / 1-transfer schedule.

References
----------
    [1] J. Liu, Y. Wang, G. Sun, and T. Pang, "Solving Highly Expensive
        Optimization Problems via Evolutionary Expected Improvement," IEEE
        Transactions on Systems, Man, and Cybernetics: Systems, vol. 53, no. 8,
        pp. 4843-4855, 2023.

Notes
-----
Corrected against the reference implementation (MT-EEI-BO, v2.0):

- Persistent per-task CMA-ES advancing one generation per iteration (was a full
  CMA-ES restart every iteration); shares the ST-EEI-BO components.
- A single global transfer schedule shared by all tasks (was tracked per task and
  could desync): ``switch_interval`` no-transfer iterations then one transfer
  iteration, repeating.
- Transfer uses the other task's CMA-ES distribution snapshot from the same
  iteration (all tasks advance CMA-ES first, then all tasks query), matching the
  reference's two-pass structure.
- Cross-task distribution mapping is dimension pad/truncation in the unified
  [0, 1] space (the reference's bound rescaling is the identity there).
- EEI maximized in log space; see EEI_BO for the shared acquisition details.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.07.18
Version: 2.0
"""
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

from ddmtolab.Algorithms.STSO.EEI_BO import (
    init_cma_state, fit_gp_ei, cma_step, eei_next_point,
)
from ddmtolab.Methods.Algo_Methods.algo_utils import *

warnings.filterwarnings("ignore")


class EEI_BO_plus:
    """
    Evolutionary Expected Improvement based Bayesian Optimization for MTOPs.

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
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, switch_interval=6,
                 cma_popsize=100, sigma0=0.5, n2=30, max_nfes2=6000,
                 save_data=True, save_path='./Data', name='EEI-BO+', disable_tqdm=True):
        """
        Initialize EEI-BO+ algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        switch_interval : int, optional
            Number of no-transfer iterations between transfer iterations
            (default: 6, the reference's 6-no-transfer / 1-transfer schedule)
        cma_popsize : int, optional
            Number of surrogate-ranked samples per CMA-ES generation (default: 100)
        sigma0 : float, optional
            Initial CMA-ES step size in the [0, 1] space (default: 0.5)
        n2 : int, optional
            Population size of the DE that maximizes EEI (default: 30)
        max_nfes2 : int, optional
            Function evaluations of the DE that maximizes EEI (default: 6000)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EEI-BO+')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.switch_interval = switch_interval
        self.cma_popsize = cma_popsize
        self.sigma0 = sigma0
        self.n2 = n2
        self.max_nfes2 = max_nfes2
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute EEI-BO+.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.double
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Persistent CMA-ES state per task, seeded at each task's sample mean
        cma_states = [init_cma_state(decs[i], dims[i], self.cma_popsize, self.sigma0)
                      for i in range(nt)]

        # Global transfer schedule (shared by all tasks)
        transfer_mode = False
        mode_counter = 1

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Pass 1: fit GP and advance each active task's CMA-ES one generation
            gps, logeis = {}, {}
            for i in active_tasks:
                gp, logEI = fit_gp_ei(decs[i], objs[i], data_type)
                gps[i], logeis[i] = gp, logEI
                cma_step(cma_states[i], gp, data_type)

            # Snapshot post-update distributions for cross-task transfer
            snapshots = {i: (cma_states[i]['m_dec'].copy(),
                             cma_states[i]['sigma'] ** 2 * cma_states[i]['C'].copy())
                         for i in active_tasks}

            # Pass 2: build EEI (own or transferred distribution) and query each task
            for i in active_tasks:
                if transfer_mode:
                    sources = [t for t in active_tasks if t != i]
                    if sources:
                        source = int(np.random.choice(sources))
                        mu_s, Sigma_s = snapshots[source]
                        mu, Sigma_real = _adapt_distribution(mu_s, Sigma_s, dims[source], dims[i])
                    else:
                        mu, Sigma_real = snapshots[i]
                else:
                    mu, Sigma_real = snapshots[i]

                candidate = eei_next_point(logeis[i], mu, Sigma_real, dims[i],
                                           self.n2, self.max_nfes2, data_type)

                new_objs, _ = evaluation_single(problem, candidate, i)
                decs[i], objs[i] = vstack_groups((decs[i], candidate), (objs[i], new_objs))

                nfes_per_task[i] += 1
                pbar.update(1)

            # Advance the global schedule once per iteration
            if (not transfer_mode) and mode_counter == self.switch_interval:
                transfer_mode = True
                mode_counter = 0
            elif transfer_mode:
                transfer_mode = False
            mode_counter += 1

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


def _adapt_distribution(mu_source, Sigma_source, dim_source, dim_target):
    """
    Map a source task's CMA-ES distribution (mu, Sigma) to a target task's space.

    In DDMTOLab all tasks share the unified [0, 1] space, so the reference's
    bound rescaling is the identity and only dimensionality is adjusted:
    truncate to the first dims when the source is larger, pad the mean with
    uniform-random coordinates and the covariance with the identity when the
    target is larger.

    Parameters
    ----------
    mu_source : np.ndarray
        Source mean, shape (dim_source,).
    Sigma_source : np.ndarray
        Source real covariance sigma^2 C, shape (dim_source, dim_source).
    dim_source, dim_target : int
        Source and target dimensionalities.

    Returns
    -------
    mu_new, Sigma_new : tuple(np.ndarray, np.ndarray)
        Adapted mean (dim_target,) and covariance (dim_target, dim_target).
    """
    if dim_source == dim_target:
        return mu_source.copy(), Sigma_source.copy()

    if dim_source > dim_target:
        mu_new = mu_source[:dim_target].copy()
        Sigma_new = Sigma_source[:dim_target, :dim_target].copy()
    else:
        mu_new = np.concatenate([mu_source, np.random.rand(dim_target - dim_source)])
        Sigma_new = np.eye(dim_target)
        Sigma_new[:dim_source, :dim_source] = Sigma_source

    return mu_new, Sigma_new

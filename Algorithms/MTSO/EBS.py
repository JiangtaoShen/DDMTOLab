"""
EBS (Evolutionary Biocoenosis-based Symbiosis)

This module implements the EBS algorithm for evolutionary many-tasking optimization.

References
----------
    [1] Liaw, R. T., & Ting, C. K. (2019). Evolutionary many-tasking based on biocoenosis through symbiosis: A framework and benchmark problems. In 2019 IEEE Congress on Evolutionary Computation (CEC) (pp. 2266-2273). IEEE.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.09
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from Methods.Algo_Methods.algo_utils import *


class EBS:
    """
    Evolutionary Biocoenosis-based Symbiosis for many-task optimization.

    EBS uses multiple CMA-ES instances with adaptive information exchange among tasks.
    Each task maintains two CMA-ES distributions:
    - One updated when knowledge transfer occurs
    - One updated when no knowledge transfer occurs

    The information exchange probability is controlled adaptively based on the
    improvement ratio from self-generated offspring versus offspring from other tasks.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
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

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, use_n=True,
                 gen_init=10, save_data=True, save_path='./TestData',
                 name='EBS_test', disable_tqdm=True):
        """
        Initialize EBS Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: None, will use 4+3*log(D))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial step size for CMA-ES (default: 0.3)
        use_n : bool, optional
            If True, use provided n; if False, use 4+3*log(D) (default: True)
        gen_init : int, optional
            Number of initial generations for alternating CMA-ES before using gamma (default: 10)
            During this phase, two CMA-ES alternate (one without transfer, one with transfer)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EBS_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma0 = sigma0
        self.use_n = use_n
        self.gen_init = gen_init
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EBS Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d_max = np.max(dims)  # Unified dimension
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize CMA-ES parameters for each task (using unified dimension)
        params = []
        for t in range(nt):
            # Use unified dimension for optimization
            dim = d_max

            # Determine population size based on original task dimension
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dims[t]))

            mu = int(lam / 2)

            # Recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)

            # Step size control parameters
            cs = (mueff + 2) / (dim + mueff + 5)
            damps = 1 + cs + 2 * max(np.sqrt((mueff - 1) / (dim + 1)) - 1, 0)

            # Covariance update parameters
            cc = (4 + mueff / dim) / (4 + dim + 2 * mueff / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + 2 * mueff / 2))

            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Initialize mean vector (random in unified space)
            m_dec_init = np.random.rand(dim)

            # Distribution for self-generated offspring (no knowledge transfer)
            params_s = {
                'm_dec': m_dec_init.copy(),
                'ps': np.zeros(dim),
                'pc': np.zeros(dim),
                'B': np.eye(dim),
                'D': np.ones(dim),
                'C': np.eye(dim),
                'invsqrtC': np.eye(dim),
                'sigma': self.sigma0,
                'eigenFE': 0
            }

            # Distribution for knowledge transfer offspring
            params_o = {
                'm_dec': m_dec_init.copy(),
                'ps': np.zeros(dim),
                'pc': np.zeros(dim),
                'B': np.eye(dim),
                'D': np.ones(dim),
                'C': np.eye(dim),
                'invsqrtC': np.eye(dim),
                'sigma': self.sigma0,
                'eigenFE': 0
            }

            params.append({
                'dim': dim,  # Unified dimension
                'real_dim': dims[t],  # Real dimension for this task
                'lam': lam,
                'mu': mu,
                'weights': weights,
                'mueff': mueff,
                'cs': cs,
                'damps': damps,
                'cc': cc,
                'c1': c1,
                'cmu': cmu,
                'chiN': chiN,
                'params_s': params_s,  # Self distribution (no transfer)
                'params_o': params_o,  # Other distribution (with transfer)
            })

        # Initialize tracking variables
        nfes_per_task = [0] * nt
        decs = [None] * nt  # In unified space
        objs = [None] * nt
        cons = [None] * nt
        best_objs = [np.inf] * nt  # Best-so-far objective for each task
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        # Initialize information exchange statistics
        improvements_s = [0] * nt
        evals_s = [0] * nt
        improvements_o = [0] * nt
        evals_o = [0] * nt
        gamma = [0.0] * nt  # Probability of information exchange (will be computed after init phase)

        # Initialize generation counter for each task (for alternating during init phase)
        gen_count = [0] * nt

        pbar = tqdm(total=sum(max_nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Step 1: Determine transfer flags and generate offspring accordingly
            transfer_flags = []
            offspring_list = []

            for i in active_tasks:
                p = params[i]

                # Decide whether to perform knowledge transfer
                if gen_count[i] < self.gen_init:
                    # During init phase: alternate between no transfer and transfer
                    # Even generations: no transfer (is_transfer=False)
                    # Odd generations: transfer (is_transfer=True)
                    is_transfer = (gen_count[i] % 2 == 1)
                else:
                    # After init phase: use gamma probability
                    is_transfer = np.random.rand() < gamma[i]

                transfer_flags.append(is_transfer)

                # Select distribution based on transfer decision
                if is_transfer:
                    ps = p['params_o']  # Use knowledge transfer distribution
                else:
                    ps = p['params_s']  # Use self distribution

                # Generate offspring using selected CMA-ES distribution
                sample_decs = ebs_cmaes_generation(
                    m_dec=ps['m_dec'],
                    sigma=ps['sigma'],
                    B=ps['B'],
                    D=ps['D'],
                    lam=p['lam']
                )
                offspring_list.append(sample_decs)

            # Concatenate all offspring (all in unified space d_max)
            concat_offspring = np.vstack(offspring_list)

            # Step 2: For each task, select candidates based on transfer decision
            candidate_list = []

            for idx, i in enumerate(active_tasks):
                p = params[i]
                is_transfer = transfer_flags[idx]

                if is_transfer:
                    # Perform knowledge transfer: sample from concatenate offspring
                    # Randomly select lambda candidates from concatenate offspring
                    n_candidates = min(p['lam'], concat_offspring.shape[0])
                    candidate_indices = np.random.choice(
                        concat_offspring.shape[0],
                        size=n_candidates,
                        replace=False
                    )
                    candidate_decs = concat_offspring[candidate_indices].copy()

                    # If we need more candidates, duplicate some
                    while candidate_decs.shape[0] < p['lam']:
                        extra_idx = np.random.choice(concat_offspring.shape[0])
                        candidate_decs = np.vstack([candidate_decs, concat_offspring[extra_idx:extra_idx + 1]])

                    candidate_list.append(candidate_decs)
                else:
                    # No knowledge transfer: use self-generated offspring
                    candidate_list.append(offspring_list[idx])

            # Step 3: Evaluate, update population and CMA-ES parameters
            for idx, i in enumerate(active_tasks):
                p = params[i]
                candidate_decs = candidate_list[idx]  # In unified space
                is_transfer = transfer_flags[idx]

                # Convert to real space for evaluation (truncate to real dimension)
                candidate_decs_real = candidate_decs[:, :dims[i]]
                candidate_decs_real = np.clip(candidate_decs_real, 0, 1)  # Ensure bounds

                # Evaluate candidates (in real space)
                sample_objs, sample_cons = evaluation_single(problem, candidate_decs_real, i)

                # Sort by constraint violation first, then by objective
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                sort_indices = np.lexsort((sample_objs.flatten(), cvs))

                # Sort in unified space
                sorted_decs = candidate_decs[sort_indices]
                sorted_objs = sample_objs[sort_indices]
                sorted_cons = sample_cons[sort_indices]

                # Update evaluation counts
                if is_transfer:
                    evals_o[i] += p['lam']
                else:
                    evals_s[i] += p['lam']

                # Check if best-so-far is improved
                best_candidate_obj = sorted_objs[0, 0]
                if best_candidate_obj < best_objs[i]:
                    best_objs[i] = best_candidate_obj
                    if is_transfer:
                        improvements_o[i] += 1
                    else:
                        improvements_s[i] += 1

                # Increment generation counter
                gen_count[i] += 1

                # Update gamma after init phase is complete
                if gen_count[i] == self.gen_init:
                    # Compute gamma based on accumulated statistics
                    if evals_s[i] > 0 and evals_o[i] > 0:
                        R_s = improvements_s[i] / evals_s[i]
                        R_o = improvements_o[i] / evals_o[i]
                        if (R_s + R_o) > 0:
                            gamma[i] = R_o / (R_s + R_o)
                        else:
                            gamma[i] = 0.0  # Default if no improvements
                    else:
                        gamma[i] = 0.0  # Default
                elif gen_count[i] > self.gen_init:
                    # Continue updating gamma based on accumulated statistics
                    if evals_s[i] > 0 and evals_o[i] > 0:
                        R_s = improvements_s[i] / evals_s[i]
                        R_o = improvements_o[i] / evals_o[i]
                        if (R_s + R_o) > 0:
                            gamma[i] = R_o / (R_s + R_o)

                # Update current population (store in unified space)
                decs[i] = sorted_decs
                objs[i] = sorted_objs
                cons[i] = sorted_cons

                nfes_per_task[i] += p['lam']
                pbar.update(p['lam'])

                # Convert to real space for history (truncate to real dimension)
                decs_real = sorted_decs[:, :dims[i]]
                append_history(all_decs[i], decs_real, all_objs[i], sorted_objs, all_cons[i], sorted_cons)

                # Update the appropriate CMA-ES distribution
                if is_transfer:
                    # Update knowledge transfer CMA-ES
                    _update_cmaes_params(
                        p['params_o'], sorted_decs, p,
                        nfes_per_task[i]
                    )
                else:
                    # Update self CMA-ES
                    _update_cmaes_params(
                        p['params_s'], sorted_decs, p,
                        nfes_per_task[i]
                    )

        pbar.close()
        runtime = time.time() - start_time

        # Collect final EBS parameters
        ebs_params = []
        for i in range(nt):
            p = params[i]
            ebs_params.append({
                'params_s': {k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in p['params_s'].items()},
                'params_o': {k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in p['params_o'].items()},
                'gamma': gamma[i],
                'improvements_s': improvements_s[i],
                'improvements_o': improvements_o[i],
                'evals_s': evals_s[i],
                'evals_o': evals_o[i],
            })

        # Save results (all_decs are already in real space)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name, save_data=self.save_data
        )

        return results, ebs_params


def _update_cmaes_params(cmaes_params, sorted_decs, task_params, nfes):
    """
    Update CMA-ES distribution parameters.

    Parameters
    ----------
    cmaes_params : dict
        CMA-ES parameter dictionary to update (params_s or params_o)
    sorted_decs : np.ndarray
        Sorted offspring decision variables (best first), shape (lam, dim)
    task_params : dict
        Task-level parameters (weights, mu, cs, cc, c1, cmu, etc.)
    nfes : int
        Current number of function evaluations
    """
    p = cmaes_params
    tp = task_params

    # Update mean decision variables
    old_dec = p['m_dec'].copy()
    p['m_dec'] = tp['weights'] @ sorted_decs[:tp['mu']]

    # Update evolution paths
    diff = (p['m_dec'] - old_dec) / p['sigma']
    p['ps'] = (1 - tp['cs']) * p['ps'] + \
              np.sqrt(tp['cs'] * (2 - tp['cs']) * tp['mueff']) * (p['invsqrtC'] @ diff)

    ps_norm = np.linalg.norm(p['ps'])
    hsig = ps_norm / np.sqrt(1 - (1 - tp['cs']) ** (2 * nfes / tp['lam'])) / tp['chiN'] < 1.4 + 2 / (tp['dim'] + 1)

    p['pc'] = (1 - tp['cc']) * p['pc'] + \
              hsig * np.sqrt(tp['cc'] * (2 - tp['cc']) * tp['mueff']) * diff

    # Update covariance matrix
    artmp = (sorted_decs[:tp['mu']] - old_dec) / p['sigma']
    delta = (1 - hsig) * tp['cc'] * (2 - tp['cc'])
    p['C'] = (1 - tp['c1'] - tp['cmu']) * p['C'] + \
             tp['c1'] * (np.outer(p['pc'], p['pc']) + delta * p['C']) + \
             tp['cmu'] * (artmp.T @ np.diag(tp['weights']) @ artmp)

    # Update step size
    p['sigma'] = p['sigma'] * np.exp(tp['cs'] / tp['damps'] * (ps_norm / tp['chiN'] - 1))

    # Update eigendecomposition periodically
    if nfes - p['eigenFE'] > tp['lam'] / (tp['c1'] + tp['cmu']) / tp['dim'] / 10:
        p['eigenFE'] = nfes
        p['C'] = np.triu(p['C']) + np.triu(p['C'], 1).T

        eigvals, eigvecs = np.linalg.eigh(p['C'])

        if np.min(eigvals) <= 0:
            p['B'] = np.eye(tp['dim'])
            p['D'] = np.ones(tp['dim'])
            p['C'] = np.eye(tp['dim'])
            print(f"Warning: Covariance matrix not positive definite, resetting to identity.")
        else:
            p['B'] = eigvecs
            p['D'] = np.sqrt(eigvals)

        p['invsqrtC'] = p['B'] @ np.diag(1.0 / p['D']) @ p['B'].T


def ebs_cmaes_generation(m_dec: np.ndarray, sigma: float, B: np.ndarray,
                         D: np.ndarray, lam: int = None) -> np.ndarray:
    """
    Generate offspring population using CMA-ES sampling strategy.

    Parameters
    ----------
    m_dec : np.ndarray
        Mean decision vector, shape (d,)
    sigma : float
        Step size (global scaling factor)
    B : np.ndarray
        Eigenvector matrix from covariance matrix decomposition, shape (d, d)
    D : np.ndarray
        Square root of eigenvalues (standard deviations), shape (d,)
    lam : int, optional
        Number of offspring to generate (default: None)

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (lam, d)
    """
    d = len(m_dec)

    if lam is None:
        lam = int(4 + 3 * np.log(d))

    offdecs = np.zeros((lam, d))

    for i in range(lam):
        z = np.random.randn(d)
        offdec = m_dec + sigma * (B @ (D * z))
        offdec = np.clip(offdec, 0, 1)
        offdecs[i] = offdec

    return offdecs
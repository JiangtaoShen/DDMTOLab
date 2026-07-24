"""
sep-CMA-ES (Separable Covariance Matrix Adaptation Evolution Strategy)

This module implements the sep-CMA-ES algorithm for single-objective optimization problems.
sep-CMA-ES achieves linear time and space complexity by using a diagonal covariance matrix.

References
----------
    [1] Ros, R., & Hansen, N. (2008). A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity. Parallel Problem Solving from Nature, PPSN X, 296-305. DOI: 10.1007/978-3-540-87700-4_30

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.27
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class sep_CMA_ES:
    """
    sep-CMA-ES for single-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, use_n=True,
                 save_data=True, save_path='./Data', name='sep-CMA-ES', disable_tqdm=True):
        """
        Initialize sep-CMA-ES Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: None, will use 4+3*log(D))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial step size (default: 0.3)
        use_n : bool, optional
            If True, use provided n; if False, use 4+3*log(D) (default: True)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'sep_CMA_ES_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma0 = sigma0
        self.use_n = use_n
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the sep-CMA-ES Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize parameters for each task
        params = []
        for t in range(nt):
            dim = problem.dims[t]

            # Determine population size
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dim))

            mu = (lam + 1) // 2  # MATLAB: mu = round(lambda / 2), round-half-away-from-zero

            # Recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)

            # Step size control parameters (different from standard CMA-ES)
            cs = (mueff + 2) / (dim + mueff + 3)
            damps = 1 + cs + 2 * max(np.sqrt((mueff - 1) / (dim + 1)) - 1, 0)

            # Covariance update parameters (different from standard CMA-ES)
            cc = 4 / (4 + dim)
            ccov = (1 / mueff) * (2 / (dim + np.sqrt(2)) ** 2) + \
                   (1 - 1 / mueff) * min(1, (2 * mueff - 1) / ((dim + 2) ** 2 + mueff))
            ccov = (dim + 2) / 3 * ccov

            # Initialize
            m_dec = np.random.rand(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            C = np.ones(dim)  # Diagonal covariance matrix (vector form)
            sigma = self.sigma0
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            params.append({
                'dim': dim, 'lam': lam, 'mu': mu, 'weights': weights, 'mueff': mueff,
                'cs': cs, 'damps': damps, 'cc': cc, 'ccov': ccov,
                'm_dec': m_dec, 'ps': ps, 'pc': pc, 'C': C,
                'sigma': sigma, 'chiN': chiN
            })

        # Initialize tracking variables
        nfes_per_task = [0] * nt
        decs = [None] * nt
        objs = [None] * nt
        cons = [None] * nt
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                p = params[i]

                # Generate offspring using sep_cmaes_generation
                sample_decs = sep_cmaes_generation(
                    m_dec=p['m_dec'],
                    sigma=p['sigma'],
                    C=p['C'],
                    lam=p['lam']
                )

                # Evaluate samples
                sample_objs, sample_cons = evaluation_single(problem, sample_decs, i)

                # Sort by constraint violation first, then by objective
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                sort_indices = np.lexsort((sample_objs.flatten(), cvs))

                sample_decs = sample_decs[sort_indices]
                sample_objs = sample_objs[sort_indices]
                sample_cons = sample_cons[sort_indices]

                # Update current population
                decs[i] = sample_decs
                objs[i] = sample_objs
                cons[i] = sample_cons

                nfes_per_task[i] += p['lam']
                pbar.update(p['lam'])

                # Append to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

                # Update mean decision variables
                old_dec = p['m_dec'].copy()
                p['m_dec'] = p['weights'] @ sample_decs[:p['mu']]

                # Update evolution paths
                diff = (p['m_dec'] - old_dec) / p['sigma']
                p['ps'] = (1 - p['cs']) * p['ps'] + \
                          np.sqrt(p['cs'] * (2 - p['cs']) * p['mueff']) * (diff / np.sqrt(p['C']))

                ps_norm = np.linalg.norm(p['ps'])
                hsig = ps_norm / np.sqrt(1 - (1 - p['cs']) ** (2 * nfes_per_task[i] / p['lam'])) / p[
                    'chiN'] < 1.4 + 2 / (p['dim'] + 1)

                p['pc'] = (1 - p['cc']) * p['pc'] + \
                          hsig * np.sqrt(p['cc'] * (2 - p['cc']) * p['mueff']) * diff

                # Update diagonal covariance matrix
                artmp = (sample_decs[:p['mu']] - old_dec) / p['sigma']
                delta = (1 - hsig) * p['cc'] * (2 - p['cc'])
                p['C'] = (1 - p['ccov']) * p['C'] + \
                         (p['ccov'] / p['mueff']) * (p['pc'] ** 2 + delta * p['C']) + \
                         p['ccov'] * (1 - 1 / p['mueff']) * (artmp.T ** 2 @ p['weights'])

                # Update step size
                p['sigma'] = p['sigma'] * np.exp(p['cs'] / p['damps'] * (ps_norm / p['chiN'] - 1))

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(
            all_decs=all_decs,
            all_objs=all_objs,
            runtime=runtime,
            max_nfes=nfes_per_task,
            all_cons=all_cons,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        return results


def sep_cmaes_generation(m_dec: np.ndarray, sigma: float, C: np.ndarray, lam: int = None) -> np.ndarray:
    """
    Generate offspring population using sep-CMA-ES sampling strategy.

    Uses diagonal covariance matrix for linear time and space complexity.

    Parameters
    ----------
    m_dec : np.ndarray
        Mean decision vector, shape (d,)
    sigma : float
        Step size (global scaling factor)
    C : np.ndarray
        Diagonal elements of covariance matrix, shape (d,)
    lam : int, optional
        Number of offspring to generate (default: None)

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (lam, d)
    """
    d = len(m_dec)

    # If lam is None, generate a default number
    if lam is None:
        lam = int(4 + 3 * np.log(d))

    offdecs = np.zeros((lam, d))

    for i in range(lam):
        z = np.random.randn(d)
        offdec = m_dec + sigma * (np.sqrt(C) * z)
        offdec = np.clip(offdec, 0, 1)
        offdecs[i] = offdec

    return offdecs
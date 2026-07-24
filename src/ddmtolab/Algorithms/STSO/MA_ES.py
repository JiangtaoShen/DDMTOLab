"""
MA-ES (Matrix Adaptation Evolution Strategy)

This module implements the MA-ES algorithm for single-objective optimization problems.
MA-ES uses matrix adaptation instead of covariance matrix adaptation for efficiency.

References
----------
    [1] Bayer, H. G., & Sendhoff, B. (2017). Simplify Your Covariance Matrix Adaptation Evolution Strategy. IEEE Transactions on Evolutionary Computation, 21(5), 746-759. DOI: 10.1109/TEVC.2017.2680320

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


class MA_ES:
    """
    MA-ES for single-objective optimization.

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
                 save_data=True, save_path='./Data', name='MA-ES', disable_tqdm=True):
        """
        Initialize MA-ES Algorithm.

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
            Name for the experiment (default: 'MA_ES_test')
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
        Execute the MA-ES Algorithm.

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

            # Determine population size (lambda)
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dim))

            mu = lam // 2  # Beyer & Sendhoff (2017): mu = floor(lambda / 2)

            # Recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)

            # Step size control parameters
            cs = (mueff + 2) / (dim + mueff + 5)
            damps = 1 + cs + 2 * max(np.sqrt((mueff - 1) / (dim + 1)) - 1, 0)

            # Matrix adaptation parameters (c1 and cw correspond to c1 and cmu in CMA-ES)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cw = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + 2 * mueff / 2))

            # Initialize (M1)
            y = np.random.rand(dim)  # mean in search space
            s = np.zeros(dim)  # evolution path for step size
            M = np.eye(dim)  # transformation matrix
            sigma = self.sigma0
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            params.append({
                'dim': dim, 'lam': lam, 'mu': mu, 'weights': weights, 'mueff': mueff,
                'cs': cs, 'damps': damps, 'c1': c1, 'cw': cw,
                'y': y, 's': s, 'M': M, 'sigma': sigma, 'chiN': chiN
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

                # M4-M5: Sample z and compute d
                z = np.random.randn(p['lam'], p['dim'])  # (M4)
                d = z @ p['M'].T  # (M5): d̃ = M z̃

                # M6: Generate and evaluate offspring
                sample_decs = p['y'] + p['sigma'] * d
                sample_decs = np.clip(sample_decs, 0, 1)  # boundary handling

                sample_objs, sample_cons = evaluation_single(problem, sample_decs, i)

                # M8: Sort offspring population
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                sort_indices = np.lexsort((sample_objs.flatten(), cvs))

                sample_decs = sample_decs[sort_indices]
                sample_objs = sample_objs[sort_indices]
                sample_cons = sample_cons[sort_indices]
                z = z[sort_indices]  # sort z accordingly
                d = d[sort_indices]  # sort d accordingly

                # Update current population
                decs[i] = sample_decs
                objs[i] = sample_objs
                cons[i] = sample_cons

                nfes_per_task[i] += p['lam']
                pbar.update(p['lam'])

                # Append to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

                # M9: Update mean - y(g+1) = y(g) + σ(g) ⟨d̃(g)⟩_w
                d_weighted = p['weights'] @ d[:p['mu']]
                p['y'] = p['y'] + p['sigma'] * d_weighted

                # M10: Update evolution path s - in z-space
                z_weighted = p['weights'] @ z[:p['mu']]
                p['s'] = (1 - p['cs']) * p['s'] + \
                         np.sqrt(p['mueff'] * p['cs'] * (2 - p['cs'])) * z_weighted

                # M11: Update transformation matrix M
                # M(g+1) = M(g) * [I + c1/2 * (s*s' - I) + cw/2 * (⟨z*z'⟩_w - I)]
                s_outer = np.outer(p['s'], p['s'])

                # Weighted sum of z outer products
                z_outer_w = np.zeros((p['dim'], p['dim']))
                for j in range(p['mu']):
                    z_outer_w += p['weights'][j] * np.outer(z[j], z[j])

                identity = np.eye(p['dim'])

                # Matrix update (M11)
                p['M'] = p['M'] @ (
                        identity +
                        p['c1'] / 2 * (s_outer - identity) +
                        p['cw'] / 2 * (z_outer_w - identity)
                )

                # M12: Update step size
                norm_s = np.linalg.norm(p['s'])
                p['sigma'] = p['sigma'] * np.exp(p['cs'] / p['damps'] * (norm_s / p['chiN'] - 1))

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


def maes_generation(y: np.ndarray, sigma: float, M: np.ndarray, lam: int = None) -> tuple:
    """
    Generate offspring population using MA-ES sampling strategy.

    Parameters
    ----------
    y : np.ndarray
        Mean vector, shape (d,)
    sigma : float
        Step size (global scaling factor)
    M : np.ndarray
        Transformation matrix, shape (d, d)
    lam : int, optional
        Number of offspring to generate (default: None)

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (lam, d)
    z : np.ndarray
        Standard normal samples, shape (lam, d)
    d : np.ndarray
        Transformed samples, shape (lam, d)
    """
    dim = len(y)

    if lam is None:
        lam = int(4 + 3 * np.log(dim))

    # M4: Sample z
    z = np.random.randn(lam, dim)

    # M5: Compute d = M * z
    d = z @ M.T

    # M6: Generate offspring
    offdecs = y + sigma * d
    offdecs = np.clip(offdecs, 0, 1)

    return offdecs, z, d
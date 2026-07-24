"""
OpenAI-ES (OpenAI Evolution Strategies)

This module implements the OpenAI-ES algorithm for single-objective optimization problems.
OpenAI-ES uses antithetic sampling and momentum-based gradient descent.

References
----------
    [1] Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv:1703.03864 [stat.ML].

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


class OpenAI_ES:
    """
    OpenAI-ES for single-objective optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, sigma=1.0, lr=1e-3, momentum=0.9,
                 save_data=True, save_path='./Data', name='OpenAI-ES', disable_tqdm=True):
        """
        Initialize OpenAI-ES Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (must be even, default: None, will use 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma : float, optional
            Noise standard deviation (default: 1.0)
        lr : float, optional
            Learning rate (default: 1e-3)
        momentum : float, optional
            Momentum coefficient (default: 0.9)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'OpenAI_ES_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma = sigma
        self.lr = lr
        self.momentum = momentum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the OpenAI-ES Algorithm.

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

            # Determine population size (must be even for antithetic sampling)
            N = par_list(self.n, nt)[t]
            if N % 2 != 0:
                N = N + 1

            # Normalize sigma based on problem range
            # MATLAB: range = mean(Prob.Ub{t} - Prob.Lb{t});
            # MATLAB: sigma{t} = Algo.sigma / range;
            lb, ub = problem.bounds[t]
            range_val = np.mean(ub - lb)
            sigma = self.sigma / range_val

            # Initialize mean and momentum
            # MATLAB: x{t} = mean(unifrnd(zeros(Prob.D(t), N), ones(Prob.D(t), N)), 2);
            x = np.mean(np.random.rand(dim, N), axis=1)  # Random initialization, then average
            v = np.zeros(dim)  # Momentum vector

            params.append({
                'dim': dim, 'N': N, 'sigma': sigma, 'x': x, 'v': v
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

                # Antithetic Sampling
                # MATLAB: Z_half = randn(Prob.D(t), N / 2);
                # MATLAB: Z = [Z_half, -Z_half];
                Z_half = np.random.randn(p['dim'], p['N'] // 2)
                Z = np.hstack([Z_half, -Z_half])  # Shape: (dim, N)

                # MATLAB: X = repmat(x{t}, 1, N) + sigma{t} * Z;
                X = p['x'][:, np.newaxis] + p['sigma'] * Z  # Shape: (dim, N)

                # Decode samples (transpose to get (N, dim))
                # MATLAB applies no boundary handling: raw (unclipped) decisions
                # are mapped linearly to real space and evaluated as-is
                sample_decs = X.T

                # Also evaluate the mean
                # MATLAB: mean_sample.Dec = x{t}';
                mean_dec = p['x'][np.newaxis, :]

                # Combine samples and mean for evaluation
                # MATLAB: sample{t} = Algo.Evaluation([sample{t}, mean_sample], Prob, t);
                all_sample_decs = np.vstack([sample_decs, mean_dec])

                # Evaluate fitness
                all_objs_eval, all_cons_eval = evaluation_single(problem, all_sample_decs, i)

                # MATLAB: sample{t} = sample{t}(1:N); (mean is dropped from the
                # population but its evaluation still counts and enters history)
                sample_objs = all_objs_eval[:-1]
                sample_cons = all_cons_eval[:-1]

                # Update current population (samples first, mean last, as in MATLAB)
                decs[i] = all_sample_decs
                objs[i] = all_objs_eval
                cons[i] = all_cons_eval

                nfes_per_task[i] += p['N'] + 1  # N samples + 1 mean
                pbar.update(p['N'] + 1)

                # Append to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

                # Centered rank shaping
                # MATLAB: [~, sortIdx] = sort(fitness);
                # MATLAB: ranks(sortIdx) = N - 1:-1:0; % Minimizing fitness
                # MATLAB: shaped = ranks / (N - 1) - 0.5;
                fitness = sample_objs.flatten()
                sort_idx = np.argsort(fitness, kind='stable')  # MATLAB sort is stable
                ranks = np.zeros(p['N'])
                ranks[sort_idx] = np.arange(p['N'] - 1, -1, -1)
                shaped = ranks / (p['N'] - 1) - 0.5

                # Gradient estimation
                # MATLAB: grad = (Z * shaped') / (N * sigma{t});
                grad = (Z @ shaped) / (p['N'] * p['sigma'])

                # Momentum update
                # MATLAB: v{t} = Algo.momentum * v{t} + (1 - Algo.momentum) * grad;
                p['v'] = self.momentum * p['v'] + (1 - self.momentum) * grad

                # Update mean
                # MATLAB: x{t} = x{t} + Algo.lr * v{t};
                p['x'] = p['x'] + self.lr * p['v']

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


def openai_es_generation(x: np.ndarray, sigma: float, N: int = None) -> tuple:
    """
    Generate offspring population using OpenAI-ES antithetic sampling.

    Parameters
    ----------
    x : np.ndarray
        Mean vector, shape (d,)
    sigma : float
        Noise standard deviation
    N : int, optional
        Number of offspring to generate (must be even, default: None)

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (N, d)
    Z : np.ndarray
        Noise samples, shape (d, N)
    """
    d = len(x)

    # Default population size
    if N is None:
        N = 100

    # Ensure N is even
    if N % 2 != 0:
        N = N + 1

    # Antithetic sampling
    Z_half = np.random.randn(d, N // 2)
    Z = np.hstack([Z_half, -Z_half])

    # Generate offspring
    X = x[:, np.newaxis] + sigma * Z
    offdecs = X.T
    offdecs = np.clip(offdecs, 0, 1)

    return offdecs, Z
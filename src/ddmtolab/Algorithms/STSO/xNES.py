"""
xNES (Exponential Natural Evolution Strategies)

This module implements the xNES algorithm for single-objective optimization problems.
xNES uses natural gradients to adapt the search distribution.

References
----------
    [1] Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J. (2010). Exponential Natural Evolution Strategies. Proceedings of the 12th Annual Conference on Genetic and Evolutionary Computation, 393-400.
    [2] Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J., & Schmidhuber, J. (2014). Natural Evolution Strategies. Journal of Machine Learning Research, 15(27), 949-980.

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
from scipy.linalg import expm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class xNES:
    """
    xNES for single-objective optimization.

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
                 save_data=True, save_path='./Data', name='xNES', disable_tqdm=True):
        """
        Initialize xNES Algorithm.

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
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'xNES')
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
        Execute the xNES Algorithm.

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
                N = par_list(self.n, nt)[t]
            else:
                N = int(4 + 3 * np.log(dim))  # MATLAB: fix(4 + 3 * log(Prob.D(t)))

            # Learning rates
            # MATLAB: etax{t} = 1;
            etax = 1.0
            # MATLAB: etas{t} = (9 + 3 * log(Prob.D(t))) / (5 * Prob.D(t) * sqrt(Prob.D(t)));
            etas = (9 + 3 * np.log(dim)) / (5 * dim * np.sqrt(dim))
            # MATLAB: etaB{t} = etas{t};
            etaB = etas

            # Fitness shaping weights
            # MATLAB: shape{t} = max(0.0, log(N{t} / 2 + 1.0) - log(1:N{t}));
            # MATLAB: shape{t} = shape{t} / sum(shape{t}) - 1 / N{t};
            shape = np.maximum(0.0, np.log(N / 2 + 1.0) - np.log(np.arange(1, N + 1)))
            shape = shape / np.sum(shape) - 1 / N

            # Initialize distribution parameters
            # MATLAB: x{t} = rand(Prob.D(t), 1);
            x = np.random.rand(dim)
            # MATLAB: s{t} = Algo.sigma0;
            s = self.sigma0
            # MATLAB: B{t} = eye(Prob.D(t)); % B = A/s; A*A' = C = covariance matrix
            B = np.eye(dim)

            params.append({
                'dim': dim, 'N': N, 'etax': etax, 'etas': etas, 'etaB': etaB,
                'shape': shape, 'x': x, 's': s, 'B': B
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

                # Step 1: Sampling & importance mixing
                # MATLAB: Z{t} = randn(Prob.D(t), N{t});
                # MATLAB: X{t} = repmat(x{t}, 1, N{t}) + s{t} * B{t} * Z{t};
                Z = np.random.randn(p['dim'], p['N'])
                X = p['x'][:, np.newaxis] + p['s'] * (p['B'] @ Z)

                # Convert to samples (transpose to get shape (N, dim))
                sample_decs = X.T

                # Evaluate samples with boundary constraint handling
                sample_objs, sample_cons, bound_cvs = self._evaluation_and_boundary(
                    problem, sample_decs, i
                )

                # Update current population
                decs[i] = sample_decs
                objs[i] = sample_objs
                cons[i] = sample_cons

                nfes_per_task[i] += p['N']
                pbar.update(p['N'])

                # Append to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

                # Step 2: Fitness reshaping
                # MATLAB: [~, rank] = sortrows([sample.CVs + boundCVs, sample.Objs], [1, 2]);
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                total_cvs = cvs + bound_cvs
                sort_indices = np.lexsort((sample_objs.flatten(), total_cvs))

                # MATLAB: weights{t}(rank{t}) = shape{t};
                weights = np.zeros(p['N'])
                weights[sort_indices] = p['shape']

                # Step 3: Compute the gradient for x, s, and B
                # MATLAB: dx = etax{t} * s{t} * B{t} * (Z{t} * weights{t}');
                dx = p['etax'] * p['s'] * (p['B'] @ (Z @ weights))

                # MATLAB: JM = (repmat(weights{t}, Prob.D(t), 1) .* Z{t}) * Z{t}' - sum(weights{t}) * eye(Prob.D(t));
                JM = (Z * weights[np.newaxis, :]) @ Z.T - np.sum(weights) * np.eye(p['dim'])

                # MATLAB: Js = trace(JM) / Prob.D(t);
                Js = np.trace(JM) / p['dim']

                # MATLAB: ds = 0.5 * etas{t} * Js;
                ds = 0.5 * p['etas'] * Js

                # MATLAB: dB = 0.5 * etaB{t} * (JM - Js * eye(Prob.D(t)));
                dB = 0.5 * p['etaB'] * (JM - Js * np.eye(p['dim']))

                # Step 4: Compute the update
                # MATLAB: x{t} = x{t} + dx;
                p['x'] = p['x'] + dx

                # MATLAB: s{t} = s{t} * exp(ds);
                p['s'] = p['s'] * np.exp(ds)

                # MATLAB: B{t} = B{t} * expm(dB);
                # MATLAB: B{t} = triu(B{t}) + triu(B{t}, 1)'; % enforce symmetry
                p['B'] = p['B'] @ expm(dB)
                p['B'] = np.triu(p['B']) + np.triu(p['B'], 1).T

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

    def _evaluation_and_boundary(self, problem, sample_decs, task_idx):
        """
        Evaluate samples with boundary constraint handling.

        Parameters
        ----------
        problem : MTOP
            Problem instance
        sample_decs : np.ndarray
            Sample decision variables, shape (n_samples, dim)
        task_idx : int
            Task index

        Returns
        -------
        sample_objs : np.ndarray
            Objective values, shape (n_samples, n_objs)
        sample_cons : np.ndarray
            Constraint values, shape (n_samples, n_cons)
        bound_cvs : np.ndarray
            Boundary constraint violations, shape (n_samples,)
        """
        # MATLAB: tempDec = max(0, min(1, sample(i).Dec));
        # MATLAB: boundCVs(i) = sum((sample(i).Dec - tempDec).^2);
        temp_decs = np.clip(sample_decs, 0, 1)
        bound_cvs = np.sum((sample_decs - temp_decs) ** 2, axis=1)

        # MATLAB: sample = Algo.Evaluation(sample, Prob, t);
        # Evaluate the raw (unclipped) decisions, exactly as MATLAB does;
        # out-of-bounds solutions are only penalized via boundCVs in the ranking
        sample_objs, sample_cons = evaluation_single(problem, sample_decs, task_idx)

        # MATLAB: boundCVs(boundCVs > 0) = boundCVs(boundCVs > 0) + max(sample.CVs);
        cvs = np.sum(np.maximum(0, sample_cons), axis=1)
        if np.any(bound_cvs > 0) and len(cvs) > 0:
            max_cv = np.max(cvs) if np.max(cvs) > 0 else 0
            bound_cvs[bound_cvs > 0] = bound_cvs[bound_cvs > 0] + max_cv

        return sample_objs, sample_cons, bound_cvs


def xnes_generation(x: np.ndarray, s: float, B: np.ndarray, N: int = None) -> np.ndarray:
    """
    Generate offspring population using xNES sampling strategy.

    Parameters
    ----------
    x : np.ndarray
        Mean vector, shape (d,)
    s : float
        Step size (global scaling factor)
    B : np.ndarray
        Transformation matrix, shape (d, d)
        Note: B = A/s where A*A' = C (covariance matrix)
    N : int, optional
        Number of offspring to generate (default: None)

    Returns
    -------
    offdecs : np.ndarray
        Offspring decision variables, shape (N, d)
    """
    d = len(x)

    # If N is None, generate a default number
    if N is None:
        N = int(4 + 3 * np.log(d))

    # MATLAB: Z{t} = randn(Prob.D(t), N{t});
    # MATLAB: X{t} = repmat(x{t}, 1, N{t}) + s{t} * B{t} * Z{t};
    Z = np.random.randn(d, N)
    X = x[:, np.newaxis] + s * (B @ Z)

    # Transpose to get shape (N, d) and clip to [0, 1]
    offdecs = X.T
    offdecs = np.clip(offdecs, 0, 1)

    return offdecs
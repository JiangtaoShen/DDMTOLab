"""
Multi-task Max-value (entropy search) Bayesian Optimization (MUMBO)

This code implements multi-task optimization based on the MUMBO. Unlike the original, this implementation
ignores task cost differences and the fidelity discount mechanism (ρ parameter), using a round-robin strategy to
sequentially add samples to each task. All tasks are treated as equally important in a standard multi-task optimization
setting, rather than a multi-fidelity or target-task-oriented scenario.

References
----------
    [1] Moss, Henry B., David S. Leslie, and Paul Rayson. "Mumbo: Multi-task max-value Bayesian \
        optimization." Joint European Conference on Machine Learning and Knowledge Discovery in \
        Databases. Cham: Springer International Publishing, 2020.

    [2] Wang, Zi, and Stefanie Jegelka. "Max-value entropy search for efficient Bayesian \
        optimization." International Conference on Machine Learning. PMLR, 2017.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.18
Version: 1.0
"""
from tqdm import tqdm
import torch
import time
import numpy as np
from botorch.optim import optimize_acqf
from torch.distributions import Normal
from Methods.Algo_Methods.algo_utils import *
from Methods.Algo_Methods.bo_utils import mtgp_build
import warnings

warnings.filterwarnings("ignore")


class MUMBO:
    """
    Multi-task Max-value (entropy search) Bayesian Optimization (MUMBO)

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

    def __init__(self, problem, n_initial=None, max_nfes=None, save_data=True,
                 save_path='./TestData', name='MUMBO_test', disable_tqdm=True):
        """
        Initialize MUMBO

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MUMBO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute MUMBO

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

        # Initialize samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Round-robin optimization loop
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Build multi-task Gaussian process with normalized objectives
            objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
            mtgp = mtgp_build(decs, objs_normalized, dims, data_type=data_type)

            # Round-robin: add one point to each active task
            for i in active_tasks:
                # Select next sample point via MES acquisition function
                candidate_np = mes_next_point(mtgp=mtgp, objs_normalized=objs_normalized, task_id=i, dims=dims, nt=nt,
                                              data_type=data_type)

                # Evaluate on target task
                obj, _ = evaluation_single(problem, candidate_np, i)

                # Update data
                decs[i], objs[i] = vstack_groups((decs[i], candidate_np), (objs[i], obj))
                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def get_mes_samples(objs_normalized, n_samples=10, data_type=torch.double):
    """
    Sample potential global optimum values using Gumbel distribution.

    Parameters
    ----------
    objs_normalized : list of ndarray
        Normalized objective values for all tasks
    n_samples : int, optional
        Number of samples to generate (default: 10)
    data_type : torch.dtype, optional
        Data type for tensors (default: torch.double)

    Returns
    -------
    torch.Tensor
        Sampled potential global optimum values, shape (n_samples,)
    """
    with torch.no_grad():
        # Convert all observations to negative values (for maximization)
        all_neg_objs = torch.cat([-torch.as_tensor(o, dtype=data_type) for o in objs_normalized])
        y_max = all_neg_objs.max()

        # Sample from Gumbel distribution
        sampler = torch.distributions.Gumbel(loc=0, scale=0.01)
        samples = y_max + sampler.sample((n_samples,)).abs().to(data_type)
        return samples


def mes_acquisition(X, mtgp, g_samples, task_id, nt):
    """
    Compute Max-value Entropy Search acquisition function.

    Parameters
    ----------
    X : torch.Tensor
        Candidate points, shape (..., d)
    mtgp : MultiTaskGP
        Multi-task Gaussian process model
    g_samples : torch.Tensor
        Sampled potential global optimum values
    task_id : int
        Target task index
    nt : int
        Number of tasks

    Returns
    -------
    torch.Tensor
        MES acquisition values, shape (...)
    """
    # Construct input with task indicator
    task_feature = torch.full((*X.shape[:-1], 1), task_id / (nt - 1),
                              dtype=X.dtype, device=X.device)
    X_with_task = torch.cat([X, task_feature], dim=-1)

    # Get posterior predictions
    posterior = mtgp.posterior(X_with_task)
    mu = posterior.mean.squeeze(-1)
    sigma = torch.sqrt(posterior.variance.squeeze(-1)).clamp(min=1e-10)

    # Compute information gain
    total_info_gain = torch.zeros_like(mu)
    normal = Normal(torch.tensor(0.0, dtype=mu.dtype, device=mu.device),
                    torch.tensor(1.0, dtype=mu.dtype, device=mu.device))

    for g_star in g_samples:
        gamma = (g_star - mu) / sigma
        pdf_g = torch.exp(normal.log_prob(gamma))
        cdf_g = normal.cdf(gamma).clamp(min=1e-10)

        # Entropy reduction term
        info_gain = (gamma * pdf_g) / (2 * cdf_g) - torch.log(cdf_g)
        total_info_gain += info_gain

    avg_info_gain = total_info_gain / len(g_samples)

    return avg_info_gain.view(-1)


def mes_next_point(mtgp, objs_normalized, task_id, dims, nt, data_type=torch.double):
    """
    Select next evaluation point using MES acquisition function.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Multi-task Gaussian process model
    objs_normalized : list of ndarray
        Normalized objective values for all tasks
    task_id : int
        Target task index
    dims : list
        Dimensionality of each task
    nt : int
        Number of tasks
    data_type : torch.dtype, optional
        Data type for tensors (default: torch.double)

    Returns
    -------
    ndarray
        Selected decision variables, shape (1, dim_task)
    """
    # Generate samples for max-value
    g_samples = get_mes_samples(objs_normalized, n_samples=10, data_type=data_type)

    # Define search bounds for the target task
    dim = dims[task_id]
    bounds = torch.stack([
        torch.zeros(dim, dtype=data_type),
        torch.ones(dim, dtype=data_type)
    ])

    # Define acquisition function wrapper
    def acq_wrapper(X):
        return mes_acquisition(X, mtgp, g_samples, task_id, nt)

    # Optimize acquisition function
    candidate, _ = optimize_acqf(
        acq_function=acq_wrapper,  # type: ignore
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=100,
    )

    # Convert to numpy
    candidate_np = candidate.detach().cpu().numpy().squeeze().reshape(1, -1)

    return candidate_np
"""
Multi-fidelity Upper-confidence-bound Multi-armed Bandit Optimization (MUMBO)

This module implements MUMBO for expensive multi-task optimization with knowledge transfer via multi-task Gaussian processes.

References
----------
    [1] Swersky, Kevin, Jasper Snoek, and Ryan P. Adams. "Multi-task bayesian optimization."
        Advances in neural information processing systems 26 (2013).

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


def get_mumbo_samples(objs_normalized, n_samples=10, data_type=torch.double):
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


def mumbo_utility_func(X, mtgp, costs, g_samples, active_tasks, nt):
    """
    Compute MUMBO utility (information gain per unit cost).

    Parameters
    ----------
    X : torch.Tensor
        Candidate points with task indices, shape (..., d+1)
    mtgp : MultiTaskGP
        Multi-task Gaussian process model
    costs : list
        Cost vector for each task
    g_samples : torch.Tensor
        Sampled potential global optimum values
    active_tasks : list
        List of active task indices
    nt : int
        Number of tasks

    Returns
    -------
    torch.Tensor
        MUMBO utility values, shape (...)
    """
    # Get posterior predictions
    posterior = mtgp.posterior(X)
    mu = posterior.mean.squeeze(-1)  # Shape: (...)
    sigma = torch.sqrt(posterior.variance.squeeze(-1)).clamp(min=1e-6)  # Shape: (...)

    # Extract and map task indices
    task_indices_raw = X[..., -1]
    task_id = torch.round(task_indices_raw * (nt - 1)).long()

    # Compute information gain
    total_info_gain = torch.zeros_like(mu)
    for g_star in g_samples:
        gamma = (g_star - mu) / sigma
        normal = Normal(torch.tensor(0.0, dtype=mu.dtype, device=mu.device),
                        torch.tensor(1.0, dtype=mu.dtype, device=mu.device))
        pdf_g = torch.exp(normal.log_prob(gamma))
        cdf_g = normal.cdf(gamma).clamp(min=1e-8)

        # Core entropy reduction term (Formula 5)
        info_gain = (gamma * pdf_g) / (2 * cdf_g) - torch.log(cdf_g)
        total_info_gain += info_gain

    avg_info_gain = total_info_gain / len(g_samples)

    # Apply cost normalization
    costs_tensor = torch.as_tensor(costs, dtype=mu.dtype, device=mu.device)
    current_costs = costs_tensor[task_id]

    # Create mask for active tasks
    mask = torch.zeros_like(avg_info_gain)
    for t in active_tasks:
        mask[task_id == t] = 1.0

    # Return utility: information gain per unit cost, masked by active tasks
    utility = (avg_info_gain * mask) / current_costs

    # Ensure output is properly shaped for optimize_acqf
    return utility.view(-1)


def mumbo_next_point(mtgp, costs, objs_normalized, dims, nt, active_tasks, data_type=torch.double):
    """
    Select next evaluation point and task using MUMBO acquisition function.

    Parameters
    ----------
    mtgp : MultiTaskGP
        Multi-task Gaussian process model
    costs : list
        Cost vector for each task
    objs_normalized : list of ndarray
        Normalized objective values for all tasks
    dims : list
        Dimensionality of each task
    nt : int
        Number of tasks
    active_tasks : list
        List of active task indices
    data_type : torch.dtype, optional
        Data type for tensors (default: torch.double)

    Returns
    -------
    candidate_x : ndarray
        Selected decision variables, shape (1, dim_chosen_task)
    chosen_task : int
        Selected task index
    """
    # 1. Generate g_samples
    g_samples = get_mumbo_samples(objs_normalized, n_samples=10, data_type=data_type)

    # 2. Define search bounds: [0, 1]^(max_dim + 1)
    max_dim = max(dims)
    bounds = torch.stack([
        torch.zeros(max_dim + 1, dtype=data_type),
        torch.ones(max_dim + 1, dtype=data_type)
    ])

    # 3. Define acquisition function wrapper
    def acq_wrapper(X):
        return mumbo_utility_func(X, mtgp, costs, g_samples, active_tasks, nt)

    # 4. Optimize acquisition function
    try:
        candidate_full, _ = optimize_acqf(
            acq_function=acq_wrapper,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )
    except Exception as e:
        print(f"Warning: optimize_acqf failed with error: {e}")
        # Fallback: random selection from active tasks
        chosen_task = np.random.choice(active_tasks)
        candidate_x = np.random.rand(1, dims[chosen_task])
        return candidate_x, chosen_task

    # 5. Parse results
    res = candidate_full.detach().cpu().numpy().squeeze()
    z_val = res[-1]
    chosen_task = int(np.round(z_val * (nt - 1)))

    # Handle out-of-bounds task selection
    if chosen_task not in active_tasks:
        # Select active task with minimum cost
        chosen_task = active_tasks[np.argmin([costs[t] for t in active_tasks])]

    # Extract decision variables for chosen task
    candidate_x = res[:dims[chosen_task]].reshape(1, -1)

    return candidate_x, chosen_task


class MUMBO:
    """
    Multi-fidelity Upper-confidence-bound Multi-armed Bandit Optimization for
    expensive multi-task optimization problems.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'n_objs': '1',
        'n_cons': '0',
        'n_initial': 'unequal',
        'max_nfes': 'unequal',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'cost_aware': 'True'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        """
        Get algorithm information.

        Parameters
        ----------
        print_info : bool, optional
            Whether to print information (default: True)

        Returns
        -------
        dict
            Algorithm information dictionary
        """
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, costs=None, n_initial=None, max_nfes=None, save_data=True,
                 save_path='./TestData', name='MUMBO_test', disable_tqdm=True):
        """
        Initialize Multi-fidelity Upper-confidence-bound Multi-armed Bandit Optimization algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        costs : list or np.ndarray, optional
            Cost vector for each task/fidelity level (default: all ones)
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

        # Initialize costs: default to all ones if not provided
        if costs is None:
            self.costs = [1.0] * problem.n_tasks
        else:
            self.costs = list(costs)
            if len(self.costs) != problem.n_tasks:
                raise ValueError(f"Length of costs ({len(self.costs)}) must match "
                                 f"number of tasks ({problem.n_tasks})")

    def optimize(self):
        """
        Execute the Multi-fidelity Upper-confidence-bound Multi-armed Bandit Optimization algorithm.

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
        nfes = sum(n_initial_per_task)

        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < sum(max_nfes_per_task):
            # Identify active tasks
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # Normalize objectives and build multi-task GP
            objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
            mtgp = mtgp_build(decs, objs_normalized, dims, data_type=data_type)

            # Select next point and task using MUMBO
            candidate_np, chosen_task = mumbo_next_point(
                mtgp=mtgp,
                costs=self.costs,
                objs_normalized=objs_normalized,
                dims=dims,
                nt=nt,
                active_tasks=active_tasks,
                data_type=data_type
            )

            # Evaluate on chosen task
            obj, _ = evaluation_single(problem, candidate_np, chosen_task)

            # Update data
            decs[chosen_task], objs[chosen_task] = vstack_groups(
                (decs[chosen_task], candidate_np),
                (objs[chosen_task], obj)
            )
            append_history(all_decs[chosen_task], decs[chosen_task],
                           all_objs[chosen_task], objs[chosen_task])

            nfes_per_task[chosen_task] += 1
            nfes += 1
            pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(
            all_decs=all_decs,
            all_objs=all_objs,
            runtime=runtime,
            max_nfes=nfes_per_task,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        return results


# Test code
if __name__ == "__main__":
    from Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D

    problem = CEC17MTSO_10D().P1()
    costs = [1.0, 2.5]  # Task 2 is 2.5x more expensive than Task 1

    results = MUMBO(
        problem,
        costs=costs,
        n_initial=10,
        max_nfes=50,
        disable_tqdm=False
    ).optimize()

    print("Optimization completed!")
    print(f"Runtime: {results.runtime:.2f} seconds")
"""
A Surrogate-Assisted Evolutionary Framework for Expensive Multitask Optimization Problems (SELF)

This module implements SELF using multi-task Gaussian processes and Bayesian optimization for expensive multi-task optimization.

References
----------
    [1] Tan, Shenglian, et al. "A surrogate-assisted evolutionary framework for expensive multitask
        optimization problems." IEEE Transactions on Evolutionary Computation (2024).

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.18
Version: 1.0
"""
import numpy as np
from tqdm import tqdm
import torch
import time

from Methods.Algo_Methods.bo_utils import mtgp_predict, mtgp_build, mtgp_task_corr
from Methods.mtop import MTOP
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.models.transforms import Standardize
from Methods.Algo_Methods.algo_utils import *
from Algorithms.STSO.DE import DE
import warnings
warnings.filterwarnings("ignore")


class SELF:
    """
    Surrogate-Assisted Evolutionary Framework for expensive multi-task optimization.

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
        'n_initial': 'equal',
        'max_nfes': 'unequal, controlled by SELF',
        'expensive': 'True',
        'knowledge_transfer': 'True'
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

    def __init__(self, problem, max_nfes=None, np=10, F=0.5, CR=0.9, ng=50, nl=50, save_data=True, save_path='./TestData',
                 name='SELF_test', disable_tqdm=True):
        """
        Initialize SELF algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        np : int, optional
            Population size (default: 10)
        F : float, optional
            Mutation factor for DE (default: 0.5)
        CR : float, optional
            Crossover rate for DE (default: 0.9)
        ng : int, optional
            Number of trial vectors in global knowledge transfer phase (default: 50)
        nl : int, optional
            Sample size for training GP model in local knowledge transfer phase (default: 50)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'SELF_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.np = np if max_nfes is not None else 10
        self.F = F
        self.CR = CR
        self.ng = ng
        self.nl = nl
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SELF algorithm with three phases: global transfer, local optimization, and local transfer.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        data_type = torch.double
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.np, nt)
        nfes_per_task = [0] * nt
        max_nfes = self.max_nfes * nt

        # Initialize samples using Latin Hypercube Sampling
        decs = initialization(problem, self.np, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes = self.np * nt
        for i in range(nt):
            nfes_per_task[i] += self.np

        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        # Working population for evolutionary updates
        pop_decs = copy.deepcopy(decs)
        pop_objs = copy.deepcopy(objs)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:

            # === Global Knowledge Transfer Phase ===
            # Build multi-task Gaussian process and extract task correlations
            objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
            mtgp = mtgp_build(decs, objs_normalized, dims, data_type=data_type)
            task_corr = mtgp_task_corr(mtgp)

            for i in range(nt):
                for j in range(self.np):
                    # Generate candidates using DE guided by current best
                    off_decs = de_generation_with_core(pop_decs[i], pop_objs[i], pop_decs[i][j], self.ng, self.F,
                                                       self.CR)

                    # Predict objectives using MTGP surrogate
                    pred_objs, pred_std = mtgp_predict(mtgp=mtgp, off_decs=off_decs, task_id=i, dims=dims, nt=nt,
                                                       data_type=data_type)

                    # Evaluate candidate with minimum predicted objective
                    best_idx = np.argmin(pred_objs)
                    best_off_dec = off_decs[[best_idx], :]
                    true_obj, _ = evaluation_single(problem, best_off_dec, i)

                    # Greedy update: replace if offspring is better
                    if true_obj < pop_objs[i][j]:
                        pop_decs[i][j] = best_off_dec[0]
                        pop_objs[i][j] = true_obj[0]

                    decs[i], objs[i] = vstack_groups((decs[i], best_off_dec), (objs[i], true_obj))

                    nfes += 1
                    nfes_per_task[i] += 1
                    pbar.update(1)

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

            # === Local Optimization Phase ===
            for i in range(nt):
                # Select top nl individuals for local GP model
                if len(decs[i]) <= self.nl:
                    nearest_decs = decs[i]
                    nearest_objs = objs[i]
                else:
                    best_indices = np.argsort(objs[i].flatten())[:self.nl]
                    nearest_decs = decs[i][best_indices]
                    nearest_objs = objs[i][best_indices]

                # Generate next point via Bayesian optimization with LogEI
                candidate = bo_next_point_de(nearest_decs, nearest_objs, dims[i], data_type)
                true_obj, _ = evaluation_single(problem, candidate, i)

                # Update working population if candidate improves
                if np.any(true_obj < pop_objs[i]):
                    is_duplicate = np.any(np.all(np.isclose(pop_decs[i], candidate[0]), axis=1))
                    if not is_duplicate:
                        worse_indices = np.where(pop_objs[i] > true_obj)[0]
                        if len(worse_indices) > 0:
                            worst_idx = worse_indices[np.argmax(pop_objs[i][worse_indices])]
                            pop_decs[i][worst_idx] = candidate[0]
                            pop_objs[i][worst_idx] = true_obj[0]

                decs[i], objs[i] = vstack_groups((decs[i], candidate), (objs[i], true_obj))

                nfes += 1
                nfes_per_task[i] += 1
                pbar.update(1)

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

            # === Local Knowledge Transfer Phase ===
            for i in range(nt):
                transfer_samples = []

                # Probabilistic transfer based on task correlation
                for j in range(nt):
                    if i == j:
                        continue
                    if np.random.rand() < abs(task_corr[i][j]):
                        best_idx = np.argmin(pop_objs[j])
                        transfer_samples.append(pop_decs[j][best_idx])

                if len(transfer_samples) > 0:
                    transfer_samples = np.array(transfer_samples)

                    # Adjust dimensions to match target task
                    if transfer_samples.shape[1] > dims[i]:
                        transfer_samples = transfer_samples[:, :dims[i]]
                    elif transfer_samples.shape[1] < dims[i]:
                        padding = np.zeros((transfer_samples.shape[0], dims[i] - transfer_samples.shape[1]))
                        transfer_samples = np.hstack((transfer_samples, padding))

                    true_obj, _ = evaluation_single(problem, transfer_samples, i)

                    # Select best transfer sample and update population
                    best_idx = np.argmin(true_obj)
                    best_sample = transfer_samples[[best_idx], :]
                    best_sample_obj = true_obj[best_idx]

                    if np.any(best_sample_obj < pop_objs[i]):
                        is_duplicate = np.any(np.all(np.isclose(pop_decs[i], best_sample[0]), axis=1))
                        if not is_duplicate:
                            worse_indices = np.where(pop_objs[i] > best_sample_obj)[0]
                            if len(worse_indices) > 0:
                                worst_idx = worse_indices[np.argmax(pop_objs[i][worse_indices])]
                                pop_decs[i][worst_idx] = best_sample[0]
                                pop_objs[i][worst_idx] = true_obj[0]

                    decs[i], objs[i] = vstack_groups((decs[i], transfer_samples), (objs[i], true_obj))

                    nfes += len(transfer_samples)
                    nfes_per_task[i] += len(transfer_samples)
                    pbar.update(len(transfer_samples))

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs, nfes_per_task = trim_excess_evaluations(all_decs, all_objs, nt, [self.max_nfes] * nt,
                                                                    nfes_per_task)

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def de_generation_with_core(parents, parents_objs, core_parent, n_off, F, CR):
    """
    Generate offspring using hybrid DE mutation strategy.

    Parameters
    ----------
    parents : np.ndarray
        Parent solutions of shape (n, d)
    parents_objs : np.ndarray
        Objective values of parents of shape (n,) or (n, 1)
    core_parent : np.ndarray
        Current individual being evolved of shape (1, d) or (d,)
    n_off : int
        Number of offspring to generate
    F : float
        Differential weight (mutation scale factor)
    CR : float
        Crossover rate in [0, 1] for binomial crossover

    Returns
    -------
    offdecs : np.ndarray
        Offspring array of shape (n_off, d), clipped to [0, 1]

    Notes
    -----
    Uses a hybrid strategy with 50% probability each:
    - DE/best/1 with binomial crossover: v = best + F*(r2 - r3)
    - DE/current/1 without crossover: v = current + F*(r2 - r3)
    """
    n, d = parents.shape

    # Ensure correct dimensions
    if core_parent.ndim == 2:
        core_parent = core_parent.squeeze()
    if parents_objs.ndim == 2:
        parents_objs = parents_objs.flatten()

    # Find best parent
    best_id = np.argmin(parents_objs)
    best_parent = parents[best_id]

    offdecs = np.zeros((n_off, d), dtype=float)

    # Generate offspring
    for j in range(n_off):
        id_set = np.random.permutation(n)
        r1, r2, r3 = id_set[0], id_set[1], id_set[2]

        if np.random.rand() < 0.5:
            # DE/best/1 with binomial crossover
            v = best_parent + F * (parents[r2] - parents[r3])
            trial = np.where(np.random.rand(d) < CR, v, parents[r1])
        else:
            # DE/current/1 without crossover
            trial = core_parent + F * (parents[r2] - parents[r3])

        offdecs[j] = np.clip(trial, 0.0, 1.0)

    return offdecs


def bo_next_point_de(decs, objs, dim, data_type=torch.float):
    """
    Generate next sampling point using Bayesian Optimization with Log Expected Improvement.

    Parameters
    ----------
    decs : np.ndarray
        Decision variables (training data) of shape (n_samples, dim)
    objs : np.ndarray
        Objective values (training data) of shape (n_samples, 1)
    dim : int
        Dimension of the problem
    data_type : torch.dtype, optional
        Data type for torch tensors (default: torch.float)

    Returns
    -------
    candidate_np : np.ndarray
        Next sampling point of shape (1, dim)

    Notes
    -----
    Builds a Gaussian Process with the provided data, constructs a Log Expected Improvement
    acquisition function, and optimizes it using Differential Evolution to find the next
    most promising point to evaluate.
    """
    # Prepare training data for Gaussian Process
    train_X = torch.tensor(decs, dtype=data_type)
    train_Y = torch.tensor(-objs, dtype=data_type)

    # Build and fit Gaussian Process model
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Build Log Expected Improvement acquisition function
    best_f = train_Y.max()
    logEI = LogExpectedImprovement(model=gp, best_f=best_f)

    # Wrap LogEI as numpy function for DE optimizer
    def logEI_func(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_torch = torch.tensor(x, dtype=data_type)
        with torch.no_grad():
            logei_value = logEI(x_torch)
        logei_np = logei_value.detach().cpu().numpy()
        return -logei_np.flatten() if x.shape[0] == 1 else -logei_np

    # Optimize LogEI using Differential Evolution
    problem = MTOP()
    problem.add_task(logEI_func, dim=dim)
    de = DE(problem, n=50, max_nfes=5000, F=0.5, CR=0.9, save_data=False, disable_tqdm=True)
    result = de.optimize()

    return result.best_decs
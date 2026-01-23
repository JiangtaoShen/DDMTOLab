"""
Evolutionary Expected Improvement based Bayesian Optimization (EEI-BO)

This module implements Bayesian Optimization for expensive single-objective optimization problems
using an evolutionary approach to optimize the Expected Improvement acquisition function.

References
----------
    [1] Liu, Jiao, et al. "Solving highly expensive optimization problems via evolutionary expected improvement." \
        IEEE Transactions on Systems, Man, and Cybernetics: Systems 53.8 (2023): 4843-4855.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.17
Version: 1.0
"""
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
from ddmtolab.Algorithms.STSO.CMAES import CMAES
import torch
from ddmtolab.Methods.mtop import MTOP
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from ddmtolab.Algorithms.STSO.DE import DE
from botorch.acquisition import LogExpectedImprovement
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings
import time

warnings.filterwarnings("ignore")


class EEIBO:
    """
    Evolutionary Expected Improvement based Bayesian Optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n1=50, max_nfes1=500, n2=30, max_nfes2=600,
                 save_data=True, save_path='./TestData', name='EEIBO_test', disable_tqdm=True):
        """
        Initialize EEI-BO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        n1: int, optional
            Population size of CMA-ES (default: 50)
        max_nfes1: int, optional
            Maximum number of function evaluations of CMA-ES (default: 500)
        n2: int, optional
            Population size of DE (default: 30)
        max_nfes2: int, optional
            Maximum number of function evaluations of DE (default: 6000)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EEIBO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.n1 = n1
        self.max_nfes1 = max_nfes1
        self.n2 = n2
        self.max_nfes2 = max_nfes2
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Evolutionary Expected Improvement based Bayesian Optimization algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Reorganize initial data into task-specific history lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Identify tasks that have not exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Build RBF surrogate model from current samples
                rbf_i = RBFInterpolator(decs[i], objs[i].flatten())

                def rbf(x):
                    return rbf_i(x)

                surrogate_problem = MTOP()
                surrogate_problem.add_task(rbf, dim=dims[i])

                # Use CMA-ES to extract distribution parameters from surrogate
                _, params = CMAES(surrogate_problem, n=self.n1, max_nfes=self.max_nfes1, save_data=False).optimize()

                params_i = params[0]
                mu = params_i['m_dec']  # Mean of the distribution
                sigma = params_i['sigma']  # Step size
                C = params_i['C']  # Covariance matrix

                # Select next candidate using Evolutionary Expected Improvement
                candidate = eei_bo_next_point(decs[i], objs[i], dims[i], mu, sigma, C, self.n2, self.max_nfes2,
                                              data_type=data_type)

                # Evaluate the candidate solution on true objective
                new_objs, _ = evaluation_single(problem, candidate, i)

                # Update dataset with new sample
                decs[i], objs[i] = vstack_groups((decs[i], candidate), (objs[i], new_objs))

                nfes_per_task[i] += 1
                pbar.update(1)

                # Store cumulative history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Build and save optimization results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


# def eei_bo_next_point(decs, objs, dim, mu, sigma, C, n2, max_nfes2, data_type=torch.float):
#     """
#     Select next candidate point using Evolutionary Expected Improvement criterion.
#
#     This function combines the traditional Expected Improvement acquisition function
#     with a probability distribution derived from CMA-ES, creating an Evolutionary
#     Expected Improvement (EEI) acquisition function that balances exploration and
#     exploitation while incorporating evolutionary search information.
#
#     Parameters
#     ----------
#     decs : ndarray
#         Current decision variables (samples)
#     objs : ndarray
#         Current objective values
#     dim : int
#         Dimensionality of the problem
#     mu : ndarray
#         Mean vector from CMA-ES distribution
#     sigma : float
#         Step size from CMA-ES
#     C : ndarray
#         Covariance matrix from CMA-ES
#     n2 : int
#         Population size of DE
#     max_nfes2 : int
#         Maximum number of function evaluations of DE
#     data_type : torch.dtype, optional
#         Data type for PyTorch tensors (default: torch.float)
#
#     Returns
#     -------
#     ndarray
#         Next candidate point to evaluate
#     """
#     # Fit Gaussian Process surrogate model
#     train_X = torch.tensor(decs, dtype=data_type)
#     train_Y = torch.tensor(-objs, dtype=data_type)  # Negative for maximization
#     gp = SingleTaskGP(train_X=train_X, train_Y=train_Y, outcome_transform=Standardize(m=1))
#     mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#     fit_gpytorch_mll(mll)
#
#     # Prepare Expected Improvement acquisition function
#     best_f = train_Y.max()
#     logEI = LogExpectedImprovement(model=gp, best_f=best_f)
#
#     # Compute real covariance matrix and its properties
#     Sigma_Real = sigma ** 2 * C
#     det_C = np.linalg.det(Sigma_Real)
#     C_inv = np.linalg.inv(Sigma_Real)
#     d = dim
#
#     def EEI_func(x):
#         """
#         Evolutionary Expected Improvement acquisition function.
#
#         EEI(x) = EI(x) * P(x), where:
#         - EI(x) is the Expected Improvement
#         - P(x) is the probability density from CMA-ES distribution
#
#         Parameters
#         ----------
#         x : ndarray
#             Candidate point(s) to evaluate
#
#         Returns
#         -------
#         float or ndarray
#             Negative EEI value(s) for minimization
#         """
#         x = x.reshape(1, -1) if x.ndim == 1 else x
#
#         # Compute Expected Improvement
#         x_torch = torch.tensor(x, dtype=data_type)
#         with torch.no_grad():
#             log_ei = logEI(x_torch).detach().cpu().numpy().flatten()
#
#         ei = np.exp(log_ei)
#
#         # Compute probability density from multivariate normal distribution
#         normalization = 1.0 / (det_C * (2 * np.pi) ** (d / 2))
#         diff = x - mu
#         mahalanobis = np.sum(diff @ C_inv * diff, axis=1)
#         P = normalization * np.exp(-0.5 * mahalanobis)
#
#         # Combine EI with probability density
#         eei = ei * P
#
#         return -float(eei) if x.shape[0] == 1 else -eei
#
#     # Optimize EEI acquisition function using Differential Evolution
#     problem = MTOP()
#     problem.add_task(EEI_func, dim=dim)
#     result = DE(problem, n=n2, max_nfes=max_nfes2, F=0.5, CR=0.9, save_data=False, disable_tqdm=True).optimize()
#
#     return result.best_decs

def eei_acquisition_function(mu, sigma, C, dim, logEI, data_type=torch.float):
    """
    Create Evolutionary Expected Improvement acquisition function.

    Parameters
    ----------
    mu : ndarray
        Mean vector from CMA-ES distribution
    sigma : float
        Step size from CMA-ES
    C : ndarray
        Covariance matrix from CMA-ES
    dim : int
        Dimensionality of the problem
    logEI : LogExpectedImprovement
        Log Expected Improvement acquisition function
    data_type : torch.dtype, optional
        Data type for PyTorch tensors (default: torch.float)

    Returns
    -------
    callable
        EEI acquisition function
    """
    # Compute real covariance matrix and its properties
    Sigma_Real = sigma ** 2 * C
    det_C = np.linalg.det(Sigma_Real)
    C_inv = np.linalg.inv(Sigma_Real)
    d = dim

    def EEI_func(x):
        """
        Evolutionary Expected Improvement acquisition function.

        EEI(x) = EI(x) * P(x), where:
        - EI(x) is the Expected Improvement
        - P(x) is the probability density from CMA-ES distribution

        Parameters
        ----------
        x : ndarray
            Candidate point(s) to evaluate

        Returns
        -------
        float or ndarray
            Negative EEI value(s) for minimization
        """
        x = x.reshape(1, -1) if x.ndim == 1 else x

        # Compute Expected Improvement
        x_torch = torch.tensor(x, dtype=data_type)
        with torch.no_grad():
            log_ei = logEI(x_torch).detach().cpu().numpy().flatten()

        ei = np.exp(log_ei)

        # Compute probability density from multivariate normal distribution
        normalization = 1.0 / (det_C * (2 * np.pi) ** (d / 2))
        diff = x - mu
        mahalanobis = np.sum(diff @ C_inv * diff, axis=1)
        P = normalization * np.exp(-0.5 * mahalanobis)

        # Combine EI with probability density
        eei = ei * P

        return -float(eei) if x.shape[0] == 1 else -eei

    return EEI_func


def eei_bo_next_point(decs, objs, dim, mu, sigma, C, n2, max_nfes2, data_type=torch.float):
    """
    Select next candidate point using Evolutionary Expected Improvement criterion.

    This function combines the traditional Expected Improvement acquisition function
    with a probability distribution derived from CMA-ES, creating an Evolutionary
    Expected Improvement (EEI) acquisition function that balances exploration and
    exploitation while incorporating evolutionary search information.

    Parameters
    ----------
    decs : ndarray
        Current decision variables (samples)
    objs : ndarray
        Current objective values
    dim : int
        Dimensionality of the problem
    mu : ndarray
        Mean vector from CMA-ES distribution
    sigma : float
        Step size from CMA-ES
    C : ndarray
        Covariance matrix from CMA-ES
    n2 : int
        Population size of DE
    max_nfes2 : int
        Maximum number of function evaluations of DE
    data_type : torch.dtype, optional
        Data type for PyTorch tensors (default: torch.float)

    Returns
    -------
    ndarray
        Next candidate point to evaluate
    """
    # Fit Gaussian Process surrogate model
    train_X = torch.tensor(decs, dtype=data_type)
    train_Y = torch.tensor(-objs, dtype=data_type)  # Negative for maximization
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Prepare Expected Improvement acquisition function
    best_f = train_Y.max()
    logEI = LogExpectedImprovement(model=gp, best_f=best_f)

    # Create EEI acquisition function
    EEI_func = eei_acquisition_function(mu, sigma, C, dim, logEI, data_type)

    # Optimize EEI acquisition function using Differential Evolution
    problem = MTOP()
    problem.add_task(EEI_func, dim=dim)
    result = DE(problem, n=n2, max_nfes=max_nfes2, F=0.5, CR=0.9, save_data=False, disable_tqdm=True).optimize()

    return result.best_decs
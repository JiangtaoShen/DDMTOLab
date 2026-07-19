"""
Evolutionary Expected Improvement based Bayesian Optimization (EEI-BO)

This module implements EEI-BO for expensive single-objective optimization. A GP
surrogate provides the Expected Improvement (EI), and a persistent CMA-ES search
distribution co-evolves alongside the BO loop: at every BO iteration the CMA-ES
state advances a single generation (its samples ranked by the GP surrogate), and
the resulting Gaussian density P(x) = N(x; mu, sigma^2 C) weights EI to form the
Evolutionary Expected Improvement acquisition

    EEI(x) = EI(x) * P(x),

which is maximized by differential evolution to select the next query point.

References
----------
    [1] J. Liu, Y. Wang, G. Sun, and T. Pang, "Solving Highly Expensive
        Optimization Problems via Evolutionary Expected Improvement," IEEE
        Transactions on Systems, Man, and Cybernetics: Systems, vol. 53, no. 8,
        pp. 4843-4855, 2023.

Notes
-----
Corrected against the reference implementation (ST-EEI-BO, v2.0):

- The CMA-ES distribution is now persistent and advances exactly one generation
  per BO iteration (the paper's co-evolution). The previous version re-ran a full
  CMA-ES optimization from a random restart every iteration.
- CMA-ES starts from the mean of the initial samples with sigma0 = 0.5 (half the
  [0, 1] search range), matching the reference init.
- EEI is maximized in log space (log EI + log P) to avoid the underflow of EI*P
  when EI is tiny; the argmax is identical to the reference EI*P.
- CMA-ES samples are ranked by the GP posterior mean (equivalent, for ranking, to
  the reference's separate RBF surrogate and matching the MT reference).

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
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.acquisition import LogExpectedImprovement

from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Algorithms.STSO.DE import DE
from ddmtolab.Methods.Algo_Methods.algo_utils import *

warnings.filterwarnings("ignore")


class EEI_BO:
    """
    Evolutionary Expected Improvement based Bayesian Optimization.

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

    def __init__(self, problem, n_initial=None, max_nfes=None, cma_popsize=100, sigma0=0.5,
                 n2=30, max_nfes2=6000, save_data=True, save_path='./Data', name='EEI-BO',
                 disable_tqdm=True):
        """
        Initialize EEI-BO algorithm.

        Parameters
        ----------
        problem : MTOP
            Optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
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
            Name for the experiment (default: 'EEI-BO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
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
        Execute EEI-BO.

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

        # Persistent CMA-ES state per task, seeded at the sample mean
        cma_states = [init_cma_state(decs[i], dims[i], self.cma_popsize, self.sigma0)
                      for i in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                gp, logEI = fit_gp_ei(decs[i], objs[i], data_type)

                # One persistent CMA-ES generation, ranked by the GP surrogate
                cma_step(cma_states[i], gp, data_type)
                mu = cma_states[i]['m_dec']
                Sigma_real = cma_states[i]['sigma'] ** 2 * cma_states[i]['C']

                candidate = eei_next_point(logEI, mu, Sigma_real, dims[i],
                                           self.n2, self.max_nfes2, data_type)

                new_objs, _ = evaluation_single(problem, candidate, i)
                decs[i], objs[i] = vstack_groups((decs[i], candidate), (objs[i], new_objs))

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# ---------------------------------------------------------------------------
# Shared EEI-BO components (reused by the multi-task EEI-BO+)
# ---------------------------------------------------------------------------

def init_cma_state(decs, dim, popsize, sigma0):
    """Initialize a persistent CMA-ES state seeded at the sample mean."""
    state = cmaes_init_params(dim, lam=popsize, sigma0=sigma0)
    state['m_dec'] = decs.mean(axis=0)
    state['_evals'] = 0
    return state


def fit_gp_ei(decs, objs, data_type=torch.double):
    """
    Fit a GP on (decs, -objs) and build the log Expected Improvement criterion.

    Objectives are negated so the model works in a maximization frame; best_f is
    the incumbent in that frame.
    """
    train_X = torch.tensor(decs, dtype=data_type)
    train_Y = torch.tensor(-objs, dtype=data_type)
    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max())
    return gp, logEI


def cma_step(state, gp, data_type=torch.double):
    """
    Advance the persistent CMA-ES state by one generation.

    Samples ``lam`` points from N(mu, sigma^2 C), ranks them by the GP posterior
    mean (the surrogate; higher mean = smaller true objective, ranked best-first),
    and updates the CMA-ES state in place.
    """
    offdecs = cmaes_sample(state['m_dec'], state['sigma'], state['B'], state['D'], state['lam'])

    xt = torch.tensor(offdecs, dtype=data_type)
    gp.eval()
    with torch.no_grad():
        surrogate = gp.posterior(xt).mean.squeeze(-1).cpu().numpy()

    order = np.argsort(-surrogate)  # best (largest -obj) first
    state['_evals'] += state['lam']
    cmaes_update(state, offdecs[order], state['_evals'])


def _build_eei_objective(logEI, mu, Sigma_real, dim, data_type):
    """
    Build the (negated, for minimization) log-EEI objective log EI(x) + log P(x).

    P is the density of N(mu, Sigma_real); working in log space avoids the
    underflow of EI*P when EI is tiny while preserving the argmax.
    """
    mu = np.asarray(mu, dtype=float).flatten()
    S = np.asarray(Sigma_real, dtype=float) + 1e-12 * np.eye(dim)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:  # numerical fallback: diagonal-load until PD
        S = S + (abs(logdet) + 1.0) * np.eye(dim)
        sign, logdet = np.linalg.slogdet(S)
    log_norm = -0.5 * dim * np.log(2.0 * np.pi) - 0.5 * logdet

    def eei_obj(x):
        x2 = np.atleast_2d(x)
        xt = torch.tensor(x2, dtype=data_type)
        with torch.no_grad():
            log_ei = logEI(xt).detach().cpu().numpy().flatten()
        diff = x2 - mu
        maha = np.sum(diff * np.linalg.solve(S, diff.T).T, axis=1)
        log_p = log_norm - 0.5 * maha
        neg_log_eei = -(log_ei + log_p)  # DE minimizes
        return float(neg_log_eei[0]) if x2.shape[0] == 1 else neg_log_eei

    return eei_obj


def eei_next_point(logEI, mu, Sigma_real, dim, n2, max_nfes2, data_type=torch.double):
    """
    Select the next query point by maximizing EEI = EI * P with differential evolution.

    Parameters
    ----------
    logEI : LogExpectedImprovement
        Log EI acquisition built on the current GP.
    mu : np.ndarray
        Mean of the CMA-ES search distribution, shape (dim,).
    Sigma_real : np.ndarray
        Real covariance sigma^2 C of the CMA-ES search distribution, shape (dim, dim).
    dim : int
        Task dimensionality.
    n2, max_nfes2 : int
        DE population size and evaluation budget for maximizing EEI.

    Returns
    -------
    np.ndarray
        Next query point, shape (1, dim).
    """
    eei_obj = _build_eei_objective(logEI, mu, Sigma_real, dim, data_type)

    problem = MTOP()
    problem.add_task(eei_obj, dim=dim)
    result = DE(problem, n=n2, max_nfes=max_nfes2, F=0.5, CR=0.9,
                save_data=False, disable_tqdm=True).optimize()

    return np.asarray(result.best_decs[0]).reshape(1, -1)

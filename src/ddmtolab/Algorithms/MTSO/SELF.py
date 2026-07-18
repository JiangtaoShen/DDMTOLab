"""
A Surrogate-Assisted Evolutionary Framework for Expensive Multitask Optimization Problems (SELF)

This module implements SELF using multi-task Gaussian processes and Bayesian
optimization for expensive multi-task optimization. Each optimization cycle
runs two phases (paper Algorithm 1): a global knowledge transfer phase, where
DE assisted with an MTGP surrogate preselects one candidate per individual by
the lower confidence bound (Eq. 18), and a local knowledge transfer phase,
where task-local BO improves each population's best individual (Algorithm 3)
and the improved best individuals are transferred across tasks with the
correlation-based probability p_T (Eq. 23, Algorithm 4).

References
----------
    [1] S. Tan, Y. Wang, G. Sun, T. Pang, and K. Tang, "A surrogate-assisted
        evolutionary framework for expensive multitask optimization problems,"
        IEEE Transactions on Evolutionary Computation, vol. 29, no. 3,
        pp. 779-793, 2025.

Notes
-----
Corrected against the paper (v2.0):
- Global phase uses DE/rand/1 with binomial crossover against the current
  individual (Algorithm 2, Eqs. 19-20) and preselects candidates by the LCB
  mean - std (Eq. 18), not the posterior mean alone.
- Local BO builds the GP on the n nearest evaluated solutions to the best
  individual and maximizes EI inside the bounding box of those solutions
  (Eq. 21); the BO candidate competes for the best individual's slot.
- Transfer moves the BO-improved best individuals X_I (not population bests)
  with probability p_T from the MTGP task correlation (no absolute value);
  the best transferred solution competes for the best individual's slot.
- Paper parameter defaults: NP=10, lambda=50, F=0.6, CR=0.7, n=50,
  MaxFEs=200 per task (400 total for two tasks).
Validated on the paper's 10-D CEC17-MTSO test suite (Table II protocol).

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.07.18
Version: 2.0
"""
import numpy as np
from tqdm import tqdm
import torch
import time

from ddmtolab.Methods.Algo_Methods.bo_utils import mtgp_predict, mtgp_build, mtgp_task_corr
from ddmtolab.Methods.mtop import MTOP
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.models.transforms import Standardize
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.STSO.DE import DE
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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'equal',
        'max_nfes': 'unequal, controlled by SELF'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, max_nfes=None, np=None, F=0.6, CR=0.7, ng=50, nl=50, save_data=True, save_path='./Data',
                 name='SELF', disable_tqdm=True):
        """
        Initialize SELF algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200,
            the paper's setting of MaxFEs = 400 for two tasks)
        np : int, optional
            Population size NP per task (default: 10, paper setting)
        F : float, optional
            Mutation factor for DE (default: 0.6, paper setting)
        CR : float, optional
            Crossover rate for DE (default: 0.7, paper setting)
        ng : int, optional
            Number of trial vectors lambda per individual in the global
            knowledge transfer phase (default: 50, paper setting)
        nl : int, optional
            Number n of nearest evaluated solutions used to train the local
            GP model in the local knowledge transfer phase (default: 50,
            paper setting)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SELF')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.np = np if np is not None else 10
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
        Execute the SELF algorithm (paper Algorithm 1).

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
        nfes_per_task = [0] * nt
        max_nfes = self.max_nfes * nt

        # Line 1-4: initialize populations by LHS, evaluate, fill databases
        decs = initialization(problem, self.np, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes = self.np * nt
        for i in range(nt):
            nfes_per_task[i] += self.np

        # Working populations P_m (databases are decs/objs)
        pop_decs = copy.deepcopy(decs)
        pop_objs = copy.deepcopy(objs)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:

            # ==================== Global Knowledge Transfer Phase ====================
            # Line 7: establish MTGP on all evaluated data
            objs_normalized, _, _ = normalize(objs, axis=0, method='minmax')
            mtgp = mtgp_build(decs, objs_normalized, dims, data_type=data_type)
            task_corr = mtgp_task_corr(mtgp)

            # Lines 8-10: DE_MTGP per task (Algorithm 2)
            for i in range(nt):
                for j in range(self.np):
                    # DE/rand/1 + binomial crossover vs the current individual
                    # (Eqs. 19-20): lambda trial vectors
                    off_decs = de_rand_1_bin_trials(pop_decs[i], pop_decs[i][j],
                                                    self.ng, self.F, self.CR)

                    # Preselect by LCB mean - std of the MTGP (Eq. 18)
                    pred_objs, pred_std = mtgp_predict(mtgp=mtgp, off_decs=off_decs, task_id=i, dims=dims, nt=nt,
                                                       data_type=data_type)
                    lcb = pred_objs.flatten() - pred_std.flatten()

                    best_idx = np.argmin(lcb)
                    best_off_dec = off_decs[[best_idx], :]
                    true_obj, _ = evaluation_single(problem, best_off_dec, i)

                    # Line 9 (Alg. 2): survival between x_{m,i} and o*_{m,i}
                    if true_obj < pop_objs[i][j]:
                        pop_decs[i][j] = best_off_dec[0]
                        pop_objs[i][j] = true_obj[0]

                    decs[i], objs[i] = vstack_groups((decs[i], best_off_dec), (objs[i], true_obj))

                    nfes += 1
                    nfes_per_task[i] += 1
                    pbar.update(1)

            # ==================== Local Knowledge Transfer Phase ====================
            # Lines 12-16: BO per task (Algorithm 3); collect improved bests X_I
            transfer_pool = []
            for i in range(nt):
                # Best individual x_Bm and its position in P_m
                best_pos = int(np.argmin(pop_objs[i].flatten()))
                x_best = pop_decs[i][best_pos]

                # Local GP on the nl nearest evaluated solutions to x_Bm
                dists = np.linalg.norm(decs[i] - x_best[None, :], axis=1)
                near_idx = np.argsort(dists)[:min(self.nl, len(dists))]
                nearest_decs = decs[i][near_idx]
                nearest_objs = objs[i][near_idx]

                # Maximize EI inside the bounding box of the nearest solutions
                # (Eq. 21), then evaluate the improved best individual x_Im
                candidate = bo_next_point_de(nearest_decs, nearest_objs, dims[i], data_type)
                true_obj, _ = evaluation_single(problem, candidate, i)
                transfer_pool.append(candidate[0])

                # Line 7 (Alg. 3): better of (x_Im, x_Bm) takes the best slot
                if true_obj[0, 0] < pop_objs[i][best_pos, 0]:
                    pop_decs[i][best_pos] = candidate[0]
                    pop_objs[i][best_pos] = true_obj[0]

                decs[i], objs[i] = vstack_groups((decs[i], candidate), (objs[i], true_obj))

                nfes += 1
                nfes_per_task[i] += 1
                pbar.update(1)

            # Lines 17-20: adaptive knowledge transfer of X_I (Algorithm 4)
            for i in range(nt):
                transfer_samples = []
                for j in range(nt):
                    if i == j:
                        continue
                    # p_T from the MTGP task correlation (Eq. 23); negative
                    # correlation means the transfer never fires
                    if np.random.rand() < task_corr[i][j]:
                        sample = transfer_pool[j]
                        if len(sample) > dims[i]:
                            sample = sample[:dims[i]]
                        elif len(sample) < dims[i]:
                            sample = np.concatenate([sample, np.zeros(dims[i] - len(sample))])
                        transfer_samples.append(sample)

                if len(transfer_samples) > 0:
                    transfer_samples = np.array(transfer_samples)

                    true_obj, _ = evaluation_single(problem, transfer_samples, i)

                    # Line 14 (Alg. 4): best transferred solution competes for
                    # the best individual's slot
                    best_idx = np.argmin(true_obj)
                    best_sample = transfer_samples[best_idx]
                    best_sample_obj = true_obj[best_idx]

                    best_pos = int(np.argmin(pop_objs[i].flatten()))
                    if best_sample_obj[0] < pop_objs[i][best_pos, 0]:
                        pop_decs[i][best_pos] = best_sample
                        pop_objs[i][best_pos] = best_sample_obj

                    decs[i], objs[i] = vstack_groups((decs[i], transfer_samples), (objs[i], true_obj))

                    nfes += len(transfer_samples)
                    nfes_per_task[i] += len(transfer_samples)
                    pbar.update(len(transfer_samples))

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        # Trim excess samples so each task has exactly max_nfes_per_task points
        max_nfes_per_task = par_list(self.max_nfes, nt)
        all_decs, all_objs, nfes_per_task = trim_excess_evaluations(
            all_decs, all_objs, nt, max_nfes_per_task, nfes_per_task
        )
        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def de_rand_1_bin_trials(parents, current, n_off, F, CR):
    """
    Generate trial vectors for one individual per paper Algorithm 2.

    DE/rand/1 mutation (Eq. 19): v = x_r1 + F * (x_r2 - x_r3) with distinct
    random parents, followed by binomial crossover (Eq. 20) between v and the
    current individual with a forced dimension j_rd.

    Parameters
    ----------
    parents : np.ndarray
        Population P_m of shape (n, d)
    current : np.ndarray
        Current individual x_{m,i} of shape (d,) or (1, d)
    n_off : int
        Number lambda of trial vectors to generate
    F : float
        Scaling factor
    CR : float
        Crossover control parameter in [0, 1]

    Returns
    -------
    trials : np.ndarray
        Trial vectors of shape (n_off, d), clipped to [0, 1]
    """
    n, d = parents.shape
    if current.ndim == 2:
        current = current.squeeze(0)

    trials = np.zeros((n_off, d), dtype=float)
    for c in range(n_off):
        r1, r2, r3 = np.random.permutation(n)[:3]
        v = parents[r1] + F * (parents[r2] - parents[r3])

        cross = np.random.rand(d) <= CR
        cross[np.random.randint(d)] = True
        trials[c] = np.where(cross, v, current)

    return np.clip(trials, 0.0, 1.0)


def bo_next_point_de(decs, objs, dim, data_type=torch.float):
    """
    Solve the EI subproblem of paper Eq. 21 for the local BO step.

    Builds a local GP on the given (nearest) solutions and maximizes log
    expected improvement inside the bounding box of those solutions using
    DE/rand/1/bin, as prescribed by the paper.

    Parameters
    ----------
    decs : np.ndarray
        Local training decision variables of shape (n_samples, dim)
    objs : np.ndarray
        Local training objective values of shape (n_samples, 1)
    dim : int
        Dimension of the task
    data_type : torch.dtype, optional
        Data type for torch tensors (default: torch.float)

    Returns
    -------
    candidate_np : np.ndarray
        Next sampling point of shape (1, dim)
    """
    # Prepare training data for the local GP (negated for maximization)
    train_X = torch.tensor(decs, dtype=data_type)
    train_Y = torch.tensor(-objs, dtype=data_type)

    gp = SingleTaskGP(train_X=train_X, train_Y=train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # f_min in Eq. 21 is the best value among the local solutions, which
    # include the best individual (distance zero to itself)
    best_f = train_Y.max()
    logEI = LogExpectedImprovement(model=gp, best_f=best_f)

    # Bounding box [L~, U~] of the local solutions (Eq. 21 constraints)
    lb = decs.min(axis=0)
    ub = decs.max(axis=0)
    span = np.maximum(ub - lb, 1e-12)

    # Inner DE searches [0, 1]^dim mapped affinely into the bounding box
    def logEI_func(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_box = lb[None, :] + x * span[None, :]
        x_torch = torch.tensor(x_box, dtype=data_type)
        with torch.no_grad():
            logei_value = logEI(x_torch)
        logei_np = logei_value.detach().cpu().numpy()
        return -logei_np.flatten() if x.shape[0] == 1 else -logei_np

    problem = MTOP()
    problem.add_task(logEI_func, dim=dim)
    de = DE(problem, n=50, max_nfes=5000, F=0.5, CR=0.9, save_data=False, disable_tqdm=True)
    result = de.optimize()

    best_x = np.asarray(result.best_decs[0]).flatten()
    return (lb + best_x * span).reshape(1, -1)

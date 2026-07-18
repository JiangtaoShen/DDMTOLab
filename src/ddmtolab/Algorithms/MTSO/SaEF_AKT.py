"""
SaEF-AKT: Surrogate-Assisted Evolutionary Framework with Adaptive Knowledge Transfer

This module implements SaEF-AKT for expensive multi-task single-objective optimization.

References
----------
    [1] Z. Huang, J. Zhong, and W. N. N. Yu, "Surrogate-Assisted Evolutionary Framework with Adaptive Knowledge Transfer for Multi-Task Optimization," IEEE Trans. Emerg. Topics Comput., vol. 9, no. 4, pp. 1930-1944, 2021.

Notes
-----
Verified against the paper (v1.1). Structure matches Algorithm 2 and the
component equations: local GP on the NC nearest + NR most-recent points, merit
functions f_M = t_hat - g*s_t with g in {0, 1, 2, 4} optimized inside the
hypercube around x_best (Eqs. 15-16), KLD similarity between the training-data
normal distributions (Eq. 21), Softmax-normalized reciprocal similarity
(Eqs. 23-24), pheromone/similarity transfer selection (Eqs. 22, 25), and
pheromone volatilize/reward/clamp (Eqs. 26-28). Default parameters follow the
paper: NC = NR = 60, Phe_a = 0.1, Phe_max = 1.0, Phe_min = 0.01, P_max = 0.9,
DE population 15 with 2000 merit evaluations.

Correction: the GPOP phase now evaluates the optima of all four merit functions
with the true fitness each generation (the defining exploration mechanism of
GPOP), instead of the previous single best-predicted candidate.

Deliberate substitutions (inner-solver / modeling choices that do not change the
algorithm's semantics): the merit functions and GP hyper-parameters are
optimized with the platform's DE / botorch's L-BFGS rather than the paper's
JADE, and the GP uses botorch's default Matern 5/2 kernel rather than the
paper's squared-exponential-with-periodic kernel (more robust on general,
non-periodic landscapes).

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.07.18
Version: 1.1
"""
import time
import warnings
import numpy as np
import torch
from tqdm import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *

warnings.filterwarnings("ignore")


class SaEF_AKT:
    """
    Surrogate-Assisted Evolutionary Framework with Adaptive Knowledge Transfer.

    This algorithm features:
    - Local Gaussian Process modeling (NC nearest + NR recent points)
    - Multiple merit functions with different exploration-exploitation balance (g = 0, 1, 2, 4)
    - DE-based optimization of merit functions on GP surrogate
    - KL divergence-based task similarity measurement
    - Pheromone-based adaptive auxiliary task selection for knowledge transfer

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
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None,
                 NC=60, NR=60, Phe_a=0.1, Phe_max=1.0, Phe_min=0.01,
                 P_max=0.9, g_values=None,
                 de_pop=15, de_merit_evals=2000,
                 save_data=True, save_path='./Data',
                 name='SaEF-AKT', disable_tqdm=True):
        """
        Initialize SaEF-AKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 100)
        NC : int, optional
            Number of nearest neighbor points for local GP (default: 60)
        NR : int, optional
            Number of most recent evaluation points for local GP (default: 60)
        Phe_a : float, optional
            Pheromone evaporation rate (default: 0.1)
        Phe_max : float, optional
            Maximum pheromone concentration (default: 1.0)
        Phe_min : float, optional
            Minimum pheromone concentration (default: 0.01)
        P_max : float, optional
            Probability of selecting the task with maximum transfer probability (default: 0.9)
        g_values : list of float, optional
            Exploration weights for merit functions (default: [0, 1, 2, 4])
        de_pop : int, optional
            DE population size for merit function optimization (default: 15)
        de_merit_evals : int, optional
            Max DE evaluations for merit function optimization (default: 2000)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SaEF-AKT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 100
        self.NC = NC
        self.NR = NR
        self.Phe_a = Phe_a
        self.Phe_max = Phe_max
        self.Phe_min = Phe_min
        self.P_max = P_max
        self.g_values = g_values if g_values is not None else [0, 1, 2, 4]
        self.de_pop = de_pop
        self.de_merit_evals = de_merit_evals
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SaEF-AKT algorithm.

        The main loop follows Algorithm 3 in the paper:
        Phase 1: GPOP search + real evaluation for each task (1 FE per task)
        Phase 2: Similarity measurement via KL divergence on updated databases
        Phase 3: Knowledge transfer + pheromone update for each task (1 FE per task)

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = [n_initial_per_task[i] for i in range(nt)]

        # Initialize pheromone matrix: tau[r, s] = pheromone of s assisting r
        tau = np.full((nt, nt), (self.Phe_max + self.Phe_min) / 2)
        np.fill_diagonal(tau, 0.0)

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            # ===== Phase 1: Surrogate-Assisted EA (GPOP) =====
            # For each task: build a local GP and evaluate the optima of all four
            # merit functions (g = 0, 1, 2, 4) with the real fitness (GPOP).
            for i in active_tasks:
                if nfes_per_task[i] >= max_nfes_per_task[i]:
                    continue

                candidates = self._gpop_search(decs[i], objs[i], dims[i])
                for candidate in candidates:
                    if nfes_per_task[i] >= max_nfes_per_task[i]:
                        break
                    candidate = candidate.reshape(1, -1)
                    obj_new, _ = evaluation_single(problem, candidate, i)
                    decs[i] = np.vstack([decs[i], candidate])
                    objs[i] = np.vstack([objs[i], obj_new])
                    nfes_per_task[i] += 1
                    pbar.update(1)

            # ===== Phase 2: Similarity Measurement via KL Divergence =====
            # Uses updated databases (including Phase 1 evaluations)
            train_data = {}
            for i in active_tasks:
                train_data[i] = self._select_training_data(decs[i], objs[i])

            eta = self._compute_similarity(train_data, dims, active_tasks, nt)

            # ===== Phase 3: Knowledge Transfer + Pheromone Update =====
            for r in active_tasks:
                if nfes_per_task[r] >= max_nfes_per_task[r]:
                    continue

                # Select auxiliary task
                at = self._select_auxiliary_task(r, tau, eta, nt, active_tasks)

                if at is not None:
                    # Transfer the optimal solution (x_best) of auxiliary task AT(r)
                    best_idx_at = np.argmin(objs[at].flatten())
                    transfer_dec = decs[at][best_idx_at].copy()

                    # Dimension alignment
                    if len(transfer_dec) < dims[r]:
                        transfer_dec = np.pad(transfer_dec, (0, dims[r] - len(transfer_dec)),
                                              mode='constant', constant_values=0.5)
                    elif len(transfer_dec) > dims[r]:
                        transfer_dec = transfer_dec[:dims[r]]

                    transfer_dec = np.clip(transfer_dec, 0, 1).reshape(1, -1)

                    # Evaluate transferred solution with real fitness of task r
                    obj_transfer, _ = evaluation_single(problem, transfer_dec, r)
                    nfes_per_task[r] += 1
                    pbar.update(1)

                    # Pheromone evaporation
                    tau[r, at] = (1 - self.Phe_a) * tau[r, at]

                    # Success reinforcement: if transfer improves best-so-far
                    if obj_transfer.flatten()[0] < np.min(objs[r]):
                        tau[r, at] = self.Phe_max

                    # Clamp pheromone
                    tau[r, at] = np.clip(tau[r, at], self.Phe_min, self.Phe_max)

                    # Add evaluated solution to task r's dataset
                    decs[r] = np.vstack([decs[r], transfer_dec])
                    objs[r] = np.vstack([objs[r], obj_transfer])

        pbar.close()
        runtime = time.time() - start_time

        # Build staircase history for analysis
        all_decs, all_objs = build_staircase_history(decs, objs, k=1)

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    def _select_training_data(self, decs_t, objs_t):
        """
        Select local training data: NC nearest to x_best + NR most recent.

        Parameters
        ----------
        decs_t : np.ndarray
            All decision variables for this task, shape (n, dim)
        objs_t : np.ndarray
            All objective values for this task, shape (n, 1)

        Returns
        -------
        np.ndarray
            Selected training decisions, shape (n_train, dim)
        """
        n = len(decs_t)
        best_idx = np.argmin(objs_t.flatten())
        x_best = decs_t[best_idx]

        # NC nearest neighbors
        nc = min(self.NC, n)
        distances = cdist(x_best.reshape(1, -1), decs_t)[0]
        nearest_idx = np.argsort(distances)[:nc]

        # NR most recent
        nr = min(self.NR, n)
        recent_idx = np.arange(max(0, n - nr), n)

        # Union of both sets
        selected_idx = np.unique(np.concatenate([nearest_idx, recent_idx]))

        return decs_t[selected_idx]

    def _gpop_search(self, decs_t, objs_t, dim_t):
        """
        GPOP: GP-based local search with multiple merit functions.

        Parameters
        ----------
        decs_t : np.ndarray
            All decision variables for this task, shape (n, dim)
        objs_t : np.ndarray
            All objective values for this task, shape (n, 1)
        dim_t : int
            Dimension of this task

        Returns
        -------
        list of np.ndarray
            Optima of the four merit functions (g = 0, 1, 2, 4), each of shape
            (dim_t,); near-duplicates removed. Empty if the GP cannot be built.
        """
        n = len(decs_t)
        best_idx = np.argmin(objs_t.flatten())
        x_best = decs_t[best_idx]

        # Select local training data
        nc = min(self.NC, n)
        distances = cdist(x_best.reshape(1, -1), decs_t)[0]
        nearest_idx = np.argsort(distances)[:nc]

        nr = min(self.NR, n)
        recent_idx = np.arange(max(0, n - nr), n)
        selected_idx = np.unique(np.concatenate([nearest_idx, recent_idx]))

        train_x = decs_t[selected_idx]
        train_y = objs_t[selected_idx].flatten()

        # Define local search bounds based on NC nearest points range
        nc_decs = decs_t[nearest_idx]
        d = np.max(nc_decs, axis=0) - np.min(nc_decs, axis=0)
        lb = np.clip(x_best - d / 2, 0, 1)
        ub = np.clip(x_best + d / 2, 0, 1)

        # Ensure minimum range
        too_small = (ub - lb) < 1e-6
        lb[too_small] = np.clip(x_best[too_small] - 0.05, 0, 1)
        ub[too_small] = np.clip(x_best[too_small] + 0.05, 0, 1)

        # Build GP model
        try:
            gp, y_mean, y_std = self._build_gp(train_x, train_y)
        except Exception:
            return []

        # Optimize each merit function f_M = mean - g*std (g = 0, 1, 2, 4). Each
        # optimum is a distinct exploration/exploitation trade-off; per the GPOP
        # procedure all of them are returned to be evaluated with the true
        # fitness (g=0 exploits the surrogate mean, larger g probes uncertain
        # regions). Near-duplicates are dropped to avoid wasting evaluations.
        candidates = []
        for g in self.g_values:
            try:
                cand = self._optimize_merit(gp, y_mean, y_std, lb, ub, dim_t, g)
            except Exception:
                continue
            if not any(np.allclose(cand, c, atol=1e-6) for c in candidates):
                candidates.append(cand)

        return candidates

    def _build_gp(self, train_x, train_y):
        """
        Build and fit a local GP model.

        Parameters
        ----------
        train_x : np.ndarray
            Training decisions, shape (n, dim)
        train_y : np.ndarray
            Training objectives, shape (n,)

        Returns
        -------
        tuple
            (gp_model, y_mean, y_std) for denormalization
        """
        train_x_t = torch.tensor(train_x, dtype=torch.double)
        train_y_t = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)

        # Standardize Y
        y_mean = train_y_t.mean()
        y_std = train_y_t.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, dtype=torch.double)
        train_y_norm = (train_y_t - y_mean) / y_std

        gp = SingleTaskGP(train_x_t, train_y_norm)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        return gp, y_mean, y_std

    def _batch_merit(self, gp, y_mean, y_std, X, g):
        """
        Batch evaluate merit function for a population.

        Parameters
        ----------
        gp : SingleTaskGP
            Fitted GP model
        y_mean, y_std : torch.Tensor
            Standardization parameters
        X : np.ndarray
            Population, shape (n, dim)
        g : float
            Exploration weight

        Returns
        -------
        np.ndarray
            Merit values, shape (n,)
        """
        X_t = torch.tensor(X, dtype=torch.double)
        with torch.no_grad():
            posterior = gp.posterior(X_t)
            means = posterior.mean.squeeze(-1).numpy() * y_std.item() + y_mean.item()
            stds = posterior.variance.squeeze(-1).sqrt().numpy() * y_std.item()
        return means - g * stds

    def _optimize_merit(self, gp, y_mean, y_std, lb, ub, dim, g):
        """
        Optimize merit function using DE within local bounds (batch evaluation).

        Parameters
        ----------
        gp : SingleTaskGP
            Fitted GP model
        y_mean, y_std : torch.Tensor
            Standardization parameters
        lb, ub : np.ndarray
            Lower and upper bounds, shape (dim,)
        dim : int
            Decision space dimension
        g : float
            Exploration weight

        Returns
        -------
        np.ndarray
            Best candidate found, shape (dim,)
        """
        pop_size = self.de_pop
        max_gen = self.de_merit_evals // pop_size
        F, CR = 0.5, 0.9

        # Initialize population within local bounds
        pop = lb + (ub - lb) * np.random.rand(pop_size, dim)

        # Batch evaluate initial population
        fit = self._batch_merit(gp, y_mean, y_std, pop, g)

        for _ in range(max_gen):
            # Generate all trials at once
            trials = np.empty_like(pop)
            for i in range(pop_size):
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trials[i] = np.where(mask, mutant, pop[i])

            # Batch evaluate all trials
            trial_fit = self._batch_merit(gp, y_mean, y_std, trials, g)

            # Selection
            improved = trial_fit < fit
            pop[improved] = trials[improved]
            fit[improved] = trial_fit[improved]

        return pop[np.argmin(fit)]

    def _compute_similarity(self, train_data, dims, active_tasks, nt):
        """
        Compute task similarity matrix based on KL divergence.

        Parameters
        ----------
        train_data : dict
            Selected training data per task {task_idx: np.ndarray}
        dims : list of int
            Dimensions per task
        active_tasks : list of int
            Active task indices
        nt : int
            Total number of tasks

        Returns
        -------
        np.ndarray
            Normalized similarity matrix, shape (nt, nt)
        """
        eta = np.zeros((nt, nt))

        for r in active_tasks:
            for s in active_tasks:
                if r == s:
                    continue

                # Align dimensions to min(dim_r, dim_s)
                d_min = min(dims[r], dims[s])
                data_r = train_data[r][:, :d_min]
                data_s = train_data[s][:, :d_min]

                kld = self._kl_divergence(data_r, data_s)

                # Similarity = 1 / KLD (avoid division by zero)
                if kld > 1e-10:
                    eta[r, s] = 1.0 / kld
                else:
                    eta[r, s] = 1e10

        # Softmax normalization per row
        for r in active_tasks:
            row = eta[r, :]
            nonzero = row > 0
            if np.any(nonzero):
                row_vals = row[nonzero]
                # Clip for numerical stability
                row_vals = np.clip(row_vals, -500, 500)
                exp_vals = np.exp(row_vals - np.max(row_vals))
                row[nonzero] = exp_vals / np.sum(exp_vals)
                eta[r, :] = row

        return eta

    @staticmethod
    def _kl_divergence(data_0, data_1):
        """
        Compute KL divergence between two multivariate normal distributions
        fitted to the data.

        Parameters
        ----------
        data_0 : np.ndarray
            Data from distribution 0, shape (n0, d)
        data_1 : np.ndarray
            Data from distribution 1, shape (n1, d)

        Returns
        -------
        float
            KL(N_0 || N_1)
        """
        k = data_0.shape[1]

        m0 = np.mean(data_0, axis=0)
        m1 = np.mean(data_1, axis=0)

        # Empirical covariance with regularization
        C0 = np.cov(data_0, rowvar=False) + 1e-6 * np.eye(k)
        C1 = np.cov(data_1, rowvar=False) + 1e-6 * np.eye(k)

        try:
            C1_inv = np.linalg.inv(C1)
            sign0, logdet0 = np.linalg.slogdet(C0)
            sign1, logdet1 = np.linalg.slogdet(C1)

            if sign0 <= 0 or sign1 <= 0:
                return 1e10

            diff = m1 - m0
            kld = 0.5 * (np.trace(C1_inv @ C0) + diff @ C1_inv @ diff - k + logdet1 - logdet0)
            return max(kld, 0.0)
        except np.linalg.LinAlgError:
            return 1e10

    def _select_auxiliary_task(self, r, tau, eta, nt, active_tasks):
        """
        Select auxiliary task for knowledge transfer using pheromone and similarity.

        Parameters
        ----------
        r : int
            Target task index
        tau : np.ndarray
            Pheromone matrix, shape (nt, nt)
        eta : np.ndarray
            Similarity matrix, shape (nt, nt)
        nt : int
            Number of tasks
        active_tasks : list of int
            Active task indices

        Returns
        -------
        int or None
            Selected auxiliary task index, or None if no valid task
        """
        candidates = [s for s in active_tasks if s != r]
        if not candidates:
            return None

        # Compute transfer probabilities: P_kt(r, s) = tau(r,s) * eta(r,s) / sum
        probs = np.array([tau[r, s] * eta[r, s] for s in candidates])
        total = np.sum(probs)

        if total <= 0:
            return candidates[np.random.randint(len(candidates))]

        probs = probs / total

        # Selection rule
        if np.random.rand() < self.P_max:
            # Greedy: select task with maximum probability
            return candidates[np.argmax(probs)]
        else:
            # Roulette wheel selection
            cumsum = np.cumsum(probs)
            r_val = np.random.rand()
            for j, c in enumerate(cumsum):
                if r_val <= c:
                    return candidates[j]
            return candidates[-1]

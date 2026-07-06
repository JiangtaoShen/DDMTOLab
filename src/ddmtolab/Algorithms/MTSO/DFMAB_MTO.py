"""
DFMAB-MTO (Dual-Feedback Multi-Armed Bandit for Expensive Multi-Task Optimization)

This module implements the DFMAB-MTO algorithm using Discounted UCB (D-UCB) for
adaptive surrogate model selection and knowledge transfer control.

References
----------
    [1] Shen, J., et al. "Dual Feedback Multi-Armed Bandit for Expensive Multitask Optimization."

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12
Version: 1.1
"""
import time
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
from ddmtolab.Algorithms.STSO.CMA_ES import CMA_ES
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Surrogate Pool
# =============================================================================

class SurrogateModel(ABC):
    """Abstract base class for surrogate models."""

    @abstractmethod
    def fit(self, X, y):
        """Fit the surrogate model."""

    @abstractmethod
    def predict(self, X):
        """Predict objective values. Returns (mean, std)."""

    @abstractmethod
    def cv_rmse(self, X, y, n_folds=5):
        """Compute K-Fold Cross-Validation RMSE."""


class GPSurrogate(SurrogateModel):
    """Gaussian Process surrogate using BoTorch SingleTaskGP (via bo_utils)."""

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = gp_build(X, y)

    def predict(self, X):
        pred_objs, pred_std = gp_predict(self.model, X)
        return pred_objs.flatten(), pred_std.flatten()

    def cv_rmse(self, X, y, n_folds=5):
        n = len(X)
        if n < n_folds:
            n_folds = max(2, n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        errors = []
        for train_idx, val_idx in kf.split(X):
            try:
                gp = gp_build(X[train_idx], y[train_idx])
                pred, _ = gp_predict(gp, X[val_idx])
                errors.append(mean_squared_error(y[val_idx].flatten(), pred.flatten()))
            except Exception:
                errors.append(1e6)
        return np.sqrt(np.mean(errors))


class RBFSurrogate(SurrogateModel):
    """Radial Basis Function surrogate using algo_utils rbf_build/rbf_predict."""

    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.flatten().copy()
        mS, mY = dsmerge(self.X_train, self.y_train)
        self.X_train = mS
        self.y_train = mY
        self.model = rbf_build(self.X_train, self.y_train)

    def predict(self, X):
        mu = rbf_predict(self.model, self.X_train, X)
        return mu, np.zeros_like(mu)

    def cv_rmse(self, X, y, n_folds=5):
        n = len(X)
        if n < n_folds:
            n_folds = max(2, n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        errors = []
        for train_idx, val_idx in kf.split(X):
            try:
                X_tr, y_tr = X[train_idx], y[train_idx].flatten()
                mS, mY = dsmerge(X_tr, y_tr)
                model = rbf_build(mS, mY)
                pred = rbf_predict(model, mS, X[val_idx])
                errors.append(mean_squared_error(y[val_idx].flatten(), pred))
            except Exception:
                errors.append(1e6)
        return np.sqrt(np.mean(errors))


class PRSurrogate(SurrogateModel):
    """Polynomial Regression surrogate (degree=2) with Ridge regularization."""

    def __init__(self, degree=2, alpha=1e-3):
        self.degree = degree
        self.alpha = alpha
        self.poly = None
        self.model = None

    def fit(self, X, y):
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_poly, y.flatten())

    def predict(self, X):
        X_poly = self.poly.transform(X)
        mu = self.model.predict(X_poly)
        return mu, np.zeros_like(mu)

    def cv_rmse(self, X, y, n_folds=5):
        n = len(X)
        if n < n_folds:
            n_folds = max(2, n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        errors = []
        for train_idx, val_idx in kf.split(X):
            try:
                poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                X_tr = poly.fit_transform(X[train_idx])
                model = Ridge(alpha=self.alpha)
                model.fit(X_tr, y[train_idx].flatten())
                X_val = poly.transform(X[val_idx])
                pred = model.predict(X_val)
                errors.append(mean_squared_error(y[val_idx].flatten(), pred))
            except Exception:
                errors.append(1e6)
        return np.sqrt(np.mean(errors))


class KNNSurrogate(SurrogateModel):
    """K-Nearest Neighbors regression surrogate.

    Uses distance-weighted KNN regression. k is set adaptively as
    min(max(5, n//5), n-1) to balance bias and variance.
    """

    def __init__(self):
        self.model = None
        self.k = 5

    def fit(self, X, y):
        n = len(X)
        self.k = min(max(5, n // 5), n - 1)
        self.model = KNeighborsRegressor(
            n_neighbors=self.k, weights='distance', metric='minkowski', p=2
        )
        self.model.fit(X, y.flatten())

    def predict(self, X):
        mu = self.model.predict(X)
        # Estimate std via leave-one-out variance of neighbor targets
        dists, idxs = self.model.kneighbors(X)
        y_train = self.model._y
        neighbor_vals = y_train[idxs]  # (n_query, k)
        std = np.std(neighbor_vals, axis=1)
        return mu, std

    def cv_rmse(self, X, y, n_folds=5):
        n = len(X)
        if n < n_folds:
            n_folds = max(2, n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        errors = []
        for train_idx, val_idx in kf.split(X):
            try:
                k = min(max(5, len(train_idx) // 5), len(train_idx) - 1)
                model = KNeighborsRegressor(
                    n_neighbors=k, weights='distance', metric='minkowski', p=2
                )
                model.fit(X[train_idx], y[train_idx].flatten())
                pred = model.predict(X[val_idx])
                errors.append(mean_squared_error(y[val_idx].flatten(), pred))
            except Exception:
                errors.append(1e6)
        return np.sqrt(np.mean(errors))


SURROGATE_POOL = {'GP': GPSurrogate, 'RBF': RBFSurrogate, 'PR': PRSurrogate, 'KNN': KNNSurrogate}


# =============================================================================
# MAB Algorithm
# =============================================================================

class DiscountedUCB:
    """Discounted UCB with exponential forgetting for non-stationarity."""

    def __init__(self, n_arms, gamma=0.95):
        self.n_arms = n_arms
        self.t = 0
        self.gamma = gamma
        self.discounted_rewards = np.zeros(n_arms)
        self.discounted_counts = np.ones(n_arms) * 1e-6
        self.discounted_total = 1e-6

    def select_arm(self, rewards):
        self.t += 1
        mu = rewards.copy()
        exploration = np.sqrt(2 * np.log(max(self.discounted_total, 1)) /
                              np.maximum(self.discounted_counts, 1e-6))
        return int(np.argmax(mu + exploration))

    def update(self, arm, reward):
        self.discounted_rewards *= self.gamma
        self.discounted_counts *= self.gamma
        self.discounted_total = self.discounted_total * self.gamma + 1
        self.discounted_rewards[arm] += reward
        self.discounted_counts[arm] += 1


# =============================================================================
# Dual Feedback Signal Computation
# =============================================================================

class DualFeedbackMAB1:
    """MAB-1: Surrogate Model Selection with dual feedback signals."""

    def __init__(self, model_names, mab, gamma=0.95):
        self.model_names = model_names
        self.n_arms = len(model_names)
        self.mab = mab
        self.gamma = gamma
        self.improvement_counts = np.zeros(self.n_arms)
        self.n_improvements = np.zeros(self.n_arms)
        self.cv_errors = np.ones(self.n_arms)

    def compute_reward(self):
        """Compute fused reward: R = λ * r_proxy + (1-λ) * r_true."""
        rewards = np.zeros(self.n_arms)
        for m in range(self.n_arms):
            r_proxy = -self.cv_errors[m]
            r_true = self.improvement_counts[m]
            lam = 1.0 / (1.0 + self.n_improvements[m])
            rewards[m] = lam * r_proxy + (1 - lam) * r_true
        return rewards

    def select_model(self):
        rewards = self.compute_reward()
        return self.mab.select_arm(rewards)

    def update_cv(self, arm, cv_rmse):
        self.cv_errors[arm] = cv_rmse

    def update_improvement(self, arm, improved):
        self.improvement_counts *= self.gamma
        if improved:
            self.improvement_counts[arm] += 1.0
            self.n_improvements[arm] += 1


class DualFeedbackMAB2:
    """MAB-2: Knowledge Transfer Control with dual feedback signals.

    Proxy signal: KL divergence similarity between task data distributions
    (adapted from SaEF-AKT). Fits multivariate normals to each task's
    decision variable data and computes pairwise KL divergence as similarity.
    Self-arm gets similarity = 1.0.
    """

    def __init__(self, n_tasks, task_idx, mab, gamma=0.95):
        self.n_tasks = n_tasks
        self.task_idx = task_idx
        self.n_arms = n_tasks
        self.mab = mab
        self.gamma = gamma
        self.transfer_counts = np.zeros(n_tasks)

    def compute_reward(self, similarity_matrix):
        """Compute fused reward: r_proxy * w1 + r_true * w2.

        Parameters
        ----------
        similarity_matrix : np.ndarray, shape (nt, nt)
            KL divergence similarity matrix from compute_kl_similarity().
        """
        w1, w2 = 0.5, 0.5
        i = self.task_idx

        raw_proxy = np.zeros(self.n_arms)
        for j in range(self.n_arms):
            if j == i:
                raw_proxy[j] = 1.0
            else:
                raw_proxy[j] = similarity_matrix[i, j]

        rewards = np.zeros(self.n_arms)
        for j in range(self.n_arms):
            r_true = self.transfer_counts[j]
            rewards[j] = raw_proxy[j] * w1 + r_true * w2
        return rewards

    def select_source(self, similarity_matrix):
        rewards = self.compute_reward(similarity_matrix)
        return self.mab.select_arm(rewards)

    def update_transfer(self, arm, improved):
        self.transfer_counts *= self.gamma
        if improved:
            self.transfer_counts[arm] += 1.0


# =============================================================================
# KL Divergence Similarity Metric (adapted from SaEF-AKT)
# =============================================================================

def kl_divergence(data_0, data_1):
    """Compute KL divergence between two multivariate normal distributions
    fitted to the data.

    Parameters
    ----------
    data_0 : np.ndarray, shape (n0, d)
        Data from distribution 0.
    data_1 : np.ndarray, shape (n1, d)
        Data from distribution 1.

    Returns
    -------
    float
        KL(N_0 || N_1). Returns 1e10 on numerical failure.
    """
    k = data_0.shape[1]
    m0 = np.mean(data_0, axis=0)
    m1 = np.mean(data_1, axis=0)
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


def compute_kl_similarity(decs, dims, active_tasks, nt):
    """Compute pairwise task similarity matrix via KL divergence.

    For each pair (i, j), fit multivariate normals to decision variable data,
    compute KL divergence, convert to similarity via 1/(1+KLD) ∈ [0, 1].

    Parameters
    ----------
    decs : list of np.ndarray
        Decision variables per task. decs[i] has shape (n_i, dims[i]).
    dims : list of int
        Dimensions per task.
    active_tasks : list of int
        Indices of active tasks.
    nt : int
        Total number of tasks.

    Returns
    -------
    eta : np.ndarray, shape (nt, nt)
        Similarity matrix. eta[i, j] ∈ [0, 1], higher = more similar.
        Self-entries eta[i, i] = 0 (self-arm handled separately in MAB-2).
    """
    eta = np.zeros((nt, nt))
    for r in active_tasks:
        for s in active_tasks:
            if r == s:
                continue
            d_min = min(dims[r], dims[s])
            data_r = decs[r][:, :d_min]
            data_s = decs[s][:, :d_min]
            kld = kl_divergence(data_r, data_s)
            eta[r, s] = 1.0 / (1.0 + kld)

    return eta


# =============================================================================
# CMA-ES Surrogate Optimizer
# =============================================================================

def optimize_surrogate_cmaes(surrogate, dim, pop_size=50, w_max=200):
    """Optimize surrogate with CMA-ES, return best solution.

    Parameters
    ----------
    surrogate : SurrogateModel
        Fitted surrogate model with predict() method.
    dim : int
        Problem dimension.
    pop_size : int
        CMA-ES population size (default: 50).
    w_max : int
        Maximum number of CMA-ES generations (default: 200).

    Returns
    -------
    best_x : np.ndarray, shape (dim,)
        Best solution found.
    """
    def surrogate_func(x):
        pred, _ = surrogate.predict(x)
        return pred.reshape(-1, 1)

    surrogate_problem = MTOP()
    surrogate_problem.add_task(surrogate_func, dim=dim)
    cmaes = CMA_ES(surrogate_problem, n=pop_size, max_nfes=pop_size * w_max,
                    save_data=False, disable_tqdm=True)
    results = cmaes.optimize()
    return np.asarray(results.best_decs[0]).flatten()


# =============================================================================
# DFMAB-MTO Algorithm
# =============================================================================

class DFMAB_MTO:
    """
    Dual-Feedback Multi-Armed Bandit for Expensive Multi-Task Optimization.

    Two-layer MAB framework embedded within a surrogate-assisted optimization loop:
    - MAB-1: Adaptive surrogate model selection (GP, RBF, PR, KNN)
    - MAB-2: Adaptive knowledge transfer control (self vs. source tasks)

    Both MABs receive dual feedback signals (immediate proxy + delayed true reward)
    and use Discounted UCB (D-UCB) for arm selection.

    Parameters
    ----------
    problem : MTOP
        Multi-task optimization problem instance.
    n_initial : int or List[int], optional
        Number of initial LHS samples per task (default: 50).
    max_nfes : int or List[int], optional
        Maximum function evaluations per task (default: 200).
    gamma : float, optional
        Discount factor for D-UCB and delayed signals (default: 0.95).
    surrogate_pool : list of str, optional
        Surrogate model names from {'GP', 'RBF', 'PR', 'KNN'}
        (default: ['GP', 'RBF', 'PR', 'KNN']).
    cv_folds : int, optional
        Number of folds for cross-validation in MAB-1 immediate signal (default: 5).
    cmaes_pop_size : int, optional
        Population size for CMA-ES surrogate optimization (default: 50).
    cmaes_w_max : int, optional
        Max generations for CMA-ES surrogate optimization (default: 200).
    gen_gap : int, optional
        Iterations between MAB-1 model re-selections (default: 5).
    save_data : bool, optional
        Whether to save results (default: True).
    save_path : str, optional
        Path to save results (default: './Data').
    name : str, optional
        Algorithm name (default: 'DFMAB-MTO').
    disable_tqdm : bool, optional
        Whether to disable progress bar (default: True).
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
                 gamma=0.95, surrogate_pool=None, cv_folds=5,
                 cmaes_pop_size=50, cmaes_w_max=200, gen_gap=5,
                 save_data=True, save_path='./Data', name='DFMAB-MTO',
                 disable_tqdm=True):
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.gamma = gamma
        self.surrogate_pool = surrogate_pool if surrogate_pool is not None else ['GP', 'RBF', 'PR', 'KNN']
        self.cv_folds = cv_folds
        self.cmaes_pop_size = cmaes_pop_size
        self.cmaes_w_max = cmaes_w_max
        self.gen_gap = gen_gap
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def _create_mab(self, n_arms):
        """Create a Discounted UCB MAB instance."""
        return DiscountedUCB(n_arms, gamma=self.gamma)

    def optimize(self):
        """
        Execute the DFMAB-MTO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime.
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        nfes_per_task = n_initial_per_task.copy()
        model_names = self.surrogate_pool
        n_models = len(model_names)

        # --- Initialization ---
        decs = initialization(problem, self.n_initial, method='lhs', the_same=True)
        objs, _ = evaluation(problem, decs)

        # Initialize MAB-1 (model selection) per task
        mab1_list = []
        for i in range(nt):
            mab = self._create_mab(n_models)
            mab1_list.append(DualFeedbackMAB1(model_names, mab, gamma=self.gamma))

        # Initialize MAB-2 (transfer control) per task
        mab2_list = []
        for i in range(nt):
            mab = self._create_mab(nt)
            mab2_list.append(DualFeedbackMAB2(nt, i, mab, gamma=self.gamma))

        # Track current surrogate model index per task
        current_model_idx = [0] * nt
        current_surrogates = [None] * nt
        iteration = 0

        # --- Monitoring Data ---
        monitor = {
            'model_selection': [[] for _ in range(nt)],   # MAB-1: selected model idx per iter
            'cv_errors': [[] for _ in range(nt)],          # MAB-1: CV-RMSE per iter
            'transfer_source': [[] for _ in range(nt)],    # MAB-2: selected source per iter
            'improved': [[] for _ in range(nt)],            # Whether improvement occurred
            'best_obj': [[] for _ in range(nt)],            # Best objective per iter
            'similarity': [],                               # KL similarity matrices
            'mab1_rewards': [[] for _ in range(nt)],        # MAB-1 fused rewards
            'mab2_rewards': [[] for _ in range(nt)],        # MAB-2 fused rewards
            'model_names': model_names,
            'n_tasks': nt,
        }

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # --- Main Loop ---
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            iteration += 1

            # Step 1 & 2: MAB-1 selects surrogate and builds it
            best_solutions = [None] * nt
            for i in active_tasks:
                # MAB-1 model selection (every gen_gap iterations or first iteration)
                if iteration == 1 or iteration % self.gen_gap == 0:
                    current_model_idx[i] = mab1_list[i].select_model()

                model_name = model_names[current_model_idx[i]]
                surrogate = SURROGATE_POOL[model_name]()

                try:
                    surrogate.fit(decs[i], objs[i])
                    current_surrogates[i] = surrogate

                    # Step 3: Update MAB-1 immediate signal (CV error)
                    cv_err = surrogate.cv_rmse(decs[i], objs[i], n_folds=self.cv_folds)
                    mab1_list[i].update_cv(current_model_idx[i], cv_err)
                except Exception:
                    # Fallback: keep previous surrogate or use RBF
                    if current_surrogates[i] is None:
                        fallback = RBFSurrogate()
                        fallback.fit(decs[i], objs[i])
                        current_surrogates[i] = fallback
                    cv_err = 1e6
                    mab1_list[i].update_cv(current_model_idx[i], cv_err)

                # Step 4: Generate self-candidate via CMA-ES minimizing surrogate
                best_solutions[i] = optimize_surrogate_cmaes(
                    current_surrogates[i], dims[i],
                    pop_size=self.cmaes_pop_size, w_max=self.cmaes_w_max
                )

            # Compute KL divergence similarity matrix for MAB-2
            similarity_matrix = compute_kl_similarity(decs, dims, active_tasks, nt)
            monitor['similarity'].append(similarity_matrix.copy())

            # Step 5 & 6: MAB-2 selects transfer source for each task
            for i in active_tasks:
                if nfes_per_task[i] >= max_nfes_per_task[i]:
                    continue

                best_f_i = np.min(objs[i])

                # Build candidate list: arm j = candidate from task j
                candidates = [None] * nt
                for j in range(nt):
                    if best_solutions[j] is not None:
                        candidates[j] = align_dimensions(best_solutions[j], dims[i], fill='zero')

                # Record MAB-1 rewards before selection
                mab1_rewards = mab1_list[i].compute_reward()
                monitor['mab1_rewards'][i].append(mab1_rewards.copy())

                # MAB-2 selects source via KL divergence similarity
                mab2_rewards = mab2_list[i].compute_reward(similarity_matrix)
                monitor['mab2_rewards'][i].append(mab2_rewards.copy())
                selected_source = mab2_list[i].select_source(similarity_matrix)

                # Get the candidate to evaluate
                candidate = candidates[selected_source]
                if candidate is None:
                    candidate = candidates[i]
                candidate = np.clip(candidate, 0, 1).reshape(1, -1)

                # Remove duplicate
                if is_duplicate(candidate, decs[i]):
                    candidate = candidate + np.random.randn(*candidate.shape) * 0.01
                    candidate = np.clip(candidate, 0, 1)

                # Step 4: Evaluate selected candidate (true expensive evaluation)
                new_obj, _ = evaluation_single(problem, candidate, i)

                # Check if improvement occurred
                improved = new_obj.flatten()[0] < best_f_i

                # Step 5: Update MAB-1 delayed signal
                mab1_list[i].update_improvement(current_model_idx[i], improved)

                # Step 6: Update MAB-2 delayed signal
                mab2_list[i].update_transfer(selected_source, improved)

                # Update MAB internal state
                reward_val = 1.0 if improved else 0.0
                mab1_list[i].mab.update(current_model_idx[i], reward_val)
                mab2_list[i].mab.update(selected_source, reward_val)

                # Record monitoring data
                monitor['model_selection'][i].append(current_model_idx[i])
                monitor['cv_errors'][i].append(cv_err)
                monitor['transfer_source'][i].append(selected_source)
                monitor['improved'][i].append(improved)
                monitor['best_obj'][i].append(min(best_f_i, new_obj.flatten()[0]))

                # Add new data to database
                decs[i] = np.vstack([decs[i], candidate])
                objs[i] = np.vstack([objs[i], new_obj])

                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Build staircase history and save results
        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data,
            monitor=monitor
        )
        self.monitor = monitor

        return results

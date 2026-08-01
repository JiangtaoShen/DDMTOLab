"""
Surrogate-Assisted EA with Model and Infill Criterion Auto-Configuration (AutoSAEA)

This module implements AutoSAEA for expensive single-objective optimization problems.

References
----------
    [1] Xie, L., Li, G., Wang, Z., Cui, L., & Gong, M. (2023). Surrogate-assisted evolutionary algorithm with model and infill criterion auto-configuration. IEEE Transactions on Evolutionary Computation.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.19
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBF_kernel, ConstantKernel as C, WhiteKernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class AutoSAEA:
    """
    Surrogate-Assisted EA with Model and Infill Criterion Auto-Configuration.

    Uses a Two-Level UCB multi-armed bandit to adaptively select from 8
    model-criterion combinations:
    - {RBF, prescreening}, {RBF, local search}
    - {GP, LCB}, {GP, EI}
    - {PRS, prescreening}, {PRS, local search}
    - {KNN, exploitation}, {KNN, exploration}
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

    # Arm indices
    ARM_RBF_PRE = 0
    ARM_GP_LCB = 1
    ARM_RBF_LS = 2
    ARM_GP_EI = 3
    ARM_PRS_PRE = 4
    ARM_PRS_LS = 5
    ARM_KNN_EOI = 6
    ARM_KNN_EOR = 7
    N_ARMS = 8

    # High-level model mapping
    ARM_TO_MODEL = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
    MODEL_TO_ARMS = {0: [0, 2], 1: [1, 3], 2: [4, 5], 3: [6, 7]}

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, save_data=True,
                 save_path='./Data', name='AutoSAEA', disable_tqdm=True):
        """
        Initialize AutoSAEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'AutoSAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the AutoSAEA algorithm.

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

        # Generate initial samples
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        current_decs = [decs[i].copy() for i in range(nt)]
        current_objs = [objs[i].copy() for i in range(nt)]

        # Initialize bandit state per task
        bandit_states = []
        for _ in range(nt):
            bandit_states.append({
                'model_rewards': {m: [] for m in range(4)},
                'arm_rewards': {a: [] for a in range(self.N_ARMS)}
            })

        popsize = 50

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                X = current_decs[i]
                Y = current_objs[i]
                Y_flat = Y.flatten()

                # Iteration id (1-indexed)
                id_val = nfes_per_task[i] - n_initial_per_task[i] + 1

                # Elite initialization (top popsize by fitness)
                n_elite = min(len(Y_flat), popsize)
                elite_idx = np.argsort(Y_flat)[:n_elite]
                pop = X[elite_idx].copy()
                pop_objs = Y_flat[elite_idx].copy()

                # Generate shared DE/best/1 offspring
                offspring = self._de_best1(pop, pop_objs)

                # Select arm
                bs = bandit_states[i]
                if id_val <= self.N_ARMS:
                    arm_id = id_val - 1  # Round-robin
                else:
                    arm_id = self._select_arm(id_val, bs)

                # Execute arm
                candidate = self._execute_arm(arm_id, pop, pop_objs, offspring, Y_flat, dim)

                # Handle failure (GP arms may fail)
                arm_failed = candidate is None
                if arm_failed:
                    candidate = np.random.rand(1, dim)

                # Check duplicate BEFORE perturbation (MATLAB: reward=0 for near-duplicates)
                is_duplicate = False
                if not arm_failed:
                    dup_dist = cdist(candidate, X, metric='chebyshev').min()
                    if dup_dist < 5e-3:
                        is_duplicate = True

                # Ensure uniqueness (perturb for evaluation)
                candidate = self._ensure_uniqueness(candidate, X, dim)

                # Evaluate on real function
                obj, _ = evaluation_single(problem, candidate, i)

                # Compute reward (duplicate or failed arms get reward=0)
                if arm_failed or is_duplicate:
                    reward = 0.0
                else:
                    reward = self._low_level_r(pop_objs, obj.flatten()[0])

                # Update bandit
                model_id = self.ARM_TO_MODEL[arm_id]
                bs['model_rewards'][model_id].append(reward)
                bs['arm_rewards'][arm_id].append(reward)

                # Update dataset
                current_decs[i] = np.vstack([current_decs[i], candidate])
                current_objs[i] = np.vstack([current_objs[i], obj])
                nfes_per_task[i] += 1
                pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [current_decs[i].copy() for i in range(nt)]
        db_objs = [current_objs[i].copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data
        )
        return results

    # ==================== Bandit ====================

    def _tl_ucb(self, rewards, id_val, q_value, alpha=2.5):
        """Two-Level UCB value computation."""
        return q_value + np.sqrt(alpha * np.log(id_val) / len(rewards))

    def _select_arm(self, id_val, bandit_state):
        """Select arm via Two-Level UCB at both high and low levels."""
        mr = bandit_state['model_rewards']
        ar = bandit_state['arm_rewards']

        # High-level: select model
        u_model = np.zeros(4)
        for m in range(4):
            u_model[m] = self._tl_ucb(mr[m], id_val, np.mean(mr[m]))

        max_val = np.max(u_model)
        best_models = np.where(np.abs(u_model - max_val) < 1e-12)[0]
        selected_model = np.random.choice(best_models)

        # Low-level: select arm within model
        arms = self.MODEL_TO_ARMS[selected_model]
        u_arms = np.zeros(2)
        for j, arm_id in enumerate(arms):
            u_arms[j] = self._tl_ucb(ar[arm_id], id_val, np.mean(ar[arm_id]))

        max_val = np.max(u_arms)
        best_idx = np.where(np.abs(u_arms - max_val) < 1e-12)[0]
        selected_local = np.random.choice(best_idx)

        return arms[selected_local]

    def _low_level_r(self, Y_elite, y_new):
        """Rank-based reward: [0, 1], higher is better."""
        N = len(Y_elite)
        Y_all = np.concatenate([Y_elite.flatten(), [y_new]])
        sorted_idx = np.argsort(Y_all, kind='stable')
        position = np.where(sorted_idx == N)[0][0] + 1
        return -position / N + (N + 1) / N

    # ==================== Arm Dispatch ====================

    def _execute_arm(self, arm_id, pop, pop_objs, offspring, Y_full, dim):
        """Execute the selected arm and return candidate solution (or None on failure)."""
        if arm_id == self.ARM_RBF_PRE:
            return self._rbf_pre_arm(pop, pop_objs, offspring)
        elif arm_id == self.ARM_GP_LCB:
            return self._gp_lcb_arm(pop, pop_objs, offspring)
        elif arm_id == self.ARM_RBF_LS:
            return self._rbf_ls_arm(pop, pop_objs, dim)
        elif arm_id == self.ARM_GP_EI:
            return self._gp_ei_arm(pop, pop_objs, offspring, Y_full)
        elif arm_id == self.ARM_PRS_PRE:
            return self._prs_pre_arm(pop, pop_objs, offspring)
        elif arm_id == self.ARM_PRS_LS:
            return self._prs_ls_arm(pop, pop_objs, dim)
        elif arm_id == self.ARM_KNN_EOI:
            return self._knn_eoi_arm(pop, pop_objs, offspring)
        elif arm_id == self.ARM_KNN_EOR:
            return self._knn_eor_arm(pop, pop_objs, offspring)
        return None

    # ==================== Arm Implementations ====================

    def _rbf_pre_arm(self, pop, pop_objs, offspring):
        """Arm 0: {RBF, prescreening} - evaluate offspring on RBF, return best."""
        try:
            model = self._build_rbf_model(pop, pop_objs)
            pred = model(offspring).flatten()
            best_idx = np.argmin(pred)
            return offspring[best_idx:best_idx + 1]
        except Exception:
            return None

    def _gp_lcb_arm(self, pop, pop_objs, offspring):
        """Arm 1: {GP, LCB} - evaluate offspring with LCB (w=2), return best."""
        try:
            mean, std = self._build_gpr_predict(pop, pop_objs, offspring)
            lcb = mean - 2.0 * std
            best_idx = np.argmin(lcb)
            return offspring[best_idx:best_idx + 1]
        except Exception:
            return None

    def _rbf_ls_arm(self, pop, pop_objs, dim):
        """Arm 2: {RBF, local search} - EA optimization on RBF within local bounds."""
        try:
            model = self._build_rbf_model(pop, pop_objs)
            lb_local, ub_local = self._get_local_bounds(pop)
            return self._local_search_ea(model, dim, lb_local, ub_local)
        except Exception:
            return None

    def _gp_ei_arm(self, pop, pop_objs, offspring, Y_full):
        """Arm 3: {GP, EI} - Expected Improvement on GP, return best."""
        try:
            mean, std = self._build_gpr_predict(pop, pop_objs, offspring)
            y_best = np.min(Y_full)
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (y_best - mean) / (std + 1e-10)
                ei = (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
                ei[std < 1e-10] = 0.0
            best_idx = np.argmax(ei)
            return offspring[best_idx:best_idx + 1]
        except Exception:
            return None

    def _prs_pre_arm(self, pop, pop_objs, offspring):
        """Arm 4: {PRS, prescreening} - evaluate offspring on PRS, return best."""
        try:
            model = self._build_prs_model(pop, pop_objs)
            pred = model(offspring).flatten()
            best_idx = np.argmin(pred)
            return offspring[best_idx:best_idx + 1]
        except Exception:
            return None

    def _prs_ls_arm(self, pop, pop_objs, dim):
        """Arm 5: {PRS, local search} - EA optimization on PRS within local bounds."""
        try:
            model = self._build_prs_model(pop, pop_objs)
            lb_local, ub_local = self._get_local_bounds(pop)
            return self._local_search_ea(model, dim, lb_local, ub_local)
        except Exception:
            return None

    def _knn_eoi_arm(self, pop, pop_objs, offspring):
        """Arm 6: {KNN, L1-exploitation} - minimize max-distance to level-1 parents."""
        try:
            return self._knn_arm_core(pop, pop_objs, offspring, exploitation=True)
        except Exception:
            return None

    def _knn_eor_arm(self, pop, pop_objs, offspring):
        """Arm 7: {KNN, L1-exploration} - maximize min-distance to level-1 parents."""
        try:
            return self._knn_arm_core(pop, pop_objs, offspring, exploitation=False)
        except Exception:
            return None

    def _knn_arm_core(self, pop, pop_objs, offspring, exploitation=True):
        """Core KNN arm with 5-level stratification."""
        n = len(pop_objs)
        level = 5

        # Sort by objective (pop is already sorted from elite init, but be explicit)
        sorted_idx = np.argsort(pop_objs)
        X_sorted = pop[sorted_idx]

        # Assign labels: 1=best, 5=worst
        labels = np.ceil(np.arange(1, n + 1) * level / n).astype(int)
        parents_level1 = X_sorted[labels == 1]

        # Train 1-NN classifier
        knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski', p=2)
        knn.fit(X_sorted, labels)

        # Predict labels for offspring
        pred_labels = knn.predict(offspring)

        # Select offspring with best predicted label
        min_label = np.min(pred_labels)
        selected = np.where(pred_labels == min_label)[0]

        # Compute distances to level-1 parents
        dist = cdist(offspring[selected], parents_level1)

        if exploitation:
            dist_metric = np.max(dist, axis=1)
            best_local = np.argmin(dist_metric)
        else:
            dist_metric = np.min(dist, axis=1)
            best_local = np.argmax(dist_metric)

        idx = selected[best_local]
        return offspring[idx:idx + 1]

    # ==================== Surrogate Models ====================

    def _build_rbf_model(self, X, Y):
        """Build Gaussian RBF surrogate model (MATLAB newrbe equivalent)."""
        predict = newrbe_surrogate(X, Y)

        def model(x):
            return np.asarray(predict(x)).reshape(-1, 1)

        return model

    def _build_gpr_predict(self, X, Y, X_pred):
        """Build GP model and return (mean, std) predictions at X_pred."""
        Y_flat = Y.flatten()
        kernel = C(1.0, (1e-3, 1e3)) * RBF_kernel(1.0, (1e-2, 1e2)) + WhiteKernel(1e-5, (1e-10, 1e-1))
        gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=5, random_state=42
        )
        gpr.fit(X, Y_flat)
        mean, std = gpr.predict(X_pred, return_std=True)
        return mean.flatten(), std.flatten()

    def _build_prs_model(self, X, Y):
        """Build Pure Quadratic Regression Surrogate (no interaction terms)."""
        Y_flat = Y.flatten()
        X_aug = np.hstack([X, X ** 2])
        reg = LinearRegression()
        reg.fit(X_aug, Y_flat)

        def model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_aug = np.hstack([x, x ** 2])
            return reg.predict(x_aug).reshape(-1, 1)

        return model

    # ==================== EA Local Search ====================

    def _local_search_ea(self, surrogate_func, dim, lb, ub, popsize=50, n_gen=100):
        """EA-based local search on surrogate within local bounds."""
        popsize = popsize if popsize % 2 == 0 else popsize + 1

        # Latin hypercube initialization within local bounds (MATLAB pop_ini 'random')
        pop = lb + (ub - lb) * qmc.LatinHypercube(d=dim).random(popsize)
        pop_objs = surrogate_func(pop).flatten()

        for _ in range(n_gen):
            offspring = self._ea_offspring(pop, lb, ub)
            off_objs = surrogate_func(offspring).flatten()
            pop, pop_objs = self._roulette_wheel_selection(
                pop, pop_objs, offspring, off_objs, popsize)

        best_idx = np.argmin(pop_objs)
        return pop[best_idx:best_idx + 1]

    def _ea_offspring(self, parents, lb, ub, muc=15, mum=15, probswap=0.5):
        """EA offspring: SBX crossover + polynomial mutation + variable swap."""
        popsize, dim = parents.shape
        range_vec = ub - lb
        range_vec = np.where(range_vec < 1e-10, 1e-10, range_vec)

        # Normalize to [0,1] within local bounds
        pop_norm = (parents - lb) / range_vec
        offspring = np.zeros((popsize, dim))
        ind_order = np.random.permutation(popsize)

        for i in range(popsize // 2):
            p1 = ind_order[i]
            p2 = ind_order[i + popsize // 2]

            # SBX crossover
            u = np.random.rand(dim)
            cf = np.where(u <= 0.5,
                          (2 * u) ** (1 / (muc + 1)),
                          (2 * (1 - u)) ** (-1 / (muc + 1)))

            child1 = np.clip(0.5 * ((1 + cf) * pop_norm[p1] + (1 - cf) * pop_norm[p2]), 0, 1)
            child2 = np.clip(0.5 * ((1 + cf) * pop_norm[p2] + (1 - cf) * pop_norm[p1]), 0, 1)

            # Polynomial mutation
            for child in [child1, child2]:
                for j in range(dim):
                    if np.random.rand() < 1 / dim:
                        u_val = np.random.rand()
                        if u_val <= 0.5:
                            delta = (2 * u_val) ** (1 / (1 + mum)) - 1
                            child[j] += delta * child[j]
                        else:
                            delta = 1 - (2 * (1 - u_val)) ** (1 / (1 + mum))
                            child[j] += delta * (1 - child[j])
                child[:] = np.clip(child, 0, 1)

            # Variable swap
            swap = np.random.rand(dim) >= probswap
            temp = child2[swap].copy()
            child2[swap] = child1[swap]
            child1[swap] = temp

            # Denormalize back to local bounds
            offspring[i] = lb + child1 * range_vec
            offspring[i + popsize // 2] = lb + child2 * range_vec

        return offspring

    def _roulette_wheel_selection(self, pop, pop_objs, offspring, off_objs, popsize):
        """Roulette wheel selection (inverse fitness, matching MATLAB)."""
        total_pop = np.vstack([pop, offspring])
        total_objs = np.concatenate([pop_objs, off_objs])

        shift = min(np.min(total_objs), 0)
        fit = 1.0 / (total_objs - shift + 1e-6)
        cum_fit = np.cumsum(fit)
        cum_fit /= cum_fit[-1]

        idx = np.searchsorted(cum_fit, np.random.rand(popsize))
        idx = np.clip(idx, 0, len(total_objs) - 1)

        return total_pop[idx].copy(), total_objs[idx].copy()

    # ==================== DE Offspring ====================

    def _de_best1(self, parents, objs, F=0.5, CR=0.8):
        """DE/best/1/bin offspring generation in [0,1] space."""
        popsize, dim = parents.shape
        best_idx = np.argmin(objs)
        best = parents[best_idx]

        offspring = parents.copy()
        for i in range(popsize):
            idxs = np.delete(np.arange(popsize), i)
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            mutant = best + F * (parents[r1] - parents[r2])
            j_rand = np.random.randint(dim)
            mask = (np.random.rand(dim) <= CR) | (np.arange(dim) == j_rand)
            offspring[i, mask] = mutant[mask]

        return np.clip(offspring, 0, 1)

    # ==================== Utilities ====================

    def _get_local_bounds(self, pop):
        """Compute local bounds from population with degenerate dimension handling."""
        lb_local = np.min(pop, axis=0)
        ub_local = np.max(pop, axis=0)
        mask = (ub_local - lb_local) < 1e-10
        lb_local[mask] = np.maximum(0.0, lb_local[mask] - 0.05)
        ub_local[mask] = np.minimum(1.0, ub_local[mask] + 0.05)
        return lb_local, ub_local

    def _ensure_uniqueness(self, candidate, X, dim, epsilon=5e-3, n_scales=50, max_trials=1000):
        """Perturb the candidate until it is not too close to existing samples."""
        scales = np.linspace(0.1, 1.0, n_scales)
        c = 0
        while cdist(candidate, X, metric='chebyshev').min() < epsilon and c < max_trials:
            perturbation = scales[c % n_scales] * (np.random.rand(1, dim) - 0.5)
            candidate = np.clip(candidate + perturbation, 0.0, 1.0)
            c += 1
        return candidate

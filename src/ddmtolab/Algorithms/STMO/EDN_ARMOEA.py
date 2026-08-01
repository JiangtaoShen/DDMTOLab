"""
Efficient Dropout Neural Network based AR-MOEA (EDN-ARMOEA)

This module implements EDN-ARMOEA for computationally expensive multi/many-objective optimization.
It uses a dropout neural network as surrogate model with MC dropout for uncertainty estimation,
combined with an adaptive reference point based evolutionary algorithm (AR-MOEA).

References
----------
    [1] D. Guo, X. Wang, K. Gao, Y. Jin, J. Ding, and T. Chai. Evolutionary optimization of high-dimensional multiobjective and many-objective expensive problems assisted by a dropout neural network. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(4): 2084-2097.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
import warnings

warnings.filterwarnings("ignore")


class EDN_ARMOEA:
    """
    Efficient Dropout Neural Network based AR-MOEA for expensive
    multi/many-objective optimization.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, delta=0.05, wmax=20, ke=3,
                 save_data=True, save_path='./Data', name='EDN-ARMOEA', disable_tqdm=True):
        """
        Initialize EDN-ARMOEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size per task (default: 100)
        delta : float, optional
            Threshold for judging diversity improvement (default: 0.05)
        wmax : int, optional
            Number of generations before updating models (default: 20)
        ke : int, optional
            Number of solutions to be re-evaluated in each iteration (default: 3)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EDN-ARMOEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.delta = delta
        self.wmax = wmax
        self.ke = ke
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EDN-ARMOEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        n_initial_per_task = par_list(
            self.n_initial if self.n_initial is not None else [11 * d - 1 for d in dims], nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Per-task state: nets, ratio_old and the size of the last infill batch
        nets = [None] * nt
        ratio_olds = [None] * nt
        n_new_last = [0] * nt

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                M = n_objs[i]
                D = dims[i]
                popsize = n_per_task[i]

                # Generate reference vectors
                W, _ = uniform_point(popsize, M)

                # Prepare training data: SelectTrainData(A, 11*D-1, length(New))
                if n_new_last[i] > 0:
                    tr_x, tr_y = _select_train_data(decs[i], objs[i], 11 * D - 1,
                                                    n_new_last[i])
                else:
                    tr_x, tr_y = decs[i].copy(), objs[i].copy()

                # Normalize training data to [-1, 1]
                tr_xx, ps = _mapminmax(tr_x)
                tr_yy, qs = _mapminmax(tr_y)

                # MATLAB trains the net once before the loop (80000 steps) and then
                # calls updatemodel (8000 steps) at the top of every iteration
                if nets[i] is None:
                    nets[i] = _train_model(tr_xx, tr_yy, D, M, n_iter=80000)
                _update_model(nets[i], tr_xx, tr_yy, D, n_iter=8000)

                # Generate random population and estimate
                pop_dec = np.random.rand(popsize, D)
                pop_obj, pop_mse = _estimate(pop_dec, nets[i], ps, qs, M)

                # UpdateRefPoint
                archive_obj, ref_point, range_val, ratio = _update_ref_point(pop_obj, W, None)
                if ratio_olds[i] is None:
                    ratio_olds[i] = ratio

                # Inner evolution loop
                for w in range(self.wmax - 1):
                    mating_pool = _mating_selection(pop_obj, ref_point, range_val)
                    offspring_dec = ga_generation(pop_dec[mating_pool], muc=20, mum=20)
                    offspring_obj, offspring_mse = _estimate(offspring_dec, nets[i], ps, qs, M)

                    archive_obj, ref_point, range_val, ratio = _update_ref_point(
                        np.vstack([archive_obj, offspring_obj]), W, range_val)

                    mediate_dec = np.vstack([pop_dec, offspring_dec])
                    mediate_obj = np.vstack([pop_obj, offspring_obj])
                    mediate_mse = np.vstack([pop_mse, offspring_mse])

                    indices, range_val = _environmental_selection(
                        mediate_obj, ref_point, range_val, popsize)

                    pop_dec = mediate_dec[indices]
                    pop_obj = mediate_obj[indices]
                    pop_mse = mediate_mse[indices]

                # Select individuals for expensive evaluation
                flag = int(ratio_olds[i] - ratio < self.delta)
                pop_new = _individual_select(pop_dec, pop_obj, pop_mse, self.ke, flag)
                ratio_olds[i] = ratio

                # Drop points already evaluated; PlatEMO does not deduplicate, so keep
                # the raw candidates when the filter would empty the batch
                filtered = remove_duplicates(pop_new, decs[i])
                if filtered.shape[0] > 0:
                    pop_new = filtered

                if pop_new.shape[0] > 0:
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if pop_new.shape[0] > remaining:
                        pop_new = pop_new[:remaining]

                    new_objs, _ = evaluation_single(problem, pop_new, i)
                    decs[i] = np.vstack([decs[i], pop_new])
                    objs[i] = np.vstack([objs[i], new_objs])
                    nfes_per_task[i] += pop_new.shape[0]
                    n_new_last[i] = pop_new.shape[0]
                    pbar.update(pop_new.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.ke)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# Dropout Neural Network
# =============================================================================

class _DropoutNet:
    """Manual dropout neural network matching MATLAB implementation.

    Architecture: Input -> Dropout(0.2) -> FC(D,40) -> ReLU -> Dropout(0.5)
                  -> FC(40,40) -> Tanh -> FC(40,M)

    Uses manual SGD with weight decay on weights only (not biases).
    """

    def __init__(self, D, M, neuron_n=40):
        self.drop_p = [0.2, 0.5]
        self.decay = 1e-5
        self.learn_r = 0.01

        # Initialize weights (LeCun init: N(0, sqrt(1/fan_in)))
        self.W1 = np.random.randn(D, neuron_n) * np.sqrt(1.0 / D)
        self.W2 = np.random.randn(neuron_n, neuron_n) * np.sqrt(1.0 / neuron_n)
        self.W3 = np.random.randn(neuron_n, M) * np.sqrt(1.0 / neuron_n)

        # Initialize biases
        self.B1 = np.ones((1, neuron_n)) * 0.1 * np.sqrt(1.0 / neuron_n)
        self.B2 = np.random.randn(1, neuron_n) * np.sqrt(1.0 / neuron_n)
        self.B3 = np.random.randn(1, M) * np.sqrt(1.0 / M)

    def train_step(self, x, y):
        """One training step: forward + backward + SGD update."""
        N = x.shape[0]
        dp = self.drop_p

        # Forward
        x1, mask1 = _dropout(x, dp[0])
        x2 = x1 @ self.W1 + self.B1
        x3 = np.maximum(0, x2)  # ReLU
        x4, mask4 = _dropout(x3, dp[1])
        x5 = x4 @ self.W2 + self.B2
        x6 = np.tanh(x5)
        x7 = x6 @ self.W3 + self.B3

        # Backward
        e = x7 - y
        dW3 = x6.T @ e
        dB3 = np.sum(e, axis=0, keepdims=True)
        dx6 = e @ self.W3.T

        dx5 = dx6 * (1 - x6 ** 2)  # tanh derivative
        dW2 = x4.T @ dx5
        dB2 = np.sum(dx5, axis=0, keepdims=True)
        dx4 = dx5 @ self.W2.T

        # Dropout backward: zero out dropped, scale kept
        dx3 = dx4 * mask4 / (1 - dp[1])
        dx2 = dx3 * (x3 > 0).astype(float)  # ReLU derivative

        dW1 = x1.T @ dx2
        dB1 = np.sum(dx2, axis=0, keepdims=True)

        # SGD update with weight decay on weights only
        lr = self.learn_r
        self.W1 -= (self.decay * self.W1 + dW1) / N * lr
        self.W2 -= (self.decay * self.W2 + dW2) / N * lr
        self.W3 -= (self.decay * self.W3 + dW3) / N * lr
        self.B1 -= dB1 / N * lr
        self.B2 -= dB2 / N * lr
        self.B3 -= dB3 / N * lr

    def forward_dropout(self, x):
        """Forward pass WITH dropout (for MC estimation)."""
        dp = self.drop_p
        x1, _ = _dropout(x, dp[0])
        x2 = x1 @ self.W1 + self.B1
        x3 = np.maximum(0, x2)
        x4, _ = _dropout(x3, dp[1])
        x5 = x4 @ self.W2 + self.B2
        x6 = np.tanh(x5)
        x7 = x6 @ self.W3 + self.B3
        return x7


def _dropout(x, p):
    """Inverted dropout: zero with prob p, scale by 1/(1-p)."""
    mask = (np.random.rand(*x.shape) >= p).astype(float)
    return x * mask / (1 - p), mask


# =============================================================================
# Model Training / Update / Estimation
# =============================================================================

def _mapminmax(data):
    """Normalize each feature to [-1, 1]. Returns normalized data and params."""
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1.0
    normalized = 2.0 * (data - mins) / ranges - 1.0
    return normalized, (mins, maxs, ranges)


def _mapminmax_apply(data, params):
    """Apply normalization using existing params."""
    mins, maxs, ranges = params
    return 2.0 * (data - mins) / ranges - 1.0


def _mapminmax_reverse(data, params):
    """Reverse normalization."""
    mins, maxs, ranges = params
    return (data + 1.0) * ranges / 2.0 + mins


def _train_model(tr_xx, tr_yy, D, M, n_iter=80000):
    """Train the dropout neural network from scratch."""
    net = _DropoutNet(D, M)
    N = tr_xx.shape[0]
    batch_size = D

    for j in range(n_iter):
        idx = np.random.randint(0, N, size=batch_size)
        x_batch = tr_xx[idx]
        y_batch = tr_yy[idx]
        net.train_step(x_batch, y_batch)

    return net


def _update_model(net, tr_xx, tr_yy, D, n_iter=8000):
    """Fine-tune the dropout neural network."""
    N = tr_xx.shape[0]
    batch_size = D

    for j in range(n_iter):
        idx = np.random.randint(0, N, size=batch_size)
        x_batch = tr_xx[idx]
        y_batch = tr_yy[idx]
        net.train_step(x_batch, y_batch)


def _estimate(pop_dec, net, ps, qs, M):
    """MC dropout estimation: 100 stochastic forward passes."""
    x = _mapminmax_apply(pop_dec, ps)
    N = x.shape[0]
    n_mc = 100

    sum_y = np.zeros((N, M))
    sum_ysq = np.zeros((N, M))

    for _ in range(n_mc):
        y_pred = net.forward_dropout(x)
        y_orig = _mapminmax_reverse(y_pred, qs)
        sum_y += y_orig
        sum_ysq += y_orig ** 2

    mu = sum_y / n_mc
    s2 = sum_ysq / n_mc
    std = np.sqrt(np.maximum(s2 - mu ** 2, 0))

    return mu, std


# =============================================================================
# AR-MOEA Selection Components
# =============================================================================

def _cal_distance(pop_obj, ref_point):
    """Calculate distance between each solution and each adjusted reference point."""
    N = pop_obj.shape[0]
    NR = ref_point.shape[0]

    # Handle zero-norm solutions
    norms = np.sqrt(np.sum(pop_obj ** 2, axis=1))
    zero_idx = norms == 0
    if np.any(zero_idx):
        pop_obj = pop_obj.copy()
        pop_obj[zero_idx] += 1e-6 * np.random.rand(np.sum(zero_idx), pop_obj.shape[1])

    # Cosine similarity
    norm_p = np.sqrt(np.sum(pop_obj ** 2, axis=1, keepdims=True))  # (N, 1)
    norm_r = np.sqrt(np.sum(ref_point ** 2, axis=1, keepdims=True))  # (NR, 1)

    cosine = 1 - cdist(pop_obj, ref_point, 'cosine')  # (N, NR)
    cosine = np.clip(cosine, -1, 1)

    # d1 and d2
    d1 = norm_p * cosine  # (N, NR)
    d2 = norm_p * np.sqrt(np.maximum(1 - cosine ** 2, 0))  # (N, NR)

    # Adjust reference points
    nearest = np.argmin(d2, axis=0)  # (NR,)
    scale = d1[nearest, np.arange(NR)] / norm_r.ravel()  # (NR,)
    adjusted_ref = ref_point * scale[:, np.newaxis]

    # Euclidean distance
    distance = cdist(pop_obj, adjusted_ref)
    return distance


def _update_ref_point(archive_obj, W, range_val):
    """Adaptive reference point management."""
    # Delete dominated solutions and duplicates
    if archive_obj.shape[0] > 0:
        front_no, _ = nd_sort(archive_obj, 1)
        nd_mask = front_no == 1
        archive_obj = archive_obj[nd_mask]
        # MATLAB unique(...,'rows') both deduplicates and sorts rows lexicographically
        if archive_obj.shape[0] > 1:
            archive_obj = np.unique(archive_obj, axis=0)

    NA = archive_obj.shape[0]
    NW = W.shape[0]

    # Update ideal point
    if range_val is not None:
        range_val = range_val.copy()
        range_val[0] = np.minimum(range_val[0], archive_obj.min(axis=0)) if NA > 0 else range_val[0]
    elif NA > 0:
        range_val = np.vstack([archive_obj.min(axis=0), archive_obj.max(axis=0)])

    if NA <= 1:
        return archive_obj, W.copy(), range_val, 0.0

    # Translate archive
    t_archive = archive_obj - range_val[0]

    # Scale weight vectors
    w_scaled = W * (range_val[1] - range_val[0])

    # Find contributing solutions and valid weight vectors
    distance = _cal_distance(t_archive, w_scaled)  # (NA, NW)
    nearest_p = np.argmin(distance, axis=0)  # (NW,) - nearest solution per ref point
    contributing_s = np.unique(nearest_p)

    nearest_w = np.argmin(distance, axis=1)  # (NA,) - nearest ref point per solution
    valid_w = np.unique(nearest_w[contributing_s])

    # Update archive: expand with diversity
    choose = np.zeros(NA, dtype=bool)
    choose[contributing_s] = True

    cosine = 1 - cdist(t_archive, t_archive, 'cosine')
    np.fill_diagonal(cosine, 0)

    target_size = min(3 * NW, NA)
    while np.sum(choose) < target_size:
        unselected = np.where(~choose)[0]
        if len(unselected) == 0:
            break
        max_cos = np.max(cosine[np.ix_(unselected, np.where(choose)[0])], axis=1)
        x = np.argmin(max_cos)
        choose[unselected[x]] = True

    archive_obj = archive_obj[choose]
    t_archive = t_archive[choose]

    # Update reference points
    ref_candidates = np.vstack([w_scaled[valid_w], t_archive])
    n_valid = len(valid_w)
    choose_ref = np.zeros(ref_candidates.shape[0], dtype=bool)
    choose_ref[:n_valid] = True

    cosine_ref = 1 - cdist(ref_candidates, ref_candidates, 'cosine')
    np.fill_diagonal(cosine_ref, 0)

    target_ref = min(NW, ref_candidates.shape[0])
    while np.sum(choose_ref) < target_ref:
        unselected = np.where(~choose_ref)[0]
        if len(unselected) == 0:
            break
        selected = np.where(choose_ref)[0]
        max_cos = np.max(cosine_ref[np.ix_(unselected, selected)], axis=1)
        x = np.argmin(max_cos)
        choose_ref[unselected[x]] = True

    ref_point = ref_candidates[choose_ref]
    ratio = len(valid_w) / NW

    return archive_obj, ref_point, range_val, ratio


def _environmental_selection(obj, ref_point, range_val, N):
    """AR-MOEA environmental selection: ND sort + last front selection."""
    n_pop = obj.shape[0]
    front_no, max_fno = nd_sort(obj, N)

    # Select all solutions in fronts < max front
    next_mask = front_no < max_fno
    # Select from last front
    last = np.where(front_no == max_fno)[0]
    remain_n = N - np.sum(next_mask)

    if remain_n > 0 and len(last) > 0:
        chosen_last = _last_selection(obj[last], ref_point, range_val, remain_n)
        next_mask[last[chosen_last]] = True

    indices = np.where(next_mask)[0]

    # Update range
    range_val = range_val.copy()
    range_val[1] = np.max(obj, axis=0)
    diff = range_val[1] - range_val[0]
    range_val[1, diff < 1e-6] = 1.0

    return indices, range_val


def _igd_ns_fitness(rank, dis, convergence, noncontributing, N):
    """
    IGD-NS metric contribution of every solution (shared by mating/environmental
    selection). Vectorized transcription of the per-solution MATLAB loop.

    Parameters
    ----------
    rank : np.ndarray
        Solution indices sorted per reference point, shape (n_rows, NR)
    dis : np.ndarray
        Corresponding sorted distances, shape (n_rows, NR)
    convergence : np.ndarray
        Minimum distance of each solution to any reference point, shape (N,)
    noncontributing : np.ndarray
        Boolean mask of solutions that are nearest to no reference point
    N : int
        Total number of solutions the indices refer to

    Returns
    -------
    metric : np.ndarray
        Metric value of each solution (inf for solutions that are not evaluated)
    METRIC : float
        Metric value of the whole set
    """
    METRIC = np.sum(dis[0, :]) + np.sum(convergence[noncontributing])
    metric = np.full(N, np.inf)

    # Noncontributing solutions
    metric[noncontributing] = METRIC - convergence[noncontributing]

    # Contributing solutions: aggregate the reference points each of them owns
    owners = rank[0, :]
    seconds = rank[1, :] if rank.shape[0] > 1 else rank[0, :]
    s0 = np.bincount(owners, weights=dis[0, :], minlength=N)
    s1 = np.bincount(owners, weights=dis[1, :] if dis.shape[0] > 1 else dis[0, :],
                     minlength=N)

    # Set (not multiset) of runners-up per owner, restricted to noncontributing
    second_mat = np.zeros((N, N), dtype=bool)
    second_mat[owners, seconds] = True
    second_mat &= noncontributing[np.newaxis, :]
    conv_sum = second_mat @ convergence

    contributing = np.zeros(N, dtype=bool)
    contributing[owners] = True
    metric[contributing] = (METRIC - s0[contributing] + s1[contributing]
                            - conv_sum[contributing])
    return metric, METRIC


def _last_selection(pop_obj, ref_point, range_val, K):
    """Select K solutions from the last front based on IGD-NS metric contribution."""
    N = pop_obj.shape[0]
    NR = ref_point.shape[0]

    # Translate by ideal point
    translated = pop_obj - range_val[0]
    distance = _cal_distance(translated, ref_point)  # (N, NR)
    convergence = np.min(distance, axis=1)  # (N,)

    # Sort distance per reference point (column-wise)
    rank = np.argsort(distance, axis=0)  # (N, NR)
    dis = np.sort(distance, axis=0)  # (N, NR)

    remain = np.ones(N, dtype=bool)

    while np.sum(remain) > K:
        # Identify noncontributing solutions
        noncontributing = remain.copy()
        noncontributing[rank[0, :]] = False

        metric, _ = _igd_ns_fitness(rank, dis, convergence, noncontributing, N)

        # Delete worst
        metric[~remain] = np.inf
        delete = int(np.argmin(metric))
        remain[delete] = False

        # Update dis and rank: remove the deleted solution from every column
        # (MATLAB reshapes in column-major order, hence the transposes)
        n_now = int(np.sum(remain))
        keep = (rank != delete).T
        dis = dis.T[keep].reshape(NR, n_now).T
        rank = rank.T[keep].reshape(NR, n_now).T

    return np.where(remain)[0]


def _mating_selection(obj, ref_point, range_val):
    """AR-MOEA mating selection based on IGD-NS fitness."""
    N = obj.shape[0]

    translated = obj - range_val[0]
    distance = _cal_distance(translated, ref_point)
    convergence = np.min(distance, axis=1)

    rank = np.argsort(distance, axis=0)
    dis = np.sort(distance, axis=0)

    noncontributing = np.ones(N, dtype=bool)
    noncontributing[rank[0, :]] = False

    fitness, _ = _igd_ns_fitness(rank, dis, convergence, noncontributing, N)

    # Binary tournament selection on -Fitness, i.e. keep the larger IGD-NS metric
    n_select = int(np.ceil(N / 2)) * 2
    if n_select < 2:
        n_select = 2
    pool = np.zeros(n_select, dtype=int)
    for k in range(n_select):
        i1, i2 = np.random.randint(0, N, size=2)
        pool[k] = i1 if fitness[i1] > fitness[i2] else i2  # Higher metric = better diversity
    return pool


# =============================================================================
# Individual Selection (Infill)
# =============================================================================

def _individual_select(pop_dec, pop_obj, pop_mse, ke, flag):
    """Select ke individuals using k-means clustering.

    flag=0: select by max uncertainty (exploration)
    flag=1: select by min convergence (exploitation)
    """
    from sklearn.cluster import KMeans

    N = pop_dec.shape[0]
    actual_ke = min(ke, N)

    try:
        kmeans = KMeans(n_clusters=actual_ke, n_init=10, random_state=None)
        labels = kmeans.fit_predict(pop_obj)
    except Exception:
        # Fallback: random selection
        idx = np.random.choice(N, size=actual_ke, replace=False)
        return pop_dec[idx]

    selected = []
    unique_labels = np.unique(labels)

    if flag == 0:
        # Max uncertainty per cluster
        uncertainty = np.mean(pop_mse, axis=1)
        for lbl in unique_labels:
            cluster_idx = np.where(labels == lbl)[0]
            best = cluster_idx[np.argmax(uncertainty[cluster_idx])]
            selected.append(best)
    else:
        # Min convergence per cluster
        convergence = np.sqrt(np.sum(pop_obj ** 2, axis=1))
        for lbl in unique_labels:
            cluster_idx = np.where(labels == lbl)[0]
            best = cluster_idx[np.argmin(convergence[cluster_idx])]
            selected.append(best)

    return pop_dec[np.array(selected)]


# =============================================================================
# Training Data Selection
# =============================================================================

def _select_train_data(all_decs, all_objs, N1, N2):
    """Select diverse training data.

    Prioritize recent N2 solutions, then greedily fill to N1 by max cosine diversity.
    """
    NA = all_objs.shape[0]
    N1 = min(N1, NA)
    N2 = min(N2, NA)

    # Translate objectives
    tr_y = all_objs - all_objs.min(axis=0)

    # Cosine similarity matrix
    norms = np.sqrt(np.sum(tr_y ** 2, axis=1, keepdims=True))
    norms[norms < 1e-10] = 1e-10
    tr_y_norm = tr_y / norms
    cosine = tr_y_norm @ tr_y_norm.T
    np.fill_diagonal(cosine, 0)

    # Start with recent N2 solutions (last N2 entries)
    choose = np.zeros(NA, dtype=bool)
    choose[NA - N2:] = True

    while np.sum(choose) < N1:
        unselected = np.where(~choose)[0]
        selected = np.where(choose)[0]
        if len(unselected) == 0:
            break
        max_cos = np.max(cosine[np.ix_(unselected, selected)], axis=1)
        x = np.argmin(max_cos)
        choose[unselected[x]] = True

    return all_decs[choose], tr_y[choose]

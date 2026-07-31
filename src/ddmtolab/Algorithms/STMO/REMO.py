"""
Expensive Multiobjective Optimization by Relation Learning and Prediction (REMO)

This module implements REMO for computationally expensive multi/many-objective
optimization. A three-hidden-layer pattern-recognition network learns the ternary
relation (better / equal / worse) between pairs of decision vectors, and the learnt
relation is used to rank surrogate offspring without any objective regression.

References
----------
    [1] H. Hao, A. Zhou, H. Qian, and H. Zhang. Expensive multiobjective optimization by relation learning and prediction. IEEE Transactions on Evolutionary Computation, 2022, 26(5): 1157-1170.

Notes
-----
Author: Haowei Guo
Email: ghw@mail.nwpu.edu.cn
Date: 2026.01.16
Version: 2.0
"""
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform, cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import (
    get_algorithm_information, par_list, initialization, evaluation,
    evaluation_single, build_staircase_history, build_save_results, nd_sort)
import warnings

warnings.filterwarnings("ignore")


class REMO:
    """
    Expensive Multiobjective Optimization by Relation Learning and Prediction.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, k=6, gmax=3000,
                 save_data=True, save_path='./Data', name='REMO', disable_tqdm=True):
        """
        Initialize REMO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1 if dim <= 10 else 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        n : int or List[int], optional
            Population size per task (MATLAB Problem.N, default: 100)
        k : int, optional
            Number of reference solutions (default: 6)
        gmax : int, optional
            Number of solutions evaluated by the relation model (default: 3000)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'REMO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.n = n
        self.k = k
        self.gmax = gmax
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the REMO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims

        # MATLAB: N = 11*D-1 for D <= 10, otherwise 100
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 if dims[i] <= 10 else 100
                                  for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # decs/objs double as the MATLAB Archive: every expensively evaluated solution
        # Population: the raw initial design first, then RefSelect(Archive, Problem.N)
        pop_decs_list = [d.copy() for d in decs]
        pop_objs_list = [o.copy() for o in objs]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                pop_decs = pop_decs_list[i]
                pop_objs = pop_objs_list[i]
                k = min(self.k, pop_decs.shape[0])

                # Reference solutions and PBI-based binary catalog
                ref_idx = _ref_select(pop_objs, k)
                ref_decs = pop_decs[ref_idx]
                ref_objs = pop_objs[ref_idx]
                catalog = _get_output_pbi(pop_objs, ref_objs)

                # Relation pairs and stratified train/test split
                xxs, yys = _get_relation_pairs(pop_decs, catalog)
                net, scaler = None, None
                if xxs.shape[0] > 0:
                    train_in, train_out, test_in, test_out = _data_process(xxs, yys)
                    if train_in.shape[0] > 0:
                        scaler = _MapMinMax().fit(train_in)
                        x_dim = train_in.shape[1]
                        net = _RelationNet(x_dim)
                        _train_relation_net(net, scaler.transform(train_in),
                                            _onehot_index(train_out))

                s_model = {'net': net, 'scaler': scaler,
                           'X': pop_decs, 'Y': catalog}

                # Relation-model-assisted selection
                next_decs = _r_surrogate_assisted_selection(
                    ref_decs, pop_decs, self.gmax, s_model)

                if next_decs is not None and next_decs.shape[0] > 0:
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if next_decs.shape[0] > remaining:
                        next_decs = next_decs[:remaining]

                    next_objs, _ = evaluation_single(problem, next_decs, i)

                    decs[i] = np.vstack([decs[i], next_decs])
                    objs[i] = np.vstack([objs[i], next_objs])

                    nfes_per_task[i] += next_decs.shape[0]
                    pbar.update(next_decs.shape[0])

                # Population = RefSelect(Archive, Problem.N)
                survivors = _ref_select(objs[i], n_per_task[i])
                pop_decs_list[i] = decs[i][survivors]
                pop_objs_list[i] = objs[i][survivors]

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# PlatEMO OperatorGA (local copy)
# =============================================================================

def _operator_ga(parent, pro_c=1.0, dis_c=15.0, pro_m=1.0, dis_m=5.0):
    """
    Exact port of PlatEMO ``OperatorGA`` for real variables on [0, 1].

    The shared ``ga_generation`` helper shuffles the parents before pairing and
    emits an extra child for odd population sizes. REMO depends on the PlatEMO
    pairing (first half x second half, i.e. the top-ranked candidates crossed
    with the reference solutions), so a local copy is used.

    Parameters
    ----------
    parent : np.ndarray
        Parent decision variables, shape (N, D)
    pro_c, dis_c : float
        Crossover probability and SBX distribution index
    pro_m, dis_m : float
        Expected number of mutated variables and PM distribution index

    Returns
    -------
    offspring : np.ndarray
        Offspring decision variables, shape (2*floor(N/2), D)
    """
    parent = np.asarray(parent, dtype=float)
    half = parent.shape[0] // 2
    if half == 0:
        return np.zeros((0, parent.shape[1]))

    p1 = parent[:half]
    p2 = parent[half:2 * half]
    N, D = p1.shape

    mu = np.random.rand(N, D)
    beta = np.zeros((N, D))
    low = mu <= 0.5
    beta[low] = (2 * mu[low]) ** (1.0 / (dis_c + 1))
    beta[~low] = (2 - 2 * mu[~low]) ** (-1.0 / (dis_c + 1))
    beta = beta * (-1.0) ** np.random.randint(0, 2, size=(N, D))
    beta[np.random.rand(N, D) < 0.5] = 1
    beta[np.repeat(np.random.rand(N, 1) > pro_c, D, axis=1)] = 1
    offspring = np.vstack([(p1 + p2) / 2 + beta * (p1 - p2) / 2,
                           (p1 + p2) / 2 - beta * (p1 - p2) / 2])

    n2 = offspring.shape[0]
    site = np.random.rand(n2, D) < pro_m / D
    mu = np.random.rand(n2, D)
    offspring = np.clip(offspring, 0.0, 1.0)

    temp = site & (mu <= 0.5)
    offspring[temp] = offspring[temp] + (
            (2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - offspring[temp]) ** (dis_m + 1))
            ** (1.0 / (dis_m + 1)) - 1)

    temp = site & (mu > 0.5)
    offspring[temp] = offspring[temp] + (
            1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * offspring[temp] ** (dis_m + 1))
            ** (1.0 / (dis_m + 1)))

    return offspring


# =============================================================================
# Reference Solution Selection (RefSelect / RSEA strategy)
# =============================================================================

def _ref_select(pop_objs, k):
    """
    Select k reference solutions with the RSEA radar-grid strategy.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objective values, shape (N, M)
    k : int
        Number of solutions to select

    Returns
    -------
    index : np.ndarray
        Indices of the selected solutions, shape (k,)
    """
    N = pop_objs.shape[0]
    k = min(k, N)

    front_no, max_fno = nd_sort(pop_objs, k)
    next_idx = np.where(front_no <= max_fno)[0]

    pmin = pop_objs.min(axis=0) + 1e-6
    pmax = pop_objs.max(axis=0)
    if np.all(pmax > pmin):
        norm_objs = (pop_objs - pmin) / (pmax - pmin)
    else:
        norm_objs = pop_objs.copy()

    is_chosen = np.isin(next_idx, np.where(front_no < max_fno)[0])
    div = int(np.ceil(np.sqrt(k)))
    choose = _last_selection(norm_objs[next_idx], is_chosen, div, k)
    return next_idx[choose]


def _last_selection(pop_obj, choose, div, k):
    """
    Radar-grid based selection balancing convergence and diversity.

    Parameters
    ----------
    pop_obj : np.ndarray
        Normalized objective values, shape (N, M)
    choose : np.ndarray
        Boolean mask of solutions already selected (fronts before the last one)
    div : int
        Number of radar-grid divisions
    k : int
        Total number of solutions to select

    Returns
    -------
    choose : np.ndarray
        Boolean mask of selected solutions
    """
    N, M = pop_obj.shape
    choose = np.asarray(choose).copy().astype(bool)

    # Extreme solution by the PBI perpendicular distance to the (1,...,1) direction
    ones_vec = np.ones((1, M))
    norm = np.sqrt(np.sum(pop_obj ** 2, axis=1))
    cosine = 1 - cdist(pop_obj, ones_vec, metric='cosine').flatten()
    pbi = norm * np.sqrt(np.clip(1 - cosine ** 2, 0, None))
    choose[np.argmin(pbi)] = True

    # Convergence
    con = np.sum(pop_obj, axis=1)
    max_con = np.max(con)
    if max_con > 0:
        con = con / max_con

    site, rloc = _radar_grid(pop_obj, div)
    rdis = squareform(pdist(rloc))
    np.fill_diagonal(rdis, np.inf)

    crowd_g = np.zeros(int(np.max(site)) + 1)
    if np.any(choose):
        uniq, counts = np.unique(site[choose], return_counts=True)
        crowd_g[uniq] = counts

    while np.sum(choose) < k:
        remain_s = np.where(~choose)[0]
        if remain_s.size == 0:
            break
        remain_g = np.unique(site[remain_s])
        best_g = remain_g[crowd_g[remain_g] == np.min(crowd_g[remain_g])]
        current = remain_s[np.isin(site[remain_s], best_g)]

        chosen_idx = np.where(choose)[0]
        if chosen_idx.size > 0:
            min_rdis = np.min(rdis[np.ix_(current, chosen_idx)], axis=1)
        else:
            min_rdis = np.zeros(current.size)
        fitness = 0.1 * M * con[current] - min_rdis
        best = int(np.argmin(fitness))

        choose[current[best]] = True
        crowd_g[site[current[best]]] += 1

    return choose


def _radar_grid(pop_obj, div):
    """
    Compute the radar coordinates and grid index of each solution.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    div : int
        Number of grid divisions

    Returns
    -------
    site : np.ndarray
        Grid index of each solution, shape (N,)
    rloc : np.ndarray
        Radar coordinates, shape (N, 2)
    """
    N, M = pop_obj.shape
    theta = np.linspace(0, 2 * np.pi * (M - 1) / M, M)
    row_sum = np.maximum(np.sum(pop_obj, axis=1), 1e-10)

    rloc = np.zeros((N, 2))
    rloc[:, 0] = np.sum(pop_obj * np.cos(theta), axis=1) / row_sum
    rloc[:, 1] = np.sum(pop_obj * np.sin(theta), axis=1) / row_sum
    rloc = (rloc + 1) / 2

    yl = np.min(rloc, axis=0)
    yu = np.max(rloc, axis=0)
    denom = yu - yl
    denom[denom < 1e-10] = 1.0
    nrloc = (rloc - yl) / denom

    gloc = np.floor(nrloc * div).astype(int)
    gloc = np.clip(gloc, 0, div - 1)

    _, inverse = np.unique(gloc, axis=0, return_inverse=True)
    return np.asarray(inverse).ravel(), rloc


# =============================================================================
# PBI-based Catalog (GetOutput_PBI)
# =============================================================================

def _split_data(pop_objs, ref_objs, delt):
    """
    Split the population with a penalty-based boundary intersection threshold.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objective values, shape (N, M)
    ref_objs : np.ndarray
        Reference objective values, shape (k, M)
    delt : float
        Penalty parameter of the PBI aggregation

    Returns
    -------
    output : np.ndarray
        Binary catalog, 1 for solutions inside the PBI boundary and 0 otherwise
    rate : float
        Fraction of solutions labelled 1
    """
    N = pop_objs.shape[0]
    output = np.ones(N, dtype=int)

    # Nearest reference direction by cosine similarity
    ref_index = np.argmin(cdist(pop_objs, ref_objs, metric='cosine'), axis=1)
    Z = np.min(pop_objs, axis=0)

    for j in range(ref_objs.shape[0]):
        sub = np.where(ref_index == j)[0]
        if sub.size == 0:
            continue
        bound = ref_objs[j]
        w = bound - Z
        norm_w = np.sqrt(np.sum(w ** 2))
        if norm_w < 1e-12:
            continue
        W = w / norm_w                                  # unit direction, ||W|| = 1
        sub_vec = pop_objs[sub] - Z
        norm_p = np.sqrt(np.sum(sub_vec ** 2, axis=1))
        norm_p = np.where(norm_p < 1e-12, 1e-12, norm_p)
        cosine_p = (sub_vec @ W) / norm_p - 1e-6
        g = norm_p * cosine_p + delt * norm_p * np.sqrt(
            np.clip(1 - cosine_p ** 2, 0, None))
        norm_r = np.sqrt(np.sum((bound - Z) ** 2))
        if norm_r < 1e-12:
            continue
        g = g / norm_r
        output[sub[g > 1]] = 0

    rate = float(np.sum(output == 1)) / N
    return output, rate


def _get_output_pbi(pop_objs, ref_objs):
    """
    Self-adaptive PBI catalog: bisect the penalty until 0.3 <= rate <= 0.7.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objective values, shape (N, M)
    ref_objs : np.ndarray
        Reference objective values, shape (k, M)

    Returns
    -------
    output : np.ndarray
        Binary catalog, shape (N,)
    """
    delt_l, delt_u = -20.0, 20.0
    rate = 0.0
    output = None

    while rate > 0.7 or rate < 0.3:
        delt_c = (delt_l + delt_u) / 2
        if abs(delt_l - delt_u) < 1e-1:
            break
        output, rate = _split_data(pop_objs, ref_objs, delt_c)
        if rate > 0.7:
            delt_l = delt_c
        elif rate < 0.3:
            delt_u = delt_c

    if output is None:
        output = np.ones(pop_objs.shape[0], dtype=int)
    return output


# =============================================================================
# Relation Pairs and Data Processing
# =============================================================================

def _combvec_pairs(a, b):
    """
    All ordered pairs [a_i, b_j] laid out in MATLAB ``combvec`` column order.

    Parameters
    ----------
    a, b : np.ndarray
        Decision matrices, shapes (n1, D) and (n2, D)

    Returns
    -------
    pairs : np.ndarray
        Concatenated pairs, shape (n1*n2, 2*D); the index of [a_i, b_j] is j*n1+i
    """
    n1, D = a.shape
    n2 = b.shape[0]
    if n1 == 0 or n2 == 0:
        return np.zeros((0, 2 * D))
    left = np.tile(a, (n2, 1))              # a varies fastest
    right = np.repeat(b, n1, axis=0)
    return np.hstack([left, right])


def _get_relation_pairs(decs, catalog):
    """
    Build the ternary relation training set from the catalog of the population.

    Pairs of two class-1 solutions or two class-0 solutions are labelled 0,
    (class 1, class 0) pairs are labelled 1 and (class 0, class 1) pairs are
    labelled -1. The intra-class blocks are subsampled to balance the set.

    Parameters
    ----------
    decs : np.ndarray
        Population decision variables, shape (N, D)
    catalog : np.ndarray
        Binary catalog from ``_get_output_pbi``, shape (N,)

    Returns
    -------
    xxs : np.ndarray
        Pair features, shape (P, 2*D)
    ls : np.ndarray
        Pair labels in {-1, 0, 1}, shape (P,)
    """
    D = decs.shape[1]
    c1 = decs[catalog == 1]
    c2 = decs[catalog != 1]
    n1, n2 = c1.shape[0], c2.shape[0]

    c1c1 = _combvec_pairs(c1, c1)
    c1c2 = _combvec_pairs(c1, c2)
    c2c1 = _combvec_pairs(c2, c1)
    c2c2 = _combvec_pairs(c2, c2)

    # Drop the self-pairs (index j*n+i with i == j)
    if n1 > 0:
        c1c1 = np.delete(c1c1, np.arange(n1) * (n1 + 1), axis=0)
    if n2 > 0:
        c2c2 = np.delete(c2c2, np.arange(n2) * (n2 + 1), axis=0)

    def _sample(arr, num):
        num = int(num)
        if num <= 0:
            return arr[:0]
        if arr.shape[0] > num:
            return arr[np.random.choice(arr.shape[0], num, replace=False)]
        return arr

    t_num = int(np.ceil(c1c2.shape[0] / 2))
    if c1c1.shape[0] > t_num and c2c2.shape[0] > t_num:
        c1c1 = _sample(c1c1, t_num)
        c2c2 = _sample(c2c2, t_num)
    elif c1c1.shape[0] < t_num:
        c2c2 = _sample(c2c2, t_num * 2 - c1c1.shape[0])
    elif c2c2.shape[0] < t_num:
        c1c1 = _sample(c1c1, t_num * 2 - c2c2.shape[0])

    blocks = [c1c1, c2c2, c1c2, c2c1]
    if all(b.shape[0] == 0 for b in blocks):
        return np.zeros((0, 2 * D)), np.zeros(0)

    xxs = np.vstack([b for b in blocks if b.shape[0] > 0])
    ls = np.concatenate([np.zeros(c1c1.shape[0]), np.zeros(c2c2.shape[0]),
                         np.ones(c1c2.shape[0]), -np.ones(c2c1.shape[0])])
    return xxs, ls


def _data_process(xxs, yys):
    """
    Class-stratified 3/4 train, 1/4 test split followed by a global shuffle.

    Parameters
    ----------
    xxs : np.ndarray
        Pair features, shape (P, 2*D)
    yys : np.ndarray
        Pair labels, shape (P,)

    Returns
    -------
    train_in, train_out, test_in, test_out : np.ndarray
        The shuffled split
    """
    pha = 0.75
    train_idx, test_idx = [], []
    for label in (0, 1, -1):
        idx = np.where(yys == label)[0]
        if idx.size == 0:
            continue
        n_sel = int(np.ceil(pha * idx.size))
        perm = np.random.permutation(idx.size)
        train_idx.append(idx[perm[:n_sel]])
        test_idx.append(idx[perm[n_sel:]])

    train_idx = np.concatenate(train_idx) if train_idx else np.zeros(0, dtype=int)
    test_idx = np.concatenate(test_idx) if test_idx else np.zeros(0, dtype=int)

    train_idx = train_idx[np.random.permutation(train_idx.size)]
    test_idx = test_idx[np.random.permutation(test_idx.size)]

    return xxs[train_idx], yys[train_idx], xxs[test_idx], yys[test_idx]


# =============================================================================
# Relation Model
# =============================================================================

class _MapMinMax:
    """Per-feature min-max scaling to [-1, 1] (MATLAB ``mapminmax``)."""

    def __init__(self):
        self.xmin = None
        self.xrange = None

    def fit(self, X):
        self.xmin = np.min(X, axis=0)
        xmax = np.max(X, axis=0)
        self.xrange = xmax - self.xmin
        self.xrange[self.xrange < 1e-12] = 1.0
        return self

    def transform(self, X):
        if self.xmin is None:
            return X
        return 2.0 * (X - self.xmin) / self.xrange - 1.0


def _onehot_index(labels):
    """Map labels {1, 0, -1} to class indices {0, 1, 2} (MATLAB ``onehotconv``)."""
    idx = np.ones(labels.shape[0], dtype=np.int64)
    idx[labels == 1] = 0
    idx[labels == -1] = 2
    return torch.tensor(idx, dtype=torch.long)


class _RelationNet(nn.Module):
    """
    Pattern-recognition network ``patternnet([ceil(1.5*xDim), xDim, ceil(xDim/2)])``.

    MATLAB's patternnet uses tansig hidden layers and a softmax output layer; the
    softmax is folded into ``CrossEntropyLoss`` during training and applied
    explicitly at prediction time.
    """

    def __init__(self, x_dim):
        super().__init__()
        h1 = int(np.ceil(x_dim * 1.5))
        h2 = int(x_dim)
        h3 = int(np.ceil(x_dim / 2))
        self.net = nn.Sequential(
            nn.Linear(x_dim, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, h3), nn.Tanh(),
            nn.Linear(h3, 3)
        )

    def forward(self, x):
        return self.net(x)


def _train_relation_net(net, X, y, max_epochs=1000, max_fail=6, lr=0.01):
    """
    Train the relation network, mirroring the patternnet training defaults.

    MATLAB's ``train`` uses trainscg with a random 70/15/15 split, cross-entropy
    performance, at most 1000 epochs and early stopping after 6 consecutive
    validation failures. The optimizer here is full-batch Adam; the schedule,
    split ratios and stopping rule follow the MATLAB defaults.

    Parameters
    ----------
    net : _RelationNet
        Network to train
    X : np.ndarray
        Normalized pair features, shape (n, 2*D)
    y : torch.Tensor
        Class indices, shape (n,)
    max_epochs : int
        Maximum number of epochs
    max_fail : int
        Number of consecutive validation failures before stopping
    lr : float
        Adam learning rate
    """
    n = X.shape[0]
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_t = y

    perm = np.random.permutation(n)
    n_val = max(1, int(round(0.15 * n))) if n >= 4 else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if tr_idx.size == 0:
        tr_idx, val_idx = perm, np.zeros(0, dtype=int)

    X_tr, y_tr = X_t[tr_idx], y_t[tr_idx]
    use_val = val_idx.size > 0
    if use_val:
        X_va, y_va = X_t[val_idx], y_t[val_idx]

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = np.inf
    best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
    fails = 0

    net.train()
    for _ in range(max_epochs):
        optimizer.zero_grad()
        loss = loss_fn(net(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        if use_val:
            with torch.no_grad():
                val = float(loss_fn(net(X_va), y_va))
            if val < best_val - 1e-6:
                best_val = val
                best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
                fails = 0
            else:
                fails += 1
                if fails >= max_fail:
                    break

    if use_val:
        net.load_state_dict(best_state)
    net.eval()


def _model_predict(s_model, pairs):
    """
    Softmax class probabilities of the relation model for a block of pairs.

    Parameters
    ----------
    s_model : dict
        Relation model with keys ``net`` and ``scaler``
    pairs : np.ndarray
        Pair features, shape (P, 2*D)

    Returns
    -------
    probs : np.ndarray
        Class probabilities, shape (P, 3) ordered as (label 1, label 0, label -1)
    """
    net, scaler = s_model['net'], s_model['scaler']
    if net is None:
        return np.full((pairs.shape[0], 3), 1.0 / 3.0)
    with torch.no_grad():
        x = torch.tensor(scaler.transform(pairs), dtype=torch.float32)
        return torch.softmax(net(x), dim=1).numpy()


def _model_select(s_model, candidates):
    """
    Score candidates by their predicted relation to the catalogued population.

    Parameters
    ----------
    s_model : dict
        Relation model with the population ``X`` and its catalog ``Y``
    candidates : np.ndarray
        Candidate decision variables, shape (n, D)

    Returns
    -------
    ind : np.ndarray
        Candidate indices in descending score order
    scores : np.ndarray
        Relation scores in [-4, 4], shape (n,)
    """
    X, Y = s_model['X'], s_model['Y']
    n_cand = candidates.shape[0]
    c1 = X[Y == 1]
    c2 = X[Y != 1]
    n_c1, n_c2 = c1.shape[0], c2.shape[0]

    if n_cand == 0 or (n_c1 == 0 and n_c2 == 0):
        return np.arange(n_cand), np.zeros(n_cand)

    block = 2 * (n_c1 + n_c2)
    all_pairs = np.zeros((block * n_cand, 2 * X.shape[1]))
    for i in range(n_cand):
        o = i * block
        xi = np.tile(candidates[i], (max(n_c1, n_c2), 1))
        if n_c1 > 0:
            all_pairs[o:o + n_c1] = np.hstack([c1, xi[:n_c1]])
            all_pairs[o + n_c1:o + 2 * n_c1] = np.hstack([xi[:n_c1], c1])
        if n_c2 > 0:
            o2 = o + 2 * n_c1
            all_pairs[o2:o2 + n_c2] = np.hstack([c2, xi[:n_c2]])
            all_pairs[o2 + n_c2:o2 + 2 * n_c2] = np.hstack([xi[:n_c2], c2])

    probs = _model_predict(s_model, all_pairs)

    scores = np.zeros(n_cand)
    for i in range(n_cand):
        o = i * block
        s1 = s2 = 0.0
        if n_c1 > 0:
            p_c1xi = probs[o:o + n_c1].mean(axis=0)
            s1 += p_c1xi[1] + p_c1xi[2]
            s2 += p_c1xi[0]
            p_xic1 = probs[o + n_c1:o + 2 * n_c1].mean(axis=0)
            s1 += p_xic1[1] + p_xic1[0]
            s2 += p_xic1[2]
        if n_c2 > 0:
            o2 = o + 2 * n_c1
            p_c2xi = probs[o2:o2 + n_c2].mean(axis=0)
            s1 += p_c2xi[2]
            s2 += p_c2xi[1] + p_c2xi[0]
            p_xic2 = probs[o2 + n_c2:o2 + 2 * n_c2].mean(axis=0)
            s1 += p_xic2[0]
            s2 += p_xic2[1] + p_xic2[2]
        scores[i] = s1 - s2

    return np.argsort(-scores, kind='stable'), scores


def _r_surrogate_assisted_selection(ref_decs, pop_decs, wmax, s_model):
    """
    Relation-model-assisted search for the next expensive evaluations.

    Parameters
    ----------
    ref_decs : np.ndarray
        Reference solutions, shape (k, D)
    pop_decs : np.ndarray
        Population decision variables, shape (N, D)
    wmax : int
        Budget of relation-model evaluations
    s_model : dict
        Relation model

    Returns
    -------
    next_decs : np.ndarray
        Candidate solutions for expensive evaluation
    """
    next_decs = _operator_ga(np.vstack([pop_decs, ref_decs]))
    n_ref = ref_decs.shape[0]

    i = 0
    while i < wmax and next_decs.shape[0] > 0:
        sorted_idx, _ = _model_select(s_model, next_decs)
        parents = next_decs[sorted_idx[:min(n_ref, next_decs.shape[0])]]
        next_decs = _operator_ga(np.vstack([parents, ref_decs]))
        i += next_decs.shape[0]

    if next_decs.shape[0] == 0:
        return next_decs

    _, scores = _model_select(s_model, next_decs)
    good = scores > 3.9
    if np.sum(good) < 4:
        order = np.argsort(-scores, kind='stable')
        return next_decs[order[:min(4, next_decs.shape[0])]]
    return next_decs[good]

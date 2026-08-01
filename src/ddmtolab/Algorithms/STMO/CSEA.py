"""
Classification-based Surrogate-assisted Evolutionary Algorithm (CSEA)

This module implements CSEA for computationally expensive multi/many-objective optimization.
It uses a neural network classifier to distinguish promising solutions from non-promising ones
relative to reference solutions, with adaptive surrogate-assisted selection strategies.

References
----------
    [1] L. Pan, C. He, Y. Tian, H. Wang, X. Zhang, and Y. Jin. A classification based surrogate-assisted evolutionary algorithm for expensive many-objective optimization. IEEE Transactions on Evolutionary Computation, 2019, 23(1): 74-88.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.16
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import pdist, squareform, cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class CSEA:
    """
    Classification-based Surrogate-assisted Evolutionary Algorithm for expensive
    multi/many-objective optimization.

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
                 save_data=True, save_path='./Data', name='CSEA', disable_tqdm=True):
        """
        Initialize CSEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: min(11*dim-1, 109))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size per task (MATLAB Problem.N, default: 100)
        k : int, optional
            Number of reference solutions (default: 6)
        gmax : int, optional
            Number of solutions evaluated by surrogate model (default: 3000)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'CSEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.k = k
        self.gmax = gmax
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the CSEA algorithm.

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

        # Set default initial samples: min(11*dim - 1, 109)
        if self.n_initial is None:
            n_initial_per_task = [min(11 * dims[i] - 1, 109) for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Population size (MATLAB Problem.N)
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Archive: all evaluated solutions
        arc_decs = [d.copy() for d in decs]
        arc_objs = [o.copy() for o in objs]

        # MATLAB: Population is the raw initial design on the first iteration and
        # is refreshed with RefSelect(Arc, Problem.N) at the end of every iteration
        pop_decs_list = [d.copy() for d in decs]
        pop_objs_list = [o.copy() for o in objs]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                pop_decs = pop_decs_list[i]
                pop_objs = pop_objs_list[i]
                k = min(self.k, pop_decs.shape[0])

                # Select k reference solutions
                ref_decs, ref_objs = _ref_select(pop_decs, pop_objs, k)

                # Classify: 1 = not dominated by all ref points, 0 = dominated
                labels = _get_output(pop_objs, ref_objs)

                rr = np.sum(labels) / len(labels)
                tr = min(rr, 1 - rr) * 0.5

                # Stratified train/test split
                train_X, train_Y, test_X, test_Y = _data_process(pop_decs, labels)

                # Train neural network classifier
                net = _build_network(dim)
                _train_network(net, train_X, train_Y, epochs=100, batch_size=32)

                # Calculate error rates on test set
                p0, p1 = _calc_error_rates(net, test_X, test_Y)

                # Surrogate-assisted selection
                next_decs = _surrogate_assisted_selection(
                    net, p0, p1, ref_decs, pop_decs, self.gmax, tr
                )

                if next_decs is not None and len(next_decs) > 0:
                    # Drop points already evaluated; PlatEMO does not deduplicate, so
                    # fall back to the raw candidates when the filter empties the batch
                    filtered = remove_duplicates(next_decs, decs[i])
                    if filtered.shape[0] > 0:
                        next_decs = filtered

                    # Limit evaluations to remaining budget
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if next_decs.shape[0] > remaining:
                        next_decs = next_decs[:remaining]

                    # Evaluate with expensive function
                    next_objs, _ = evaluation_single(problem, next_decs, i)

                    # Update archive
                    arc_decs[i] = np.vstack([arc_decs[i], next_decs])
                    arc_objs[i] = np.vstack([arc_objs[i], next_objs])
                    decs[i] = np.vstack([decs[i], next_decs])
                    objs[i] = np.vstack([objs[i], next_objs])

                    nfes_per_task[i] += next_decs.shape[0]
                    pbar.update(next_decs.shape[0])

                # Population = RefSelect(Arc, Problem.N)
                pop_decs_list[i], pop_objs_list[i] = _ref_select(
                    arc_decs[i], arc_objs[i], n_per_task[i])

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [decs[i].copy() for i in range(nt)]
        db_objs = [objs[i].copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

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

    The shared ``ga_generation`` helper shuffles the parent matrix before pairing
    and emits one extra child for odd population sizes. CSEA relies on the
    deterministic PlatEMO pairing (first half x second half, i.e. promising
    solutions crossed with the reference solutions), so a local copy is used.

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

    # Simulated binary crossover
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

    # Polynomial mutation with lower = 0, upper = 1 (clip before mutating)
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
# Neural Network Classifier
# =============================================================================

class _CSEANet(nn.Module):
    """Neural network classifier for CSEA."""

    def __init__(self, input_dim):
        super().__init__()
        hidden_size = int(np.ceil(input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def _build_network(input_dim):
    """Build CSEA classifier network."""
    return _CSEANet(input_dim)


def _train_network(net, train_X, train_Y, epochs=100, batch_size=32):
    """
    Train the classifier with ADAM optimizer and MSE loss (regression layer in MATLAB).

    Parameters
    ----------
    net : _CSEANet
        Neural network model
    train_X : np.ndarray
        Training inputs, shape (n, d)
    train_Y : np.ndarray
        Training labels, shape (n,)
    epochs : int
        Number of training epochs
    batch_size : int
        Mini-batch size
    """
    device = torch.device('cpu')
    net = net.to(device)

    # Z-score normalization (matching MATLAB featureInputLayer 'zscore')
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std < 1e-10] = 1.0
    train_X_norm = (train_X - mean) / std

    # Store normalization parameters
    net._input_mean = mean
    net._input_std = std

    # BatchNorm needs at least two samples per batch
    if train_X.shape[0] < 2:
        return

    net.train()

    X_tensor = torch.tensor(train_X_norm, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(train_Y, dtype=torch.float32, device=device).unsqueeze(1)

    dataset = TensorDataset(X_tensor, Y_tensor)
    # BatchNorm1d cannot process a singleton batch; drop it when the tail would be 1
    n_train = X_tensor.shape[0]
    drop_last = n_train > batch_size and n_train % batch_size == 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            pred = net(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()


def _predict(net, X):
    """
    Predict labels using the trained classifier.

    Parameters
    ----------
    net : _CSEANet
        Trained neural network
    X : np.ndarray
        Input data, shape (n, d)

    Returns
    -------
    predictions : np.ndarray
        Predicted labels, shape (n,)
    """
    device = torch.device('cpu')
    net.eval()

    # Apply same normalization as training
    X_norm = (X - net._input_mean) / net._input_std

    with torch.no_grad():
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
        pred = net(X_tensor).cpu().numpy().flatten()

    return pred


def _calc_error_rates(net, test_X, test_Y):
    """
    Calculate classification error rates for class 0 and class 1.

    Parameters
    ----------
    net : _CSEANet
        Trained neural network
    test_X : np.ndarray
        Test inputs, shape (n, d)
    test_Y : np.ndarray
        Test labels, shape (n,)

    Returns
    -------
    p0 : float
        Error rate for class 1 (good solutions misclassified)
    p1 : float
        Error rate for class 0 (bad solutions misclassified)
    """
    test_pred = _predict(net, test_X)

    # p0: error for class 1 (IndexGood = TestOut==1)
    good_mask = test_Y == 1
    if np.sum(good_mask) > 0:
        p0 = np.sum(np.abs(test_Y[good_mask] - test_pred[good_mask])) / np.sum(good_mask)
    else:
        p0 = 0.0

    # p1: error for class 0 (IndexGood = ~IndexGood)
    bad_mask = ~good_mask
    if np.sum(bad_mask) > 0:
        p1 = np.sum(np.abs(test_Y[bad_mask] - test_pred[bad_mask])) / np.sum(bad_mask)
    else:
        p1 = 0.0

    return p0, p1


# =============================================================================
# Reference Solution Selection (RefSelect)
# =============================================================================

def _ref_select(pop_decs, pop_objs, k):
    """
    Select k reference solutions using RSEA strategy.

    Parameters
    ----------
    pop_decs : np.ndarray
        Decision variables, shape (N, D)
    pop_objs : np.ndarray
        Objective values, shape (N, M)
    k : int
        Number of reference solutions to select

    Returns
    -------
    ref_decs : np.ndarray
        Selected reference decision variables
    ref_objs : np.ndarray
        Selected reference objective values
    """
    N = pop_objs.shape[0]
    k = min(k, N)

    # Non-dominated sorting to select candidate set
    front_no, max_fno = nd_sort(pop_objs, k)
    next_idx = np.where(front_no <= max_fno)[0]

    # Normalize objectives
    pmin = pop_objs.min(axis=0) + 1e-6
    pmax = pop_objs.max(axis=0)
    if np.all(pmax > pmin):
        norm_objs = (pop_objs - pmin) / (pmax - pmin)
    else:
        norm_objs = pop_objs.copy()

    # Environmental selection
    is_chosen = np.isin(next_idx, np.where(front_no < max_fno)[0])
    div = int(np.ceil(np.sqrt(k)))
    choose = _last_selection(norm_objs[next_idx], is_chosen, div, k)

    selected = next_idx[choose]
    return pop_decs[selected], pop_objs[selected]


def _last_selection(pop_obj, choose, div, k):
    """
    Select solutions based on radar grid for convergence-diversity balance.

    Parameters
    ----------
    pop_obj : np.ndarray
        Normalized objective values, shape (N, M)
    choose : np.ndarray
        Boolean mask of already chosen solutions
    div : int
        Grid divisions
    k : int
        Total number to select

    Returns
    -------
    choose : np.ndarray
        Boolean mask of selected solutions
    """
    N, M = pop_obj.shape
    choose = choose.copy().astype(bool)

    # Identify extreme solutions using PBI metric
    ones_vec = np.ones((1, M))
    # PBI: perpendicular distance from solution to weight vector (1,...,1)
    norm = np.sqrt(np.sum(pop_obj ** 2, axis=1))
    cosine = 1 - cdist(pop_obj, ones_vec, metric='cosine').flatten()
    pbi = norm * np.sqrt(np.clip(1 - cosine ** 2, 0, None))
    extreme = np.argmin(pbi)
    choose[extreme] = True

    # Calculate convergence (sum of objectives)
    con = np.sum(pop_obj, axis=1)
    max_con = np.max(con)
    if max_con > 0:
        con = con / max_con

    # Calculate radar grid
    site, rloc = _radar_grid(pop_obj, div)
    rdis = squareform(pdist(rloc))
    np.fill_diagonal(rdis, np.inf)

    # Grid crowding
    max_site = np.max(site) + 1
    crowd_g = np.zeros(max_site)
    if np.any(choose):
        chosen_sites = site[choose]
        unique_sites, counts = np.unique(chosen_sites, return_counts=True)
        crowd_g[unique_sites] = counts

    # Select k solutions
    while np.sum(choose) < k:
        remain_s = np.where(~choose)[0]
        if len(remain_s) == 0:
            break
        remain_g = np.unique(site[remain_s])
        min_crowd = np.min(crowd_g[remain_g])
        best_g = remain_g[crowd_g[remain_g] == min_crowd]
        current = remain_s[np.isin(site[remain_s], best_g)]

        # Fitness: balance convergence and diversity
        chosen_idx = np.where(choose)[0]
        if len(chosen_idx) > 0:
            min_rdis = np.min(rdis[np.ix_(current, chosen_idx)], axis=1)
        else:
            min_rdis = np.zeros(len(current))
        fitness = 0.1 * M * con[current] - min_rdis
        best = np.argmin(fitness)

        choose[current[best]] = True
        crowd_g[site[current[best]]] += 1

    return choose


# =============================================================================
# Radar Grid
# =============================================================================

def _radar_grid(pop_obj, div):
    """
    Calculate radar grid index of each solution.

    Parameters
    ----------
    pop_obj : np.ndarray
        Objective values, shape (N, M)
    div : int
        Number of grid divisions

    Returns
    -------
    site : np.ndarray
        Grid site index for each solution, shape (N,)
    rloc : np.ndarray
        Radar coordinates, shape (N, 2)
    """
    N, M = pop_obj.shape

    # Calculate radar coordinates
    theta = np.linspace(0, 2 * np.pi * (M - 1) / M, M)
    row_sum = np.sum(pop_obj, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-10)

    rloc = np.zeros((N, 2))
    rloc[:, 0] = np.sum(pop_obj * np.cos(theta), axis=1) / row_sum.flatten()
    rloc[:, 1] = np.sum(pop_obj * np.sin(theta), axis=1) / row_sum.flatten()
    rloc = (rloc + 1) / 2

    # Normalize radar locations
    yl = np.min(rloc, axis=0)
    yu = np.max(rloc, axis=0)
    denom = yu - yl
    denom[denom < 1e-10] = 1.0
    nrloc = (rloc - yl) / denom

    # Grid location
    gloc = np.floor(nrloc * div).astype(int)
    gloc = np.clip(gloc, 0, div - 1)

    # Map grid locations to unique site indices (sortrows(unique(GLoc,'rows')) in MATLAB)
    unique_gloc, inverse = np.unique(gloc, axis=0, return_inverse=True)
    site = np.asarray(inverse).ravel()

    return site, rloc


# =============================================================================
# Classification Output
# =============================================================================

def _get_output(pop_obj, ref_obj):
    """
    Classify solutions: 1 if not dominated by all reference points, 0 otherwise.

    A solution gets label 1 if for EVERY reference point, the solution is better
    in at least one objective.

    Parameters
    ----------
    pop_obj : np.ndarray
        Population objectives, shape (N, M)
    ref_obj : np.ndarray
        Reference objectives, shape (k, M)

    Returns
    -------
    output : np.ndarray
        Binary labels, shape (N,)
    """
    N = pop_obj.shape[0]
    output = np.ones(N, dtype=float)

    for j in range(ref_obj.shape[0]):
        # For each ref point, check if solution has at least one objective <= ref
        output = output * np.any(pop_obj <= ref_obj[j], axis=1).astype(float)

    return output


# =============================================================================
# Data Processing
# =============================================================================

def _data_process(inputs, outputs):
    """
    Stratified train/test split (75/25) preserving class balance.

    Parameters
    ----------
    inputs : np.ndarray
        Input features, shape (N, D)
    outputs : np.ndarray
        Binary labels, shape (N,)

    Returns
    -------
    train_in, train_out, test_in, test_out : np.ndarray
        Split data
    """
    idx1 = np.where(outputs > 0.5)[0]
    idx0 = np.where(outputs <= 0.5)[0]

    # Select 75% of each class for training
    n_train1 = int(np.ceil(0.75 * len(idx1)))
    n_train0 = int(np.ceil(0.75 * len(idx0)))

    perm1 = np.random.permutation(len(idx1))
    perm0 = np.random.permutation(len(idx0))

    train_idx1 = idx1[perm1[:n_train1]]
    train_idx0 = idx0[perm0[:n_train0]]
    train_idx = np.concatenate([train_idx1, train_idx0])

    test_idx = np.setdiff1d(np.arange(len(outputs)), train_idx)

    train_in = inputs[train_idx]
    train_out = outputs[train_idx]
    test_in = inputs[test_idx]
    test_out = outputs[test_idx]

    return train_in, train_out, test_in, test_out


# =============================================================================
# Surrogate-Assisted Selection
# =============================================================================

def _surrogate_assisted_selection(net, p0, p1, ref_decs, pop_decs, gmax, tr):
    """
    Select promising solutions using surrogate predictions with adaptive strategies.

    Parameters
    ----------
    net : _CSEANet
        Trained classifier
    p0 : float
        Error rate for class 1
    p1 : float
        Error rate for class 0
    ref_decs : np.ndarray
        Reference solution decisions, shape (k, D)
    pop_decs : np.ndarray
        Population decisions, shape (N, D)
    gmax : int
        Maximum surrogate evaluations
    tr : float
        Classification threshold

    Returns
    -------
    next_decs : np.ndarray or None
        Selected solutions for expensive evaluation
    """
    a = tr
    b = 1 - tr

    # Generate initial candidates via GA operators (OperatorGA with {1,15,1,5})
    combined = np.vstack([pop_decs, ref_decs])
    next_decs = _operator_ga(combined, pro_c=1.0, dis_c=15.0, pro_m=1.0, dis_m=5.0)
    label = _predict(net, next_decs)

    i = 0
    n_ref = ref_decs.shape[0]

    if p0 < 0.4 or (p1 < a and p0 < b):
        # Strategy 1: Focus on high-prediction solutions (classifier reliable for good class)
        while i < gmax:
            sorted_idx = np.argsort(label)[::-1]  # descending
            input_decs = next_decs[sorted_idx[:n_ref]]
            combined = np.vstack([input_decs, ref_decs])
            next_decs = _operator_ga(combined, pro_c=1.0, dis_c=15.0, pro_m=1.0, dis_m=5.0)
            label = _predict(net, next_decs)
            i += next_decs.shape[0]
        # Select solutions predicted as good
        selected = next_decs[label > 0.9]
        # PlatEMO can return an empty batch here, which stalls the outer loop;
        # fall back to the single most promising candidate
        next_decs = selected if selected.shape[0] > 0 else next_decs[[np.argmax(label)]]

    elif p0 > b and p1 < a:
        # Strategy 2: Classifier unreliable, random selection
        rand_idx = np.random.randint(0, len(next_decs))
        next_decs = next_decs[rand_idx:rand_idx + 1]

    elif p1 > b:
        # Strategy 3: Focus on low-prediction solutions (classifier reverses labels)
        while i < gmax:
            sorted_idx = np.argsort(label)  # ascending
            input_decs = next_decs[sorted_idx[:n_ref]]
            combined = np.vstack([input_decs, ref_decs])
            next_decs = _operator_ga(combined, pro_c=1.0, dis_c=15.0, pro_m=1.0, dis_m=5.0)
            label = _predict(net, next_decs)
            i += next_decs.shape[0]
        # Select solutions predicted as bad (but likely good due to reversed classifier)
        selected = next_decs[label < 0.1]
        next_decs = selected if selected.shape[0] > 0 else next_decs[[np.argmin(label)]]

    else:
        # Fallback: random selection
        rand_idx = np.random.randint(0, len(next_decs))
        next_decs = next_decs[rand_idx:rand_idx + 1]

    if len(next_decs) == 0:
        return None

    return next_decs

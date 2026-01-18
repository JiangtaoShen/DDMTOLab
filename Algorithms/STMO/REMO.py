"""
Expensive Multiobjective Optimization by Relation Learning and Prediction (REMO)

This module implements the REMO algorithm. It utilizes a neural network to learn
and predict dominance relationships between candidate solutions and reference solutions,
guiding the evolutionary search process efficiently under limited evaluation budgets.

References
----------
    [1] H. Hao, A. Zhou, H. Qian, and H. Zhang. Expensive multiobjective
        optimization by relation learning and prediction. IEEE Transactions on
        Evolutionary Computation, 2022.

Notes
-----
Author: Haowei Guo
Email: ghw@mail.nwpu.edu.cn
Date: 2026.01.16
Version: 1.1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from tqdm import tqdm
import time

# [Platform Integration] Import ALL standard utils
from Methods.Algo_Methods.algo_utils import (
    get_algorithm_information, initialization, evaluation, evaluation_single,
    init_history, append_history, build_save_results,
    nd_sort, ga_generation
)

class REMO:
    """
    Expensive Multiobjective Optimization by Relation Learning and Prediction (REMO)

    Attributes
    ----------
    algorithm_information : dict
    Dictionary containing algorithm capabilities (e.g., supported objectives, constraints).
    """
    algorithm_information = {
        'n_tasks': '1-N',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=50, max_nfes=300, k=6, gmax=3000,
                 save_data=True, save_path=None, name='REMO', disable_tqdm=False, **kwargs):
        """
        Initialize the REMO algorithm parameters.

        Parameters
        ----------
        problem : MTOP
            The optimization problem instance.
        n : int or List[int]
            Population size (default: 50).
        max_nfes : int or List[int]
            Maximum number of function evaluations (default: 300).
        k : int
            Number of reference solutions used for relation learning (default: 6).
        gmax : int
            Maximum total steps for internal surrogate-assisted evolution (default: 3000).
        """
        self.problem = problem
        self.raw_n = n
        self.raw_max_nfes = max_nfes
        self.k = k
        self.gmax = gmax
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def optimize(self):
        """
        Execute the main optimization loop of REMO.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks

        # 1. Parameter Parsing
        n_per_task = []
        for t in range(nt):
            if isinstance(self.raw_n, int) and self.raw_n == 50:
                D = problem.n_vars[t]
                val = 11 * D - 1 if D <= 10 else 100
                n_per_task.append(val)
            else:
                val = self.raw_n if np.isscalar(self.raw_n) else self.raw_n[t]
                n_per_task.append(int(val))

        if np.isscalar(self.raw_max_nfes):
            max_nfes_per_task = [int(self.raw_max_nfes)] * nt
        else:
            max_nfes_per_task = [int(x) for x in self.raw_max_nfes]

        # 2. Initialization
        decs_list = initialization(problem, n=n_per_task, method='lhs')
        objs_list, cons_list = evaluation(problem, decs_list)

        for t in range(nt):
            target = n_per_task[t]
            current = decs_list[t].shape[0]
            if current > target:
                decs_list[t] = decs_list[t][:target]
                objs_list[t] = objs_list[t][:target]
                if cons_list[t] is not None:
                    cons_list[t] = cons_list[t][:target]

        normalized_cons = []
        for t in range(nt):
            if cons_list[t] is None:
                normalized_cons.append(np.zeros((len(objs_list[t]), 1)))
            else:
                normalized_cons.append(cons_list[t])

        all_decs, all_objs, all_cons = init_history(decs_list, objs_list, normalized_cons)

        pop_decs_list = [decs.copy() for decs in decs_list]
        pop_objs_list = [objs.copy() for objs in objs_list]
        pop_cons_list = [c.copy() for c in normalized_cons]

        nfes_per_task = [len(d) for d in pop_decs_list]
        total_max_nfes = sum(max_nfes_per_task)
        total_current_nfes = sum(nfes_per_task)
        pbar = tqdm(total=total_max_nfes, initial=total_current_nfes,
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # 3. Main Loop
        while sum(nfes_per_task) < total_max_nfes:
            active_tasks = [t for t in range(nt) if nfes_per_task[t] < max_nfes_per_task[t]]
            if not active_tasks:
                break

            for t in active_tasks:
                curr_pop_decs = pop_decs_list[t]
                curr_pop_objs = pop_objs_list[t]
                curr_pop_cons = pop_cons_list[t]
                n_target = n_per_task[t]

                # 3.1 Select Reference Solutions (Representative subset)
                ref_indices = ref_select(curr_pop_objs, curr_pop_cons, self.k)
                ref_decs = curr_pop_decs[ref_indices]
                ref_objs = curr_pop_objs[ref_indices]

                # 3.2 Data Preparation
                catalog = get_output_pbi(curr_pop_objs, ref_objs)
                xxs, yys = get_relation_pairs(curr_pop_decs, catalog)

                # 3.3 Model Training
                if len(xxs) > 0:
                    train_in, train_out, test_in, test_out = data_process(xxs, yys)

                    scaler = MapMinMax()
                    train_in_nor = scaler.fit_transform(train_in)
                    train_out_indices = onehot_encoding_indices(train_out)

                    x_dim = train_in.shape[1]
                    net = RelationNet(x_dim).to(self.device)
                    # Train model (even if labels are all 0, it mimics MATLAB behavior)
                    train_model(net, train_in_nor, train_out_indices, self.device)
                else:
                    # Fallback if absolutely no pairs generated (shouldn't happen with MATLAB logic)
                    scaler = MapMinMax()
                    x_dim = curr_pop_decs.shape[1] * 2
                    net = RelationNet(x_dim).to(self.device)

                s_model = {'scaler': scaler, 'net': net, 'device': self.device,
                           'X': curr_pop_decs, 'Y': catalog}

                # 3.4 Surrogate Assisted Selection (Internal Evolution)
                next_decs = r_surrogate_assisted_selection(
                    problem, ref_decs, curr_pop_decs, self.gmax, s_model, t
                )

                # 3.5 Real Evaluation & Environment Selection
                if next_decs is not None and len(next_decs) > 0:
                    new_objs, new_cons = evaluation_single(problem, next_decs, t)
                    if new_cons is None: new_cons = np.zeros((len(new_objs), 1))

                    combined_decs = np.vstack((curr_pop_decs, next_decs))
                    combined_objs = np.vstack((curr_pop_objs, new_objs))
                    combined_cons = np.vstack((curr_pop_cons, new_cons))

                    if len(combined_decs) > n_target:
                        survivor_indices = ref_select(combined_objs, combined_cons, n_target)

                        if len(survivor_indices) < n_target:
                            all_indices = np.arange(len(combined_decs))
                            remaining = np.setdiff1d(all_indices, survivor_indices)
                            needed = n_target - len(survivor_indices)

                            if len(remaining) >= needed:
                                additional = remaining[:needed]
                            else:
                                additional = np.random.choice(survivor_indices, needed, replace=True)
                            survivor_indices = np.concatenate((survivor_indices, additional))

                        if len(survivor_indices) > n_target:
                            survivor_indices = survivor_indices[:n_target]

                        curr_pop_decs = combined_decs[survivor_indices]
                        curr_pop_objs = combined_objs[survivor_indices]
                        curr_pop_cons = combined_cons[survivor_indices]
                    else:
                        curr_pop_decs, curr_pop_objs, curr_pop_cons = combined_decs, combined_objs, combined_cons

                    pop_decs_list[t] = curr_pop_decs
                    pop_objs_list[t] = curr_pop_objs
                    pop_cons_list[t] = curr_pop_cons

                    n_new = len(next_decs)
                    nfes_per_task[t] += n_new
                    pbar.update(n_new)

                    append_history(all_decs[t], pop_decs_list[t], all_objs[t], pop_objs_list[t], all_cons[t], pop_cons_list[t])

        pbar.close()
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, all_cons=all_cons,
            runtime=time.time() - start_time, max_nfes=nfes_per_task,
            bounds=problem.bounds, save_path=self.save_path, filename=self.name, save_data=self.save_data
        )
        return results

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_relation_pairs(decs, catalog):
    """
    Construct training pairs for the relation learning model.

    Data Balancing Strategy:
    - Calculates a target size based on cross-class pairs (C1-C2).
    - Performs lazy downsampling on intra-class pairs (C1-C1, C2-C2) only if
      they exceed the target size significantly.
    """
    c1 = decs[catalog == 1]
    c2 = decs[catalog != 1]

    def pairs(a, b):
        if len(a) == 0 or len(b) == 0: return np.zeros((0, decs.shape[1]*2))
        return np.hstack((np.repeat(a, len(b), axis=0), np.tile(b, (len(a), 1))))

    c1c1 = pairs(c1, c1)
    c2c2 = pairs(c2, c2)
    # Remove self-pairs
    if len(c1) > 0: c1c1 = c1c1[np.arange(len(c1c1)) % (len(c1)+1) != 0]
    if len(c2) > 0: c2c2 = c2c2[np.arange(len(c2c2)) % (len(c2)+1) != 0]

    c1c2 = pairs(c1, c2)
    c2c1 = pairs(c2, c1)

    target = int(np.ceil(len(c1c2) / 2))

    def sample_if_needed(arr, n):
        if n == 0: return arr
        if len(arr) == 0: return arr
        if len(arr) > n:
            idx = np.random.choice(len(arr), n, replace=False)
            return arr[idx]
        return arr

    if target > 0:
        if len(c1c1) > target and len(c2c2) > target:
            c1c1 = sample_if_needed(c1c1, target)
            c2c2 = sample_if_needed(c2c2, target)
        elif len(c1c1) < target and len(c2c2) > 0:
            needed = target * 2 - len(c1c1)
            c2c2 = sample_if_needed(c2c2, needed)
        elif len(c2c2) < target and len(c1c1) > 0:
            needed = target * 2 - len(c2c2)
            c1c1 = sample_if_needed(c1c1, needed)

    arrays_to_stack = [x for x in [c1c1, c2c2, c1c2, c2c1] if len(x) > 0]

    if not arrays_to_stack:
        return np.zeros((0, decs.shape[1]*2)), np.zeros(0)

    xxs = np.vstack(arrays_to_stack)

    l_c1c1 = np.zeros(len(c1c1))
    l_c2c2 = np.zeros(len(c2c2))
    l_c1c2 = np.ones(len(c1c2))
    l_c2c1 = -1 * np.ones(len(c2c1))
    yys = np.concatenate([l_c1c1, l_c2c2, l_c1c2, l_c2c1])

    return xxs, yys

def get_output_pbi(pop_objs, ref_objs):
    """
    Classify solutions using an adaptive Penalty-based Boundary Intersection (PBI) metric.

    The function iteratively adjusts the penalty parameter to maintain a reasonable
    ratio of 'Good' (1) to 'Bad' (0) solutions, typically between 0.3 and 0.7.
    """
    N = pop_objs.shape[0]
    output = np.ones(N, dtype=int)
    dists = cdist(pop_objs, ref_objs, metric='cosine')
    ref_idx = np.argmin(dists, axis=1)
    Z = np.min(pop_objs, axis=0)
    delt_l, delt_u = -20.0, 20.0

    for _ in range(20):
        delt_c = (delt_l + delt_u) / 2
        if abs(delt_l - delt_u) < 1e-1: break

        curr_output = np.ones(N, dtype=int)
        my_ref = ref_objs[ref_idx]
        w = my_ref - Z
        w_norm = np.linalg.norm(w, axis=1) + 1e-10
        W = w / w_norm[:, None]
        vec = pop_objs - Z
        d1 = np.abs(np.sum(vec * W, axis=1))
        d2 = np.linalg.norm(vec - d1[:, None] * W, axis=1)
        g = (d1 + delt_c * d2) / (np.linalg.norm(my_ref - Z, axis=1) + 1e-10)

        curr_output[g > 1] = 0
        ratio = np.sum(curr_output == 1) / N

        if ratio > 0.7: delt_l = delt_c
        elif ratio < 0.3: delt_u = delt_c
        else:
            output = curr_output
            break

    return output

def ref_select(pop_obj, pop_con, k):
    """
    Select representative solutions from the population using radar-grid diversity maintenance.
    """
    N = pop_obj.shape[0]
    k = min(k, N)
    front_no, max_f_no = nd_sort(pop_obj, k)
    next_indices = np.where(front_no <= max_f_no)[0]

    pre_selected_mask = front_no[next_indices] < max_f_no
    choose_mask = pre_selected_mask.copy()

    p_min = np.min(pop_obj, axis=0)
    p_max = np.max(pop_obj, axis=0)
    denom = p_max - p_min
    denom[denom == 0] = 1e-6
    norm_obj = (pop_obj[next_indices] - p_min) / denom

    if np.sum(choose_mask) == 0:
        ones_vec = np.ones((1, pop_obj.shape[1]))
        cosine = 1 - cdist(norm_obj, ones_vec, metric='cosine').flatten()
        sine = np.sqrt(1 - cosine**2)
        d2 = np.linalg.norm(norm_obj, axis=1) * sine
        choose_mask[np.argmin(d2)] = True

    sub_cons = pop_con[next_indices]
    choose_mask = last_selection_radar(norm_obj, sub_cons, choose_mask, int(np.ceil(np.sqrt(k))), k)
    return next_indices[choose_mask]

def last_selection_radar(pop_obj, pop_con, choose_mask, div, k):
    """
    Diversity selection helper based on radar grid mapping.
    """
    N, M = pop_obj.shape
    theta = np.linspace(0, 2*np.pi*(M-1)/M, M)
    sum_p = np.sum(pop_obj, axis=1, keepdims=True) + 1e-10
    r_loc = np.column_stack((
        np.sum(pop_obj * np.cos(theta), axis=1) / sum_p.flatten(),
        np.sum(pop_obj * np.sin(theta), axis=1) / sum_p.flatten()
    ))
    r_loc = (r_loc + 1) / 2
    gl_min = np.min(r_loc, axis=0)
    gl_max = np.max(r_loc, axis=0)

    diff = gl_max - gl_min
    diff[diff == 0] = 1.0
    g_loc = np.floor((r_loc - gl_min) / diff * div).astype(int)
    g_loc = np.clip(g_loc, 0, div-1)

    g_loc_view = np.ascontiguousarray(g_loc).view(np.dtype((np.void, g_loc.dtype.itemsize * g_loc.shape[1])))
    _, site = np.unique(g_loc_view, return_inverse=True)
    site = site.flatten()

    r_dis = cdist(r_loc, r_loc)
    np.fill_diagonal(r_dis, np.inf)

    crowd_g = np.bincount(site[choose_mask], minlength=np.max(site)+1)
    con_violation = np.sum(np.maximum(0, pop_con), axis=1)

    while np.sum(choose_mask) < k:
        remain = np.where(~choose_mask)[0]
        if len(remain) == 0: break

        remain_grids = site[remain]
        if len(remain_grids) == 0: break

        unique_remain_grids = np.unique(remain_grids)
        min_crowd = np.min(crowd_g[unique_remain_grids])
        best_grids = unique_remain_grids[crowd_g[unique_remain_grids] == min_crowd]

        current_mask = np.isin(site[remain], best_grids)
        current = remain[current_mask]

        min_dist = np.min(r_dis[current][:, choose_mask], axis=1) if np.sum(choose_mask) > 0 else 0
        fitness = 0.1 * M * con_violation[current] - min_dist

        best_idx = current[np.argmin(fitness)]
        choose_mask[best_idx] = True
        crowd_g[site[best_idx]] += 1
    return choose_mask


def data_process(xxs, yys):
    """
    Split data into training and testing sets with global shuffling.
    """
    pha = 0.75
    train_idx, test_idx = [], []
    for label in [0, 1, -1]:
        idx = np.where(yys == label)[0]
        n_sel = int(np.ceil(pha * len(idx)))
        perm = np.random.permutation(len(idx))
        train_idx.extend(idx[perm[:n_sel]])
        test_idx.extend(idx[perm[n_sel:]])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    perm_train = np.random.permutation(len(train_idx))
    train_idx = train_idx[perm_train]

    perm_test = np.random.permutation(len(test_idx))
    test_idx = test_idx[perm_test]
    # ==========================================

    return xxs[train_idx], yys[train_idx], xxs[test_idx], yys[test_idx]


def r_surrogate_assisted_selection(problem, ref_decs, pop_decs, wmax, s_model, task_id):
    """
    Surrogate-Assisted Internal Evolution.

    Uses the trained relation model to guide a Genetic Algorithm (GA) in searching
    for promising solutions without performing expensive real evaluations.

    - Limits the final output count to prevent excessive expensive evaluations.
    """
    # Initialize Population
    input_pop = np.vstack((pop_decs, ref_decs))
    next_pop = ga_generation(input_pop, 15, 5)

    # Internal Evolution Loop
    i = 0
    while i < wmax:
        # Evaluate current generation using the surrogate model
        sorted_idx, _ = model_select(s_model, next_pop)

        # Select parents
        n_ref = len(ref_decs)
        parents_indices = sorted_idx[:min(len(sorted_idx), n_ref)]
        input_for_ga = next_pop[parents_indices]

        # Generate NEXT generation (Overwriting old generation)
        mating_pool = np.vstack((input_for_ga, ref_decs))
        next_pop = ga_generation(mating_pool, 15, 5)

        i += len(next_pop)

    # Final Selection from the last virtual generation
    _, scores = model_select(s_model, next_pop)

    # Filter solutions with high predicted quality
    high_quality_mask = scores > 3.9
    good_indices = np.where(high_quality_mask)[0]
    sorted_indices = np.argsort(scores)[::-1]

    # Constraints on output size (Min 4, Max 10)
    min_output = 4
    max_output = 10

    if len(good_indices) < min_output:
        final_indices = sorted_indices[:min_output]
    elif len(good_indices) > max_output:

        final_indices = sorted_indices[:max_output]
    else:
        final_indices = good_indices

    return next_pop[final_indices]


def model_select(s_model, candidates):
    """
    Vectorized Model Selection.

    Batches all candidate-reference pairs into a single tensor for efficient
    inference
    """
    scaler, net, device = s_model['scaler'], s_model['net'], s_model['device']
    X, Y = s_model['X'], s_model['Y']

    if len(X) == 0: return np.arange(len(candidates)), np.zeros(len(candidates))

    c1 = X[Y == 1]
    c2 = X[Y != 1]
    n_c1, n_c2 = len(c1), len(c2)
    n_cand = len(candidates)

    # 1: Data Preparation
    all_pairs_list = []
    slice_indices = [0]

    for xi in candidates:
        xi_rep = xi.reshape(1, -1)

        if n_c1 > 0:
            all_pairs_list.append(np.hstack((c1, np.tile(xi_rep, (n_c1, 1)))))
            all_pairs_list.append(np.hstack((np.tile(xi_rep, (n_c1, 1)), c1)))
        if n_c2 > 0:
            all_pairs_list.append(np.hstack((c2, np.tile(xi_rep, (n_c2, 1)))))
            all_pairs_list.append(np.hstack((np.tile(xi_rep, (n_c2, 1)), c2)))

        count = (2 * n_c1 if n_c1 > 0 else 0) + (2 * n_c2 if n_c2 > 0 else 0)
        slice_indices.append(slice_indices[-1] + count)

    if not all_pairs_list:
        return np.arange(n_cand), np.zeros(n_cand)

    # 2: Inference
    full_batch = np.vstack(all_pairs_list)
    batch_norm = scaler.transform(full_batch)

    net.eval()
    with torch.no_grad():
        tensor_data = torch.tensor(batch_norm, dtype=torch.float32).to(device)
        all_probs = torch.softmax(net(tensor_data), dim=1).cpu().numpy()

    #3: Scoring
    scores = np.zeros(n_cand)
    for i in range(n_cand):

        start, end = slice_indices[i], slice_indices[i + 1]
        probs = all_probs[start:end]


        idx, score_val = 0, 0
        if n_c1 > 0:
            p_c1xi = probs[idx: idx + n_c1]
            p_xic1 = probs[idx + n_c1: idx + 2 * n_c1]
            idx += 2 * n_c1
            score_val += np.mean(p_c1xi[:, 1] + p_c1xi[:, 2] + p_xic1[:, 0] + p_xic1[:, 1]) \
                         - np.mean(p_c1xi[:, 0] + p_xic1[:, 2])
        if n_c2 > 0:
            p_c2xi = probs[idx: idx + n_c2]
            p_xic2 = probs[idx + n_c2: idx + 2 * n_c2]
            score_val += np.mean(p_c2xi[:, 2] + p_xic2[:, 0]) \
                         - np.mean(p_c2xi[:, 0] + p_c2xi[:, 1] + p_xic2[:, 1] + p_xic2[:, 2])
        scores[i] = score_val

    return np.argsort(scores)[::-1], scores

class MapMinMax:
    """
    Min-Max normalization utility.
    """
    def __init__(self):
        self.min = None
        self.max = None
    def fit_transform(self, X):
        self.min, self.max = np.min(X, axis=0), np.max(X, axis=0)
        return self.transform(X)
    def transform(self, X):
        if self.min is None: return X
        denom = self.max - self.min
        denom[denom==0] = 1.0
        return (X - self.min) / denom * 2 - 1

def onehot_encoding_indices(labels):
    mapping = {1:0, 0:1, -1:2}
    return torch.tensor(np.array([mapping[l] for l in labels]), dtype=torch.long)

class RelationNet(nn.Module):
    """
    Feed-forward Neural Network for relationship prediction.
    """
    def __init__(self, x_dim):
        super().__init__()
        h1 = int(np.ceil(x_dim*2.5))
        h2 = int(np.ceil(x_dim * 2))
        h3 =int(x_dim*1.5)
        h4 = int(x_dim)
        h5 =int(np.ceil(x_dim/2))

        self.net = nn.Sequential(
            nn.Linear(x_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, h4), nn.ReLU(),
            nn.Linear(h4, h5), nn.ReLU(),
            nn.Linear(h5, h5), nn.ReLU(),
            nn.Linear(h5, 3)
        )
    def forward(self, x): return self.net(x)


def train_model(net, X, y, device, epochs=100, lr=1.0):
    """
    Train the network using L-BFGS optimizer.
    L-BFGS is chosen for its efficiency in handling full-batch optimization.
    """
    X_t = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y_t = y.to(device)

    # L-BFGS configuration
    optimizer = optim.LBFGS(net.parameters(), lr=lr,
                            history_size=10,
                            max_iter=20,
                            line_search_fn='strong_wolfe')

    loss_fn = nn.CrossEntropyLoss()
    net.train()

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            output = net(X_t)
            loss = loss_fn(output, y_t)
            loss.backward()
            return loss
        optimizer.step(closure)

# Adam optimizer
# def train_model(net, X, y, device, epochs=1000, lr=0.01):
#     opt = optim.Adam(net.parameters(), lr=lr)
#     loss_fn = nn.CrossEntropyLoss()
#     X_t, y_t = torch.tensor(np.array(X), dtype=torch.float32).to(device), y.to(device)
#     net.train()
#     for _ in range(epochs):
#         opt.zero_grad()
#         loss_fn(net(X_t), y_t).backward()
#         opt.step()
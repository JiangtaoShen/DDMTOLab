"""
Scenario-based Self-Learning Transfer Differential Evolution (SSLT-DE)

This module implements SSLT-DE for multi-task optimization using a DQN-based
reinforcement learning framework to adaptively select among four knowledge
transfer scenarios.

References
----------
    [1] Z. Yuan, G. Dai, L. Peng, M. Wang, Z. Song, and X. Chen, "Scenario-based
        self-learning transfer framework for multi-task optimization problems,"
        Knowledge-Based Systems, vol. 325, p. 113824, 2025.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


# ---------------------------------------------------------------------------
# Dropout network hyper-parameters (MToP: SSLT/Dropout/{trainmodel,updatemodel,
# trainNet,testNet,iniA}.m).  Everything below matches the reference except the
# number of mini-batch steps: MATLAB runs 80000 steps to build and 8000 steps to
# update, i.e. ~16M sample gradients per build, which is intractable here.  The
# round counts are therefore scaled down by 10x while keeping the 10:1 ratio.
# ---------------------------------------------------------------------------
_BUILD_ROUNDS = 8000
_UPDATE_ROUNDS = 800
_BATCH_SIZE = 200
_LEARN_RATE = 0.01
_WEIGHT_DECAY = 1e-05
_HIDDEN = 40
_DROP_P = (0.2, 0.5)


# ============================================================================
# Helper Functions
# ============================================================================

def _matlab_round(x):
    """MATLAB round(): half away from zero (numpy rounds half to even)."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def _safe_div(num, den):
    """
    MATLAB evaluates the ratios below without guarding the denominator (a zero
    denominator yields Inf/NaN which then poisons the network inputs).  Zero
    denominators are mapped to 0 here so the state stays finite; every other
    case is bit-identical to the reference.
    """
    return float(num) / float(den) if den != 0 else 0.0


def _wasserstein_1d(u_decs, v_decs):
    """
    1D Wasserstein distance between two flattened population arrays
    (SSLT_DE.m ``Ws`` + ``find_interval``).
    """
    u = np.sort(u_decs.ravel())
    v = np.sort(v_decs.ravel())
    all_vals = np.unique(np.concatenate([u, v]))
    if len(all_vals) < 2:
        return 0.0
    u_cdf = np.searchsorted(u, all_vals[:-1], side='right') / len(u)
    v_cdf = np.searchsorted(v, all_vals[:-1], side='right') / len(v)
    return float(np.sum(np.abs(u_cdf - v_cdf) * np.diff(all_vals)))


def _dispersion_metric(decs, objs):
    """
    Mean pairwise squared distance among the top 10% individuals
    (SSLT_DE.m ``Dispersion``).

    Notes
    -----
    The MATLAB reference indexes ``M_pop(i + 1:end)`` linearly instead of
    row-wise, so ``up`` accumulates vectors of different lengths and the
    function raises a size-mismatch error for any M >= 3 (M = round(0.1 * N),
    i.e. M = 10 at the default N = 100).  The intended scalar reading -- the
    sum of squared distances over all elite pairs -- is used here.
    """
    M = max(_matlab_round(0.1 * len(objs)), 1)
    rank = np.argsort(objs.flatten(), kind='stable')
    top = decs[rank[:M]]
    if M <= 1:
        return 0.0
    total = 0.0
    for i in range(M - 1):
        diff = top[i + 1:] - top[i]
        total += np.sum(diff ** 2)
    return total / (M * (M - 1))


def _dispersion_type(decs, objs, decs_old, objs_old):
    """Compare dispersion: 1=decreasing, 2=same, 3=increasing."""
    dm = _dispersion_metric(decs, objs)
    dm_old = _dispersion_metric(decs_old, objs_old)
    if dm - dm_old < 0:
        return 1
    elif dm - dm_old == 0:
        return 2
    else:
        return 3


def _convergence_dist(decs_old, decs_new):
    """Euclidean distance between old and new population centers."""
    c_old = np.mean(decs_old, axis=0)
    c_new = np.mean(decs_new, axis=0)
    return float(np.sqrt(np.sum((c_old - c_new) ** 2)))


def _smooth(decs, objs):
    """
    Keep the best individual of every consecutive triple (SSLT_DE.m ``Smooth``).

    MATLAB deletes the 2nd- and 3rd-best member of each triple and keeps
    everything else, so any trailing individuals that do not fill a complete
    triple survive.
    """
    n = len(decs)
    delete = set()
    for i in range(0, n - 2, 3):
        order = np.argsort(objs[i:i + 3].flatten(), kind='stable')
        delete.add(i + order[1])
        delete.add(i + order[2])
    keep = [i for i in range(n) if i not in delete]
    return decs[keep], objs[keep]


def _de_crossover_single(trial, target, CR):
    """
    Binomial crossover for a single pair, matching MToP's ``DE_Crossover``:
    the trial value is kept where rand < CR (plus one forced position) and the
    target value is taken elsewhere.
    """
    d = len(trial)
    mask = np.random.rand(d) < CR
    mask[np.random.randint(d)] = True
    offspring = target.copy()
    offspring[mask] = trial[mask]
    return offspring


def _de_generation_sslt(parents, F, CR):
    """
    DE/rand/1/bin exactly as SSLT_DE.m ``Generation``.

    x1, x2 and x3 are distinct from each other but -- unlike the shared
    ``de_generation`` helper -- they are *not* excluded from equalling the
    target index i.
    """
    n, d = parents.shape
    off = np.zeros((n, d))
    for i in range(n):
        x1 = np.random.randint(n)
        x2 = np.random.randint(n)
        while x2 == x1:
            x2 = np.random.randint(n)
        x3 = np.random.randint(n)
        while x3 == x2 or x3 == x1:
            x3 = np.random.randint(n)
        v = parents[x1] + F * (parents[x2] - parents[x3])
        off[i] = _de_crossover_single(v, parents[i], CR)
    return np.clip(off, 0.0, 1.0)


def _selection_tournament(p_objs, p_cons, o_objs, o_cons):
    """
    One-to-one selection matching MToP's ``Selection_Tournament`` (epsilon = 0):
    the parent is replaced only if both are infeasible and the offspring has a
    strictly lower CV, or both are feasible and the offspring has a strictly
    lower objective.  Ties always keep the parent.
    """
    n = p_objs.shape[0]
    p_cv = np.sum(np.maximum(0, p_cons), axis=1) \
        if p_cons.shape[1] > 0 else np.zeros(n)
    o_cv = np.sum(np.maximum(0, o_cons), axis=1) \
        if o_cons.shape[1] > 0 else np.zeros(n)
    replace_cv = (p_cv > o_cv) & (p_cv > 0) & (o_cv > 0)
    equal_cv = (p_cv <= 0) & (o_cv <= 0)
    replace_f = p_objs[:, 0] > o_objs[:, 0]
    return (equal_cv & replace_f) | replace_cv


def _constrained_rank(objs, cons):
    """Sort ascending by constraint violation then objective (sortrows [1,2])."""
    n = objs.shape[0]
    cv = np.sum(np.maximum(0, cons), axis=1) if cons.shape[1] > 0 \
        else np.zeros(n)
    return np.lexsort((objs[:, 0], cv))


def _pad_cons(cons_real, max_c):
    """Pad a per-task constraint matrix to the unified width max_c."""
    n = cons_real.shape[0]
    out = np.zeros((n, max_c))
    if max_c > 0 and cons_real.shape[1] > 0:
        c = min(max_c, cons_real.shape[1])
        out[:, :c] = cons_real[:, :c]
    return out


def _normalize(X):
    """Min-max normalize columns to [-1, 1], matching MATLAB mapminmax."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1.0
    return 2.0 * (X - mins) / rng - 1.0, mins, maxs


def _normalize_apply(x, mins, maxs):
    """Apply saved min-max normalization."""
    rng = maxs - mins
    rng[rng == 0] = 1.0
    return 2.0 * (x - mins) / rng - 1.0


# ============================================================================
# Q-Network for DQN (port of MToP SSLT/Dropout)
# ============================================================================

class _QNet(nn.Module):
    """
    Port of MToP's ``trainNet``/``testNet``: 7 -> 40 (ReLU) -> 40 (tanh) -> 1,
    with dropout p = 0.2 on the input and p = 0.5 on the first hidden
    activation.  MATLAB's ``testNet`` applies dropout at inference time as well,
    so predictions are stochastic; that behaviour is reproduced here.

    The reference network has 7 outputs (reward plus the 6 next-state features)
    but only column 1 is ever read, and ``updatemodel`` broadcasts the scalar
    Q-target onto all 7 columns.  A single output is therefore equivalent.
    """

    def __init__(self, input_dim=7, hidden_dim=_HIDDEN, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.drop_in = nn.Dropout(_DROP_P[0])
        self.drop_hidden = nn.Dropout(_DROP_P[1])

        # iniA.m: W ~ N(0, 1/fan_in); B{1} = 0.1 * sqrt(1/n); B{2}, B{3} random
        with torch.no_grad():
            for layer in (self.fc1, self.fc2, self.fc3):
                fan_in = layer.weight.shape[1]
                layer.weight.normal_(0.0, np.sqrt(1.0 / fan_in))
            self.fc1.bias.fill_(0.1 * np.sqrt(1.0 / hidden_dim))
            self.fc2.bias.normal_(0.0, np.sqrt(1.0 / hidden_dim))
            self.fc3.bias.normal_(0.0, np.sqrt(1.0 / output_dim))

    def forward(self, x):
        x = self.drop_in(x)
        x = torch.relu(self.fc1(x))
        x = self.drop_hidden(x)
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def _train_qnet(model, X, y, rounds):
    """Mini-batch SGD exactly as trainNet.m (0.5 * squared error, lr = 0.01)."""
    weights = [model.fc1.weight, model.fc2.weight, model.fc3.weight]
    biases = [model.fc1.bias, model.fc2.bias, model.fc3.bias]
    optimizer = torch.optim.SGD(
        [{'params': weights, 'weight_decay': _WEIGHT_DECAY / _BATCH_SIZE},
         {'params': biases, 'weight_decay': 0.0}], lr=_LEARN_RATE)

    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    n = X_t.shape[0]

    model.train()
    for _ in range(rounds):
        sel = torch.randint(0, n, (_BATCH_SIZE,))
        pred = model(X_t[sel])
        loss = 0.5 * ((pred - y_t[sel]) ** 2).sum() / _BATCH_SIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _predict_qnet(model, X):
    """Forward pass with dropout active, matching testNet.m."""
    model.train()
    with torch.no_grad():
        out = model(torch.tensor(np.asarray(X), dtype=torch.float32))
    return out.numpy().reshape(-1)


# ============================================================================
# SSLT-DE Algorithm
# ============================================================================

class SSLT_DE:
    """
    Scenario-based Self-Learning Transfer Differential Evolution.

    Uses a DQN-based reinforcement learning framework to adaptively select
    among four knowledge transfer scenarios:
    1. No transfer (standard DE/rand/1/bin)
    2. Shape transfer (shift smoothed source toward target center)
    3. Bi-directional transfer (DE on merged populations)
    4. Domain transfer (direction-guided from best source-target difference)

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
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None,
                 threshold=150, gap=50, gamma=0.9, epsilon=0.8,
                 F=0.5, CR=0.9,
                 save_data=True, save_path='./Data', name='SSLT-DE',
                 disable_tqdm=True):
        """
        Initialize SSLT-DE algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        threshold : int, optional
            Number of generations before building DQN (default: 150)
        gap : int, optional
            DQN update interval in generations (default: 50)
        gamma : float, optional
            Discount factor for Q-learning (default: 0.9)
        epsilon : float, optional
            Epsilon-greedy exploration rate (default: 0.8)
        F : float, optional
            DE mutation scale factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.9)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SSLT-DE')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.threshold = threshold
        self.gap = gap
        self.gamma = gamma
        self.epsilon = epsilon
        self.F = F
        self.CR = CR
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SSLT-DE algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Convert to unified space for cross-task operations
        pop_decs, pop_cons = space_transfer(
            problem=problem, decs=decs, cons=cons, type='uni', padding='mid')
        pop_objs = [o.copy() for o in objs]
        maxD = pop_decs[0].shape[1]
        maxC = pop_cons[0].shape[1]

        # Per-task DQN state
        data_task = [[] for _ in range(nt)]       # Experience replay buffer
        model_built = [False] * nt                 # Whether DQN is built
        count_task = [0] * nt                      # Update counter
        q_model = [None] * nt                      # Q-network per task
        norm_params = [None] * nt                  # Normalization parameters

        # Store previous generation populations
        pop_decs_old = [d.copy() for d in pop_decs]
        pop_objs_old = [o.copy() for o in pop_objs]

        # MToP increments Algo.Gen inside notTerminated before the loop body
        # runs, so the first executed generation is Gen = 2.
        gen = 2
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            for t in range(nt):
                if nfes >= max_nfes:
                    break

                # Random source task
                s = np.random.randint(nt)
                while s == t and nt > 1:
                    s = np.random.randint(nt)

                # ============================================================
                # Compute state features
                # ============================================================
                min_old_t = np.min(pop_objs_old[t])
                min_cur_t = np.min(pop_objs[t])
                min_old_s = np.min(pop_objs_old[s])
                min_cur_s = np.min(pop_objs[s])

                conv_target = _safe_div(min_old_t - min_cur_t, min_old_t)
                conv_source = _safe_div(min_old_s - min_cur_s, min_old_s)
                wsd = _wasserstein_1d(pop_decs[t], pop_decs[s])
                ls_target = _dispersion_type(pop_decs[t], pop_objs[t],
                                             pop_decs_old[t], pop_objs_old[t])
                ls_source = _dispersion_type(pop_decs[s], pop_objs[s],
                                             pop_decs_old[s], pop_objs_old[s])
                pha = nfes / max_nfes

                state = np.array([conv_source, conv_target, wsd,
                                  ls_target, ls_source, pha])

                # ============================================================
                # Action selection
                # ============================================================
                if gen <= self.threshold:
                    action = np.random.randint(1, 5)
                elif not model_built[t]:
                    # Build DQN model
                    exp = np.array(data_task[t])
                    X_raw = exp[:, :7]
                    y_raw = exp[:, 7]
                    X_norm, x_min, x_max = _normalize(X_raw)
                    y_norm, y_min, y_max = _normalize(y_raw.reshape(-1, 1))
                    y_norm = y_norm.flatten()

                    q_model[t] = _QNet()
                    _train_qnet(q_model[t], X_norm, y_norm, _BUILD_ROUNDS)
                    norm_params[t] = (x_min, x_max, y_min, y_max)
                    model_built[t] = True
                    action = np.random.randint(1, 5)
                else:
                    # Epsilon-greedy
                    if np.random.rand() > self.epsilon:
                        action = np.random.randint(1, 5)
                    else:
                        x_min, x_max, y_min, y_max = norm_params[t]
                        cand = np.array([np.append(state, a)
                                         for a in range(1, 5)])
                        cand = _normalize_apply(cand, x_min, x_max)
                        q_vals = _predict_qnet(q_model[t], cand)
                        action = int(np.argmax(q_vals)) + 1

                # ============================================================
                # Execute action
                # ============================================================
                pop_decs_old[t] = pop_decs[t].copy()
                pop_objs_old[t] = pop_objs[t].copy()

                if action == 1:
                    # No KT: standard DE/rand/1/bin
                    off_decs = _de_generation_sslt(pop_decs[t], self.F, self.CR)
                    off_objs, off_cons_real = evaluation_single(
                        problem, off_decs[:, :dims[t]], t)
                    off_cons = _pad_cons(off_cons_real, maxC)
                    nfes += n
                    pbar.update(n)

                    # One-to-one Selection_Tournament
                    better = _selection_tournament(
                        pop_objs[t], pop_cons[t], off_objs, off_cons)
                    pop_decs[t][better] = off_decs[better]
                    pop_objs[t][better] = off_objs[better]
                    pop_cons[t][better] = off_cons[better]

                elif action == 2:
                    # Shape KT: shift smoothed source toward target center.
                    # MATLAB does not clip the shifted decisions before
                    # evaluating them, so no clipping is applied here either.
                    sm_s_decs, sm_s_objs = _smooth(pop_decs[s], pop_objs[s])
                    sm_t_decs, _ = _smooth(pop_decs[t], pop_objs[t])

                    center_t = np.mean(sm_t_decs, axis=0)
                    center_s = np.mean(sm_s_decs, axis=0)
                    shifted = sm_s_decs + (center_t - center_s)

                    n_shifted = len(shifted)
                    sh_objs, sh_cons_real = evaluation_single(
                        problem, shifted[:, :dims[t]], t)
                    sh_cons = _pad_cons(sh_cons_real, maxC)
                    nfes += n_shifted
                    pbar.update(n_shifted)

                    # Elite selection (target U shifted)
                    merged_decs = np.vstack([pop_decs[t], shifted])
                    merged_objs = np.vstack([pop_objs[t], sh_objs])
                    merged_cons = np.vstack([pop_cons[t], sh_cons])
                    sel = selection_elit(objs=merged_objs, n=n,
                                         cons=merged_cons)
                    pop_decs[t] = merged_decs[sel]
                    pop_objs[t] = merged_objs[sel]
                    pop_cons[t] = merged_cons[sel]

                elif action == 3:
                    # Bi-KT: DE on merged populations
                    merged = np.vstack([pop_decs[t], pop_decs[s]])
                    n_merged = len(merged)

                    off_decs = np.zeros_like(merged)
                    for i in range(n_merged):
                        # DE/current-to-rand/1 on the merged population
                        idxs = list(range(n_merged))
                        idxs.remove(i)
                        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                        v = merged[i] + self.F * (merged[r1] - merged[i]) \
                            + 0.5 * (merged[r2] - merged[r3])
                        off_decs[i] = _de_crossover_single(v, merged[i], self.CR)
                    off_decs = np.clip(off_decs, 0, 1)

                    off_objs, off_cons_real = evaluation_single(
                        problem, off_decs[:, :dims[t]], t)
                    off_cons = _pad_cons(off_cons_real, maxC)
                    nfes += n_merged
                    pbar.update(n_merged)

                    # Elite selection (target U offspring)
                    merged_sel = np.vstack([pop_decs[t], off_decs])
                    merged_sel_objs = np.vstack([pop_objs[t], off_objs])
                    merged_sel_cons = np.vstack([pop_cons[t], off_cons])
                    sel = selection_elit(objs=merged_sel_objs, n=n,
                                         cons=merged_sel_cons)
                    pop_decs[t] = merged_sel[sel]
                    pop_objs[t] = merged_sel_objs[sel]
                    pop_cons[t] = merged_sel_cons[sel]

                elif action == 4:
                    # Domain KT: direction-guided transfer
                    rank_s = _constrained_rank(pop_objs[s], pop_cons[s])
                    rank_t = _constrained_rank(pop_objs[t], pop_cons[t])
                    direction = pop_decs[s][rank_s[0]] - pop_decs[t][rank_t[0]]

                    num = max(1, _matlab_round(pha * 10))
                    perm = np.random.permutation(n)

                    off_decs = np.zeros((num, maxD))
                    for i in range(num):
                        idx = perm[i % n]
                        off_decs[i] = _de_crossover_single(
                            pop_decs[t][idx], direction, self.CR)
                    off_decs = np.clip(off_decs, 0, 1)

                    off_objs, off_cons_real = evaluation_single(
                        problem, off_decs[:, :dims[t]], t)
                    off_cons = _pad_cons(off_cons_real, maxC)
                    nfes += num
                    pbar.update(num)

                    # Elite selection (target U offspring)
                    merged_decs = np.vstack([pop_decs[t], off_decs])
                    merged_objs = np.vstack([pop_objs[t], off_objs])
                    merged_cons = np.vstack([pop_cons[t], off_cons])
                    sel = selection_elit(objs=merged_objs, n=n,
                                         cons=merged_cons)
                    pop_decs[t] = merged_decs[sel]
                    pop_objs[t] = merged_objs[sel]
                    pop_cons[t] = merged_cons[sel]

                # ============================================================
                # Compute reward and store experience
                # ============================================================
                fold = np.min(pop_objs_old[t])
                f = np.min(pop_objs[t])
                fold_mean = np.mean(pop_objs_old[t])
                f_mean = np.mean(pop_objs[t])

                imp_rate = _safe_div(fold - f, fold)
                pop_rate = _safe_div(fold_mean - f_mean, fold_mean)
                move_dis = _convergence_dist(pop_decs_old[t], pop_decs[t])

                vals = np.array([imp_rate, pop_rate, move_dis])
                max_val, min_val = vals.max(), vals.min()
                rng = max_val - min_val
                if rng != 0:
                    imp_rate_n = (imp_rate - min_val) / rng
                    pop_rate_n = (pop_rate - min_val) / rng
                    move_dis_n = (move_dis - min_val) / rng
                else:
                    imp_rate_n = pop_rate_n = move_dis_n = 0.0

                pha_new = nfes / max_nfes
                reward = (imp_rate_n + pop_rate_n + move_dis_n) * pha_new

                # New state features
                conv_new_target = _safe_div(
                    np.min(pop_objs_old[t]) - np.min(pop_objs[t]),
                    np.min(pop_objs_old[t]))
                conv_new_source = _safe_div(
                    np.min(pop_objs_old[s]) - np.min(pop_objs[s]),
                    np.min(pop_objs_old[s]))
                wsd_new = _wasserstein_1d(pop_decs[s], pop_decs[t])
                ls_new_target = _dispersion_type(pop_decs[t], pop_objs[t],
                                                 pop_decs_old[t], pop_objs_old[t])
                ls_new_source = _dispersion_type(pop_decs[s], pop_objs[s],
                                                 pop_decs_old[s], pop_objs_old[s])

                record = np.array([
                    conv_source, conv_target, wsd, ls_target, ls_source, pha, action,
                    reward, conv_new_source, conv_new_target, wsd_new,
                    ls_new_target, ls_new_source, pha_new
                ])
                data_task[t].append(record)
                if len(data_task[t]) > 500:
                    data_task[t].pop(0)

                # ============================================================
                # Update DQN periodically
                # ============================================================
                if model_built[t]:
                    count_task[t] += 1
                    if count_task[t] > self.gap:
                        exp = np.array(data_task[t])
                        X_raw = exp[:, :7]
                        rewards_raw = exp[:, 7]

                        X_norm, x_min, x_max = _normalize(X_raw)

                        # Bootstrapped target: R + gamma * max(Q)
                        max_q = np.max(_predict_qnet(q_model[t], X_norm))
                        target_q = rewards_raw + self.gamma * max_q

                        y_norm, y_min, y_max = _normalize(
                            target_q.reshape(-1, 1))
                        y_norm = y_norm.flatten()

                        norm_params[t] = (x_min, x_max, y_min, y_max)
                        _train_qnet(q_model[t], X_norm, y_norm, _UPDATE_ROUNDS)
                        count_task[t] = 0

            # Record history in real space
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs,
                           all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

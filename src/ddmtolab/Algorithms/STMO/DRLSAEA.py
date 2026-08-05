"""
Deep Reinforcement Learning-assisted Surrogate-Assisted Evolutionary Algorithm (DRL-SAEA)

This module implements DRL-SAEA for computationally expensive constrained multi-objective
optimization. It uses a Double Deep Q-Network (DDQN) to dynamically select among three
constraint handling strategies for surrogate model management.

References
----------
    [1] S. Shao, Y. Tian, and Y. Zhang. Deep reinforcement learning assisted surrogate model management for expensive constrained multi-objective optimization. Swarm and Evolutionary Computation, 2025, 92: 101817.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build, gp_predict
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# DDQN Components
# ============================================================================

class QNetwork(nn.Module):
    """Simple feedforward Q-network with tanh hidden layers and linear output."""

    def __init__(self, n_states, n_actions, hidden_layers=(5, 5)):
        super().__init__()
        layers = []
        in_dim = n_states
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ExperienceReplay:
    """Circular buffer for storing (S, A, R, S') transitions."""

    def __init__(self, n_states, max_size=100):
        self.max_size = max_size
        self.n_states = n_states
        # buffer columns: [S (n_states), A (1), R (1), S' (n_states)]
        self.buffer = np.zeros((max_size, 2 * n_states + 2))
        self.index = 0
        self.overflow = False

    def insert(self, state, action, reward, next_state):
        idx = self.index % self.max_size
        row = np.concatenate([state, [action, reward], next_state])
        self.buffer[idx] = row
        self.index += 1
        if self.index >= self.max_size:
            self.overflow = True

    def get_batch(self, batch_size):
        max_idx = self.max_size if self.overflow else self.index
        if max_idx == 0:
            return None, None, None, None
        indices = np.random.randint(0, max_idx, size=batch_size)
        batch = self.buffer[indices]
        ns = self.n_states
        S = batch[:, :ns]
        A = batch[:, ns].astype(int)
        R = batch[:, ns + 1]
        Sn = batch[:, ns + 2:]
        return S, A, R, Sn

    @property
    def size(self):
        return self.max_size if self.overflow else self.index


class DDQN:
    """Double Deep Q-Network for action selection in DRL-SAEA."""

    def __init__(self, n_states, n_actions, hidden_layers=(5, 5), max_er=100,
                 discount=0.999, lr=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.memory = ExperienceReplay(n_states, max_er)

        self.agent = QNetwork(n_states, n_actions, hidden_layers)
        self.target = QNetwork(n_states, n_actions, hidden_layers)
        self.copy_weights_agent_to_target()

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def copy_weights_agent_to_target(self):
        self.target.load_state_dict(self.agent.state_dict())

    def action(self, state):
        """Select action with biased exploration (favoring argmax Q)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.agent(state_t).squeeze(0).numpy()
        # Biased random: argmax repeated n_actions times + all actions
        best_action = np.argmax(q_vals)
        candidates = np.concatenate([
            np.full(self.n_actions, best_action),
            np.arange(self.n_actions)
        ])
        return int(np.random.choice(candidates))

    def store(self, state, action, reward, next_state):
        self.memory.insert(state, action, reward, next_state)

    def experience_replay(self, batch_size, epochs=20):
        # Experience_Replay.get_batch samples the stored transitions *with*
        # replacement, so a replay is performed even before the buffer holds
        # `batch_size` entries (the very first replay happens at step 8 with a
        # batch of 16).
        if self.memory.size == 0:
            return
        S, A, R, Sn = self.memory.get_batch(batch_size)
        S_t = torch.FloatTensor(S)
        Sn_t = torch.FloatTensor(Sn)
        R_t = torch.FloatTensor(R)

        # DDQN: use agent to select best next action, target to evaluate it
        with torch.no_grad():
            q_current = self.agent(S_t)
            q_next_agent = self.agent(Sn_t)
            q_next_target = self.target(Sn_t)
            next_best_actions = q_next_agent.argmax(dim=1)
            q_next_vals = q_next_target[torch.arange(q_next_agent.shape[0]), next_best_actions]

            # Build target Q values
            q_target = q_current.clone()
            for i in range(q_target.shape[0]):
                q_target[i, A[i]] = R_t[i] + self.discount * q_next_vals[i]

        # Train.  MATLAB calls train(net,S',Q_target), i.e. a full supervised
        # fit of the batch; a single gradient step is far too weak an analogue,
        # so a short optimisation run is performed instead.
        for _ in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.agent(S_t), q_target)
            loss.backward()
            self.optimizer.step()


def _env_selection(pop_dec, pop_obj, NI, M, status):
    """
    Environmental selection using SPEA2 fitness and truncation.

    Parameters
    ----------
    pop_dec : np.ndarray, shape (N, D)
    pop_obj : np.ndarray, shape (N, M_ext) - may include constraint columns
    NI : int, target size
    M : int, number of real objectives
    status : int, action index (1, 2, or 3)

    Returns
    -------
    pop_dec, pop_obj, fitness : selected arrays
    """
    # Remove duplicates
    _, unique_idx = np.unique(pop_obj[:, :M], axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    pop_dec = pop_dec[unique_idx]
    pop_obj = pop_obj[unique_idx]

    N = pop_dec.shape[0]
    if N == 0:
        return pop_dec, pop_obj, np.array([])

    # Calculate fitness based on action status
    if status == 1:
        real_obj = pop_obj[:, :M]
        cv = np.maximum(0, pop_obj[:, -1:])
        fitness = spea2_fitness(real_obj, cv)
    elif status == 2:
        real_obj = pop_obj[:, :M]
        pop_con = pop_obj[:, M:]
        cv = np.sum(np.maximum(0, pop_con), axis=1, keepdims=True)
        fitness = spea2_fitness(np.hstack([real_obj, cv]))
    else:  # status == 3
        fitness = spea2_fitness(pop_obj)

    # Selection: prefer fitness < 1, fill/truncate as needed
    next_mask = fitness < 1
    if np.sum(next_mask) < NI:
        rank = np.argsort(fitness)
        next_mask[:] = False
        next_mask[rank[:min(NI, N)]] = True
    elif np.sum(next_mask) > NI:
        if status != 3:
            real_obj_sel = pop_obj[next_mask, :M]
        else:
            real_obj_sel = pop_obj[next_mask]
        kept_indices = spea2_truncation(real_obj_sel, NI)
        temp = np.where(next_mask)[0]
        next_mask[:] = False
        next_mask[temp[kept_indices]] = True

    return pop_dec[next_mask], pop_obj[next_mask], fitness[next_mask]


# ============================================================================
# State and Reward Generation
# ============================================================================

def _compute_hv(objs, optimum):
    """
    Hypervolume of a solution set, following PlatEMO's ``HV`` metric.

    ``HV.m`` normalizes the (feasible non-dominated) objectives with
    ``fmin = min(min(PopObj,[],1),0)`` and ``fmax = max(optimum,[],1)``, divides
    by ``(fmax-fmin)*1.1``, discards every point outside the unit box and
    measures the volume dominating the reference point ``(1,...,1)``.  Note that
    ``fmin`` is taken from the set being measured, so the two hypervolumes of
    ``GenerateSample`` use slightly different normalizations - this is
    reproduced verbatim.

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (N, M)
    optimum : np.ndarray
        Upper reference (``max([Last.objs;Cur.objs])``), shape (M,)

    Returns
    -------
    hv : float
    """
    if objs.shape[0] == 0:
        return 0.0

    # Only the non-dominated subset contributes (Population.best)
    front_no, _ = nd_sort(objs, objs.shape[0])
    objs = objs[front_no == 1]
    if objs.shape[0] == 0:
        return 0.0

    fmin = np.minimum(np.min(objs, axis=0), 0.0)
    denom = (np.asarray(optimum, dtype=float) - fmin) * 1.1
    denom[denom <= 0] = 1e-12
    P = (objs - fmin) / denom
    P = P[~np.any(P > 1, axis=1)]
    if P.shape[0] == 0:
        return 0.0

    ref = np.ones(P.shape[1])
    if P.shape[1] < 4:
        return _exact_hv(P, ref)
    # PlatEMO switches to Monte-Carlo sampling for M >= 4
    n_samples = 100000
    samples = np.random.uniform(0.0, 1.0, size=(n_samples, P.shape[1]))
    dominated = np.zeros(n_samples, dtype=bool)
    for i in range(P.shape[0]):
        dominated |= np.all(samples >= P[i], axis=1)
    return float(np.mean(dominated))


def _exact_hv(objs, ref):
    """
    Exact hypervolume by recursive slicing (hypervolume-by-slicing-objectives).

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (N, M), all dominating ``ref``
    ref : np.ndarray
        Reference point, shape (M,)

    Returns
    -------
    hv : float
    """
    n, M = objs.shape
    if n == 0:
        return 0.0
    if M == 1:
        return float(max(ref[0] - np.min(objs[:, 0]), 0.0))
    if M == 2:
        return _exact_hv_2d(objs, ref)

    order = np.argsort(objs[:, -1])
    sorted_objs = objs[order]
    hv = 0.0
    for k in range(n):
        z = sorted_objs[k, -1]
        z_next = sorted_objs[k + 1, -1] if k + 1 < n else ref[-1]
        if z_next > z:
            hv += _exact_hv(sorted_objs[:k + 1, :-1], ref[:-1]) * (z_next - z)
    return hv


def _exact_hv_2d(objs, ref_point):
    """Exact 2D hypervolume via sweep line."""
    N = objs.shape[0]
    if N == 0:
        return 0.0
    # Keep the non-dominated staircase (ties on f1 resolved by the smaller f2)
    order = np.lexsort((objs[:, 1], objs[:, 0]))
    sorted_objs = objs[order]
    nd = [sorted_objs[0]]
    for i in range(1, N):
        if sorted_objs[i, 1] < nd[-1][1]:
            nd.append(sorted_objs[i])
    nd = np.array(nd)

    hv = 0.0
    for i in range(len(nd)):
        x_width = (nd[i + 1, 0] if i + 1 < len(nd) else ref_point[0]) - nd[i, 0]
        y_height = ref_point[1] - nd[i, 1]
        hv += max(x_width, 0.0) * max(y_height, 0.0)
    return hv


def _generate_sample(last_objs, last_cons, curr_objs, curr_cons, fe_ratio):
    """
    Generate the state/reward pair for the DDQN (``GenerateSample``).

    Note that ``GenerateSample`` uses the *current* ``Problem.FE/Problem.maxFE``
    for both the returned state and next state, and the returned ``state``
    overwrites the state that was used to pick the action, so the stored
    transition carries the up-to-date evaluation ratio in both entries.

    Parameters
    ----------
    last_objs : np.ndarray, shape (N1, M), last archive objectives
    last_cons : np.ndarray, shape (N1, C), last archive constraints
    curr_objs : np.ndarray, shape (N2, M), current archive objectives
    curr_cons : np.ndarray, shape (N2, C), current archive constraints
    fe_ratio : float, FE / maxFE at the time of the call

    Returns
    -------
    last_state : np.ndarray, shape (4,)
    curr_state : np.ndarray, shape (4,)
    reward : float
    """
    # Compute HV with the shared upper reference
    all_objs = np.vstack([last_objs, curr_objs])
    optimum = np.max(all_objs, axis=0)

    last_hv = _compute_hv(last_objs, optimum)
    curr_hv = _compute_hv(curr_objs, optimum)

    # Reward 1: HV improvement ratio
    if last_hv == 0 or np.isnan(last_hv):
        # MATLAB only guards the 0/0 NaN case; the 0-denominator guard is kept
        # here so that an all-zero initial HV cannot inject Inf into the net.
        reward1 = 0.0
    else:
        reward1 = (curr_hv - last_hv) / last_hv
    if np.isnan(reward1):
        reward1 = 0.0

    # Reward 2: CV improvement ratio
    last_cv_total = np.sum(np.maximum(0, last_cons))
    curr_cv_total = np.sum(np.maximum(0, curr_cons))
    if last_cv_total == 0:
        reward2 = 0.0
    elif curr_cv_total < last_cv_total:
        reward2 = abs((curr_cv_total - last_cv_total) / last_cv_total)
    else:
        reward2 = -abs((curr_cv_total - last_cv_total) / last_cv_total)

    reward = reward1 + reward2

    # States (MATLAB's var() is the unbiased sample variance)
    last_state = np.array([
        np.sum(np.var(last_objs, axis=0, ddof=1)) if last_objs.shape[0] > 1 else 0.0,
        np.sum(last_objs),
        last_cv_total,
        fe_ratio
    ])
    curr_state = np.array([
        np.sum(np.var(curr_objs, axis=0, ddof=1)) if curr_objs.shape[0] > 1 else 0.0,
        np.sum(curr_objs),
        curr_cv_total,
        fe_ratio
    ])

    return last_state, curr_state, reward


# ============================================================================
# Archive and Population Update
# ============================================================================

def _update_archive(arc_decs, arc_objs, arc_cons, N):
    """
    Update archive: deduplicate, SPEA2 selection with truncation.

    Parameters
    ----------
    arc_decs, arc_objs, arc_cons : archive data
    N : int, max archive size

    Returns
    -------
    arc_decs, arc_objs, arc_cons : updated archive
    """
    # Deduplicate
    _, unique_idx = np.unique(arc_objs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    arc_decs = arc_decs[unique_idx]
    arc_objs = arc_objs[unique_idx]
    arc_cons = arc_cons[unique_idx]

    if arc_decs.shape[0] > N:
        fitness = spea2_fitness(arc_objs, arc_cons)
        next_mask = fitness < 1
        if np.sum(next_mask) < N:
            rank = np.argsort(fitness)
            next_mask[:] = False
            next_mask[rank[:N]] = True
        elif np.sum(next_mask) > N:
            kept_indices = spea2_truncation(arc_objs[next_mask], N)
            temp = np.where(next_mask)[0]
            next_mask[:] = False
            next_mask[temp[kept_indices]] = True
        arc_decs = arc_decs[next_mask]
        arc_objs = arc_objs[next_mask]
        arc_cons = arc_cons[next_mask]

    return arc_decs, arc_objs, arc_cons


def _update_population(pop_decs, pop_objs, pop_cons, new_decs, new_objs, new_cons,
                        N, status):
    """
    Update population: remove duplicates with new, SPEA2 selection to N, append new.

    Parameters
    ----------
    pop_decs, pop_objs, pop_cons : current population
    new_decs, new_objs, new_cons : newly evaluated solutions
    N : int, target size (NI - mu)
    status : int, action index

    Returns
    -------
    pop_decs, pop_objs, pop_cons : updated population
    """
    # Remove solutions in population that duplicate with new
    if new_objs.shape[0] > 0:
        keep = []
        for i in range(pop_objs.shape[0]):
            is_dup = np.any(np.all(np.abs(pop_objs[i] - new_objs) < 1e-10, axis=1))
            if not is_dup:
                keep.append(i)
        if len(keep) > 0:
            keep = np.array(keep)
            pop_decs = pop_decs[keep]
            pop_objs = pop_objs[keep]
            pop_cons = pop_cons[keep]
        else:
            pop_decs = np.empty((0, pop_decs.shape[1]))
            pop_objs = np.empty((0, pop_objs.shape[1]))
            pop_cons = np.empty((0, pop_cons.shape[1]))

    # Deduplicate within population
    if pop_objs.shape[0] > 0:
        _, unique_idx = np.unique(pop_objs, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        pop_decs = pop_decs[unique_idx]
        pop_objs = pop_objs[unique_idx]
        pop_cons = pop_cons[unique_idx]

    # SPEA2 selection to N
    if pop_decs.shape[0] > N:
        if status == 1:
            fitness = spea2_fitness(pop_objs, pop_cons)
        elif status == 2:
            cv = np.sum(np.maximum(0, pop_cons), axis=1, keepdims=True)
            fitness = spea2_fitness(np.hstack([pop_objs, cv]))
        else:  # status == 3
            fitness = spea2_fitness(pop_objs, pop_cons)

        next_mask = fitness < 1
        if np.sum(next_mask) < N:
            rank = np.argsort(fitness)
            next_mask[:] = False
            next_mask[rank[:N]] = True
        elif np.sum(next_mask) > N:
            kept_indices = spea2_truncation(pop_objs[next_mask], N)
            temp = np.where(next_mask)[0]
            next_mask[:] = False
            next_mask[temp[kept_indices]] = True
        pop_decs = pop_decs[next_mask]
        pop_objs = pop_objs[next_mask]
        pop_cons = pop_cons[next_mask]

    # Append new solutions
    if new_decs.shape[0] > 0:
        pop_decs = np.vstack([pop_decs, new_decs])
        pop_objs = np.vstack([pop_objs, new_objs])
        pop_cons = np.vstack([pop_cons, new_cons])

    return pop_decs, pop_objs, pop_cons


# ============================================================================
# Normalize Constraint Violation
# ============================================================================

def _normalize_cv(pop_con):
    """
    Normalize constraint violations and return aggregated scalar CV per solution.

    Parameters
    ----------
    pop_con : np.ndarray, shape (N, C), raw constraint values

    Returns
    -------
    cv : np.ndarray, shape (N, 1), normalized aggregate CV
    """
    pop_con = np.maximum(0, pop_con)
    cmin = np.min(pop_con, axis=0)
    cmax = np.max(pop_con, axis=0)
    denom = cmax - cmin
    denom[denom == 0] = 1.0
    pop_con_norm = (pop_con - cmin) / denom
    # Handle NaN from zero columns
    pop_con_norm = np.nan_to_num(pop_con_norm, nan=0.0)
    return np.sum(pop_con_norm, axis=1, keepdims=True)


# ============================================================================
# Main Algorithm
# ============================================================================

class DRLSAEA:
    """
    Deep Reinforcement Learning-assisted Surrogate-Assisted Evolutionary Algorithm
    for expensive constrained multi-objective optimization.

    Uses DDQN to select among 3 constraint handling strategies:
    - Action 0: objectives + normalized aggregate CV
    - Action 1: objectives + individual normalized active constraints
    - Action 2: objectives only (unconstrained)

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
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100,
                 wmax=20, mu=5,
                 save_data=True, save_path='./Data', name='DRL-SAEA', disable_tqdm=True):
        """
        Initialize DRL-SAEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Archive/population size per task (default: 100)
        wmax : int, optional
            Number of inner GA generations on surrogates (default: 20)
        mu : int, optional
            Number of real evaluated solutions per iteration (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'DRL-SAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.wmax = wmax
        self.mu = mu
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the DRL-SAEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs
        n_cons = problem.n_cons

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # History tracking
        has_cons = any(c.shape[1] > 0 for c in cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Per-task optimization
        for task_i in range(nt):
            m = n_objs[task_i]
            dim = dims[task_i]
            nc = n_cons[task_i]
            NI = n_initial_per_task[task_i]
            N_archive = n_per_task[task_i]

            # Population and archive for this task
            pop_decs = decs[task_i].copy()
            pop_objs = objs[task_i].copy()
            pop_cons = cons[task_i].copy() if nc > 0 else np.zeros((NI, 0))

            # Initialize archive: Archive = UpdateArchive(Population,Problem.N)
            # is applied unconditionally, i.e. the archive is only trimmed when
            # it exceeds N and dominated solutions are otherwise retained.
            arc_decs, arc_objs, arc_cons_full = _update_archive(
                pop_decs.copy(), pop_objs.copy(),
                pop_cons.copy() if nc > 0 else np.zeros((NI, 1)),
                N_archive
            )
            arc_cons = arc_cons_full if nc > 0 else np.zeros((arc_decs.shape[0], 0))

            last_arc_objs = arc_objs.copy()
            last_arc_cons = arc_cons.copy() if nc > 0 else np.zeros((arc_objs.shape[0], 1))

            # DDQN setup
            num_actions = 3
            num_states = 4
            exp_replay_freq = 8
            copy_weights_freq = exp_replay_freq + 3  # 11
            batch_size = 16
            step = 0
            ddqn = DDQN(num_states, num_actions, (5, 5), max_er=100)

            # Initial state
            fe_ratio = nfes_per_task[task_i] / max_nfes_per_task[task_i]
            cv_total = np.sum(np.maximum(0, last_arc_cons)) if nc > 0 else 0.0
            state = np.array([
                np.sum(np.var(arc_objs, axis=0, ddof=1)) if arc_objs.shape[0] > 1 else 0.0,
                np.sum(arc_objs),
                cv_total,
                fe_ratio
            ])

            while nfes_per_task[task_i] < max_nfes_per_task[task_i]:
                action = ddqn.action(state)
                step += 1

                # Build surrogate models based on action
                if action == 0:
                    # Action 1 in MATLAB: objectives + normalized aggregate CV
                    status = 1
                    if nc > 0:
                        cv_col = _normalize_cv(pop_cons)
                        surr_objs = np.hstack([pop_objs, cv_col])
                    else:
                        surr_objs = np.hstack([pop_objs, np.zeros((pop_objs.shape[0], 1))])
                    M_surr = surr_objs.shape[1]
                    models = []
                    for j in range(M_surr):
                        gp = gp_build(pop_decs, surr_objs[:, j:j+1], data_type)
                        models.append(gp)
                    fitness = spea2_fitness(pop_objs, pop_cons if nc > 0 else None)

                elif action == 1:
                    # Action 2 in MATLAB: objectives + individual active normalized constraints
                    status = 2
                    if nc > 0:
                        raw_con = np.maximum(0, pop_cons)
                        max_con = np.max(raw_con, axis=0)
                        active_idx = np.where(max_con > 0)[0]
                        if len(active_idx) > 0:
                            active_con = raw_con[:, active_idx]
                            cmin = np.min(active_con, axis=0)
                            cmax = np.max(active_con, axis=0)
                            denom = cmax - cmin
                            denom[denom == 0] = 1.0
                            active_con_norm = (active_con - cmin) / denom
                            surr_objs = np.hstack([pop_objs, active_con_norm])
                        else:
                            surr_objs = pop_objs.copy()
                    else:
                        surr_objs = pop_objs.copy()
                    M_surr = surr_objs.shape[1]
                    models = []
                    for j in range(M_surr):
                        gp = gp_build(pop_decs, surr_objs[:, j:j+1], data_type)
                        models.append(gp)
                    # Fitness with aggregated CV as extra objective
                    if nc > 0:
                        cv_agg = np.sum(np.maximum(0, pop_cons), axis=1, keepdims=True)
                        fitness = spea2_fitness(np.hstack([pop_objs, cv_agg]))
                    else:
                        fitness = spea2_fitness(pop_objs)

                else:  # action == 2
                    # Action 3 in MATLAB: objectives only
                    status = 3
                    surr_objs = pop_objs.copy()
                    M_surr = m
                    models = []
                    for j in range(M_surr):
                        gp = gp_build(pop_decs, surr_objs[:, j:j+1], data_type)
                        models.append(gp)
                    fitness = spea2_fitness(pop_objs)

                # Inner GA evolution on surrogates
                inner_decs = pop_decs.copy()
                inner_objs = surr_objs.copy()
                inner_fitness = fitness.copy()

                for w in range(self.wmax):
                    # Tournament selection and GA generation.  TournamentSelection
                    # picks the *minimum* SPEA2 fitness, which is also the
                    # convention of the shared helper, so the fitness is passed
                    # unchanged.
                    mating_pool = tournament_selection(2, NI, inner_fitness)
                    off_decs = ga_generation(inner_decs[mating_pool], muc=20.0, mum=20.0)
                    inner_decs = np.vstack([inner_decs, off_decs])

                    # Predict all solutions using surrogates
                    N_inner = inner_decs.shape[0]
                    pred_objs = np.zeros((N_inner, M_surr))
                    for j in range(M_surr):
                        pred_j, _ = gp_predict(models[j], inner_decs, data_type)
                        pred_objs[:, j] = pred_j.ravel()

                    inner_objs = pred_objs

                    # Environmental selection
                    inner_decs, inner_objs, inner_fitness = _env_selection(
                        inner_decs, inner_objs, NI, m, status
                    )

                # Select mu best for expensive evaluation
                sel_decs, _, _ = _env_selection(inner_decs, inner_objs, self.mu, m, status)

                # Remove duplicates and respect the remaining budget
                sel_decs = remove_duplicates(sel_decs, decs[task_i])
                sel_decs = sel_decs[:max_nfes_per_task[task_i] - nfes_per_task[task_i]]
                if sel_decs.shape[0] == 0:
                    # If all duplicates, generate random solutions
                    sel_decs = np.random.rand(self.mu, dim)
                    sel_decs = remove_duplicates(sel_decs, decs[task_i])
                    sel_decs = sel_decs[:max_nfes_per_task[task_i] - nfes_per_task[task_i]]
                    if sel_decs.shape[0] == 0:
                        continue

                # Expensive evaluation
                new_objs, new_cons = evaluation_single(problem, sel_decs, task_i)
                n_new = sel_decs.shape[0]

                # Update cumulative dataset
                decs[task_i] = np.vstack([decs[task_i], sel_decs])
                objs[task_i] = np.vstack([objs[task_i], new_objs])
                if nc > 0:
                    cons[task_i] = np.vstack([cons[task_i], new_cons])

                # Update population
                pop_decs, pop_objs, pop_cons = _update_population(
                    pop_decs, pop_objs, pop_cons if nc > 0 else np.zeros((pop_decs.shape[0], max(1, nc))),
                    sel_decs, new_objs, new_cons if nc > 0 else np.zeros((n_new, max(1, nc))),
                    NI - self.mu, status
                )
                # Restore cons shape for unconstrained
                if nc == 0:
                    pop_cons = np.zeros((pop_decs.shape[0], 0))

                # Update archive
                combined_arc_decs = np.vstack([arc_decs, sel_decs])
                combined_arc_objs = np.vstack([arc_objs, new_objs])
                if nc > 0:
                    combined_arc_cons = np.vstack([arc_cons, new_cons])
                else:
                    combined_arc_cons = np.zeros((combined_arc_objs.shape[0], 1))
                arc_decs, arc_objs, arc_cons_full = _update_archive(
                    combined_arc_decs, combined_arc_objs, combined_arc_cons, N_archive
                )
                if nc == 0:
                    arc_cons = np.zeros((arc_decs.shape[0], 0))
                else:
                    arc_cons = arc_cons_full

                nfes_per_task[task_i] += n_new
                pbar.update(n_new)

                # Generate state/reward for DDQN
                curr_cons_for_state = arc_cons if nc > 0 else np.zeros((arc_objs.shape[0], 1))
                last_cons_for_state = last_arc_cons if nc > 0 else np.zeros((last_arc_objs.shape[0], 1))
                fe_ratio = nfes_per_task[task_i] / max_nfes_per_task[task_i]

                last_state, curr_state, reward = _generate_sample(
                    last_arc_objs, last_cons_for_state,
                    arc_objs, curr_cons_for_state,
                    fe_ratio
                )

                # GenerateSample also returns the state, which overwrites the
                # state used for the action before it is stored.
                ddqn.store(last_state, action, reward, curr_state)
                state = curr_state
                last_arc_objs = arc_objs.copy()
                last_arc_cons = arc_cons.copy() if nc > 0 else np.zeros((arc_objs.shape[0], 1))

                # Periodic DDQN updates
                if step % exp_replay_freq == 0:
                    ddqn.experience_replay(batch_size)
                if step % copy_weights_freq == 0:
                    ddqn.copy_weights_agent_to_target()


        pbar.close()
        runtime = time.time() - start_time

        if has_cons:
            all_decs, all_objs, all_cons = build_staircase_history(decs, objs, k=self.mu, db_cons=cons)
        else:
            all_decs, all_objs = build_staircase_history(decs, objs, k=self.mu)
            all_cons = None
        results = build_save_results(
            all_decs, all_objs, runtime, nfes_per_task,
            all_cons=all_cons if has_cons else None,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )
        return results

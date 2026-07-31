"""
Evolutionary Multi-task with Effective Transfer (EMT-ET)

This module implements EMT-ET for multi-task multi-objective optimization problems.

References
----------
    [1] Lin, Jiabin, Hai-Lin Liu, Kay Chen Tan, and Fangqing Gu. "An Effective Knowledge Transfer Approach for Multiobjective Multitasking Optimization." IEEE Transactions on Cybernetics 51.6 (2021): 3238-3248.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.16
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def _platemo_tournament_selection(K, N, *fitness):
    """
    Exact port of PlatEMO's ``TournamentSelection`` (MToP ``TournamentSelection.m``).

    Candidates are compared lexicographically on the given fitness keys (lower is
    better). Solutions sharing identical fitness values also share a rank, so a
    tournament among tied candidates is won by the (random) first draw, i.e.
    uniformly at random. The shared ``tournament_selection`` helper instead ranks
    with a total order, which would always let the lowest-index candidate win --
    a systematic bias here, because the G transferred individuals are all given
    the same top fitness as the incumbent best.
    """
    fits = np.column_stack([np.asarray(f, dtype=float).ravel() for f in fitness])
    _, loc = np.unique(fits, axis=0, return_inverse=True)
    loc = loc.ravel()
    parents = np.random.randint(0, fits.shape[0], size=(K, N))
    best = np.argmin(loc[parents], axis=0)
    return parents[best, np.arange(N)]


def _sbx_crossover_unclipped(par_dec1, par_dec2, mu):
    """
    Simulated binary crossover (MToP ``GA_Crossover``).

    Unlike the shared ``crossover`` helper the offspring are NOT clipped to
    [0, 1] here: the MATLAB reference clips only once at the end of Generation,
    after polynomial mutation has acted on the raw crossover output.
    """
    d = par_dec1.shape[0]
    u = np.random.rand(d)
    beta = np.zeros(d)
    mask = u <= 0.5
    beta[mask] = (2 * u[mask]) ** (1 / (mu + 1))
    beta[~mask] = (2 * (1 - u[~mask])) ** (-1 / (mu + 1))
    beta *= (-1.0) ** np.random.randint(0, 2, size=d)
    beta[np.random.rand(d) < 0.5] = 1.0

    off_dec1 = 0.5 * ((1 + beta) * par_dec1 + (1 - beta) * par_dec2)
    off_dec2 = 0.5 * ((1 + beta) * par_dec2 + (1 - beta) * par_dec1)
    return off_dec1, off_dec2


def _poly_mutation_unclipped(dec, mu):
    """
    Polynomial mutation (MToP ``GA_Mutation``) with per-gene probability 1/D.

    Operates on the possibly out-of-bounds crossover output and does NOT clip;
    the caller clips once afterwards, matching the MATLAB reference.
    """
    d = dec.shape[0]
    dec = dec.copy()
    prob_m = 1 / d
    for j in range(d):
        if np.random.rand() < prob_m:
            u = np.random.rand()
            if u <= 0.5:
                delta = (2 * u + (1 - 2 * u) * (1 - dec[j]) ** (mu + 1)) ** (1 / (mu + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * dec[j] ** (mu + 1)) ** (1 / (mu + 1))
            dec[j] += delta
    return dec


def _emt_et_generation(parents, muc, mum):
    """
    Offspring generation matching ``EMT_ET.Generation`` exactly.

    Parent ``i`` is paired with parent ``i + floor(L/2)`` for ``i = 0..ceil(L/2)-1``
    -- the mating pool is *not* permuted first (it already comes from a random
    tournament), and the single clip to [0, 1] happens only after mutation.
    For odd ``L`` this yields ``L + 1`` offspring, exactly as in MATLAB.

    Parameters
    ----------
    parents : np.ndarray
        Mating pool, shape (L, d)
    muc, mum : float
        Distribution indices of SBX and polynomial mutation

    Returns
    -------
    off_decs : np.ndarray
        Offspring, shape (2 * ceil(L / 2), d)
    parent_idx : np.ndarray
        For each offspring, the row of ``parents`` it was copied from. EMT-ET
        offspring inherit the transfer bookkeeping (``isTrans``/``OriginTask``)
        of that parent.
    """
    length, d = parents.shape
    half = length // 2
    n_pairs = int(np.ceil(length / 2))

    parent_idx = np.empty(2 * n_pairs, dtype=int)
    parent_idx[0::2] = np.arange(n_pairs)
    parent_idx[1::2] = np.arange(n_pairs) + half

    off_decs = np.empty((2 * n_pairs, d))
    count = 0
    for i in range(n_pairs):
        c1, c2 = _sbx_crossover_unclipped(parents[i, :], parents[i + half, :], muc)
        c1 = _poly_mutation_unclipped(c1, mum)
        c2 = _poly_mutation_unclipped(c2, mum)
        off_decs[count] = np.clip(c1, 0.0, 1.0)
        off_decs[count + 1] = np.clip(c2, 0.0, 1.0)
        count += 2

    return off_decs, parent_idx


class EMT_ET:
    """
    Evolutionary Multi-task with Effective Transfer.

    This algorithm features:
    - Adaptive knowledge transfer seeded by previously successful transfers
    - Transfer solutions taken from the neighbourhood of a successful immigrant
    - Multiplicative perturbation of transferred solutions
    - NSGA-II based environmental selection

    Notes
    -----
    Following the MTO-Platform reference implementation, the whole population is
    evolved in a *unified* :math:`[0, 1]^{\\max_t D_t}` space; when an individual
    is evaluated on task ``t`` only its first :math:`D_t` genes are used. Solutions
    are therefore exchanged verbatim between tasks of different dimensionality.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, G=8, P=0.5, muc=20.0, mum=15.0,
                 save_data=True, save_path='./Data', name='EMT-ET', disable_tqdm=True):
        """
        Initialize EMT-ET algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        G : int, optional
            Number of transfer solutions per generation (default: 8)
        P : float, optional
            Probability of perturbing a transferred solution (default: 0.5)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'EMTET_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.G = G
        self.P = P
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMT-ET algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize in the unified [0, 1]^max(D) space (MToP Initialization uses
        # rand(1, max(D)), i.e. the spare genes of the shorter tasks are random).
        decs = space_transfer(problem, initialization(problem, n_per_task),
                              type='uni', padding='random')

        objs, cons = [], []
        for t in range(nt):
            objs_t, cons_t = evaluation_single(problem, decs[t][:, :dims[t]], t)
            objs.append(objs_t)
            cons.append(cons_t)
        nfes_per_task = n_per_task.copy()

        # Transfer bookkeeping carried by each individual
        is_trans = [np.zeros(n_per_task[t], dtype=bool) for t in range(nt)]
        origin_task = [np.full(n_per_task[t], -1, dtype=int) for t in range(nt)]
        front_no = []

        # Initial NSGA-II sorting: the population is kept ordered by quality, so
        # its index doubles as its mating fitness later on.
        for t in range(nt):
            rank_t, front_no_t, _ = self._nsga2_sort(objs[t], cons[t])
            order = np.argsort(rank_t, kind='stable')
            decs[t] = decs[t][order]
            objs[t] = objs[t][order]
            cons[t] = cons[t][order]
            is_trans[t] = is_trans[t][order]
            origin_task[t] = origin_task[t][order]
            front_no.append(front_no_t[order])

        all_decs, all_objs, all_cons = init_history(
            [decs[t][:, :dims[t]] for t in range(nt)], objs, cons)

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            active_tasks = [t for t in range(nt) if nfes_per_task[t] < max_nfes_per_task[t]]
            if not active_tasks:
                break

            for t in active_tasks:
                n_t = n_per_task[t]

                # === Step 1: Transfer ===
                transfer_decs, transfer_origin = self._transfer(decs, is_trans, origin_task, front_no, t)

                transfer_objs, transfer_cons = evaluation_single(problem, transfer_decs[:, :dims[t]], t)
                nfes_per_task[t] += transfer_decs.shape[0]
                pbar.update(transfer_decs.shape[0])

                # The incumbents are no longer "newly transferred"; the immigrants are
                is_trans[t] = np.zeros(decs[t].shape[0], dtype=bool)

                decs[t] = np.vstack([decs[t], transfer_decs])
                objs[t] = np.vstack([objs[t], transfer_objs])
                cons[t] = np.vstack([cons[t], transfer_cons])
                is_trans[t] = np.concatenate([is_trans[t], np.ones(transfer_decs.shape[0], dtype=bool)])
                origin_task[t] = np.concatenate([origin_task[t], transfer_origin])

                # === Step 2: Generation ===
                # Mating fitness: the incumbents keep their quality index 1..N, the
                # G immigrants are all given the best possible fitness of 1.
                mating_fitness = np.concatenate([np.arange(1, n_t + 1),
                                                 np.ones(transfer_decs.shape[0])])
                mating_pool = _platemo_tournament_selection(2, n_t - self.G, mating_fitness)

                off_decs, parent_idx = _emt_et_generation(decs[t][mating_pool], self.muc, self.mum)
                # Offspring are copies of their parent, so they inherit its
                # transfer bookkeeping (MToP: offspring(count) = population(p1)).
                off_is_trans = is_trans[t][mating_pool][parent_idx]
                off_origin = origin_task[t][mating_pool][parent_idx]

                off_objs, off_cons = evaluation_single(problem, off_decs[:, :dims[t]], t)
                nfes_per_task[t] += off_decs.shape[0]
                pbar.update(off_decs.shape[0])

                # === Step 3: Selection ===
                decs[t] = np.vstack([decs[t], off_decs])
                objs[t] = np.vstack([objs[t], off_objs])
                cons[t] = np.vstack([cons[t], off_cons])
                is_trans[t] = np.concatenate([is_trans[t], off_is_trans])
                origin_task[t] = np.concatenate([origin_task[t], off_origin])

                rank_t, front_no_t, _ = self._nsga2_sort(objs[t], cons[t])
                order = np.argsort(rank_t, kind='stable')[:n_t]

                decs[t] = decs[t][order]
                objs[t] = objs[t][order]
                cons[t] = cons[t][order]
                is_trans[t] = is_trans[t][order]
                origin_task[t] = origin_task[t][order]
                front_no[t] = front_no_t[order]

                append_history(all_decs[t], decs[t][:, :dims[t]],
                               all_objs[t], objs[t],
                               all_cons[t], cons[t])

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _transfer(self, decs, is_trans, origin_task, front_no, t):
        """
        Build the G solutions transferred into task t.

        If the previous generation kept transferred individuals in the first
        non-dominated front, their decision-space neighbourhoods in the source
        task are exploited; otherwise G solutions are drawn at random from the
        other tasks. Every transferred solution is then perturbed with
        probability P.

        Parameters
        ----------
        decs : list of np.ndarray
            Unified-space decision variables for all tasks
        is_trans : list of np.ndarray
            Transfer flags for all tasks
        origin_task : list of np.ndarray
            Source task index of every transferred individual
        front_no : list of np.ndarray
            Front numbers for all tasks
        t : int
            Target task index

        Returns
        -------
        transfer_decs : np.ndarray
            Decision variables of the G transferred solutions, shape (G, d_uni)
        transfer_origin : np.ndarray
            Source task index of each transferred solution, shape (G,)
        """
        nt = len(decs)
        transfer_decs, transfer_origin = [], []

        # Immigrants (or their descendants) that reached the first front
        successful = np.where(is_trans[t] & (front_no[t] < 2))[0]

        if successful.size > 0:
            g_temp = int(np.ceil(self.G / successful.size))
            for s_idx in successful:
                ot = int(origin_task[t][s_idx])
                distances = np.sqrt(np.sum((decs[ot] - decs[t][s_idx]) ** 2, axis=1))
                nearest = np.argsort(distances, kind='stable')
                for j in range(min(g_temp, nearest.size)):
                    transfer_decs.append(decs[ot][nearest[j]].copy())
                    transfer_origin.append(ot)
            transfer_decs = transfer_decs[:self.G]
            transfer_origin = transfer_origin[:self.G]

        if len(transfer_decs) < self.G:
            task_pool = [k for k in range(nt) if k != t]
            while len(transfer_decs) < self.G:
                ot = task_pool[np.random.randint(len(task_pool))]
                transfer_decs.append(decs[ot][np.random.randint(decs[ot].shape[0])].copy())
                transfer_origin.append(ot)

        transfer_decs = np.asarray(transfer_decs, dtype=float)

        # Disturb
        for i in range(self.G):
            if np.random.rand() < self.P:
                transfer_decs[i] = np.clip(2 * np.random.rand() * transfer_decs[i], 0.0, 1.0)

        return transfer_decs, np.asarray(transfer_origin, dtype=int)

    @staticmethod
    def _nsga2_sort(objs, cons=None):
        """
        Sort solutions based on NSGA-II criteria (MToP ``NSGA2Sort``).

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (pop_size, n_obj)
        cons : np.ndarray, optional
            Constraint values of shape (pop_size, n_con)

        Returns
        -------
        rank : np.ndarray
            Position of each solution in the sorted order, shape (pop_size,)
        front_no : np.ndarray
            Front number of each solution, shape (pop_size,)
        crowd_dis : np.ndarray
            Crowding distance of each solution, shape (pop_size,)
        """
        pop_size = objs.shape[0]

        if cons is not None and cons.size > 0:
            front_no, _ = nd_sort(objs, cons, pop_size)
        else:
            front_no, _ = nd_sort(objs, pop_size)

        crowd_dis = crowding_distance(objs, front_no)

        # Ascending front number, then descending crowding distance
        order = np.lexsort((-crowd_dis, front_no))

        rank = np.empty(pop_size, dtype=int)
        rank[order] = np.arange(pop_size)

        return rank, front_no, crowd_dis

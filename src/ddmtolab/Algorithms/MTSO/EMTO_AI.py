"""
Evolutionary Multi-Task Optimization with Adaptive Intensity (EMTO-AI)

This module implements EMTO-AI for multi-task optimization with adaptive knowledge
transfer intensity based on cross-task competitiveness evaluation.

References
----------
    [1] Zhou, Xinyu, et al. "Evolutionary Multi-Task Optimization With Adaptive
        Intensity of Knowledge Transfer." IEEE Transactions on Emerging Topics
        in Computational Intelligence, 1-13, 2024.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class EMTO_AI:
    """
    Evolutionary Multi-Task Optimization with Adaptive Intensity of Knowledge Transfer.

    Uses a DE-based multi-factorial framework with:

    - Per-task elite archives for inter-task knowledge transfer
    - DE mutation with archive base vector for transfer, DE/rand/1 for intra-task
    - Binomial crossover
    - Adaptive transfer intensity (rmp) updated periodically by cross-evaluating
      each task's subpopulation on other tasks and measuring competitiveness

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

    def __init__(self, problem, n=None, max_nfes=None, F=0.5, CR=0.6,
                 rate=0.05, gap_gen=35, save_data=True, save_path='./Data',
                 name='EMTO-AI', disable_tqdm=True):
        """
        Initialize EMTO-AI algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        F : float, optional
            DE mutation scale factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.6)
        rate : float, optional
            Archive size as fraction of per-task population (default: 0.05)
        gap_gen : int, optional
            Generation interval for transfer intensity update (default: 35)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EMTO-AI')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.F = F
        self.CR = CR
        self.rate = rate
        self.gap_gen = gap_gen
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMTO-AI algorithm.

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
        max_nfes = self.max_nfes * nt
        pop_size = n * nt

        # Initialize population and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        # Cross-task evaluations land on the foreign task, so the per-task counts
        # diverge from the scalar budget counter
        nfes_per_task = [n] * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified space
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons,
                                            type='uni', padding='mid')
        pop_objs = objs
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        # Per-individual flag: 1 = transfer offspring that survived, 0 = not
        pop_is_off = [np.zeros(n, dtype=int) for _ in range(nt)]

        # Initialize archive: top rate% individuals per task
        arc_len = max(1, int(np.ceil(n * self.rate)))
        arc_decs = []
        for t in range(nt):
            sel = selection_elit(objs=pop_objs[t], n=arc_len, cons=pop_cons[t])
            arc_decs.append(pop_decs[t][sel].copy())

        # Initial transfer intensity. Initialization_MF evaluates every individual
        # on every task, so the reference already has a full cross-task objective
        # matrix here; reproduce it with one extra sweep per foreign task.
        cross_objs = self._cross_evaluate(problem, pop_decs, pop_objs, nt, dims, nfes_per_task)
        nfes += n * nt * (nt - 1)
        rmp = np.clip(_compute_rmp(cross_objs, pop_objs, nt), 0.0, 1.0)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        # MToP increments Algo.Gen inside notTerminated before the first loop body,
        # so the first generation executed by the reference runs with Gen == 2.
        gen = 2
        while nfes < max_nfes:
            # Merge populations from all tasks
            m_decs, m_objs, m_cons, m_sfs = vstack_groups(
                pop_decs, pop_objs, pop_cons, pop_sfs)

            # --- Generation ---
            off_decs = np.zeros_like(m_decs)
            off_objs = np.full_like(m_objs, np.inf)
            off_cons = np.zeros_like(m_cons)
            off_sfs = np.zeros_like(m_sfs)
            off_is_transfer = np.zeros(pop_size, dtype=int)

            maxD = m_decs.shape[1]

            for i in range(pop_size):
                sf_i = m_sfs[i].item()

                if np.random.rand() < rmp[sf_i]:
                    # --- Knowledge transfer: archive-base DE mutation ---
                    # Pick a random other task
                    other_tasks = [t for t in range(nt) if t != sf_i]
                    o = other_tasks[np.random.randint(len(other_tasks))]

                    # x1 from other task's archive
                    x1 = arc_decs[o][np.random.randint(len(arc_decs[o]))]

                    # x2 is drawn from the same task; the reference re-checks x2's
                    # skill factor when resampling x3, so x3 is only required to
                    # differ from i and x2 and may come from any task.
                    same_idx = np.where(m_sfs.flatten() == sf_i)[0]
                    same_idx = same_idx[same_idx != i]
                    if len(same_idx) == 0:
                        cands = np.arange(pop_size)
                        x2 = int(np.random.choice(cands[cands != i]))
                    else:
                        x2 = int(np.random.choice(same_idx))

                    cands = np.arange(pop_size)
                    cands = cands[(cands != i) & (cands != x2)]
                    x3 = int(np.random.choice(cands))

                    # DE mutation: archive_base + F*(x2 - x3)
                    mutant = x1 + self.F * (m_decs[x2] - m_decs[x3])
                    trial = _de_binomial_crossover(mutant, m_decs[i], self.CR, maxD)

                    off_decs[i] = trial
                    off_sfs[i] = np.random.randint(nt)  # random task assignment
                    off_is_transfer[i] = 1
                else:
                    # --- Intra-task: DE/rand/1/bin ---
                    same_idx = np.where(m_sfs.flatten() == sf_i)[0]
                    same_idx = same_idx[same_idx != i]
                    if len(same_idx) < 3:
                        cands = np.arange(pop_size)
                        cands = cands[cands != i]
                        chosen = np.random.choice(cands, 3, replace=False)
                    else:
                        chosen = np.random.choice(same_idx, 3, replace=False)

                    mutant = m_decs[chosen[0]] + self.F * (m_decs[chosen[1]] - m_decs[chosen[2]])
                    trial = _de_binomial_crossover(mutant, m_decs[i], self.CR, maxD)

                    off_decs[i] = trial
                    off_sfs[i] = sf_i
                    off_is_transfer[i] = 0

                # Out-of-bounds handling: replace with random in [0,1]
                rand_dec = np.random.rand(maxD)
                oob = (off_decs[i] > 1) | (off_decs[i] < 0)
                off_decs[i][oob] = rand_dec[oob]

            # --- Evaluation ---
            for idx in range(pop_size):
                t = off_sfs[idx].item()
                off_objs[idx], off_cons[idx] = evaluation_single(
                    problem, off_decs[idx, :dims[t]], t)
                nfes_per_task[t] += 1

            nfes += pop_size
            pbar.update(pop_size)

            # --- Selection: merge parents + offspring, keep best n per task ---
            merged_decs = np.vstack([m_decs, off_decs])
            merged_objs = np.vstack([m_objs, off_objs])
            merged_cons = np.vstack([m_cons, off_cons])
            merged_sfs = np.vstack([m_sfs, off_sfs])
            merged_is_off = np.concatenate([
                np.zeros(pop_size, dtype=int), off_is_transfer])

            pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []
            pop_is_off = []
            for t in range(nt):
                indices = np.where(merged_sfs.flatten() == t)[0]
                t_decs, t_objs, t_cons = select_by_index(
                    indices, merged_decs, merged_objs, merged_cons)
                t_is_off = merged_is_off[indices]

                sel = selection_elit(objs=t_objs, n=n, cons=t_cons)
                pop_decs.append(t_decs[sel])
                pop_objs.append(t_objs[sel])
                pop_cons.append(t_cons[sel])
                pop_sfs.append(np.full((n, 1), t))
                pop_is_off.append(t_is_off[sel])

            # --- Update archive ---
            for t in range(nt):
                transfer_mask = pop_is_off[t] == 1
                tsf_decs = pop_decs[t][transfer_mask]

                if len(tsf_decs) >= arc_len:
                    arc_decs[t] = tsf_decs[:arc_len].copy()
                elif len(tsf_decs) == 0:
                    sel = selection_elit(objs=pop_objs[t], n=arc_len, cons=pop_cons[t])
                    arc_decs[t] = pop_decs[t][sel].copy()
                else:
                    # Remove oldest, add new transfer survivors
                    n_remove = min(len(tsf_decs), len(arc_decs[t]))
                    arc_decs[t] = np.vstack([
                        arc_decs[t][n_remove:], tsf_decs])[:arc_len]

            # Reset is_offspring flags
            pop_is_off = [np.zeros(n, dtype=int) for _ in range(nt)]

            # --- Update transfer intensity every gap_gen generations ---
            if gen % self.gap_gen == 0:
                # Cross-evaluate: evaluate each task's subpop on every other task
                # cross_objs[t][o] = task t's subpop objectives when evaluated on task o
                cross_objs = self._cross_evaluate(
                    problem, pop_decs, pop_objs, nt, dims, nfes_per_task)
                nfes += n * nt * (nt - 1)
                pbar.update(n * nt * (nt - 1))

                # Compute rmp: for each target task o, measure how competitive
                # source tasks' individuals are on task o
                rmp = np.clip(_compute_rmp(cross_objs, pop_objs, nt), 0.0, 1.0)

            # Record history (transform back to real space)
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs, all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    @staticmethod
    def _cross_evaluate(problem, pop_decs, pop_objs, nt, dims, nfes_per_task=None):
        """
        Evaluate every task's subpopulation on every other task.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        pop_decs : list of np.ndarray
            Per-task decision variables in the unified space
        pop_objs : list of np.ndarray
            Per-task objective values on the task the individuals belong to
        nt : int
            Number of tasks
        dims : list of int
            Decision variable dimensionality of each task

        Returns
        -------
        cross_objs : list of list of np.ndarray
            ``cross_objs[t][o]`` holds task ``t``'s subpopulation evaluated on
            task ``o``; the diagonal reuses the already known objectives.
        """
        cross_objs = [[None] * nt for _ in range(nt)]
        for t in range(nt):
            cross_objs[t][t] = pop_objs[t]
            for o in range(nt):
                if o == t:
                    continue
                c_objs, _ = evaluation_single(problem, pop_decs[t][:, :dims[o]], o)
                if nfes_per_task is not None:
                    nfes_per_task[o] += pop_decs[t].shape[0]
                cross_objs[t][o] = c_objs
        return cross_objs


# ============================================================
# Helper functions
# ============================================================

def _de_binomial_crossover(mutant, target, CR, d):
    """
    DE binomial crossover between mutant and target vectors.

    Parameters
    ----------
    mutant : np.ndarray
        Mutant vector, shape (d,)
    target : np.ndarray
        Target (parent) vector, shape (d,)
    CR : float
        Crossover rate in [0, 1]
    d : int
        Dimensionality

    Returns
    -------
    trial : np.ndarray
        Trial vector, shape (d,)
    """
    trial = target.copy()
    j_rand = np.random.randint(d)
    mask = np.random.rand(d) < CR
    mask[j_rand] = True
    trial[mask] = mutant[mask]
    return trial


def _get_nums(t_objs_on_o, o_objs_on_o):
    """
    Count how many individuals from source task are competitive on target task.

    Implements the getNums function from MATLAB: for each individual i in the
    source subpopulation, compare against all individuals in the target
    subpopulation. If the net wins (wins - losses) >= 0, the individual is
    considered competitive.

    Parameters
    ----------
    t_objs_on_o : np.ndarray
        Source task's objectives evaluated on target task, shape (n,)
    o_objs_on_o : np.ndarray
        Target task's native objectives, shape (n,)

    Returns
    -------
    better_num : int
        Number of competitive individuals from source task
    """
    n_t = len(t_objs_on_o)
    n_o = len(o_objs_on_o)
    better_num = 0

    for i in range(n_t):
        wins = np.sum(t_objs_on_o[i] < o_objs_on_o)
        losses = np.sum(t_objs_on_o[i] > o_objs_on_o)
        if wins - losses >= 0:
            better_num += 1

    return better_num


def _compute_rmp(cross_objs, pop_objs, nt):
    """
    Compute rmp from cross-task evaluation results.

    For each source task t and target task o (o != t):
        Evaluate how many of task t's individuals are competitive on task o.
        rmp[o] = better_num / len(subpop_t)

    For K > 2 tasks, rmp[o] is averaged over all source tasks t != o.

    This matches the MATLAB logic where rmp[o] controls transfer intensity
    for task o: higher means more knowledge transfer for that task.

    Parameters
    ----------
    cross_objs : list of list of np.ndarray
        cross_objs[t][o] = task t's subpop evaluated on task o, shape (n, 1)
    pop_objs : list of np.ndarray
        Native objective values per task, shape (n, 1)
    nt : int
        Number of tasks

    Returns
    -------
    rmp : np.ndarray
        Updated transfer intensity per task, shape (nt,)
    """
    rmp = np.zeros(nt)

    for o in range(nt):
        total_ratio = 0.0
        count = 0
        for t in range(nt):
            if t == o:
                continue
            t_on_o = cross_objs[t][o][:, 0]
            o_native = pop_objs[o][:, 0]
            better_num = _get_nums(t_on_o, o_native)
            total_ratio += better_num / len(t_on_o)
            count += 1
        if count > 0:
            rmp[o] = total_ratio / count

    return rmp

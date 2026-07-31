"""
Multi-Task Evolutionary Algorithm with Hierarchical Knowledge Transfer Strategy (MTEA-HKTS)

This module implements MTEA-HKTS for multi-task optimization using KLD-based
variable ordering, adaptive knowledge transfer with hierarchical strategy
selection, and alternating GA/DE operators.

References
----------
    [1] Zhao, Ben, et al. "A Multi-Task Evolutionary Algorithm for Solving
        the Problem of Transfer Targets." Information Sciences, 681: 121214,
        2024.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MTEA_HKTS:
    """
    Multi-Task EA with Hierarchical Knowledge Transfer Strategy.

    Uses KLD-based decision variable alignment across tasks, adaptive
    transfer probability control via a task selection table, and
    alternating GA (SBX+PM) / DE (rand/1/bin) operators.

    Three operation modes per generation:
    - sign=0 (10%): Separate transferred population evaluated independently
    - sign=1 (9%): Transferred individuals replace worst, standard GA/DE
    - sign=2 (81%): Transferred individuals in temp pop, cross-population GA/DE

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

    def __init__(self, problem, n=None, max_nfes=None, pTransfer=0.5,
                 mu=2, mum=5, F=0.5, CR=0.5, minx=0.1, Lb=0.1, Ub=0.7,
                 save_data=True, save_path='./Data', name='MTEA-HKTS',
                 disable_tqdm=True):
        """
        Initialize MTEA-HKTS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        pTransfer : float, optional
            Initial transfer portion (default: 0.5)
        mu : float, optional
            SBX crossover distribution index (default: 2)
        mum : float, optional
            Polynomial mutation distribution index (default: 5)
        F : float, optional
            DE mutation factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.5)
        minx : float, optional
            Minimum scale boundary (default: 0.1)
        Lb : float, optional
            Lower bound for transfer probability (default: 0.1)
        Ub : float, optional
            Upper bound for transfer probability (default: 0.7)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTEA-HKTS')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.pTransfer = pTransfer
        self.mu = mu
        self.mum = mum
        self.F = F
        self.CR = CR
        self.minx = minx
        self.Lb = Lb
        self.Ub = Ub
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTEA-HKTS algorithm.

        Returns
        -------
        Results
            Optimization results
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

        # Convert to unified space
        pop_decs, pop_cons = space_transfer(
            problem=problem, decs=decs, cons=cons, type='uni', padding='mid')
        pop_objs = objs
        maxD = pop_decs[0].shape[1]
        maxC = pop_cons[0].shape[1]

        # MATLAB sorts the initial population only for multi-objective tasks
        # (MTEA_HKTS.m:70-75); for single-objective it stays unsorted.

        # Transfer probability table (diagonal = 0)
        scale = np.full((nt, nt), self.pTransfer)
        table = np.full((nt, nt), 0.5)
        np.fill_diagonal(table, 0.0)

        # Archive: 3N individuals per task (decs + objs + cons)
        arch_decs = []
        arch_objs = []
        arch_cons = []
        for t in range(nt):
            idx = np.random.randint(n, size=3 * n)
            arch_decs.append(pop_decs[t][idx].copy())
            arch_objs.append(pop_objs[t][idx].copy())
            arch_cons.append(pop_cons[t][idx].copy())

        # MATLAB's Algo.Gen equals 2 during the first loop body because
        # notTerminated() already counted the initialization generation.
        gen = 1
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            gen += 1
            for t in range(nt):
                n_t = len(pop_decs[t])

                # --- Operation mode ---
                if np.random.rand() < 0.9:
                    sign = 1 if np.random.rand() < 0.1 else 2
                else:
                    sign = 0

                # --- Source task selection (roulette wheel) ---
                m_pt = _select_task(table[t])

                # --- Transfer option ---
                if np.random.rand() < table[t, m_pt]:
                    option = 1  # KLD-aligned transfer
                else:
                    m_pt = t
                    option = 2  # Self with exclusion

                if np.random.rand() > 0.9:
                    option = 0  # Random mapping

                # --- Variable ordering (using archives) ---
                order = _var_order(arch_decs[m_pt], arch_decs[t],
                                   dims[m_pt], dims[t], option)

                # --- Population preparation ---
                transpop_decs = None
                if sign == 1 or sign == 2:
                    nTransfer = min(_mround(scale[t, m_pt] * n_t), n_t)
                    temp_decs = pop_decs[t][::-1].copy()  # worst first
                    if nTransfer > 0:
                        temp_decs[:nTransfer] = _m_transfer(
                            pop_decs[m_pt], pop_decs[t],
                            dims[m_pt], dims[t], nTransfer, order, option)
                else:  # sign == 0
                    nTransfer = min(_mround(0.1 * n_t), n_t)
                    temp_decs = pop_decs[t].copy()
                    if nTransfer > 0:
                        transpop_decs = _m_transfer(
                            pop_decs[m_pt], pop_decs[t],
                            dims[m_pt], dims[t], nTransfer, order, option)

                # --- Generation ---
                op = 'GA' if t % 2 == 0 else 'DE'
                if sign == 1 or sign == 0:
                    if op == 'GA':
                        off_decs = _gen_ga(temp_decs, self.mu, self.mum)
                    else:
                        off_decs = _gen_de(temp_decs, self.F, self.CR)
                else:  # sign == 2
                    if op == 'GA':
                        off_decs = _gen_ga1(
                            pop_decs[t], temp_decs, self.mu, self.mum)
                    else:
                        off_decs = _gen_de1(
                            pop_decs[t], temp_decs, self.F, self.CR)

                # --- Evaluate offspring ---
                o_objs, o_cons_r = evaluation_single(
                    problem, off_decs[:, :dims[t]], t)
                o_cons = np.zeros((len(off_decs), maxC))
                if maxC > 0 and o_cons_r.shape[1] > 0:
                    o_cons[:, :o_cons_r.shape[1]] = o_cons_r
                nfes += len(off_decs)
                pbar.update(len(off_decs))

                # --- Merge and select ---
                if sign == 0 and transpop_decs is not None:
                    tp_objs, tp_cons_r = evaluation_single(
                        problem, transpop_decs[:, :dims[t]], t)
                    tp_cons = np.zeros((len(transpop_decs), maxC))
                    if maxC > 0 and tp_cons_r.shape[1] > 0:
                        tp_cons[:, :tp_cons_r.shape[1]] = tp_cons_r
                    nfes += len(transpop_decs)
                    pbar.update(len(transpop_decs))
                    m_decs = np.vstack([pop_decs[t], off_decs, transpop_decs])
                    m_objs = np.vstack([pop_objs[t], o_objs, tp_objs])
                    m_cons = np.vstack([pop_cons[t], o_cons, tp_cons])
                else:
                    m_decs = np.vstack([pop_decs[t], off_decs])
                    m_objs = np.vstack([pop_objs[t], o_objs])
                    m_cons = np.vstack([pop_cons[t], o_cons])
                    tp_objs = None

                # Sort by objective, select top N
                si = np.argsort(m_objs[:, 0])
                pop_decs[t] = m_decs[si[:n_t]]
                pop_objs[t] = m_objs[si[:n_t]]
                pop_cons[t] = m_cons[si[:n_t]]

                # --- Update archive (MATLAB updatearchive + Selection_Tournament) ---
                seg = gen % 3
                s_idx = seg * n_t
                a_slice = np.arange(s_idx, s_idx + n_t)
                a_cv = np.sum(np.maximum(0, arch_cons[t][a_slice]), axis=1)
                p_cv = np.sum(np.maximum(0, pop_cons[t]), axis=1)
                replace_cv = (a_cv > p_cv) & (a_cv > 0) & (p_cv > 0)
                equal_cv = (a_cv <= 0) & (p_cv <= 0)
                replace_f = arch_objs[t][a_slice, 0] > pop_objs[t][:, 0]
                replace = (equal_cv & replace_f) | replace_cv
                if np.any(replace):
                    tgt = a_slice[replace]
                    arch_decs[t][tgt] = pop_decs[t][replace]
                    arch_objs[t][tgt] = pop_objs[t][replace]
                    arch_cons[t][tgt] = pop_cons[t][replace]

                # --- Transfer quality tracking ---
                # MATLAB: [~, ia] = intersect(pop.Decs, X.Decs, 'rows'); sum(ia)
                # intersect returns the (lowest) indices of the unique common rows.
                rev_pop = pop_decs[t][::-1]  # worst first
                quality = transpop_decs if sign == 0 else off_decs

                ia_sum = 0
                if quality is not None and len(quality) > 0:
                    q_set = {np.ascontiguousarray(row).tobytes()
                             for row in quality}
                    seen = set()
                    for r_idx in range(n_t):
                        key = np.ascontiguousarray(rev_pop[r_idx]).tobytes()
                        if key in q_set and key not in seen:
                            seen.add(key)
                            ia_sum += (r_idx + 1)

                norm = n_t / 2.0 * (n_t + 1)
                ratio = ia_sum / norm if norm > 0 else 0

                if sign == 0:
                    scale[t, m_pt] = 0.1 + ratio * 0.4
                else:
                    scale[t, m_pt] = self.minx + ratio * (0.5 - self.minx)

                # --- Transfer probability table update ---
                if m_pt != t and option != 0:
                    denom = scale[t, m_pt] + scale[t, t]
                    if denom > 0:
                        temp_val = (scale[t, m_pt] - scale[t, t]) / denom
                    else:
                        temp_val = 0
                    w = 0.1 + np.random.rand() * 0.8
                    table[t, m_pt] = (
                        self.Lb + w * (table[t, m_pt] - self.Lb) +
                        (1 - w) * temp_val * (self.Ub - self.Lb))
                    if np.isnan(table[t, m_pt]) or table[t, m_pt] < self.Lb:
                        table[t, m_pt] = self.Lb
                    if table[t, m_pt] > self.Ub:
                        table[t, m_pt] = self.Ub

            # Record history
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs,
                           all_cons, real_cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)
        return results


# ============================================================
# Helper functions
# ============================================================

def _mround(value):
    """MATLAB ``round``: half away from zero (numpy/Python round half to even)."""
    return int(np.floor(np.abs(value) + 0.5)) * (1 if value >= 0 else -1)


def _select_task(table_row):
    """Roulette wheel selection based on transfer probability row."""
    s = np.sum(table_row)
    if s <= 0:
        return np.random.randint(len(table_row))
    probs = table_row / s
    return np.random.choice(len(table_row), p=probs)


def _reshape_stats(decs, dims):
    """
    MATLAB ``reshape([pop.Dec], dims, [])`` statistics.

    ``[pop.Dec]`` concatenates the (maxD-wide) decision rows into a single
    stream, which is then reshaped column-major into ``dims`` rows.  When
    ``dims == maxD`` this reduces to the ordinary per-column statistics;
    for unequal task dimensions MATLAB deliberately (if accidentally) mixes
    variables across individuals, which is reproduced here.
    """
    stream = np.ascontiguousarray(decs).reshape(-1)
    if dims > 0 and stream.size % dims == 0:
        mat = stream.reshape(dims, -1, order='F')
    else:  # not reshapeable (MATLAB would error) - fall back to columns
        mat = decs[:, :dims].T
    return mat.max(axis=1), mat.min(axis=1), mat.mean(axis=1)


def _var_order(prev_decs, this_decs, prev_dims, this_dims, option):
    """
    KLD-based decision variable ordering between two populations (varOrder2.m).

    Maps each target dimension to the source dimension with minimum KLD.

    Notes
    -----
    ``varOrder2.m`` line 44 rescales ``temp1(j)`` with a *single* subscript on
    the (N x maxD) deviation matrix, so MATLAB rescales the j-th element in
    column-major order rather than the j-th column.  That linear indexing is
    reproduced here for behavioural equivalence: in practice the source
    variances entering the KLD are the raw (unscaled) column variances, while
    the first column-major elements accumulate the range ratios across the
    target-dimension loop.
    """
    if option == 0:
        return np.random.randint(prev_dims, size=this_dims)

    n_prev = prev_decs.shape[0]
    n_this = this_decs.shape[0]

    prev_max = np.max(prev_decs, axis=0)
    prev_min = np.min(prev_decs, axis=0)
    this_max = np.max(this_decs, axis=0)
    this_min = np.min(this_decs, axis=0)
    prev_range = prev_max - prev_min
    this_range = this_max - this_min

    m_prev = np.mean(prev_decs, axis=0)
    m_this = np.mean(this_decs, axis=0)

    temp1 = np.asfortranarray((prev_decs - m_prev) ** 2)
    temp2 = (this_decs - m_this) ** 2
    sum2 = np.sum(temp2, axis=0) / max(n_this - 1, 1)

    # Column-major (linear-index) view onto temp1, matching MATLAB temp1(j)
    flat = temp1.T.reshape(-1)
    k = min(prev_dims, flat.size)

    order = np.zeros(this_dims, dtype=int)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for i in range(this_dims):
            upd = prev_range[:k] != this_range[i]
            if np.any(upd):
                ratio = this_range[i] / prev_range[:k]
                flat[:k][upd] *= ratio[upd] ** 2
            sum1 = np.sum(temp1[:, :prev_dims], axis=0) / max(n_prev - 1, 1)

            kld = (np.log2(np.sqrt(sum1) / np.sqrt(sum2[i])) +
                   (sum2[i] + (m_prev[:prev_dims] - m_this[i]) ** 2) /
                   (2 * sum1) - 0.5)
            kld = np.where(np.isnan(kld), np.inf, kld)

            if option == 2 and i < prev_dims:
                kld[i] = np.max(kld) + 1  # exclude same-index match

            order[i] = int(np.argmin(kld))
    return order


def _m_transfer(prev_decs, this_decs, prev_dims, this_dims,
                n_transfer, order, option):
    """
    Transfer and transform decision variables from source to target
    (m_transfer1.m).

    Scales variables by range ratio and shifts by mean difference
    when source/target distributions don't overlap sufficiently.
    """
    prev_max, prev_min, m_prev = _reshape_stats(prev_decs, prev_dims)
    this_max, this_min, m_this = _reshape_stats(this_decs, this_dims)
    prev_max = prev_max.copy()
    prev_min = prev_min.copy()
    prev_range = prev_max - prev_min
    this_range = this_max - this_min

    n_transfer = int(min(n_transfer, len(prev_decs)))
    new_decs = this_decs[:n_transfer].copy()
    if n_transfer == 0 or this_dims == 0:
        return new_decs

    j = np.asarray(order, dtype=int)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if option != 0:
            # Cumulative in-place rescaling of the source bounds (order matters
            # when several target dims map onto the same source dim)
            for i in range(this_dims):
                jj = j[i]
                sc = this_range[i] / prev_range[jj]
                prev_max[jj] = (prev_max[jj] - m_prev[jj]) * sc + m_prev[jj]
                prev_min[jj] = (prev_min[jj] - m_prev[jj]) * sc + m_prev[jj]

            vals = ((prev_decs[:n_transfer][:, j] - m_prev[j]) *
                    (this_range[:this_dims] / prev_range[j]) + m_prev[j])
            keep = (((prev_min[j] <= this_max[:this_dims]) &
                     (prev_max[j] >= this_max[:this_dims]) &
                     (m_prev[j] <= this_max[:this_dims])) |
                    ((prev_max[j] >= this_min[:this_dims]) &
                     (prev_min[j] <= this_min[:this_dims]) &
                     (m_prev[j] >= this_min[:this_dims])))
        else:
            vals = prev_decs[:n_transfer][:, j].astype(float).copy()
            keep = np.zeros(this_dims, dtype=bool)

        shift = ~keep
        vals[:, shift] += (m_this[:this_dims] - m_prev[j])[shift]

    new_decs[:, :this_dims] = np.clip(np.nan_to_num(vals, nan=0.0), 0, 1)
    return new_decs


def _gen_ga(pop_decs, mu, mum):
    """Standard GA generation: SBX crossover + polynomial mutation."""
    n_pop, D = pop_decs.shape
    perm = np.random.permutation(n_pop)
    # MATLAB: for i = 1:ceil(N/2) with p2 = indorder(i + fix(N/2))
    n_pairs = -(-n_pop // 2)
    off = np.zeros((n_pairs * 2, D))

    for i in range(n_pairs):
        p1 = perm[i]
        p2 = perm[i + n_pop // 2]
        c1, c2 = crossover(pop_decs[p1], pop_decs[p2], mu=mu)
        off[2 * i] = np.clip(mutation(c1, mu=mum), 0, 1)
        off[2 * i + 1] = np.clip(mutation(c2, mu=mum), 0, 1)
    return off


def _gen_de(pop_decs, F, CR):
    """Standard DE/rand/1/bin with random boundary repair."""
    n_pop, D = pop_decs.shape
    off = np.zeros_like(pop_decs)

    for i in range(n_pop):
        indices = np.arange(n_pop)
        indices = indices[indices != i]
        a, b, c = np.random.choice(indices, 3, replace=False)

        v = pop_decs[a] + F * (pop_decs[b] - pop_decs[c])
        # Binomial crossover
        u = pop_decs[i].copy()
        j_rand = np.random.randint(D)
        mask = np.random.rand(D) < CR
        mask[j_rand] = True
        u[mask] = v[mask]

        # Random boundary repair
        rand_dec = np.random.rand(D)
        u[u > 1] = rand_dec[u > 1]
        u[u < 0] = rand_dec[u < 0]
        off[i] = u
    return off


def _gen_ga1(pop_decs, temp_decs, mu, mum):
    """Cross-population GA: SBX between temp and original, then mutation."""
    n_pop, D = pop_decs.shape
    perm = np.random.permutation(n_pop)
    off = np.zeros((n_pop, D))

    for i in range(n_pop):
        p1 = perm[i]
        c1, _ = crossover(temp_decs[i], pop_decs[p1], mu=mu)
        off[i] = np.clip(mutation(c1, mu=mum), 0, 1)
    return off


def _gen_de1(pop_decs, temp_decs, F, CR):
    """Cross-population DE: mutation from original, crossover with temp."""
    n_pop, D = pop_decs.shape
    off = np.zeros_like(pop_decs)

    for i in range(n_pop):
        indices = np.arange(n_pop)
        indices = indices[indices != i]
        a, b, c = np.random.choice(indices, 3, replace=False)

        v = pop_decs[a] + F * (pop_decs[b] - pop_decs[c])
        # Binomial crossover with temp (not original)
        u = temp_decs[i].copy()
        j_rand = np.random.randint(D)
        mask = np.random.rand(D) < CR
        mask[j_rand] = True
        u[mask] = v[mask]

        rand_dec = np.random.rand(D)
        u[u > 1] = rand_dec[u > 1]
        u[u < 0] = rand_dec[u < 0]
        off[i] = u
    return off

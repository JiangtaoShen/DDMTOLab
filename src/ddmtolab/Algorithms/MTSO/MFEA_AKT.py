"""
Multifactorial Evolutionary Algorithm with Adaptive Knowledge Transfer (MFEA-AKT)

This module implements MFEA-AKT for multi-task optimization with adaptive crossover
operator selection for inter-task knowledge transfer.

References
----------
    [1] Zhou, Lei, et al. "Toward Adaptive Knowledge Transfer in Multifactorial
        Evolutionary Computation." IEEE Transactions on Cybernetics, 51(5):
        2563-2576, 2021.

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


class MFEA_AKT:
    """
    Multifactorial Evolutionary Algorithm with Adaptive Knowledge Transfer.

    Extends MFEA with 6 crossover operators for inter-task transfer and an
    adaptive mechanism to select the best operator based on improvement tracking.

    The 6 crossover operators are:
        0: Two-point crossover
        1: Uniform crossover
        2: Arithmetical crossover (r=0.25)
        3: Geometric crossover (r=0.2)
        4: BLX-alpha crossover (a=0.3)
        5: SBX crossover

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

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, gap=20, muc=2, mum=5,
                 save_data=True, save_path='./Data', name='MFEA-AKT', disable_tqdm=True):
        """
        Initialize MFEA-AKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.3)
        gap : int, optional
            History window size for operator selection fallback (default: 20)
        muc : float, optional
            Distribution index for SBX crossover (default: 2)
        mum : float, optional
            Distribution index for polynomial mutation (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-AKT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.gap = gap
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MFEA-AKT algorithm.

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
        pop_size = n * nt  # total population size

        # Initialize population and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified space
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons,
                                            type='uni', padding='mid')
        pop_objs = objs
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        # Individual_AKT bookkeeping: CXFactor = randi(6), isTran = 0, parNum = 0.
        # parNum is stored 0-based; -1 stands for MATLAB's "parNum == 0" (untracked).
        pop_cxf = [np.random.randint(0, 6, size=(n,)) for _ in range(nt)]
        pop_parnum = [np.full(n, -1, dtype=int) for _ in range(nt)]

        # Record of best CX factor per generation (for fallback selection)
        cfb_record = []

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        # MToP increments Algo.Gen inside notTerminated before the first loop body,
        # so the first generation executed by the reference runs with Gen == 2.
        gen = 2
        while nfes < max_nfes:
            # Merge populations from all tasks into single arrays
            m_decs, m_objs, m_cons, m_sfs = vstack_groups(
                pop_decs, pop_objs, pop_cons, pop_sfs)
            m_cxf = np.concatenate(pop_cxf)  # (pop_size,)
            m_parnum = np.concatenate(pop_parnum)  # (pop_size,)

            # --- Generation ---
            off_decs = np.zeros_like(m_decs)
            off_objs = np.full_like(m_objs, np.inf)
            off_cons = np.zeros_like(m_cons)
            off_sfs = np.zeros_like(m_sfs)
            off_cxf = np.zeros(pop_size, dtype=int)
            off_parnum = np.full(pop_size, -1, dtype=int)

            indorder = np.random.permutation(pop_size)
            n_pairs = pop_size // 2

            for pair in range(n_pairs):
                # MToP pairs indorder(i) with indorder(i + fix(len/2))
                p1 = indorder[pair]
                p2 = indorder[pair + n_pairs]
                sf1 = m_sfs[p1].item()
                sf2 = m_sfs[p2].item()
                i1, i2 = 2 * pair, 2 * pair + 1

                # MToP starts from `offspring(count) = population(p1)`, so CXFactor
                # and parNum are inherited from the copied parent unless overwritten.
                off_cxf[i1], off_cxf[i2] = m_cxf[p1], m_cxf[p2]
                off_parnum[i1], off_parnum[i2] = m_parnum[p1], m_parnum[p2]

                if sf1 == sf2 or np.random.rand() < self.rmp:
                    p = (p1, p2)
                    if sf1 == sf2:
                        # Same task: SBX crossover, isTran = 0
                        off_decs[i1], off_decs[i2] = crossover(
                            m_decs[p1], m_decs[p2], mu=self.muc)
                        off_cxf[i1] = m_cxf[p1]
                        off_cxf[i2] = m_cxf[p2]
                        is_tran = False
                    else:
                        # Different tasks: hyberCX with adaptive operator, isTran = 1
                        alpha = m_cxf[p[np.random.randint(2)]]
                        off_decs[i1], off_decs[i2] = _hyber_crossover(
                            m_decs[p1], m_decs[p2], alpha, self.muc)
                        off_cxf[i1] = alpha
                        off_cxf[i2] = alpha
                        is_tran = True

                    # Task imitation (random parent's task); parNum only refreshed
                    # for transferred offspring, matching `if isTran == 1`.
                    for k in (i1, i2):
                        rand_p = p[np.random.randint(2)]
                        off_sfs[k] = m_sfs[rand_p]
                        if is_tran:
                            off_parnum[k] = rand_p
                else:
                    # No transfer: mutation (CXFactor/parNum stay inherited)
                    off_decs[i1] = mutation(m_decs[p1], mu=self.mum)
                    off_decs[i2] = mutation(m_decs[p2], mu=self.mum)
                    off_sfs[i1] = sf1
                    off_sfs[i2] = sf2

                # Clip to [0, 1]
                off_decs[i1] = np.clip(off_decs[i1], 0, 1)
                off_decs[i2] = np.clip(off_decs[i2], 0, 1)

            # --- Evaluation ---
            for idx in range(pop_size):
                t = off_sfs[idx].item()
                dec_trimmed = off_decs[idx, :dims[t]]
                off_objs[idx], off_cons[idx] = evaluation_single(
                    problem, dec_trimmed, t)

            nfes += pop_size
            pbar.update(pop_size)

            # --- Calculate best CXFactor ---
            # MToP seeds imp_num with zeros, so only strictly positive relative
            # improvements are ever recorded and `any(imp_num)` decides the branch.
            tracked = off_parnum >= 0
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_imp = np.zeros(pop_size)
                if np.any(tracked):
                    pfc = m_objs[off_parnum[tracked], 0]
                    cfc = off_objs[tracked, 0]
                    rel_imp[tracked] = (pfc - cfc) / pfc

            imp_num = np.zeros(6)
            for idx in np.flatnonzero(tracked):
                cx = off_cxf[idx]
                if rel_imp[idx] > imp_num[cx]:
                    imp_num[cx] = rel_imp[idx]

            if np.any(imp_num):
                # Best operator is the one with highest max improvement
                max_idx = int(np.argmax(imp_num))
            else:
                # Fallback: most frequent best operator over the last `gap` records.
                # MToP takes max() of an all-zero counter when the window is empty,
                # which resolves to the first operator.
                if len(cfb_record) > 0:
                    start = max(0, len(cfb_record) - self.gap)
                    recent = cfb_record[start:]
                    counts = np.bincount(recent, minlength=6)
                    max_idx = int(np.argmax(counts))
                else:
                    max_idx = 0

            cfb_record.append(max_idx)

            # --- Adaptive CXFactor update ---
            for idx in range(pop_size):
                if tracked[idx]:
                    if rel_imp[idx] < 0:
                        # Offspring worsened -> adopt best operator
                        off_cxf[idx] = max_idx
                else:
                    # Untracked offspring: 50% best operator, 50% random operator
                    off_cxf[idx] = (max_idx if np.random.randint(2) == 0
                                    else np.random.randint(0, 6))

            # --- Selection: merge parents + offspring, keep best n per task ---
            merged_decs = np.vstack([m_decs, off_decs])
            merged_objs = np.vstack([m_objs, off_objs])
            merged_cons = np.vstack([m_cons, off_cons])
            merged_sfs = np.vstack([m_sfs, off_sfs])
            merged_cxf = np.concatenate([m_cxf, off_cxf])
            merged_parnum = np.concatenate([m_parnum, off_parnum])

            pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []
            pop_cxf, pop_parnum = [], []
            for t in range(nt):
                indices = np.where(merged_sfs.flatten() == t)[0]
                t_decs, t_objs, t_cons = select_by_index(
                    indices, merged_decs, merged_objs, merged_cons)
                t_cxf = merged_cxf[indices]
                t_parnum = merged_parnum[indices]

                sel = selection_elit(objs=t_objs, n=n, cons=t_cons)
                pop_decs.append(t_decs[sel])
                pop_objs.append(t_objs[sel])
                pop_cons.append(t_cons[sel])
                pop_sfs.append(np.full((n, 1), t))
                pop_cxf.append(t_cxf[sel])
                pop_parnum.append(t_parnum[sel])

            # Record history (transform back to real space for storage)
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs, all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results


# ============================================================
# Crossover operators for knowledge transfer
# ============================================================

def _hyber_crossover(par1, par2, alpha, muc=2):
    """
    Apply one of 6 crossover operators based on alpha.

    Parameters
    ----------
    par1, par2 : np.ndarray
        Parent decision vectors, shape (d,)
    alpha : int
        Crossover operator index (0-5)
    muc : float
        Distribution index for SBX (operator 5)

    Returns
    -------
    off1, off2 : np.ndarray
        Offspring decision vectors, shape (d,)
    """
    if alpha == 0:
        off1 = _tp_crossover(par1, par2)
        off2 = _tp_crossover(par2, par1)
    elif alpha == 1:
        off1 = _uf_crossover(par1, par2)
        off2 = _uf_crossover(par2, par1)
    elif alpha == 2:
        off1 = _ari_crossover(par1, par2)
        off2 = _ari_crossover(par2, par1)
    elif alpha == 3:
        off1 = _geo_crossover(par1, par2)
        off2 = _geo_crossover(par2, par1)
    elif alpha == 4:
        off1 = _blxa_crossover(par1, par2, a=0.3)
        off2 = _blxa_crossover(par2, par1, a=0.3)
    else:  # alpha == 5
        off1, off2 = crossover(par1, par2, mu=muc)
    return off1, off2


def _tp_crossover(par1, par2):
    """Two-point crossover."""
    d = len(par1)
    i, j = sorted(np.random.randint(0, d, size=2))
    off = par1.copy()
    off[i:j + 1] = par2[i:j + 1]
    return off


def _uf_crossover(par1, par2):
    """Uniform crossover."""
    mask = np.random.randint(0, 2, size=len(par1)).astype(bool)
    off = par1.copy()
    off[mask] = par2[mask]
    return off


def _ari_crossover(par1, par2, r=0.25):
    """Arithmetical crossover with ratio r."""
    return r * par1 + (1 - r) * par2


def _geo_crossover(par1, par2, r=0.2):
    """Geometric crossover with ratio r (MToP: ParDec1^r * ParDec2^(1-r))."""
    return np.power(par1, r) * np.power(par2, 1 - r)


def _blxa_crossover(par1, par2, a=0.3):
    """BLX-alpha crossover."""
    cmin = np.minimum(par1, par2)
    cmax = np.maximum(par1, par2)
    interval = cmax - cmin
    low = cmin - interval * a
    high = cmax + interval * a
    return low + (high - low) * np.random.rand(len(par1))

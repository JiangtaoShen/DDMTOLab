"""
Multifactorial Evolutionary Algorithm Based on Diffusion Gradient Descent (MFEA-DGD)

This module implements MFEA-DGD for multi-task optimization using gradient
estimation via finite differences with random orthogonal directions.

References
----------
    [1] Liu, Zhaobo, et al. "Multifactorial Evolutionary Algorithm Based on
        Diffusion Gradient Descent." IEEE Transactions on Cybernetics,
        1-13, 2023.

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


class MFEA_DGD:
    """
    Multifactorial Evolutionary Algorithm Based on Diffusion Gradient Descent.

    Uses gradient estimation via random finite differences to guide crossover
    and mutation operators:
    - Random perturbation direction from Gaussian distribution
    - Finite-difference gradient estimation per parent pair
    - Gradient-guided blend crossover with opposition-based learning (OBL)
    - Gradient descent mutation for non-transfer offspring
    - Adaptive sigma randomly selected each generation

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements

    Notes
    -----
    One generation costs ``4 * n * n_tasks + 2`` evaluations: two probes per parent
    per pair for the finite-difference gradient (``2 * n * n_tasks``), the main
    offspring (``n * n_tasks``), and the surviving probe buffer that the reference
    implementation re-evaluates together with the offspring (``n * n_tasks + 2``).
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

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.7, gamma=0.1,
                 save_data=True, save_path='./Data', name='MFEA-DGD',
                 disable_tqdm=True):
        """
        Initialize MFEA-DGD algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.7)
        gamma : float, optional
            Smoothing factor for gradient norm tracking (default: 0.1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-DGD')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.gamma = gamma
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MFEA-DGD algorithm.

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
        pop_size = n * nt

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

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        # MToP increments Algo.Gen inside notTerminated before the first loop body,
        # so the first generation executed by the reference runs with Gen == 2.
        gen = 2
        while nfes < max_nfes:
            # Random sigma from {10^-1, ..., 10^-5}
            sigma = 10.0 ** (-np.random.randint(1, 6))

            # Compute per-task decision variable bounds
            max_dec = []
            min_dec = []
            for t in range(nt):
                max_dec.append(np.max(pop_decs[t], axis=0))
                min_dec.append(np.min(pop_decs[t], axis=0))

            # Merge populations
            m_decs, m_objs, m_cons, m_sfs = vstack_groups(
                pop_decs, pop_objs, pop_cons, pop_sfs)

            maxD = m_decs.shape[1]
            n_obj = m_objs.shape[1]
            n_con = m_cons.shape[1]
            n_pairs = pop_size // 2

            # Preallocate main offspring
            main_off_decs = np.zeros((pop_size, maxD))
            main_off_sfs = np.zeros((pop_size, 1), dtype=int)

            # MToP's `offspring2` buffer. Pair i writes slots (2i, 2i+1, 2i+2, 2i+3)
            # while the loop counter advances by 2, so every pair overwrites the two
            # trailing slots of its predecessor. Exactly 2*n_pairs + 2 probes survive.
            probe_decs = np.zeros((2 * n_pairs + 2, maxD))
            probe_sfs = np.zeros((2 * n_pairs + 2, 1), dtype=int)

            shuffled = np.random.permutation(pop_size)

            # MToP re-initialises L at the start of every Generation call
            L = 0.0

            for pair_idx in range(n_pairs):
                p1 = shuffled[pair_idx]
                p2 = shuffled[pair_idx + n_pairs]
                sf1 = m_sfs[p1].item()
                sf2 = m_sfs[p2].item()
                base = 2 * pair_idx

                # Drawn once per pair in MToP
                k = 0.7 + 0.6 * np.random.rand()

                # RandOrthMat(D, 1) returns the raw (un-normalised) Gaussian column
                sd = np.random.randn(maxD)

                # --- Gradient estimation via finite differences ---
                QWE = np.zeros((2, maxD))
                parents = (p1, p2)
                factors = (sf1, sf2)

                # offspring2(count .. count+3) are copies of p1, p2, p1, p2, so the
                # skill factors follow the copied individuals even though the decision
                # vectors are later overwritten by the probes of a single parent.
                probe_sfs[base + 0] = sf1
                probe_sfs[base + 1] = sf2
                probe_sfs[base + 2] = sf1
                probe_sfs[base + 3] = sf2

                for x in range(2):
                    pidx = parents[x]
                    ft = factors[x]

                    probe_pos = m_decs[pidx] + sd * sigma
                    probe_neg = m_decs[pidx] - sd * sigma

                    # MToP evaluates the probes without clipping them back to [0, 1];
                    # clipping would flatten the finite difference at the box boundary.
                    obj_pos, _ = evaluation_single(problem, probe_pos[:dims[ft]], ft)
                    obj_neg, _ = evaluation_single(problem, probe_neg[:dims[ft]], ft)
                    nfes += 2
                    pbar.update(2)

                    # Finite-difference gradient estimate
                    L1 = obj_pos[0, 0] - obj_neg[0, 0]
                    QWE[x, :] = sd * L1 / sigma

                    probe_decs[base + 2 * x] = probe_pos
                    probe_decs[base + 2 * x + 1] = probe_neg

                # Update smoothed gradient norm L.
                # MATLAB norm() of a matrix is the spectral norm (largest singular
                # value), not the Frobenius norm.
                qwe_norm = np.linalg.norm(QWE, 2)
                if qwe_norm > L:
                    L = (1 - self.gamma) * qwe_norm + self.gamma * L

                # Avoid division by zero
                L_safe = max(L, 1e-15)

                idx1 = base
                idx2 = base + 1

                if sf1 == sf2 or np.random.rand() < self.rmp:
                    # --- Transfer: gradient-guided crossover + OBL ---
                    r1 = np.random.randint(2)
                    r2 = 1 - r1
                    factor = factors[np.random.randint(2)]

                    # Gradient-guided blend crossover
                    off_dec1 = _dgd_crossover(
                        m_decs[parents[r1]], m_decs[parents[r2]],
                        QWE, L_safe, sigma, maxD)

                    main_off_decs[idx1] = off_dec1

                    # MToP: `if rand() > mod(Algo.Gen, 2)` selects plain OBL on even
                    # generations and bound-reflected opposition on odd generations.
                    if gen % 2 == 0:
                        main_off_decs[idx2] = 1.0 - off_dec1
                    else:
                        main_off_decs[idx2] = (
                            k * (max_dec[factor] + min_dec[factor]) - off_dec1)

                    # Task imitation: random parent's MFFactor
                    main_off_sfs[idx1] = factors[np.random.randint(2)]
                    main_off_sfs[idx2] = factors[np.random.randint(2)]
                else:
                    # --- No transfer: gradient descent mutation ---
                    main_off_decs[idx1] = m_decs[p1] - QWE[0, :] * sigma / L_safe
                    main_off_decs[idx2] = m_decs[p2] - QWE[1, :] * sigma / L_safe

                    main_off_sfs[idx1] = sf1
                    main_off_sfs[idx2] = sf2

                # Clip to [0, 1] (MToP clips `offspring` only, never `offspring2`)
                main_off_decs[idx1] = np.clip(main_off_decs[idx1], 0, 1)
                main_off_decs[idx2] = np.clip(main_off_decs[idx2], 0, 1)

            # --- Evaluate main offspring ---
            main_off_objs = np.full((pop_size, n_obj), np.inf)
            main_off_cons = np.zeros((pop_size, n_con))
            for idx in range(pop_size):
                t = main_off_sfs[idx].item()
                main_off_objs[idx], main_off_cons[idx] = evaluation_single(
                    problem, main_off_decs[idx, :dims[t]], t)
            nfes += pop_size
            pbar.update(pop_size)

            # --- Re-evaluate the surviving probes on their own skill factor ---
            # MToP appends offspring2 to the per-task offspring list and evaluates
            # the whole list again, so each surviving probe costs a second FE.
            n_probes = probe_decs.shape[0]
            probe_objs = np.full((n_probes, n_obj), np.inf)
            probe_cons = np.zeros((n_probes, n_con))
            for idx in range(n_probes):
                t = probe_sfs[idx].item()
                probe_objs[idx], probe_cons[idx] = evaluation_single(
                    problem, probe_decs[idx, :dims[t]], t)
            nfes += n_probes
            pbar.update(n_probes)

            # --- Selection: merge parents + main offspring + probes ---
            merged_decs = np.vstack([m_decs, main_off_decs, probe_decs])
            merged_objs = np.vstack([m_objs, main_off_objs, probe_objs])
            merged_cons = np.vstack([m_cons, main_off_cons, probe_cons])
            merged_sfs = np.vstack([m_sfs, main_off_sfs, probe_sfs])

            pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []
            for t in range(nt):
                indices = np.where(merged_sfs.flatten() == t)[0]
                t_decs, t_objs, t_cons = select_by_index(
                    indices, merged_decs, merged_objs, merged_cons)
                sel = selection_elit(objs=t_objs, n=n, cons=t_cons)
                pop_decs.append(t_decs[sel])
                pop_objs.append(t_objs[sel])
                pop_cons.append(t_cons[sel])
                pop_sfs.append(np.full((n, 1), t))

            # Record history
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
# Operators
# ============================================================

def _dgd_crossover(par_dec1, par_dec2, QWE, L, sigma, D):
    """
    Gradient-guided blend crossover.

    Applies gradient descent step to both parents, then performs
    a random blend crossover.

    Parameters
    ----------
    par_dec1 : np.ndarray
        First parent (shape (D,)), already selected by random r1
    par_dec2 : np.ndarray
        Second parent (shape (D,)), already selected by random r2
    QWE : np.ndarray
        Gradient estimates, shape (2, D)
    L : float
        Smoothed gradient norm (> 0)
    sigma : float
        Perturbation step size
    D : int
        Decision vector dimensionality

    Returns
    -------
    off_dec : np.ndarray
        Offspring decision vector, shape (D,)
    """
    u = np.random.rand(D)
    cf = np.zeros(D)
    r1 = 0.6 * np.random.rand()
    r2 = -0.6 * np.random.rand()
    cf[u <= 0.5] = r1
    cf[u > 0.5] = r2

    # Gradient descent step on both parents
    p1 = par_dec1 - QWE[0, :] * sigma / L
    p2 = par_dec2 - QWE[1, :] * sigma / L

    # Blend crossover (only first offspring is used in MATLAB)
    off_dec = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
    return off_dec

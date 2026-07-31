"""
Block-Level Knowledge Transfer DE (BLKT-DE)

This module implements BLKT-DE for multi-task optimization using block-level
decision variable decomposition with k-means clustering for cross-task
knowledge transfer.

References
----------
    [1] Jiang, Yi, et al. "Block-Level Knowledge Transfer for Evolutionary
        Multitask Optimization." IEEE Transactions on Cybernetics, 1-14, 2023.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class BLKT_DE:
    """
    Block-Level Knowledge Transfer DE.

    Decomposes decision variables into fixed-size blocks, clusters them
    across all tasks and individuals using k-means, then performs DE/rand/1
    within each cluster for cross-task knowledge transfer. Combined with
    standard DE/rand/1/bin offspring per task.

    Adaptive block size (divD) and cluster count (divK) based on per-task
    improvement: full reset if all tasks stagnate, slight perturbation if
    some tasks stagnate, no change if all improve.

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

    def __init__(self, problem, n=None, max_nfes=None, F=0.5, CR=0.7,
                 save_data=True, save_path='./Data', name='BLKT-DE',
                 disable_tqdm=True):
        """
        Initialize BLKT-DE algorithm.

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
            DE crossover rate (default: 0.7)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'BLKT-DE')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.F = F
        self.CR = CR
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the BLKT-DE algorithm.

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

        # Initialize population and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Block parameters
        maxD = min(dims)
        divD = np.random.randint(1, maxD + 1)
        minK = 2
        maxK = n // 2
        divK = np.random.randint(minK, maxK + 1)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            # === Build correspondence vectors ===
            # Each entry: (task, individual, dim_start, dim_end)
            corre = []
            for t in range(nt):
                n_blocks = int(np.ceil(dims[t] / divD))
                for i in range(n):
                    for j in range(n_blocks):
                        d_start = j * divD
                        d_end = min((j + 1) * divD, dims[t])
                        corre.append((t, i, d_start, d_end))
            n_total_blocks = len(corre)

            # Extract block values (padded to divD)
            dim_val = np.zeros((n_total_blocks, divD))
            for idx in range(n_total_blocks):
                t, i, ds, de = corre[idx]
                dim_val[idx, :de - ds] = decs[t][i, ds:de]

            # === K-means clustering ===
            actual_k = max(2, min(divK, n_total_blocks))
            km = KMeans(n_clusters=actual_k, n_init=1, max_iter=100)
            labels = km.fit_predict(dim_val)

            # Group block indices by cluster
            subpops = [[] for _ in range(actual_k)]
            for idx in range(n_total_blocks):
                subpops[labels[idx]].append(idx)

            # === Block-level DE within clusters ===
            off_block_vals = {}  # block index -> new block values

            for k in range(actual_k):
                cluster = subpops[k]
                cs = len(cluster)
                if cs < 4:
                    continue
                for i_loc in range(cs):
                    # Select 3 distinct others from the cluster
                    candidates = list(range(cs))
                    candidates.remove(i_loc)
                    chosen = np.random.choice(candidates, 3, replace=False)
                    r1, r2, r3 = chosen

                    dp1 = dim_val[cluster[r1]]
                    dp2 = dim_val[cluster[r2]]
                    dp3 = dim_val[cluster[r3]]

                    # DE/rand/1 mutation
                    v = dp1 + self.F * (dp2 - dp3)
                    v = np.clip(v, 0, 1)

                    # Binomial crossover with original block
                    u = dim_val[cluster[i_loc]].copy()
                    j_rand = np.random.randint(divD)
                    mask = np.random.rand(divD) < self.CR
                    mask[j_rand] = True
                    u[mask] = v[mask]

                    off_block_vals[cluster[i_loc]] = u

            # === Reassemble offspring1 (block-transferred copy) ===
            offspring1_decs = [d.copy() for d in decs]
            for idx, new_val in off_block_vals.items():
                t, i, ds, de = corre[idx]
                block_len = de - ds
                offspring1_decs[t][i, ds:de] = new_val[:block_len]

            # === Per-task: standard DE + combine + evaluate + select ===
            succ_flags = [False] * nt

            for t in range(nt):
                # offspring2: standard DE/rand/1/bin
                off2_decs = de_generation(decs[t], self.F, self.CR)

                # Combine offspring1[t] and offspring2, randomly sample n
                combined_off = np.vstack([off2_decs, offspring1_decs[t]])
                perm = np.random.permutation(combined_off.shape[0])[:n]
                off_decs_t = np.clip(combined_off[perm], 0, 1)

                # Evaluate offspring
                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_t, t)
                nfes += n
                pbar.update(n)

                # Track best before selection. MToP's succ_flag is the Flag
                # returned by Algo.Evaluation, i.e. whether the newly evaluated
                # offspring beat the incumbent best under min_FP (feasibility
                # first, then objective). Because Selection_Elit is elitist and
                # uses the same order, that is exactly "the population best
                # improved" measured lexicographically on (CV, Obj).
                best_before = _best_key(objs[t], cons[t])

                # Elitist selection (parents + offspring)
                merged_decs = np.vstack([decs[t], off_decs_t])
                merged_objs = np.vstack([objs[t], off_objs_t])
                merged_cons = np.vstack([cons[t], off_cons_t])
                sel = selection_elit(objs=merged_objs, n=n, cons=merged_cons)
                decs[t] = merged_decs[sel]
                objs[t] = merged_objs[sel]
                cons[t] = merged_cons[sel]

                # Check improvement
                if _best_key(objs[t], cons[t]) < best_before:
                    succ_flags[t] = True

            # Record history
            append_history(all_decs, decs, all_objs, objs, all_cons, cons)

            # === Adaptive block parameters ===
            if all(not f for f in succ_flags):
                # All tasks stagnated: full random reset
                divD = np.random.randint(1, maxD + 1)
                divK = np.random.randint(minK, maxK + 1)
            elif any(not f for f in succ_flags):
                # Some tasks stagnated: slight perturbation (+-1).
                # MATLAB draws uniformly from {x-1, x, x+1} and *then* clamps,
                # so the clamped endpoint keeps the extra probability mass.
                divD = min(maxD, max(1, np.random.randint(divD - 1, divD + 2)))
                divK = min(maxK, max(minK, np.random.randint(divK - 1, divK + 2)))

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results


def _best_key(objs, cons):
    """
    Lexicographic (CV, Obj) key of the best individual, matching MToP's
    ``min_FP`` / ``sortrows([CVs, Objs], [1, 2])`` ordering.
    """
    n = objs.shape[0]
    cv = np.sum(np.maximum(0, cons), axis=1) if cons is not None \
        and cons.shape[1] > 0 else np.zeros(n)
    b = np.lexsort((objs[:, 0], cv))[0]
    return float(cv[b]), float(objs[b, 0])

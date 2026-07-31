"""
Meta-Knowledge Transfer-based Differential Evolution (MKTDE)

This module implements MKTDE for multi-task optimization using centroid-based
meta-knowledge transfer between tasks in a multi-population DE framework.

References
----------
    [1] Li, Jian-Yu, et al. "A Meta-Knowledge Transfer-Based Differential
        Evolution for Multitask Optimization." IEEE Transactions on
        Evolutionary Computation, 26(4): 719-734, 2022.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Date: 2026.02.22
Version: 2.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MKTDE:
    """
    Meta-Knowledge Transfer-based Differential Evolution.

    Uses centroid alignment to transform source task solutions into the
    target task's search distribution, creating an extended donor pool for
    DE/rand/1/bin. The base vector x1 is selected from the current task
    only, while difference vectors x2, x3 come from the combined pool.
    Additionally, an elite solution from the source task is transferred
    each generation.

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
                 save_data=True, save_path='./Data', name='MKTDE',
                 disable_tqdm=True):
        """
        Initialize MKTDE algorithm.

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
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MKTDE')
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
        Execute the MKTDE algorithm.

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

        # Initialize and evaluate in real space
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Convert decisions to the unified max-D space used by MToP. MToP
        # initializes the whole unified genome with rand(1, max(D)), so the
        # padding genes are U[0, 1] rather than a constant.
        pop_decs = space_transfer(problem=problem, decs=decs, type='uni', padding='random')
        pop_objs = objs
        # Constraints stay in each task's native width: MToP keeps one Con
        # vector per task and padding them to c_max would inflate the CV of
        # tasks that declare fewer constraints than the maximum.
        pop_cons = [c.copy() for c in cons]
        maxD = pop_decs[0].shape[1]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            # Compute centroids in unified space
            centroids = [np.mean(pop_decs[t], axis=0) for t in range(nt)]

            # Source task selection (random, different from t)
            source_tasks = []
            for t in range(nt):
                s = np.random.randint(nt)
                while s == t:
                    s = np.random.randint(nt)
                source_tasks.append(s)

            # --- Generation and selection per task ---
            for t in range(nt):
                s = source_tasks[t]
                ct, cs = centroids[t], centroids[s]

                # Meta-knowledge transfer: align source via centroids
                spop_transformed = pop_decs[s] - cs + ct

                # Combined donor pool: current task + transformed source
                popf = np.vstack([pop_decs[t], spop_transformed])
                n_combined = len(popf)

                # DE/rand/1/bin generation
                off_decs = np.zeros((n, maxD))
                for i in range(n):
                    # x1 from current task population only
                    x1 = np.random.randint(n)
                    while x1 == i:
                        x1 = np.random.randint(n)
                    # x2, x3 from combined pool
                    x2 = np.random.randint(n_combined)
                    while x2 == i or x2 == x1:
                        x2 = np.random.randint(n_combined)
                    x3 = np.random.randint(n_combined)
                    while x3 == i or x3 == x1 or x3 == x2:
                        x3 = np.random.randint(n_combined)

                    # DE/rand/1 mutation
                    v = pop_decs[t][x1] + self.F * (popf[x2] - popf[x3])

                    # Binomial crossover
                    u = pop_decs[t][i].copy()
                    j_rand = np.random.randint(maxD)
                    mask = np.random.rand(maxD) < self.CR
                    mask[j_rand] = True
                    u[mask] = v[mask]

                    off_decs[i] = np.clip(u, 0, 1)

                # Evaluate offspring (trim to task dimension)
                off_objs_t, off_cons_t = evaluation_single(
                    problem, off_decs[:, :dims[t]], t)
                nfes += n
                pbar.update(n)

                # Elitist selection (parents + offspring → best n)
                merged_decs = np.vstack([pop_decs[t], off_decs])
                merged_objs = np.vstack([pop_objs[t], off_objs_t])
                merged_cons = np.vstack([pop_cons[t], off_cons_t])
                sel = selection_elit(objs=merged_objs, n=n, cons=merged_cons)
                pop_decs[t] = merged_decs[sel]
                pop_objs[t] = merged_objs[sel]
                pop_cons[t] = merged_cons[sel]

            # --- Elite solution transfer ---
            for t in range(nt):
                s = source_tasks[t]
                # Replace last (worst) with first (best) from source task
                elite_dec = pop_decs[s][0].copy()

                elite_obj, elite_con = evaluation_single(
                    problem, elite_dec[:dims[t]].reshape(1, -1), t)
                nfes += 1
                pbar.update(1)

                pop_decs[t][-1] = elite_dec
                pop_objs[t][-1] = elite_obj.flatten()
                pop_cons[t][-1] = elite_con.flatten()

            # Record history in real space
            real_decs = space_transfer(problem, decs=pop_decs, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs,
                           all_cons, pop_cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

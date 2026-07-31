"""
Multi-task Evolutionary Algorithm Based on Decomposition with Transfer of Search Directions (MTEA-D-TSD)

This module implements MTEA-D-TSD for multi-task multi-objective optimization problems.

References
----------
    [1] Y. Li, W. Gong, and Q. Gu, "Transfer Search Directions Among Decomposed Subtasks for Evolutionary Multitasking in Multiobjective Optimization," in Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '24), 2024, pp. 557-565.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MTEA_D_TSD:
    """
    Multi-task Evolutionary Algorithm Based on Decomposition with Transfer of Search Directions.

    This algorithm features:
    - MOEA/D framework with Tchebycheff decomposition
    - Search direction (SD) tracking for each individual
    - Cross-task transfer of search directions based on cosine similarity
    - Adaptive per-individual transfer rate
    - DE/rand/1 mutation with polynomial mutation

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

    def __init__(self, problem, n=None, max_nfes=None,
                 TR0=0.2, CF=0.4, SNum=10, Delta=0.9, NR=2,
                 F=0.5, CR=0.9, MuM=15,
                 save_data=True, save_path='./Data',
                 name='MTEA-D-TSD', disable_tqdm=True):
        """
        Initialize MTEA-D-TSD algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        TR0 : float, optional
            Initial transfer rate (default: 0.2)
        CF : float, optional
            Cumulative factor for search direction update (default: 0.4)
        SNum : int, optional
            Number of random samples for source selection (default: 10)
        Delta : float, optional
            Probability of choosing parents from local neighborhood (default: 0.9)
        NR : int, optional
            Maximum number of solutions replaced per offspring (default: 2)
        F : float, optional
            DE mutation factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.9)
        MuM : float, optional
            PM mutation distribution index (default: 15)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTEA-D-TSD')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.TR0 = TR0
        self.CF = CF
        self.SNum = SNum
        self.Delta = Delta
        self.NR = NR
        self.F = F
        self.CR = CR
        self.MuM = MuM
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTEA-D-TSD algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        no = problem.n_objs
        dims = problem.dims
        d_max = max(dims)
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate weight vectors and neighborhoods
        W = []
        Nt = []
        DT = []
        B = []

        for t in range(nt):
            w_t, n_t = uniform_point(n_per_task[t], no[t])
            W.append(w_t)
            Nt.append(n_t)
            n_per_task[t] = n_t

            dt = int(np.ceil(n_t / 10))
            DT.append(dt)

            distances = squareform(pdist(w_t))
            neighbors = np.argsort(distances, axis=1, kind='stable')[:, :dt]
            B.append(neighbors)

        # Initialize populations in unified d_max space and evaluate
        decs = []
        objs = []
        cons = []
        for t in range(nt):
            decs_t = np.random.rand(Nt[t], d_max)
            objs_t, cons_t = evaluation_single(problem, decs_t[:, :dims[t]], t)
            decs.append(decs_t)
            objs.append(objs_t)
            cons.append(cons_t)
        nfes_per_task = Nt.copy()
        nfes = sum(Nt)

        # Native-space history
        all_decs = [[d[:, :dims[t]].copy()] for t, d in enumerate(decs)]
        all_objs = [[o.copy()] for o in objs]
        all_cons = [[c.copy()] for c in cons]

        # Ideal points
        Z = [np.min(objs[t], axis=0) for t in range(nt)]

        # Search directions (d_max) and transfer rates per individual
        SD = [np.zeros((Nt[t], d_max)) for t in range(nt)]
        TR = [np.full(Nt[t], self.TR0) for t in range(nt)]

        # Transfer flag and SD neighborhoods
        trans_flag = False
        SD_B = None
        RD_B = None

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=nfes,
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Main loop: one shared evaluation budget across all tasks, as in MToP
        while nfes < total_nfes:
            # Start transfer after 10% budget consumed
            if not trans_flag and nfes > 0.1 * total_nfes and self.TR0 > 0:
                SD_B = []
                RD_B = []
                for t in range(nt):
                    sd_b_t = []
                    rd_b_t = []
                    for i in range(Nt[t]):
                        sd_b_i = np.zeros((DT[t], 2), dtype=int)
                        rd_b_i = np.ones(DT[t])
                        for j in range(DT[t]):
                            k, jj = self._source_select(
                                SD[t][i], SD, nt, Nt)
                            sd_b_i[j] = [k, jj]
                        sd_b_t.append(sd_b_i)
                        rd_b_t.append(rd_b_i)
                    SD_B.append(sd_b_t)
                    RD_B.append(rd_b_t)
                trans_flag = True

            # Snapshot old population
            old_decs = [d.copy() for d in decs]

            for t in np.random.permutation(nt):
                for i in np.random.permutation(Nt[t]):
                    # Select neighborhood
                    PL = B[t][i][np.random.permutation(DT[t])]
                    PG = np.random.permutation(Nt[t])
                    P = PL if np.random.rand() < self.Delta else PG

                    # MToP builds the offspring by copying the whole individual i,
                    # so it carries i's search direction and transfer rate along
                    # to every subproblem it later replaces
                    off_sd = SD[t][i].copy()
                    off_tr = TR[t][i]

                    flag = False
                    if trans_flag and np.random.rand() < TR[t][i]:
                        flag = True
                        # Search-direction transfer
                        sd_idx = np.random.randint(DT[t])
                        k, j = SD_B[t][i][sd_idx]

                        # Scale source SD to match target SD magnitude
                        src_sd = SD[k][j]
                        src_norm = np.linalg.norm(src_sd)
                        tgt_norm = np.linalg.norm(SD[t][i])
                        if src_norm > 0.0:
                            sd = src_sd / src_norm * tgt_norm
                        else:
                            # MATLAB would produce NaN here; a null direction keeps
                            # the offspring equal to its parent instead.
                            sd = np.zeros(d_max)

                        off_dec = decs[t][i] + 2 * np.random.rand() * sd
                        off_dec = np.clip(off_dec, 0, 1)
                    else:
                        # DE/rand/1 + DE crossover + PM mutation
                        if np.random.rand() < 0.5:
                            x1, x2, x3 = decs[t][i], decs[t][P[0]], decs[t][P[1]]
                        else:
                            x1, x2, x3 = decs[t][i], decs[t][P[0]], old_decs[t][P[1]]

                        # DE mutation
                        off_dec = x1 + self.F * (x2 - x3)
                        # DE binomial crossover in d_max space (MToP DE_Crossover)
                        replace = np.random.rand(d_max) > self.CR
                        replace[np.random.randint(d_max)] = False
                        off_dec = np.where(replace, decs[t][i], off_dec)
                        # PM mutation
                        off_dec = mutation(off_dec, mu=self.MuM)
                        off_dec = np.clip(off_dec, 0, 1)

                    # Evaluate (trim to task dim)
                    off_obj, off_con = evaluation_single(
                        problem, off_dec[:dims[t]].reshape(1, -1), t)
                    nfes_per_task[t] += 1
                    nfes += 1
                    pbar.update(1)

                    # Update ideal point
                    Z[t] = np.minimum(Z[t], off_obj[0])

                    # Tchebycheff selection
                    g_old = np.max(np.abs(objs[t][P] - Z[t]) * W[t][P], axis=1)
                    g_new_val = np.max(np.abs(off_obj[0] - Z[t]) * W[t][P], axis=1)

                    # Constraint violation
                    CV_off = np.sum(np.maximum(0, off_con[0]))
                    CV_pop = np.sum(np.maximum(0, cons[t][P]), axis=1)

                    # Replace where better: (same CV & better Tchebycheff) or less CV
                    replace_mask = ((g_old >= g_new_val) & (CV_pop == CV_off)) | (CV_pop > CV_off)
                    replace_idx = P[replace_mask][:self.NR]

                    if len(replace_idx) > 0:
                        decs[t][replace_idx] = off_dec
                        objs[t][replace_idx] = off_obj[0]
                        cons[t][replace_idx] = off_con[0]
                        SD[t][replace_idx] = off_sd
                        TR[t][replace_idx] = off_tr

                    # Update TR and SD neighborhoods
                    if flag and len(replace_idx) > 0:
                        TR[t][i] = min(0.5, TR[t][i] * 1.1)
                        RD_B[t][i][sd_idx] += 1
                    elif flag and len(replace_idx) == 0:
                        TR[t][i] = TR[t][i] * 0.9 + self.TR0 / 2 * 0.1
                        RD_B[t][i][sd_idx] -= 1
                        if RD_B[t][i][sd_idx] <= 0:
                            k_new, jj_new = self._source_select(
                                SD[t][i], SD, nt, Nt)
                            SD_B[t][i][sd_idx] = [k_new, jj_new]
                            RD_B[t][i][sd_idx] = 1

            # Update search directions: SD = CF*SD + (1-CF)*variation
            for t in range(nt):
                for i in range(Nt[t]):
                    variation = decs[t][i] - old_decs[t][i]
                    if np.any(variation != 0):
                        SD[t][i] = self.CF * SD[t][i] + (1 - self.CF) * variation

            # Record the maintained populations of every task once per generation
            append_history(all_decs, [decs[t][:, :dims[t]] for t in range(nt)],
                           all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    def _source_select(self, self_sd, SD, nt, Nt):
        """
        Select source search-direction neighbor based on cosine similarity.

        Randomly samples SNum candidates from all tasks and selects the one
        with the highest cosine similarity to self_sd.

        Parameters
        ----------
        self_sd : np.ndarray
            Search direction of the target individual
        SD : list of np.ndarray
            Search directions for all tasks
        nt : int
            Number of tasks
        Nt : list of int
            Population sizes per task

        Returns
        -------
        k : int
            Selected source task index
        jj : int
            Selected source individual index

        Notes
        -----
        A zero-norm search direction makes the cosine undefined; MATLAB stores a
        NaN and ``max`` silently skips it, which is reproduced here with
        ``np.nan`` plus ``nanargmax`` (falling back to the first sample when all
        candidates are undefined).
        """
        index = np.empty((self.SNum, 2), dtype=int)
        cosine_sim = np.empty(self.SNum)
        self_norm = np.linalg.norm(self_sd)

        for s in range(self.SNum):
            k = np.random.randint(nt)
            j = np.random.randint(Nt[k])
            index[s] = (k, j)
            src_norm = np.linalg.norm(SD[k][j])
            denom = self_norm * src_norm
            cosine_sim[s] = np.dot(self_sd, SD[k][j]) / denom if denom > 0.0 else np.nan

        # A perfectly aligned direction is the individual's own; discard it
        cosine_sim[cosine_sim == 1] = -1.0

        if np.all(np.isnan(cosine_sim)):
            idx = 0
        else:
            idx = int(np.nanargmax(cosine_sim))

        return int(index[idx, 0]), int(index[idx, 1])

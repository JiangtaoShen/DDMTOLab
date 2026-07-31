"""
Multi-task Evolutionary Algorithm via Diversity- and Convergence-Oriented Knowledge Transfer (MTEA-DCK)

This module implements MTEA-DCK for multi-task multi-objective optimization problems.

References
----------
    [1] Y. Li, D. Li, W. Gong, and Q. Gu, "Multiobjective Multitask Optimization via Diversity- and Convergence-Oriented Knowledge Transfer," IEEE Trans. Syst., Man, Cybern., Syst., vol. 55, no. 3, pp. 2367-2379, 2025.

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
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MTEA_DCK:
    """
    Multi-task Evolutionary Algorithm via Diversity- and Convergence-Oriented Knowledge Transfer.

    This algorithm features:
    - Competitive Swarm Optimizer (CSO) framework with winner/loser pairing
    - DE-based generation with diversified knowledge transfer (DKT) via region mapping
    - CSO-based generation with convergent knowledge transfer (CKT) via fragment swap
    - Adaptive per-individual parameters (F, CR, TRD) with Cauchy/Normal perturbation
    - SPEA2 environmental selection

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
                 Tau=0.1, TRC0=0.3,
                 save_data=True, save_path='./Data',
                 name='MTEA-DCK', disable_tqdm=True):
        """
        Initialize MTEA-DCK algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        Tau : float, optional
            Probability of random parameter reset (default: 0.1)
        TRC0 : float, optional
            Initial transfer rate for convergent knowledge transfer (default: 0.3)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTEA-DCK')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.Tau = Tau
        self.TRC0 = TRC0
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTEA-DCK algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d_max = max(dims)
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize populations in unified d_max space
        decs = []
        objs = []
        cons = []
        for t in range(nt):
            decs_t = np.random.rand(n_per_task[t], d_max)
            objs_t, cons_t = evaluation_single(problem, decs_t[:, :dims[t]], t)
            decs.append(decs_t)
            objs.append(objs_t)
            cons.append(cons_t)
        nfes_per_task = [n_per_task[t] for t in range(nt)]
        nfes = sum(nfes_per_task)

        # SPEA2 environmental selection on the initial population (a no-op on the
        # membership, but it fixes the fitness values and the population order)
        fitness = []
        vel = []
        F_param = []
        CR_param = []
        TRD_param = []
        for t in range(nt):
            idx, fit = self._spea2_select(objs[t], cons[t], n_per_task[t])
            decs[t] = decs[t][idx]
            objs[t] = objs[t][idx]
            cons[t] = cons[t][idx]
            fitness.append(fit)
            vel.append(np.zeros((n_per_task[t], d_max)))
            F_param.append(np.random.rand(n_per_task[t]))
            CR_param.append(np.random.rand(n_per_task[t]))
            TRD_param.append(np.random.rand(n_per_task[t]))

        # Native-space history
        all_decs = [[d[:, :dims[t]].copy()] for t, d in enumerate(decs)]
        all_objs = [[o.copy()] for o in objs]
        all_cons = [[c.copy()] for c in cons]

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=nfes,
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Main loop: one shared evaluation budget across all tasks, as in MToP
        while nfes < total_nfes:
            # Winner/Loser tournament for each task
            winners = []
            losers = []
            uni_upper = []
            uni_lower = []
            for t in range(nt):
                N_t = n_per_task[t]
                half = N_t // 2
                rnd_idx = np.random.permutation(N_t)
                lose_idx = rnd_idx[:half].copy()
                win_idx = rnd_idx[half:2 * half].copy()
                # Swap: ensure winner has better (lower) fitness
                swap_mask = fitness[t][win_idx] > fitness[t][lose_idx]
                temp = win_idx[swap_mask].copy()
                win_idx[swap_mask] = lose_idx[swap_mask]
                lose_idx[swap_mask] = temp
                winners.append(win_idx)
                losers.append(lose_idx)
                # Winner region bounds
                uni_upper.append(np.max(decs[t][win_idx], axis=0))
                uni_lower.append(np.min(decs[t][win_idx], axis=0))

            for t in range(nt):
                N_t = n_per_task[t]
                union_idx = np.concatenate([winners[t], losers[t]])

                # DE generation with diversified KT (from winners)
                off_de = self._generation_de(
                    decs, vel, F_param, CR_param, TRD_param,
                    winners, losers, union_idx, t, nt,
                    uni_upper, uni_lower)
                off_de_decs, off_de_V, off_de_F, off_de_CR, off_de_TRD = off_de

                # Evaluate DE offspring
                off_de_objs, off_de_cons = evaluation_single(
                    problem, off_de_decs[:, :dims[t]], t)
                nfes_per_task[t] += len(off_de_objs)
                nfes += len(off_de_objs)
                pbar.update(len(off_de_objs))

                # CSO generation with convergent KT (from losers). MToP reads the
                # evaluation counter here, i.e. after the DE offspring are spent.
                factor = nfes / total_nfes
                off_cso = self._generation_cso(
                    decs, vel, F_param, CR_param, TRD_param,
                    winners, losers, t, nt, factor)
                off_cso_decs, off_cso_V, off_cso_F, off_cso_CR, off_cso_TRD = off_cso

                # Evaluate CSO offspring
                off_cso_objs, off_cso_cons = evaluation_single(
                    problem, off_cso_decs[:, :dims[t]], t)
                nfes_per_task[t] += len(off_cso_objs)
                nfes += len(off_cso_objs)
                pbar.update(len(off_cso_objs))

                # Merge all populations
                merged_decs = np.vstack([decs[t], off_de_decs, off_cso_decs])
                merged_objs = np.vstack([objs[t], off_de_objs, off_cso_objs])
                merged_cons = np.vstack([cons[t], off_de_cons, off_cso_cons])
                merged_V = np.vstack([vel[t], off_de_V, off_cso_V])
                merged_F = np.concatenate([F_param[t], off_de_F, off_cso_F])
                merged_CR = np.concatenate([CR_param[t], off_de_CR, off_cso_CR])
                merged_TRD = np.concatenate([TRD_param[t], off_de_TRD, off_cso_TRD])

                # SPEA2 environmental selection. The fitness that drives the next
                # winner/loser pairing is the one computed on the merged
                # population, not a value recomputed on the survivors.
                idx, fit = self._spea2_select(merged_objs, merged_cons, N_t)
                decs[t] = merged_decs[idx]
                objs[t] = merged_objs[idx]
                cons[t] = merged_cons[idx]
                vel[t] = merged_V[idx]
                F_param[t] = merged_F[idx]
                CR_param[t] = merged_CR[idx]
                TRD_param[t] = merged_TRD[idx]
                fitness[t] = fit

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

    @staticmethod
    def _spea2_select(objs, cons, N):
        """
        SPEA2 environmental selection.

        Parameters
        ----------
        objs : np.ndarray
            Objective values, shape (pop_size, n_objs)
        cons : np.ndarray
            Constraint violations, shape (pop_size, n_cons)
        N : int
            Number of individuals to select

        Returns
        -------
        idx : np.ndarray
            Indices of the selected individuals, ordered by ascending fitness
        fit : np.ndarray
            Fitness of the selected individuals, computed on the full input
            population (never recomputed on the survivors)
        """
        fit = spea2_fitness(objs, cons)
        next_mask = fit < 1
        n_nd = int(np.sum(next_mask))
        if n_nd <= N:
            idx = np.argsort(fit, kind='stable')[:N]
        else:
            nd_indices = np.where(next_mask)[0]
            selected = spea2_truncation(objs[nd_indices], N)
            idx = nd_indices[selected]
        # Sort the surviving population by fitness (MToP Selection_SPEA2)
        idx = idx[np.argsort(fit[idx], kind='stable')]
        return idx, fit[idx]

    def _generation_de(self, decs, vel, F_param, CR_param, TRD_param,
                       winners, losers, union_idx, t, nt,
                       uni_upper, uni_lower):
        """
        DE generation with diversified knowledge transfer.

        For each winner individual, generates an offspring using DE/current-to-rand/1
        with optional cross-task diversified knowledge transfer via region mapping
        and particle reversal.

        Parameters
        ----------
        decs : list of np.ndarray
            Decision variables per task
        vel : list of np.ndarray
            Velocities per task
        F_param, CR_param, TRD_param : list of np.ndarray
            Adaptive parameters per task
        winners, losers : list of np.ndarray
            Winner and loser indices per task
        union_idx : np.ndarray
            Union of winner and loser indices for task t
        t : int
            Current task index
        nt : int
            Number of tasks
        uni_upper, uni_lower : list of np.ndarray
            Winner region bounds per task

        Returns
        -------
        tuple
            (off_decs, off_V, off_F, off_CR, off_TRD)
        """
        n_win = len(winners[t])
        d = decs[t].shape[1]
        task_pool = [k for k in range(nt) if k != t]

        off_decs = np.zeros((n_win, d))
        off_V = np.zeros((n_win, d))
        off_F = np.zeros(n_win)
        off_CR = np.zeros(n_win)
        off_TRD = np.zeros(n_win)

        for i in range(n_win):
            wi = winners[t][i]

            # Adaptive parameter perturbation
            new_F = self._cauchy_rand(F_param[t][wi], 0.1)
            while new_F <= 0:
                new_F = self._cauchy_rand(F_param[t][wi], 0.1)
            new_F = min(1.0, new_F)
            new_CR = np.clip(np.random.normal(CR_param[t][wi], 0.1), 0, 1)
            new_TRD = np.clip(np.random.normal(TRD_param[t][wi], 0.1), 0, 1)

            # Tau-based random reset
            if np.random.rand() < self.Tau:
                new_F = np.random.rand()
            if np.random.rand() < self.Tau:
                new_CR = np.random.rand()
            if np.random.rand() < self.Tau:
                new_TRD = np.random.rand()

            off_F[i] = new_F
            off_CR[i] = new_CR
            off_TRD[i] = new_TRD

            # Select donor indices
            x1 = np.random.randint(n_win)
            while x1 == i:
                x1 = np.random.randint(n_win)
            x2 = np.random.randint(n_win)
            while x2 == i or x2 == x1:
                x2 = np.random.randint(n_win)
            x3 = np.random.randint(len(union_idx))
            while x3 == i or x3 == x1 or x3 == x2:
                x3 = np.random.randint(len(union_idx))

            w_x1 = winners[t][x1]
            w_x2 = winners[t][x2]
            u_x3 = union_idx[x3]

            if np.random.rand() < new_TRD and len(task_pool) > 0:
                # Diversified Knowledge Transfer
                k = task_pool[np.random.randint(len(task_pool))]
                src_win_idx = x1 % len(winners[k])
                src_i = winners[k][src_win_idx]

                diver_dec = decs[k][src_i].copy()
                diver_v = vel[k][src_i].copy()

                # Particle reversal (50% chance)
                if np.random.rand() < 0.5:
                    diver_dec = (uni_lower[k] + uni_upper[k]) - diver_dec
                    diver_v = -diver_v

                # Region mapping: source winner region -> target winner region
                range_k = uni_upper[k] - uni_lower[k]
                range_t = uni_upper[t] - uni_lower[t]
                safe = range_k > 1e-20
                diver_dec_mapped = diver_dec.copy()
                diver_dec_mapped[safe] = (
                    (diver_dec[safe] - uni_lower[k][safe]) / range_k[safe]
                    * range_t[safe] + uni_lower[t][safe])
                diver_dec_mapped[~safe] = uni_lower[t][~safe]

                # Polynomial mutation on mapped dec
                diver_dec_mapped = mutation(diver_dec_mapped, mu=20)

                # Velocity mapping
                diver_v_mapped = diver_v.copy()
                diver_v_mapped[safe] = diver_v[safe] / range_k[safe] * range_t[safe]
                diver_v_mapped[~safe] = 0.0

                # DE mutation with diversified source
                off_dec = (diver_dec_mapped
                           + new_F * (decs[t][w_x1] - diver_dec_mapped)
                           + new_F * (decs[t][w_x2] - decs[t][u_x3]))
                off_v = (diver_v_mapped
                         + new_F * (vel[t][w_x1] - diver_v_mapped)
                         + new_F * (vel[t][w_x2] - vel[t][u_x3]))
            else:
                # Standard DE/current-to-rand/1
                off_dec = (decs[t][wi]
                           + new_F * (decs[t][w_x1] - decs[t][wi])
                           + new_F * (decs[t][w_x2] - decs[t][u_x3]))
                off_v = (vel[t][wi]
                         + new_F * (vel[t][w_x1] - vel[t][wi])
                         + new_F * (vel[t][w_x2] - vel[t][u_x3]))

            # DE binomial crossover on [Dec; V] together
            mutant = np.vstack([off_dec.reshape(1, -1), off_v.reshape(1, -1)])
            parent = np.vstack([decs[t][wi].reshape(1, -1), vel[t][wi].reshape(1, -1)])
            crossed = self._de_crossover(mutant, parent, new_CR)
            off_dec = crossed[0]
            off_v = crossed[1]

            # Boundary repair (only for Dec, using winner as parent)
            off_dec = self._boundary_repair(off_dec, decs[t][wi])

            off_decs[i] = off_dec
            off_V[i] = off_v

        return off_decs, off_V, off_F, off_CR, off_TRD

    def _generation_cso(self, decs, vel, F_param, CR_param, TRD_param,
                        winners, losers, t, nt, factor):
        """
        CSO generation with convergent knowledge transfer.

        For each loser individual, generates an offspring using velocity-based
        position update toward the paired winner, with optional cross-task
        convergent knowledge transfer via fragment swap.

        Parameters
        ----------
        decs : list of np.ndarray
            Decision variables per task
        vel : list of np.ndarray
            Velocities per task
        F_param, CR_param, TRD_param : list of np.ndarray
            Adaptive parameters per task
        winners, losers : list of np.ndarray
            Winner and loser indices per task
        t : int
            Current task index
        nt : int
            Number of tasks
        factor : float
            Progress ratio (sum_nfes / total_nfes)

        Returns
        -------
        tuple
            (off_decs, off_V, off_F, off_CR, off_TRD)
        """
        n_lose = len(losers[t])
        d = decs[t].shape[1]
        task_pool = [k for k in range(nt) if k != t]
        TRC = self.TRC0 * ((1 - factor) ** 2)

        off_decs = np.zeros((n_lose, d))
        off_V = np.zeros((n_lose, d))
        off_F = np.zeros(n_lose)
        off_CR = np.zeros(n_lose)
        off_TRD = np.zeros(n_lose)

        for i in range(n_lose):
            li = losers[t][i]
            wi = winners[t][i]

            # Parameter learning from winner
            off_F[i] = F_param[t][wi]
            off_CR[i] = CR_param[t][wi]
            off_TRD[i] = TRD_param[t][wi]

            # Velocity update
            if np.random.rand() < TRC and len(task_pool) > 0:
                # Convergent Knowledge Transfer
                k = task_pool[np.random.randint(len(task_pool))]
                src_win_idx = np.random.randint(len(winners[k]))
                conver_dec = decs[k][winners[k][src_win_idx]].copy()

                # Fragment swap: replace some dims with target winner
                swap_mask = np.random.rand(d) < np.random.rand()
                conver_dec[swap_mask] = decs[t][wi][swap_mask]

                new_v = (np.random.rand() * vel[t][li]
                         + np.random.rand() * (conver_dec - decs[t][li]))
            else:
                # Standard CSO velocity update toward winner
                new_v = (np.random.rand() * vel[t][li]
                         + np.random.rand() * (decs[t][wi] - decs[t][li]))

            # Position update
            off_dec = decs[t][li] + new_v

            # Boundary repair (using loser as parent)
            off_dec = self._boundary_repair(off_dec, decs[t][li])

            off_decs[i] = off_dec
            off_V[i] = new_v

        return off_decs, off_V, off_F, off_CR, off_TRD

    @staticmethod
    def _cauchy_rand(loc, scale):
        """Generate a Cauchy random number with given location and scale."""
        return loc + scale * np.random.standard_cauchy()

    @staticmethod
    def _de_crossover(mutant, parent, cr):
        """
        Binomial crossover for a 2xD matrix (Dec and V together).

        A single 1xD crossover mask is drawn and applied to both rows, so the
        position and the velocity of an offspring always inherit the same set of
        dimensions from the mutant (MToP ``DE_Crossover`` operates column-wise).

        Parameters
        ----------
        mutant : np.ndarray
            Mutant vectors, shape (2, D)
        parent : np.ndarray
            Parent vectors, shape (2, D)
        cr : float
            Crossover rate

        Returns
        -------
        np.ndarray
            Crossed-over vectors, shape (2, D)
        """
        d = mutant.shape[1]
        result = mutant.copy()
        replace = np.random.rand(d) > cr
        replace[np.random.randint(d)] = False
        result[:, replace] = parent[:, replace]
        return result

    @staticmethod
    def _boundary_repair(dec, parent_dec):
        """
        Random repair boundary handling.

        For lower boundary violations, replace with random value in [0, parent].
        For upper boundary violations, replace with random value in [parent, 1].

        Parameters
        ----------
        dec : np.ndarray
            Decision vector to repair
        parent_dec : np.ndarray
            Parent decision vector for repair reference

        Returns
        -------
        np.ndarray
            Repaired decision vector in [0, 1]
        """
        d = len(dec)
        vio_low = dec < 0
        if np.any(vio_low):
            rnd_lower = np.random.rand(d) * parent_dec
            dec[vio_low] = rnd_lower[vio_low]
        vio_up = dec > 1
        if np.any(vio_up):
            rnd_upper = parent_dec + np.random.rand(d) * (1 - parent_dec)
            dec[vio_up] = rnd_upper[vio_up]
        return np.clip(dec, 0, 1)

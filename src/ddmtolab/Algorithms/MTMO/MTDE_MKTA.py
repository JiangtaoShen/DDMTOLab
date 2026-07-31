"""
Multi-objective Multi-task Differential Evolution with Multiple Knowledge Types and Transfer Adaptation (MTDE-MKTA)

This module implements MTDE-MKTA for multi-task multi-objective optimization problems.

References
----------
    [1] Li, Yanchi, and Wenyin Gong. "Multiobjective Multitask Optimization With Multiple Knowledge Types and Transfer Adaptation." IEEE Transactions on Evolutionary Computation 29.1 (2025): 205-216.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.18
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MTDE_MKTA:
    """
    Multi-objective Multi-task Differential Evolution with Multiple Knowledge Types and Transfer Adaptation.

    This algorithm features:
    - Self-adaptive parameters (F, CR, TR, KP) for each individual
    - Rank-based DE parent selection
    - Two knowledge transfer types: direct transfer and distribution-based transfer

    Notes
    -----
    Following the MTO-Platform reference implementation, the whole population is
    evolved in a *unified* :math:`[0, 1]^{\\max_t D_t}` space; when an individual
    is evaluated on task ``t`` only its first :math:`D_t` genes are used. This is
    what makes a decision vector directly exchangeable between tasks of different
    dimensionality.

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

    def __init__(self, problem, n=None, max_nfes=None, tau1=0.2, tau2=0.1,
                 save_data=True, save_path='./Data', name='MTDE-MKTA', disable_tqdm=True):
        """
        Initialize MTDE-MKTA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        tau1 : float, optional
            Mutation probability for F and CR parameters (default: 0.2)
        tau2 : float, optional
            Mutation probability for TR and KP parameters (default: 0.1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MTDEMKTA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.tau1 = tau1
        self.tau2 = tau2
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTDE-MKTA algorithm.

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

        # Self-adaptive parameters carried by every individual
        params = []
        for t in range(nt):
            params.append({
                'F': 0.2 + np.random.rand(n_per_task[t]),   # F in [0.2, 1.2]
                'CR': np.random.rand(n_per_task[t]),        # CR in [0, 1]
                'TR': np.random.rand(n_per_task[t]),        # Knowledge transfer rate
                'KP': np.random.rand(n_per_task[t])         # Knowledge type proportion
            })

        # Initial SPEA2 environmental selection (sorts the population by fitness)
        # and the initial distribution models
        fitness, models = [], []
        for t in range(nt):
            decs[t], objs[t], cons[t], params[t], fit_t = self._selection_spea2(
                decs[t], objs[t], cons[t], params[t], n_per_task[t]
            )
            fitness.append(fit_t)
            models.append({
                'mean': np.mean(decs[t], axis=0),
                'std': np.std(decs[t], axis=0, ddof=1) + 1e-100
            })

        all_decs, all_objs, all_cons = init_history(
            [decs[t][:, :dims[t]] for t in range(nt)], objs, cons)

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            active_tasks = [t for t in range(nt) if nfes_per_task[t] < max_nfes_per_task[t]]
            if not active_tasks:
                break

            # Rank of every individual w.r.t. its SPEA2 fitness (1 = best), and
            # the exponentially smoothed per-task decision-space distribution.
            ranks = []
            alpha = 0.5
            for t in range(nt):
                order = np.argsort(fitness[t], kind='stable')
                rank_t = np.empty(len(fitness[t]), dtype=int)
                rank_t[order] = np.arange(1, len(fitness[t]) + 1)
                ranks.append(rank_t)

                models[t]['mean'] = alpha * models[t]['mean'] + (1 - alpha) * np.mean(decs[t], axis=0)
                models[t]['std'] = (alpha * models[t]['std']
                                    + (1 - alpha) * np.std(decs[t], axis=0, ddof=1) + 1e-100)

            # All offspring are generated from the *same* pre-update populations
            off_decs_all, off_params_all = [], []
            for t in active_tasks:
                off_decs_t, off_params_t = self._generation(decs, params, ranks, models, t)
                off_decs_all.append(off_decs_t)
                off_params_all.append(off_params_t)

            # Evaluate and select task by task
            for idx, t in enumerate(active_tasks):
                off_decs_t = off_decs_all[idx]
                off_params_t = off_params_all[idx]

                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_t[:, :dims[t]], t)
                nfes_per_task[t] += off_decs_t.shape[0]
                pbar.update(off_decs_t.shape[0])

                merged_decs = np.vstack([decs[t], off_decs_t])
                merged_objs = np.vstack([objs[t], off_objs_t])
                merged_cons = np.vstack([cons[t], off_cons_t])
                merged_params = {key: np.concatenate([params[t][key], off_params_t[key]])
                                 for key in ('F', 'CR', 'TR', 'KP')}

                decs[t], objs[t], cons[t], params[t], fitness[t] = self._selection_spea2(
                    merged_decs, merged_objs, merged_cons, merged_params, n_per_task[t]
                )

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

    def _generation(self, decs, params, ranks, models, t):
        """
        Generate offspring for task t using rank-based DE with knowledge transfer.

        Parameters
        ----------
        decs : list of np.ndarray
            Unified-space decision variables for all tasks
        params : list of dict
            Adaptive parameters for all tasks
        ranks : list of np.ndarray
            1-based ranks for all tasks (based on SPEA2 fitness, 1 = best)
        models : list of dict
            Distribution models for all tasks
        t : int
            Current task index

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables of shape (N, d_uni)
        off_params : dict
            Offspring adaptive parameters
        """
        pop = decs[t]
        Np, d = pop.shape
        nt = len(decs)

        off_decs = np.zeros((Np, d))
        off_params = {key: np.zeros(Np) for key in ('F', 'CR', 'TR', 'KP')}

        for i in range(Np):
            # Parameter disturbance with Gaussian noise
            off_F = float(np.clip(np.random.normal(params[t]['F'][i], 0.1), 0.2, 1.2))
            off_CR = float(np.clip(np.random.normal(params[t]['CR'][i], 0.1), 0.0, 1.0))
            off_TR = float(np.clip(np.random.normal(params[t]['TR'][i], 0.1), 0.0, 1.0))

            # Cyclic boundary for KP
            off_KP = float(np.random.normal(params[t]['KP'][i], 0.1))
            if off_KP < 0:
                off_KP = 1 + off_KP
            elif off_KP > 1:
                off_KP = off_KP - 1

            # Parameter mutation
            if np.random.rand() < self.tau1:
                off_F = 0.2 + np.random.rand()
            if np.random.rand() < self.tau1:
                off_CR = np.random.rand()
            if np.random.rand() < self.tau2:
                off_TR = np.random.rand()
            if np.random.rand() < self.tau2:
                off_KP = np.random.rand()

            off_params['F'][i] = off_F
            off_params['CR'][i] = off_CR
            off_params['TR'][i] = off_TR
            off_params['KP'][i] = off_KP

            # Select individuals (rank-DE): x1 and x2 are accepted with
            # probability (Np - rank) / Np, x3 is drawn uniformly.
            x1 = self._rank_selection(ranks[t], Np, (i,))
            x2 = self._rank_selection(ranks[t], Np, (i, x1))
            x3 = int(np.random.randint(Np))
            while x3 == i or x3 == x1 or x3 == x2:
                x3 = int(np.random.randint(Np))

            x_dec_i = pop[i]
            x_dec1 = pop[x1]
            x_dec2 = pop[x2]
            x_dec3 = pop[x3]

            # Knowledge transfer: replace the second difference vector member
            if np.random.rand() < off_TR:
                k = int(np.random.randint(nt))          # help task
                while k == t:
                    k = int(np.random.randint(nt))
                np_k = decs[k].shape[0]

                if off_KP > 0.5:
                    # Solution-level knowledge: use the source solution as it is
                    x_dec_k = decs[k][np.random.randint(np_k)]
                else:
                    # Distribution-level knowledge: standardise with the source
                    # model and map onto the target model
                    x_dec_k = decs[k][np.random.randint(np_k)]
                    x_dec_k = (x_dec_k - models[k]['mean']) / models[k]['std']
                    x_dec_k = models[t]['mean'] + models[t]['std'] * x_dec_k
                x_dec2 = x_dec_k

            # DE/rand/1 mutation + binomial crossover (MToP DE_Crossover)
            mutant = x_dec1 + off_F * (x_dec2 - x_dec3)
            replace = np.random.rand(d) > off_CR
            replace[np.random.randint(d)] = False
            child = np.where(replace, x_dec_i, mutant)

            off_decs[i] = np.clip(child, 0.0, 1.0)

        return off_decs, off_params

    @staticmethod
    def _rank_selection(rank, Np, exclude):
        """
        Rank-based selection: accept a uniformly drawn index with probability
        ``(Np - rank[x]) / Np`` and reject the excluded indices.

        Parameters
        ----------
        rank : np.ndarray
            1-based ranks of individuals (1 = best, Np = worst)
        Np : int
            Population size
        exclude : tuple of int
            Indices that must not be selected

        Returns
        -------
        selected : int
            Selected individual index
        """
        for _ in range(10000):
            x = int(np.random.randint(Np))
            if np.random.rand() <= (Np - rank[x]) / Np and x not in exclude:
                return x

        # Safety net for degenerate populations (MATLAB would spin forever here)
        candidates = [j for j in range(Np) if j not in exclude]
        return int(np.random.choice(candidates))

    @staticmethod
    def _selection_spea2(decs, objs, cons, params, n):
        """
        SPEA2 environmental selection (MToP ``Selection_SPEA2`` with Epsilon = 0).

        Parameters
        ----------
        decs : np.ndarray
            Decision variables of shape (pop_size, d_uni)
        objs : np.ndarray
            Objective values of shape (pop_size, n_obj)
        cons : np.ndarray
            Constraint values of shape (pop_size, n_con)
        params : dict
            Adaptive parameters with keys 'F', 'CR', 'TR', 'KP'
        n : int
            Target population size

        Returns
        -------
        selected_decs : np.ndarray
            Selected decision variables
        selected_objs : np.ndarray
            Selected objective values
        selected_cons : np.ndarray
            Selected constraint values
        selected_params : dict
            Selected adaptive parameters
        fitness : np.ndarray
            Fitness values of the selected population, sorted ascending
        """
        pop_size = objs.shape[0]
        if pop_size == 0:
            return decs, objs, cons, params, np.array([])

        n = min(n, pop_size)
        fitness = spea2_fitness(objs, cons)

        next_mask = fitness < 1
        n_selected = int(np.sum(next_mask))

        if n_selected < n:
            # Not enough non-dominated solutions: take the n best fitness values
            order = np.argsort(fitness, kind='stable')
            next_mask = np.zeros(pop_size, dtype=bool)
            next_mask[order[:n]] = True
        elif n_selected > n:
            # Too many: truncate the most crowded ones away
            candidate = np.where(next_mask)[0]
            keep = spea2_truncation(objs[candidate], n)
            next_mask = np.zeros(pop_size, dtype=bool)
            next_mask[candidate[keep]] = True

        selected = np.where(next_mask)[0]
        selected = selected[np.argsort(fitness[selected], kind='stable')]

        selected_params = {key: params[key][selected] for key in ('F', 'CR', 'TR', 'KP')}

        return decs[selected], objs[selected], cons[selected], selected_params, fitness[selected]

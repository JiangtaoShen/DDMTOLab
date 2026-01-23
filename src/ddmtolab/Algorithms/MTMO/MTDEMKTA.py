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


class MTDEMKTA:
    """
    Multi-objective Multi-task Differential Evolution with Multiple Knowledge Types and Transfer Adaptation.

    This algorithm features:
    - Self-adaptive parameters (F, CR, TR, KP) for each individual
    - Rank-based DE parent selection
    - Two knowledge transfer types: direct transfer and distribution-based transfer

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, tau1=0.2, tau2=0.1,
                 save_data=True, save_path='./TestData', name='MTDEMKTA_test', disable_tqdm=True):
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
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize adaptive parameters for each individual
        # params[t] is a dict with keys: 'F', 'CR', 'TR', 'KP'
        params = []
        for t in range(nt):
            params_t = {
                'F': 0.2 + np.random.rand(n_per_task[t]),  # F in [0.2, 1.2]
                'CR': np.random.rand(n_per_task[t]),  # CR in [0, 1]
                'TR': np.random.rand(n_per_task[t]),  # Transfer rate in [0, 1]
                'KP': np.random.rand(n_per_task[t])  # Knowledge type proportion in [0, 1]
            }
            params.append(params_t)

        # Initial SPEA2 selection and get fitness
        fitness = []
        for t in range(nt):
            decs[t], objs[t], cons[t], params[t], fit_t = self._selection_spea2(
                decs[t], objs[t], cons[t], params[t], n_per_task[t]
            )
            fitness.append(fit_t)

        # Initialize distribution models for each task
        models = []
        for t in range(nt):
            model_t = {
                'mean': np.mean(decs[t], axis=0),
                'std': np.std(decs[t], axis=0) + 1e-100
            }
            models.append(model_t)

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            active_tasks = [t for t in range(nt) if nfes_per_task[t] < max_nfes_per_task[t]]
            if not active_tasks:
                break

            # Update ranks for all tasks based on fitness
            ranks = []
            for t in range(nt):
                # Get rank from fitness (lower fitness = better rank)
                sorted_indices = np.argsort(fitness[t])
                rank_t = np.empty(len(fitness[t]), dtype=int)
                rank_t[sorted_indices] = np.arange(len(fitness[t]))
                ranks.append(rank_t)

            # Update distribution models with exponential moving average
            alpha = 0.5
            for t in range(nt):
                models[t]['mean'] = alpha * models[t]['mean'] + (1 - alpha) * np.mean(decs[t], axis=0)
                models[t]['std'] = alpha * models[t]['std'] + (1 - alpha) * (np.std(decs[t], axis=0) + 1e-100)

            # Generate offspring for all tasks
            off_decs_all = []
            off_params_all = []
            for t in active_tasks:
                off_decs_t, off_params_t = self._generation(
                    decs, params, ranks, models, t, n_per_task
                )
                off_decs_all.append(off_decs_t)
                off_params_all.append(off_params_t)

            # Evaluate and select for each task
            for idx, t in enumerate(active_tasks):
                off_decs_t = off_decs_all[idx]
                off_params_t = off_params_all[idx]

                # Evaluate offspring
                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_t, t)
                nfes_per_task[t] += len(off_decs_t)
                pbar.update(len(off_decs_t))

                # Merge parents and offspring
                merged_decs = np.vstack([decs[t], off_decs_t])
                merged_objs = np.vstack([objs[t], off_objs_t])
                merged_cons = np.vstack([cons[t], off_cons_t]) if cons[t] is not None else off_cons_t
                merged_params = {
                    'F': np.concatenate([params[t]['F'], off_params_t['F']]),
                    'CR': np.concatenate([params[t]['CR'], off_params_t['CR']]),
                    'TR': np.concatenate([params[t]['TR'], off_params_t['TR']]),
                    'KP': np.concatenate([params[t]['KP'], off_params_t['KP']])
                }

                # SPEA2 selection
                decs[t], objs[t], cons[t], params[t], fitness[t] = self._selection_spea2(
                    merged_decs, merged_objs, merged_cons, merged_params, n_per_task[t]
                )

                # Append to history
                append_history(all_decs[t], decs[t], all_objs[t], objs[t], all_cons[t], cons[t])

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, decs, params, ranks, models, t, n_per_task):
        """
        Generate offspring for task t using rank-based DE with knowledge transfer.

        Parameters
        ----------
        decs : list of np.ndarray
            Decision variables for all tasks
        params : list of dict
            Adaptive parameters for all tasks
        ranks : list of np.ndarray
            Ranks for all tasks (based on fitness)
        models : list of dict
            Distribution models for all tasks
        t : int
            Current task index
        n_per_task : list of int
            Population sizes for all tasks

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables of shape (N, dim)
        off_params : dict
            Offspring adaptive parameters
        """
        Np = n_per_task[t]
        nt = len(decs)
        dim = decs[t].shape[1]

        off_decs = np.zeros((Np, dim))
        off_params = {
            'F': np.zeros(Np),
            'CR': np.zeros(Np),
            'TR': np.zeros(Np),
            'KP': np.zeros(Np)
        }

        for i in range(Np):
            # Parameter disturbance with Gaussian noise
            off_F = np.random.normal(params[t]['F'][i], 0.1)
            off_F = np.clip(off_F, 0.2, 1.2)

            off_CR = np.random.normal(params[t]['CR'][i], 0.1)
            off_CR = np.clip(off_CR, 0, 1)

            off_TR = np.random.normal(params[t]['TR'][i], 0.1)
            off_TR = np.clip(off_TR, 0, 1)

            off_KP = np.random.normal(params[t]['KP'][i], 0.1)
            # Cyclic boundary for KP
            if off_KP < 0:
                off_KP = 1 + off_KP
            elif off_KP > 1:
                off_KP = off_KP - 1

            # Parameter mutation with probability
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

            # Rank-based DE parent selection
            # x1: selected with probability proportional to (Np - rank) / Np
            x1 = self._rank_selection(ranks[t], Np, exclude=[i])
            x2 = self._rank_selection(ranks[t], Np, exclude=[i, x1])
            x3 = self._random_selection(Np, exclude=[i, x1, x2])

            xDeci = decs[t][i]
            xDec1 = decs[t][x1]
            xDec2 = decs[t][x2]
            xDec3 = decs[t][x3]

            # Knowledge transfer
            if np.random.rand() < off_TR:
                # Select a helper task (different from current task)
                k = np.random.randint(nt)
                while k == t:
                    k = np.random.randint(nt)

                Np_k = n_per_task[k]

                # Two types of knowledge transfer based on KP
                if off_KP > 0.5:
                    # Type 1: Direct transfer - randomly select a solution from helper task
                    xDeck = decs[k][np.random.randint(Np_k)]
                    # Align dimensions to current task
                    xDeck = self._align_dimensions(xDeck, dim)
                else:
                    # Type 2: Distribution-based transfer
                    # Transform solution from helper task's distribution to current task's distribution
                    xDeck = decs[k][np.random.randint(Np_k)]
                    dim_k = len(xDeck)

                    # Standardize using helper task's distribution
                    xDeck_normalized = (xDeck - models[k]['mean']) / models[k]['std']

                    # Align dimensions after normalization
                    xDeck_normalized_aligned = self._align_dimensions(xDeck_normalized, dim)

                    # Get aligned mean and std for current task
                    mean_t = models[t]['mean']
                    std_t = models[t]['std']

                    # Transform to current task's distribution
                    xDeck = mean_t + std_t * xDeck_normalized_aligned

                # Use transferred solution as x2
                xDec2 = xDeck

            # DE/rand/1 mutation
            mutant = xDec1 + off_F * (xDec2 - xDec3)

            # DE binomial crossover
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < off_CR
            mask[j_rand] = True
            offspring_dec = np.where(mask, mutant, xDeci)

            # Boundary handling
            offspring_dec = np.clip(offspring_dec, 0, 1)

            off_decs[i] = offspring_dec

        return off_decs, off_params

    def _rank_selection(self, rank, Np, exclude=None):
        """
        Rank-based selection: select individual with probability proportional to (Np - rank) / Np.

        Parameters
        ----------
        rank : np.ndarray
            Ranks of individuals (0-indexed, lower is better)
        Np : int
            Population size
        exclude : list, optional
            Indices to exclude from selection

        Returns
        -------
        selected : int
            Selected individual index
        """
        if exclude is None:
            exclude = []

        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.randint(Np)
            # Accept with probability (Np - rank[x]) / Np
            # Higher probability for lower rank (better individuals)
            if x not in exclude and np.random.rand() < (Np - rank[x]) / Np:
                return x

        # Fallback: random selection excluding specified indices
        candidates = [i for i in range(Np) if i not in exclude]
        return np.random.choice(candidates)

    def _random_selection(self, Np, exclude=None):
        """
        Random selection excluding specified indices.

        Parameters
        ----------
        Np : int
            Population size
        exclude : list, optional
            Indices to exclude from selection

        Returns
        -------
        selected : int
            Selected individual index
        """
        if exclude is None:
            exclude = []

        candidates = [i for i in range(Np) if i not in exclude]
        return np.random.choice(candidates)

    def _align_dimensions(self, dec, target_dim):
        """
        Align decision variable dimensions to target dimension.

        Parameters
        ----------
        dec : np.ndarray
            Decision variable of shape (dim,)
        target_dim : int
            Target dimension

        Returns
        -------
        aligned_dec : np.ndarray
            Aligned decision variable of shape (target_dim,)
        """
        current_dim = len(dec)
        if current_dim == target_dim:
            return dec.copy()
        elif current_dim < target_dim:
            # Pad with random values
            padding = np.random.rand(target_dim - current_dim)
            return np.concatenate([dec, padding])
        else:
            # Truncate
            return dec[:target_dim].copy()

    def _selection_spea2(self, decs, objs, cons, params, n):
        """
        SPEA2 environmental selection.

        Parameters
        ----------
        decs : np.ndarray
            Decision variables of shape (pop_size, dim)
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
            Fitness values of selected population
        """
        pop_size = len(objs)
        if pop_size == 0:
            return decs, objs, cons, params, np.array([])

        n = min(n, pop_size)

        # Calculate constraint violation
        if cons is not None and cons.size > 0:
            cvs = np.sum(np.maximum(0, cons), axis=1)
        else:
            cvs = np.zeros(pop_size)

        # Calculate SPEA2 fitness
        fitness = self._cal_spea2_fitness(objs, cvs)

        # Environmental selection
        next_mask = fitness < 1
        n_selected = np.sum(next_mask)

        if n_selected < n:
            # Not enough: select top n by fitness
            sorted_indices = np.argsort(fitness)
            next_mask = np.zeros(pop_size, dtype=bool)
            next_mask[sorted_indices[:n]] = True
        elif n_selected > n:
            # Too many: truncation
            selected_indices = np.where(next_mask)[0]
            selected_objs_temp = objs[selected_indices]
            del_indices = self._truncation(selected_objs_temp, n_selected - n)
            next_mask[selected_indices[del_indices]] = False

        # Apply selection
        selected_indices = np.where(next_mask)[0]

        # Sort by fitness
        sorted_fitness_indices = np.argsort(fitness[selected_indices])
        selected_indices = selected_indices[sorted_fitness_indices]

        selected_decs = decs[selected_indices]
        selected_objs = objs[selected_indices]
        selected_cons = cons[selected_indices] if cons is not None else None
        selected_params = {
            'F': params['F'][selected_indices],
            'CR': params['CR'][selected_indices],
            'TR': params['TR'][selected_indices],
            'KP': params['KP'][selected_indices]
        }
        selected_fitness = fitness[selected_indices]

        return selected_decs, selected_objs, selected_cons, selected_params, selected_fitness

    def _cal_spea2_fitness(self, objs, cvs):
        """
        Calculate SPEA2 fitness values.

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (pop_size, n_obj)
        cvs : np.ndarray
            Constraint violations of shape (pop_size,)

        Returns
        -------
        fitness : np.ndarray
            SPEA2 fitness values of shape (pop_size,)
        """
        n = len(objs)
        if n == 0:
            return np.array([])

        # Detect dominance relations
        dominate = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if cvs[i] < cvs[j]:
                    dominate[i, j] = True
                elif cvs[i] > cvs[j]:
                    dominate[j, i] = True
                else:
                    # Compare objectives (minimization)
                    better_i = np.any(objs[i] < objs[j])
                    worse_i = np.any(objs[i] > objs[j])
                    if better_i and not worse_i:
                        dominate[i, j] = True
                    elif worse_i and not better_i:
                        dominate[j, i] = True

        # Strength S(i) = number of solutions dominated by i
        s = np.sum(dominate, axis=1)

        # Raw fitness R(i) = sum of S(j) for all j that dominate i
        r = np.zeros(n)
        for i in range(n):
            dominating_indices = np.where(dominate[:, i])[0]
            r[i] = np.sum(s[dominating_indices])

        # Density D(i)
        distance = cdist(objs, objs)
        np.fill_diagonal(distance, np.inf)
        distance_sorted = np.sort(distance, axis=1)
        k_neighbor = int(np.floor(np.sqrt(n)))
        k_neighbor = max(0, min(k_neighbor, n - 1))
        d = 1.0 / (distance_sorted[:, k_neighbor] + 2)

        # Total fitness = R + D
        fitness = r + d
        return fitness

    def _truncation(self, objs, k):
        """
        Truncation operator for SPEA2.

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (pop_size, n_obj)
        k : int
            Number of solutions to delete

        Returns
        -------
        del_indices : np.ndarray
            Indices of solutions to delete
        """
        n = len(objs)
        del_mask = np.zeros(n, dtype=bool)

        distance = cdist(objs, objs)
        np.fill_diagonal(distance, np.inf)

        while np.sum(del_mask) < k:
            remain_indices = np.where(~del_mask)[0]
            remain_dist = distance[remain_indices][:, remain_indices]
            sorted_dist = np.sort(remain_dist, axis=1)
            sorted_rows_indices = np.lexsort(np.rot90(sorted_dist))
            del_idx_in_remain = sorted_rows_indices[0]
            del_mask[remain_indices[del_idx_in_remain]] = True

        return np.where(del_mask)[0]
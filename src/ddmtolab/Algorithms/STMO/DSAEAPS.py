"""
Dual Surrogate assisted Evolutionary Algorithm based on Parallel Search (DSAEA-PS)

This module implements DSAEA-PS for computationally expensive multi/many-objective optimization.

References
----------
    [1] Shen, Jiangtao, et al. "A dual surrogate assisted evolutionary algorithm based on parallel search for expensive multi/many-objective optimization." Applied Soft Computing 148 (2023): 110879.

Notes
-----
Author: Jiangtao Shen
Date: 2025.01.12
Version: 1.0
"""
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.interpolate import RBFInterpolator
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
from ddmtolab.Algorithms.STMO.RVEA import RVEA
from ddmtolab.Algorithms.STMO.IBEA import IBEA
from ddmtolab.Algorithms.STMO.NSGAIISDR import NSGAIISDR, nd_sort_sdr
from ddmtolab.Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class DSAEAPS:
    """
    Dual surrogate assisted evolutionary algorithm based on parallel search for expensive
    multi/many-objective optimization.

    This algorithm uses two types of surrogate models:
    1. Gaussian Process (GP) models for each objective
    2. RBF model for predicting SDR-based front numbers

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100, wmax=20, mu=5,
                 save_data=True, save_path='./TestData', name='DSAEAPS_test', disable_tqdm=True):
        """
        Initialize DSAEA-PS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size per task (default: 100)
        wmax : int, optional
            Number of generations for surrogate optimization (default: 20)
        mu : int, optional
            Number of re-evaluated solutions at each generation (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'DSAEAPS_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.wmax = wmax
        self.mu = mu
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the DSAEA-PS algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        n_per_task = par_list(self.n, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Reorganize initial data into task-specific history lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task, interval=self.mu)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task, interval=self.mu)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                m = n_objs[i]
                dim = dims[i]

                # ===== Step 1: Build surrogate models =====
                # Build GP models for each objective using mo_gp_build
                gp_models = mo_gp_build(decs[i], objs[i], data_type)

                # Normalize objectives and compute SDR front numbers for RBF training
                objs_norm, _, _ = normalize(objs[i], axis=0, method='minmax')
                front_no_sdr, _ = nd_sort_sdr(objs_norm, len(objs[i]))

                # Build RBF model for predicting front numbers
                rbf_model = RBFInterpolator(decs[i], front_no_sdr)

                # ===== Step 2: Optimize surrogates using RVEA, IBEA, NSGA-II-SDR =====
                # Create surrogate problem wrapper using mo_gp_predict
                def surrogate_func(x, models=gp_models, dtype=data_type):
                    return mo_gp_predict(models, x, dtype, mse=False)

                surrogate_problem = MTOP()
                surrogate_problem.add_task(objective_func=surrogate_func, dim=dim)

                # Run RVEA on surrogate
                rvea = RVEA(surrogate_problem, n=n_per_task[i], max_nfes=n_per_task[i] * self.wmax, disable_tqdm=True)
                results_rvea = rvea.optimize()
                pop_decs_1 = results_rvea.best_decs[0]
                pop_objs_1 = results_rvea.best_objs[0]

                # Run IBEA on surrogate
                ibea = IBEA(surrogate_problem, n=n_per_task[i], max_nfes=n_per_task[i] * self.wmax, disable_tqdm=True)
                results_ibea = ibea.optimize()
                pop_decs_2 = results_ibea.best_decs[0]
                pop_objs_2 = results_ibea.best_objs[0]

                # Run NSGA-II-SDR on surrogate
                nsgaiisdr = NSGAIISDR(surrogate_problem, n=n_per_task[i], max_nfes=n_per_task[i] * self.wmax, disable_tqdm=True)
                results_nsgaiisdr = nsgaiisdr.optimize()
                pop_decs_3 = results_nsgaiisdr.best_decs[0]
                pop_objs_3 = results_nsgaiisdr.best_objs[0]

                # Merge populations
                C_pop_decs = np.vstack([pop_decs_1, pop_decs_2, pop_decs_3])
                C_pop_objs = np.vstack([pop_objs_1, pop_objs_2, pop_objs_3])

                # ===== Step 3: Dual front number computation =====
                # Normalize merged population objectives
                C_pop_objs_norm, _, _ = normalize(C_pop_objs, axis=0, method='minmax')

                # Front number based on GP (SDR sorting)
                FN_GP, max_FN = nd_sort_sdr(C_pop_objs_norm, len(C_pop_objs))

                # Predict front numbers using RBF model
                FN_RBF_raw = rbf_model(C_pop_decs)

                # Cluster RBF predictions into max_FN classes
                FN_RBF = self._cluster_front_numbers(FN_RBF_raw, max_FN)

                # ===== Step 4: Selection based on dual front numbers =====
                # Create bi-objective matrix: [FN_GP + FN_RBF, |FN_GP - FN_RBF|]
                ss = np.column_stack([FN_GP + FN_RBF, np.abs(FN_GP - FN_RBF)])

                # Perform standard non-dominated sorting on ss
                front_no_ss, _ = nd_sort(ss, len(ss))

                # Get first front indices
                index_F1 = np.where(front_no_ss == 1)[0]

                # ===== Step 5: Select mu solutions for re-evaluation =====
                if len(index_F1) <= self.mu:
                    pop_new_decs = C_pop_decs[index_F1]
                else:
                    # Use IBEA-based selection
                    F1_decs = C_pop_decs[index_F1]
                    F1_objs = C_pop_objs[index_F1]
                    pop_new_decs = self._se_ibea(F1_decs, F1_objs, objs[i], self.mu)

                # Remove duplicates
                pop_new_decs = remove_duplicates(pop_new_decs, decs[i])

                if pop_new_decs.shape[0] > 0:
                    # Evaluate new solutions
                    new_objs, _ = evaluation_single(problem, pop_new_decs, i)

                    # Update archive
                    decs[i] = np.vstack([decs[i], pop_new_decs])
                    objs[i] = np.vstack([objs[i], new_objs])

                    nfes_per_task[i] += pop_new_decs.shape[0]
                    pbar.update(pop_new_decs.shape[0])

                    append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    def _cluster_front_numbers(self, fn_raw, max_fn):
        """
        Cluster continuous front number predictions into discrete classes.

        Parameters
        ----------
        fn_raw : np.ndarray
            Raw front number predictions from RBF model
        max_fn : int
            Maximum number of fronts (clusters)

        Returns
        -------
        fn_clustered : np.ndarray
            Clustered front numbers (1 to max_fn)
        """
        if max_fn <= 1 or len(fn_raw) <= max_fn:
            return np.ones(len(fn_raw), dtype=int)

        # K-means clustering
        kmeans = KMeans(n_clusters=max_fn, random_state=42, n_init=10)
        labels = kmeans.fit_predict(fn_raw.reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()

        # Assign front numbers based on cluster center ordering
        fn_clustered = np.zeros(len(fn_raw), dtype=int)
        sorted_center_indices = np.argsort(centers)

        for rank, center_idx in enumerate(sorted_center_indices):
            fn_clustered[labels == center_idx] = rank + 1

        return fn_clustered

    def _se_ibea(self, pop_decs, pop_objs, archive_objs, mu):
        """
        IBEA-based selection for infill points.

        Parameters
        ----------
        pop_decs : np.ndarray
            Candidate decision variables, shape (N_pop, D)
        pop_objs : np.ndarray
            Candidate objective values, shape (N_pop, M)
        archive_objs : np.ndarray
            Archive objective values, shape (N_archive, M)
        mu : int
            Number of solutions to select

        Returns
        -------
        selected_decs : np.ndarray
            Selected decision variables, shape (mu, D)
        """
        kappa = 0.05

        # Combined normalization
        combined_objs = np.vstack([archive_objs, pop_objs])
        combined_norm, _, _ = normalize(combined_objs, axis=0, method='minmax')

        n_archive = len(archive_objs)
        archive_objs_norm = combined_norm[:n_archive]
        pop_objs_norm = combined_norm[n_archive:]

        # SDR-based front filtering
        if n_archive > 0:
            fn_archive, _ = nd_sort_sdr(archive_objs_norm, n_archive)
            archive_objs_norm = archive_objs_norm[fn_archive == 1]

        fn_pop, _ = nd_sort_sdr(pop_objs_norm, len(pop_objs_norm))
        nd_mask = fn_pop == 1
        pop_objs_norm = pop_objs_norm[nd_mask]
        pop_decs_filtered = pop_decs[nd_mask]

        n_pop = len(pop_objs_norm)
        if n_pop <= mu:
            return pop_decs_filtered

        # Combined objectives: pop first, then archive
        n_archive = len(archive_objs_norm)
        C_objs = np.vstack([pop_objs_norm, archive_objs_norm]) if n_archive > 0 else pop_objs_norm

        # Calculate IBEA fitness
        fitness, I, C = ibea_fitness(C_objs, kappa)

        # Track remaining indices (True = remaining)
        N = len(C_objs)
        remaining = np.ones(N, dtype=bool)
        n_pop_remaining = n_pop

        while n_pop_remaining > mu:
            # Get indices of remaining solutions
            remaining_idx = np.where(remaining)[0]

            # Find and remove solution with minimum fitness
            x = remaining_idx[np.argmin(fitness[remaining_idx])]
            remaining[x] = False

            # Update fitness after removal
            if C[x] > 1e-10:
                fitness += np.exp(-I[x, :] / C[x] / kappa)

            # Update count if removed from pop (first n_pop elements)
            if x < n_pop:
                n_pop_remaining -= 1

        # Return selected solutions from pop
        selected_idx = np.where(remaining[:n_pop])[0]
        return pop_decs_filtered[selected_idx]
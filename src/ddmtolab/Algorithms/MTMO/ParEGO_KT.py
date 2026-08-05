"""
ParEGO with Knowledge Transfer (ParEGO-KT)

This module implements ParEGO-KT for expensive multitask multiobjective optimization.
It extends ParEGO with cross-task knowledge transfer using Spearman rank correlation
to identify and leverage beneficial task relationships.

References
----------
    [1] J. Knowles. "ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems." IEEE Transactions on Evolutionary Computation, 2006, 10(1): 50-66.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.18
Version: 1.0
"""
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import torch
from ddmtolab.Methods.Algo_Methods.bo_utils import bo_next_point
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings
import time

warnings.filterwarnings("ignore")


class ParEGO_KT:
    """
    ParEGO with knowledge transfer for expensive multitask multiobjective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[1, M]',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, n_weights=None, max_nfes=None, rho=0.05,
                 save_data=True, save_path='./Data', name='ParEGO-KT', disable_tqdm=True):
        """
        Initialize ParEGO-KT algorithm with knowledge transfer.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim - 1)
        n_weights : int or List[int], optional
            Number of reference weight vectors per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        rho : float, optional
            Augmentation coefficient for augmented Tchebycheff scalarization (default: 0.05)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'ParEGO-KT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.n_weights = n_weights if n_weights is not None else 100
        # Keep the caller's request as given; the budget declared by the tasks is
        # consulted at run time and only used when max_nfes was left unset.
        self.max_nfes = max_nfes
        self.rho = rho
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the ParEGO-KT algorithm with knowledge transfer.

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
        no = problem.n_objs

        # Set default initial samples: 11*dim - 1
        if self.n_initial is None:
            n_initial_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)

        max_nfes_per_task = resolve_budget(problem, self.max_nfes, nt, default=200)
        n_weights_per_task = par_list(self.n_weights, nt)

        # Generate uniformly distributed weight vectors for each task
        W = []
        for i in range(nt):
            w_i, actual_n = uniform_point(n_weights_per_task[i], no[i])
            W.append(w_i)
            n_weights_per_task[i] = actual_n

        # ============ Knowledge Transfer: Initialize with shared samples ============
        # Generate initial samples in [0,1] space using LHS (the_same=True)
        decs = initialization(problem, n_initial_per_task, method='lhs', the_same=True)
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # ============ Compute task similarity based on initial samples ============
        # Compute ranking for each task using NSGA-II sorting
        ranks = []
        for i in range(nt):
            rank_i, _, _ = nsga2_sort(objs[i])
            ranks.append(rank_i)

        # Compute similarity matrix using Spearman's rank correlation
        similarity_matrix = self._compute_similarity_matrix(ranks, n_initial_per_task)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # ============ BO Sampling Phase ============
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Randomly select a weight vector
                weight_idx = np.random.randint(0, n_weights_per_task[i])
                weight = W[i][weight_idx]

                # Scalarize multi-objective values using augmented Tchebycheff
                scalar_objs = self._scalarize(objs[i], weight)

                # Fit GP surrogate and select next candidate via BO with EI
                candidate_np = bo_next_point(dims[i], decs[i], scalar_objs, data_type=data_type)

                # Evaluate the candidate solution
                obj, _ = evaluation_single(problem, candidate_np, i)

                # Update dataset with new sample
                decs[i], objs[i] = vstack_groups((decs[i], candidate_np), (objs[i], obj))

                nfes_per_task[i] += 1
                pbar.update(1)

            # ============ Knowledge Transfer Phase ============
            # After all tasks complete one BO iteration, perform knowledge transfer
            for i in active_tasks:
                if nfes_per_task[i] >= max_nfes_per_task[i]:
                    continue

                # Find the most similar task to task i
                most_similar_task = self._find_most_similar_task(i, similarity_matrix)

                # Generate random number for transfer decision
                transfer_prob = similarity_matrix[i, most_similar_task]
                if np.random.rand() < transfer_prob:
                    # Find the best solution in the most similar task
                    weight_idx = np.random.randint(0, n_weights_per_task[most_similar_task])
                    weight = W[most_similar_task][weight_idx]
                    scalar_objs_source = self._scalarize(objs[most_similar_task], weight)

                    best_idx = np.argmin(scalar_objs_source)
                    best_solution = decs[most_similar_task][best_idx]

                    # Transfer solution: truncate or pad to match target task dimension
                    transferred_solution = self._adapt_solution(best_solution, dims[most_similar_task], dims[i])

                    # Evaluate transferred solution on target task
                    obj_transfer, _ = evaluation_single(problem, transferred_solution, i)

                    # Update dataset
                    decs[i], objs[i] = vstack_groups((decs[i], transferred_solution), (objs[i], obj_transfer))

                    nfes_per_task[i] += 1
                    pbar.update(1)

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    def _compute_similarity_matrix(self, ranks, n_initial_per_task):
        """
        Compute task similarity matrix based on Spearman's rank correlation.

        Parameters
        ----------
        ranks : List[np.ndarray]
            List of rank arrays for each task
        n_initial_per_task : List[int]
            Number of initial samples per task

        Returns
        -------
        similarity_matrix : np.ndarray
            Similarity matrix of shape (n_tasks, n_tasks) with values in [0, 1]
        """
        nt = len(ranks)
        similarity_matrix = np.zeros((nt, nt))

        # Use minimum number of samples for correlation computation
        min_samples = min(n_initial_per_task)

        for i in range(nt):
            for j in range(nt):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Compute Spearman correlation on the common samples
                    rank_i = ranks[i][:min_samples]
                    rank_j = ranks[j][:min_samples]

                    correlation, _ = spearmanr(rank_i, rank_j)

                    # Convert correlation from [-1, 1] to [0, 1]
                    # correlation = 1 means identical ranking (high similarity)
                    # correlation = -1 means opposite ranking (low similarity)
                    similarity = (correlation + 1) / 2
                    similarity_matrix[i, j] = max(0.0, min(1.0, similarity))

        return similarity_matrix

    def _find_most_similar_task(self, task_idx, similarity_matrix):
        """
        Find the most similar task to the given task.

        Parameters
        ----------
        task_idx : int
            Index of the target task
        similarity_matrix : np.ndarray
            Task similarity matrix

        Returns
        -------
        most_similar_idx : int
            Index of the most similar task
        """
        similarities = similarity_matrix[task_idx].copy()
        similarities[task_idx] = -1  # Exclude self
        most_similar_idx = np.argmax(similarities)
        return most_similar_idx

    def _adapt_solution(self, solution, source_dim, target_dim):
        """
        Adapt solution from source task to target task by truncating or padding.

        Parameters
        ----------
        solution : np.ndarray
            Solution from source task of shape (source_dim,)
        source_dim : int
            Dimension of source task
        target_dim : int
            Dimension of target task

        Returns
        -------
        adapted_solution : np.ndarray
            Adapted solution of shape (target_dim,)
        """
        if source_dim >= target_dim:
            # Truncate to target dimension
            return solution[:target_dim].reshape(1, -1)
        else:
            # Pad with random values in [0, 1]
            padding = np.random.rand(target_dim - source_dim)
            return np.concatenate([solution, padding]).reshape(1, -1)

    def _scalarize(self, objs, weight):
        """
        Scalarize multi-objective values using augmented Tchebycheff approach.

        Parameters
        ----------
        objs : np.ndarray
            Multi-objective values of shape (N, M)
        weight : np.ndarray
            Weight vector of shape (M,)

        Returns
        -------
        scalar_objs : np.ndarray
            Scalarized objective values of shape (N, 1)
        """
        # Normalize objectives to [0, 1] range
        obj_min = np.min(objs, axis=0)
        obj_max = np.max(objs, axis=0)
        obj_range = obj_max - obj_min
        obj_range = np.maximum(obj_range, 1e-10)
        normalized_objs = (objs - obj_min) / obj_range

        # Augmented Tchebycheff scalarization
        weighted_objs = normalized_objs * weight
        max_term = np.max(weighted_objs, axis=1)
        aug_term = self.rho * np.sum(weighted_objs, axis=1)
        scalar_objs = max_term + aug_term

        return scalar_objs.reshape(-1, 1)

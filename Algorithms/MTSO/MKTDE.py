"""
Meta-Knowledge Transfer-based Differential Evolution (MKTDE)

This module implements MKTDE for multi-task single-objective optimization problems.

References
----------
    [1] Li, Jian-Yu, Zhi-Hui Zhan, Kay Chen Tan, and Jun Zhang. "A Meta-Knowledge Transfer-Based Differential Evolution for Multitask Optimization." IEEE Transactions on Evolutionary Computation 26.4 (2022): 719-734. https://doi.org/10.1109/TEVC.2021.3131236

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.01.04
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from Methods.Algo_Methods.algo_utils import *


class MKTDE:
    """
    Meta-Knowledge Transfer-based Differential Evolution for multi-task optimization.

    This algorithm features:
    - Meta-knowledge transfer via centroid-based solution transformation
    - DE/rand/1/bin mutation strategy with extended population
    - Elite solution transfer between tasks
    - Elitist selection mechanism

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '2-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
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

    def __init__(self, problem, n=None, max_nfes=None, F=0.5, CR=0.6,
                 save_data=True, save_path='./TestData', name='MKTDE_test', disable_tqdm=True):
        """
        Initialize MKTDE algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        F : float, optional
            DE mutation factor (default: 0.5)
        CR : float, optional
            DE crossover rate (default: 0.6)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MKTDE_test')
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
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        # Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            # Compute centroids for all tasks
            centroids = []
            for t in range(nt):
                centroid = np.mean(decs[t], axis=0)
                centroids.append(centroid)

            # Source task selection for each task
            source_tasks = []
            for t in range(nt):
                s = np.random.randint(nt)
                while s == t:
                    s = np.random.randint(nt)
                source_tasks.append(s)

            # Generation and selection for each task
            for t in range(nt):
                if nfes_per_task[t] >= max_nfes_per_task[t]:
                    continue

                s = source_tasks[t]
                ct = centroids[t]  # Current task centroid
                cs = centroids[s]  # Source task centroid

                # Generate offspring using meta-knowledge transfer
                off_decs = self._generation(decs[t], decs[s], ct, cs, dims[t], dims[s])
                off_objs, off_cons = evaluation_single(problem, off_decs, t)
                nfes_per_task[t] += len(off_decs)
                pbar.update(len(off_decs))

                # Elitist selection: compare parent vs offspring one-by-one
                cvs = np.sum(np.maximum(0, cons[t]), axis=1) if cons[t] is not None and cons[t].size > 0 else np.zeros(len(objs[t]))
                for i in range(len(decs[t])):
                    parent_cv = cvs[i]
                    parent_obj = objs[t][i, 0]
                    off_cv = np.sum(np.maximum(0, off_cons[i])) if off_cons is not None and off_cons[i].size > 0 else 0
                    off_obj = off_objs[i, 0]

                    # Constrained comparison: prefer lower CV, then lower objective
                    if off_cv < parent_cv or (off_cv == parent_cv and off_obj < parent_obj):
                        decs[t][i] = off_decs[i]
                        objs[t][i] = off_objs[i]
                        if cons[t] is not None:
                            cons[t][i] = off_cons[i]

            # Elite solution transfer: replace worst solution with elite from source task
            for t in range(nt):
                if nfes_per_task[t] >= max_nfes_per_task[t]:
                    continue

                s = source_tasks[t]

                # Find elite (best) solution from source task
                cvs_s = np.sum(np.maximum(0, cons[s]), axis=1) if cons[s] is not None and cons[s].size > 0 else np.zeros(len(objs[s]))
                elite_idx = self._get_best_index(objs[s][:, 0], cvs_s)
                elite_dec = decs[s][elite_idx].copy()

                # Align dimensions if necessary
                elite_dec = self._align_dimensions(elite_dec, dims[t])

                # Evaluate elite in current task
                elite_dec_2d = elite_dec.reshape(1, -1)
                elite_obj, elite_con = evaluation_single(problem, elite_dec_2d, t)
                nfes_per_task[t] += 1
                pbar.update(1)

                # Replace the last (worst after sorting) individual
                # First sort population by fitness
                cvs_t = np.sum(np.maximum(0, cons[t]), axis=1) if cons[t] is not None and cons[t].size > 0 else np.zeros(len(objs[t]))
                worst_idx = self._get_worst_index(objs[t][:, 0], cvs_t)

                decs[t][worst_idx] = elite_dec
                objs[t][worst_idx] = elite_obj[0]
                if cons[t] is not None:
                    cons[t][worst_idx] = elite_con[0]

            # Append to history
            for t in range(nt):
                append_history(all_decs[t], decs[t], all_objs[t], objs[t], all_cons[t], cons[t])

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, pop_decs, spop_decs, ct, cs, dim_t, dim_s):
        """
        Generate offspring using meta-knowledge transfer-based DE.

        The meta-knowledge transfer transforms solutions from the source task
        to the target task space using centroid alignment: x_s - cs + ct

        Parameters
        ----------
        pop_decs : np.ndarray
            Current task population, shape (n, dim_t)
        spop_decs : np.ndarray
            Source task population, shape (n_s, dim_s)
        ct : np.ndarray
            Centroid of current task, shape (dim_t,)
        cs : np.ndarray
            Centroid of source task, shape (dim_s,)
        dim_t : int
            Dimension of current task
        dim_s : int
            Dimension of source task

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (n, dim_t)
        """
        n = len(pop_decs)

        # Transform source population to target task space via centroid alignment
        # Align dimensions first
        spop_aligned = np.zeros((len(spop_decs), dim_t))
        cs_aligned = self._align_dimensions(cs, dim_t)

        for i in range(len(spop_decs)):
            dec_aligned = self._align_dimensions(spop_decs[i], dim_t)
            # Meta-knowledge transfer: x_s - cs + ct
            spop_aligned[i] = dec_aligned - cs_aligned + ct

        # Combine current population with transformed source population
        popf_dec = np.vstack([pop_decs, spop_aligned])
        n_combined = len(popf_dec)

        off_decs = np.zeros((n, dim_t))

        for i in range(n):
            # Select x1 from current population (different from i)
            x1 = np.random.randint(n)
            while x1 == i:
                x1 = np.random.randint(n)

            # Select x2 from combined population (different from i and x1)
            x2 = np.random.randint(n_combined)
            while x2 == i or x2 == x1:
                x2 = np.random.randint(n_combined)

            # Select x3 from combined population (different from i, x1, x2)
            x3 = np.random.randint(n_combined)
            while x3 == i or x3 == x2 or x3 == x1:
                x3 = np.random.randint(n_combined)

            # DE/rand/1 mutation
            mutant = pop_decs[x1] + self.F * (popf_dec[x2] - popf_dec[x3])

            # DE binomial crossover
            off_dec = self._de_crossover(mutant, pop_decs[i])

            # Boundary handling
            off_decs[i] = np.clip(off_dec, 0, 1)

        return off_decs

    def _de_crossover(self, mutant, target):
        """
        Perform DE binomial crossover.

        Parameters
        ----------
        mutant : np.ndarray
            Mutant vector, shape (dim,)
        target : np.ndarray
            Target vector, shape (dim,)

        Returns
        -------
        trial : np.ndarray
            Trial vector after crossover, shape (dim,)
        """
        dim = len(target)
        j_rand = np.random.randint(dim)
        mask = np.random.rand(dim) < self.CR
        mask[j_rand] = True  # Ensure at least one dimension from mutant
        trial = np.where(mask, mutant, target)
        return trial

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
            # Pad with random values in [0, 1]
            padding = np.random.rand(target_dim - current_dim)
            return np.concatenate([dec, padding])
        else:
            # Truncate to target dimension
            return dec[:target_dim].copy()

    def _get_best_index(self, objs, cvs):
        """
        Get index of the best individual considering constraints.

        Parameters
        ----------
        objs : np.ndarray
            Objective values, shape (n,)
        cvs : np.ndarray
            Constraint violations, shape (n,)

        Returns
        -------
        best_idx : int
            Index of the best individual
        """
        # Sort by CV first, then by objective
        indices = np.lexsort((objs, cvs))
        return indices[0]

    def _get_worst_index(self, objs, cvs):
        """
        Get index of the worst individual considering constraints.

        Parameters
        ----------
        objs : np.ndarray
            Objective values, shape (n,)
        cvs : np.ndarray
            Constraint violations, shape (n,)

        Returns
        -------
        worst_idx : int
            Index of the worst individual
        """
        # Sort by CV first, then by objective
        indices = np.lexsort((objs, cvs))
        return indices[-1]
"""
Multi-task Evolutionary Algorithm with Adaptive Knowledge Transfer via Anomaly Detection (MTEA-AD)

This module implements MTEA-AD for multi-task optimization with adaptive knowledge transfer
using anomaly detection to identify beneficial solutions from other tasks.

References
----------
    [1] Wang, Chao, et al. "Solving multitask optimization problems with adaptive knowledge transfer via anomaly detection." IEEE Transactions on Evolutionary Computation 26.2 (2021): 304-318.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.12
Version: 1.0
"""
import time
from tqdm import tqdm
import numpy as np
from scipy.stats import multivariate_normal
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MTEAAD:
    """
    Multi-task Evolutionary Algorithm with Adaptive Knowledge Transfer via Anomaly Detection.

    Uses a Gaussian-based anomaly detection model to adaptively identify and transfer
    beneficial solutions from other tasks during optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, TRP=0.1, muc=2.0, mum=5.0,
                 save_data=True, save_path='./TestData', name='MTEA_AD_test', disable_tqdm=True):
        """
        Initialize MTEA-AD algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        TRP : float, optional
            Transfer probability - probability of knowledge transfer in each generation (default: 0.1)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 5.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MTEA_AD_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.TRP = TRP
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MTEA-AD algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
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

        # Initialize epsilon (anomaly detection parameter) for each task
        epsilon = np.zeros(nt)

        gen = 1
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for t in active_tasks:
                # Generate offspring through crossover and mutation
                off_decs = self._generation(decs[t])

                # Knowledge transfer with probability TRP
                if np.random.rand() < self.TRP:
                    # Set NL parameter for anomaly detection
                    if gen == 1:
                        NL = 1.0
                    else:
                        NL = epsilon[t]

                    # Collect populations from other tasks (limit to 10 for many-task problems)
                    kpool = [k for k in active_tasks if k != t]
                    if len(kpool) > 10:
                        kpool = np.random.choice(kpool, size=10, replace=False).tolist()

                    # Gather historical population from other tasks
                    his_pop_dec = np.vstack([decs[k][:, :dims[t]] for k in kpool]) if kpool else np.empty((0, dims[t]))

                    if his_pop_dec.shape[0] > 0:
                        # Learn anomaly detection model and get transfer solutions
                        tfsol = self._learn_anomaly_detection(off_decs, his_pop_dec, NL)

                        # Clip transferred solutions to [0, 1]
                        tfsol = np.clip(tfsol, 0.0, 1.0)

                        if tfsol.shape[0] > 0:
                            # Evaluate offspring and transferred solutions
                            off_objs, off_cons = evaluation_single(problem, off_decs, t)
                            tf_objs, tf_cons = evaluation_single(problem, tfsol, t)

                            # Merge parent, offspring, and transferred populations
                            merged_objs, merged_decs, merged_cons = vstack_groups(
                                (objs[t], off_objs, tf_objs),
                                (decs[t], off_decs, tfsol),
                                (cons[t], off_cons, tf_cons)
                            )

                            # Elitist selection
                            index = selection_elit(merged_objs, n_per_task[t], merged_cons)
                            objs[t], decs[t], cons[t] = select_by_index(index, merged_objs, merged_decs, merged_cons)

                            # Calculate success rate for parameter adaptation
                            # Count how many transferred solutions were selected
                            parent_off_size = n_per_task[t] + n_per_task[t]  # parent + offspring
                            succ_num = np.sum(index >= parent_off_size)
                            epsilon[t] = succ_num / tfsol.shape[0] if tfsol.shape[0] > 0 else 0

                            nfes_per_task[t] += n_per_task[t] + tfsol.shape[0]
                            pbar.update(n_per_task[t] + tfsol.shape[0])
                        else:
                            # No transfer solutions, proceed normally
                            off_objs, off_cons = evaluation_single(problem, off_decs, t)
                            objs[t], decs[t], cons[t] = vstack_groups(
                                (objs[t], off_objs), (decs[t], off_decs), (cons[t], off_cons)
                            )
                            index = selection_elit(objs[t], n_per_task[t], cons[t])
                            objs[t], decs[t], cons[t] = select_by_index(index, objs[t], decs[t], cons[t])

                            nfes_per_task[t] += n_per_task[t]
                            pbar.update(n_per_task[t])
                    else:
                        # No other tasks available for transfer
                        off_objs, off_cons = evaluation_single(problem, off_decs, t)
                        objs[t], decs[t], cons[t] = vstack_groups(
                            (objs[t], off_objs), (decs[t], off_decs), (cons[t], off_cons)
                        )
                        index = selection_elit(objs[t], n_per_task[t], cons[t])
                        objs[t], decs[t], cons[t] = select_by_index(index, objs[t], decs[t], cons[t])

                        nfes_per_task[t] += n_per_task[t]
                        pbar.update(n_per_task[t])
                else:
                    # No knowledge transfer this generation
                    off_objs, off_cons = evaluation_single(problem, off_decs, t)

                    # Merge parent and offspring populations
                    objs[t], decs[t], cons[t] = vstack_groups(
                        (objs[t], off_objs), (decs[t], off_decs), (cons[t], off_cons)
                    )

                    # Elitist selection
                    index = selection_elit(objs[t], n_per_task[t], cons[t])
                    objs[t], decs[t], cons[t] = select_by_index(index, objs[t], decs[t], cons[t])

                    nfes_per_task[t] += n_per_task[t]
                    pbar.update(n_per_task[t])

                append_history(all_decs[t], decs[t], all_objs[t], objs[t], all_cons[t], cons[t])

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, population):
        # Generate offspring through crossover, mutation and gene swapping.

        n_pop = population.shape[0]
        d = population.shape[1]
        offspring = np.zeros((n_pop, d))

        # Random pairing
        ind_order = np.random.permutation(n_pop)

        count = 0
        for i in range(n_pop // 2):
            p1 = ind_order[i]
            p2 = ind_order[i + n_pop // 2]

            # Crossover
            off1, off2 = crossover(population[p1, :], population[p2, :], mu=self.muc)

            # Mutation
            off1 = mutation(off1, mu=self.mum)
            off2 = mutation(off2, mu=self.mum)

            # Gene swapping (as in MATLAB)
            swap_indicator = np.random.rand(d) < 0.5
            temp = off1[swap_indicator].copy()
            off1[swap_indicator] = off2[swap_indicator]
            off2[swap_indicator] = temp

            # Clip to [0, 1]
            off1 = np.clip(off1, 0.0, 1.0)
            off2 = np.clip(off2, 0.0, 1.0)

            offspring[count, :] = off1
            offspring[count + 1, :] = off2
            count += 2

        return offspring

    def _learn_anomaly_detection(self, curr_pop, his_pop, NL):
        """
        Learn anomaly detection model to identify candidate transferred solutions.

        Uses a multivariate Gaussian distribution fitted on the current population
        to score solutions from historical populations. Solutions with high scores
        (low anomaly) are selected for transfer.

        Parameters
        ----------
        curr_pop : np.ndarray
            Current task population of shape (n, d)
        his_pop : np.ndarray
            Historical population from other tasks of shape (m, d)
        NL : float
            Anomaly detection threshold parameter in [0, 1].
            Controls the proportion of solutions selected for transfer.
            NL=1 selects all solutions with score >= minimum
            NL=0 selects only solutions with score >= maximum

        Returns
        -------
        tfsol : np.ndarray
            Candidate transferred solutions of shape (k, d)

        Notes
        -----
        The algorithm fits a Gaussian model on the current population (with added noise
        to ensure positive definiteness) and evaluates historical solutions against this
        model. Solutions that appear "normal" (high PDF value) under the current task's
        distribution are selected for transfer.
        """
        d = curr_pop.shape[1]

        # Add random samples to ensure covariance matrix is positive definite
        n_samples = max(1, int(0.01 * curr_pop.shape[0]))
        rand_mat = np.random.rand(n_samples, d)
        curr_pop_augmented = np.vstack([curr_pop, rand_mat])

        # Fit multivariate Gaussian model
        mean = np.mean(curr_pop_augmented, axis=0)
        cov = np.cov(curr_pop_augmented, rowvar=False)

        # Ensure covariance matrix is positive definite by adding small diagonal
        cov = cov + 1e-6 * np.eye(d)

        # Get unique historical solutions
        his_pop_unique = np.unique(his_pop, axis=0)

        # Ensure dimension compatibility
        if his_pop_unique.shape[1] > d:
            his_pop_unique = his_pop_unique[:, :d]
        elif his_pop_unique.shape[1] < d:
            n_pad = d - his_pop_unique.shape[1]
            his_pop_unique = np.hstack([
                his_pop_unique,
                np.random.rand(his_pop_unique.shape[0], n_pad)
            ])

        # Calculate anomaly scores (PDF values)
        try:
            scores = multivariate_normal.pdf(his_pop_unique, mean=mean, cov=cov)
        except np.linalg.LinAlgError:
            # If covariance is still singular, use diagonal covariance
            var = np.var(curr_pop_augmented, axis=0) + 1e-6
            scores = np.prod(
                np.exp(-0.5 * ((his_pop_unique - mean) ** 2) / var) / np.sqrt(2 * np.pi * var),
                axis=1
            )

        # Sort scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        # Determine threshold based on NL
        if NL == 0:
            threshold = sorted_scores[0]  # Only select maximum
        else:
            idx = min(int(np.ceil(len(sorted_scores) * NL)), len(sorted_scores) - 1)
            threshold = sorted_scores[idx]

        # Select solutions with score >= threshold
        selected_mask = scores >= threshold
        tfsol = his_pop_unique[selected_mask]

        return tfsol
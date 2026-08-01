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
class MTEA_AD:
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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
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

    def __init__(self, problem, n=None, max_nfes=None, TRP=0.1, muc=2.0, mum=5.0,
                 save_data=True, save_path='./Data', name='MTEA-AD', disable_tqdm=True):
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
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MTEA-AD')
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

        # MToP evolves one unified genome of length max(D) per individual and
        # truncates to D(t) only at evaluation time. The genes beyond D(t) take
        # part in crossover/mutation and - decisive for MTEA-AD - in the
        # anomaly-detection model that scores solutions coming from other
        # tasks, so the populations must be kept in the unified space here.
        # Padding is U[0, 1] to match MToP's rand(1, max(D)) initialization.
        pop_decs = space_transfer(problem=problem, decs=decs, type='uni', padding='random')

        # Initialize epsilon (anomaly detection parameter) for each task
        epsilon = np.zeros(nt)

        # MToP increments Algo.Gen once before entering the loop body, so the
        # first generation the reference executes already has Gen == 2.
        gen = 2
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for t in active_tasks:
                # Generate offspring through crossover, mutation, gene swapping
                off_decs = self._generation(pop_decs[t])

                # Knowledge transfer with probability TRP
                tfsol = None
                if np.random.rand() < self.TRP:
                    # Set NL parameter for anomaly detection. MToP tests
                    # Algo.Gen == 1, which is unreachable because Gen is
                    # already 2 in the first loop body; the branch is kept for
                    # fidelity, so in practice NL starts from epsilon = 0.
                    NL = 1.0 if gen == 1 else epsilon[t]

                    # Collect populations from other tasks
                    # (MToP: randperm(T - 1, min(T - 1, 10)) to bound the cost
                    # of the many-task case)
                    kpool = [k for k in range(nt) if k != t]
                    if kpool:
                        kpool = list(np.random.choice(kpool, size=min(len(kpool), 10),
                                                      replace=False))
                        his_pop_dec = np.vstack([pop_decs[k] for k in kpool])

                        # Learn anomaly detection model and get transfer solutions
                        tfsol = np.clip(self._learn_anomaly_detection(off_decs, his_pop_dec, NL),
                                        0.0, 1.0)
                        if tfsol.shape[0] == 0:
                            tfsol = None

                if tfsol is not None:
                    # Evaluate offspring and transferred solutions
                    off_objs, off_cons = evaluation_single(problem, off_decs[:, :dims[t]], t)
                    tf_objs, tf_cons = evaluation_single(problem, tfsol[:, :dims[t]], t)

                    # Merge parent, offspring, and transferred populations
                    merged_objs, merged_decs, merged_cons = vstack_groups(
                        (objs[t], off_objs, tf_objs),
                        (pop_decs[t], off_decs, tfsol),
                        (cons[t], off_cons, tf_cons)
                    )

                    # Elitist selection
                    index = selection_elit(merged_objs, n_per_task[t], merged_cons)
                    objs[t], pop_decs[t], cons[t] = select_by_index(
                        index, merged_objs, merged_decs, merged_cons)

                    # Parameter adaptation via elitism: MToP counts the
                    # survivors whose rank exceeds N + numel(offspring), i.e.
                    # the transferred solutions that entered the new population
                    parent_off_size = n_per_task[t] + off_decs.shape[0]
                    succ_num = int(np.sum(index >= parent_off_size))
                    epsilon[t] = succ_num / tfsol.shape[0]

                    n_eval = off_decs.shape[0] + tfsol.shape[0]
                else:
                    # No knowledge transfer this generation
                    off_objs, off_cons = evaluation_single(problem, off_decs[:, :dims[t]], t)

                    # Merge parent and offspring populations
                    merged_objs, merged_decs, merged_cons = vstack_groups(
                        (objs[t], off_objs), (pop_decs[t], off_decs), (cons[t], off_cons)
                    )

                    # Elitist selection
                    index = selection_elit(merged_objs, n_per_task[t], merged_cons)
                    objs[t], pop_decs[t], cons[t] = select_by_index(
                        index, merged_objs, merged_decs, merged_cons)

                    n_eval = off_decs.shape[0]

                nfes_per_task[t] += n_eval
                pbar.update(n_eval)

                # History is stored in each task's own (real) decision space
                append_history(all_decs[t], pop_decs[t][:, :dims[t]],
                               all_objs[t], objs[t], all_cons[t], cons[t])

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, population):
        """
        Offspring generation matching MTO-Platform ``MTEA_AD.Generation``.

        A random permutation is split in halves and parent i is paired with
        parent i + floor(N/2) for i = 1..ceil(N/2). Each pair yields two
        children (SBX, polynomial mutation, then per-gene swapping with
        probability 0.5) and the single clip to [0, 1] happens only at the
        very end. For odd N this produces N + 1 offspring, exactly as in
        MATLAB (MTEA-AD does not truncate the offspring list).
        """
        n_pop, d = population.shape
        ind_order = np.random.permutation(n_pop)
        half = n_pop // 2
        n_pairs = int(np.ceil(n_pop / 2))
        offspring = np.empty((2 * n_pairs, d))

        count = 0
        for i in range(n_pairs):
            p1 = population[ind_order[i], :]
            p2 = population[ind_order[i + half], :]

            # Crossover then mutation, both unclipped
            off1, off2 = sbx_crossover_unclipped(p1, p2, self.muc)
            off1 = poly_mutation_unclipped(off1, self.mum)
            off2 = poly_mutation_unclipped(off2, self.mum)

            # Gene swapping
            swap_indicator = np.random.rand(d) < 0.5
            temp = off1[swap_indicator].copy()
            off1[swap_indicator] = off2[swap_indicator]
            off2[swap_indicator] = temp

            # Single boundary handling at the end
            offspring[count, :] = np.clip(off1, 0.0, 1.0)
            offspring[count + 1, :] = np.clip(off2, 0.0, 1.0)
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
            Controls the proportion of solutions selected for transfer:
            the threshold is the score ranked at ceil(len(Y) * NL).
            NL=1 selects every solution, NL=0 falls back to MToP's
            "at least one" rule described below.

        Returns
        -------
        tfsol : np.ndarray
            Candidate transferred solutions of shape (k, d)

        Notes
        -----
        The algorithm fits a Gaussian model on the current offspring (with added
        noise to ensure positive definiteness) and evaluates historical solutions
        against this model. Solutions that appear "normal" (high PDF value) under
        the current task's distribution are selected for transfer.
        """
        d = curr_pop.shape[1]

        # MATLAB: nsamples = floor(0.01 * size(curr_pop, 1)), which is 0 for
        # populations smaller than 100 - do not force a sample here.
        n_samples = int(np.floor(0.01 * curr_pop.shape[0]))
        if n_samples > 0:
            curr_pop = np.vstack([curr_pop, np.random.rand(n_samples, d)])

        # Fit multivariate Gaussian model
        # MATLAB: sstd = cov(curr_pop) + (10e-6) * eye(d), and 10e-6 == 1e-5
        mean = np.mean(curr_pop, axis=0)
        cov = np.cov(curr_pop, rowvar=False) + 1e-5 * np.eye(d)

        # Get unique historical solutions (MATLAB unique(..., 'rows') also
        # returns them sorted lexicographically, which np.unique matches)
        dec = np.unique(his_pop, axis=0)

        # Calculate anomaly scores (PDF values)
        try:
            scores = multivariate_normal.pdf(dec[:, :d], mean=mean, cov=cov, allow_singular=True)
        except (np.linalg.LinAlgError, ValueError):
            # If covariance is still degenerate, fall back to a diagonal model
            var = np.var(curr_pop, axis=0) + 1e-5
            scores = np.prod(
                np.exp(-0.5 * ((dec[:, :d] - mean) ** 2) / var) / np.sqrt(2 * np.pi * var),
                axis=1
            )
        scores = np.atleast_1d(scores)
        n_scores = scores.shape[0]

        # Determine the threshold exactly as MToP does
        if NL == 0:
            # MATLAB uses mm = Y(1): the score of the FIRST row of the unique
            # (lexicographically sorted) matrix, not the largest score. It only
            # guarantees that the number of transferred individuals is not 0.
            threshold = scores[0]
        else:
            order = np.argsort(-scores, kind='stable')
            # MATLAB ii(ceil(n * NL)) is 1-based
            idx = int(np.ceil(n_scores * NL)) - 1
            idx = min(max(idx, 0), n_scores - 1)
            threshold = scores[order[idx]]

        # Select solutions with score >= threshold
        return dec[scores >= threshold, :]
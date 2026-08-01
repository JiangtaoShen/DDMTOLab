"""
Strength Pareto Evolutionary Algorithm 2 (SPEA2)

This module implements SPEA2 for multi-objective optimization problems.

References
----------
    [1] Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the Strength Pareto Evolutionary Algorithm For Multiobjective Optimization. In Evolutionary Methods for Design, Optimization and Control with Applications to Industrial Problems. Proceedings of the EUROGEN'2001.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.13
Version: 1.1
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class SPEA2:
    """
    Strength Pareto Evolutionary Algorithm 2 for multi-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=20.0, epsilon=0, save_data=True,
                 save_path='./Data', name='SPEA2', disable_tqdm=True):
        """
        Initialize SPEA2 algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 20.0)
        epsilon : float, optional
            Constraint epsilon value for epsilon-constraint method (default: 0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SPEA2')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.muc = muc
        self.mum = mum
        self.epsilon = epsilon
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SPEA2 algorithm.

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

        # Calculate initial fitness for each task
        fitness = []
        for i in range(nt):
            fitness.append(self._cal_fitness(objs[i], cons[i]))

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Parent selection via binary tournament based on fitness
                matingpool = tournament_selection(2, n_per_task[i], fitness[i])

                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i][matingpool, :], muc=self.muc, mum=self.mum)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs),
                                                          (cons[i], off_cons))

                # Environmental selection to obtain the next population
                selected_indices, fitness[i] = self._spea2_selection(objs[i], cons[i], n_per_task[i])
                objs[i], decs[i], cons[i] = select_by_index(selected_indices, objs[i], decs[i], cons[i])

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _cal_fitness(self, objs, cons):
        """
        Calculate SPEA2 fitness with (epsilon-relaxed) constrained dominance.

        Parameters
        ----------
        objs : ndarray
            Objective values with shape (pop_size, n_objs)
        cons : ndarray or None
            Constraint values with shape (pop_size, n_cons)

        Returns
        -------
        fitness : ndarray
            SPEA2 fitness values with shape (pop_size,). Lower is better;
            values < 1 indicate non-dominated solutions.
        """
        if cons is None:
            return spea2_fitness(objs)

        # Epsilon-constraint handling: violations below epsilon count as feasible
        cv = np.sum(np.maximum(0, cons), axis=1)
        cv[cv < self.epsilon] = 0
        return spea2_fitness(objs, cv[:, None])

    def _spea2_selection(self, objs, cons, N):
        """
        Environmental selection for SPEA2 algorithm.

        Parameters
        ----------
        objs : ndarray
            Objective values with shape (pop_size, n_objs)
        cons : ndarray or None
            Constraint values with shape (pop_size, n_cons)
        N : int
            Number of individuals to select

        Returns
        -------
        selected_indices : ndarray
            Indices of selected individuals (in original order)
        selected_fitness : ndarray
            Fitness values of selected individuals
        """
        # Calculate fitness for all individuals
        fitness = self._cal_fitness(objs, cons)

        # Environmental selection: keep non-dominated solutions (fitness < 1)
        next_selected = fitness < 1

        if np.sum(next_selected) < N:
            # Fill with the best dominated solutions by fitness
            sorted_indices = np.argsort(fitness, kind='stable')
            next_selected[sorted_indices[:N]] = True
        elif np.sum(next_selected) > N:
            # Truncate the non-dominated set by iteratively deleting the most crowded
            candidates = np.where(next_selected)[0]
            keep = spea2_truncation(objs[candidates], N)
            next_selected = np.zeros_like(next_selected)
            next_selected[candidates[keep]] = True

        selected_indices = np.where(next_selected)[0]
        selected_fitness = fitness[selected_indices]

        return selected_indices, selected_fitness

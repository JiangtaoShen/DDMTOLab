"""
Coevolutionary Framework for Constrained Multiobjective Optimization (CCMO)

This module implements CCMO for constrained multi-objective optimization problems.

References
----------
    [1] Tian, Ye, et al. "A Coevolutionary Framework for Constrained Multiobjective Optimization Problems." IEEE Transactions on Evolutionary Computation 25.1 (2021): 102-116.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.14
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Algorithms.STMO.SPEA2 import SPEA2
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class CCMO:
    """
    Coevolutionary Framework for Constrained Multiobjective Optimization.

    CCMO uses two co-evolving populations:
    - Population 1: Optimizes objectives with strict constraint handling (epsilon=0)
    - Population 2: Optimizes objectives with relaxed constraints (epsilon=inf)

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
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=15.0,
                 save_data=True, save_path='./TestData', name='CCMO_test',
                 disable_tqdm=True):
        """
        Initialize CCMO algorithm.

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
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'CCMO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        # Create SPEA2 instance for reusing methods
        self._spea2 = SPEA2(problem, n, max_nfes, muc, mum, epsilon=0,
                            save_data=False, disable_tqdm=True)

    def optimize(self):
        """
        Execute the CCMO algorithm.

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

        # Initialize two populations for each task
        # Population 1: strict constraint handling (epsilon=0)
        decs1 = initialization(problem, n_per_task)
        objs1, cons1 = evaluation(problem, decs1)

        # Population 2: relaxed constraint handling (epsilon=inf)
        decs2 = initialization(problem, n_per_task)
        objs2, cons2 = evaluation(problem, decs2)

        nfes_per_task = [2 * n for n in n_per_task]  # Two populations initialized

        # History tracking uses population 1
        all_decs, all_objs, all_cons = init_history(decs1, objs1, cons1)

        # Calculate initial fitness for both populations using SPEA2 selection
        fitness1 = []
        fitness2 = []
        for i in range(nt):
            # Population 1: epsilon = 0 (strict constraints)
            selected_indices1, selected_fitness1 = self._selection_spea2(
                objs1[i], cons1[i], n_per_task[i], epsilon=0
            )
            objs1[i] = objs1[i][selected_indices1]
            decs1[i] = decs1[i][selected_indices1]
            cons1[i] = cons1[i][selected_indices1] if cons1[i] is not None else None
            fitness1.append(selected_fitness1.copy())

            # Population 2: epsilon = inf (relaxed constraints)
            selected_indices2, selected_fitness2 = self._selection_spea2(
                objs2[i], cons2[i], n_per_task[i], epsilon=np.inf
            )
            objs2[i] = objs2[i][selected_indices2]
            decs2[i] = decs2[i][selected_indices2]
            cons2[i] = cons2[i][selected_indices2] if cons2[i] is not None else None
            fitness2.append(selected_fitness2.copy())

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Parent selection from both populations
                matingpool1 = tournament_selection(2, n_per_task[i], fitness1[i])
                matingpool2 = tournament_selection(2, n_per_task[i], fitness2[i])

                # Generate offspring from both populations
                off_decs1 = ga_generation(decs1[i][matingpool1, :], muc=self.muc, mum=self.mum)
                off_decs2 = ga_generation(decs2[i][matingpool2, :], muc=self.muc, mum=self.mum)

                # Combine offspring from both populations
                off_decs_combined = np.vstack([off_decs1, off_decs2])
                off_objs, off_cons = evaluation_single(problem, off_decs_combined, i)

                # Update both populations by merging with offspring
                objs1[i], decs1[i], cons1[i] = vstack_groups(
                    (objs1[i], off_objs),
                    (decs1[i], off_decs_combined),
                    (cons1[i], off_cons)
                )

                objs2[i], decs2[i], cons2[i] = vstack_groups(
                    (objs2[i], off_objs),
                    (decs2[i], off_decs_combined),
                    (cons2[i], off_cons)
                )

                # Environmental selection for both populations using SPEA2 selection
                # Population 1: epsilon = 0 (strict constraints)
                selected_indices1, selected_fitness1 = self._selection_spea2(
                    objs1[i], cons1[i], n_per_task[i], epsilon=0
                )
                objs1[i] = objs1[i][selected_indices1]
                decs1[i] = decs1[i][selected_indices1]
                cons1[i] = cons1[i][selected_indices1] if cons1[i] is not None else None
                fitness1[i] = selected_fitness1

                # Population 2: epsilon = inf (relaxed constraints)
                selected_indices2, selected_fitness2 = self._selection_spea2(
                    objs2[i], cons2[i], n_per_task[i], epsilon=np.inf
                )
                objs2[i] = objs2[i][selected_indices2]
                decs2[i] = decs2[i][selected_indices2]
                cons2[i] = cons2[i][selected_indices2] if cons2[i] is not None else None
                fitness2[i] = selected_fitness2

                # Update evaluation count (2*N offspring evaluated)
                nfes_per_task[i] += 2 * n_per_task[i]
                pbar.update(2 * n_per_task[i])

                # Update history with population 1 (strict constraints)
                append_history(all_decs[i], decs1[i], all_objs[i], objs1[i], all_cons[i], cons1[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results using population 1
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _selection_spea2(self, objs, cons, N, epsilon=0):
        """
        SPEA2-based environmental selection with epsilon constraint handling.

        This method reuses SPEA2's selection logic with custom epsilon values.

        Parameters
        ----------
        objs : ndarray
            Objective values with shape (pop_size, n_objs)
        cons : ndarray or None
            Constraint values with shape (pop_size, n_cons)
        N : int
            Number of individuals to select
        epsilon : float, optional
            Epsilon threshold for constraint violation (default: 0)

        Returns
        -------
        selected_indices : ndarray
            Indices of selected individuals
        selected_fitness : ndarray
            Fitness values of selected individuals
        """
        # Temporarily set epsilon for constraint violation calculation
        original_epsilon = self._spea2.epsilon
        self._spea2.epsilon = epsilon

        # Use SPEA2's selection method
        selected_indices, selected_fitness = self._spea2._spea2_selection(objs, cons, N)

        # Restore original epsilon
        self._spea2.epsilon = original_epsilon

        return selected_indices, selected_fitness
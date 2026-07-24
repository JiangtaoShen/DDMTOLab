"""
Multistage Evolutionary Algorithm (MSEA)

This module implements MSEA for better diversity preservation in multi-objective optimization.

References
----------
    [1] Tian, Ye, et al. "A multistage evolutionary algorithm for better diversity preservation in multiobjective optimization." IEEE Transactions on Systems, Man, and Cybernetics: Systems, 51.9 (2021): 5880-5894.

Notes
-----
Author: [Your Name]
Email: [Your Email]
Date: 2025.12.12
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MSEA:
    """
    Multistage Evolutionary Algorithm for diversity preservation in multi-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, 3]',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=20.0, save_data=True, save_path='./Data',
                 name='MSEA', disable_tqdm=True):
        """
        Initialize MSEA algorithm.

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
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MSEA_test')
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

    def optimize(self):
        """
        Execute the MSEA algorithm.

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
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs = init_history(decs, objs)

        # Perform initial non-dominated sorting for each task
        front_no = []
        for i in range(nt):
            front_i, _ = nd_sort(objs[i], np.inf)
            front_no.append(front_i)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for task_id in active_tasks:
                # Normalize the population based on the first front
                pop_obj = objs[task_id].copy()
                first_front_mask = front_no[task_id] == 1
                fmax = np.max(pop_obj[first_front_mask], axis=0)
                fmin = np.min(pop_obj[first_front_mask], axis=0)

                # Avoid division by zero
                frange = fmax - fmin
                frange = np.where(frange < 1e-10, 1.0, frange)

                pop_obj_norm = (pop_obj - fmin) / frange

                # Calculate the distance between each two solutions
                distance = squareform(pdist(pop_obj_norm))
                np.fill_diagonal(distance, np.inf)

                # Local search: generate N offspring one by one
                for i in range(n_per_task[task_id]):
                    # Calculate diversity indicator for each solution
                    sorted_dist = np.sort(distance, axis=1)
                    div = sorted_dist[:, 0] + 0.01 * sorted_dist[:, 1]

                    # Stage 1: Determine the evolutionary stage
                    if np.max(front_no[task_id]) > 1:
                        stage = 1  # Convergence stage
                    elif np.min(div) < np.max(div) / 2:
                        stage = 2  # Exploitation stage
                    else:
                        stage = 3  # Exploration stage

                    # Stage 2: Parent selection based on stage
                    # tournament_selection: lower fitness values are better (as in PlatEMO)
                    if stage == 1:
                        # Convergence: select by front number (lower better), then convergence (lower better)
                        convergence = np.sum(pop_obj_norm, axis=1)
                        mating_pool = tournament_selection(2, 2, front_no[task_id], convergence)
                    elif stage == 2:
                        # Exploitation: the most isolated solution (max div) and one with high diversity
                        mating_pool = np.zeros(2, dtype=int)
                        mating_pool[0] = np.argmax(div)
                        mating_pool[1] = tournament_selection(2, 1, -div)[0]
                    else:
                        # Exploration: one with low convergence and one with high diversity
                        mating_pool = np.zeros(2, dtype=int)
                        convergence = np.sum(pop_obj_norm, axis=1)
                        mating_pool[0] = tournament_selection(2, 1, convergence)[0]
                        mating_pool[1] = tournament_selection(2, 1, -div)[0]

                    # Generate one offspring (OperatorGAhalf: SBX child of the ordered
                    # parent pair followed by polynomial mutation; parent order matters)
                    off_dec, _ = crossover(decs[task_id][mating_pool[0], :],
                                           decs[task_id][mating_pool[1], :], mu=self.muc)
                    off_dec = mutation(off_dec, mu=self.mum)[np.newaxis, :]
                    off_obj, _ = evaluation_single(problem, off_dec, task_id)
                    nfes_per_task[task_id] += 1
                    pbar.update(1)
                    off_obj_norm = (off_obj - fmin) / frange

                    # Update front numbers with the new offspring
                    combined_obj_norm = np.vstack([pop_obj_norm, off_obj_norm])
                    new_front = self._update_front_add(combined_obj_norm, front_no[task_id].copy())

                    # Skip if offspring is dominated
                    if new_front[-1] > 1:
                        continue

                    # Calculate distances from offspring to all population members
                    off_dist = np.linalg.norm(pop_obj_norm - off_obj_norm, axis=1)

                    # Stage 3: Determine replacement strategy
                    if np.max(new_front) > 1:
                        stage = 1
                    elif np.min(div) < np.max(div) / 2:
                        stage = 2
                    else:
                        stage = 3

                    # Stage 4: Decide whether to replace and which solution to replace
                    replace = False
                    q = -1

                    if stage == 1:
                        # Convergence stage: replace worst solution in worst front
                        worst_front = np.max(new_front)
                        worse_indices = np.where(new_front[:-1] == worst_front)[0]
                        convergence_worse = np.sum(pop_obj_norm[worse_indices], axis=1)
                        q = worse_indices[np.argmax(convergence_worse)]
                        off_dist[q] = np.inf
                        replace = True
                    elif stage == 2:
                        # Exploitation stage: replace if offspring improves diversity
                        q = np.argmin(div)
                        off_dist[q] = np.inf
                        sorted_off_dist = np.sort(off_dist)
                        off_div = sorted_off_dist[0] + 0.01 * sorted_off_dist[1]
                        if off_div >= div[q]:
                            replace = True
                    else:
                        # Exploration stage: replace nearest if offspring is better in convergence and diversity
                        q = np.argmin(off_dist)
                        off_dist[q] = np.inf
                        sorted_off_dist = np.sort(off_dist)
                        off_div = sorted_off_dist[0] + 0.01 * sorted_off_dist[1]
                        if np.sum(off_obj_norm[0]) <= np.sum(pop_obj_norm[q]) and off_div >= div[q]:
                            replace = True

                    # Perform replacement
                    if replace:
                        # Update front numbers by removing q-th solution and adding offspring
                        new_front_updated = self._update_front_remove(combined_obj_norm, new_front, q)
                        # Reorder: move offspring (last position) to q-th position
                        front_no[task_id] = np.concatenate([
                            new_front_updated[:q],
                            new_front_updated[-1:],
                            new_front_updated[q:-1]
                        ])

                        # Update population
                        decs[task_id][q] = off_dec[0]
                        objs[task_id][q] = off_obj[0]
                        pop_obj_norm[q] = off_obj_norm[0]

                        # Update distances
                        distance[q, :] = off_dist
                        distance[:, q] = off_dist

                append_history(all_decs[task_id], decs[task_id], all_objs[task_id], objs[task_id])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _update_front_add(self, pop_obj, front_no):
        """
        Update front numbers when adding a new solution (last in pop_obj).

        Mirrors PlatEMO's UpdateFront.m (add branch). Note that, as in the
        MATLAB code, weak dominance (all objectives <=) is used, so a
        duplicate of an existing solution is pushed to the next front.

        Parameters
        ----------
        pop_obj : np.ndarray
            Combined objective values of shape (N+1, M), where last solution is new
        front_no : np.ndarray
            Current front numbers of shape (N,)

        Returns
        -------
        new_front_no : np.ndarray
            Updated front numbers of shape (N+1,)
        """
        N, M = pop_obj.shape
        new_front_no = np.append(np.asarray(front_no, dtype=float), 0.0)
        move = np.zeros(N, dtype=bool)
        move[-1] = True
        current_f = 1

        # Locate the front number of the new solution: the first front in which
        # no member weakly dominates it
        while True:
            members = np.where(new_front_no[:N - 1] == current_f)[0]
            if members.size > 0 and np.any(np.all(pop_obj[members] <= pop_obj[-1], axis=1)):
                current_f += 1
            else:
                break

        # Move down the dominated solutions front by front
        while np.any(move):
            next_move = np.zeros(N, dtype=bool)
            members = np.where(new_front_no == current_f)[0]
            if members.size > 0:
                movers = pop_obj[move]
                dominated = np.any(np.all(movers[np.newaxis, :, :] <= pop_obj[members][:, np.newaxis, :],
                                          axis=2), axis=1)
                next_move[members] = dominated
            new_front_no[move] = current_f
            current_f += 1
            move = next_move

        return new_front_no

    def _update_front_remove(self, pop_obj, front_no, x):
        """
        Update front numbers when removing the x-th solution.

        Mirrors PlatEMO's UpdateFront.m (delete branch), using weak dominance
        (all objectives <=) as in the MATLAB code.

        Parameters
        ----------
        pop_obj : np.ndarray
            Objective values of shape (N, M)
        front_no : np.ndarray
            Current front numbers of shape (N,)
        x : int
            Index of solution to remove

        Returns
        -------
        new_front_no : np.ndarray
            Updated front numbers of shape (N-1,)
        """
        N, M = pop_obj.shape
        front_no = np.asarray(front_no, dtype=float).copy()
        move = np.zeros(N, dtype=bool)
        move[x] = True
        current_f = front_no[x] + 1

        while np.any(move):
            next_move = np.zeros(N, dtype=bool)

            # Solutions in the current front weakly dominated by a moved solution
            members = np.where(front_no == current_f)[0]
            if members.size > 0:
                movers = pop_obj[move]
                dominated = np.any(np.all(movers[np.newaxis, :, :] <= pop_obj[members][:, np.newaxis, :],
                                          axis=2), axis=1)
                candidates = members[dominated]

                # They are promoted only if no remaining solution of the previous
                # front weakly dominates them
                if candidates.size > 0:
                    keepers = pop_obj[(front_no == current_f - 1) & ~move]
                    if keepers.shape[0] > 0:
                        blocked = np.any(np.all(keepers[np.newaxis, :, :] <= pop_obj[candidates][:, np.newaxis, :],
                                                axis=2), axis=1)
                        next_move[candidates] = ~blocked
                    else:
                        next_move[candidates] = True

            front_no[move] = current_f - 2
            current_f += 1
            move = next_move

        # Remove the x-th solution
        new_front_no = np.delete(front_no, x)
        return new_front_no
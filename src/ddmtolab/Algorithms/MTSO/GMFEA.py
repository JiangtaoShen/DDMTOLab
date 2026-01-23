"""
Generalized Multifactorial Evolutionary Algorithm (G-MFEA)

This module implements G-MFEA for multi-task optimization with adaptive knowledge transfer.

References
----------
    [1] Ding, Jinliang, et al. "Generalized Multitasking for Evolutionary Optimization of Expensive Problems." IEEE Transactions on Evolutionary Computation 23.1 (2019): 44-58. https://doi.org/10.1109/TEVC.2017.2785351

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.12
Version: 1.0
"""
import time
import copy
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class GMFEA:
    """
    Generalized Multifactorial Evolutionary Algorithm for multi-task optimization.

    This algorithm features:
    - Adaptive knowledge transfer via task-pair specific transfer vectors
    - Dimension shuffling for heterogeneous task alignment
    - Translation strategy based on population centroids

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
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=2.0, mum=5.0,
                 phi=0.1, theta=0.02, top=0.4, save_data=True, save_path='./TestData',
                 name='GMFEA_test', disable_tqdm=True):
        """
        Initialize G-MFEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability for inter-task crossover (default: 0.3)
        muc : float, optional
            Distribution index for SBX crossover (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (default: 5.0)
        phi : float, optional
            Threshold ratio to activate translation (default: 0.1)
        theta : float, optional
            Interval ratio for translation frequency (default: 0.02)
        top : float, optional
            Ratio of top individuals to estimate current optimums (default: 0.4)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'GMFEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.muc = muc
        self.mum = mum
        self.phi = phi
        self.theta = theta
        self.top = top
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the G-MFEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        d_max = max(dims)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt
        max_gen = max_nfes // (n * nt)

        # Center of decision space
        mid_num = 0.5 * np.ones(d_max)

        # Initialize alpha (adaptive coefficient) and mean vectors
        alpha = 0.0
        mean_t = {t: np.zeros(d_max) for t in range(nt)}

        # Initialize transfer matrix (task-pair specific)
        # transfer[t1, t2] is the transfer vector from task t1 to task t2
        transfer = {}
        for t1 in range(nt):
            for t2 in range(nt):
                if t1 != t2:
                    transfer[(t1, t2)] = np.zeros(d_max)

        # Initialize dimension shuffling orders (task-pair specific)
        # inorder[t1, t2] stores the permutation for aligning t1 and t2
        inorder = {}
        for t1 in range(nt - 1):
            for t2 in range(t1 + 1, nt):
                inorder[(t1, t2)] = np.random.permutation(d_max)

        # Initialize population in unified space
        pop_decs = np.random.rand(n * nt, d_max)
        pop_objs = np.full((n * nt, 1), np.inf)
        pop_cvs = np.full((n * nt, 1), 0.0)
        pop_sfs = np.zeros((n * nt, 1), dtype=int)  # Skill factors

        # Assign skill factors (n individuals per task)
        for t in range(nt):
            pop_sfs[t * n:(t + 1) * n] = t

        # Get per-task populations
        pop_dec_per_task = {t: pop_decs[pop_sfs.flatten() == t].copy() for t in range(nt)}

        # Initial dimension shuffling: align lower-dim populations with higher-dim ones
        for t1 in range(nt - 1):
            for t2 in range(t1 + 1, nt):
                if dims[t1] > dims[t2]:
                    p1, p2 = t1, t2  # p1 is higher-dim
                else:
                    p1, p2 = t2, t1

                # Borrow genetic material from higher-dim task for lower-dim task
                indices = np.random.randint(0, n, size=n)
                int_pop = pop_dec_per_task[p1][indices].copy()
                int_pop[:, inorder[(t1, t2)][:dims[p2]]] = pop_dec_per_task[p2][:, :dims[p2]]
                pop_dec_per_task[p2] = int_pop

                # Update unified population
                pop_decs[pop_sfs.flatten() == p2] = int_pop

        # Evaluate initial population
        nfes = 0
        for t in range(nt):
            task_indices = np.where(pop_sfs.flatten() == t)[0]
            for idx in task_indices:
                dec_t = pop_decs[idx, :dims[t]].reshape(1, -1)
                obj_t, con_t = evaluation_single(problem, dec_t, t)
                pop_objs[idx] = obj_t[0, 0]
                cv_t = np.sum(np.maximum(0, con_t[0])) if con_t is not None and con_t.size > 0 else 0
                pop_cvs[idx] = cv_t
                nfes += 1

        # Initialize history
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        # Store initial population
        for t in range(nt):
            task_indices = np.where(pop_sfs.flatten() == t)[0]
            all_decs[t].append(pop_decs[task_indices, :dims[t]].copy())
            all_objs[t].append(pop_objs[task_indices].copy())
            all_cons[t].append(pop_cvs[task_indices].copy())

        # Progress bar
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        gen = 1
        while nfes < max_nfes:
            # Generation
            off_decs, off_sfs = self._generation(pop_decs, pop_sfs, transfer, nt, d_max)

            # Evaluate offspring
            off_objs = np.full((len(off_decs), 1), np.inf)
            off_cvs = np.full((len(off_decs), 1), 0.0)

            for i in range(len(off_decs)):
                t = off_sfs[i, 0]
                dec_t = off_decs[i, :dims[t]].reshape(1, -1)
                obj_t, con_t = evaluation_single(problem, dec_t, t)
                off_objs[i] = obj_t[0, 0]
                cv_t = np.sum(np.maximum(0, con_t[0])) if con_t is not None and con_t.size > 0 else 0
                off_cvs[i] = cv_t
                nfes += 1
                pbar.update(1)

                if nfes >= max_nfes:
                    break

            # Selection: merge and select best n per task
            merged_decs = np.vstack([pop_decs, off_decs[:len(off_objs)]])
            merged_objs = np.vstack([pop_objs, off_objs])
            merged_cvs = np.vstack([pop_cvs, off_cvs])
            merged_sfs = np.vstack([pop_sfs, off_sfs[:len(off_objs)]])

            pop_decs, pop_objs, pop_cvs, pop_sfs = self._selection(
                merged_decs, merged_objs, merged_cvs, merged_sfs, n, nt
            )

            # Update per-task populations and ranks
            pop_dec_per_task = {}
            pop_rank_per_task = {}
            for t in range(nt):
                task_indices = np.where(pop_sfs.flatten() == t)[0]
                pop_dec_per_task[t] = pop_decs[task_indices].copy()
                # Sort by CV then objective
                task_objs = pop_objs[task_indices].flatten()
                task_cvs = pop_cvs[task_indices].flatten()
                pop_rank_per_task[t] = np.lexsort((task_objs, task_cvs))

            # Update alpha and mean vectors at specified intervals
            if gen >= self.phi * max_gen and gen % max(1, round(self.theta * max_gen)) == 0:
                alpha = (nfes / max_nfes) ** 2
                for t in range(nt):
                    top_num = max(1, round(self.top * n))
                    top_indices = pop_rank_per_task[t][:top_num]
                    mean_t[t] = np.mean(pop_dec_per_task[t][top_indices], axis=0)

            # Update dimension shuffling and transfer vectors
            for t1 in range(nt - 1):
                for t2 in range(t1 + 1, nt):
                    # New random permutation
                    inorder[(t1, t2)] = np.random.permutation(d_max)

                    if dims[t1] > dims[t2]:
                        p1, p2 = t1, t2  # p1 is higher-dim
                    else:
                        p1, p2 = t2, t1

                    # Borrow genetic material from higher-dim task
                    indices = np.random.randint(0, len(pop_dec_per_task[p1]), size=len(pop_dec_per_task[p2]))
                    int_pop = pop_dec_per_task[p1][indices].copy()
                    int_pop[:, inorder[(t1, t2)][:dims[p2]]] = pop_dec_per_task[p2][:, :dims[p2]]

                    # Re-evaluate the aligned population
                    task_indices = np.where(pop_sfs.flatten() == p2)[0]
                    for i, idx in enumerate(task_indices):
                        pop_decs[idx] = int_pop[i]
                        dec_t = int_pop[i, :dims[p2]].reshape(1, -1)
                        obj_t, con_t = evaluation_single(problem, dec_t, p2)
                        pop_objs[idx] = obj_t[0, 0]
                        cv_t = np.sum(np.maximum(0, con_t[0])) if con_t is not None and con_t.size > 0 else 0
                        pop_cvs[idx] = cv_t
                        nfes += 1
                        pbar.update(1)

                        if nfes >= max_nfes:
                            break

                    # Update per-task population
                    pop_dec_per_task[p2] = int_pop

                    # Calculate transfer vectors
                    # int_mean: mean of p2 mapped to p1's space
                    int_mean = mean_t[p1].copy()
                    int_mean[inorder[(t1, t2)][:dims[p2]]] = mean_t[p2][:dims[p2]]

                    transfer[(p1, p2)] = alpha * (mid_num - mean_t[p1])
                    transfer[(p2, p1)] = alpha * (mid_num - int_mean)

                    if nfes >= max_nfes:
                        break
                if nfes >= max_nfes:
                    break

            # Store history
            for t in range(nt):
                task_indices = np.where(pop_sfs.flatten() == t)[0]
                all_decs[t].append(pop_decs[task_indices, :dims[t]].copy())
                all_objs[t].append(pop_objs[task_indices].copy())
                all_cons[t].append(pop_cvs[task_indices].copy())

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=max_nfes_per_task, all_cons=all_cons,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, pop_decs, pop_sfs, transfer, nt, d_max):
        """
        Generate offspring using assortative mating with transfer.

        Parameters
        ----------
        pop_decs : np.ndarray
            Population decision variables, shape (pop_size, d_max)
        pop_sfs : np.ndarray
            Population skill factors, shape (pop_size, 1)
        transfer : dict
            Transfer vectors, transfer[(t1, t2)] is vector from t1 to t2
        nt : int
            Number of tasks
        d_max : int
            Maximum dimension

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (pop_size, d_max)
        off_sfs : np.ndarray
            Offspring skill factors, shape (pop_size, 1)
        """
        pop_size = len(pop_decs)
        off_decs = np.zeros((pop_size, d_max))
        off_sfs = np.zeros((pop_size, 1), dtype=int)

        # Shuffle for random pairing
        ind_order = np.random.permutation(pop_size)

        count = 0
        for i in range(pop_size // 2):
            p1 = ind_order[i]
            p2 = ind_order[i + pop_size // 2]
            sf1 = pop_sfs[p1, 0]
            sf2 = pop_sfs[p2, 0]

            if sf1 == sf2:
                # Same task: direct crossover
                off_dec1, off_dec2 = crossover(pop_decs[p1], pop_decs[p2], mu=self.muc)
                # Random imitation
                off_sfs[count, 0] = np.random.choice([sf1, sf2])
                off_sfs[count + 1, 0] = np.random.choice([sf1, sf2])

            elif np.random.rand() < self.rmp:
                # Different tasks with RMP: crossover with transfer
                t_dec1 = pop_decs[p1] + transfer[(sf1, sf2)]
                t_dec2 = pop_decs[p2] + transfer[(sf2, sf1)]
                off_dec1, off_dec2 = crossover(t_dec1, t_dec2, mu=self.muc)
                off_dec1 = off_dec1 - transfer[(sf1, sf2)]
                off_dec2 = off_dec2 - transfer[(sf2, sf1)]
                # Random imitation
                off_sfs[count, 0] = np.random.choice([sf1, sf2])
                off_sfs[count + 1, 0] = np.random.choice([sf1, sf2])

            else:
                # Different tasks without transfer: mutation only
                off_dec1 = mutation(pop_decs[p1].copy(), mu=self.mum)
                off_dec2 = mutation(pop_decs[p2].copy(), mu=self.mum)
                # Keep original skill factors
                off_sfs[count, 0] = sf1
                off_sfs[count + 1, 0] = sf2

            # Boundary handling
            off_decs[count] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1] = np.clip(off_dec2, 0, 1)
            count += 2

        return off_decs[:count], off_sfs[:count]

    def _selection(self, all_decs, all_objs, all_cvs, all_sfs, n, nt):
        """
        Environmental selection: keep best n individuals per task.

        Parameters
        ----------
        all_decs : np.ndarray
            All decision variables, shape (total, d_max)
        all_objs : np.ndarray
            All objective values, shape (total, 1)
        all_cvs : np.ndarray
            All constraint violations, shape (total, 1)
        all_sfs : np.ndarray
            All skill factors, shape (total, 1)
        n : int
            Population size per task
        nt : int
            Number of tasks

        Returns
        -------
        pop_decs, pop_objs, pop_cvs, pop_sfs : np.ndarray
            Selected population arrays
        """
        selected_decs = []
        selected_objs = []
        selected_cvs = []
        selected_sfs = []

        for t in range(nt):
            task_indices = np.where(all_sfs.flatten() == t)[0]
            task_decs = all_decs[task_indices]
            task_objs = all_objs[task_indices]
            task_cvs = all_cvs[task_indices]
            task_sfs = all_sfs[task_indices]

            # Sort by CV first, then objective
            sort_indices = np.lexsort((task_objs.flatten(), task_cvs.flatten()))
            top_n = sort_indices[:n]

            selected_decs.append(task_decs[top_n])
            selected_objs.append(task_objs[top_n])
            selected_cvs.append(task_cvs[top_n])
            selected_sfs.append(task_sfs[top_n])

        return (np.vstack(selected_decs), np.vstack(selected_objs),
                np.vstack(selected_cvs), np.vstack(selected_sfs))
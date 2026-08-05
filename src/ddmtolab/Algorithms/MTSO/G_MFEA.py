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
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
def _matlab_round(x):
    """Round half away from zero, matching MATLAB's ``round``."""
    return int(np.floor(np.abs(x) + 0.5) * np.sign(x))


class G_MFEA:
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
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=2.0, mum=5.0,
                 phi=0.1, theta=0.02, top=0.4, save_data=True, save_path='./Data',
                 name='G-MFEA', disable_tqdm=True):
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
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'G-MFEA')
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
        c_max = max(problem.n_cons)
        max_nfes = self.max_nfes * nt
        # Reference generation budget maxFE / (N * T), kept in floating point
        gen_ref = max_nfes / (n * nt)
        theta_gen = max(1, _matlab_round(self.theta * gen_ref))
        top_num = _matlab_round(self.top * n)

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

        def evaluate_group(dec_matrix, sf_vector):
            """Evaluate every row on the task named by its skill factor."""
            m = dec_matrix.shape[0]
            objs = np.zeros((m, 1))
            cons = np.zeros((m, c_max))
            sf_flat = np.asarray(sf_vector).flatten()
            for t in range(nt):
                idx_t = np.where(sf_flat == t)[0]
                if idx_t.size == 0:
                    continue
                objs_t, cons_t = evaluation_single(problem, dec_matrix[idx_t][:, :dims[t]], t, unified=True)
                nfes_per_task[t] += idx_t.size
                objs[idx_t, :] = objs_t[:, :1]
                if c_max > 0:
                    cons[idx_t, :] = cons_t[:, :c_max]
            return objs, cons

        # Initialize population in unified space; skill factor = owning task
        pop_decs = np.vstack(space_transfer(problem=problem, decs=initialization(problem, n),
                                            type='uni', padding='random'))
        pop_sfs = np.repeat(np.arange(nt), n).reshape(-1, 1)

        # Initial dimension shuffling: a task borrows the genes it does not own from the
        # higher-dimensional task of the pair, scattered by a random permutation. The
        # reference reads the *pre-shuffle* decisions of both tasks for every pair.
        pop_snapshot = {t: pop_decs[pop_sfs.flatten() == t].copy() for t in range(nt)}
        for t1 in range(nt - 1):
            for t2 in range(t1 + 1, nt):
                inorder[(t1, t2)] = np.random.permutation(d_max)
                p1, p2 = (t1, t2) if dims[t1] > dims[t2] else (t2, t1)
                indices = np.random.randint(0, pop_snapshot[p1].shape[0], size=pop_snapshot[p2].shape[0])
                int_pop = pop_snapshot[p1][indices].copy()
                int_pop[:, inorder[(t1, t2)][:dims[p2]]] = pop_snapshot[p2][:, :dims[p2]]
                pop_decs[pop_sfs.flatten() == p2] = int_pop

        # Skill factors split the unified population unevenly, so the per-task
        # counts are accumulated inside evaluate_group and reported
        nfes_per_task = [0] * nt

        # Evaluate initial population on its own task
        pop_objs, pop_cons = evaluate_group(pop_decs, pop_sfs)
        nfes = pop_decs.shape[0]

        # Initialize history in the native (per-task) search space
        task_decs, task_cons = self._split_by_task(problem, pop_decs, pop_cons, pop_sfs, nt)
        task_objs = [pop_objs[pop_sfs.flatten() == t] for t in range(nt)]
        all_decs, all_objs, all_cons = init_history(task_decs, task_objs, task_cons)

        # Progress bar
        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        gen = 1
        while nfes < max_nfes:
            # Generation
            off_decs, off_sfs = self._generation(pop_decs, pop_sfs, transfer, nt, d_max)

            # Evaluate offspring
            off_objs, off_cons = evaluate_group(off_decs, off_sfs)
            nfes += off_decs.shape[0]
            pbar.update(off_decs.shape[0])

            # Selection: merge and select best n per task
            merged_decs = np.vstack([pop_decs, off_decs])
            merged_objs = np.vstack([pop_objs, off_objs])
            merged_cons = np.vstack([pop_cons, off_cons])
            merged_sfs = np.vstack([pop_sfs, off_sfs])

            pop_decs, pop_objs, pop_cons, pop_sfs = self._selection(
                merged_decs, merged_objs, merged_cons, merged_sfs, n, nt
            )

            # Record the selected elite. The reference reports Algo.Best, which the
            # decision-variable shuffling below cannot degrade, so the history is taken
            # here rather than after the shuffled population has been re-evaluated.
            task_decs, task_cons = self._split_by_task(problem, pop_decs, pop_cons, pop_sfs, nt)
            task_objs = [pop_objs[pop_sfs.flatten() == t] for t in range(nt)]
            append_history(all_decs, task_decs, all_objs, task_objs, all_cons, task_cons)

            # Update per-task populations and ranks
            pop_snapshot = {}
            pop_rank_per_task = {}
            pop_cvs = np.sum(np.maximum(0, pop_cons), axis=1)
            for t in range(nt):
                task_indices = np.where(pop_sfs.flatten() == t)[0]
                pop_snapshot[t] = pop_decs[task_indices].copy()
                # Sort by CV then objective
                pop_rank_per_task[t] = constrained_sort(pop_objs[task_indices].flatten(),
                                                        pop_cvs[task_indices])

            # Update alpha and mean vectors at specified intervals
            if gen >= self.phi * gen_ref and gen % theta_gen == 0:
                alpha = (nfes / max_nfes) ** 2
                for t in range(nt):
                    top_indices = pop_rank_per_task[t][:top_num]
                    mean_t[t] = np.mean(pop_snapshot[t][top_indices], axis=0)

            # Update dimension shuffling and transfer vectors
            for t1 in range(nt - 1):
                for t2 in range(t1 + 1, nt):
                    # New random permutation
                    inorder[(t1, t2)] = np.random.permutation(d_max)

                    if dims[t1] > dims[t2]:
                        p1, p2 = t1, t2  # p1 is higher-dim
                    else:
                        p1, p2 = t2, t1

                    # Borrow genetic material from the higher-dimensional task; both
                    # operands come from the post-selection snapshot, never from an
                    # already re-shuffled population
                    indices = np.random.randint(0, pop_snapshot[p1].shape[0], size=pop_snapshot[p2].shape[0])
                    int_pop = pop_snapshot[p1][indices].copy()
                    int_pop[:, inorder[(t1, t2)][:dims[p2]]] = pop_snapshot[p2][:, :dims[p2]]

                    # Re-evaluate the aligned population
                    task_indices = np.where(pop_sfs.flatten() == p2)[0]
                    pop_decs[task_indices] = int_pop
                    objs_p2, cons_p2 = evaluation_single(problem, int_pop[:, :dims[p2]], p2, unified=True)
                    pop_objs[task_indices, :] = objs_p2[:, :1]
                    if c_max > 0:
                        pop_cons[task_indices, :] = cons_p2[:, :c_max]
                    nfes += int_pop.shape[0]
                    nfes_per_task[p2] += int_pop.shape[0]
                    pbar.update(int_pop.shape[0])

                    # Calculate transfer vectors
                    # int_mean: mean of p2 mapped to p1's space
                    int_mean = mean_t[p1].copy()
                    int_mean[inorder[(t1, t2)][:dims[p2]]] = mean_t[p2][:dims[p2]]

                    transfer[(p1, p2)] = alpha * (mid_num - mean_t[p1])
                    transfer[(p2, p1)] = alpha * (mid_num - int_mean)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, all_cons=all_cons,
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
        half = pop_size // 2
        n_pairs = int(np.ceil(pop_size / 2))
        off_decs = np.zeros((2 * n_pairs, d_max))
        off_sfs = np.zeros((2 * n_pairs, 1), dtype=int)

        # Shuffle for random pairing: the reference pairs order[i] with
        # order[i + floor(N / 2)] for i = 1..ceil(N / 2)
        ind_order = np.random.permutation(pop_size)

        count = 0
        for i in range(n_pairs):
            p1 = ind_order[i]
            p2 = ind_order[i + half]
            sf1 = pop_sfs[p1, 0]
            sf2 = pop_sfs[p2, 0]
            parent_sfs = (sf1, sf2)

            if sf1 == sf2:
                # Same task: direct crossover
                off_dec1, off_dec2 = sbx_crossover_unclipped(pop_decs[p1], pop_decs[p2], self.muc)
                # Random imitation: each child picks a parent independently
                off_sfs[count, 0] = parent_sfs[np.random.randint(2)]
                off_sfs[count + 1, 0] = parent_sfs[np.random.randint(2)]

            elif np.random.rand() < self.rmp:
                # Different tasks with RMP: translate both parents towards the shared
                # search region, mate there, and translate the children back
                t_dec1 = pop_decs[p1] + transfer[(sf1, sf2)]
                t_dec2 = pop_decs[p2] + transfer[(sf2, sf1)]
                off_dec1, off_dec2 = sbx_crossover_unclipped(t_dec1, t_dec2, self.muc)
                off_dec1 = off_dec1 - transfer[(sf1, sf2)]
                off_dec2 = off_dec2 - transfer[(sf2, sf1)]
                # Random imitation: each child picks a parent independently
                off_sfs[count, 0] = parent_sfs[np.random.randint(2)]
                off_sfs[count + 1, 0] = parent_sfs[np.random.randint(2)]

            else:
                # Different tasks without transfer: mutation only
                off_dec1 = mutation(pop_decs[p1].copy(), mu=self.mum)
                off_dec2 = mutation(pop_decs[p2].copy(), mu=self.mum)
                # Keep original skill factors
                off_sfs[count, 0] = sf1
                off_sfs[count + 1, 0] = sf2

            # Boundary handling (single repair, after the translation is undone)
            off_decs[count] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1] = np.clip(off_dec2, 0, 1)
            count += 2

        return off_decs[:count], off_sfs[:count]

    @staticmethod
    def _split_by_task(problem, pop_decs, pop_cons, pop_sfs, nt):
        """
        Split a unified population into per-task decision and constraint matrices.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        pop_decs : np.ndarray
            Unified decision variables, shape (total, d_max)
        pop_cons : np.ndarray
            Unified constraint values, shape (total, c_max)
        pop_sfs : np.ndarray
            Skill factors, shape (total, 1)
        nt : int
            Number of tasks

        Returns
        -------
        task_decs : list[np.ndarray]
            Decision variables truncated to each task's own dimension
        task_cons : list[np.ndarray]
            Constraint values truncated to each task's own constraint count
        """
        sf_flat = pop_sfs.flatten()
        decs = [pop_decs[sf_flat == t] for t in range(nt)]
        cons = [pop_cons[sf_flat == t] for t in range(nt)]
        return space_transfer(problem=problem, decs=decs, cons=cons, type='real')

    def _selection(self, all_decs, all_objs, all_cons, all_sfs, n, nt):
        """
        Environmental selection: keep best n individuals per task.

        Parameters
        ----------
        all_decs : np.ndarray
            All decision variables, shape (total, d_max)
        all_objs : np.ndarray
            All objective values, shape (total, 1)
        all_cons : np.ndarray
            All constraint values, shape (total, c_max)
        all_sfs : np.ndarray
            All skill factors, shape (total, 1)
        n : int
            Population size per task
        nt : int
            Number of tasks

        Returns
        -------
        pop_decs, pop_objs, pop_cons, pop_sfs : np.ndarray
            Selected population arrays
        """
        selected_decs = []
        selected_objs = []
        selected_cons = []
        selected_sfs = []

        for t in range(nt):
            task_indices = np.where(all_sfs.flatten() == t)[0]
            task_decs = all_decs[task_indices]
            task_objs = all_objs[task_indices]
            task_cons = all_cons[task_indices]
            task_sfs = all_sfs[task_indices]

            # Sort by CV first, then objective, and keep the best n
            top_n = selection_elit(objs=task_objs, n=n, cons=task_cons)

            selected_decs.append(task_decs[top_n])
            selected_objs.append(task_objs[top_n])
            selected_cons.append(task_cons[top_n])
            selected_sfs.append(task_sfs[top_n])

        return (np.vstack(selected_decs), np.vstack(selected_objs),
                np.vstack(selected_cons), np.vstack(selected_sfs))
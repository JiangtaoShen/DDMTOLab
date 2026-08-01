"""
Multiobjective Evolutionary Multitasking via Explicit Autoencoding (MO-EMEA)

This module implements the MO_EMEA algorithm for multi-task multi-objective optimization problems with knowledge transfer.

References
----------
    [1] L. Feng, L. Zhou, J. Zhong, A. Gupta, Y. -S. Ong, K. -C. Tan, and A. K. Qin. "Evolutionary Multitasking via Explicit Autoencoding." IEEE Transactions on Cybernetics, 49(9): 3457-3470, 2019.

Notes
-----
The code is developed in accordance with the MATLAB-based MTO-platform framework.

Author: Jing Wang
Email:
Date: 2026.01.09
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
def selection_spea2(objs, cons, n, epsilon=0.0):
    """
    Environmental selection of SPEA2 (MToP ``Selection_SPEA2``).

    Parameters
    ----------
    objs : np.ndarray
        Objective values of the merged population, shape (n_pop, n_objs)
    cons : np.ndarray or None
        Constraint values of the merged population, shape (n_pop, n_cons)
    n : int
        Target population size
    epsilon : float, optional
        Epsilon-constraint tolerance (default: 0.0)

    Returns
    -------
    index : np.ndarray
        Indices of the surviving individuals, sorted by ascending fitness
    fitness : np.ndarray
        SPEA2 fitness of the surviving individuals, ascending
    """
    n_pop = objs.shape[0]
    n = max(0, min(n, n_pop))
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0)

    if cons is None or cons.size == 0:
        cv = np.zeros(n_pop)
    else:
        cv = np.sum(np.maximum(0, cons), axis=1)
    cv = np.where(cv < epsilon, 0.0, cv)

    fitness = spea2_fitness(objs, cv[:, None])

    next_mask = fitness < 1
    n_selected = int(np.sum(next_mask))

    if n_selected < n:
        order = np.argsort(fitness, kind='stable')
        next_mask = np.zeros(n_pop, dtype=bool)
        next_mask[order[:n]] = True
    elif n_selected > n:
        selected = np.where(next_mask)[0]
        keep = spea2_truncation(objs[selected], n)
        next_mask = np.zeros(n_pop, dtype=bool)
        next_mask[selected[keep]] = True

    index = np.where(next_mask)[0]
    order = np.argsort(fitness[index], kind='stable')
    index = index[order]
    return index, fitness[index]
def mda(curr_pop, his_pop, his_best_solution):
    """
    Marginalized denoising autoencoder mapping (MToP ``mDA``).

    Learns a linear map from the source-domain initial population to the
    target-domain initial population and applies it to the source elites.

    Parameters
    ----------
    curr_pop : np.ndarray
        Target-domain population, shape (n, d_curr)
    his_pop : np.ndarray
        Source-domain population, shape (n, d_his)
    his_best_solution : np.ndarray
        Source-domain elites to map, shape (n_best, d_his)

    Returns
    -------
    inj_solution : np.ndarray
        Mapped solutions in the target domain, shape (n_best, d_curr)
    """
    curr_pop = np.asarray(curr_pop, dtype=float)
    his_pop = np.asarray(his_pop, dtype=float)
    his_best_solution = np.asarray(his_best_solution, dtype=float)

    n_curr, curr_len = curr_pop.shape
    n_his, tmp_len = his_pop.shape
    if n_curr != n_his:
        raise ValueError(
            f"mDA requires equal population sizes, got {n_curr} and {n_his}")

    # Zero-pad the shorter genome so both domains share one width
    if curr_len < tmp_len:
        curr_pop = np.hstack([curr_pop, np.zeros((n_curr, tmp_len - curr_len))])
    elif curr_len > tmp_len:
        his_pop = np.hstack([his_pop, np.zeros((n_his, curr_len - tmp_len))])

    xx = curr_pop.T
    noise = his_pop.T
    d, n = xx.shape

    xxb = np.vstack([xx, np.ones((1, n))])
    noise_xb = np.vstack([noise, np.ones((1, n))])

    Q = noise_xb @ noise_xb.T
    P = xxb @ noise_xb.T
    reg = 1e-5 * np.eye(d + 1)
    reg[-1, -1] = 0.0

    # MATLAB "P / (Q + reg)" is a right division, i.e. solve W (Q+reg) = P
    A = Q + reg
    try:
        W = np.linalg.solve(A.T, P.T).T
    except np.linalg.LinAlgError:
        W = P @ np.linalg.pinv(A)

    # Drop the bias row and column
    W = W[:-1, :-1]

    if curr_len <= tmp_len:
        tmp_solution = (W @ his_best_solution.T).T
        return tmp_solution[:, :curr_len]

    pad = np.zeros((his_best_solution.shape[0], curr_len - tmp_len))
    return (W @ np.hstack([his_best_solution, pad]).T).T


class MO_EMEA:
    """
    Multi-task Multi-objective Evolutionary Multitasking via Explicit Autoencoding (MO_EMEA).

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n=None, max_nfes=None, operator='SP/NS',
                 s_num=None, t_gap=None, mu_c=None, mu_m=None, save_data=True,
                 save_path='./Data', name='MO-EMEA', disable_tqdm=True):
        """
        Initialize MO-EMEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        operator : str, optional
            Selection operator(s) split with '/', e.g., 'SP/NS' (default: 'SP/NS')

            - 'SP': SPEA2 selection
            - 'NS': NSGA-II selection
        s_num : int, optional
            Number of solutions for knowledge transfer (default: 10)
        t_gap : int, optional
            Generation gap for knowledge transfer (default: 10)
        mu_c : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20)
        mu_m : float, optional
            Distribution index for polynomial mutation (PM) (default: 15)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MO-EMEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        # Common parameters
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        self.operator = operator
        self.s_num = s_num if s_num is not None else 10
        self.t_gap = t_gap if t_gap is not None else 10
        self.mu_c = mu_c if mu_c is not None else 20
        self.mu_m = mu_m if mu_m is not None else 15

        self.operators = [op.strip() for op in self.operator.split('/')]
        self.nt = problem.n_tasks
        self.n_per_task = par_list(self.n, self.nt)
        self.max_nfes_per_task = par_list(self.max_nfes, self.nt)

    def optimize(self):
        """
        Execute MO_EMEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = self.nt
        dims = problem.dims
        d_max = max(dims)
        n_per_task = self.n_per_task
        max_nfes_per_task = self.max_nfes_per_task

        # 1. Initialization. MToP evolves one unified genome of length max(D)
        #    per individual and truncates to D(t) only at evaluation time, so
        #    the padded genes take part in crossover/mutation and are carried
        #    along by the autoencoder injection.
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = list(n_per_task)
        pop_decs = space_transfer(problem=problem, decs=decs, type='uni', padding='random')

        # 2. Initial SPEA2 selection (sorts the population by fitness)
        fitness = [None] * nt
        for t in range(nt):
            index, fitness[t] = selection_spea2(objs[t], cons[t], n_per_task[t])
            pop_decs[t] = pop_decs[t][index]
            objs[t] = objs[t][index]
            cons[t] = cons[t][index]

        # MToP records the population inside notTerminated, i.e. after this
        # first selection, so the history starts from the sorted population.
        all_decs, all_objs, all_cons = init_history(
            [pop_decs[t][:, :dims[t]] for t in range(nt)], objs, cons)

        # Autoencoder anchor: the sorted initial population of every task
        init_pop_dec = [pop_decs[t][:, :dims[t]].copy() for t in range(nt)]

        # 3. Progress bar
        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # MToP increments Algo.Gen once inside notTerminated before the loop
        # body runs, so the first executed generation already sees Gen == 2.
        gen = 1

        # 4. Main optimization loop
        while sum(nfes_per_task) < total_nfes:
            gen += 1

            for t in range(nt):
                if nfes_per_task[t] >= max_nfes_per_task[t]:
                    continue

                # 4.1 Binary tournament on the SPEA2 / NSGA-II fitness
                mating_pool = platemo_tournament_selection(2, n_per_task[t], fitness[t])

                # 4.2 GA offspring. MO-EMEA pairs the mating pool
                #     deterministically (i with i + floor(N/2)); the pool is
                #     already randomized by the tournament.
                off_decs = self._generation(pop_decs[t][mating_pool])

                # 4.3 Knowledge transfer via explicit autoencoding
                if self.s_num > 0 and gen % self.t_gap == 0:
                    inject_decs = self._knowledge_transfer(t, init_pop_dec, pop_decs, dims, d_max)
                    if inject_decs.shape[0] > 0:
                        n_inject = min(inject_decs.shape[0], off_decs.shape[0])
                        replace_idx = np.random.permutation(off_decs.shape[0])[:n_inject]
                        off_decs[replace_idx] = inject_decs[:n_inject]

                # 4.4 Evaluate offspring on the task's own decision space
                off_objs, off_cons = evaluation_single(problem, off_decs[:, :dims[t]], t)
                nfes_per_task[t] += off_decs.shape[0]
                pbar.update(off_decs.shape[0])

                # 4.5 Merge parents and offspring
                merged_decs = np.vstack([pop_decs[t], off_decs])
                merged_objs = np.vstack([objs[t], off_objs])
                merged_cons = np.vstack([cons[t], off_cons])

                # 4.6 Environmental selection, alternating over the operator
                #     list exactly as MToP's mod(t - 1, numel(operator)) + 1
                op = self.operators[t % len(self.operators)]
                if op == 'SP':
                    index, fitness[t] = selection_spea2(merged_objs, merged_cons, n_per_task[t])
                elif op == 'NS':
                    rank, _, _ = nsga2_sort(merged_objs, merged_cons)
                    index = np.argsort(rank, kind='stable')[:n_per_task[t]]
                    fitness[t] = np.arange(1, n_per_task[t] + 1, dtype=float)
                else:
                    raise ValueError(f"Unknown MO-EMEA operator '{op}', expected 'SP' or 'NS'")

                pop_decs[t] = merged_decs[index]
                objs[t] = merged_objs[index]
                cons[t] = merged_cons[index]

                # 4.7 Record the maintained population in the task's own space
                append_history(all_decs[t], pop_decs[t][:, :dims[t]],
                               all_objs[t], objs[t], all_cons[t], cons[t])

        # 5. Process results
        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, parent_decs):
        """
        Generate offspring with SBX + polynomial mutation (MToP ``Generation``).

        Parent i is paired with parent i + floor(N/2) for i = 1..ceil(N/2) and
        each pair yields two children; the single clip to [0, 1] happens only
        after mutation. For an odd N this produces N + 1 offspring, exactly as
        in the MATLAB reference (which does not truncate).

        Parameters
        ----------
        parent_decs : np.ndarray
            Mating pool of shape (pop_size, d_max)

        Returns
        -------
        off_decs : np.ndarray
            Offspring of shape (2 * ceil(pop_size / 2), d_max)
        """
        pop_size, dim = parent_decs.shape
        half = pop_size // 2
        n_pairs = int(np.ceil(pop_size / 2))
        off_decs = np.empty((2 * n_pairs, dim))

        count = 0
        for i in range(n_pairs):
            p1 = parent_decs[i]
            p2 = parent_decs[i + half]

            c1, c2 = sbx_crossover_unclipped(p1, p2, self.mu_c)
            c1 = poly_mutation_unclipped(c1, self.mu_m)
            c2 = poly_mutation_unclipped(c2, self.mu_m)

            off_decs[count] = np.clip(c1, 0, 1)
            off_decs[count + 1] = np.clip(c2, 0, 1)
            count += 2

        return off_decs

    def _knowledge_transfer(self, t, init_pop_dec, pop_decs, dims, d_max):
        """
        Build the injected solutions for task ``t`` via explicit autoencoding.

        For every other task k a linear mDA map is learned between the two
        (sorted) initial populations and applied to the current elites of task
        k. The mapped genome keeps only D(t) genes; the remaining unified genes
        are drawn from U[0, 1], matching ``rand(1, max(D) - D(t))`` in MToP.

        Parameters
        ----------
        t : int
            Target task index
        init_pop_dec : list of np.ndarray
            Sorted initial populations of every task, truncated to D(k)
        pop_decs : list of np.ndarray
            Current unified populations of every task
        dims : list of int
            Decision-variable count of every task
        d_max : int
            Unified genome width

        Returns
        -------
        inject_decs : np.ndarray
            Injected solutions of shape (inject_num * (T - 1), d_max)
        """
        nt = len(pop_decs)
        if nt < 2:
            return np.zeros((0, d_max))

        # MATLAB round() is half-away-from-zero, unlike numpy's banker rounding
        inject_num = int(np.floor(self.s_num / (nt - 1) + 0.5))
        if inject_num <= 0:
            return np.zeros((0, d_max))

        chunks = []
        for k in range(nt):
            if k == t:
                continue

            n_take = min(inject_num, pop_decs[k].shape[0])
            his_best_dec = pop_decs[k][:n_take, :dims[k]]

            inject = mda(init_pop_dec[t], init_pop_dec[k], his_best_dec)

            if d_max > dims[t]:
                inject = np.hstack([inject, np.random.rand(inject.shape[0], d_max - dims[t])])
            inject = np.clip(inject, 0, 1)
            chunks.append(inject)

        if not chunks:
            return np.zeros((0, d_max))
        return np.vstack(chunks)

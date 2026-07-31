"""
Multiobjective Multifactorial Evolutionary Algorithm (MO-MFEA)

This module implements MO-MFEA for multi-objective multi-task optimization with knowledge transfer.

References
----------
    [1] Gupta, Abhishek, Yew-Soon Ong, Liang Feng, and Kay Chen Tan. "Multiobjective Multifactorial Optimization in Evolutionary Multitasking." IEEE Transactions on Cybernetics 47, no. 7 (2017): 1652-1665.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.27
Version: 1.1
"""
import time
from tqdm import tqdm
from ddmtolab.Algorithms.STMO.NSGA_II import nsga2_sort, platemo_tournament_selection
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def _sbx_crossover_unclipped(par_dec1, par_dec2, mu):
    """
    Simulated binary crossover (MTO-Platform ``GA_Crossover``).

    Unlike the shared ``crossover`` helper the offspring are NOT clipped to
    [0, 1] here: the MATLAB reference clips once at the end of ``Generation``,
    after polynomial mutation has acted on the raw crossover output.

    Parameters
    ----------
    par_dec1 : np.ndarray
        First parent decision vector, shape (d,)
    par_dec2 : np.ndarray
        Second parent decision vector, shape (d,)
    mu : float
        Distribution index for the crossover

    Returns
    -------
    off_dec1 : np.ndarray
        First offspring decision vector, shape (d,)
    off_dec2 : np.ndarray
        Second offspring decision vector, shape (d,)
    """
    d = par_dec1.shape[0]
    u = np.random.rand(d)
    beta = np.zeros(d)
    mask = u <= 0.5
    beta[mask] = (2 * u[mask]) ** (1 / (mu + 1))
    beta[~mask] = (2 * (1 - u[~mask])) ** (-1 / (mu + 1))
    beta *= (-1.0) ** np.random.randint(0, 2, size=d)
    beta[np.random.rand(d) < 0.5] = 1.0

    off_dec1 = 0.5 * ((1 + beta) * par_dec1 + (1 - beta) * par_dec2)
    off_dec2 = 0.5 * ((1 + beta) * par_dec2 + (1 - beta) * par_dec1)
    return off_dec1, off_dec2


def _poly_mutation_unclipped(dec, mu):
    """
    Polynomial mutation (MTO-Platform ``GA_Mutation``) with probability 1/D per gene.

    Operates on the possibly out-of-bounds crossover output and does NOT clip;
    the caller clips once afterwards, matching the MATLAB reference.

    Parameters
    ----------
    dec : np.ndarray
        Decision vector to mutate, shape (d,)
    mu : float
        Distribution index for the mutation

    Returns
    -------
    dec : np.ndarray
        Mutated decision vector, shape (d,)
    """
    d = dec.shape[0]
    dec = dec.copy()
    prob_m = 1 / d
    for j in range(d):
        if np.random.rand() < prob_m:
            u = np.random.rand()
            if u <= 0.5:
                delta = (2 * u + (1 - 2 * u) * (1 - dec[j]) ** (mu + 1)) ** (1 / (mu + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * dec[j] ** (mu + 1)) ** (1 / (mu + 1))
            dec[j] += delta
    return dec


class MO_MFEA:
    """
    Multiobjective Multifactorial Evolutionary Algorithm for multi-objective multi-task optimization.

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

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=20.0, mum=15.0, save_data=True,
                 save_path='./Data', name='MO-MFEA', disable_tqdm=True):
        """
        Initialize MO-MFEA algorithm.

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
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MO-MFEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MO-MFEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Population lives in the unified [0, 1] space of dimension max(dims); the
        # genes beyond a task's own dimension are initialized at random and keep
        # evolving, exactly as in the MATLAB reference (Dec = rand(1, max(D))).
        decs = space_transfer(problem, initialization(problem, n), type='uni', padding='random')

        objs, cons = [], []
        for t in range(nt):
            obj_t, con_t = evaluation_single(problem, decs[t][:, :dims[t]], t)
            objs.append(obj_t)
            cons.append(con_t)
        nfes = n * nt

        # Sort each task's population by non-dominated rank then crowding distance
        for t in range(nt):
            rank_t, _, _ = nsga2_sort(objs[t], cons[t])
            order = np.argsort(rank_t)
            decs[t], objs[t], cons[t] = decs[t][order], objs[t][order], cons[t][order]

        all_decs, all_objs, all_cons = init_history(
            [decs[t][:, :dims[t]] for t in range(nt)], objs, cons)

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        # Skill factor of the concatenated population: task-ordered blocks of size n
        pop_sfs = np.repeat(np.arange(nt), n)
        # Tournament fitness is the position inside the own (sorted) sub-population
        pool_fitness = np.tile(np.arange(n), nt)

        while nfes < max_nfes:
            # Mating pool drawn from the union of all sub-populations
            mating_pool = platemo_tournament_selection(2, n * nt, pool_fitness)
            par_decs = np.vstack(decs)[mating_pool, :]
            par_sfs = pop_sfs[mating_pool]

            off_decs, off_sfs = self._generation(par_decs, par_sfs)

            for t in range(nt):
                # Evaluation
                mask = off_sfs == t
                if not np.any(mask):
                    continue
                off_decs_t = off_decs[mask, :]
                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_t[:, :dims[t]], t)
                nfes += off_decs_t.shape[0]
                pbar.update(off_decs_t.shape[0])

                # Selection: NSGA-II sorting on the merged parent + offspring pool
                merged_decs, merged_objs, merged_cons = vstack_groups(
                    (decs[t], off_decs_t), (objs[t], off_objs_t), (cons[t], off_cons_t))
                rank_t, _, _ = nsga2_sort(merged_objs, merged_cons)
                index = np.argsort(rank_t)[:n]
                decs[t], objs[t], cons[t] = select_by_index(index, merged_decs, merged_objs, merged_cons)

            append_history(all_decs, [decs[t][:, :dims[t]] for t in range(nt)],
                           all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=max_nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, par_decs, par_sfs):
        """
        Multifactorial offspring generation (MTO-Platform ``MO_MFEA.Generation``).

        Parents are paired deterministically as (i, i + floor(L/2)); the mating
        pool order is already randomized by tournament selection. Assortative
        mating applies SBX + polynomial mutation when both parents share a skill
        factor or when a random draw falls below ``rmp``, and polynomial mutation
        alone otherwise. Each child imitates the skill factor of one of the two
        parents drawn independently at random.

        Parameters
        ----------
        par_decs : np.ndarray
            Mating pool decision variables in unified space, shape (L, d_uni)
        par_sfs : np.ndarray
            Mating pool skill factors, shape (L,)

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables, shape (2 * ceil(L / 2), d_uni)
        off_sfs : np.ndarray
            Offspring skill factors, shape (2 * ceil(L / 2),)
        """
        length, d = par_decs.shape
        half = length // 2
        n_pairs = int(np.ceil(length / 2))
        off_decs = np.empty((2 * n_pairs, d))
        off_sfs = np.empty(2 * n_pairs, dtype=int)

        count = 0
        for i in range(n_pairs):
            p1, p2 = i, i + half
            sf1, sf2 = int(par_sfs[p1]), int(par_sfs[p2])

            if sf1 == sf2 or np.random.rand() < self.rmp:
                # Crossover
                off_dec1, off_dec2 = _sbx_crossover_unclipped(par_decs[p1, :], par_decs[p2, :], self.muc)
                # Mutation
                off_dec1 = _poly_mutation_unclipped(off_dec1, self.mum)
                off_dec2 = _poly_mutation_unclipped(off_dec2, self.mum)
                # Imitation: each child picks one of the two parents independently
                pair = (sf1, sf2)
                off_sfs[count] = pair[np.random.randint(2)]
                off_sfs[count + 1] = pair[np.random.randint(2)]
            else:
                # Mutation only
                off_dec1 = _poly_mutation_unclipped(par_decs[p1, :], self.mum)
                off_dec2 = _poly_mutation_unclipped(par_decs[p2, :], self.mum)
                # Imitation
                off_sfs[count] = sf1
                off_sfs[count + 1] = sf2

            off_decs[count, :] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1, :] = np.clip(off_dec2, 0, 1)
            count += 2

        return off_decs, off_sfs

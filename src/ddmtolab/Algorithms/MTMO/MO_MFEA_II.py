"""
Multiobjective Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation (MO-MFEA-II)

This module implements MO-MFEA-II for multi-objective multi-task optimization with knowledge transfer.

References
----------
    [1] Bali, Kavitesh Kumar, Abhishek Gupta, Yew-Soon Ong, and Puay Siew Tan. "Cognizant Multitasking in Multiobjective Multifactorial Evolution: MO-MFEA-II." IEEE Transactions on Cybernetics 51, no. 4 (2021): 1784-1796.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.16
Version: 1.1
"""
import time
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from ddmtolab.Methods.Algo_Methods.algo_utils import *
class MO_MFEA_II:
    """
    Multiobjective Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation.

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

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=15.0, swap=0.5, save_data=True,
                 save_path='./Data', name='MO-MFEA-II', disable_tqdm=True):
        """
        Initialize MO-MFEA-II.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        swap : float, optional
            Variable swap probability of the uniform crossover applied to the
            two children of an assortative mating (default: 0.5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MO-MFEA-II')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.muc = muc
        self.mum = mum
        self.swap = swap
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MO-MFEA-II algorithm.

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
        # Skill factors split the unified population unevenly, so the per-task
        # counts are tracked alongside the scalar budget counter and reported
        nfes_per_task = [n] * nt

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
            # Learn the RMP matrix online at every generation from the populations
            rmp_matrix = learnRMP(decs, dims)

            # Mating pool drawn from the union of all sub-populations
            mating_pool = platemo_tournament_selection(2, n * nt, pool_fitness)
            par_decs = np.vstack(decs)[mating_pool, :]
            par_sfs = pop_sfs[mating_pool]

            off_decs, off_sfs = self._generation(par_decs, par_sfs, rmp_matrix)

            for t in range(nt):
                # Evaluation
                mask = off_sfs == t
                if not np.any(mask):
                    continue
                off_decs_t = off_decs[mask, :]
                off_objs_t, off_cons_t = evaluation_single(problem, off_decs_t[:, :dims[t]], t)
                nfes += off_decs_t.shape[0]
                nfes_per_task[t] += off_decs_t.shape[0]
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
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _generation(self, par_decs, par_sfs, rmp_matrix):
        """
        Multifactorial offspring generation (MTO-Platform ``MO_MFEA_II.Generation``).

        Parents are paired deterministically as (i, i + floor(L/2)); the mating
        pool order is already randomized by tournament selection. Inter-task
        mating happens with the online learned probability ``rmp_matrix[sf1, sf2]``
        and is followed by a uniform variable swap between the two children.
        Otherwise each parent is crossed with another randomly picked member of
        its own task and the resulting child receives the partner's genes at the
        swapped positions.

        Parameters
        ----------
        par_decs : np.ndarray
            Mating pool decision variables in unified space, shape (L, d_uni)
        par_sfs : np.ndarray
            Mating pool skill factors, shape (L,)
        rmp_matrix : np.ndarray
            Online learned random mating probability matrix, shape (nt, nt)

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
            rmp = rmp_matrix[sf1, sf2]

            if sf1 == sf2 or np.random.rand() < rmp:
                # Crossover
                off_dec1, off_dec2 = sbx_crossover_unclipped(par_decs[p1, :], par_decs[p2, :], self.muc)
                # Mutation
                off_dec1 = poly_mutation_unclipped(off_dec1, self.mum)
                off_dec2 = poly_mutation_unclipped(off_dec2, self.mum)
                # Variable swap (uniform crossover between the two children)
                swap_indicator = np.random.rand(d) >= self.swap
                temp = off_dec2[swap_indicator].copy()
                off_dec2[swap_indicator] = off_dec1[swap_indicator]
                off_dec1[swap_indicator] = temp
                # Imitation: each child picks one of the two parents independently
                pair = (sf1, sf2)
                off_sfs[count] = pair[np.random.randint(2)]
                off_sfs[count + 1] = pair[np.random.randint(2)]
            else:
                # Randomly pick another individual from the same task
                children = []
                for x, p in enumerate((p1, p2)):
                    sf = int(par_sfs[p])
                    find_idx = np.where(par_sfs == sf)[0]
                    idx = find_idx[np.random.randint(find_idx.shape[0])]
                    # Guard against a task contributing a single parent, which
                    # would make the MATLAB rejection loop spin forever
                    while idx == p and find_idx.shape[0] > 1:
                        idx = find_idx[np.random.randint(find_idx.shape[0])]
                    # Crossover
                    child, temp_child = sbx_crossover_unclipped(par_decs[p, :], par_decs[idx, :], self.muc)
                    # Mutation
                    child = poly_mutation_unclipped(child, self.mum)
                    temp_child = poly_mutation_unclipped(temp_child, self.mum)
                    # Variable swap (uniform crossover, one directional)
                    swap_indicator = np.random.rand(d) >= self.swap
                    child[swap_indicator] = temp_child[swap_indicator]
                    children.append(child)
                    # Imitation
                    off_sfs[count + x] = sf
                off_dec1, off_dec2 = children

            off_decs[count, :] = np.clip(off_dec1, 0, 1)
            off_decs[count + 1, :] = np.clip(off_dec2, 0, 1)
            count += 2

        return off_decs, off_sfs


def learnRMP(subpops, vars):
    """
    Learn the random mating probability (RMP) matrix online (MTO-Platform ``learnRMP``).

    A univariate Gaussian model is fitted per task on its population augmented
    with ``floor(0.1 * N)`` uniformly random samples. For every task pair the
    RMP is the minimizer over [0, 1] of the negative log-likelihood of a mixture
    of the two models, perturbed by ``N(0, 0.01)`` noise and clamped to [0, 1].

    Parameters
    ----------
    subpops : list of np.ndarray
        Population of each task in unified space, ``subpops[i]`` has shape (n, d_uni)
    vars : list of int
        Number of decision variables of each task

    Returns
    -------
    rmp_matrix : np.ndarray
        Symmetric RMP matrix of shape (n_tasks, n_tasks) with unit diagonal
    """
    numtasks = len(subpops)
    max_dim = int(np.max(vars))
    rmp_matrix = np.eye(numtasks)

    # Add noise and build probabilistic models
    probmodel = []
    for i in range(numtasks):
        nsamples = subpops[i].shape[0]
        nrandsamples = int(np.floor(0.1 * nsamples))
        rand_mat = np.random.rand(nrandsamples, max_dim)
        combined = np.vstack([subpops[i], rand_mat])
        probmodel.append({
            'nsamples': nsamples,
            'mean': np.mean(combined, axis=0),
            'stdev': np.std(combined, axis=0, ddof=1),
        })

    for i in range(numtasks):
        for j in range(i + 1, numtasks):
            dim = int(min(vars[i], vars[j]))
            popdata = []
            for src in (i, j):
                data = subpops[src][:, :dim]
                probmatrix = np.ones((probmodel[src]['nsamples'], 2))
                for col, mdl in enumerate((i, j)):
                    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
                        pdf = norm.pdf(data, probmodel[mdl]['mean'][:dim], probmodel[mdl]['stdev'][:dim])
                        probmatrix[:, col] = np.prod(pdf, axis=1)
                popdata.append(probmatrix)

            result = minimize_scalar(lambda x: loglik(x, popdata, numtasks), bounds=(0, 1), method='bounded')
            value = max(0.0, result.x + np.random.normal(0, 0.01))
            value = min(value, 1.0)
            rmp_matrix[i, j] = value
            rmp_matrix[j, i] = value

    return rmp_matrix


def loglik(rmp, popdata, ntasks):
    """
    Negative log-likelihood of the two-task mixture model for a candidate RMP.

    Parameters
    ----------
    rmp : float
        Candidate random mating probability in [0, 1]
    popdata : list of np.ndarray
        Two matrices of shape (n_samples, 2) holding, for each population, the
        likelihood of every individual under the two univariate Gaussian models
    ntasks : int
        Total number of tasks in the multi-task problem

    Returns
    -------
    f : float
        Negative log-likelihood, smaller is better
    """
    f = 0.0
    for i in range(2):
        probmatrix = popdata[i].copy()
        for j in range(2):
            if i == j:
                probmatrix[:, j] *= (1 - (0.5 * (ntasks - 1) * rmp / ntasks))
            else:
                probmatrix[:, j] *= 0.5 * (ntasks - 1) * rmp / ntasks
        with np.errstate(divide='ignore', invalid='ignore'):
            f += np.sum(-np.log(np.sum(probmatrix, axis=1)))
    return f

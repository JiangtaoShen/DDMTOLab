"""
Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation (MFEA-II)

This module implements MFEA-II for multi-task optimization with knowledge transfer across tasks.

References
----------
    [1] Bali, Kavitesh Kumar, et al. "Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II." IEEE Transactions on Evolutionary Computation 24.1 (2019): 69-83.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.16
Version: 1.0
"""
import time
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.MTSO.MFEA import mfea_selection
class MFEA_II:
    """
    Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation

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

    def __init__(self, problem, n=None, max_nfes=None, muc=2.0, mum=5.0, swap=0.5, save_data=True, save_path='./Data',
                 name='MFEA-II', disable_tqdm=True):
        """
        Initialize MFEA-II.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        muc : float, optional
            Distribution index for SBX crossover (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (default: 5.0)
        swap : float, optional
            Variable swap probability threshold; a gene is swapped between the two
            children when ``rand() >= swap`` (default: 0.5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-II')
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
        Execute the Multifactorial Evolutionary Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes = self.max_nfes * nt

        # Initialize population and evaluate for each task
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        # Skill factors split the unified population unevenly, so the per-task
        # counts are tracked alongside the scalar budget counter and reported
        nfes_per_task = [n] * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform populations to unified search space for knowledge transfer. The
        # reference initialises the whole unified vector with rand(1, max(D)), so the
        # padded dimensions must be uniform random (not zeros). Constraints are padded
        # separately with zeros so that padding never fabricates a violation.
        pop_decs = space_transfer(problem=problem, decs=decs, type='uni', padding='random')
        _, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni', padding='zero')
        pop_objs = objs

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:

            # Learn RMP matrix online
            rmpMatrix = learnRMP(pop_decs, dims)

            # Merge populations from all tasks into single arrays
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_cons, pop_sfs)

            n_pop, d_max = pop_decs.shape
            half = n_pop // 2
            n_pairs = int(np.ceil(n_pop / 2))

            off_decs = np.zeros((2 * n_pairs, d_max))
            off_objs = np.zeros((2 * n_pairs, pop_objs.shape[1]))
            off_cons = np.zeros((2 * n_pairs, pop_cons.shape[1]))
            off_sfs = np.zeros((2 * n_pairs, 1), dtype=int)

            # Randomly pair individuals for assortative mating: the reference pairs
            # order[i] with order[i + floor(N / 2)] for i = 1..ceil(N / 2)
            ind_order = np.random.permutation(n_pop)

            count = 0
            for i in range(n_pairs):
                p1 = ind_order[i]
                p2 = ind_order[i + half]
                sf1 = pop_sfs[p1].item()
                sf2 = pop_sfs[p2].item()
                parent_sfs = (sf1, sf2)
                rmp_value = rmpMatrix[sf1, sf2]

                # Cross-task transfer: crossover if same task or the learned rmp fires
                if sf1 == sf2 or np.random.rand() < rmp_value:
                    off_dec1, off_dec2 = sbx_crossover_unclipped(pop_decs[p1, :], pop_decs[p2, :], self.muc)
                    off_dec1 = poly_mutation_unclipped(off_dec1, self.mum)
                    off_dec2 = poly_mutation_unclipped(off_dec2, self.mum)
                    # Variable swap (uniform crossover) between the two children
                    swap_mask = np.random.rand(d_max) >= self.swap
                    tmp = off_dec2[swap_mask].copy()
                    off_dec2[swap_mask] = off_dec1[swap_mask]
                    off_dec1[swap_mask] = tmp
                    # Vertical cultural transmission: each child independently imitates
                    # the skill factor of one uniformly chosen parent
                    off_sfs[count] = parent_sfs[np.random.randint(2)]
                    off_sfs[count + 1] = parent_sfs[np.random.randint(2)]
                    children = (off_dec1, off_dec2)
                else:
                    # No transfer: each parent mates with another random individual
                    # drawn from its own task
                    children = []
                    for x, p in enumerate((p1, p2)):
                        sf = pop_sfs[p].item()
                        # Find all individuals with the same skill factor, excluding p
                        same_sf_indices = np.where(pop_sfs.flatten() == sf)[0]
                        same_sf_indices = same_sf_indices[same_sf_indices != p]
                        idx = np.random.choice(same_sf_indices) if same_sf_indices.size > 0 else p

                        off_dec_curr, off_dec_temp = sbx_crossover_unclipped(
                            pop_decs[p, :], pop_decs[idx, :], self.muc)
                        off_dec_curr = poly_mutation_unclipped(off_dec_curr, self.mum)
                        off_dec_temp = poly_mutation_unclipped(off_dec_temp, self.mum)
                        # One-way variable swap: the discarded sibling donates its genes
                        swap_mask = np.random.rand(d_max) >= self.swap
                        off_dec_curr[swap_mask] = off_dec_temp[swap_mask]
                        children.append(off_dec_curr)
                        # Inherit skill factor from parent
                        off_sfs[count + x] = sf

                # Single boundary repair, after crossover / mutation / swap
                off_decs[count, :] = np.clip(children[0], 0, 1)
                off_decs[count + 1, :] = np.clip(children[1], 0, 1)
                count += 2

            # Evaluate every offspring on the task named by its skill factor
            for t in range(nt):
                idx_t = np.where(off_sfs.flatten() == t)[0]
                if idx_t.size == 0:
                    continue
                objs_t, cons_t = evaluation_single(problem, off_decs[idx_t][:, :dims[t]], t, unified=True)
                off_objs[idx_t, :] = objs_t[:, :off_objs.shape[1]]
                if off_cons.shape[1] > 0:
                    off_cons[idx_t, :] = cons_t[:, :off_cons.shape[1]]
                nfes += idx_t.size
                nfes_per_task[t] += idx_t.size
                pbar.update(idx_t.size)

            # Merge parents and offspring populations
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(
                (pop_decs, off_decs), (pop_objs, off_objs), (pop_cons, off_cons), (pop_sfs, off_sfs)
            )

            # Environmental selection: keep best n individuals per task
            pop_decs, pop_objs, pop_cons, pop_sfs = mfea_selection(pop_decs, pop_objs, pop_cons, pop_sfs, n, nt)

            # Transform back to native search space
            decs, cons = space_transfer(problem=problem, decs=pop_decs, cons=pop_cons, type='real')

            append_history(all_decs, decs, all_objs, pop_objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def learnRMP(subpops, vars):
    """
    Learn the relationship matrix (RMP) between multiple tasks.

    Parameters
    ----------
    subpops : list
        List of subpopulations, either as numpy arrays or dicts with 'data' key.
        Each subpopulation contains solution variables for one task.
    vars : list or array-like
        Dimensionality (number of variables) for each task.

    Returns
    -------
    rmpMatrix : np.ndarray
        Symmetric relationship matrix of shape (numtasks, numtasks).
        rmpMatrix[i,j] indicates the similarity between task i and task j.
        Diagonal elements are 1.0, off-diagonal values are in [0, 1].

    Notes
    -----
    The RMP (Relationship Matrix of Problems) quantifies inter-task similarities
    by computing probabilistic overlap between learned Gaussian models.
    Higher RMP values indicate stronger task relationships, enabling better
    knowledge transfer in multi-task optimization.
    """
    # Convert to dict format if needed
    if isinstance(subpops, list) and isinstance(subpops[0], np.ndarray):
        subpops = [{'data': pop} for pop in subpops]

    numtasks = len(subpops)
    maxDim = max(vars)
    rmpMatrix = np.eye(numtasks)

    # Add noise and build probabilistic models
    probmodel = []
    for i in range(numtasks):
        model = {}
        model['nsamples'] = subpops[i]['data'].shape[0]
        nrandsamples = int(np.floor(0.1 * model['nsamples']))

        # Create random samples with maxDim columns
        randMat = np.random.rand(nrandsamples, maxDim)

        # Combine original data (already in the unified maxDim space) with the noise
        combined_data = np.vstack([subpops[i]['data'], randMat])
        model['mean'] = np.mean(combined_data, axis=0)
        model['stdev'] = np.std(combined_data, axis=0, ddof=1)

        probmodel.append(model)

    # Compute pairwise similarities
    for i in range(numtasks):
        for j in range(i + 1, numtasks):
            popdata = [
                {'probmatrix': np.ones((probmodel[i]['nsamples'], 2))},
                {'probmatrix': np.ones((probmodel[j]['nsamples'], 2))}
            ]

            Dim = min(vars[i], vars[j])

            # Likelihood of each member of subpopulation i under model i and model j
            # (product of the univariate normal densities over the first Dim genes;
            # vectorised form of the reference's element-wise double loop)
            data_i = subpops[i]['data'][:, :Dim]
            data_j = subpops[j]['data'][:, :Dim]
            mean_i, std_i = probmodel[i]['mean'][:Dim], probmodel[i]['stdev'][:Dim]
            mean_j, std_j = probmodel[j]['mean'][:Dim], probmodel[j]['stdev'][:Dim]

            popdata[0]['probmatrix'][:, 0] = np.prod(norm.pdf(data_i, mean_i, std_i), axis=1)
            popdata[0]['probmatrix'][:, 1] = np.prod(norm.pdf(data_i, mean_j, std_j), axis=1)
            popdata[1]['probmatrix'][:, 0] = np.prod(norm.pdf(data_j, mean_i, std_i), axis=1)
            popdata[1]['probmatrix'][:, 1] = np.prod(norm.pdf(data_j, mean_j, std_j), axis=1)

            # Optimize to find RMP value (MATLAB fminbnd, whose default TolX is 1e-4)
            result = minimize_scalar(
                lambda x: loglik(x, popdata, numtasks),
                bounds=(0, 1),
                method='bounded',
                options={'xatol': 1e-4}
            )

            rmp_value = max(0, result.x + np.random.normal(0, 0.01))
            rmp_value = min(rmp_value, 1)

            rmpMatrix[i, j] = rmp_value
            rmpMatrix[j, i] = rmp_value

    return rmpMatrix


def loglik(rmp, popdata, ntasks):
    """
    Compute the negative log-likelihood for a given RMP value.

    Parameters
    ----------
    rmp : float
        Relationship matrix parameter value in [0, 1] to evaluate.
        Represents the strength of inter-task relationship.
    popdata : list
        List of dicts, each containing 'probmatrix' of shape (nsamples, 2).
        probmatrix[:, 0] are probabilities under own task model,
        probmatrix[:, 1] are probabilities under other task model.
    ntasks : int
        Total number of tasks in the multi-task problem.

    Returns
    -------
    f : float
        Negative log-likelihood value. Lower values indicate better fit
        of the RMP parameter to the observed probability distributions.

    Notes
    -----
    This function is used as the objective in optimization to find the optimal
    RMP value that maximizes the likelihood of observing the population data
    under a mixture model with inter-task knowledge transfer.
    """
    f = 0

    # Make a copy to avoid modifying the original
    popdata_copy = [{'probmatrix': pop['probmatrix'].copy()} for pop in popdata]

    for i in range(2):
        for j in range(2):
            if i == j:
                popdata_copy[i]['probmatrix'][:, j] *= (1 - (0.5 * (ntasks - 1) * rmp / ntasks))
            else:
                popdata_copy[i]['probmatrix'][:, j] *= 0.5 * (ntasks - 1) * rmp / ntasks

        # Compute negative log-likelihood
        f += np.sum(-np.log(np.sum(popdata_copy[i]['probmatrix'], axis=1)))

    return f
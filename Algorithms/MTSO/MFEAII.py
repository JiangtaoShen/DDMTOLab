"""
Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation (MFEA-II)

This module implements MFEA-II for multi-task optimization with knowledge transfer across tasks.

References
----------
    [1] Bali, Kavitesh Kumar, et al. "Multifactorial evolutionary algorithm with online transfer parameter \
        estimation: MFEA-II." IEEE Transactions on Evolutionary Computation 24.1 (2019): 69-83.

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
from Methods.Algo_Methods.algo_utils import *
from Algorithms.MTSO.MFEA import mfea_selection


class MFEAII:
    """
    Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation

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

    def __init__(self, problem, n=None, max_nfes=None, save_data=True, save_path='./TestData',
                 name='MFEAII_test', disable_tqdm=True):
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
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MFEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
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
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize population and evaluate for each task
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform populations to unified search space for knowledge transfer
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons, type='uni')
        pop_objs = objs

        # Skill factor indicates which task each individual belongs to
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}", disable=self.disable_tqdm)

        while nfes < max_nfes:

            # Learn RMP matrix online
            rmpMatrix = learnRMP(pop_decs, dims)

            # Merge populations from all tasks into single arrays
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(pop_decs, pop_objs, pop_cons, pop_sfs)

            off_decs = np.zeros_like(pop_decs)
            off_objs = np.zeros_like(pop_objs)
            off_cons = np.zeros_like(pop_cons)
            off_sfs = np.zeros_like(pop_sfs)

            # Randomly pair individuals for assortative mating
            shuffled_index = np.random.permutation(pop_decs.shape[0])

            for i in range(0, len(shuffled_index), 2):
                p1 = shuffled_index[i]
                p2 = shuffled_index[i + 1]
                sf1 = pop_sfs[p1].item()
                sf2 = pop_sfs[p2].item()
                rmp_value = rmpMatrix[sf1, sf2]

                # Cross-task transfer: crossover if same task or rmp condition met
                if sf1 == sf2 or np.random.rand() < rmp_value:
                    off_dec1, off_dec2 = crossover(pop_decs[p1, :], pop_decs[p2, :], mu=2)
                    off_decs[i, :] = off_dec1
                    off_decs[i + 1, :] = off_dec2
                    off_sfs[i] = np.random.choice([sf1, sf2])
                    off_sfs[i + 1] = sf1 if off_sfs[i] == sf2 else sf2
                else:
                    # No transfer: randomly pick individuals from same task for crossover
                    for x, p in enumerate([p1, p2]):
                        sf = pop_sfs[p].item()
                        # Find all individuals with the same skill factor
                        same_sf_indices = np.where(pop_sfs.flatten() == sf)[0]
                        # Remove current individual from candidates
                        same_sf_indices = same_sf_indices[same_sf_indices != p]
                        # Randomly select another individual from the same task
                        idx = np.random.choice(same_sf_indices)

                        # Crossover with the selected individual
                        off_dec_curr, _ = crossover(pop_decs[p, :], pop_decs[idx, :], mu=2)
                        off_dec_curr = mutation(off_dec_curr, mu=5)
                        off_decs[i + x, :] = off_dec_curr
                        # Inherit skill factor from parent
                        off_sfs[i + x] = sf

                # Trim to task dimensionality and evaluate offspring
                task_idx1 = off_sfs[i].item()
                task_idx2 = off_sfs[i + 1].item()

                off_dec1_trimmed = off_decs[i, :dims[task_idx1]]
                off_dec2_trimmed = off_decs[i + 1, :dims[task_idx2]]
                off_objs[i, :], off_cons[i, :] = evaluation_single(problem, off_dec1_trimmed, task_idx1)
                off_objs[i + 1, :], off_cons[i + 1, :] = evaluation_single(problem, off_dec2_trimmed, task_idx2)

            # Merge parents and offspring populations
            pop_decs, pop_objs, pop_cons, pop_sfs = vstack_groups(
                (pop_decs, off_decs), (pop_objs, off_objs), (pop_cons, off_cons), (pop_sfs, off_sfs)
            )

            # Environmental selection: keep best n individuals per task
            pop_decs, pop_objs, pop_cons, pop_sfs = mfea_selection(pop_decs, pop_objs, pop_cons, pop_sfs, n, nt)

            # Transform back to native search space
            decs, cons = space_transfer(problem=problem, decs=pop_decs, cons=pop_cons, type='real')

            nfes += n * nt
            pbar.update(n * nt)

            append_history(all_decs, decs, all_objs, pop_objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=max_nfes_per_task,
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

        # Pad subpops data to maxDim with ZEROS (to match MATLAB behavior)
        current_data = subpops[i]['data']
        padded_data = current_data

        # Combine original data with random samples
        combined_data = np.vstack([padded_data, randMat])
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

            # Compute probabilities for population i
            for k in range(probmodel[i]['nsamples']):
                for l in range(Dim):
                    popdata[0]['probmatrix'][k, 0] *= norm.pdf(
                        subpops[i]['data'][k, l],
                        probmodel[i]['mean'][l],
                        probmodel[i]['stdev'][l]
                    )
                    popdata[0]['probmatrix'][k, 1] *= norm.pdf(
                        subpops[i]['data'][k, l],
                        probmodel[j]['mean'][l],
                        probmodel[j]['stdev'][l]
                    )

            # Compute probabilities for population j
            for k in range(probmodel[j]['nsamples']):
                for l in range(Dim):
                    popdata[1]['probmatrix'][k, 0] *= norm.pdf(
                        subpops[j]['data'][k, l],
                        probmodel[i]['mean'][l],
                        probmodel[i]['stdev'][l]
                    )
                    popdata[1]['probmatrix'][k, 1] *= norm.pdf(
                        subpops[j]['data'][k, l],
                        probmodel[j]['mean'][l],
                        probmodel[j]['stdev'][l]
                    )

            # Optimize to find RMP value
            result = minimize_scalar(
                lambda x: loglik(x, popdata, numtasks),
                bounds=(0, 1),
                method='bounded'
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
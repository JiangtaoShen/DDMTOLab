"""
Genetic Algorithm (GA)

This module implements the Genetic Algorithm for single-objective optimization problems.

References
----------
    [1] David E. Goldberg. Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley, Reading, MA, 1989.

    [2] John H. Holland. Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence. University of Michigan Press, Ann Arbor, MI, 1st edition, 1975. Reprinted by MIT Press in 1992.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.11.11
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


def _sbx_crossover_unclipped(par_dec1, par_dec2, mu):
    """
    Simulated binary crossover (MTO-Platform GA_Crossover).

    Unlike the shared ``crossover`` helper, offspring are NOT clipped to
    [0, 1] here: the MATLAB reference clips only once at the end of
    Generation, after mutation has acted on the raw crossover output.
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
    Polynomial mutation (MTO-Platform GA_Mutation) with prob 1/D per gene.

    Operates on the possibly out-of-bounds crossover output and does NOT
    clip; the caller clips once afterwards, matching the MATLAB reference.
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


def _ga_generation_matlab(parents, muc, mum):
    """
    Offspring generation matching MTO-Platform GA.Generation exactly.

    A random permutation is split in halves and parent i is paired with
    parent i + floor(N/2) for i = 1..ceil(N/2); each pair yields two
    children (SBX then polynomial mutation), and the single clip to
    [0, 1] happens only after mutation. For odd N this produces N + 1
    offspring (the middle permutation index is used twice), exactly as
    in MATLAB.
    """
    n, d = parents.shape
    order = np.random.permutation(n)
    half = n // 2
    n_pairs = int(np.ceil(n / 2))
    offdecs = np.empty((2 * n_pairs, d))
    count = 0
    for i in range(n_pairs):
        p1 = parents[order[i], :]
        p2 = parents[order[i + half], :]
        c1, c2 = _sbx_crossover_unclipped(p1, p2, muc)
        c1 = _poly_mutation_unclipped(c1, mum)
        c2 = _poly_mutation_unclipped(c2, mum)
        offdecs[count] = np.clip(c1, 0, 1)
        offdecs[count + 1] = np.clip(c2, 0, 1)
        count += 2
    return offdecs


class GA:
    """
    Genetic Algorithm for single-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
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

    def __init__(self, problem, n=None, max_nfes=None, muc=2.0, mum=5.0, save_data=True, save_path='./Data',
                 name='GA', disable_tqdm=True):
        """
        Initialize Genetic Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 2.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 5.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'GA_test')
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
        Execute the Genetic Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population in [0,1] space and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Generate offspring through crossover and mutation
                off_decs = _ga_generation_matlab(decs[i], self.muc, self.mum)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)
                n_off = off_decs.shape[0]

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups(
                    (objs[i], off_objs),
                    (decs[i], off_decs),
                    (cons[i], off_cons)
                )

                # Calculate constraint violations
                cvs = np.sum(np.maximum(0, cons[i]), axis=1)

                # Selection based on constraint violation first, then objective
                # Sort by constraint violation (ascending), then by objective (ascending)
                sort_indices = np.lexsort((objs[i].flatten(), cvs))

                # Select top n_per_task[i] individuals
                index = sort_indices[:n_per_task[i]]

                objs[i], decs[i], cons[i] = select_by_index(index,objs[i], decs[i], cons[i])

                nfes_per_task[i] += n_off
                pbar.update(n_off)

                # Append current population to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
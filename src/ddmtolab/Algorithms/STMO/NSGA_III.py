"""
Nondominated Sorting Genetic Algorithm III (NSGA-III)

This module implements NSGA-III for many-objective optimization problems.

References
----------
    [1] Deb, Kalyanmoy, and Himanshu Jain. "An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: Solving problems with box constraints." IEEE Transactions on Evolutionary Computation 18.4 (2014): 577-601.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.12
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class NSGA_III:
    """
    Nondominated Sorting Genetic Algorithm III for many-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=20.0, save_data=True, save_path='./Data',
                 name='NSGA-III', disable_tqdm=True):
        """
        Initialize NSGA-III algorithm.

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
            Name for the experiment (default: 'NSGA-III_test')
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
        Execute the NSGA-III algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        no = problem.n_objs
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate uniformly distributed reference points for each task
        Z = []
        for i in range(nt):
            z_i, n = uniform_point(n_per_task[i], no[i])
            Z.append(z_i)
            n_per_task[i] = n

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize ideal point (minimum objective values among feasible solutions).
        # PlatEMO keeps Zmin empty until a feasible solution appears; an empty Zmin
        # is replaced by ones(1,M) inside the environmental selection.
        Zmin = []
        for i in range(nt):
            feasible_mask = np.all(cons[i] <= 0, axis=1)
            if np.any(feasible_mask):
                Zmin.append(np.min(objs[i][feasible_mask], axis=0))
            else:
                Zmin.append(None)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Calculate constraint violation for parent selection
                CV = np.sum(np.maximum(0, cons[i]), axis=1)

                # Parent selection via binary tournament based on constraint violation
                # (uniformly random among tied candidates, as in PlatEMO)
                matingpool = platemo_tournament_selection(2, n_per_task[i], CV)

                # Generate offspring through crossover and mutation
                off_decs = ga_generation(decs[i][matingpool, :], muc=self.muc, mum=self.mum)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Update ideal point with feasible offspring
                feasible_mask = np.all(off_cons <= 0, axis=1)
                if np.any(feasible_mask):
                    zmin_off = np.min(off_objs[feasible_mask], axis=0)
                    Zmin[i] = zmin_off if Zmin[i] is None else np.minimum(Zmin[i], zmin_off)

                # Merge parent and offspring populations
                objs[i], decs[i], cons[i] = vstack_groups((objs[i], off_objs), (decs[i], off_decs),
                                                          (cons[i], off_cons))

                # Environmental selection
                objs[i], decs[i], cons[i] = self._environmental_selection(
                    objs[i], decs[i], cons[i], n_per_task[i], Z[i], Zmin[i]
                )

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results

    def _environmental_selection(self, objs, decs, cons, N, Z, Zmin):
        """
        Environmental selection based on non-dominated sorting and reference points.

        Parameters
        ----------
        objs : np.ndarray
            Objective values of shape (pop_size, M)
        decs : np.ndarray
            Decision variables of shape (pop_size, D)
        cons : np.ndarray
            Constraint values of shape (pop_size, K)
        N : int
            Number of solutions to select
        Z : np.ndarray
            Reference points of shape (NZ, M)
        Zmin : np.ndarray or None
            Ideal point of shape (M,), or None if no feasible solution has been
            found yet (replaced by ones, as in PlatEMO)

        Returns
        -------
        selected_objs : np.ndarray
            Selected objective values
        selected_decs : np.ndarray
            Selected decision variables
        selected_cons : np.ndarray
            Selected constraint values
        """
        if Zmin is None:
            Zmin = np.ones(objs.shape[1])

        # Perform non-dominated sorting
        front_no, max_front = nd_sort(objs, cons, N)

        # Select all solutions in fronts before the last front
        Next = front_no < max_front

        # If we haven't filled N solutions, select from the last front
        if np.sum(Next) < N:
            Last = np.where(front_no == max_front)[0]
            objs_before_last = objs[Next]
            objs_last = objs[Last]

            # Select K solutions from the last front
            K = N - np.sum(Next)
            Choose = self._last_selection(objs_before_last, objs_last, K, Z, Zmin)

            # Mark selected solutions from last front
            selected_from_last = Last[Choose]
            Next[selected_from_last] = True

        # Return selected solutions
        selected_objs = objs[Next]
        selected_decs = decs[Next]
        selected_cons = cons[Next]

        return selected_objs, selected_decs, selected_cons

    def _last_selection(self, objs_before, objs_last, K, Z, Zmin):
        """
        Select K solutions from the last front using reference point association.

        Parameters
        ----------
        objs_before : np.ndarray
            Objectives of solutions in fronts before the last front of shape (N1, M)
        objs_last : np.ndarray
            Objectives of solutions in the last front of shape (N2, M)
        K : int
            Number of solutions to select from the last front
        Z : np.ndarray
            Reference points of shape (NZ, M)
        Zmin : np.ndarray
            Ideal point of shape (M,)

        Returns
        -------
        Choose : np.ndarray
            Boolean array indicating selected solutions from the last front
        """
        # Combine all objectives and translate by ideal point
        PopObj = np.vstack([objs_before, objs_last]) - Zmin
        N, M = PopObj.shape
        N1 = objs_before.shape[0]
        N2 = objs_last.shape[0]
        NZ = Z.shape[0]

        # Normalization
        # Step 1: Detect extreme points
        Extreme = np.zeros(M, dtype=int)
        w = np.eye(M) + 1e-6

        for i in range(M):
            # Find the solution that minimizes the maximum ratio to weight vector
            ratios = PopObj / w[i]
            max_ratios = np.max(ratios, axis=1)
            Extreme[i] = np.argmin(max_ratios)

        # Step 2: Calculate intercepts of the hyperplane constructed by the
        # extreme points and the axes. PlatEMO falls back to the maximum
        # objective values only when the intercepts contain NaN (a singular
        # system in NumPy raises instead of returning NaN/Inf).
        try:
            Hyperplane = np.linalg.solve(PopObj[Extreme, :], np.ones(M))
            with np.errstate(divide='ignore'):
                a = 1.0 / Hyperplane
            if np.any(np.isnan(a)):
                a = np.max(PopObj, axis=0)
        except np.linalg.LinAlgError:
            a = np.max(PopObj, axis=0)

        # Step 3: Normalize objectives
        with np.errstate(divide='ignore', invalid='ignore'):
            PopObj = PopObj / a

        # Associate each solution with reference points
        # Calculate cosine distance to each reference point
        cosine = 1 - cdist(PopObj, Z, metric='cosine')

        # Calculate perpendicular distance to reference vectors
        norm_PopObj = np.linalg.norm(PopObj, axis=1, keepdims=True)
        Distance = norm_PopObj * np.sqrt(1 - cosine ** 2)

        # Associate each solution with its nearest reference point
        pi = np.argmin(Distance, axis=1)
        d = np.min(Distance, axis=1)

        # Calculate the number of associated solutions (excluding last front) for each reference point
        rho = np.bincount(pi[:N1], minlength=NZ)

        # Environmental selection
        Choose = np.zeros(N2, dtype=bool)
        Zchoose = np.ones(NZ, dtype=bool)

        # Select K solutions one by one
        while np.sum(Choose) < K:
            # Select the least crowded reference point
            Temp = np.where(Zchoose)[0]
            if len(Temp) == 0:
                break

            # Find reference points with minimum associated solutions
            min_rho = np.min(rho[Temp])
            Jmin = Temp[rho[Temp] == min_rho]

            # Randomly select one from the least crowded reference points
            j = Jmin[np.random.randint(len(Jmin))]

            # Find solutions in last front associated with reference point j
            I = np.where((~Choose) & (pi[N1:] == j))[0]

            if len(I) > 0:
                # Select one solution associated with this reference point
                if rho[j] == 0:
                    # If no solution associated yet, select the closest one
                    s = np.argmin(d[N1 + I])
                else:
                    # Otherwise, randomly select one
                    s = np.random.randint(len(I))

                Choose[I[s]] = True
                rho[j] += 1
            else:
                # No solution associated with this reference point, mark it as unavailable
                Zchoose[j] = False

        return Choose


def platemo_tournament_selection(K, N, *fitness):
    """
    Exact port of PlatEMO's TournamentSelection.

    Candidates are compared lexicographically on the given fitness keys
    (lower values are better). Solutions with identical fitness values share
    the same rank, so a tournament among tied candidates is decided by the
    (random) draw order, i.e. uniformly at random. In particular, when all
    fitness values are equal (e.g. all-zero constraint violations), mating
    selection is uniformly random, as in PlatEMO.

    Parameters
    ----------
    K : int
        Tournament size
    N : int
        Number of parents to select
    *fitness : np.ndarray
        One or more fitness vectors of equal length (primary key first)

    Returns
    -------
    index : np.ndarray
        Indices of the selected parents, shape (N,)
    """
    fits = np.column_stack([np.asarray(f, dtype=float).ravel() for f in fitness])
    _, loc = np.unique(fits, axis=0, return_inverse=True)
    loc = loc.ravel()
    parents = np.random.randint(0, fits.shape[0], size=(K, N))
    best = np.argmin(loc[parents], axis=0)
    return parents[best, np.arange(N)]
"""
Two-Archive Algorithm 2 (Two_Arch2)

This module implements Two_Arch2 for many-objective optimization problems.

References
----------
    [1] Wang, H., Jiao, L., & Yao, X. (2015). Two_Arch2: An improved two-archive algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 19(4), 524-541.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.13
Version: 1.1
"""
from tqdm import tqdm
import time
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class TwoArch2:
    """
    Two-Archive Algorithm 2 for many-objective optimization.

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
        'n_cons': '0',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, CA_size=None, p=None, save_data=True,
                 save_path='./Data', name='Two_Arch2', disable_tqdm=True):
        """
        Initialize Two_Arch2 algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        CA_size : int or None, optional
            Convergence archive size (default: None, will be set to population size)
        p : float or None, optional
            Parameter for fractional distance (default: None, will be set to 1/M)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'Two_Arch2_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.CA_size = CA_size
        self.p = p
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the Two_Arch2 algorithm.

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

        # Initialize archives for each task
        CAs = []  # Convergence Archive
        DAs = []  # Diversity Archive
        CA_sizes = []
        ps = []

        for i in range(nt):
            # Set CA size and p parameter for this task
            CA_sizes.append(self.CA_size if self.CA_size is not None else n_per_task[i])
            ps.append(self.p if self.p is not None else 1.0 / objs[i].shape[1])

            # Initialize archives
            CA_i = self._update_CA(None, objs[i], decs[i], CA_sizes[i])
            DA_i = self._update_DA(None, objs[i], decs[i], n_per_task[i], ps[i])

            CAs.append(CA_i)
            DAs.append(DA_i)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Get CA and DA for current task
                CA_objs_i, CA_decs_i = CAs[i]
                DA_objs_i, DA_decs_i = DAs[i]

                # Mating selection
                parentC_decs, parentM_decs = self._mating_selection(
                    CA_objs_i, CA_decs_i, DA_objs_i, DA_decs_i, n_per_task[i]
                )

                # ParentC: SBX crossover only (proC=1, disC=20, proM=0),
                # pairing CA tournament winners with random DA members
                off_decs_C = operator_ga_real(parentC_decs, 1.0, 20.0, 0.0, 0.0)

                # ParentM: polynomial mutation only (proC=0, proM=1, disM=20)
                off_decs_M = operator_ga_real(parentM_decs, 0.0, 0.0, 1.0, 20.0)

                # Combine offspring
                off_decs = np.vstack([off_decs_C, off_decs_M])
                off_objs, _ = evaluation_single(problem, off_decs, i)

                # Update archives
                CAs[i] = self._update_CA((CA_objs_i, CA_decs_i), off_objs, off_decs, CA_sizes[i])
                DAs[i] = self._update_DA((DA_objs_i, DA_decs_i), off_objs, off_decs, n_per_task[i], ps[i])

                # Update main population with DA for tracking
                objs[i], decs[i] = DAs[i]

                nfes_per_task[i] += off_decs.shape[0]
                pbar.update(off_decs.shape[0])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results (using DA as final population)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        results.CAs = CAs
        results.DAs = DAs

        return results

    def _mating_selection(self, CA_objs, CA_decs, DA_objs, DA_decs, N):
        """
        Mating selection of Two_Arch2.

        Parameters
        ----------
        CA_objs : ndarray
            Convergence archive objectives
        CA_decs : ndarray
            Convergence archive decisions
        DA_objs : ndarray
            Diversity archive objectives
        DA_decs : ndarray
            Diversity archive decisions
        N : int
            Population size

        Returns
        -------
        parentC_decs : ndarray
            Parents for crossover: ceil(N/2) CA tournament winners followed by
            ceil(N/2) random DA members (pairing CA x DA in the crossover operator)
        parentM_decs : ndarray
            Parents for mutation: N random CA members
        """
        CA_size = CA_objs.shape[0]
        DA_size = DA_objs.shape[0]
        half = int(np.ceil(N / 2))

        # Pareto-dominance tournament between two random CA members
        CA_parent1 = np.random.randint(0, CA_size, size=half)
        CA_parent2 = np.random.randint(0, CA_size, size=half)
        dominate = (np.any(CA_objs[CA_parent1] < CA_objs[CA_parent2], axis=1).astype(int)
                    - np.any(CA_objs[CA_parent1] > CA_objs[CA_parent2], axis=1).astype(int))
        winners = np.where(dominate == 1, CA_parent1, CA_parent2)

        # First half: CA winners; second half: random DA members
        parentC_decs = np.vstack([CA_decs[winners],
                                  DA_decs[np.random.randint(0, DA_size, size=half)]])

        # Parents for mutation: random CA members
        parentM_decs = CA_decs[np.random.randint(0, CA_size, size=N)]

        return parentC_decs, parentM_decs

    def _update_CA(self, CA, new_objs, new_decs, max_size):
        """
        Update Convergence Archive (CA) by indicator-based (IBEA-style) selection.

        Parameters
        ----------
        CA : tuple or None
            Current CA (objs, decs) or None
        new_objs : ndarray
            New objectives to add
        new_decs : ndarray
            New decisions to add
        max_size : int
            Maximum size of CA

        Returns
        -------
        CA_objs : ndarray
            Updated CA objectives
        CA_decs : ndarray
            Updated CA decisions
        """
        if CA is None:
            CA_objs = new_objs
            CA_decs = new_decs
        else:
            CA_objs, CA_decs = CA
            CA_objs = np.vstack([CA_objs, new_objs])
            CA_decs = np.vstack([CA_decs, new_decs])

        N = CA_objs.shape[0]
        if N <= max_size:
            return CA_objs, CA_decs

        # IBEA fitness with kappa = 0.05 (normalized additive epsilon indicator)
        kappa = 0.05
        F, I, C = ibea_fitness(CA_objs, kappa)

        # Iteratively delete the worst solution and update the fitness
        choose = np.arange(N)
        while len(choose) > max_size:
            x = np.argmin(F[choose])
            F = F + np.exp(-I[choose[x], :] / C[choose[x]] / kappa)
            choose = np.delete(choose, x)

        return CA_objs[choose], CA_decs[choose]

    def _update_DA(self, DA, new_objs, new_decs, max_size, p):
        """
        Update Diversity Archive (DA): non-dominated solutions truncated by
        Lp-norm-based diversity maintenance.

        Parameters
        ----------
        DA : tuple or None
            Current DA (objs, decs) or None
        new_objs : ndarray
            New objectives to add
        new_decs : ndarray
            New decisions to add
        max_size : int
            Maximum size of DA
        p : float
            Parameter of the fractional (Lp) distance

        Returns
        -------
        DA_objs : ndarray
            Updated DA objectives
        DA_decs : ndarray
            Updated DA decisions
        """
        # Combine current DA and new solutions
        if DA is None:
            DA_objs = new_objs
            DA_decs = new_decs
        else:
            DA_objs, DA_decs = DA
            DA_objs = np.vstack([DA_objs, new_objs])
            DA_decs = np.vstack([DA_decs, new_decs])

        # Keep only the non-dominated solutions
        N = DA_objs.shape[0]
        front_no, _ = nd_sort(DA_objs, N)
        non_dominated_mask = front_no == 1

        DA_objs = DA_objs[non_dominated_mask]
        DA_decs = DA_decs[non_dominated_mask]

        N = DA_objs.shape[0]
        if N <= max_size:
            return DA_objs, DA_decs

        # Select the extreme solutions first (min and max of each objective)
        choose = np.zeros(N, dtype=bool)
        choose[np.argmin(DA_objs, axis=0)] = True
        choose[np.argmax(DA_objs, axis=0)] = True

        if np.sum(choose) > max_size:
            # Randomly delete several solutions
            chosen_indices = np.where(choose)[0]
            to_remove = np.random.choice(chosen_indices, size=np.sum(choose) - max_size, replace=False)
            choose[to_remove] = False
        elif np.sum(choose) < max_size:
            # Add several solutions by truncation strategy (Lp-norm distance)
            diff = np.abs(DA_objs[:, np.newaxis, :] - DA_objs[np.newaxis, :, :])
            distance = np.sum(diff ** p, axis=2) ** (1.0 / p)
            np.fill_diagonal(distance, np.inf)

            while np.sum(choose) < max_size:
                remaining = np.where(~choose)[0]
                # Select the remaining solution farthest from the chosen ones
                min_distances = np.min(distance[np.ix_(remaining, np.where(choose)[0])], axis=1)
                choose[remaining[np.argmax(min_distances)]] = True

        return DA_objs[choose], DA_decs[choose]


def operator_ga_real(parent_decs, proC, disC, proM, disM):
    """
    Real-coded genetic operators (PlatEMO OperatorGA, GAreal) in [0, 1] space.

    The parents are split into two halves; row k of the first half is paired
    with row k of the second half, and each pair generates two offspring.

    Parameters
    ----------
    parent_decs : ndarray
        Parent decision variables in [0, 1], shape (n, d)
    proC : float
        Probability of doing crossover
    disC : float
        Distribution index of simulated binary crossover
    proM : float
        Expectation of the number of mutated variables
    disM : float
        Distribution index of polynomial mutation

    Returns
    -------
    offspring : ndarray
        Offspring decision variables in [0, 1], shape (2 * floor(n/2), d)
    """
    n = parent_decs.shape[0]
    half = n // 2
    parent1 = parent_decs[:half, :]
    parent2 = parent_decs[half:2 * half, :]
    N, D = parent1.shape

    # Simulated binary crossover
    beta = np.zeros((N, D))
    mu = np.random.rand(N, D)
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
    beta = beta * (-1.0) ** np.random.randint(0, 2, size=(N, D))
    beta[np.random.rand(N, D) < 0.5] = 1
    beta[np.repeat(np.random.rand(N, 1) > proC, D, axis=1)] = 1
    offspring = np.vstack([(parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
                           (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2])

    # Polynomial mutation (lower = 0, upper = 1)
    offspring = np.clip(offspring, 0, 1)
    site = np.random.rand(2 * N, D) < proM / D
    mu = np.random.rand(2 * N, D)
    temp = site & (mu <= 0.5)
    offspring[temp] = offspring[temp] + ((2 * mu[temp] + (1 - 2 * mu[temp]) *
                      (1 - offspring[temp]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    temp = site & (mu > 0.5)
    offspring[temp] = offspring[temp] + (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) *
                      offspring[temp] ** (disM + 1)) ** (1 / (disM + 1)))

    return np.clip(offspring, 0, 1)

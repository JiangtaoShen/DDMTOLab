"""
Classification and Pareto Domination Based Multi-Objective Evolutionary Algorithm (CPS-MOEA)

This module implements CPS-MOEA for multi-objective optimization problems.

References
----------
    [1] J. Zhang, A. Zhou, and G. Zhang. "A classification and Pareto domination based multiobjective evolutionary algorithm." Proceedings of the IEEE Congress on Evolutionary Computation, 2015, 2883-2890.

Notes
-----
Author: Converted from MATLAB implementation
Date: 2025.01.22
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class CPSMOEA:
    """
    Classification and Pareto Domination Based Multi-Objective Evolutionary Algorithm.

    This algorithm uses KNN-based classification to distinguish between good and bad
    solutions, and uses this information to guide offspring generation via differential
    evolution with surrogate-assisted pre-selection.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-3',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, M=3, CR=1.0, F=0.5, proM=1.0, disM=20.0,
                 k_neighbors=5, save_data=True, save_path='./TestData', name='CPS-MOEA_test',
                 disable_tqdm=True):
        """
        Initialize CPS-MOEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        M : int, optional
            Number of candidate offsprings generated per solution (default: 3)
        CR : float, optional
            Crossover probability for differential evolution (default: 1.0)
        F : float, optional
            Scaling factor for differential evolution (default: 0.5)
        proM : float, optional
            Expected number of mutated variables (default: 1.0)
        disM : float, optional
            Distribution index for polynomial mutation (default: 20.0)
        k_neighbors : int, optional
            Number of neighbors for KNN classification (default: 5)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'CPS-MOEA_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.M = M
        self.CR = CR
        self.F = F
        self.proM = proM
        self.disM = disM
        self.k_neighbors = k_neighbors
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        # Initialize KNN model storage
        self.knn_models = []

    def optimize(self):
        """
        Execute the CPS-MOEA algorithm.

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
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize Pgood and Pbad for each task
        Pgood_decs, Pgood_objs, Pgood_cons = [], [], []
        Pbad_decs, Pbad_objs, Pbad_cons = [], [], []

        # Initialize KNN models for each task
        self.knn_models = [{'data': None, 'label': None} for _ in range(nt)]

        for i in range(nt):
            half_n = n_per_task[i] // 2
            pgood_d, pgood_o, pgood_c, pbad_d, pbad_o, pbad_c = self._nds_split(
                decs[i], objs[i], cons[i], half_n
            )
            Pgood_decs.append(pgood_d)
            Pgood_objs.append(pgood_o)
            Pgood_cons.append(pgood_c)
            Pbad_decs.append(pbad_d)
            Pbad_objs.append(pbad_o)
            Pbad_cons.append(pbad_c)

            # Train initial KNN model
            self._train_knn(i, pgood_d, pbad_d)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Generate offsprings using DE and KNN-based classification
                off_decs = self._generate_offspring(i, decs[i], n_per_task[i])
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Merge current population and offspring
                objs[i], decs[i], cons[i] = vstack_groups(
                    (objs[i], off_objs), (decs[i], off_decs), (cons[i], off_cons)
                )

                # Environmental selection: NDS-based selection to keep N individuals
                rank, front_no, crowd_dis = nsga2_sort(objs[i], cons[i])
                index = np.argsort(rank)[:n_per_task[i]]
                objs[i], decs[i], cons[i] = select_by_index(index, objs[i], decs[i], cons[i])

                # Update Pgood and Pbad sets
                front_no_off = self._compute_front_no(off_objs, off_cons)
                good_mask = front_no_off == 1

                # Merge offspring with Pgood and Pbad
                new_pgood_decs, new_pgood_objs, new_pgood_cons = vstack_groups(
                    (Pgood_decs[i], off_decs[good_mask]),
                    (Pgood_objs[i], off_objs[good_mask]),
                    (Pgood_cons[i], off_cons[good_mask] if off_cons is not None else None)
                )

                new_pbad_decs, new_pbad_objs, new_pbad_cons = vstack_groups(
                    (Pbad_decs[i], off_decs[~good_mask]),
                    (Pbad_objs[i], off_objs[~good_mask]),
                    (Pbad_cons[i], off_cons[~good_mask] if off_cons is not None else None)
                )

                # Select best half for Pgood and Pbad
                half_n = n_per_task[i] // 2
                Pgood_decs[i], Pgood_objs[i], Pgood_cons[i], _, _, _ = self._nds_split(
                    new_pgood_decs, new_pgood_objs, new_pgood_cons, half_n
                )
                Pbad_decs[i], Pbad_objs[i], Pbad_cons[i], _, _, _ = self._nds_split(
                    new_pbad_decs, new_pbad_objs, new_pbad_cons, half_n
                )

                # Update KNN model
                self._train_knn(i, Pgood_decs[i], Pbad_decs[i])

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
            all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data
        )

        return results

    def _nds_split(self, decs, objs, cons, K):
        """
        Sort population and split into good (Pgood) and bad (Pbad) sets.

        Parameters
        ----------
        decs : np.ndarray
            Decision variables
        objs : np.ndarray
            Objective values
        cons : np.ndarray or None
            Constraint values
        K : int
            Size of each set

        Returns
        -------
        Pgood_decs, Pgood_objs, Pgood_cons, Pbad_decs, Pbad_objs, Pbad_cons
        """
        rank, front_no, crowd_dis = nsga2_sort(objs, cons)
        sorted_idx = np.argsort(rank)

        # Select top K as Pgood
        pgood_idx = sorted_idx[:K]
        # Select bottom K as Pbad
        pbad_idx = sorted_idx[-K:]

        pgood_decs = decs[pgood_idx]
        pgood_objs = objs[pgood_idx]
        pgood_cons = cons[pgood_idx] if cons is not None else None

        pbad_decs = decs[pbad_idx]
        pbad_objs = objs[pbad_idx]
        pbad_cons = cons[pbad_idx] if cons is not None else None

        return pgood_decs, pgood_objs, pgood_cons, pbad_decs, pbad_objs, pbad_cons

    def _train_knn(self, task_idx, good_decs, bad_decs):
        """
        Train KNN model for classification.

        Parameters
        ----------
        task_idx : int
            Task index
        good_decs : np.ndarray
            Decision variables of good solutions
        bad_decs : np.ndarray
            Decision variables of bad solutions
        """
        data = np.vstack([good_decs, bad_decs])
        labels = np.hstack([np.ones(len(good_decs), dtype=bool),
                            np.zeros(len(bad_decs), dtype=bool)])

        self.knn_models[task_idx] = {'data': data, 'label': labels}

    def _predict_knn(self, task_idx, test_decs):
        """
        Predict labels using KNN model.

        Parameters
        ----------
        task_idx : int
            Task index
        test_decs : np.ndarray
            Test decision variables

        Returns
        -------
        np.ndarray
            Boolean array indicating predicted class (True=good, False=bad)
        """
        model = self.knn_models[task_idx]

        # Compute pairwise distances
        distances = np.sqrt(((test_decs[:, np.newaxis, :] - model['data'][np.newaxis, :, :]) ** 2).sum(axis=2))

        # Find k nearest neighbors
        k_nearest_idx = np.argsort(distances, axis=1)[:, :self.k_neighbors]

        # Get labels of k nearest neighbors
        k_nearest_labels = model['label'][k_nearest_idx]

        # Predict: majority voting (>50% of neighbors are good)
        predictions = np.sum(k_nearest_labels, axis=1) > (self.k_neighbors / 2)

        return predictions

    def _generate_offspring(self, task_idx, parent_decs, N):
        """
        Generate offspring using DE and KNN-based classification pre-selection.

        Parameters
        ----------
        task_idx : int
            Task index
        parent_decs : np.ndarray
            Parent decision variables
        N : int
            Number of offsprings to generate

        Returns
        -------
        np.ndarray
            Selected offspring decision variables
        """
        M = self.M
        pop_size = len(parent_decs)

        # Generate M candidate offsprings for each parent
        all_candidates = []

        for _ in range(M):
            # Random selection for DE
            parent1_idx = np.arange(pop_size)
            parent2_idx = np.random.randint(0, pop_size, size=pop_size)
            parent3_idx = np.random.randint(0, pop_size, size=pop_size)

            parent1 = parent_decs[parent1_idx]
            parent2 = parent_decs[parent2_idx]
            parent3 = parent_decs[parent3_idx]

            # Differential evolution operator
            candidates = self._operator_de(parent1, parent2, parent3)
            all_candidates.append(candidates)

        # Stack all candidates: shape (N, M, D)
        all_candidates = np.array(all_candidates).transpose(1, 0, 2)  # (N, M, D)

        # Flatten for KNN prediction
        flat_candidates = all_candidates.reshape(-1, all_candidates.shape[-1])

        # KNN classification
        predictions = self._predict_knn(task_idx, flat_candidates)
        predictions = predictions.reshape(N, M).astype(float)

        # Add random noise for tie-breaking
        predictions += np.random.rand(N, M) * 0.01

        # Select best candidate for each parent
        best_idx = np.argmax(predictions, axis=1)
        offspring = all_candidates[np.arange(N), best_idx, :]

        return offspring

    def _operator_de(self, parent1, parent2, parent3):
        """
        Differential evolution operator with polynomial mutation in [0,1] space.

        Parameters
        ----------
        parent1, parent2, parent3 : np.ndarray
            Parent decision variables

        Returns
        -------
        np.ndarray
            Offspring decision variables
        """
        N, D = parent1.shape

        # Differential evolution: offspring = parent1 + F * (parent2 - parent3)
        site = np.random.rand(N, D) < self.CR
        offspring = parent1.copy()
        offspring[site] = offspring[site] + self.F * (parent2[site] - parent3[site])

        # Polynomial mutation
        site = np.random.rand(N, D) < (self.proM / D)
        mu = np.random.rand(N, D)

        # Ensure offspring is within [0,1] before mutation
        offspring = np.clip(offspring, 0.0, 1.0)

        # Mutation for mu <= 0.5
        temp = site & (mu <= 0.5)
        delta = (2.0 * mu + (1.0 - 2.0 * mu) *
                 (1.0 - offspring) ** (self.disM + 1)) ** (1.0 / (self.disM + 1)) - 1.0
        offspring[temp] = offspring[temp] + delta[temp]

        # Mutation for mu > 0.5
        temp = site & (mu > 0.5)
        delta = 1.0 - (2.0 * (1.0 - mu) + 2.0 * (mu - 0.5) *
                       (1.0 - (1.0 - offspring)) ** (self.disM + 1)) ** (1.0 / (self.disM + 1))
        offspring[temp] = offspring[temp] + delta[temp]

        # Final bound check to ensure [0,1]
        offspring = np.clip(offspring, 0.0, 1.0)

        return offspring

    def _compute_front_no(self, objs, cons=None):
        """
        Compute front numbers for solutions.

        Parameters
        ----------
        objs : np.ndarray
            Objective values
        cons : np.ndarray, optional
            Constraint values

        Returns
        -------
        np.ndarray
            Front number for each solution
        """
        pop_size = objs.shape[0]
        if cons is not None:
            front_no, _ = nd_sort(objs, cons, pop_size)
        else:
            front_no, _ = nd_sort(objs, pop_size)
        return front_no

def nsga2_sort(objs, cons=None):
    """
    Sort solutions based on NSGA-II criteria using non-dominated sorting and crowding distance.

    Parameters
    ----------
    objs : np.ndarray
        Objective value matrix of shape (pop_size, n_obj)
    cons : np.ndarray, optional
        Constraint matrix of shape (pop_size, n_con). If None, no constraints are considered (default: None)

    Returns
    -------
    rank : np.ndarray
        Ranking of each solution (0-based index after sorting) of shape (pop_size,).
        rank[i] indicates the position of solution i in the sorted order
    front_no : np.ndarray
        Non-dominated front number of each solution of shape (pop_size,)
    crowd_dis : np.ndarray
        Crowding distance of each solution of shape (pop_size,)

    Notes
    -----
    Solutions are sorted first by front number (ascending), then by crowding distance (descending).
    Larger crowding distance values indicate better diversity preservation.
    """
    pop_size = objs.shape[0]

    # Perform non-dominated sorting
    if cons is not None:
        front_no, _ = nd_sort(objs, cons, pop_size)
    else:
        front_no, _ = nd_sort(objs, pop_size)

    # Calculate crowding distance for diversity preservation
    crowd_dis = crowding_distance(objs, front_no)

    # Sort by front number (ascending), then by crowding distance (descending)
    sorted_indices = np.lexsort((-crowd_dis, front_no))

    # Create rank array: rank[i] gives the sorted position of solution i
    rank = np.empty(pop_size, dtype=int)
    rank[sorted_indices] = np.arange(pop_size)

    return rank, front_no, crowd_dis

# from Problems.STMO.DTLZ import DTLZ, SETTINGS
# problem = DTLZ().DTLZ1()
# results = CPSMOEA(problem).optimize()
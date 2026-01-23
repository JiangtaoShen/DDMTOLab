"""
Multiple Classifiers-assisted Evolutionary Algorithm based on Decomposition (MCEAD)

This module implements MCEAD for multi-objective optimization problems.

References
----------
    [1] T. Sonoda and M. Nakata. Multiple classifiers-assisted evolutionary algorithm based on decomposition for high-dimensional multi-objective problems. IEEE Transactions on Evolutionary Computation, 2022.

Notes
-----
Author: Haowei Guo
Email: ghw@mail.nwpu.edu.cn
Date: 2026.01.06
Version: 1.0
"""

import numpy as np
from tqdm import tqdm
import time
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MCEAD:
    """
    Multiple Classifiers-assisted Evolutionary Algorithm based on Decomposition for multi-objective optimization

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '2-M',
        'cons': 'unequal',
        'n_cons': '0-C',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem,  n=None, max_nfes=None, delta=0.9, nr=2, r_max=10,
                 save_data=True, save_path=None, name='MCEAD_test', disable_tqdm=True):
        """
        Initialize MCEAD algorithm.

        Parameters
        ----------
        problem : MTOP
            Problem instance.
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        delta : float, optional
            Probability of choosing parents from neighborhood (default: 0.9).
        nr : int, optional
            Maximum number of solutions replaced by each offspring (default: 2).
        r_max : int, optional
            Maximum repeat time of offspring generation (default: 10).
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path for saving results (default: './TestData')
        name : str, optional
            Name of the experiment/file (default: 'MCEAD_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 500
        self.delta = delta
        self.nr = nr
        self.r_max = r_max
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MCEAD algorithm.

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

        W_list = []
        B_list = []
        Z_list = []
        svm_groups = []

        # Stage 1: Pre-calculate weights and update population size
        for t in range(nt):
            n_objs_t = problem.n_objs[t]
            W, N = uniform_point(n_per_task[t], n_objs_t)
            n_per_task[t] = N
            W_list.append(W)

            # Compute neighboring vectors
            T = int(np.ceil(N / 10))
            dists = cdist(W, W, metric='euclidean')
            B = np.argsort(dists, axis=1)[:, :T]
            B_list.append(B)

        # Stage 2: Unified initialization
        decs_list = initialization(problem, n=n_per_task, method='lhs')
        objs_list, cons_list = evaluation(problem, decs_list)

        all_decs, all_objs, all_cons = init_history(decs_list, objs_list, cons_list)

        archive_decs_list = []
        archive_objs_list = []

        # Stage 3: Build environments task-wise (SVM, Z, Archive)
        for t in range(nt):
            N = n_per_task[t]
            objs = objs_list[t]
            decs = decs_list[t]

            # 2.4 Ideal point Z
            Z = np.min(objs, axis=0)
            Z_list.append(Z)

            # 2.5 Archive (Copy initial population)
            archive_decs_list.append(decs.copy())
            archive_objs_list.append(objs.copy())

            # 2.6 Init SVMs (one for each subproblem in task t)
            svm_list = [SVM(problem, task_id=t) for _ in range(N)]
            svm_groups.append(svm_list)

        # Progress Bar
        nfes_per_task = n_per_task.copy()
        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Stage 4: Optimization Loop
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [t for t in range(nt) if nfes_per_task[t] < max_nfes_per_task[t]]
            if not active_tasks:
                break

            for t in active_tasks:
                N = n_per_task[t]
                W = W_list[t]
                B = B_list[t]
                Z = Z_list[t]
                current_svm_list = svm_groups[t]

                # MCEAD logic: Iterate over all subproblems; switch task after one full cycle (N subproblems).
                for i in range(N):
                    if nfes_per_task[t] >= max_nfes_per_task[t]:
                        break

                    # 4.1: Model Construction
                    current_svm_list[i].model_construction(
                            archive_decs_list[t],
                            archive_objs_list[t],
                            B[i], W, Z
                    )

                    # 4.2: Choose Parents
                    if np.random.rand() < self.delta:
                        P = B[i].copy()
                    else:
                        P = np.arange(N)
                    np.random.shuffle(P)

                    # 4.3: Solution Generation
                    off_dec = solution_generation(
                        problem=problem,
                        population_decs=decs_list[t],
                        parents_indices=P,
                        svm_model=current_svm_list[i],
                        r_max=self.r_max,
                        current_idx=i,
                        task_id=t
                    )

                    # 4.4: Evaluate Offspring
                    off_obj, off_con = evaluation_single(problem, off_dec.reshape(1, -1), t)

                    nfes_per_task[t] += 1
                    pbar.update(1)

                    # Update Archive (Task t)
                    archive_decs_list[t] = np.vstack((archive_decs_list[t], off_dec))
                    archive_objs_list[t] = np.vstack((archive_objs_list[t], off_obj))

                    # 4.5: Update Reference Point
                    Z = np.min(np.vstack((Z, off_obj)), axis=0)
                    Z_list[t] = Z

                    # 4.6: Update Population
                    g_old = np.max(np.abs(objs_list[t][P] - Z) * W[P], axis=1)
                    g_new = np.max(np.abs(off_obj - Z) * W[P], axis=1)

                    replace_mask = g_old >= g_new
                    replace_indices = P[replace_mask]

                    if len(replace_indices) > self.nr:
                        replace_indices = replace_indices[:self.nr]

                    if len(replace_indices) > 0:
                        decs_list[t][replace_indices] = off_dec
                        objs_list[t][replace_indices] = off_obj
                        if cons_list[t] is not None:
                            cons_list[t][replace_indices] = off_con

                # Log historical data for the current task's generation
                append_history(all_decs[t], decs_list[t], all_objs[t], objs_list[t], all_cons[t], cons_list[t])

        pbar.close()
        runtime = time.time() - start_time

        # Save Results
        results = build_save_results(
            all_decs=all_decs,
            all_objs=all_objs,
            all_cons=all_cons,
            runtime=runtime,
            max_nfes=nfes_per_task,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        return results


class SVM:
    """
    Support Vector Machine (SVM) wrapper for classification-assisted optimization.

    This class manages a local SVM model for a specific subproblem in the MCEAD algorithm.
    It constructs a decision boundary to distinguish between promising and unpromising
    solutions based on the current population's performance in the neighborhood.

    """
    def __init__(self, problem, task_id):
        self.problem = problem
        self.task_id = task_id
        self.model = None
        self.C = 1.0
        self.gamma_matlab = 1.0
        self.gamma_sklearn = self.gamma_matlab

        # Access bounds from MTOP bounds list [(lb, ub), ...]
        self.lower = problem.bounds[task_id][0].flatten()
        self.upper = problem.bounds[task_id][1].flatten()
        self.n_vars = len(self.lower)

    def uniform_input(self, x):
        denom = self.upper - self.lower
        denom[denom == 0] = 1e-10
        return (x - self.lower) / denom

    def model_construction(self, archive_decs, archive_objs, neighbor_indices, W, Z):
        """
        Construct and train the SVM model based on neighboring solutions.

        The model classifies solutions as 'good' (label 1) or 'bad' (label -1)
        based on their Tchebycheff aggregation values.
        """
        n_samples = len(archive_decs)
        labels = -1 * np.ones(n_samples)
        c_i = set()

        for neighbor_idx in neighbor_indices:
            w_vec = W[neighbor_idx]
            g_data = np.max(np.abs(archive_objs - Z) * w_vec, axis=1)
            sorted_indices = np.argsort(g_data)

            for best_idx in sorted_indices:
                if best_idx not in c_i:
                    c_i.add(best_idx)
                    labels[best_idx] = 1
                    break

        X_train = np.array([self.uniform_input(x) for x in archive_decs])
        y_train = labels

        if len(np.unique(y_train)) > 1:
            self.model = SVC(C=self.C, kernel='rbf', gamma=self.gamma_sklearn, probability=False)
            self.model.fit(X_train, y_train)
        else:
            self.model = None

    def predict_class(self, x):
        """
        Predict the class and decision score for a candidate solution.

        Returns
        -------
        pred_class : int
            Predicted class label (1 for good, -1/0 for bad).
        score : float
            Decision function score (distance to hyperplane).
        """
        if self.model is None:
            return 1, 0.0
        x_uni = self.uniform_input(x).reshape(1, -1)
        pred_class = self.model.predict(x_uni)[0]
        score = self.model.decision_function(x_uni)[0]
        return pred_class, score


def operator_de(problem, parent1, parent2, parent3, task_id, params=None):
    """
    Perform Differential Evolution (DE) crossover and Polynomial Mutation.

    Returns
    -------
    offspring : np.ndarray
        The generated offspring decision vector.
    """
    if params is None:
        # CR=1, F=0.5, proM=1, disM=20
        CR, F, proM, disM = 1.0, 0.5, 1.0, 20.0
    else:
        CR, F, proM, disM = params

    lower = problem.bounds[task_id][0].flatten()
    upper = problem.bounds[task_id][1].flatten()
    D = len(lower)

    offspring = parent1.copy()
    site_de = np.random.rand(D) < CR
    diff = F * (parent2 - parent3)
    offspring[site_de] = offspring[site_de] + diff[site_de]

    # Polynomial Mutation
    site_pm = np.random.rand(D) < (proM / D)
    mu = np.random.rand(D)

    offspring = np.clip(offspring, lower, upper)

    temp = site_pm & (mu <= 0.5)
    l_temp, u_temp = lower[temp], upper[temp]
    o_temp, m_temp = offspring[temp], mu[temp]

    if np.any(temp):
        val = 2.0 * m_temp + (1.0 - 2.0 * m_temp) * \
              (1.0 - (o_temp - l_temp) / (u_temp - l_temp)) ** (disM + 1)
        delta = val ** (1.0 / (disM + 1)) - 1.0
        offspring[temp] = o_temp + (u_temp - l_temp) * delta

    temp = site_pm & (mu > 0.5)
    l_temp, u_temp = lower[temp], upper[temp]
    o_temp, m_temp = offspring[temp], mu[temp]

    if np.any(temp):
        val = 2.0 * (1.0 - m_temp) + 2.0 * (m_temp - 0.5) * \
              (1.0 - (u_temp - o_temp) / (u_temp - l_temp)) ** (disM + 1)
        delta = 1.0 - val ** (1.0 / (disM + 1))
        offspring[temp] = o_temp + (u_temp - l_temp) * delta

    offspring = np.clip(offspring, lower, upper)
    return offspring


def solution_generation(problem, population_decs, parents_indices, svm_model, r_max, current_idx, task_id):
    """
    Generate a new solution using DE and SVM assistance.

    Attempts to generate a solution that is predicted as 'good' (class 1) by the SVM.
    If no such solution is found within `r_max` tries, returns the one with the highest decision score.

    Returns
    -------
    y_best : np.ndarray
        The best generated offspring solution found.
    """
    x_i = population_decs[current_idx]
    d_i_max = -np.inf
    y_best = None
    P = parents_indices.copy()

    for r in range(r_max):
        p1_idx = P[0]
        p2_idx = P[1]
        x_p1 = population_decs[p1_idx]
        x_p2 = population_decs[p2_idx]

        candidate = operator_de(problem, x_i, x_p1, x_p2, task_id)
        np.random.shuffle(P)
        c_pred, score = svm_model.predict_class(candidate)

        if c_pred == 1:
            return candidate
        else:
            if r == 0:
                d_i_max = score
                y_best = candidate
            elif score > d_i_max:
                d_i_max = score
                y_best = candidate

    return y_best
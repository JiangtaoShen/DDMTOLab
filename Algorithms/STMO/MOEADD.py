"""
Multi-objective Evolutionary Algorithm Based on Decomposition and Dominance (MOEA/DD)

This module implements MOEA/DD for multi-objective optimization problems.

References
----------
    [1] Li, Ke, et al. "An evolutionary many-objective optimization algorithm based on dominance and decomposition."
        IEEE transactions on evolutionary computation 19.5 (2014): 694-716.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.12.18
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from Methods.Algo_Methods.uniform_point import uniform_point
from Methods.Algo_Methods.algo_utils import *


class MOEADD:
    """
    Multi-objective Evolutionary Algorithm Based on Decomposition and Dominance.

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
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        """
        Get algorithm information.

        Parameters
        ----------
        print_info : bool, optional
            Whether to print information (default: True)

        Returns
        -------
        dict
            Algorithm information dictionary
        """
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, delta=0.9, muc=20.0, mum=15.0, save_data=True,
                 save_path='./TestData', name='MOEADD_test', disable_tqdm=True):
        """
        Initialize MOEA/DD algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        delta : float, optional
            Probability of choosing parents locally (default: 0.9)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'MOEADD_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.delta = delta
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MOEA/DD algorithm.

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

        # Generate uniformly distributed weight vectors for each task
        W = []
        T = []  # Neighborhood size
        B = []  # Neighbor indices
        for i in range(nt):
            w_i, n = uniform_point(n_per_task[i], no[i])
            W.append(w_i)
            n_per_task[i] = n

            # Set neighborhood size to 10% of population
            T.append(int(np.ceil(n / 10)))

            # Detect the neighbors of each weight vector based on Euclidean distance
            distances = squareform(pdist(w_i))
            neighbors = np.argsort(distances, axis=1)[:, :T[i]]
            B.append(neighbors)

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize Region for each task
        Region = []
        for i in range(nt):
            # Calculate cosine similarity: 1 - cosine distance
            norm_objs = objs[i] / (np.linalg.norm(objs[i], axis=1, keepdims=True) + 1e-10)
            norm_w = W[i] / (np.linalg.norm(W[i], axis=1, keepdims=True) + 1e-10)
            cosine_similarity = np.dot(norm_objs, norm_w.T)
            region_i = np.argmax(cosine_similarity, axis=1)
            Region.append(region_i)

        # Initialize FrontNo for each task
        FrontNo = []
        for i in range(nt):
            front_no_i, _ = nd_sort(objs[i], cons[i], n_per_task[i])
            FrontNo.append(front_no_i)

        # Initialize ideal point Z for each task
        Z = []
        for i in range(nt):
            Z.append(np.min(objs[i], axis=0))

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        # Main loop
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for task_id in active_tasks:
                # For each solution
                for i in range(n_per_task[task_id]):
                    # Choose the parents
                    Ei = np.where(np.isin(Region[task_id], B[task_id][i]))[0]

                    if np.random.rand() < self.delta and len(Ei) >= 2:
                        # Local selection
                        CV_Ei = np.sum(np.maximum(0, cons[task_id][Ei]), axis=1) if cons[
                                                                                        task_id] is not None else np.zeros(
                            len(Ei))
                        P = tournament_selection(2, 2, CV_Ei)
                        P = Ei[P]
                    else:
                        # Global selection
                        CV = np.sum(np.maximum(0, cons[task_id]), axis=1) if cons[task_id] is not None else np.zeros(
                            n_per_task[task_id])
                        P = tournament_selection(2, 2, CV)

                    # Generate an offspring
                    parent_decs = decs[task_id][P, :]
                    off_decs = ga_generation(parent_decs, muc=self.muc, mum=self.mum)
                    off_objs, off_cons = evaluation_single(problem, off_decs[:1], task_id)

                    # Assign offspring to region
                    norm_off_obj = off_objs[0] / (np.linalg.norm(off_objs[0]) + 1e-10)
                    norm_w = W[task_id] / (np.linalg.norm(W[task_id], axis=1, keepdims=True) + 1e-10)
                    cosine_similarity = np.dot(norm_off_obj, norm_w.T)
                    offRegion = np.argmax(cosine_similarity)

                    # Add the offspring to the population
                    decs[task_id] = np.vstack([decs[task_id], off_decs[0]])
                    objs[task_id] = np.vstack([objs[task_id], off_objs[0]])
                    if cons[task_id] is not None and off_cons is not None:
                        cons[task_id] = np.vstack([cons[task_id], off_cons[0]])
                    Region[task_id] = np.append(Region[task_id], offRegion)

                    # Update FrontNo (add mode)
                    FrontNo[task_id] = UpdateFront(objs[task_id], FrontNo[task_id])

                    # Calculate constraint violations
                    CV = np.sum(np.maximum(0, cons[task_id]), axis=1) if cons[task_id] is not None else np.zeros(
                        len(objs[task_id]))

                    # Update the ideal point
                    Z[task_id] = np.minimum(Z[task_id], off_objs[0])

                    # Delete a solution from the population
                    if np.any(CV > 0):
                        S = np.argsort(CV)[::-1]
                        S = S[:np.sum(CV > 0)]
                        flag = False
                        x = None

                        for j in range(len(S)):
                            if np.sum(Region[task_id] == Region[task_id][S[j]]) > 1:
                                flag = True
                                x = S[j]
                                break

                        if not flag:
                            x = S[0]

                    elif np.max(FrontNo[task_id]) == 1:
                        x = LocateWorst(objs[task_id], W[task_id], Region[task_id], FrontNo[task_id], Z[task_id])

                    else:
                        Fl = np.where(FrontNo[task_id] == np.max(FrontNo[task_id]))[0]

                        if len(Fl) == 1:
                            if np.sum(Region[task_id] == Region[task_id][Fl[0]]) > 1:
                                x = Fl[0]
                            else:
                                x = LocateWorst(objs[task_id], W[task_id], Region[task_id], FrontNo[task_id],
                                                Z[task_id])
                        else:
                            SubRegion = np.unique(Region[task_id][Fl])
                            Crowd = np.bincount(Region[task_id][np.isin(Region[task_id], SubRegion)],
                                                minlength=W[task_id].shape[0])
                            Phi = np.where(Crowd == np.max(Crowd))[0]

                            PBI = CalPBI(objs[task_id], W[task_id], Region[task_id], Z[task_id],
                                         np.isin(Region[task_id], Phi))
                            PBISum = np.zeros(W[task_id].shape[0])

                            for j in range(len(PBI)):
                                PBISum[Region[task_id][j]] += PBI[j]

                            Phi = np.argmax(PBISum)
                            Phih = np.where(Region[task_id] == Phi)[0]

                            if len(Phih) > 1:
                                x = Phih[np.argmax(PBI[Phih])]
                            else:
                                x = LocateWorst(objs[task_id], W[task_id], Region[task_id], FrontNo[task_id],
                                                Z[task_id])

                    # Update FrontNo before removing (delete mode)
                    FrontNo[task_id] = UpdateFront(objs[task_id], FrontNo[task_id], x)

                    # Remove the worst solution
                    decs[task_id] = np.delete(decs[task_id], x, axis=0)
                    objs[task_id] = np.delete(objs[task_id], x, axis=0)
                    if cons[task_id] is not None:
                        cons[task_id] = np.delete(cons[task_id], x, axis=0)
                    Region[task_id] = np.delete(Region[task_id], x)

                    # Update evaluation count
                    nfes_per_task[task_id] += 1
                    pbar.update(1)

                    # Check if evaluation budget is exhausted
                    if nfes_per_task[task_id] >= max_nfes_per_task[task_id]:
                        break

                # Update history
                append_history(all_decs[task_id], decs[task_id], all_objs[task_id],
                               objs[task_id], all_cons[task_id], cons[task_id])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, all_cons=all_cons,
                                     bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results


def UpdateFront(PopObj, FrontNo, x=None):
    """
    Update the front number of each solution when a solution is added or deleted.

    Parameters
    ----------
    PopObj : np.ndarray
        Objective values of shape (N, M)
    FrontNo : np.ndarray
        Front numbers
    x : int, optional
        Index of solution to delete. If None, assumes a new solution is added at the end (default: None)

    Returns
    -------
    FrontNo : np.ndarray
        Updated front numbers
    """
    N, M = PopObj.shape

    if x is None:
        # Add a new solution (has been stored in the last of PopObj)
        FrontNo = np.append(FrontNo, 0)
        Move = np.zeros(N, dtype=bool)
        Move[N - 1] = True
        CurrentF = 1

        # Locate the front No. of the new solution
        while True:
            Dominated = False
            for i in range(N - 1):
                if FrontNo[i] == CurrentF:
                    m = 0
                    while m < M and PopObj[i, m] <= PopObj[N - 1, m]:
                        m += 1
                    Dominated = (m == M)
                    if Dominated:
                        break

            if not Dominated:
                break
            else:
                CurrentF += 1

        # Move down the dominated solutions front by front
        while np.any(Move):
            NextMove = np.zeros(N, dtype=bool)
            for i in range(N):
                if FrontNo[i] == CurrentF:
                    Dominated = False
                    for j in range(N):
                        if Move[j]:
                            m = 0
                            while m < M and PopObj[j, m] <= PopObj[i, m]:
                                m += 1
                            Dominated = (m == M)
                            if Dominated:
                                break
                    NextMove[i] = Dominated

            FrontNo[Move] = CurrentF
            CurrentF += 1
            Move = NextMove

    else:
        # Delete the x-th solution
        x = int(x)
        Move = np.zeros(N, dtype=bool)
        Move[x] = True
        CurrentF = int(FrontNo[x]) + 1

        while np.any(Move):
            NextMove = np.zeros(N, dtype=bool)
            for i in range(N):
                if FrontNo[i] == CurrentF:
                    Dominated = False
                    for j in range(N):
                        if Move[j]:
                            m = 0
                            while m < M and PopObj[j, m] <= PopObj[i, m]:
                                m += 1
                            Dominated = (m == M)
                            if Dominated:
                                break
                    NextMove[i] = Dominated

            for i in range(N):
                if NextMove[i]:
                    Dominated = False
                    for j in range(N):
                        if FrontNo[j] == CurrentF - 1 and not Move[j]:
                            m = 0
                            while m < M and PopObj[j, m] <= PopObj[i, m]:
                                m += 1
                            Dominated = (m == M)
                            if Dominated:
                                break
                    NextMove[i] = not Dominated

            FrontNo[Move] = CurrentF - 2
            CurrentF += 1
            Move = NextMove

        FrontNo = np.delete(FrontNo, x)

    return FrontNo


def LocateWorst(PopObj, W, Region, FrontNo, Z):
    """
    Detect the worst solution in the population.
    """
    Crowd = np.bincount(Region, minlength=W.shape[0])
    Phi = np.where(Crowd == np.max(Crowd))[0]
    PBI = CalPBI(PopObj, W, Region, Z, np.isin(Region, Phi))
    PBISum = np.zeros(W.shape[0])
    for j in range(len(PBI)):
        PBISum[Region[j]] += PBI[j]
    Phi = np.argmax(PBISum)
    Phih = np.where(Region == Phi)[0]
    R = Phih[FrontNo[Phih] == np.max(FrontNo[Phih])]
    x = R[np.argmax(PBI[R])]
    return int(x)


def CalPBI(PopObj, W, Region, Z, Sub):
    """
    Calculate the PBI value between each solution and its associated weight vector.
    """
    M = W.shape[1]
    Z_rep = np.tile(Z, (np.sum(Sub), 1))
    W_sub = W[Region[Sub], :]
    NormW = np.sqrt(np.sum(W_sub ** 2, axis=1))
    d1 = np.abs(np.sum((PopObj[Sub, :] - Z_rep) * W_sub, axis=1)) / NormW
    d1_NormW = d1 / NormW
    projection = Z_rep + W_sub * d1_NormW[:, np.newaxis]
    d2 = np.sqrt(np.sum((PopObj[Sub, :] - projection) ** 2, axis=1))
    PBI = np.zeros(PopObj.shape[0])
    PBI[Sub] = d1 + 5 * d2
    return PBI

#
# # Test code
# if __name__ == "__main__":
#     from Problems.STMO.DTLZ import DTLZ
#
#     problem = DTLZ().DTLZ1()
#     result = MOEADD(problem, max_nfes=10000, disable_tqdm=False).optimize()
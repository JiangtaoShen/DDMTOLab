"""
Multi-objective Evolutionary Algorithm Based on Dominance and Decomposition (MOEA/DD)

References
----------
    K. Li, K. Deb, Q. Zhang, and S. Kwong. An evolutionary many-objective
    optimization algorithm based on dominance and decomposition. IEEE
    Transactions on Evolutionary Computation, 2015, 19(5): 694-716.
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from Methods.Algo_Methods.uniform_point import uniform_point
from Methods.Algo_Methods.algo_utils import *


class MOEADD:
    """
    Many-objective evolutionary algorithm based on dominance and decomposition (MOEA/DD).

    MOEA/DD combines decomposition-based and dominance-based approaches to handle
    many-objective optimization problems effectively. It uses weight vectors for
    decomposition and Pareto dominance for environmental selection.

    Parameters
    ----------
    problem : MTOP
        Multi-task optimization problem instance
    n : int or list, optional
        Population size (default: 100)
    max_nfes : int or list, optional
        Maximum number of function evaluations (default: 10000)
    delta : float, optional
        Probability of choosing parents locally (default: 0.9)
    muc : float, optional
        Distribution index for crossover (default: 20.0)
    mum : float, optional
        Distribution index for mutation (default: 15.0)
    save_data : bool, optional
        Whether to save optimization results (default: True)
    save_path : str, optional
        Path to save results (default: './TestData')
    name : str, optional
        Name for the test run (default: 'MOEADD_test')
    disable_tqdm : bool, optional
        Whether to disable progress bar (default: True)
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
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, delta=0.9,
                 muc=20.0, mum=15.0, save_data=True, save_path='./TestData',
                 name='MOEADD_test', disable_tqdm=True):

        self.problem = problem
        self.nt = problem.n_tasks

        # Population size per task
        self.n = par_list(n if n is not None else 100, self.nt)

        # Maximum function evaluations per task
        self.max_nfes = par_list(max_nfes if max_nfes is not None else 10000, self.nt)

        # Algorithm parameters
        self.delta = delta  # Probability of local parent selection
        self.muc = muc      # Distribution index for crossover
        self.mum = mum      # Distribution index for mutation

        # Saving settings
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MOEA/DD optimization process.

        Returns
        -------
        Results
            Optimization results containing final population and performance history
        """
        # Initialize storage
        decs = [None] * self.nt
        objs = [None] * self.nt
        cons = [None] * self.nt

        W = [None] * self.nt
        B = [None] * self.nt
        Z = [None] * self.nt
        Region = [None] * self.nt
        FrontNo = [None] * self.nt

        nfes = [0] * self.nt
        history = init_history(self.nt)

        # Dimensions and objectives per task
        dd = self.problem.dims
        no = self.problem.n_objs
        nc = self.problem.n_cons

        # Initialize each task
        for i in range(self.nt):
            # Generate weight vectors
            W[i], self.n[i] = uniform_point(self.n[i], no[i])

            # Neighborhood size
            T = int(np.ceil(self.n[i] / 10))

            # Detect neighbors based on Euclidean distance
            distances = squareform(pdist(W[i]))
            B[i] = np.argsort(distances, axis=1)[:, :T]

            # Initialize population
            decs[i] = initialization(self.problem, self.n[i], method='random')[i]
            objs[i], cons[i] = evaluation_single(self.problem, decs[i], i)
            nfes[i] = self.n[i]

            # Associate each solution with nearest weight vector
            cosine = 1 - np.sum(objs[i] * W[i][:, None, :], axis=2).T / (
                np.linalg.norm(objs[i], axis=1, keepdims=True) *
                np.linalg.norm(W[i], axis=1, keepdims=True).T
            )
            Region[i] = np.argmax(1 - cosine, axis=1)

            # Non-dominated sorting
            FrontNo[i] = nd_sort(objs[i], cons[i], self.n[i])

            # Initialize ideal point
            Z[i] = np.min(objs[i], axis=0)

            # Record initial history
            append_history(history, i, objs[i], cons[i])

        # Main optimization loop
        for i in tqdm(range(self.nt), desc='Task', disable=self.disable_tqdm):
            # Evolution loop for task i
            pbar = tqdm(total=self.max_nfes[i], desc=f'Task {i+1}',
                       disable=self.disable_tqdm, leave=False)
            pbar.update(nfes[i])

            while nfes[i] < self.max_nfes[i]:
                # For each solution in the population
                for j in range(self.n[i]):
                    # Parent selection
                    if np.random.rand() < self.delta:
                        # Local selection from neighborhood
                        Ei = np.where(np.isin(Region[i], B[i][j, :]))[0]
                        if len(Ei) >= 2:
                            # Tournament selection based on constraint violation
                            CV = np.sum(np.maximum(0, cons[i][Ei]), axis=1)
                            P = Ei[tournament_selection(2, 2, CV)]
                        else:
                            # Global selection if not enough neighbors
                            CV = np.sum(np.maximum(0, cons[i]), axis=1)
                            P = tournament_selection(2, 2, CV)
                    else:
                        # Global selection
                        CV = np.sum(np.maximum(0, cons[i]), axis=1)
                        P = tournament_selection(2, 2, CV)

                    # Generate offspring
                    off_dec = ga_generation(decs[i][P], self.muc, self.mum)[0:1]
                    off_obj, off_con = evaluation_single(self.problem, off_dec, i)
                    nfes[i] += 1

                    # Add offspring to population
                    decs[i] = np.vstack([decs[i], off_dec])
                    objs[i] = np.vstack([objs[i], off_obj])
                    cons[i] = np.vstack([cons[i], off_con])

                    # Associate offspring with nearest weight vector
                    cosine = 1 - np.sum(off_obj * W[i], axis=1) / (
                        np.linalg.norm(off_obj, axis=1, keepdims=True) *
                        np.linalg.norm(W[i], axis=1, keepdims=True).T
                    )
                    off_region = np.argmax(1 - cosine, axis=1)[0]
                    Region[i] = np.append(Region[i], off_region)

                    # Update front numbers
                    FrontNo[i] = self._update_front_add(objs[i], FrontNo[i])

                    # Update ideal point
                    Z[i] = np.minimum(Z[i], off_obj[0])

                    # Environmental selection - delete one solution
                    CV = np.sum(np.maximum(0, cons[i]), axis=1)

                    if np.any(CV > 0):
                        # Delete infeasible solution
                        x = self._delete_infeasible(Region[i], CV)
                    elif np.max(FrontNo[i]) == 1:
                        # All solutions in first front - use decomposition
                        x = self._locate_worst(objs[i], W[i], Region[i], FrontNo[i], Z[i])
                    else:
                        # Mixed fronts - delete from worst front
                        Fl = np.where(FrontNo[i] == np.max(FrontNo[i]))[0]
                        if len(Fl) == 1:
                            # Only one solution in worst front
                            if np.sum(Region[i] == Region[i][Fl[0]]) > 1:
                                x = Fl[0]
                            else:
                                x = self._locate_worst(objs[i], W[i], Region[i], FrontNo[i], Z[i])
                        else:
                            # Multiple solutions in worst front
                            x = self._delete_from_worst_front(objs[i], W[i], Region[i],
                                                             FrontNo[i], Z[i], Fl)

                    # Remove the selected solution
                    decs[i] = np.delete(decs[i], x, axis=0)
                    objs[i] = np.delete(objs[i], x, axis=0)
                    cons[i] = np.delete(cons[i], x, axis=0)
                    Region[i] = np.delete(Region[i], x)
                    FrontNo[i] = self._update_front_delete(objs[i], FrontNo[i], x)

                    # Update progress
                    pbar.update(1)

                    # Record history periodically
                    if nfes[i] % self.n[i] == 0:
                        append_history(history, i, objs[i], cons[i])

                    if nfes[i] >= self.max_nfes[i]:
                        break

            pbar.close()

        # Build and return results
        results = build_save_results(decs, objs, cons, history, self.problem,
                                     self.save_data, self.save_path, self.name)
        return results

    def _calculate_pbi(self, objs, weights, region, Z, sub_mask, theta=5.0):
        """
        Calculate PBI (Penalty-based Boundary Intersection) values.

        Parameters
        ----------
        objs : ndarray
            Objective values
        weights : ndarray
            Weight vectors
        region : ndarray
            Region assignment for each solution
        Z : ndarray
            Ideal point
        sub_mask : ndarray
            Boolean mask for solutions to calculate
        theta : float
            Penalty parameter (default: 5.0)

        Returns
        -------
        ndarray
            PBI values for all solutions (0 for non-masked solutions)
        """
        n = len(objs)
        pbi = np.zeros(n)

        if not np.any(sub_mask):
            return pbi

        # Get solutions and their weights
        sub_objs = objs[sub_mask]
        sub_regions = region[sub_mask]
        sub_weights = weights[sub_regions]

        # Translate by ideal point
        translated = sub_objs - Z

        # Calculate parallel and perpendicular distances
        norm_w = np.linalg.norm(sub_weights, axis=1, keepdims=True)
        d1 = np.abs(np.sum(translated * sub_weights, axis=1)) / norm_w.flatten()
        d2 = np.linalg.norm(translated - sub_weights * (d1 / norm_w.flatten())[:, None], axis=1)

        # PBI = d1 + theta * d2
        pbi[sub_mask] = d1 + theta * d2

        return pbi

    def _locate_worst(self, objs, weights, region, front_no, Z):
        """
        Detect the worst solution in the population using PBI-based crowding.

        Parameters
        ----------
        objs : ndarray
            Objective values
        weights : ndarray
            Weight vectors
        region : ndarray
            Region assignment for each solution
        front_no : ndarray
            Front number for each solution
        Z : ndarray
            Ideal point

        Returns
        -------
        int
            Index of the worst solution
        """
        # Calculate crowding degree for each region
        crowd = np.bincount(region, minlength=len(weights))
        phi = np.where(crowd == np.max(crowd))[0]

        # Calculate PBI for solutions in crowded regions
        sub_mask = np.isin(region, phi)
        pbi = self._calculate_pbi(objs, weights, region, Z, sub_mask)

        # Sum PBI values for each region
        pbi_sum = np.zeros(len(weights))
        for j in range(len(pbi)):
            if pbi[j] > 0:
                pbi_sum[region[j]] += pbi[j]

        # Find most crowded region
        phi_idx = np.argmax(pbi_sum)
        phi_h = np.where(region == phi_idx)[0]

        if len(phi_h) > 1:
            # Select solution with maximum PBI in that region
            r_indices = phi_h[front_no[phi_h] == np.max(front_no[phi_h])]
            x = r_indices[np.argmax(pbi[r_indices])]
        else:
            x = phi_h[0]

        return x

    def _delete_infeasible(self, region, CV):
        """
        Delete an infeasible solution prioritizing most violated and crowded regions.

        Parameters
        ----------
        region : ndarray
            Region assignment for each solution
        CV : ndarray
            Constraint violation values

        Returns
        -------
        int
            Index of solution to delete
        """
        # Sort by constraint violation (descending)
        S = np.argsort(-CV)
        S = S[:np.sum(CV > 0)]

        # Try to find a solution in a crowded region
        for j in S:
            if np.sum(region == region[j]) > 1:
                return j

        # Otherwise return most violated
        return S[0]

    def _delete_from_worst_front(self, objs, weights, region, front_no, Z, Fl):
        """
        Delete a solution from the worst front using PBI-based crowding.

        Parameters
        ----------
        objs : ndarray
            Objective values
        weights : ndarray
            Weight vectors
        region : ndarray
            Region assignment for each solution
        front_no : ndarray
            Front number for each solution
        Z : ndarray
            Ideal point
        Fl : ndarray
            Indices of solutions in the worst front

        Returns
        -------
        int
            Index of solution to delete
        """
        # Get unique regions in worst front
        sub_region = np.unique(region[Fl])

        # Calculate crowding for these regions
        crowd = np.bincount(region, minlength=len(weights))
        crowd = crowd[sub_region]

        # Find most crowded regions
        phi = sub_region[crowd == np.max(crowd)]

        # Calculate PBI for solutions in crowded regions
        sub_mask = np.isin(region, phi)
        pbi = self._calculate_pbi(objs, weights, region, Z, sub_mask)

        # Sum PBI for each region
        pbi_sum = np.zeros(len(weights))
        for j in range(len(pbi)):
            if pbi[j] > 0:
                pbi_sum[region[j]] += pbi[j]

        # Find region with maximum PBI sum
        phi_idx = np.argmax(pbi_sum)
        phi_h = np.where(region == phi_idx)[0]

        if len(phi_h) > 1:
            # Select solution with maximum PBI
            x = phi_h[np.argmax(pbi[phi_h])]
        else:
            # Fallback to locate_worst
            x = self._locate_worst(objs, weights, region, front_no, Z)

        return x

    def _update_front_add(self, objs, front_no):
        """
        Update front numbers when a new solution is added (last in objs).

        Parameters
        ----------
        objs : ndarray
            Objective values (last row is new solution)
        front_no : ndarray
            Current front numbers

        Returns
        -------
        ndarray
            Updated front numbers
        """
        N, M = objs.shape
        front_no = np.append(front_no, 0)
        move = np.zeros(N, dtype=bool)
        move[N-1] = True
        current_f = 1

        # Locate the front number of the new solution
        while True:
            dominated = False
            for i in range(N-1):
                if front_no[i] == current_f:
                    # Check if solution i dominates the new solution
                    if np.all(objs[i] <= objs[-1]) and np.any(objs[i] < objs[-1]):
                        dominated = True
                        break

            if not dominated:
                break
            else:
                current_f += 1

        # Move down dominated solutions front by front
        while np.any(move):
            next_move = np.zeros(N, dtype=bool)
            for i in range(N):
                if front_no[i] == current_f:
                    dominated = False
                    for j in range(N):
                        if move[j]:
                            # Check if solution j dominates solution i
                            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                                dominated = True
                                break
                    next_move[i] = dominated

            front_no[move] = current_f
            current_f += 1
            move = next_move

        return front_no

    def _update_front_delete(self, objs, front_no, x):
        """
        Update front numbers when solution x is deleted.

        Parameters
        ----------
        objs : ndarray
            Objective values (after deletion)
        front_no : ndarray
            Front numbers before deletion
        x : int
            Index of deleted solution

        Returns
        -------
        ndarray
            Updated front numbers (after deletion)
        """
        N, M = objs.shape
        move = np.zeros(N+1, dtype=bool)
        move[x] = True
        current_f = front_no[x] + 1
        original_front_no = front_no.copy()

        while np.any(move):
            next_move = np.zeros(N+1, dtype=bool)
            for i in range(N+1):
                if original_front_no[i] == current_f:
                    dominated = False
                    for j in range(N+1):
                        if move[j]:
                            # Check domination (skip deleted solution)
                            if j < N:
                                if i < N:
                                    if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                                        dominated = True
                                        break
                    next_move[i] = dominated

            # Check if promoted solutions are dominated by previous front
            for i in range(N+1):
                if next_move[i] and i < N:
                    dominated = False
                    for j in range(N+1):
                        if original_front_no[j] == current_f - 1 and not move[j] and j < N:
                            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                                dominated = True
                                break
                    next_move[i] = not dominated

            front_no[move] = current_f - 2
            current_f += 1
            move = next_move

        # Remove the deleted solution's front number
        front_no = np.delete(front_no, x)

        return front_no

"""
Constrained Two-Archive Evolutionary Algorithm (C-TAEA)

This module implements C-TAEA for constrained multi-objective optimization problems.

References
----------
    [1] Li, Ke, et al. "Two-archive evolutionary algorithm for constrained multi-objective \
        optimization." IEEE Transactions on Evolutionary Computation 23.2 (2018): 303-315.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.01
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class CTAEA:
    """
    Constrained Two-Archive Evolutionary Algorithm for constrained multi-objective optimization.

    C-TAEA uses two co-evolving archives:
    - Convergence Archive (CA): Focuses on convergence towards the Pareto front
    - Diversity Archive (DA): Maintains diversity in the objective space

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
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, muc=20.0, mum=15.0,
                 save_data=True, save_path='./TestData', name='CTAEA_test',
                 disable_tqdm=True):
        """
        Initialize C-TAEA algorithm.

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
            Distribution index for polynomial mutation (PM) (default: 15.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'CTAEA_test')
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
        Execute the C-TAEA algorithm.

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
        for i in range(nt):
            W_i, n = uniform_point(n_per_task[i], no[i])
            W.append(W_i)
            n_per_task[i] = n

        # Initialize population and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()

        # Initialize CA and DA archives for each task
        CAs = []
        DAs = []

        for i in range(nt):
            CA_i = self._update_CA(None, objs[i], decs[i], cons[i], W[i], n_per_task[i])
            DA_i = self._update_DA(CA_i, None, objs[i], decs[i], cons[i], W[i], n_per_task[i])
            CAs.append(CA_i)
            DAs.append(DA_i)

        # History tracking uses CA
        all_decs, all_objs, all_cons = init_history(
            [CA_i['decs'] for CA_i in CAs],
            [CA_i['objs'] for CA_i in CAs],
            [CA_i['cons'] for CA_i in CAs]
        )

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                CA_i = CAs[i]
                DA_i = DAs[i]

                # Calculate the ratio of non-dominated solutions in CA and DA
                Hm_objs = np.vstack([CA_i['objs'], DA_i['objs']])
                Hm_cons = np.vstack([CA_i['cons'], DA_i['cons']]) if CA_i['cons'] is not None else None

                # Non-dominated sorting for the combined archive
                front_no, _ = nd_sort(Hm_objs, Hm_cons, Hm_objs.shape[0])

                # Calculate proportions
                CA_size = CA_i['objs'].shape[0]
                FrontNo_C = front_no[:CA_size]
                Nc = np.sum(FrontNo_C == 1)
                Pc = Nc / len(front_no)

                FrontNo_D = front_no[CA_size:]
                Nd = np.sum(FrontNo_D == 1)
                Pd = Nd / len(front_no)

                # Calculate the proportion of non-dominated solutions in CA
                front_no_CA, _ = nd_sort(CA_i['objs'], CA_i['cons'], CA_i['objs'].shape[0])
                NC = np.sum(front_no_CA == 1)
                PC = NC / CA_size

                # Generate offspring
                off_decs_list = []
                for j in range(n_per_task[i]):
                    # Select first parent based on Pc and Pd
                    if Pc > Pd:
                        P1 = self._mating_selection(CA_i)
                    else:
                        P1 = self._mating_selection(DA_i)

                    # Select second parent based on PC
                    pf = np.random.rand()
                    if pf < PC:
                        P2 = self._mating_selection(CA_i)
                    else:
                        P2 = self._mating_selection(DA_i)

                    # Combine parents
                    mating_pool = np.vstack([P1['decs'], P2['decs']])
                    offspring = ga_generation(mating_pool, muc=self.muc, mum=self.mum)
                    off_decs_list.append(offspring)

                # Combine all offspring
                off_decs = np.vstack(off_decs_list)
                off_objs, off_cons = evaluation_single(problem, off_decs, i)

                # Update CA and DA
                CA_i = self._update_CA(CA_i, off_objs, off_decs, off_cons, W[i], n_per_task[i])
                DA_i = self._update_DA(CA_i, DA_i, off_objs, off_decs, off_cons, W[i], n_per_task[i])

                CAs[i] = CA_i
                DAs[i] = DA_i

                # Update evaluation count
                nfes_per_task[i] += off_decs.shape[0]
                pbar.update(off_decs.shape[0])

                # Update history with CA
                append_history(all_decs[i], CA_i['decs'], all_objs[i], CA_i['objs'],
                               all_cons[i], CA_i['cons'])

        pbar.close()
        runtime = time.time() - start_time

        # Save results using CA
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name, save_data=self.save_data
        )

        return results

    def _mating_selection(self, archive):
        """
        Mating selection for C-TAEA using binary tournament selection.

        Parameters
        ----------
        archive : dict
            Archive containing 'objs', 'decs', and 'cons'

        Returns
        -------
        dict
            Selected individual with keys 'objs', 'decs', 'cons'
        """
        number = archive['objs'].shape[0]
        x_1 = np.random.randint(0, number)
        x_2 = np.random.randint(0, number)

        CV1 = np.sum(np.maximum(0, archive['cons'][x_1])) if archive['cons'] is not None else 0
        CV2 = np.sum(np.maximum(0, archive['cons'][x_2])) if archive['cons'] is not None else 0

        if CV1 > 0 and CV2 > 0:
            # Both infeasible, select randomly
            x = np.random.randint(0, number)
            selected_idx = x
        elif CV1 <= 0 and CV2 > 0:
            # x_1 is feasible
            selected_idx = x_1
        elif CV1 > 0 and CV2 <= 0:
            # x_2 is feasible
            selected_idx = x_2
        else:
            # Both feasible, use non-dominated sorting
            rnd_objs = np.vstack([archive['objs'][x_1], archive['objs'][x_2]])
            rnd_cons = np.vstack([archive['cons'][x_1], archive['cons'][x_2]]) if archive['cons'] is not None else None
            front_no, _ = nd_sort(rnd_objs, rnd_cons, 2)

            if front_no[0] <= front_no[1]:
                selected_idx = x_1
            else:
                selected_idx = x_2

        return {
            'objs': archive['objs'][selected_idx:selected_idx + 1],
            'decs': archive['decs'][selected_idx:selected_idx + 1],
            'cons': archive['cons'][selected_idx:selected_idx + 1] if archive['cons'] is not None else None
        }

    def _update_CA(self, CA, new_objs, new_decs, new_cons, W, N):
        """
        Update Convergence Archive (CA).

        Parameters
        ----------
        CA : dict or None
            Current CA with keys 'objs', 'decs', 'cons'
        new_objs : ndarray
            New objectives to add
        new_decs : ndarray
            New decisions to add
        new_cons : ndarray
            New constraints to add
        W : ndarray
            Weight vectors
        N : int
            Archive size

        Returns
        -------
        dict
            Updated CA with keys 'objs', 'decs', 'cons'
        """
        # Merge CA and new solutions
        if CA is None:
            Hc_objs = new_objs
            Hc_decs = new_decs
            Hc_cons = new_cons
        else:
            Hc_objs = np.vstack([CA['objs'], new_objs])
            Hc_decs = np.vstack([CA['decs'], new_decs])
            Hc_cons = np.vstack([CA['cons'], new_cons]) if CA['cons'] is not None else new_cons

        # Calculate constraint violation
        CV = np.sum(np.maximum(0, Hc_cons), axis=1) if Hc_cons is not None else np.zeros(Hc_objs.shape[0])

        # Collect feasible solutions
        Sc_mask = CV == 0
        Sc_objs = Hc_objs[Sc_mask]
        Sc_decs = Hc_decs[Sc_mask]
        Sc_cons = Hc_cons[Sc_mask] if Hc_cons is not None else None

        if len(Sc_objs) == N:
            return {'objs': Sc_objs, 'decs': Sc_decs, 'cons': Sc_cons}

        elif len(Sc_objs) > N:
            # More feasible solutions than needed
            S_objs, S_decs, S_cons = self._truncate_feasible(Sc_objs, Sc_decs, Sc_cons, W, N)
            return {'objs': S_objs, 'decs': S_decs, 'cons': S_cons}

        else:
            # Less feasible solutions than needed, add infeasible ones
            SI_mask = ~Sc_mask
            SI_objs = Hc_objs[SI_mask]
            SI_decs = Hc_decs[SI_mask]
            SI_cons = Hc_cons[SI_mask] if Hc_cons is not None else None

            S_objs, S_decs, S_cons = self._fill_with_infeasible(
                Sc_objs, Sc_decs, Sc_cons, SI_objs, SI_decs, SI_cons, W, N
            )
            return {'objs': S_objs, 'decs': S_decs, 'cons': S_cons}

    def _truncate_feasible(self, objs, decs, cons, W, N):
        """
        Truncate feasible solutions using non-dominated sorting and crowding.

        Parameters
        ----------
        objs : ndarray
            Objective values
        decs : ndarray
            Decision variables
        cons : ndarray or None
            Constraint values
        W : ndarray
            Weight vectors
        N : int
            Target size

        Returns
        -------
        tuple
            Truncated (objs, decs, cons)
        """
        # Non-dominated sorting
        front_no, max_fno = nd_sort(objs, cons, objs.shape[0])

        # Add solutions front by front
        S_indices = []
        for i in range(1, max_fno + 1):
            current_front = np.where(front_no == i)[0]
            if len(S_indices) + len(current_front) <= N:
                S_indices.extend(current_front)
            else:
                # Need to select from this front
                S_indices.extend(current_front)
                break

        S_indices = np.array(S_indices)
        S_objs = objs[S_indices]
        S_decs = decs[S_indices]
        S_cons = cons[S_indices] if cons is not None else None

        # Truncate if necessary
        while len(S_objs) > N:
            # Normalization
            Zmax = np.max(S_objs, axis=0)
            Zmin = np.min(S_objs, axis=0)
            range_vals = Zmax - Zmin
            range_vals[range_vals == 0] = 1  # Avoid division by zero

            SPopObj = (S_objs - Zmin) / range_vals

            # Associate each solution with a subregion
            cosine = 1 - cdist(SPopObj, W, metric='cosine')
            Region = np.argmax(cosine, axis=1)

            # Find the most crowded subregion
            unique_regions, counts = np.unique(Region, return_counts=True)
            most_crowded = unique_regions[np.argmax(counts)]

            # Find solutions in the most crowded subregion
            crowded_mask = Region == most_crowded
            crowded_indices = np.where(crowded_mask)[0]
            S_crowded_objs = S_objs[crowded_mask]

            # Calculate distances
            dist = cdist(S_crowded_objs, S_crowded_objs)
            np.fill_diagonal(dist, np.inf)

            # Find the closest pair
            min_dist = np.min(dist)
            closest_indices = np.where(dist == min_dist)
            row = closest_indices[0][0]

            # Map back to S indices
            closest_in_S = crowded_indices[row]

            # Calculate Tchebycheff distance for tie-breaking
            St_objs = S_objs[crowded_indices]
            Region_St = Region[crowded_indices]
            Z = np.min(St_objs, axis=0)

            g_tch = np.max(np.abs(St_objs - Z) / W[Region_St], axis=1)
            worst_idx = np.argmax(g_tch)

            # Map to S indices
            worst_in_S = crowded_indices[worst_idx]

            # Remove the worst solution
            mask = np.ones(len(S_objs), dtype=bool)
            mask[worst_in_S] = False
            S_objs = S_objs[mask]
            S_decs = S_decs[mask]
            S_cons = S_cons[mask] if S_cons is not None else None

        return S_objs, S_decs, S_cons

    def _fill_with_infeasible(self, Sc_objs, Sc_decs, Sc_cons, SI_objs, SI_decs, SI_cons, W, N):
        """
        Fill archive with infeasible solutions when there are not enough feasible ones.

        Parameters
        ----------
        Sc_objs : ndarray
            Feasible objectives
        Sc_decs : ndarray
            Feasible decisions
        Sc_cons : ndarray or None
            Feasible constraints
        SI_objs : ndarray
            Infeasible objectives
        SI_decs : ndarray
            Infeasible decisions
        SI_cons : ndarray
            Infeasible constraints
        W : ndarray
            Weight vectors
        N : int
            Target size

        Returns
        -------
        tuple
            Combined (objs, decs, cons)
        """
        # Create two-objective optimization problem: minimize CV and Tchebycheff
        CV_SI = np.sum(np.maximum(0, SI_cons), axis=1, keepdims=True)

        # Associate with subregions
        cosine = 1 - cdist(SI_objs, W, metric='cosine')
        Region_SI = np.argmax(cosine, axis=1)
        Z = np.min(SI_objs, axis=0)

        g_tch = np.max(np.abs(SI_objs - Z) / W[Region_SI], axis=1, keepdims=True)

        # Combine CV and Tchebycheff as bi-objective
        PopObj = np.hstack([CV_SI, g_tch])

        # Non-dominated sorting
        front_no, max_fno = nd_sort(PopObj, None, PopObj.shape[0])

        # Start with feasible solutions
        S_objs = Sc_objs
        S_decs = Sc_decs
        S_cons = Sc_cons

        # Add infeasible solutions front by front
        for i in range(1, max_fno + 1):
            current_front = np.where(front_no == i)[0]

            if len(S_objs) + len(current_front) <= N:
                S_objs = np.vstack([S_objs, SI_objs[current_front]])
                S_decs = np.vstack([S_decs, SI_decs[current_front]])
                S_cons = np.vstack([S_cons, SI_cons[current_front]]) if S_cons is not None else SI_cons[current_front]
            else:
                # Need to select from this front
                needed = N - len(S_objs)
                last_front_indices = current_front

                # Sort by CV
                CV_last = CV_SI[last_front_indices].flatten()
                sorted_indices = np.argsort(CV_last)[:needed]

                S_objs = np.vstack([S_objs, SI_objs[last_front_indices[sorted_indices]]])
                S_decs = np.vstack([S_decs, SI_decs[last_front_indices[sorted_indices]]])
                S_cons = np.vstack([S_cons, SI_cons[last_front_indices[sorted_indices]]]) if S_cons is not None else \
                SI_cons[last_front_indices[sorted_indices]]
                break

            if len(S_objs) >= N:
                break

        return S_objs, S_decs, S_cons

    def _update_DA(self, CA, DA, new_objs, new_decs, new_cons, W, N):
        """
        Update Diversity Archive (DA).

        Parameters
        ----------
        CA : dict
            Current CA with keys 'objs', 'decs', 'cons'
        DA : dict or None
            Current DA with keys 'objs', 'decs', 'cons'
        new_objs : ndarray
            New objectives to add
        new_decs : ndarray
            New decisions to add
        new_cons : ndarray
            New constraints to add
        W : ndarray
            Weight vectors
        N : int
            Archive size

        Returns
        -------
        dict
            Updated DA with keys 'objs', 'decs', 'cons'
        """
        # Merge DA and new solutions
        if DA is None:
            Hd_objs = new_objs.copy()
            Hd_decs = new_decs.copy()
            Hd_cons = new_cons.copy() if new_cons is not None else None
        else:
            Hd_objs = np.vstack([DA['objs'], new_objs])
            Hd_decs = np.vstack([DA['decs'], new_decs])
            Hd_cons = np.vstack([DA['cons'], new_cons]) if DA['cons'] is not None else new_cons

        # Associate solutions in CA with subregions
        cosine_CA = 1 - cdist(CA['objs'], W, metric='cosine')
        Region_CA = np.argmax(cosine_CA, axis=1)

        # Build DA iteratively - use a different approach to avoid index tracking issues
        selected_indices = []
        available_mask = np.ones(len(Hd_objs), dtype=bool)
        itr = 1
        max_itr = 10 * N  # Prevent infinite loop

        while len(selected_indices) < N and itr < max_itr:
            for region_idx in range(N):  # region_idx is the subregion index
                if len(selected_indices) >= N:
                    break

                # Count how many CA solutions are in this region
                current_c_count = np.sum(Region_CA == region_idx)

                if current_c_count < itr:
                    # Need to add solutions from Hd to this subregion
                    for j in range(itr - current_c_count):
                        if len(selected_indices) >= N:
                            break

                        # Find available candidates in Hd for this region
                        if np.sum(available_mask) == 0:
                            break

                        cosine_Hd = 1 - cdist(Hd_objs[available_mask], W, metric='cosine')
                        Region_Hd_available = np.argmax(cosine_Hd, axis=1)

                        # Get indices in the available subset that belong to current region
                        candidates_in_subset = np.where(Region_Hd_available == region_idx)[0]

                        if len(candidates_in_subset) > 0:
                            # Map back to original Hd indices
                            available_indices = np.where(available_mask)[0]
                            candidates_in_Hd = available_indices[candidates_in_subset]

                            # Get objectives and constraints for candidates
                            cand_objs = Hd_objs[candidates_in_Hd]
                            cand_cons = Hd_cons[candidates_in_Hd] if Hd_cons is not None else None

                            # Non-dominated sorting
                            front_no, _ = nd_sort(cand_objs, cand_cons, len(candidates_in_Hd))
                            nd_mask = front_no == 1
                            nd_indices_in_Hd = candidates_in_Hd[nd_mask]

                            if len(nd_indices_in_Hd) > 0:
                                # Calculate Tchebycheff distance for non-dominated solutions
                                nd_objs = Hd_objs[nd_indices_in_Hd]
                                Z = np.min(nd_objs, axis=0)

                                # Get regions for nd solutions
                                cosine_nd = 1 - cdist(nd_objs, W, metric='cosine')
                                Region_nd = np.argmax(cosine_nd, axis=1)

                                g_tch = np.max(np.abs(nd_objs - Z) / W[Region_nd], axis=1)

                                # Select the best one (minimum Tchebycheff)
                                best_local_idx = np.argmin(g_tch)
                                best_global_idx = nd_indices_in_Hd[best_local_idx]

                                # Add to selected and mark as unavailable
                                selected_indices.append(best_global_idx)
                                available_mask[best_global_idx] = False
                        else:
                            break

            itr += 1

        # Build final DA from selected indices
        if len(selected_indices) == 0:
            # Fallback: just take first N solutions
            selected_indices = list(range(min(N, len(Hd_objs))))

        # Ensure we have exactly N solutions
        if len(selected_indices) < N:
            # Fill with remaining available solutions
            remaining_indices = np.where(available_mask)[0]
            needed = N - len(selected_indices)
            if len(remaining_indices) > 0:
                selected_indices.extend(remaining_indices[:needed].tolist())
            else:
                # If no more available, just duplicate some existing ones
                dup_count = 0
                while len(selected_indices) < N:
                    selected_indices.append(selected_indices[dup_count % len(selected_indices)])
                    dup_count += 1

        selected_indices = selected_indices[:N]

        return {
            'objs': Hd_objs[selected_indices],
            'decs': Hd_decs[selected_indices],
            'cons': Hd_cons[selected_indices] if Hd_cons is not None else None
        }
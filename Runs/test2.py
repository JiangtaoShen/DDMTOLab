"""
Surrogate-Assisted Cooperative Swarm Optimization (SA-COSO)

This module implements SA-COSO for expensive single-objective optimization problems.

References
----------
    [1] C. Sun, Y. Jin, R. Cheng, J. Ding, and J. Zeng. Surrogate-assisted
        cooperative swarm optimization of high-dimensional expensive problems.
        IEEE Transactions on Evolutionary Computation, 2017, 21(4): 644-660.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.13
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from Methods.Algo_Methods.algo_utils import *
from Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class SACOSO:
    """
    Surrogate-Assisted Cooperative Swarm Optimization for expensive optimization problems.

    This algorithm uses two cooperative swarms:
    1. FES-PSO: Small swarm with Fitness Estimation Strategy to reduce evaluations
    2. RBF-SLPSO: Large swarm with RBF-assisted Social Learning PSO

    Key features:
    - Cooperative information sharing between two swarms
    - Fitness Estimation Strategy (FES) for FES-PSO
    - Social Learning mechanism for RBF-SLPSO
    - RBF surrogate model for fitness approximation

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '1-K',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n_fes=30, n_rbf=200,
                 n_sample_fes=4, n_sample_rbf=1, save_data=True, save_path='./TestData',
                 name='SACOSO_test', disable_tqdm=True):
        """
        Initialize SA-COSO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: n_fes + n_rbf)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 500)
        n_fes : int, optional
            Population size of FES-assisted PSO (default: 30)
        n_rbf : int, optional
            Population size of RBF-assisted SL-PSO (default: 200)
        n_sample_fes : int, optional
            Number of samples per iteration from FES-PSO (default: 9)
        n_sample_rbf : int, optional
            Number of samples per iteration from RBF-SLPSO (default: 1)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'SACOSO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_fes = n_fes  # FES-PSO population size
        self.n_rbf = n_rbf  # RBF-SLPSO population size
        self.n_sample_fes = n_sample_fes  # Fixed samples from FES-PSO per iteration
        self.n_sample_rbf = n_sample_rbf  # Fixed samples from RBF-SLPSO per iteration
        self.n_initial = n_initial if n_initial is not None else (n_fes + n_rbf)
        self.max_nfes = max_nfes if max_nfes is not None else 500
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        # PSO parameters
        self.c1 = 2.05  # Cognitive coefficient
        self.c2 = 1.025  # Social coefficient (GbestFES)
        self.c3 = 1.025  # Social coefficient (GbestRBF)
        self.w = 0.7298  # Inertia weight (constriction factor)

        # RBF parameters
        self.max_node = 8  # Maximum nodes for RBF

    def optimize(self):
        """
        Execute the SA-COSO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Initialize history
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task)

        # Initialize data structures for each task
        # Historical database
        hx = [decs[i].copy() for i in range(nt)]
        hf = [objs[i].flatten().copy() for i in range(nt)]

        # FES-PSO swarm
        swarm_fes = []  # [pos, obj]
        vel_fes = []
        pbest_fes = []
        pbest_fes_val = []
        gbest_fes = []
        gbest_fes_val = []

        # RBF-SLPSO swarm
        swarm_rbf = []  # [pos, obj]
        delta_rbf = []  # velocity for SL-PSO
        gbest_rbf = []
        gbest_rbf_val = []

        # Spread sum for RBF
        spread_sum = [0.0] * nt

        for i in range(nt):
            dim = dims[i]
            n_total = n_initial_per_task[i]

            # Split initial population
            n_fes_actual = min(self.n_fes, n_total)
            n_rbf_actual = min(self.n_rbf, n_total - n_fes_actual)

            # FES-PSO initialization
            pos_fes = hx[i][:n_fes_actual].copy()
            obj_fes = hf[i][:n_fes_actual].copy()
            swarm_fes.append({'pos': pos_fes, 'obj': obj_fes})

            vel_i = np.random.rand(n_fes_actual, dim) - 0.5
            vel_fes.append(vel_i)

            pbest_fes.append(pos_fes.copy())
            pbest_fes_val.append(obj_fes.copy())

            best_idx = np.argmin(obj_fes)
            gbest_fes.append(pos_fes[best_idx].copy())
            gbest_fes_val.append(obj_fes[best_idx])

            # RBF-SLPSO initialization
            if n_rbf_actual > 0:
                pos_rbf = hx[i][n_fes_actual:n_fes_actual + n_rbf_actual].copy()
                obj_rbf = hf[i][n_fes_actual:n_fes_actual + n_rbf_actual].copy()
            else:
                # Generate additional random samples if needed
                pos_rbf = np.random.rand(self.n_rbf, dim)
                obj_rbf_arr, _ = evaluation_single(problem, pos_rbf, i)
                obj_rbf = obj_rbf_arr.flatten()
                hx[i] = np.vstack([hx[i], pos_rbf])
                hf[i] = np.concatenate([hf[i], obj_rbf])
                nfes_per_task[i] += self.n_rbf

            swarm_rbf.append({'pos': pos_rbf, 'obj': obj_rbf})

            delta_i = np.random.rand(len(pos_rbf), dim) - 0.5
            delta_rbf.append(delta_i)

            best_idx_rbf = np.argmin(obj_rbf)
            gbest_rbf.append(pos_rbf[best_idx_rbf].copy())
            gbest_rbf_val.append(obj_rbf[best_idx_rbf])

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        iter_count = [0] * nt

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                iter_count[i] += 1

                # Determine global best across both swarms
                if gbest_fes_val[i] < gbest_rbf_val[i]:
                    gbest = gbest_fes[i].copy()
                    gbest_val = gbest_fes_val[i]
                else:
                    gbest = gbest_rbf[i].copy()
                    gbest_val = gbest_rbf_val[i]

                # Update spread for RBF
                pos_range = np.max(hx[i], axis=0) - np.min(hx[i], axis=0)
                spread_sum[i] += np.sqrt(np.sum(pos_range ** 2))
                spread = spread_sum[i] / iter_count[i]

                # Build RBF surrogate model
                rbf_model = self._build_rbf_model(hx[i], hf[i], spread, dim)

                # ===== FES-PSO Update =====
                new_evals_fes = self._fes_pso_update(
                    i, problem, dim, rbf_model,
                    swarm_fes[i], vel_fes[i], pbest_fes[i], pbest_fes_val[i],
                    gbest_fes[i], gbest_fes_val[i], gbest_rbf[i],
                    hx[i], hf[i], self.n_sample_fes
                )

                # Update history and counters for FES-PSO
                for pos, obj in new_evals_fes:
                    if nfes_per_task[i] >= max_nfes_per_task[i]:
                        break
                    hx[i] = np.vstack([hx[i], pos.reshape(1, -1)])
                    hf[i] = np.concatenate([hf[i], [obj]])
                    nfes_per_task[i] += 1
                    pbar.update(1)

                # Update FES gbest
                best_idx = np.argmin(pbest_fes_val[i])
                if pbest_fes_val[i][best_idx] < gbest_fes_val[i]:
                    gbest_fes[i] = pbest_fes[i][best_idx].copy()
                    gbest_fes_val[i] = pbest_fes_val[i][best_idx]

                # ===== RBF-SLPSO Update =====
                if nfes_per_task[i] < max_nfes_per_task[i]:
                    # Rebuild RBF model with updated data
                    spread = spread_sum[i] / iter_count[i]
                    rbf_model = self._build_rbf_model(hx[i], hf[i], spread, dim)

                    new_evals_rbf = self._rbf_slpso_update(
                        i, problem, dim, rbf_model,
                        swarm_rbf[i], delta_rbf[i], gbest,
                        hx[i], hf[i], self.n_sample_rbf
                    )

                    # Update history for RBF-SLPSO
                    for pos, obj in new_evals_rbf:
                        if nfes_per_task[i] >= max_nfes_per_task[i]:
                            break
                        hx[i] = np.vstack([hx[i], pos.reshape(1, -1)])
                        hf[i] = np.concatenate([hf[i], [obj]])
                        nfes_per_task[i] += 1
                        pbar.update(1)

                        # Update RBF gbest
                        if obj < gbest_rbf_val[i]:
                            gbest_rbf[i] = pos.copy()
                            gbest_rbf_val[i] = obj

                # Append to history for results
                all_decs[i].append(hx[i].copy())
                all_objs[i].append(hf[i].reshape(-1, 1).copy())

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(
            all_decs=all_decs,
            all_objs=all_objs,
            runtime=runtime,
            max_nfes=nfes_per_task,
            bounds=problem.bounds,
            save_path=self.save_path,
            filename=self.name,
            save_data=self.save_data
        )

        return results

    def _build_rbf_model(self, hx, hf, spread, dim):
        """
        Build RBF surrogate model.

        Parameters
        ----------
        hx : np.ndarray
            Historical positions
        hf : np.ndarray
            Historical fitness values
        spread : float
            Spread parameter for RBF
        dim : int
            Problem dimension

        Returns
        -------
        callable
            RBF model function
        """
        if spread <= 0:
            spread = 1.0

        try:
            rbf_interpolator = RBFInterpolator(hx, hf, kernel='gaussian', epsilon=1.0 / spread)
        except:
            rbf_interpolator = RBFInterpolator(hx, hf, kernel='thin_plate_spline')

        def rbf_model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            pred = rbf_interpolator(x)
            return pred.flatten()

        return rbf_model

    def _fes_pso_update(self, task_idx, problem, dim, rbf_model,
                        swarm, vel, pbest, pbest_val,
                        gbest_fes, gbest_fes_val, gbest_rbf,
                        hx, hf, n_sample):
        """
        FES-PSO update with fixed number of samples.

        Parameters
        ----------
        n_sample : int
            Fixed number of particles to evaluate

        Returns
        -------
        list
            List of (position, objective) tuples for new evaluations
        """
        new_evals = []
        n = len(swarm['pos'])

        # PSO velocity and position update
        r1 = np.random.rand(n, dim)
        r2 = np.random.rand(n, dim)
        r3 = np.random.rand(n, dim)

        # Velocity update: influenced by pbest, gbest_fes, and gbest_rbf
        vel_new = self.w * (vel +
                           self.c1 * r1 * (pbest - swarm['pos']) +
                           self.c2 * r2 * (gbest_fes - swarm['pos']) +
                           self.c3 * r3 * (gbest_rbf - swarm['pos']))

        # Velocity clamping
        vel_new = np.clip(vel_new, -1.0, 1.0)
        vel[:] = vel_new

        # Position update
        pos_new = swarm['pos'] + vel_new
        pos_new = np.clip(pos_new, 0.0, 1.0)
        swarm['pos'] = pos_new

        # Use surrogate model for fitness approximation
        srgt_obj = rbf_model(pos_new)
        swarm['obj'] = srgt_obj.copy()

        # Select top n_sample particles by surrogate prediction (ascending order)
        sorted_idx = np.argsort(srgt_obj)

        # Evaluate top n_sample non-duplicate particles
        eval_count = 0
        for idx in sorted_idx:
            if eval_count >= n_sample:
                break

            if not self._is_duplicate(pos_new[idx], hx):
                obj_arr, _ = evaluation_single(problem, pos_new[idx:idx + 1], task_idx)
                obj = obj_arr[0, 0]
                swarm['obj'][idx] = obj
                new_evals.append((pos_new[idx].copy(), obj))
                eval_count += 1

                # Update personal best
                if obj < pbest_val[idx]:
                    pbest[idx] = pos_new[idx].copy()
                    pbest_val[idx] = obj

        return new_evals

    def _rbf_slpso_update(self, task_idx, problem, dim, rbf_model,
                          swarm, delta, gbest, hx, hf, n_sample):
        """
        RBF-assisted SL-PSO update with fixed number of samples.

        Parameters
        ----------
        n_sample : int
            Fixed number of particles to evaluate

        Returns
        -------
        list
            List of (position, objective) tuples for new evaluations
        """
        new_evals = []
        n = len(swarm['pos'])

        # Combine swarm with historical best as demonstrators
        n_demons = min(n, len(hf))
        idx_sorted = np.argsort(hf)[:n_demons]
        demons_pos = np.vstack([swarm['pos'], hx[idx_sorted]])
        demons_obj = np.concatenate([swarm['obj'], hf[idx_sorted]])

        # Social Learning PSO update
        for j in range(n):
            # Find better individuals (demonstrators)
            better_mask = demons_obj < swarm['obj'][j]
            better_idx = np.where(better_mask)[0]

            if len(better_idx) > 0:
                # Choose demonstrators for each dimension
                chosen = np.zeros(dim)
                if len(better_idx) == 1:
                    chosen = demons_pos[better_idx[0]]
                elif len(better_idx) == 2:
                    mix = np.random.rand(dim) > 0.5
                    chosen[mix] = demons_pos[better_idx[0], mix]
                    chosen[~mix] = demons_pos[better_idx[1], ~mix]
                else:
                    for d in range(dim):
                        rand_idx = better_idx[np.random.randint(len(better_idx))]
                        chosen[d] = demons_pos[rand_idx, d]

                # Update velocity (delta) and position
                delta[j] = np.random.rand(dim) * delta[j] + np.random.rand(dim) * (chosen - swarm['pos'][j])
                delta[j] = np.clip(delta[j], -1.0, 1.0)
                swarm['pos'][j] = swarm['pos'][j] + delta[j]

        # Boundary handling
        swarm['pos'] = np.clip(swarm['pos'], 0.0, 1.0)

        # Update fitness using surrogate
        swarm['obj'] = rbf_model(swarm['pos'])

        # Select top n_sample particles by surrogate prediction
        sorted_idx = np.argsort(swarm['obj'])

        # Evaluate top n_sample non-duplicate particles
        eval_count = 0
        for idx in sorted_idx:
            if eval_count >= n_sample:
                break

            if not self._is_duplicate(swarm['pos'][idx], hx):
                obj_arr, _ = evaluation_single(problem, swarm['pos'][idx:idx + 1], task_idx)
                obj = obj_arr[0, 0]
                swarm['obj'][idx] = obj
                new_evals.append((swarm['pos'][idx].copy(), obj))
                eval_count += 1

        return new_evals

    def _is_duplicate(self, x, X, epsilon=1e-10):
        """
        Check if position x is a duplicate in X.
        """
        if len(X) == 0:
            return False

        x = x.reshape(1, -1)
        dist = cdist(x, X, metric='euclidean')
        return np.min(dist) < epsilon

# Example usage
if __name__ == "__main__":
    from Problems.MTSO.cec17_mtso import CEC17MTSO

    problem = CEC17MTSO().P1()
    results = SACOSO(problem, n_initial=100, max_nfes=300, disable_tqdm=False).optimize()

    print(f"Best objective: {results.best_objs[0]}")
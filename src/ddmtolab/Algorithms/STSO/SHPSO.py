"""
Surrogate-assisted Hierarchical Particle Swarm Optimization (SHPSO)

This module implements SHPSO for expensive single-objective optimization problems.

References
----------
    [1] Yu, Haibo, et al. "Surrogate-assisted hierarchical particle swarm optimization." Information Sciences 454 (2018): 59-72.

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
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Algorithms.STSO.SLPSO import SLPSO
from ddmtolab.Methods.mtop import MTOP
import warnings

warnings.filterwarnings("ignore")


class SHPSO:
    """
    Surrogate-assisted Hierarchical Particle Swarm Optimization for expensive optimization problems.

    This algorithm uses a two-level hierarchical structure:
    1. Upper level: RBF surrogate model optimized by SL-PSO to find promising regions
    2. Lower level: PSO swarm guided by surrogate model optimum with prescreening strategy
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

    def __init__(self, problem, n_initial=None, max_nfes=None, ps=None, mu=5,
                 save_data=True, save_path='./TestData', name='SHPSO_test', disable_tqdm=True):
        """
        Initialize SHPSO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        ps : int or List[int], optional
            Particle swarm size per task (default: 20)
        mu : int, optional
            Number of new samples per iteration (default: 1)
            - 1: Only evaluate surrogate optimum (most conservative)
            - k: Evaluate surrogate optimum + top (k-1) prescreened particles
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './TestData')
        name : str, optional
            Name for the experiment (default: 'SHPSO_test')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.ps = ps if ps is not None else 20
        self.mu = mu
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        self.c1 = 2.05  # Cognitive coefficient
        self.c2 = 2.05  # Social coefficient
        self.w = 0.7298  # Inertia weight (constriction factor)
        self.slpso_max_gen = 50

    def optimize(self):
        """
        Execute the SHPSO algorithm.

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
        ps_per_task = par_list(self.ps, nt)

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Initialize history: reorganize initial data into task-specific lists
        all_decs = reorganize_initial_data(decs, nt, n_initial_per_task, interval=self.mu)
        all_objs = reorganize_initial_data(objs, nt, n_initial_per_task, interval=self.mu)

        # Historical database for each task
        hx = [decs[i].copy() for i in range(nt)]  # Historical positions
        hf = [objs[i].flatten().copy() for i in range(nt)]  # Historical fitness

        # Initialize PSO swarm for each task
        pos = []  # Current positions
        vel = []  # Current velocities
        pbest = []  # Personal best positions
        pbestval = []  # Personal best values
        gbest = []  # Global best position
        gbestval = []  # Global best value

        for i in range(nt):
            dim = dims[i]
            ps = ps_per_task[i]

            # Initialize particle positions from initial samples
            if n_initial_per_task[i] >= ps:
                # Select ps best particles from initial samples
                idx_sorted = np.argsort(hf[i])[:ps]
                pos_i = hx[i][idx_sorted].copy()
                obj_i = hf[i][idx_sorted].copy()
            else:
                # Use all initial samples and add random particles
                pos_i = hx[i].copy()
                obj_i = hf[i].copy()
                extra = ps - len(pos_i)
                if extra > 0:
                    extra_pos = np.random.rand(extra, dim)
                    pos_i = np.vstack([pos_i, extra_pos])
                    # Evaluate extra particles
                    extra_objs, _ = evaluation_single(problem, extra_pos, i)
                    obj_i = np.concatenate([obj_i, extra_objs.flatten()])
                    # Update history
                    hx[i] = np.vstack([hx[i], extra_pos])
                    hf[i] = np.concatenate([hf[i], extra_objs.flatten()])
                    nfes_per_task[i] += extra

            pos.append(pos_i)

            # Initialize velocities
            mv = 0.5
            vel_i = -mv + 2 * mv * np.random.rand(ps, dim)
            vel.append(vel_i)

            # Initialize personal best
            pbest.append(pos_i.copy())
            pbestval.append(obj_i.copy())

            # Initialize global best
            gbestidx = np.argmin(obj_i)
            gbest.append(pos_i[gbestidx].copy())
            gbestval.append(obj_i[gbestidx])

        # Best surrogate model optimum for each task
        bestp = [gbest[i].copy() for i in range(nt)]
        besty = [gbestval[i] for i in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                ps = ps_per_task[i]

                # Determine training sample size
                if dim < 100:
                    gs = 100
                else:
                    gs = 150

                # Build global RBF surrogate model using best historical samples
                rbf_model = self._build_rbf_model(hx[i], hf[i], gs, dim)

                # Record old best
                besty_old = besty[i]
                bestp_old = bestp[i].copy()

                # ===== Upper Level: Optimize surrogate model using SL-PSO =====
                bestp_new = self._optimize_surrogate_slpso(rbf_model, hx[i], dim)

                # Exact evaluate the surrogate optimum
                besty_new_arr, _ = evaluation_single(problem, bestp_new.reshape(1, -1), i)
                besty_new = besty_new_arr[0, 0]
                nfes_per_task[i] += 1
                pbar.update(1)

                # Update history database
                if not is_duplicate(bestp_new, hx[i]):
                    hx[i] = np.vstack([hx[i], bestp_new])
                    hf[i] = np.concatenate([hf[i], [besty_new]])

                # Update surrogate optimum
                if besty_new < besty_old:
                    besty[i] = besty_new
                    bestp[i] = bestp_new.copy()
                else:
                    besty[i] = besty_old
                    bestp[i] = bestp_old.copy()

                # Update RBF model if new optimum is better
                if besty_new < besty_old:
                    rbf_model = self._build_rbf_model(hx[i], hf[i], gs, dim)

                # ===== Lower Level: PSO guided by surrogate optimum =====
                # Determine guidance strategy
                if besty[i] < gbestval[i]:
                    # Use surrogate optimum to guide particles
                    guide_pos = bestp[i]
                    # Update gbest's corresponding pbest
                    gbestidx = np.argmin(pbestval[i])
                    pbest[i][gbestidx] = bestp[i].copy()
                    pbestval[i][gbestidx] = besty[i]
                else:
                    # Use current gbest
                    guide_pos = gbest[i]

                # PSO velocity and position update
                r1 = np.random.rand(ps, dim)
                r2 = np.random.rand(ps, dim)

                # Velocity update with constriction factor
                vel[i] = self.w * (vel[i] + self.c1 * r1 * (pbest[i] - pos[i]) + self.c2 * r2 * (guide_pos - pos[i]))

                # Velocity clamping
                mv = 0.5
                vel[i] = np.clip(vel[i], -mv, mv)

                # Position update
                pos[i] = pos[i] + vel[i]

                # Boundary handling with random reinitialization
                out_lower = pos[i] < 0
                out_upper = pos[i] > 1
                pos[i] = np.where(out_lower, 0.25 * np.random.rand(ps, dim), pos[i])
                pos[i] = np.where(out_upper, 1 - 0.25 * np.random.rand(ps, dim), pos[i])

                # ===== Prescreening: Select top candidates for evaluation =====
                # Predict fitness using surrogate model
                e_pred = rbf_model(pos[i]).flatten()

                # Sort particles by predicted fitness (ascending, best first)
                sorted_idx = np.argsort(e_pred)

                # Select top (n_sample - 1) candidates (since surrogate optimum already uses 1)
                n_prescreen = self.mu - 1
                candidates_to_eval = []

                for idx in sorted_idx:
                    if len(candidates_to_eval) >= n_prescreen:
                        break
                    # Check if not duplicate in history
                    if not is_duplicate(pos[i][idx], hx[i]):
                        candidates_to_eval.append(idx)

                # Exact evaluate prescreened candidates
                for idx in candidates_to_eval:
                    if nfes_per_task[i] >= max_nfes_per_task[i]:
                        break

                    candidate_pos = pos[i][idx:idx + 1]
                    candidate_obj, _ = evaluation_single(problem, candidate_pos, i)
                    e_true = candidate_obj[0, 0]
                    nfes_per_task[i] += 1
                    pbar.update(1)

                    # Update history database
                    hx[i] = np.vstack([hx[i], candidate_pos])
                    hf[i] = np.concatenate([hf[i], [e_true]])

                    # Update personal best
                    if e_true < pbestval[i][idx]:
                        pbest[i][idx] = candidate_pos.flatten()
                        pbestval[i][idx] = e_true

                # Update global best
                gbestidx = np.argmin(pbestval[i])
                gbest[i] = pbest[i][gbestidx].copy()
                gbestval[i] = pbestval[i][gbestidx]


                # Append to history for results
                all_decs[i].append(hx[i].copy())
                all_objs[i].append(hf[i].reshape(-1, 1).copy())

        pbar.close()
        runtime = time.time() - start_time

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    # Build RBF surrogate model using best historical samples
    def _build_rbf_model(self, hx, hf, gs, dim):

        # Sort by fitness and select best gs samples
        idx_sorted = np.argsort(hf)
        n_select = min(gs, len(hf))
        idx_select = idx_sorted[:n_select]

        ghx = hx[idx_select]
        ghf = hf[idx_select]

        # Calculate spread parameter
        n_samples = len(ghf)
        if n_samples > 1:
            dist_matrix = cdist(ghx, ghx, metric='euclidean')
            max_dist = dist_matrix.max()
            spread = max_dist / (dim * n_samples) ** (1.0 / dim)
        else:
            spread = 1.0

        # Build RBF interpolator
        try:
            rbf_interpolator = RBFInterpolator(ghx, ghf, kernel='gaussian', epsilon=1.0 / spread)
        except:
            rbf_interpolator = RBFInterpolator(ghx, ghf, kernel='thin_plate_spline')

        def rbf_model(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            pred = rbf_interpolator(x)
            return pred.reshape(-1, 1)

        return rbf_model

    # Optimize surrogate model using SL-PSO
    def _optimize_surrogate_slpso(self, rbf_model, hx, dim):

        # Create surrogate problem wrapper
        def surrogate_func(x):
            return rbf_model(x)

        surrogate_problem = MTOP()
        surrogate_problem.add_task(objective_func=surrogate_func, dim=dim)

        # Population size and generations for SL-PSO
        n_pop = 100
        max_gen = self.slpso_max_gen
        max_nfes = n_pop * max_gen

        # Run SL-PSO
        slpso = SLPSO(surrogate_problem, n=n_pop, max_nfes=max_nfes, disable_tqdm=True)
        results = slpso.optimize()

        # Get best solution
        best_pos = results.best_decs[0].flatten()

        # Clip to [0, 1]
        best_pos = np.clip(best_pos, 0.0, 1.0)

        return best_pos
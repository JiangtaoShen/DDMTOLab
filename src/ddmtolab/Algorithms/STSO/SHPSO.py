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
from scipy.stats import qmc
from ddmtolab.Methods.Algo_Methods.algo_utils import *
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
        'n_tasks': '[1, K]',
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

    def __init__(self, problem, n_initial=None, max_nfes=None, ps=None,
                 save_data=True, save_path='./Data', name='SHPSO', disable_tqdm=True):
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
            Particle swarm size per task (default: 50)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SHPSO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.ps = ps if ps is not None else 50
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        self.c1 = 2.05  # Cognitive coefficient
        self.c2 = 2.05  # Social coefficient
        self.w = 0.7298  # Inertia weight (constriction factor)

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

        # Historical database for each task (surrogate training data)
        hx = [decs[i].copy() for i in range(nt)]  # Historical positions
        hf = [objs[i].flatten().copy() for i in range(nt)]  # Historical fitness

        # Full evaluation history for results (MATLAB records every evaluation
        # in CE even when a point is not admitted into the database)
        res_x = [decs[i].copy() for i in range(nt)]
        res_f = [objs[i].flatten().copy() for i in range(nt)]

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
                    res_x[i] = np.vstack([res_x[i], extra_pos])
                    res_f[i] = np.concatenate([res_f[i], extra_objs.flatten()])
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

        # Best surrogate model optimum for each task (MATLAB: besty=1e200, bestp=zeros)
        bestp = [np.zeros(dims[i]) for i in range(nt)]
        besty = [1e200 for _ in range(nt)]

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
                rbf_model, ghx, ghf, spread = self._build_rbf_model(hx[i], hf[i], gs, dim)

                # Record old best
                besty_old = besty[i]
                bestp_old = bestp[i].copy()

                # ===== Upper Level: Optimize surrogate model using SL-PSO =====
                # MATLAB passes ghx (top-gs samples) to SLPSO for search bounds
                bestp_new = self._optimize_surrogate_slpso(rbf_model, ghx, dim)

                # Exact evaluate the surrogate optimum
                besty_new_arr, _ = evaluation_single(problem, bestp_new.reshape(1, -1), i)
                besty_new = besty_new_arr[0, 0]
                nfes_per_task[i] += 1
                pbar.update(1)
                res_x[i] = np.vstack([res_x[i], bestp_new.reshape(1, -1)])
                res_f[i] = np.concatenate([res_f[i], [besty_new]])

                # Update surrogate optimum (MATLAB: compare before adding to history)
                if besty_new < besty_old:
                    besty[i] = besty_new
                    bestp[i] = bestp_new.copy()
                else:
                    besty[i] = besty_old
                    bestp[i] = bestp_old.copy()

                # Update history database with the WINNER (matching MATLAB)
                if not is_duplicate(bestp[i], hx[i]):
                    hx[i] = np.vstack([hx[i], bestp[i]])
                    hf[i] = np.concatenate([hf[i], [besty[i]]])

                # Rebuild only when the worst training sample is worse than besty;
                # MATLAB reuses the spread from the pre-update training set
                if ghf[-1] > besty[i]:
                    rbf_model, _, _, _ = self._build_rbf_model(hx[i], hf[i], gs, dim, spread=spread)

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

                # ===== Prescreening: e-pbest strategy (matching MATLAB) =====
                # Predict fitness using surrogate model
                e_pred = rbf_model(pos[i]).flatten()

                # Select candidates where surrogate predicts improvement over personal best
                candidates_to_eval = []
                for idx in range(ps):
                    if e_pred[idx] < pbestval[i][idx]:
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
                    res_x[i] = np.vstack([res_x[i], candidate_pos])
                    res_f[i] = np.concatenate([res_f[i], [e_true]])

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


        pbar.close()
        runtime = time.time() - start_time

        # Convert evaluation history to staircase structure for results
        db_decs = [res_x[i].copy() for i in range(nt)]
        db_objs = [res_f[i].reshape(-1, 1).copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    # Build RBF surrogate model using best historical samples
    def _build_rbf_model(self, hx, hf, gs, dim, spread=None):

        # Sort by fitness and select best gs samples
        idx_sorted = np.argsort(hf, kind='stable')
        n_select = min(gs, len(hf))
        idx_select = idx_sorted[:n_select]

        ghx = hx[idx_select]
        ghf = hf[idx_select]

        # Calculate spread parameter (MATLAB: max pairwise dist / (D*gs)^(1/D))
        if spread is None:
            dist_matrix = cdist(ghx, ghx, metric='euclidean')
            max_dist = dist_matrix.max()
            if max_dist > 0:
                spread = max_dist / (dim * len(ghf)) ** (1.0 / dim)
            else:
                spread = 1.0

        predict = newrbe_surrogate(ghx, ghf, spread=spread)

        def rbf_model(x):
            return np.asarray(predict(x)).reshape(-1, 1)

        return rbf_model, ghx, ghf, spread

    # Optimize surrogate model using inline SLPSO (matching MATLAB exactly)
    def _optimize_surrogate_slpso(self, rbf_model, ghx, dim):
        """
        Find near-optimum of RBF surrogate using Social Learning PSO.

        Key difference from general SL_PSO: search bounds are restricted to
        [min(ghx), max(ghx)] per dimension, matching MATLAB's implementation.
        This prevents SLPSO from exploring regions where the Gaussian RBF
        extrapolates to spurious low values.
        """
        # MATLAB: lu = [min(ghx); max(ghx)]
        lb = ghx.min(axis=0)
        ub = ghx.max(axis=0)

        # SLPSO parameters (matching MATLAB)
        M = 100
        beta = 0.01
        m = M + int(np.floor(dim / 10))
        c3 = dim / M * beta
        maxgen = 50 * dim
        minerror = 1e-6

        # Learning probability: PL(i) = (1 - (i-1)/m)^log(sqrt(ceil(d/M)))
        PL = np.array([(1 - i / m) ** np.log(np.sqrt(np.ceil(dim / M))) for i in range(m)])

        # Initialize population with LHS in [lb, ub] (MATLAB: lhsdesign)
        p = lb + (ub - lb) * qmc.LatinHypercube(d=dim).random(m)
        v = np.zeros((m, dim))

        bestever = 1e200
        best_pos = np.zeros(dim)
        flag_er = 0

        for gen in range(maxgen):
            best_old = bestever

            # Evaluate fitness using surrogate model
            fitness = rbf_model(p).flatten()

            # Sort descending (worst first, best last — matching MATLAB)
            rank = np.argsort(-fitness, kind='stable')
            fitness = fitness[rank]
            p = p[rank]
            v = v[rank]

            # Best in current population (last element after descending sort)
            besty = fitness[-1]
            bestp = p[-1].copy()

            # MATLAB min([besty, bestever]) takes the new position on ties
            if besty <= bestever:
                bestever = besty
                best_pos = bestp.copy()

            # Early stopping: 20 consecutive gens with improvement <= minerror
            error = abs(best_old - bestever)
            if error <= minerror:
                flag_er += 1
            else:
                flag_er = 0
            if flag_er >= 20:
                break

            # Center position
            center = np.mean(p, axis=0)

            # Random coefficients
            randco1 = np.random.rand(m, dim)
            randco2 = np.random.rand(m, dim)
            randco3 = np.random.rand(m, dim)

            # Social learning: select demonstrators from better particles
            # MATLAB: winidx = i + ceil(rand*(m-i)), selecting from [i+1, m] (1-indexed)
            idx = np.arange(m).reshape(-1, 1)
            max_range = m - 1 - idx  # max offset for each particle
            offsets = np.ceil(np.random.rand(m, dim) * max_range).astype(int)
            winidx = idx + offsets
            winidx = np.clip(winidx, 0, m - 1)

            # Get winner positions (per-dimension indexing)
            pwin = np.zeros((m, dim))
            for j in range(dim):
                pwin[:, j] = p[winidx[:, j], j]

            # Learning mask
            lpmask_1d = np.random.rand(m) < PL
            lpmask_1d[-1] = False  # Best particle doesn't learn
            lpmask = np.tile(lpmask_1d.reshape(-1, 1), (1, dim))

            # Velocity and position update
            v1 = randco1 * v + randco2 * (pwin - p) + c3 * randco3 * (center - p)
            p1 = p + v1

            v = np.where(lpmask, v1, v)
            p = np.where(lpmask, p1, p)

            # Boundary handling: clip to [lb, ub]
            p = np.clip(p, lb, ub)

        best_pos = np.clip(best_pos, 0.0, 1.0)
        return best_pos
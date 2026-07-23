"""
Surrogate-Assisted Cooperative Swarm Optimization (SA-COSO)

This module implements SA-COSO for expensive single-objective optimization problems.

References
----------
    [1] Sun, Chaoli, et al. "Surrogate-assisted cooperative swarm optimization of high-dimensional expensive problems." IEEE Transactions on Evolutionary Computation 21.4 (2017): 644-660.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.07.23
Version: 1.1
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class SA_COSO:
    """
    Surrogate-Assisted Cooperative Swarm Optimization for expensive optimization problems.

    Two cooperating swarms share one RBF network (MATLAB newrb, at most 8 neurons)
    trained on a bounded archive of exactly evaluated solutions:

    1. A PSO swarm whose fitness is mostly estimated by a Fitness Estimation
       Strategy (FES): plain RBF predictions plus position-relation-based
       estimates from the nearest neighbor's virtual position; estimated
       particles that look better than their personal best (under both the
       estimate and the RBF prediction) are evaluated exactly.
    2. A social-learning swarm whose particles learn dimension-wise from
       better solutions among the swarm itself and random archive members;
       its best predicted particle is evaluated exactly every iteration.
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n_fes=30, n_rbf=200,
                 save_data=True, save_path='./Data', name='SA-COSO', disable_tqdm=True):
        """
        Initialize SA-COSO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: n_fes + n_rbf)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        n_fes : int, optional
            Population size of the FES-assisted PSO swarm (default: 30)
        n_rbf : int, optional
            Population size of the RBF-assisted social-learning swarm (default: 200)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SA-COSO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_fes = n_fes
        self.n_rbf = n_rbf
        self.n_initial = n_initial if n_initial is not None else (n_fes + n_rbf)
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        # PSO parameters
        self.c1 = 2.05   # cognitive coefficient
        self.c2 = 1.025  # social coefficient (own gbest)
        self.c3 = 1.025  # social coefficient (other swarm's gbest)
        self.w = 0.7298  # constriction factor

        # RBF network parameters (MATLAB newrb)
        self.max_node = 8
        self.rbf_goal = 0.1

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

        hx = [decs[i].copy() for i in range(nt)]
        hf = [objs[i].flatten().copy() for i in range(nt)]

        # Per-task algorithm state
        states = []
        for i in range(nt):
            dim = dims[i]
            n_total = n_initial_per_task[i]
            n1 = min(self.n_fes, n_total)
            n2 = n_total - n1

            pos1 = hx[i][:n1].copy()
            fit1 = hf[i][:n1].copy()

            if n2 > 0:
                pos2 = hx[i][n1:].copy()
                fit2 = hf[i][n1:].copy()
            else:
                pos2 = np.random.rand(self.n_rbf, dim)
                fit2_arr, _ = evaluation_single(problem, pos2, i)
                fit2 = fit2_arr.flatten()
                hx[i] = np.vstack([hx[i], pos2])
                hf[i] = np.concatenate([hf[i], fit2])
                nfes_per_task[i] += len(pos2)

            st = {
                'pos1': pos1, 'fit1': fit1,
                'vel1': np.random.rand(len(pos1), dim) - 0.5,
                'pbest': pos1.copy(), 'pbestval': fit1.copy(),
                'pbest_eval': np.ones(len(pos1), dtype=bool),
                'pos2': pos2, 'fit2': fit2,
                'vel2': np.random.rand(len(pos2), dim) - 0.5,
                'his_pos_1': pos1.copy(), 'his_fit_1': fit1.copy(),
                'his_pos_2': np.zeros_like(pos1), 'his_fit_2': np.zeros(len(pos1)),
                'r1': np.zeros((len(pos1), dim)), 'r2': np.zeros((len(pos1), dim)),
                'r3': np.zeros((len(pos1), dim)),
                'archive_pos': hx[i].copy(), 'archive_fit': hf[i].copy(),
                'iter': 1,
            }

            g_idx = int(np.argmin(fit1))
            st['gbest'] = pos1[g_idx].copy()
            st['gbestval'] = float(fit1[g_idx])
            r_idx = int(np.argmin(fit2))
            st['rbfgbest'] = pos2[r_idx].copy()
            st['rbfgbestval'] = float(fit2[r_idx])

            # Initial RBF network: spread is the archive bounding-box diagonal
            spread0 = self._bbox_diagonal(st['archive_pos'])
            st['spread_sum'] = spread0
            st['model'] = newrb_surrogate(st['archive_pos'], st['archive_fit'],
                                          goal=self.rbf_goal, spread=spread0,
                                          max_neurons=self.max_node)
            states.append(st)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(nfes_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                dim = dims[i]
                st = states[i]
                model = st['model']
                n1 = len(st['pos1'])
                n2 = len(st['pos2'])

                # Exact evaluations of this iteration (for archive update)
                temp_pos = []
                temp_fit = []

                def try_eval(x_row):
                    if nfes_per_task[i] >= max_nfes_per_task[i]:
                        return None
                    obj_arr, _ = evaluation_single(problem, x_row.reshape(1, -1), i)
                    val = float(obj_arr[0, 0])
                    nfes_per_task[i] += 1
                    pbar.update(1)
                    hx[i] = np.vstack([hx[i], x_row.reshape(1, -1)])
                    hf[i] = np.concatenate([hf[i], [val]])
                    temp_pos.append(x_row.copy())
                    temp_fit.append(val)
                    return val

                # ===== Learning pool for the social-learning swarm =====
                # Current swarm particles plus distinct random archive members
                # (MATLAB's index sampling never selects the last archive entry)
                n_arc = len(st['archive_fit'])
                n_picks = min(n2, max(n_arc - 1, 0))
                picks = np.random.choice(max(n_arc - 1, 1), size=n_picks, replace=False)
                pool_pos = np.vstack([st['pos2'], st['archive_pos'][picks]])
                pool_fit = np.concatenate([st['fit2'], st['archive_fit'][picks]])

                # ===== Record swarm-1 history (pre-move positions) =====
                if st['iter'] > 1:
                    st['his_pos_2'] = st['his_pos_1'].copy()
                    st['his_fit_2'] = st['his_fit_1'].copy()
                st['his_pos_1'] = st['pos1'].copy()
                st['his_fit_1'] = st['fit1'].copy()
                sorted_index = np.argsort(st['his_fit_1'], kind='stable')

                # ===== Swarm 1: PSO move (random matrices kept for FES) =====
                st['r1'] = np.random.rand(n1, dim)
                st['r2'] = np.random.rand(n1, dim)
                st['r3'] = np.random.rand(n1, dim)
                st['vel1'] = self.w * (st['vel1'] +
                                       self.c1 * st['r1'] * (st['pbest'] - st['pos1']) +
                                       self.c2 * st['r2'] * (st['gbest'] - st['pos1']) +
                                       self.c3 * st['r3'] * (st['rbfgbest'] - st['pos1']))
                st['vel1'] = np.clip(st['vel1'], -0.5, 0.5)
                st['pos1'] = np.clip(st['pos1'] + st['vel1'], 0.0, 1.0)

                # ===== Swarm 2: dimension-wise social learning =====
                for j in range(n2):
                    better_idx = np.where(pool_fit < st['fit2'][j])[0]
                    nb = len(better_idx)
                    if nb == 0:
                        continue
                    if nb == 1:
                        chosen = pool_pos[better_idx[0]].copy()
                    elif nb == 2:
                        take_first = np.random.rand(dim) > 0.5
                        chosen = np.where(take_first,
                                          pool_pos[better_idx[0]],
                                          pool_pos[better_idx[1]])
                    else:
                        picks_d = better_idx[np.random.randint(nb, size=dim)]
                        chosen = pool_pos[picks_d, np.arange(dim)]

                    st['vel2'][j] = (np.random.rand(dim) * st['vel2'][j] +
                                     np.random.rand(dim) * (chosen - st['pos2'][j]))
                    st['vel2'][j] = np.clip(st['vel2'][j], -0.5, 0.5)
                    st['pos2'][j] = np.clip(st['pos2'][j] + st['vel2'][j], 0.0, 1.0)
                    st['fit2'][j] = float(model(st['pos2'][j])[0])

                # ===== Swarm 1: fitness determination via FES =====
                fit_known = np.zeros(n1, dtype=bool)
                fit_eval = np.zeros(n1, dtype=bool)
                determined = np.zeros(n1, dtype=bool)
                via_virtual = np.zeros(n1, dtype=bool)
                evals_before = nfes_per_task[i]

                if st['iter'] == 1:
                    for p in range(n1):
                        idx = int(sorted_index[p])
                        val = try_eval(st['pos1'][idx])
                        if val is None:
                            break
                        st['fit1'][idx] = val
                        fit_known[idx] = True
                        fit_eval[idx] = True
                else:
                    dist_mat = cdist(st['pos1'], st['pos1'])
                    np.fill_diagonal(dist_mat, 1e5)
                    for p in range(n1):
                        idx = int(sorted_index[p])
                        if not fit_known[idx]:
                            st['fit1'][idx] = float(model(st['pos1'][idx])[0])
                            fit_known[idx] = True
                            fit_eval[idx] = False
                            via_virtual[idx] = False
                        determined[idx] = True

                        # Nearest-neighbor estimation via the virtual position
                        cand = np.where(dist_mat[idx] > 0)[0]
                        if len(cand) > 0:
                            p_star = int(cand[np.argmin(dist_mat[idx, cand])])
                            mi = int(sorted_index[p_star])
                            if (not fit_known[mi]) or (fit_known[mi] and not fit_eval[mi]
                                                       and not determined[mi]):
                                est = self._virtual_estimate(st, idx, mi)
                                if est is not None:
                                    if fit_known[mi]:
                                        st['fit1'][mi] = est
                                    else:
                                        st['fit1'][mi] = min(st['fit1'][mi], est)
                                    fit_known[mi] = True
                                    fit_eval[mi] = False
                                    via_virtual[mi] = True

                        # Exact evaluation when both the estimate and the RBF
                        # prediction improve on the personal best
                        if via_virtual[idx] and not fit_eval[idx]:
                            pred = float(model(st['pos1'][idx])[0])
                            if (st['fit1'][idx] < st['pbestval'][idx]
                                    and pred < st['pbestval'][idx]):
                                val = try_eval(st['pos1'][idx])
                                if val is not None:
                                    st['fit1'][idx] = val
                                    fit_eval[idx] = True

                    # Fallback: no exact evaluation happened in this pass
                    if nfes_per_task[i] == evals_before:
                        preds = np.asarray(model(st['pos1'])).flatten()
                        errs = np.abs(st['fit1'] - preds)
                        est_mask = via_virtual & ~fit_eval
                        if np.any(est_mask):
                            avg_err = errs[est_mask].mean()
                            for idx in np.where(est_mask)[0]:
                                if errs[idx] > avg_err:
                                    val = try_eval(st['pos1'][idx])
                                    if val is None:
                                        break
                                    st['fit1'][idx] = val
                                    fit_eval[idx] = True

                # ===== Swarm 1: pbest / gbest update with certification =====
                for idx in range(n1):
                    if st['fit1'][idx] < st['pbestval'][idx]:
                        st['pbest'][idx] = st['pos1'][idx].copy()
                        st['pbestval'][idx] = st['fit1'][idx]
                        st['pbest_eval'][idx] = fit_eval[idx]

                bid = int(np.argmin(st['pbestval']))
                if st['pbest_eval'][bid]:
                    if st['pbestval'][bid] < st['gbestval']:
                        st['gbest'] = st['pbest'][bid].copy()
                        st['gbestval'] = float(st['pbestval'][bid])
                else:
                    val = try_eval(st['pbest'][bid])
                    if val is not None:
                        st['pbestval'][bid] = val
                        st['pbest_eval'][bid] = True
                        if val < st['gbestval']:
                            st['gbest'] = st['pbest'][bid].copy()
                            st['gbestval'] = val

                # ===== Swarm 2: evaluate its best predicted particle =====
                rid = int(np.argmin(st['fit2']))
                val = try_eval(st['pos2'][rid])
                if val is not None:
                    st['fit2'][rid] = val
                    if val < st['rbfgbestval']:
                        st['rbfgbest'] = st['pos2'][rid].copy()
                        st['rbfgbestval'] = val

                st['iter'] += 1

                # ===== Archive update and RBF network rebuild =====
                if temp_pos:
                    self._update_archive(st, np.asarray(temp_pos), np.asarray(temp_fit), dim)

                spread1 = self._bbox_diagonal(st['archive_pos'])
                st['spread_sum'] += spread1
                spread = st['spread_sum'] / (st['iter'] + 1)
                st['model'] = newrb_surrogate(st['archive_pos'], st['archive_fit'],
                                              goal=self.rbf_goal, spread=spread,
                                              max_neurons=self.max_node)

        pbar.close()
        runtime = time.time() - start_time

        # Convert database to staircase history structure for results
        db_decs = [hx[i].copy() for i in range(nt)]
        db_objs = [hf[i].reshape(-1, 1).copy() for i in range(nt)]
        all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)

        # Build and save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     bounds=problem.bounds, save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    @staticmethod
    def _bbox_diagonal(X):
        """Diagonal length of the bounding box of X (MATLAB spread estimate)."""
        rng = np.max(X, axis=0) - np.min(X, axis=0)
        diag = float(np.sqrt(np.sum(rng ** 2)))
        return diag if diag > 0 else 1.0

    def _virtual_estimate(self, st, idx, mi):
        """
        Estimate the fitness of particle mi from the virtual position implied
        by the PSO update relation between particles idx and mi.

        Returns None when any of the twelve reference distances is zero.
        """
        phi = self.w
        r1m, r2m, r3m = st['r1'][mi], st['r2'][mi], st['r3'][mi]

        vp = (st['pos1'][idx]
              + (1 + phi - phi * self.c1 * r1m - phi * self.c2 * r2m
                 - phi * self.c3 * r3m) * st['his_pos_1'][mi]
              + phi * st['his_pos_2'][idx]
              + phi * self.c1 * r1m * st['pbest'][mi]
              + phi * self.c2 * r2m * st['gbest']
              + phi * self.c3 * r3m * st['rbfgbest'])

        refs_a = [st['pos1'][idx], st['his_pos_1'][mi], st['his_pos_2'][idx],
                  st['pbest'][mi], st['gbest'], st['rbfgbest']]
        vals_a = [st['fit1'][idx], st['his_fit_1'][mi], st['his_fit_2'][idx],
                  st['pbestval'][mi], st['gbestval'], st['rbfgbestval']]
        refs_b = [st['pos1'][mi], st['his_pos_1'][idx], st['his_pos_2'][mi],
                  st['pbest'][idx], st['gbest'], st['rbfgbest']]
        vals_b = [st['his_fit_1'][idx], st['his_fit_2'][mi], st['pbestval'][idx],
                  st['gbestval'], st['rbfgbestval']]

        d_a = np.array([np.linalg.norm(vp - r) for r in refs_a])
        d_b = np.array([np.linalg.norm(vp - r) for r in refs_b])
        if np.any(d_a == 0) or np.any(d_b == 0):
            return None

        dist_temp1 = np.sum(1.0 / d_a)
        dist_temp2 = np.sum(1.0 / d_b)
        dist_ratio = dist_temp2 / dist_temp1
        virtual_fitness = float(np.sum(np.asarray(vals_a) / d_a))
        # d_b[0] is the distance to particle mi whose fitness is being estimated
        est = d_b[0] * (virtual_fitness * dist_ratio
                        - float(np.sum(np.asarray(vals_b) / d_b[1:])))
        return est

    def _update_archive(self, st, new_pos, new_fit, dim):
        """
        Insert this iteration's exactly evaluated points into the bounded
        archive: append while below capacity; otherwise replace the archive
        member farthest from the social-learning swarm when the new point
        lies closer to it.
        """
        max_archive = self.max_node * dim + 10

        arc_to_pop = cdist(st['archive_pos'], st['pos2'])
        minarc_to_pop = arc_to_pop.min(axis=1)
        max_id = int(np.argmax(minarc_to_pop))
        max_val = minarc_to_pop[max_id]

        for k in range(len(new_fit)):
            d_arc = cdist(new_pos[k:k + 1], st['archive_pos']).flatten()
            if np.any(d_arc < 1e-4):
                continue
            if len(st['archive_fit']) < max_archive:
                st['archive_pos'] = np.vstack([st['archive_pos'], new_pos[k:k + 1]])
                st['archive_fit'] = np.concatenate([st['archive_fit'], [new_fit[k]]])
            else:
                d_pop = cdist(new_pos[k:k + 1], st['pos2']).flatten()
                min_to_pop = d_pop.min()
                if min_to_pop < max_val:
                    st['archive_pos'][max_id] = new_pos[k]
                    st['archive_fit'][max_id] = new_fit[k]
                    minarc_to_pop[max_id] = min_to_pop
                    max_id = int(np.argmax(minarc_to_pop))
                    max_val = minarc_to_pop[max_id]

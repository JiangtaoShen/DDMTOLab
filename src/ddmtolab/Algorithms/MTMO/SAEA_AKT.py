"""
Surrogate-Assisted Evolutionary Algorithm with Adaptive Knowledge Transfer (SAEA-AKT)

This module implements SAEA-AKT for expensive multitask multiobjective optimization.
Every task is solved in turn as the target task, paired with a randomly drawn source
task, by running four components in sequence: a competitive surrogate selection that
lets a Kriging model and a heterogeneous ensemble model compete on a held-out
validation split, an adaptive solution selection that transfers either convergent or
diverse solutions from the source task depending on the current optimization state, a
surrogate-assisted evolutionary optimizer seeded with the transferred population, and
an adaptive infilling criterion that reuses the same selection to pick the samples
that are truly evaluated.

References
----------
    [1] X. Wu, S. Liu, Q. Lin, K. C. Tan, and V. C. M. Leung. Evolutionary multitasking with adaptive knowledge transfer for expensive multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2537-2551.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.03
Version: 1.0
"""
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm

from ddmtolab.Algorithms.STMO.IBEA import ibea_selection
from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.Algo_Methods.bo_utils import mo_gp_build, mo_gp_predict
import warnings

warnings.filterwarnings("ignore")


class SAEA_AKT:
    """
    Surrogate-assisted evolutionary algorithm with adaptive knowledge transfer for
    expensive multitask multiobjective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'equal',
        'n_cons': '0',
        'expensive': 'True',
        'knowledge_transfer': 'True',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, n_kt=10, n_ic=5, g_max=20,
                 p1=0.5, p2=0.75, muc=20.0, mum=20.0, save_data=True, save_path='./Data',
                 name='SAEA-AKT', disable_tqdm=True):
        """
        Initialize the SAEA-AKT algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial Latin hypercube samples per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 500)
        n_kt : int, optional
            Number of transfer solutions taken from the source task (default: 10)
        n_ic : int, optional
            Number of infill solutions truly evaluated per iteration (default: 5)
        g_max : int, optional
            Number of generations run by the surrogate-assisted optimizer (default: 20)
        p1 : float, optional
            Probability of performing knowledge transfer in one iteration (default: 0.5)
        p2 : float, optional
            Probability of assigning a sample to the surrogate training split, the
            remainder forming the validation split (default: 0.75)
        muc : float, optional
            Distribution index for simulated binary crossover (SBX) (default: 20.0)
        mum : float, optional
            Distribution index for polynomial mutation (PM) (default: 20.0)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SAEA-AKT')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 500
        self.n_kt = n_kt
        self.n_ic = n_ic
        self.g_max = g_max
        self.p1 = p1
        self.p2 = p2
        self.muc = muc
        self.mum = mum
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SAEA-AKT algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        data_type = torch.float
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims

        n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initial Latin hypercube design, truly evaluated to seed each training database
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Convergence metric of the previous visit of each task; the first visit has no
        # predecessor, so it falls into the convergence-based branch of Algorithm 3
        cm_previous = [np.inf] * nt

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break
            nfes_before_pass = sum(nfes_per_task)

            for i in active_tasks:
                dim = dims[i]

                # ============ Competitive Surrogate Selection (Algorithm 2) ============
                model = self._competitive_surrogate_selection(decs[i], objs[i], data_type)

                # The convergence metric is measured once per visit and shared by the
                # knowledge transfer and the infilling criterion of this iteration
                cm = float(np.mean(np.linalg.norm(objs[i], axis=1)))
                converging = cm <= cm_previous[i]
                cm_previous[i] = cm

                # ============ Adaptive Solution Selection (Algorithm 3) ============
                kt_decs = np.empty((0, dim))
                if nt > 1 and np.random.rand() < self.p1:
                    # One other task is drawn at random as the source task
                    candidates = [j for j in range(nt) if j != i]
                    source = candidates[np.random.randint(len(candidates))]
                    # Transferred solutions live in the target task's search space
                    source_decs = align_dimensions(decs[source], dim)
                    kt_decs = self._adaptive_solution_selection(
                        decs[i], source_decs, objs[source], self.n_kt, converging
                    )

                # ============ SAEO (Algorithm 4) ============
                pop_decs, pop_objs = self._saeo(decs[i], objs[i], kt_decs, model, data_type)

                # ============ Adaptive Infilling Criterion (Algorithm 5) ============
                n_new = min(self.n_ic, max_nfes_per_task[i] - nfes_per_task[i])
                new_decs = self._adaptive_infilling_criterion(
                    decs[i], objs[i], pop_decs, pop_objs, n_new, converging
                )

                if new_decs.shape[0] > 0:
                    # Truly evaluate the infill samples and grow the training database
                    new_objs, _ = evaluation_single(problem, new_decs, i)
                    decs[i], objs[i] = vstack_groups((decs[i], new_decs), (objs[i], new_objs))
                    nfes_per_task[i] += new_decs.shape[0]
                    pbar.update(new_decs.shape[0])

            if sum(nfes_per_task) == nfes_before_pass:
                # No task could produce a new sample in a full pass; stop
                break

        pbar.close()
        runtime = time.time() - start_time

        all_decs, all_objs = build_staircase_history(decs, objs, k=self.n_ic)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)

        return results

    # -----------------------------------------------------------------------------
    # Algorithm 2: Competitive Surrogate Selection
    # -----------------------------------------------------------------------------

    def _competitive_surrogate_selection(self, db_decs, db_objs, data_type):
        """
        Train a Kriging model and an ensemble model, keep the more reliable one.

        Every sample of the training database joins the training split with
        probability ``p2`` and the validation split otherwise. Both model types are
        built on the training split and scored on the validation split with the RMSE
        of Eq. (3), which sums the per-objective root mean squared errors, so a
        single model type serves all objectives of the task.

        Parameters
        ----------
        db_decs : np.ndarray
            Decision variables of the training database, shape (n, d)
        db_objs : np.ndarray
            Objective values of the training database, shape (n, m)
        data_type : torch.dtype
            Data type used by the Kriging models

        Returns
        -------
        model : dict
            Selected surrogate, consumed by :func:`_surrogate_predict`
        """
        n = db_decs.shape[0]

        # Probability-dependent split into training and validation samples
        is_train = np.random.rand(n) < self.p2
        if is_train.sum() < 2 or (~is_train).sum() < 1:
            # Degenerate draw: fall back on a fixed 3:1 split of a random permutation
            order = np.random.permutation(n)
            n_train = int(np.clip(round(self.p2 * n), 2, n - 1)) if n > 2 else n
            is_train = np.zeros(n, dtype=bool)
            is_train[order[:n_train]] = True

        train_decs, train_objs = db_decs[is_train], db_objs[is_train]
        valid_decs, valid_objs = db_decs[~is_train], db_objs[~is_train]

        kriging = {'type': 'kriging', 'models': mo_gp_build(train_decs, train_objs, data_type)}
        ensemble = {'type': 'ensemble', 'members': _build_ensemble(train_decs, train_objs)}

        if valid_decs.shape[0] == 0:
            return kriging

        rmse_kriging = _rmse(valid_objs, _surrogate_predict(kriging, valid_decs, data_type))
        rmse_ensemble = _rmse(valid_objs, _surrogate_predict(ensemble, valid_decs, data_type))

        return kriging if rmse_kriging < rmse_ensemble else ensemble

    # -----------------------------------------------------------------------------
    # Algorithm 3: Adaptive Solution Selection
    # -----------------------------------------------------------------------------

    def _adaptive_solution_selection(self, target_decs, source_decs, source_objs, n_sel,
                                     converging):
        """
        Select solutions from a source set, favouring convergence or diversity.

        When the convergence metric of the target task did not deteriorate, the source
        set is truncated by repeatedly discarding the solution with the smallest
        I-epsilon-plus loss of Eqs. (5) and (6), i.e. the environmental selection of
        IBEA with kappa = 0.05. Otherwise the solutions lying farthest from the target
        training database in the decision space are taken.

        Parameters
        ----------
        target_decs : np.ndarray
            Decision variables of the target training database, shape (n_t, d)
        source_decs : np.ndarray
            Decision variables of the source set, already in the target space,
            shape (n_s, d)
        source_objs : np.ndarray
            Objective values of the source set, shape (n_s, m_s)
        n_sel : int
            Number of solutions to select
        converging : bool
            True when the convergence-based branch applies

        Returns
        -------
        selected_decs : np.ndarray
            Selected decision variables, shape (min(n_sel, n_s), d)
        """
        n_source = source_decs.shape[0]
        if n_sel <= 0 or n_source == 0:
            return np.empty((0, source_decs.shape[1]))
        if n_source <= n_sel:
            return source_decs.copy()

        if converging:
            # Convergence-based selection: keep the n_sel best by I-epsilon-plus loss
            index, _ = ibea_selection(source_objs, n_sel, 0.05)
        else:
            # Diversity-based selection: the solutions farthest from the target database
            distance = np.min(cdist(source_decs, target_decs), axis=1)
            index = np.argsort(-distance)[:n_sel]

        return source_decs[index].copy()

    # -----------------------------------------------------------------------------
    # Algorithm 4: Surrogate-Assisted Evolutionary Optimizer
    # -----------------------------------------------------------------------------

    def _saeo(self, db_decs, db_objs, kt_decs, model, data_type):
        """
        Evolve the nondominated solutions of the target task under the surrogate.

        The initial population combines the nondominated solutions of the training
        database with the transferred population; offspring are produced by SBX and
        polynomial mutation, scored by the surrogate, and reduced by the NSGA-II
        environmental selection.

        Parameters
        ----------
        db_decs : np.ndarray
            Decision variables of the training database, shape (n, d)
        db_objs : np.ndarray
            Objective values of the training database, shape (n, m)
        kt_decs : np.ndarray
            Transferred decision variables from the source task, shape (n_kt, d)
        model : dict
            Surrogate returned by :func:`_competitive_surrogate_selection`
        data_type : torch.dtype
            Data type used by the Kriging models

        Returns
        -------
        pop_decs : np.ndarray
            Final population, shape (n_pop, d)
        pop_objs : np.ndarray
            Surrogate-approximated objectives of the final population, shape (n_pop, m)
        """
        dim = db_decs.shape[1]

        # Nondominated solutions of the target task seed the search
        front_no, _ = nd_sort(db_objs, db_objs.shape[0])
        pop_decs = np.vstack([db_decs[front_no == 1], kt_decs]) if kt_decs.shape[0] > 0 \
            else db_decs[front_no == 1].copy()

        # SBX needs at least one pair of parents
        if pop_decs.shape[0] < 2:
            extra = np.random.permutation(db_decs.shape[0])[:2 - pop_decs.shape[0]]
            pop_decs = np.vstack([pop_decs, db_decs[extra]]) if pop_decs.shape[0] > 0 \
                else db_decs[extra].copy()
        if pop_decs.shape[0] < 2:
            pop_decs = np.vstack([pop_decs, np.random.rand(2 - pop_decs.shape[0], dim)])

        n_pop = pop_decs.shape[0]
        pop_objs = _surrogate_predict(model, pop_decs, data_type)

        for _ in range(self.g_max):
            off_decs = ga_generation(pop_decs, self.muc, self.mum)
            off_objs = _surrogate_predict(model, off_decs, data_type)

            merged_decs = np.vstack([pop_decs, off_decs])
            merged_objs = np.vstack([pop_objs, off_objs])

            # NSGA-II environmental selection: front number, then crowding distance
            rank, _, _ = nsga2_sort(merged_objs)
            index = np.argsort(rank)[:n_pop]
            pop_decs, pop_objs = merged_decs[index], merged_objs[index]

        return pop_decs, pop_objs

    # -----------------------------------------------------------------------------
    # Algorithm 5: Adaptive Infilling Criterion
    # -----------------------------------------------------------------------------

    def _adaptive_infilling_criterion(self, db_decs, db_objs, pop_decs, pop_objs, n_new,
                                      converging):
        """
        Pick the samples of the final population that are truly evaluated.

        Candidates are the members of the final population that are nondominated with
        respect to the training database, and the adaptive solution selection of
        Algorithm 3 reduces them to ``n_new`` infill samples. Duplicates of already
        evaluated solutions are dropped and the shortfall is refilled so that the
        evaluation budget is spent exactly.

        Parameters
        ----------
        db_decs : np.ndarray
            Decision variables of the training database, shape (n, d)
        db_objs : np.ndarray
            Objective values of the training database, shape (n, m)
        pop_decs : np.ndarray
            Final population of the surrogate-assisted optimizer, shape (n_pop, d)
        pop_objs : np.ndarray
            Surrogate-approximated objectives of the final population, shape (n_pop, m)
        n_new : int
            Number of infill samples to return
        converging : bool
            True when the convergence-based branch of Algorithm 3 applies

        Returns
        -------
        new_decs : np.ndarray
            Infill decision variables, shape (n_new, d)
        """
        dim = db_decs.shape[1]
        if n_new <= 0:
            return np.empty((0, dim))

        # Candidates nondominated with the truly evaluated samples of the target task
        merged = np.vstack([pop_objs, db_objs])
        front_no, _ = nd_sort(merged, merged.shape[0])
        is_candidate = front_no[:pop_objs.shape[0]] == 1
        if not np.any(is_candidate):
            # The surrogate found nothing better; the whole population stays eligible
            is_candidate = np.ones(pop_objs.shape[0], dtype=bool)

        new_decs = self._adaptive_solution_selection(
            db_decs, pop_decs[is_candidate], pop_objs[is_candidate], n_new, converging
        )
        new_decs = remove_duplicates(new_decs, db_decs)

        if new_decs.shape[0] < n_new:
            # Refill from the rest of the population, then from fresh random samples
            spare = remove_duplicates(pop_decs, db_decs)
            spare = remove_duplicates(np.vstack([new_decs, spare]))[new_decs.shape[0]:]
            new_decs = np.vstack([new_decs, spare[:n_new - new_decs.shape[0]]])
        if new_decs.shape[0] < n_new:
            new_decs = np.vstack([new_decs, np.random.rand(n_new - new_decs.shape[0], dim)])

        return new_decs[:n_new]


# =============================================================================
# Surrogate models
# =============================================================================

def _rmse(real_objs, pred_objs):
    """
    Root mean squared error of Eq. (3), summed over the objectives.

    Parameters
    ----------
    real_objs : np.ndarray
        Real objective values of the validation samples, shape (n, m)
    pred_objs : np.ndarray
        Predicted objective values of the validation samples, shape (n, m)

    Returns
    -------
    rmse : float
        Sum over the objectives of the per-objective root mean squared error
    """
    return float(np.sum(np.sqrt(np.mean((real_objs - pred_objs) ** 2, axis=0))))


def _surrogate_predict(model, decs, data_type):
    """
    Approximate the objective values of a set of solutions.

    Parameters
    ----------
    model : dict
        Surrogate returned by ``SAEA_AKT._competitive_surrogate_selection``
    decs : np.ndarray
        Decision variables to approximate, shape (n, d)
    data_type : torch.dtype
        Data type used by the Kriging models

    Returns
    -------
    objs : np.ndarray
        Approximated objective values, shape (n, m)
    """
    if model['type'] == 'kriging':
        return mo_gp_predict(model['models'], decs, data_type)
    return np.column_stack([predict(decs) for predict in model['members']])


def _build_ensemble(decs, objs):
    """
    Build one heterogeneous ensemble model per objective.

    Each ensemble averages, with the equal weights of the reference, an RBF network
    fitted by least squares, an RBF network fitted by backpropagation and a
    least-squares support vector machine.

    Parameters
    ----------
    decs : np.ndarray
        Training decision variables, shape (n, d)
    objs : np.ndarray
        Training objective values, shape (n, m)

    Returns
    -------
    members : List[callable]
        One ``predict(decs) -> np.ndarray`` callable per objective
    """
    spread = _gaussian_spread(decs)
    members = []
    for j in range(objs.shape[1]):
        values = objs[:, j]
        rbf_ls = newrbe_surrogate(decs, values)
        rbf_bp = _rbf_bp_surrogate(decs, values, spread)
        svm_ls = _ls_svm_surrogate(decs, values, spread)

        def predict(x, _members=(rbf_ls, rbf_bp, svm_ls)):
            return np.mean([m(x) for m in _members], axis=0)

        members.append(predict)
    return members


def _gaussian_spread(decs):
    """
    Width shared by the Gaussian kernels of the ensemble members.

    Parameters
    ----------
    decs : np.ndarray
        Training decision variables, shape (n, d)

    Returns
    -------
    spread : float
        Median nonzero pairwise distance, or 1.0 when all samples coincide
    """
    distance = cdist(decs, decs)
    positive = distance[distance > 0]
    return float(np.median(positive)) if positive.size > 0 else 1.0


def _rbf_bp_surrogate(decs, values, spread, n_epochs=200, learning_rate=0.1):
    """
    Gaussian RBF network whose output layer is trained by backpropagation.

    Centers are drawn from the training samples and the output weights and bias are
    fitted by full-batch gradient descent with momentum on standardized targets,
    which makes this member differ from the least-squares RBF of the same ensemble.

    Parameters
    ----------
    decs : np.ndarray
        Training decision variables, shape (n, d)
    values : np.ndarray
        Training values of one objective, shape (n,)
    spread : float
        Width of the Gaussian basis functions
    n_epochs : int, optional
        Number of gradient descent epochs (default: 200)
    learning_rate : float, optional
        Gradient descent step size (default: 0.1)

    Returns
    -------
    predict : callable
        ``predict(x)`` accepts an array of shape (nq, d) and returns shape (nq,)
    """
    n = decs.shape[0]
    n_centers = int(min(n, max(2, 2 * np.ceil(np.sqrt(n)))))
    centers = decs[np.random.permutation(n)[:n_centers]]

    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    if std_value == 0:
        std_value = 1.0
    targets = (values - mean_value) / std_value

    activation = np.exp(-(cdist(decs, centers) / spread) ** 2)
    weights = np.zeros(n_centers)
    bias = 0.0
    velocity_w = np.zeros(n_centers)
    velocity_b = 0.0

    for _ in range(n_epochs):
        residual = activation @ weights + bias - targets
        grad_w = 2.0 * (activation.T @ residual) / n
        grad_b = 2.0 * float(np.mean(residual))
        velocity_w = 0.9 * velocity_w - learning_rate * grad_w
        velocity_b = 0.9 * velocity_b - learning_rate * grad_b
        weights = weights + velocity_w
        bias = bias + velocity_b

    def predict(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        basis = np.exp(-(cdist(x, centers) / spread) ** 2)
        return (basis @ weights + bias) * std_value + mean_value

    return predict


def _ls_svm_surrogate(decs, values, spread, alpha=1e-3):
    """
    Least-squares support vector machine, i.e. kernel ridge regression.

    Parameters
    ----------
    decs : np.ndarray
        Training decision variables, shape (n, d)
    values : np.ndarray
        Training values of one objective, shape (n,)
    spread : float
        Width of the Gaussian kernel
    alpha : float, optional
        Ridge regularization strength (default: 1e-3)

    Returns
    -------
    predict : callable
        ``predict(x)`` accepts an array of shape (nq, d) and returns shape (nq,)
    """
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    if std_value == 0:
        std_value = 1.0

    model = KernelRidge(kernel='rbf', alpha=alpha, gamma=1.0 / (2.0 * spread ** 2))
    model.fit(decs, (values - mean_value) / std_value)

    def predict(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return model.predict(x) * std_value + mean_value

    return predict

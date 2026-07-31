"""
Pairwise Comparison Surrogate-Assisted Evolutionary Algorithm (PC-SAEA)

This module implements PC-SAEA for computationally expensive multi-objective optimization.
It uses a Probabilistic Neural Network (PNN) based pairwise comparison surrogate that
classifies whether one solution is better than another, combined with adaptive selection
strategies based on surrogate reliability.

References
----------
    [1] Y. Tian, J. Hu, C. He, H. Ma, L. Zhang, and X. Zhang. A pairwise comparison based surrogate-assisted evolutionary algorithm for expensive multi-objective optimization. Swarm and Evolutionary Computation, 2023, 80: 101323.

Notes
-----
Author: Jiangtao Shen
Date: 2026.02.17
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class PCSAEA:
    """
    Pairwise Comparison Surrogate-Assisted Evolutionary Algorithm for expensive
    multi-objective optimization using PNN-based pairwise comparison surrogates.
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n_initial=None, max_nfes=None, n=100,
                 delta=0.8, gmax=3000, spread=0.1925,
                 save_data=True, save_path='./Data', name='PC-SAEA', disable_tqdm=True):
        """
        Initialize PC-SAEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: max(11*dim-1, n))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        n : int or List[int], optional
            Population size per task (default: 100)
        delta : float, optional
            Reliability threshold for surrogate (default: 0.8)
        gmax : int, optional
            Number of solutions evaluated by the surrogate model (default: 3000)
        spread : float, optional
            PNN RBF spread parameter (default: 0.1925)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'PC-SAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n = n
        self.delta = delta
        self.gmax = gmax
        self.spread = spread
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the PC-SAEA algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n_objs = problem.n_objs

        n_per_task = par_list(self.n, nt)
        if self.n_initial is None:
            # N = max(11*D-1, Problem.N)
            n_initial_per_task = [max(11 * dims[i] - 1, n_per_task[i]) for i in range(nt)]
        else:
            n_initial_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize with LHS
        decs = initialization(problem, n_initial_per_task, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # The working population starts as the *complete* initial design; it is
        # only trimmed to Problem.N by the environmental selection performed
        # after the first infill (as in PCSAEA.m).
        pop_decs = [d.copy() for d in decs]
        pop_objs = [o.copy() for o in objs]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                D = dims[i]
                N = n_per_task[i]
                rate = nfes_per_task[i] / max_nfes_per_task[i]

                # ===== Step 1: fitness and balanced training sample set =====
                train_input, train_output, Pa, Pmid = _cal_fitness_pc(
                    pop_objs[i], pop_decs[i], rate
                )

                if train_input.shape[0] < 4 or Pa.shape[0] < 1:
                    continue

                # ===== Step 2: split train/test (3/4 - 1/4, stratified) =====
                tr_in, tr_out, te_in, te_out = _data_process(train_input, train_output)

                if tr_in.shape[0] < 2:
                    continue

                # ===== Step 3: train the PNN pairwise comparison model =====
                pnn = _PNN(self.spread)
                pnn.train(tr_in, D)

                # ===== Step 4: error rate / reliability measurement =====
                # The reliability is measured against the *mid* reference set.
                error1, error2 = _assess_reliability(pnn, te_in, te_out, D, Pmid)

                # ===== Step 5: surrogate-assisted selection =====
                candidates = _surrogate_assisted_selection(
                    pnn, error1, error2, pop_decs[i], self.gmax, Pa, D, 0, self.delta
                )

                if candidates is None or candidates.shape[0] == 0:
                    continue

                # Remove duplicates and respect the remaining budget
                candidates = remove_duplicates(candidates, decs[i])
                candidates = candidates[:max_nfes_per_task[i] - nfes_per_task[i]]
                if candidates.shape[0] == 0:
                    continue

                # ===== Step 6: expensive evaluation and archive update =====
                cand_objs, _ = evaluation_single(problem, candidates, i)

                decs[i] = np.vstack([decs[i], candidates])
                objs[i] = np.vstack([objs[i], cand_objs])

                # Environmental selection is applied to the whole archive
                # (Population = EnvironmentalSelection(Arc,Problem.N))
                sel_idx = _env_selection(objs[i], N)
                pop_decs[i] = decs[i][sel_idx]
                pop_objs[i] = objs[i][sel_idx]

                nfes_per_task[i] += candidates.shape[0]
                pbar.update(candidates.shape[0])

        pbar.close()
        runtime = time.time() - start_time

        # The number of infilled solutions varies from iteration to iteration
        # (1 for the unreliable branch, floor(|Pa|/2) otherwise), so the history
        # is built with a unit step to keep the row counts exact.
        all_decs, all_objs = build_staircase_history(decs, objs, k=1)
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                                     max_nfes=nfes_per_task, bounds=problem.bounds,
                                     save_path=self.save_path, filename=self.name,
                                     save_data=self.save_data)
        return results


# =============================================================================
# Probabilistic Neural Network for Pairwise Comparison
# =============================================================================

class _PNN:
    """
    Probabilistic Neural Network for pairwise comparison classification.

    Port of ``RBFNNPC`` (PC-SAEA), which wraps MATLAB's ``newpnn``.  ``newpnn``
    stores every training pattern as an RBF centre with bias
    ``b = sqrt(-log(0.5))/spread`` and classifies a query by
    ``argmax_c sum_{q in c} exp(-(b*||x-x_q||)^2)`` (a ``compet`` layer over the
    class-indicator matrix), i.e. the class scores are *sums*, not means.
    """

    def __init__(self, spread=0.1925):
        self.spread = spread
        self.bias = np.sqrt(-np.log(0.5)) / spread
        self.train_X = None
        self.class_masks = None
        self.classes = np.array([1, 2])

    def train(self, train_input, D):
        """
        Train the PNN on all ordered pairs of the training solutions.

        Parameters
        ----------
        train_input : np.ndarray
            Training rows, shape (N, D+1); the last column holds the fitness
            value used to label the pairs.
        D : int
            Number of decision variables
        """
        N = train_input.shape[0]
        decs = train_input[:, :D]
        # RBFNNPC.train labels a pair by comparing the *fitness* column, which
        # is stored as the last column of `Input` (`FrontNo = X(:,end)`), not by
        # comparing the coarse 1/2 class labels.
        fit = train_input[:, -1]

        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        ii = ii.ravel()
        jj = jj.ravel()
        keep = ii != jj  # the diagonal (x,x) pairs are deleted
        ii, jj = ii[keep], jj[keep]

        self.train_X = np.hstack([decs[ii], decs[jj]])
        labels = np.where(fit[ii] < fit[jj], 2, 1)
        self.class_masks = [labels == c for c in self.classes]

    def _predict_raw(self, X):
        """
        Classify pairs with the PNN.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (N, 2*D)

        Returns
        -------
        predictions : np.ndarray
            Predicted classes (1 or 2), shape (N,)
        """
        d = cdist(X, self.train_X)
        e = -(self.bias * d) ** 2
        # Shifting all exponents by the row maximum scales every class score by
        # the same positive constant, so the arg-max (the `compet` layer) is
        # unchanged while the sums stay representable for large 2*D.
        e = e - np.max(e, axis=1, keepdims=True)
        act = np.exp(e)

        scores = np.column_stack([
            act[:, mask].sum(axis=1) if np.any(mask) else np.zeros(X.shape[0])
            for mask in self.class_masks
        ])
        # `compet` resolves ties in favour of the first (smallest) class.
        return self.classes[np.argmax(scores, axis=1)]

    def lastpredict(self, X, D, preference, flag):
        """
        Compare candidates against a cycling reference set (``lastpredict``).

        Parameters
        ----------
        X : np.ndarray
            Candidate rows, shape (N, >=D)
        D : int
            Number of decision variables
        preference : np.ndarray
            Reference solutions, shape (K, >=D)
        flag : int
            0 -> return the forward/reverse difference in {-1,0,1};
            1 -> return the forward class in {1,2} with conflicts marked 1.5.

        Returns
        -------
        labels : np.ndarray, shape (N,)
        """
        N = X.shape[0]
        n_pref = preference.shape[0]
        if n_pref == 0:
            return np.zeros(N)

        decs = X[:, :D]
        # MATLAB: Pref(i,:) = Preference(mod(i+numberP,numberP)+1,:) for i=1..N,
        # i.e. the 0-based reference index of the k-th candidate is (k+1)%numberP.
        pref_idx = (np.arange(N) + 1) % n_pref
        pref = preference[pref_idx, :D]

        # Forward prediction: [candidate, reference]
        Y = self._predict_raw(np.hstack([decs, pref])).astype(float)
        # Reverse prediction: [reference, candidate]
        y = self._predict_raw(np.hstack([pref, decs])).astype(float)

        result = Y - y
        if flag == 0:
            return result
        Y[result == 0] = 1.5
        return Y


# =============================================================================
# Fitness Calculation
# =============================================================================

def _cal_fitness_pc(pop_objs, pop_decs, rate):
    """
    Compute the convergence/diversity fitness and build the training set.

    Port of ``CalFitnessPC``.  ``Fitness = rate*R + (1-rate)*D`` where ``R`` is
    the min-max normalized SPEA2 raw fitness and ``D`` is the density term.

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives, shape (N, M)
    pop_decs : np.ndarray
        Decisions, shape (N, D)
    rate : float
        Progress ratio FE/maxFE

    Returns
    -------
    train_input : np.ndarray
        Training rows ``[decs, fitness]``, shape (ceil(N/2), D+1)
    train_output : np.ndarray
        Class labels (2 = good, 1 = bad), shape (ceil(N/2),)
    Pa : np.ndarray
        Preference set, the best ceil(N/4) rows of ``train_input``
    Pmid : np.ndarray
        Mid rows of ``train_input`` straddling the good/bad boundary
    """
    N, M = pop_objs.shape
    D = pop_decs.shape[1]

    if N < 4:
        return np.zeros((0, D + 1)), np.zeros(0), pop_decs, pop_decs

    # Normalize objectives
    zmin = np.min(pop_objs, axis=0)
    zmax = np.max(pop_objs, axis=0)
    obj_range = zmax - zmin
    obj_range[obj_range == 0] = 1.0
    norm_objs = (pop_objs - zmin) / obj_range

    # --- Shift-based density estimation (SDE) ---
    sde = np.zeros(N)
    k_sde = int(np.floor(np.sqrt(N)))
    for p in range(N):
        shifted = np.maximum(norm_objs, norm_objs[p])
        dist = np.sqrt(np.sum((norm_objs[p] - shifted) ** 2, axis=1))
        dist_sorted = np.sort(dist)
        # Distance(index(floor(sqrt(N))+1)): the distance to itself (0) sits at
        # position 1, so this is the floor(sqrt(N))-th nearest neighbour.
        dk = dist_sorted[min(k_sde, N - 1)]
        sde[p] = 2.0 / (dk + 2.0)

    # --- Dominance relation (SPEA2 strength / raw fitness) ---
    dominate = np.zeros((N, N), dtype=bool)
    for p in range(N):
        for q in range(p + 1, N):
            any_less = np.any(norm_objs[p] < norm_objs[q])
            any_greater = np.any(norm_objs[p] > norm_objs[q])
            if any_less and not any_greater:
                dominate[p, q] = True
            elif any_greater and not any_less:
                dominate[q, p] = True

    S = np.sum(dominate, axis=1).astype(float)
    R = np.zeros(N)
    for p in range(N):
        R[p] = np.sum(S[dominate[:, p]])

    # --- Density term ---
    # CalFitnessPC applies pdist2(...,'cosine') to the N-by-1 SDE column.  The
    # cosine distance between two positive scalars is identically 0, therefore
    # every off-diagonal entry vanishes and D collapses to the constant
    # 1/(0+2) = 0.5 for every solution.  The computation is reproduced verbatim
    # instead of being "repaired" so that the selection order matches the
    # reference implementation (which is driven by R alone).
    cos_dist = np.nan_to_num(cdist(sde.reshape(-1, 1), sde.reshape(-1, 1), metric='cosine'))
    np.fill_diagonal(cos_dist, np.inf)
    cos_dist = np.sort(cos_dist, axis=1)
    D_density = 1.0 / (cos_dist[:, min(k_sde - 1, N - 1)] + 2.0)

    # --- Combined fitness (lower is better) ---
    R_min, R_max = R.min(), R.max()
    if R_max - R_min > 0:
        R = (R - R_min) / (R_max - R_min)
    else:
        R = np.zeros(N)
    fitness = rate * R + (1.0 - rate) * D_density

    # --- Balanced training sample set ---
    order = np.argsort(fitness, kind='stable')
    n_good = int(np.ceil(N / 4))
    n_bad = int(np.ceil(N / 2)) - n_good

    good_idx = order[:n_good]
    bad_idx = order[N - n_bad:]

    sel_idx = np.concatenate([good_idx, bad_idx])
    train_input = np.hstack([pop_decs[sel_idx], fitness[sel_idx].reshape(-1, 1)])
    train_output = np.concatenate([np.full(n_good, 2.0), np.full(n_bad, 1.0)])

    Pa = train_input[:n_good]
    half = int(np.floor(N / 8))
    lo = max(0, n_good - half - 1)
    hi = min(train_input.shape[0], n_good + half)
    Pmid = train_input[lo:hi]
    if Pmid.shape[0] == 0:
        Pmid = Pa

    return train_input, train_output, Pa, Pmid


# =============================================================================
# Data Processing
# =============================================================================

def _data_process(train_input, train_output):
    """
    Stratified train/test split (3/4 - 1/4), port of ``DataProcess``.

    Parameters
    ----------
    train_input : np.ndarray
        Rows, shape (N, D+1)
    train_output : np.ndarray
        Labels, shape (N,)

    Returns
    -------
    tr_in, tr_out, te_in, te_out : np.ndarray
    """
    idx_good = np.where(train_output > 1)[0]
    idx_bad = np.where(train_output <= 1)[0]

    n_train_good = int(np.ceil(0.75 * len(idx_good)))
    n_train_bad = int(np.ceil(0.75 * len(idx_bad)))

    perm_good = np.random.permutation(len(idx_good))
    perm_bad = np.random.permutation(len(idx_bad))

    train_idx = np.concatenate([idx_good[perm_good[:n_train_good]],
                                idx_bad[perm_bad[:n_train_bad]]])
    test_idx = np.setdiff1d(np.arange(train_input.shape[0]), train_idx)

    tr_in = train_input[train_idx]
    tr_out = train_output[train_idx]

    if len(test_idx) > 0:
        te_in = train_input[test_idx]
        te_out = train_output[test_idx]
    else:
        te_in = tr_in
        te_out = tr_out

    return tr_in, tr_out, te_in, te_out


# =============================================================================
# Reliability Assessment
# =============================================================================

def _assess_reliability(pnn, te_in, te_out, D, preference):
    """
    Measure the surrogate error rates on the held-out set.

    ``TestPre = net.lastpredict(TestIn,D,Pmid,1)`` returns the forward class in
    {1,2} with conflicting (forward == reverse) predictions marked 1.5.  Both
    rates are normalized by the *total* number of test points, so
    ``Error1 + Error2 <= 1``.

    Parameters
    ----------
    pnn : _PNN
        Trained model
    te_in : np.ndarray
        Test rows, shape (N_test, D+1)
    te_out : np.ndarray
        Test class labels, shape (N_test,)
    D : int
        Number of decision variables
    preference : np.ndarray
        Reference set (``Pmid``)

    Returns
    -------
    error1 : float
        Fraction of test points classified in agreement with their label
    error2 : float
        Fraction of test points classified in disagreement with their label
    """
    n_test = te_in.shape[0]
    if n_test == 0:
        return 1.0, 1.0

    labels = pnn.lastpredict(te_in, D, preference, flag=1)
    valid = labels != 1.5
    error1 = float(np.sum(te_out[valid] == labels[valid])) / n_test
    error2 = float(np.sum(te_out[valid] != labels[valid])) / n_test
    return error1, error2


# =============================================================================
# Surrogate-Assisted Selection
# =============================================================================

def _surrogate_assisted_selection(pnn, error1, error2, input_decs, wmax, Pa, D, flag, delta):
    """
    Select promising solutions with the pairwise-comparison surrogate.

    Port of ``SurrogateAssistedSelectionPC``.  ``Input`` is the current
    population, which is mated with the preference set ``Pa``.

    Parameters
    ----------
    pnn : _PNN
        Trained model
    error1, error2 : float
        Error rates from :func:`_assess_reliability`
    input_decs : np.ndarray
        Current population decisions, shape (N, D)
    wmax : int
        Number of solutions evaluated by the surrogate model (``gmax``)
    Pa : np.ndarray
        Preference set, shape (lnum, D+1)
    D : int
        Number of decision variables
    flag : int
        ``lastpredict`` flag (0 in PCSAEA.m)
    delta : float
        Reliability threshold

    Returns
    -------
    candidates : np.ndarray or None
    """
    Pa = Pa[:, :D]
    lnum = Pa.shape[0]
    if lnum == 0:
        return None

    candidates = ga_generation(np.vstack([input_decs[:, :D], Pa]), muc=15, mum=5)
    labels = pnn.lastpredict(candidates, D, Pa, flag)

    wmax = max(1, int(wmax // lnum))
    good_next = np.zeros((wmax, D))
    good_label = np.zeros(wmax)

    threshold = 1 - delta

    if error1 < threshold:
        descending = True
    elif error2 < threshold:
        descending = False
    else:
        # The surrogate is not reliable enough: a single random offspring is
        # re-evaluated (Next = Next(randi(end),:)).
        idx = np.random.randint(0, candidates.shape[0])
        return candidates[idx:idx + 1, :D]

    for it in range(wmax):
        order = np.argsort(-labels, kind='stable') if descending else np.argsort(labels, kind='stable')
        good_next[it] = candidates[order[0], :D]
        good_label[it] = labels[order[0]]

        n_sel = min(lnum, order.shape[0])
        parents = np.vstack([candidates[order[:n_sel], :D], Pa])
        np.random.shuffle(parents)
        candidates = ga_generation(parents, muc=15, mum=5)
        labels = pnn.lastpredict(candidates, D, Pa, flag)

    n_keep = int(np.floor(lnum / 2))
    if descending:
        confident = good_label >= 0.95
        if np.sum(confident) == 0 or np.sum(confident) > n_keep:
            top = np.argsort(-good_label, kind='stable')[:max(n_keep, 1)]
            return good_next[top]
        return good_next[confident]
    else:
        confident = good_label <= -0.95
        if np.sum(confident) == 0 or np.sum(confident) > n_keep:
            top = np.argsort(good_label, kind='stable')[:max(n_keep, 1)]
            return good_next[top]
        return good_next[confident]


# =============================================================================
# Environmental Selection (SPEA2-based)
# =============================================================================

def _env_selection(pop_objs, N):
    """
    SPEA2 environmental selection (``EnvironmentalSelection`` / ``ESCalFitness``).

    Parameters
    ----------
    pop_objs : np.ndarray
        Objectives, shape (n_total, M)
    N : int
        Target size

    Returns
    -------
    selected : np.ndarray
        Ascending indices of the selected solutions
    """
    n_total = pop_objs.shape[0]
    if n_total <= N:
        return np.arange(n_total)

    fitness = spea2_fitness(pop_objs)

    next_mask = fitness < 1.0
    n_selected = int(np.sum(next_mask))

    if n_selected < N:
        selected = np.sort(np.argsort(fitness, kind='stable')[:N])
    elif n_selected > N:
        candidates = np.where(next_mask)[0]
        # Truncation() compares the sorted neighbour-distance vectors
        # lexicographically (sortrows), i.e. spea2_truncation.
        kept = spea2_truncation(pop_objs[candidates], N)
        selected = candidates[kept]
    else:
        selected = np.where(next_mask)[0]

    return selected

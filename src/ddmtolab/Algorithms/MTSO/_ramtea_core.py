"""Faithful Python reimplementation of the MATLAB RAMTEA family (RAMTEA / RAMTEA-NT /
RAMTEA-BT), aligned line-by-line with the original MATLAB source:
  F:\\...\\RTMATEA\\{RAMTEA, RAMTEA-NT, RAMTEA-BT}\\*.m

Key fidelity points (vs the earlier, divergent DDMTOLab RAMTEA.py):
  * Search in the unified [0,1] space (DDMTOLab's evaluation_single denormalizes internally).
  * Task similarity S = max(0, Spearman rank correlation) of the two tasks' objectives on
    the shared initial samples, computed ONCE (MATLAB lines 61-67).
  * RBF surrogate = Multiquadric (c=1) WITH the linear polynomial tail (rbf_build default).
  * RBF inner search = a hand-written GA: Reproduction = polynomial mutation (eta=5,
    pm=1/D) of the parents + uniform variable swap + clip to [0,1]; the SBX crossover in
    the MATLAB Reproduction.m is dead code (overwritten by the parent mutation), so it is
    intentionally NOT applied here.
  * New-sample selection / transfer:
      - RAMTEA (mode='adaptive'): if rand()<S, inject every task's surrogate-best into
        every task (each task evaluates all tasks' bests); else each task evaluates only
        its own best.
      - RAMTEA-NT (mode='none'): always own-best only (no transfer).
      - RAMTEA-BT (mode='both'): always inject all tasks' bests (unconditional transfer).
  * wmax (RBF inner-search generations): RAMTEA=50, RAMTEA-NT=50, RAMTEA-BT=30.
  * Nin (kept after each inner generation) = 50; NI (initial) = 100.
"""
import time
import numpy as np
from scipy.stats import spearmanr

from ddmtolab.Methods.Algo_Methods.algo_utils import (
    evaluation_single, build_staircase_history, build_save_results, par_list,
)


# ----------------------------------------------------------------------------- LHS
def lhs_unit(n, d):
    """Latin Hypercube Sampling in [0,1]^d (UniformPoint(...,'Latin') analogue)."""
    out = np.zeros((n, d))
    for j in range(d):
        perm = np.random.permutation(n)
        out[:, j] = (perm + np.random.rand(n)) / n
    return out


# ----------------------------------------------------------------- Multiquadric RBF
def mq_rbf_build(X, Y, c=1.0):
    """Multiquadric RBF with linear polynomial tail (rbf_build.m, bf_type='MQ', poly=1)."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)   # n x n
    Phi = np.sqrt(sq + c ** 2)
    P = np.hstack([np.ones((n, 1)), X])                          # n x (d+1)
    A = np.zeros((n + d + 1, n + d + 1))
    A[:n, :n] = Phi
    A[:n, n:] = P
    A[n:, :n] = P.T
    rhs = np.vstack([Y, np.zeros((d + 1, 1))])
    coefs = np.linalg.lstsq(A, rhs, rcond=None)[0]              # (n+d+1) x 1
    return {'X': X, 'c': c, 'coefs': coefs, 'n': n, 'd': d}


def mq_rbf_predict(model, Xq):
    """Predict with MQ-RBF (+poly). Matches rbf_predict.m: Yq = coefs' * [dist; 1; Xq']."""
    X = model['X']; c = model['c']; coefs = model['coefs']
    Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float64))
    sq = np.sum((Xq[:, None, :] - X[None, :, :]) ** 2, axis=2)   # nq x n
    Phi = np.sqrt(sq + c ** 2)
    nq = Xq.shape[0]
    Pq = np.hstack([Phi, np.ones((nq, 1)), Xq])                 # nq x (n+1+d)
    return (Pq @ coefs).flatten()


# ------------------------------------------------------------ GA operators (MToP)
def ga_mutation(dec, eta=5.0):
    """Polynomial mutation (GA_Mutation.m), pm=1/D, distribution index eta. No clipping."""
    dec = dec.copy()
    D = dec.shape[0]
    pm = 1.0 / D
    for k in range(D):
        if np.random.rand() < pm:
            u = np.random.rand()
            if u <= 0.5:
                delta = (2 * u + (1 - 2 * u) * (1 - dec[k]) ** (eta + 1)) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * dec[k] ** (eta + 1)) ** (1 / (eta + 1))
            dec[k] = dec[k] + delta
    return dec


def reproduction(PopDec, eta=5.0):
    """Reproduction.m: per random pair, mutate BOTH parents (the SBX result is overwritten
    in the MATLAB source, i.e. dead code), uniformly swap variables, clip to [0,1]."""
    N, D = PopDec.shape
    order = np.random.permutation(N)
    half = N // 2                       # fix(N/2)
    npairs = int(np.ceil(N / 2))
    Off = np.empty((2 * npairs, D))
    cnt = 0
    for i in range(npairs):
        p1 = order[i]
        p2 = order[i + half]
        o1 = ga_mutation(PopDec[p1], eta)
        o2 = ga_mutation(PopDec[p2], eta)
        swap = np.random.rand(D) >= 0.5
        tmp = o2[swap].copy()
        o2[swap] = o1[swap]
        o1[swap] = tmp
        Off[cnt] = np.clip(o1, 0.0, 1.0)
        Off[cnt + 1] = np.clip(o2, 0.0, 1.0)
        cnt += 2
    return Off[:cnt]


# --------------------------------------------------------------------- similarity
def similarity_S(objs_list):
    """S = max(0, Spearman) for 2 tasks (MATLAB); mean of pairwise max(0,Spearman) if >2."""
    nt = len(objs_list)
    if nt == 2:
        rho, _ = spearmanr(objs_list[0], objs_list[1])
        if np.isnan(rho):
            rho = 0.0
        return max(0.0, rho)
    vals = []
    for i in range(nt):
        for j in range(i + 1, nt):
            rho, _ = spearmanr(objs_list[i], objs_list[j])
            vals.append(0.0 if np.isnan(rho) else max(0.0, rho))
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------- core driver
def ramtea_core(problem, n_initial, max_nfes, wmax, mode,
                Nin=50, eta=5.0, save_data=True, save_path='./Data',
                name='RAMTEA', disable_tqdm=True):
    """Run the RAMTEA family. mode in {'adaptive','none','both'}."""
    start = time.time()
    nt = problem.n_tasks
    dims = problem.dims
    D = int(np.max(dims))
    NI = n_initial
    maxF = par_list(max_nfes, nt)

    # ---- Initialization: shared LHS in [0,1]^D, evaluate every task (LHS_MF) ----
    X = lhs_unit(NI, D)
    DecsT = [X.copy() for _ in range(nt)]            # unified [0,1]^D per task
    ObjsT = []
    for t in range(nt):
        obj, _ = evaluation_single(problem, X[:, :dims[t]], t)
        ObjsT.append(obj.flatten())
    nfes = [NI] * nt
    init_best = [float(np.min(ObjsT[t])) for t in range(nt)]

    # ---- Task similarity S (Spearman + max(0,.)), computed once on init data ----
    S = similarity_S(ObjsT)

    # --------------------------------- main loop ---------------------------------
    n_gen = 0
    n_transfer = 0
    while sum(nfes) < sum(maxF):
        active = [t for t in range(nt) if nfes[t] < maxF[t]]
        if not active:
            break

        # RBF surrogate per active task (built on that task's data, dims[t])
        models = {t: mq_rbf_build(DecsT[t][:, :dims[t]], ObjsT[t]) for t in active}

        # RBF inner GA search per active task -> surrogate-best of each task
        bestDec = {}
        for t in active:
            P = DecsT[t].copy()                       # unified [0,1]^D
            pred_keep = None
            for _w in range(wmax):
                Off = reproduction(P[:, :dims[t]], eta)
                if Off.shape[1] < D:
                    Off = np.hstack([Off, np.zeros((Off.shape[0], D - Off.shape[1]))])
                P = np.vstack([P, Off])
                pred = mq_rbf_predict(models[t], P[:, :dims[t]])
                idx = np.argsort(pred)[:Nin]
                P = P[idx]
                pred_keep = pred[idx]
            bestDec[t] = P[int(np.argmin(pred_keep))]

        # Transfer decision
        if mode == 'none':
            transfer = False
        elif mode == 'both':
            transfer = True
        else:  # adaptive
            transfer = (np.random.rand() < S)
        n_gen += 1
        n_transfer += int(transfer)

        # New-sample construction, evaluation, and database update
        if transfer:
            injected = np.vstack([bestDec[j] for j in active])   # all tasks' bests
        for t in active:
            if transfer:
                cand = injected.copy()                           # task t evaluates all bests
            else:
                cand = bestDec[t][None, :]                       # own best only
            rem = maxF[t] - nfes[t]
            if cand.shape[0] > rem:
                cand = cand[:rem]
            obj, _ = evaluation_single(problem, cand[:, :dims[t]], t)
            DecsT[t] = np.vstack([DecsT[t], cand])
            ObjsT[t] = np.concatenate([ObjsT[t], obj.flatten()])
            nfes[t] += cand.shape[0]

    runtime = time.time() - start

    # ------------------------------- build results -------------------------------
    db_decs = [DecsT[t][:, :dims[t]] for t in range(nt)]
    db_objs = [ObjsT[t].reshape(-1, 1) for t in range(nt)]
    all_decs, all_objs = build_staircase_history(db_decs, db_objs, k=1)
    results = build_save_results(
        all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes,
        bounds=problem.bounds, save_path=save_path, filename=name, save_data=save_data,
    )
    results.similarity_S = S
    results.n_gen = n_gen
    results.n_transfer = n_transfer
    results.init_best = init_best
    return results

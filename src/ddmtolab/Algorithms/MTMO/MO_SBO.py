"""
Multi-Objective Symbiosis-Based Optimization (MO-SBO)

This module implements the MO-SBO algorithm for multi-objective many-task optimization
based on symbiotic relationships in biocoenosis. The algorithm adaptively controls
knowledge transfer rates by tracking six types of symbiotic interactions: mutualism,
commensalism, parasitism, competition, amensalism, and neutralism.

References
----------
    [1] R.-T. Liaw and C.-K. Ting. "Evolutionary Manytasking Optimization Based on Symbiosis in Biocoenosis." Proceedings of the AAAI Conference on Artificial Intelligence, 33(01): 4295-4303, 2019.

Notes
-----
The code is developed in accordance with the MATLAB-based MTO-platform framework.

Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.21
Version: 1.0
"""
import time
import numpy as np
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *
def _ga_generation_matlab(parents, muc, mum):
    """
    Offspring generation matching MToP ``MO_SBO.Generation``.

    A random permutation is split in halves and parent i is paired with parent
    i + floor(N/2) for i = 1..ceil(N/2); each pair yields two children (SBX
    then polynomial mutation) and the single clip to [0, 1] happens only after
    mutation. The result is truncated back to N so the positional bookkeeping
    (``rankO``/``BelongT``) stays well defined for odd N.

    Parameters
    ----------
    parents : np.ndarray
        Parent population of shape (n, d)
    muc : float
        SBX distribution index
    mum : float
        Polynomial mutation distribution index

    Returns
    -------
    offdecs : np.ndarray
        Offspring of shape (n, d)
    """
    n, d = parents.shape
    order = np.random.permutation(n)
    half = n // 2
    n_pairs = int(np.ceil(n / 2))
    offdecs = np.empty((2 * n_pairs, d))

    count = 0
    for i in range(n_pairs):
        p1 = parents[order[i], :]
        p2 = parents[order[i + half], :]
        c1, c2 = sbx_crossover_unclipped(p1, p2, muc)
        c1 = poly_mutation_unclipped(c1, mum)
        c2 = poly_mutation_unclipped(c2, mum)
        offdecs[count] = np.clip(c1, 0, 1)
        offdecs[count + 1] = np.clip(c2, 0, 1)
        count += 2

    return offdecs[:n]


def _nsga2_rank(objs, cons):
    """
    Compute the 1-based NSGA-II rank of each individual (MToP ``NSGA2Sort``).

    Parameters
    ----------
    objs : np.ndarray
        Objective values, shape (pop_size, n_obj)
    cons : np.ndarray
        Constraint values, shape (pop_size, n_con)

    Returns
    -------
    rank : np.ndarray
        1-based rank of each individual (lower is better), shape (pop_size,)
    """
    pop_size = objs.shape[0]
    if cons is not None and cons.size > 0:
        front_no, _ = nd_sort(objs, cons, pop_size)
    else:
        front_no, _ = nd_sort(objs, pop_size)
    crowd_dis = crowding_distance(objs, front_no)
    order = np.lexsort((-crowd_dis, front_no))
    rank = np.empty(pop_size, dtype=int)
    rank[order] = np.arange(1, pop_size + 1)
    return rank


class MO_SBO:
    """
    Multi-Objective Symbiosis-Based Optimization for many-task multi-objective optimization.

    The algorithm uses symbiotic relationships between tasks to adaptively control
    knowledge transfer rates. Six types of symbiotic interactions are tracked:

    - Mutualism (MIJ): Both tasks benefit (transferred solution ranks high in both)
    - Commensalism (OIJ): One benefits, other neutral
    - Parasitism (PIJ): One benefits, other harmed
    - Competition (CIJ): Both harmed
    - Amensalism (AIJ): One harmed, other neutral
    - Neutralism (NIJ): Both neutral

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
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, benefit=0.25, harm=0.5,
                 mu_c=20, mu_m=15, save_data=True, save_path='./Data',
                 name='MO-SBO', disable_tqdm=True):
        """
        Initialize MO-SBO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        benefit : float, optional
            Beneficial factor threshold for symbiosis categorization (default: 0.25)
        harm : float, optional
            Harmful factor threshold for symbiosis categorization (default: 0.5)
        mu_c : float, optional
            Distribution index for simulated binary crossover (default: 20)
        mu_m : float, optional
            Distribution index for polynomial mutation (default: 15)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MO-SBO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.benefit = benefit
        self.harm = harm
        self.mu_c = mu_c
        self.mu_m = mu_m
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MO-SBO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives,
            constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        n = self.n
        nt = problem.n_tasks
        dims = problem.dims
        max_dim = max(dims)
        max_nfes = self.max_nfes * nt

        # Initialize the population and evaluate. MToP evolves one unified
        # genome of length max(D) per individual (Dec = rand(1, max(D))) and
        # truncates to D(t) only at evaluation time; the padded genes are
        # active because a whole genome is copied across tasks on transfer.
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        # Report what each task actually consumed rather than the requested
        # budget, which the analysis tools use to scale convergence curves
        nfes_per_task = [n] * nt

        uni_decs = []
        for t in range(nt):
            padded = np.random.rand(n, max_dim)
            padded[:, :dims[t]] = decs[t]
            uni_decs.append(padded)

        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # rankO of the current population. MToP assigns the NSGA-II rank to the
        # (unsorted) initial population and, from the first environmental
        # selection onwards, the population is kept sorted so rankO == position.
        rank_o = [_nsga2_rank(objs[t], cons[t]) for t in range(nt)]

        # Symbiosis interaction counters (initialized to 1 to avoid division by zero)
        RIJ = 0.5 * np.ones((nt, nt))  # Transfer rates
        MIJ = np.ones((nt, nt))  # Mutualism
        NIJ = np.ones((nt, nt))  # Neutralism
        CIJ = np.ones((nt, nt))  # Competition
        OIJ = np.ones((nt, nt))  # Commensalism
        PIJ = np.ones((nt, nt))  # Parasitism
        AIJ = np.ones((nt, nt))  # Amensalism

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while nfes < max_nfes:
            # === Step 1: Generate offspring for each task (SBX + PM) ===
            off_uni_decs = []
            off_rank_o = []
            off_belong_t = []
            for t in range(nt):
                off_uni_decs.append(_ga_generation_matlab(uni_decs[t], self.mu_c, self.mu_m))
                # Positional inheritance of the parent's rankO, as in MToP
                off_rank_o.append(rank_o[t].copy())
                off_belong_t.append(np.full(n, t))

            # === Step 2: Knowledge transfer driven by the symbiosis rates ===
            for t in range(nt):
                # Task with the highest transfer rate, excluding t itself
                rij_row = RIJ[t].copy()
                rij_row[t] = -np.inf
                transfer_task = int(np.argmax(rij_row))

                if np.random.rand() < RIJ[t, transfer_task]:
                    si = int(np.floor(n * RIJ[t, transfer_task]))
                    if si > 0:
                        ind1 = np.random.permutation(n)[:si]
                        ind2 = np.random.permutation(n)[:si]
                        # Copies the (possibly already transferred) offspring of
                        # the donor task, matching MToP's sequential loop
                        off_uni_decs[t][ind1] = off_uni_decs[transfer_task][ind2]
                        off_belong_t[t][ind1] = transfer_task

            # === Step 3: Evaluate, compute rankC and select for each task ===
            all_rank_c = [None] * nt
            for t in range(nt):
                off_objs_t, off_cons_t = evaluation_single(problem, off_uni_decs[t][:, :dims[t]], t)
                nfes += n
                nfes_per_task[t] += n
                pbar.update(n)

                # rankC: NSGA-II rank of the offspring among themselves
                all_rank_c[t] = _nsga2_rank(off_objs_t, off_cons_t)

                merged_uni = np.vstack([uni_decs[t], off_uni_decs[t]])
                merged_objs = np.vstack([objs[t], off_objs_t])
                merged_cons = np.vstack([cons[t], off_cons_t])

                merged_rank = _nsga2_rank(merged_objs, merged_cons)
                select_idx = np.argsort(merged_rank, kind='stable')[:n]

                uni_decs[t] = merged_uni[select_idx]
                objs[t] = merged_objs[select_idx]
                cons[t] = merged_cons[select_idx]
                decs[t] = uni_decs[t][:, :dims[t]]

                # The surviving population is stored in rank order, so rankO
                # becomes the 1-based position
                rank_o[t] = np.arange(1, n + 1)

            # === Step 4: Update the symbiosis counters ===
            for t in range(nt):
                transferred_idx = np.where(off_belong_t[t] != t)[0]
                for k in transferred_idx:
                    rc = all_rank_c[t][k]        # rank on the receiving task
                    ro = off_rank_o[t][k]        # rank inherited from the parent slot
                    src = int(off_belong_t[t][k])

                    if rc < n * self.benefit:
                        if ro < n * self.benefit:
                            MIJ[t, src] += 1  # both benefit
                        elif ro > n * (1 - self.harm):
                            PIJ[t, src] += 1  # one benefits, other harmed
                        else:
                            OIJ[t, src] += 1  # one benefits, other neutral
                    elif rc > n * (1 - self.harm):
                        if ro > n * (1 - self.harm):
                            CIJ[t, src] += 1  # both harmed
                    else:
                        if ro > n * (1 - self.harm):
                            AIJ[t, src] += 1  # one harmed, other neutral
                        elif n * self.benefit <= ro <= n * (1 - self.harm):
                            NIJ[t, src] += 1  # both neutral

            # === Step 5: Update the transfer rates ===
            RIJ = (MIJ + OIJ + PIJ) / (MIJ + OIJ + PIJ + AIJ + CIJ + NIJ)

            append_history(all_decs, decs, all_objs, objs, all_cons, cons)

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data
        )

        return results

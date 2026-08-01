"""
EBS (Evolutionary Biocoenosis-based Symbiosis)

This module implements the EBS algorithm for evolutionary many-tasking optimization.

References
----------
    [1] Liaw, R. T., & Ting, C. K. (2017). Evolutionary many-tasking based on biocoenosis through symbiosis: A framework and benchmark problems. In 2017 IEEE Congress on Evolutionary Computation (CEC) (pp. 2266-2273). IEEE. doi:10.1109/CEC.2017.7969579
    [2] Liaw, R. T., & Ting, C. K. (2020). Evolution of biocoenosis through symbiosis with fitness approximation for many-tasking optimization. Memetic Computing, 12(4), 399-417. doi:10.1007/s12293-020-00317-2
    [3] Tan, X. (2018). EBSGA.m, reference implementation of [1], in drwuHUST/MTGA. https://github.com/drwuHUST/MTGA/blob/master/EBSGA.m

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2025.01.09
Version: 1.0
"""
from tqdm import tqdm
import copy
import time
import numpy as np
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class EBS:
    """
    Evolutionary Biocoenosis-based Symbiosis for many-task optimization.

    EBS uses multiple CMA-ES instances with adaptive information exchange among tasks.
    Each task maintains two CMA-ES distributions:
    - One updated when knowledge transfer occurs
    - One updated when no knowledge transfer occurs

    The information exchange probability ``gamma_i`` is recomputed every generation
    from the transfer-rate formula of [2] (attributed there to [1])::

        gamma_i = S_io / (S_is + S_io)
        S_is = #Surpassing_is / #Evals_is
        S_io = #Surpassing_io / #Evals_io

    where the ``_is`` counters are credited to the task's own EA and the ``_io``
    counters to the union of the other tasks' EAs. There is no warm-up phase:
    generation 1 only initialises the populations, and ``gamma`` is seeded with
    ``rmp0`` and updated from generation 2 onward, matching the reference
    implementation [3].

    Known defect (reference-faithful, not a porting bug)
    ----------------------------------------------------
    ``gamma_i`` has an absorbing zero state. Transfer is the only way foreign
    offspring are ever evaluated, so once ``gamma_i`` reaches 0 (or NaN) the
    ``_io`` counters can never grow again and transfer stays dead for the rest of
    the run. Two entry conditions exist, both present in [3]:

    - ``#Evals_io == 0`` (the transfer coin has never fired): ``S_io = 0/0 = NaN``,
      ``gamma_i = NaN``, and ``rand() < NaN`` is False forever.
    - ``#Evals_io > 0`` but ``#Surpassing_io == 0``: ``S_io = 0`` and
      ``gamma_i = 0``.

    Seeding with ``rmp0 = 0.3`` makes the first condition unlikely but does not
    remove it. The optional ``gamma_min`` guard restores ergodicity but is a
    deviation from the published algorithm and is disabled by default.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, sigma0=0.3, use_n=True,
                 rmp0=0.3, gamma_min=0.0, save_data=True, save_path='./Data',
                 name='EBS', disable_tqdm=True):
        """
        Initialize EBS Algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: None, will use 4+3*log(D))
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        sigma0 : float, optional
            Initial step size for CMA-ES (default: 0.3)
        use_n : bool, optional
            If True, use provided n; if False, use 4+3*log(D) (default: True)
        rmp0 : float, optional
            Initial information-exchange probability, used for generation 2
            before any transfer statistics exist (default: 0.3, the value the
            reference implementation's driver passes to ``EBSGA.m``)
        gamma_min : float, optional
            Lower bound applied to gamma, also used in place of NaN
            (default: 0.0, i.e. disabled). Any value > 0 is a deliberate
            DEVIATION from the published algorithm: it removes the absorbing
            zero state described in the class docstring by guaranteeing that
            transfer generations keep occurring, at the cost of no longer
            reproducing the reference behaviour.
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EBS')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.sigma0 = sigma0
        self.use_n = use_n
        self.rmp0 = rmp0
        self.gamma_min = gamma_min
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EBS Algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        d_max = np.max(dims)  # Unified dimension
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize CMA-ES parameters for each task (using unified dimension)
        params = []
        for t in range(nt):
            # Determine population size based on original task dimension
            if self.use_n:
                lam = par_list(self.n, nt)[t]
            else:
                lam = int(4 + 3 * np.log(dims[t]))

            # Distribution for self-generated offspring (no knowledge transfer)
            params_s = cmaes_init_params(d_max, lam=lam, sigma0=self.sigma0)
            # Distribution for knowledge transfer offspring (same starting point)
            params_o = copy.deepcopy(params_s)
            params_o['m_dec'] = params_s['m_dec'].copy()

            params.append({
                'real_dim': dims[t],  # Real dimension for this task
                'params_s': params_s,  # Self distribution (no transfer)
                'params_o': params_o,  # Other distribution (with transfer)
            })

        # Initialize tracking variables
        nfes_per_task = [0] * nt
        decs = [None] * nt  # In unified space
        objs = [None] * nt
        cons = [None] * nt
        origins = [None] * nt  # Originating task of each retained individual
        all_decs = [[] for _ in range(nt)]
        all_objs = [[] for _ in range(nt)]
        all_cons = [[] for _ in range(nt)]

        # Information exchange statistics (cumulative over the whole run, never
        # reset). improvements_* are #Surpassing_is / #Surpassing_io and
        # evals_* are #Evals_is / #Evals_io in the formula of [2].
        improvements_s = [0] * nt
        evals_s = [0] * nt
        improvements_o = [0] * nt
        evals_o = [0] * nt
        # gamma is seeded with rmp0 and first updated at the end of generation 2,
        # exactly as EBSGA.m does (RMP = rmp*ones(1,nTasks), rmp = 0.3).
        gamma = [float(self.rmp0)] * nt

        gen = 0
        pbar = tqdm(total=sum(max_nfes_per_task), desc=f"{self.name}", disable=self.disable_tqdm)

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            gen += 1
            # Generation 1 only establishes the initial populations: EBSGA.m
            # evaluates a random population at gen 1 and starts the transfer
            # loop at gen 2. Its evaluations still count towards #Evals_is.
            is_init_gen = (gen == 1)

            # Step 1: Determine transfer flags and generate offspring accordingly
            transfer_flags = []
            offspring_list = []

            for i in active_tasks:
                p = params[i]

                # Per-generation Bernoulli draw on gamma (Algorithm 2, line 9).
                # A NaN gamma compares False, which is the reference behaviour.
                is_transfer = (not is_init_gen) and bool(np.random.rand() < gamma[i])
                transfer_flags.append(is_transfer)

                # Select distribution based on transfer decision
                if is_transfer:
                    ps = p['params_o']  # Use knowledge transfer distribution
                else:
                    ps = p['params_s']  # Use self distribution

                # Generate offspring using selected CMA-ES distribution
                sample_decs = cmaes_sample(
                    ps['m_dec'], ps['sigma'], ps['B'], ps['D'], ps['lam']
                )
                offspring_list.append(sample_decs)

            # Step 2: For each task, select candidates based on transfer decision
            n_act = len(active_tasks)
            candidate_list = []
            candidate_origins = []

            for idx, i in enumerate(active_tasks):
                p = params[i]
                lam_i = p['params_s']['lam']

                if transfer_flags[idx]:
                    # EBSGA.m lines 83-92: every candidate slot independently
                    # picks a source task uniformly over ALL tasks (the target
                    # task included), then takes an offspring of that task
                    # through a per-task permutation indexed by the slot. Slots
                    # that pick the same source therefore get distinct members,
                    # while the number taken from each source is free to vary
                    # (unlike a without-replacement draw over the flat pool).
                    slot_src = np.random.randint(0, n_act, size=lam_i)
                    perms = [np.random.permutation(o.shape[0]) for o in offspring_list]
                    candidate_decs = np.empty((lam_i, d_max))
                    for j in range(lam_i):
                        src = slot_src[j]
                        # Modulo generalises the reference (which assumes one
                        # common popSize) to unequal per-task lambda.
                        candidate_decs[j] = offspring_list[src][perms[src][j % len(perms[src])]]
                    candidate_list.append(candidate_decs)
                    candidate_origins.append(np.asarray(active_tasks, dtype=int)[slot_src])
                else:
                    # No knowledge transfer: use self-generated offspring
                    candidate_list.append(offspring_list[idx])
                    candidate_origins.append(np.full(lam_i, i, dtype=int))

            # Step 3: Evaluate, update population and CMA-ES parameters
            for idx, i in enumerate(active_tasks):
                p = params[i]
                candidate_decs = candidate_list[idx]  # In unified space
                cand_origin = candidate_origins[idx]
                is_transfer = transfer_flags[idx]
                lam_i = p['params_s']['lam']

                # Convert to real space for evaluation (truncate to real dimension)
                candidate_decs_real = candidate_decs[:, :dims[i]]
                candidate_decs_real = np.clip(candidate_decs_real, 0, 1)  # Ensure bounds

                # Evaluate candidates (in real space)
                sample_objs, sample_cons = evaluation_single(problem, candidate_decs_real, i)

                # #Evals_io / #Evals_is: split the batch by the ORIGINATING task
                # of each candidate, not by the generation's transfer flag
                # (EBSGA.m line 93 counts length(find(idt ~= idxTask))).
                n_foreign = int(np.count_nonzero(cand_origin != i))
                evals_o[i] += n_foreign
                evals_s[i] += len(cand_origin) - n_foreign

                # Sort by constraint violation first, then by objective
                cvs = np.sum(np.maximum(0, sample_cons), axis=1)
                sort_indices = constrained_sort(sample_objs, cvs)

                # Sort in unified space
                sorted_decs = candidate_decs[sort_indices]
                sorted_objs = sample_objs[sort_indices]
                sorted_cons = sample_cons[sort_indices]
                sorted_origin = cand_origin[sort_indices]

                # Elitist population update (store in unified space): merge the
                # retained population with the newly evaluated candidates and keep
                # the best lam (EBSGA.m lines 101-104 and 116-117). The origin tag
                # travels with the individual, mirroring Chromosome.skill_factor,
                # so a surviving foreign solution keeps being credited to the task
                # that produced it.
                if decs[i] is None:
                    decs[i], objs[i] = sorted_decs, sorted_objs
                    cons[i], origins[i] = sorted_cons, sorted_origin
                else:
                    pool_decs = np.vstack([decs[i], sorted_decs])
                    pool_objs = np.vstack([objs[i], sorted_objs])
                    pool_cons = np.vstack([cons[i], sorted_cons])
                    pool_origin = np.concatenate([origins[i], sorted_origin])
                    pool_cvs = np.sum(np.maximum(0, pool_cons), axis=1)
                    keep = constrained_sort(pool_objs, pool_cvs)[:lam_i]
                    decs[i], objs[i] = pool_decs[keep], pool_objs[keep]
                    cons[i], origins[i] = pool_cons[keep], pool_origin[keep]

                if not is_init_gen:
                    # #Surpassing: EBSGA.m lines 106-114 credit exactly one
                    # counter per task per generation, chosen by the origin of
                    # the best individual of the merged pool. The test there is
                    # `<= bestFitness(gen-1)`, which under elitist survival is
                    # always true because the previous best is in the pool, so
                    # the credit fires unconditionally.
                    if origins[i][0] == i:
                        improvements_s[i] += 1
                    else:
                        improvements_o[i] += 1

                    # Update the probability of information exchange
                    # (EBSGA.m lines 126-129), from generation 2 onward.
                    gamma[i] = _transfer_rate(
                        improvements_s[i], evals_s[i],
                        improvements_o[i], evals_o[i], self.gamma_min)

                nfes_per_task[i] += lam_i
                pbar.update(lam_i)

                # Convert to real space for history (truncate to real dimension)
                decs_real = decs[i][:, :dims[i]]
                append_history(all_decs[i], decs_real, all_objs[i], objs[i], all_cons[i], cons[i])

                # Update the appropriate CMA-ES distribution
                if is_transfer:
                    # Update knowledge transfer CMA-ES
                    cmaes_update(p['params_o'], sorted_decs, nfes_per_task[i])
                else:
                    # Update self CMA-ES
                    cmaes_update(p['params_s'], sorted_decs, nfes_per_task[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results (all_decs are already in real space)
        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name, save_data=self.save_data
        )

        return results


def _transfer_rate(improve_s, evals_s, improve_o, evals_o, gamma_min=0.0):
    """
    Information-exchange probability of EBS.

    Implements ``gamma_i = S_io / (S_is + S_io)`` with
    ``S_ix = #Surpassing_ix / #Evals_ix`` (Liaw & Ting 2020, attributed to the
    CEC 2017 EBS paper; identical to ``RMP = RO/(RO+RS)`` in EBSGA.m lines
    127-129).

    No smoothing, prior or clipping is applied: the reference divides straight
    through, so an unexercised branch yields ``0/0 = NaN`` and a branch with no
    surpasses yields 0. Both propagate to ``gamma`` and both are absorbing,
    because ``rand() < NaN`` and ``rand() < 0`` are always False and transfer is
    the only source of foreign evaluations. This is a defect of the published
    algorithm, reproduced here deliberately.

    Parameters
    ----------
    improve_s, evals_s : int
        #Surpassing_is and #Evals_is (the task's own EA).
    improve_o, evals_o : int
        #Surpassing_io and #Evals_io (the union of the other tasks' EAs).
    gamma_min : float, optional
        If > 0, a lower bound that also replaces NaN. DEVIATION from the
        reference, disabled by default (see EBS.__init__).

    Returns
    -------
    float
        gamma in [0, 1], or NaN when a branch has never been exercised.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        R_o = np.float64(improve_o) / np.float64(evals_o)
        R_s = np.float64(improve_s) / np.float64(evals_s)
        gamma = float(R_o / (R_o + R_s))

    if gamma_min > 0.0 and (np.isnan(gamma) or gamma < gamma_min):
        gamma = float(gamma_min)
    return gamma


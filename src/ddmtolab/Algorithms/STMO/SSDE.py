"""
Self-Organizing Surrogate-Assisted Non-Dominated Sorting Differential Evolution (SSDE)

This module implements SSDE for computationally expensive multi/many-objective optimization.
It uses a Self-Organizing Map (SOM) as a surrogate model to predict offspring quality,
combined with NSGA-II environmental selection. Only offspring that survive selection are
evaluated with the expensive objective function.

References
----------
    [1] A. F. R. Araujo, L. R. C. Farias, and A. R. C. Goncalves. Self-organizing surrogate-assisted non-dominated sorting differential evolution. Swarm and Evolutionary Computation, 2024, 91: 101703.

Notes
-----
Author: Jiangtao Shen (DDMTOLab implementation)
Email: j.shen5@exeter.ac.uk
Date: 2026.02.16
Version: 1.0
"""
from tqdm import tqdm
import time
import numpy as np
from scipy.spatial.distance import cdist
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import warnings

warnings.filterwarnings("ignore")


class SSDE:
    """
    Self-Organizing Surrogate-Assisted Non-Dominated Sorting Differential Evolution
    for expensive multi/many-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'True',
        'knowledge_transfer': 'False',
        'n_initial': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None,
                 num_nodes=None, eta0=0.2, sigma0=None,
                 save_data=True, save_path='./Data', name='SSDE', disable_tqdm=True):
        """
        Initialize SSDE algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task (default: 11*dim-1).
            This plays the role of MATLAB's ``Problem.N``: it is simultaneously
            the population size, the initial sample count, the default number of
            SOM neurons and the default neighbourhood size.
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 200)
        num_nodes : int, optional
            Number of neurons in the SOM (default: same as n_initial)
        eta0 : float, optional
            Initial learning rate for SOM training (default: 0.2)
        sigma0 : float, optional
            Initial neighborhood size for SOM (default: same as n_initial)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'SSDE')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.max_nfes = max_nfes if max_nfes is not None else 200
        self.n_initial = n_initial
        self.num_nodes = num_nodes
        self.eta0 = eta0
        self.sigma0 = sigma0
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the SSDE algorithm.

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
        n_cons = problem.n_cons

        # MATLAB: Problem.N is at once the population size and the number of
        # initially sampled solutions (Population = Problem.Initialization()).
        if self.n_initial is None:
            n_per_task = [11 * dims[i] - 1 for i in range(nt)]
        else:
            n_per_task = par_list(self.n_initial, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Generate initial samples using random initialization
        decs = initialization(problem, n_per_task, method='random')
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()

        # History tracking
        has_cons = any(nc > 0 for nc in n_cons)

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        # Per-task state
        task_states = []
        for i in range(nt):
            N = n_per_task[i]
            D = dims[i]
            M = n_objs[i]
            num_nodes = self.num_nodes if self.num_nodes is not None else N
            sigma0 = self.sigma0 if self.sigma0 is not None else float(N)

            # Initialize SOM weight vectors: W ~ N(0.5, 0.001)
            a, b = 0.001, 0.5
            W = a * np.random.randn(num_nodes, D + M) + b

            # 1D latent space: neurons at positions 1, 2, ..., num_nodes
            V = np.arange(1, num_nodes + 1, dtype=float).reshape(-1, 1)
            LDis = cdist(V, V)  # pairwise distances in latent space

            # Winning weights tracking
            winning_weights = np.zeros(num_nodes, dtype=bool)

            # Current population (indices into accumulated decs/objs)
            pop_decs = decs[i].copy()
            pop_objs = objs[i].copy()
            pop_cons = cons[i].copy() if n_cons[i] > 0 else None

            # Sample set for SOM training
            sample_decs = pop_decs.copy()
            sample_objs = pop_objs.copy()

            task_states.append({
                'W': W, 'LDis': LDis, 'V': V,
                'winning_weights': winning_weights,
                'pop_decs': pop_decs, 'pop_objs': pop_objs, 'pop_cons': pop_cons,
                'sample_decs': sample_decs, 'sample_objs': sample_objs,
                'num_nodes': num_nodes, 'sigma0': sigma0,
                'D': D, 'M': M, 'N': N,
            })

        # Safety guard: MATLAB loops forever if no offspring ever survives the
        # NSGA-II selection (no expensive evaluation is consumed in that case).
        stagnation = 0

        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break
            nfes_before_sweep = sum(nfes_per_task)

            for i in active_tasks:
                st = task_states[i]
                N = st['N']
                D = st['D']
                M = st['M']

                # Training: when enough samples accumulated
                if st['sample_decs'].shape[0] >= N:
                    st['W'] = _som_training(
                        st['W'], st['LDis'], st['sample_decs'], st['sample_objs'],
                        st['num_nodes'], self.eta0, st['sigma0'], st['winning_weights'],
                        D, M
                    )
                    st['winning_weights'] = np.zeros(st['num_nodes'], dtype=bool)
                    st['sample_decs'] = np.empty((0, D))
                    st['sample_objs'] = np.empty((0, M))

                # Operator: generate offspring, predict with SOM, select
                pop_decs = st['pop_decs']
                pop_objs = st['pop_objs']
                pop_cons = st['pop_cons']
                n_pop = pop_decs.shape[0]

                # Binary tournament on total constraint violation.
                # MATLAB: TournamentSelection(2,N,sum(max(0,Population.cons),2)).
                # For unconstrained problems every fitness is 0, so PlatEMO's
                # tie handling degenerates to uniform random sampling.
                if pop_cons is not None and pop_cons.shape[1] > 0:
                    cv = np.sum(np.maximum(0, pop_cons), axis=1)
                    mating_pool = platemo_tournament_selection(2, N, cv)
                else:
                    mating_pool = np.random.randint(0, n_pop, size=N)

                # DE offspring generation (OperatorDE, CR=1, F=0.5, proM=1, disM=20)
                donor1_idx = np.random.randint(0, n_pop, size=N)
                donor2_idx = np.random.randint(0, n_pop, size=N)
                offspring_dec = _operator_de(
                    pop_decs[mating_pool], pop_decs[donor1_idx], pop_decs[donor2_idx]
                )

                # Map offspring to nearest SOM neuron to estimate objectives
                Distance = cdist(offspring_dec, st['W'][:, :D])
                rank = np.argsort(Distance, axis=1)
                offspring_labels = st['W'][rank[:, 0], D:D + M]

                # Track winning neurons
                st['winning_weights'][rank[:, 0]] = True

                # NSGA-II selection on combined population
                combined_objs = np.vstack([pop_objs, offspring_labels])

                if pop_cons is not None and pop_cons.shape[1] > 0:
                    # Offspring get zero constraints (optimistic assumption)
                    combined_cons = np.vstack([
                        pop_cons,
                        np.zeros((offspring_labels.shape[0], pop_cons.shape[1]))
                    ])
                    front_no, max_fno = nd_sort(combined_objs, combined_cons, N)
                else:
                    combined_cons = None
                    front_no, max_fno = nd_sort(combined_objs, N)

                # Select N survivors
                Next = front_no < max_fno
                crowd_dis = crowding_distance(combined_objs, front_no)
                last_front = np.where(front_no == max_fno)[0]
                last_cd = crowd_dis[last_front]
                last_sorted = np.argsort(last_cd)[::-1]  # descending
                n_needed = N - np.sum(Next)
                for idx in last_sorted[:n_needed]:
                    Next[last_front[idx]] = True

                # Split selection into parent survivors and offspring survivors
                out = Next[:n_pop]        # which parents survive
                in_mask = Next[n_pop:]    # which offspring survive

                if np.sum(in_mask) >= 1:
                    # Evaluate surviving offspring with expensive function
                    selected_off_dec = offspring_dec[in_mask]

                    # Limit to remaining budget
                    remaining = max_nfes_per_task[i] - nfes_per_task[i]
                    if selected_off_dec.shape[0] > remaining:
                        selected_off_dec = selected_off_dec[:remaining]

                    in_count = selected_off_dec.shape[0]
                    new_objs, new_cons = evaluation_single(problem, selected_off_dec, i)

                    # Update population: replace non-surviving parents with evaluated offspring
                    # (matching MATLAB: Population(~out) = Offspring)
                    non_surviving = np.where(~out)[0]
                    n_replace = min(in_count, len(non_surviving))

                    for j in range(n_replace):
                        st['pop_decs'][non_surviving[j]] = selected_off_dec[j]
                        st['pop_objs'][non_surviving[j]] = new_objs[j]
                        if pop_cons is not None:
                            st['pop_cons'][non_surviving[j]] = new_cons[j]

                    # Accumulate samples for SOM training
                    st['sample_decs'] = np.vstack([st['sample_decs'], selected_off_dec[:in_count]])
                    st['sample_objs'] = np.vstack([st['sample_objs'], new_objs[:in_count]])

                    # Update global tracking
                    decs[i] = np.vstack([decs[i], selected_off_dec[:in_count]])
                    objs[i] = np.vstack([objs[i], new_objs[:in_count]])
                    cons[i] = np.vstack([cons[i], new_cons[:in_count]])

                    nfes_per_task[i] += in_count
                    pbar.update(in_count)

                else:
                    # No offspring survived, use current population as samples
                    st['sample_decs'] = np.vstack([st['sample_decs'], pop_decs])
                    st['sample_objs'] = np.vstack([st['sample_objs'], pop_objs])

            if sum(nfes_per_task) == nfes_before_sweep:
                stagnation += 1
                if stagnation >= 100:
                    break
            else:
                stagnation = 0

        pbar.close()
        runtime = time.time() - start_time

        if has_cons:
            all_decs, all_objs, all_cons = build_staircase_history(decs, objs, k=1, db_cons=cons)
        else:
            all_decs, all_objs = build_staircase_history(decs, objs, k=1)
            all_cons = None
        if all_cons is not None:
            results = build_save_results(
                all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                max_nfes=nfes_per_task, all_cons=all_cons, bounds=problem.bounds,
                save_path=self.save_path, filename=self.name,
                save_data=self.save_data)
        else:
            results = build_save_results(
                all_decs=all_decs, all_objs=all_objs, runtime=runtime,
                max_nfes=nfes_per_task, bounds=problem.bounds,
                save_path=self.save_path, filename=self.name,
                save_data=self.save_data)

        return results


# =============================================================================
# SOM Training
# =============================================================================

def _som_training(W, LDis, sample_decs, sample_objs, num_nodes, eta0, sigma0,
                  winning_weights, D, M):
    """
    Train the SOM surrogate model with memory-based reinitialization.

    Parameters
    ----------
    W : np.ndarray
        SOM weight matrix, shape (num_nodes, D+M)
    LDis : np.ndarray
        Pairwise distances in latent space, shape (num_nodes, num_nodes)
    sample_decs : np.ndarray
        Sample decision variables in [0,1], shape (n_samples, D)
    sample_objs : np.ndarray
        Sample objective values, shape (n_samples, M)
    num_nodes : int
        Number of SOM neurons
    eta0 : float
        Initial learning rate
    sigma0 : float
        Initial neighborhood size
    winning_weights : np.ndarray
        Boolean mask of neurons that won during previous phase, shape (num_nodes,)
    D : int
        Number of decision variables
    M : int
        Number of objectives

    Returns
    -------
    W : np.ndarray
        Updated SOM weight matrix
    """
    W = W.copy()

    # Memory-based reinitialization of non-winning neurons
    n_winning = np.sum(winning_weights)
    if n_winning > 0 and n_winning < num_nodes:
        winning_idx = np.where(winning_weights)[0]
        non_winning_idx = np.where(~winning_weights)[0]
        n_dead = len(non_winning_idx)

        # Non-dominated sorting on winning nodes' objective weights
        winning_objs = W[winning_idx, D:]
        front_no, _ = nd_sort(winning_objs, n_winning)

        # Crowding distance for diversity criterion
        crowd_dis = crowding_distance(winning_objs, front_no)

        # Rank by crowding distance: higher CD = higher weight
        sorted_cd_idx = np.argsort(crowd_dis)
        factor = np.zeros(n_winning)
        for rank_val, idx in enumerate(sorted_cd_idx):
            factor[idx] = rank_val + 1  # 1-indexed rank

        # Weighted sampling with replacement
        probs = factor / np.sum(factor)
        chosen_local = np.random.choice(n_winning, size=2 * n_dead, replace=True, p=probs)
        chosen_local = chosen_local.reshape(n_dead, 2)

        # Map local indices to global winning indices
        chosen_global = winning_idx[chosen_local]

        # Linear combination + noise
        W[non_winning_idx] = (
            (W[chosen_global[:, 0]] + W[chosen_global[:, 1]]) / 2.0
            + 0.001 * np.random.randn(n_dead, D + M)
        )

        # Repair decision space bounds
        W[:, :D] = np.clip(W[:, :D], 0.0, 1.0)

    # Concatenate samples: [decs, objs]
    # Decision variables are already in [0, 1]
    Samples = np.hstack([sample_decs, sample_objs])
    n_samples = Samples.shape[0]

    # Reset win count
    win_count_set = np.zeros(num_nodes)

    # SOM training: 50 epochs
    for epoch in range(1, 51):
        # Shuffle samples
        randpos = np.random.permutation(n_samples)

        # Per-neuron neighborhood radius and learning rate
        sigma = sigma0 * np.exp(-win_count_set / n_samples)
        eta = eta0 * np.exp(-win_count_set / n_samples)

        for ii in range(n_samples):
            s = randpos[ii]

            # Find winning neuron (nearest in decision space)
            dists = np.sum((Samples[s, :D] - W[:, :D]) ** 2, axis=1)
            u1 = np.argmin(dists)

            # First win: assign sample to neuron directly
            if win_count_set[u1] == 0:
                W[u1] = Samples[s]

            # Update win counter (limited to current epoch)
            if win_count_set[u1] < epoch:
                win_count_set[u1] += 1

            # Update winning neuron and its neighborhood
            U = LDis[u1] < sigma  # neighbors within each neuron's own sigma
            U_idx = np.where(U)[0]

            if len(U_idx) > 0:
                eta_U = eta[U_idx].reshape(-1, 1)
                dist_decay = np.exp(-LDis[u1, U_idx]).reshape(-1, 1)
                W[U_idx] += eta_U * dist_decay * (Samples[s] - W[U_idx])

    return W


# =============================================================================
# DE Operator
# =============================================================================
def _operator_de(parent1, parent2, parent3, CR=1.0, F=0.5, proM=1.0, disM=20.0):
    """
    Exact port of PlatEMO's ``OperatorDE`` (default parameters {1, 0.5, 1, 20}).

    Rand/1 differential mutation on a random subset of variables followed by
    polynomial mutation. Decision variables live in the normalised box [0, 1],
    so ``Lower = 0`` and ``Upper = 1``.

    Parameters
    ----------
    parent1 : np.ndarray
        Base vectors, shape (N, D)
    parent2, parent3 : np.ndarray
        Difference vectors, shape (N, D)
    CR : float
        Differential evolution crossover rate (PlatEMO default 1)
    F : float
        Differential weight (PlatEMO default 0.5)
    proM : float
        Expected number of mutated variables (PlatEMO default 1)
    disM : float
        Distribution index of polynomial mutation (PlatEMO default 20)

    Returns
    -------
    offspring : np.ndarray
        Offspring decision variables, shape (N, D), inside [0, 1]
    """
    N, D = parent1.shape

    # Differential evolution
    site = np.random.rand(N, D) < CR
    offspring = parent1.copy()
    offspring[site] = offspring[site] + F * (parent2[site] - parent3[site])

    # Polynomial mutation (bounds are 0/1 in the normalised space)
    site = np.random.rand(N, D) < proM / D
    mu = np.random.rand(N, D)
    offspring = np.clip(offspring, 0.0, 1.0)

    temp = site & (mu <= 0.5)
    if np.any(temp):
        x = offspring[temp]
        offspring[temp] = x + ((2 * mu[temp] + (1 - 2 * mu[temp]) *
                                (1 - x) ** (disM + 1)) ** (1 / (disM + 1)) - 1)

    temp = site & (mu > 0.5)
    if np.any(temp):
        x = offspring[temp]
        offspring[temp] = x + (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) *
                                    x ** (disM + 1)) ** (1 / (disM + 1)))

    return offspring

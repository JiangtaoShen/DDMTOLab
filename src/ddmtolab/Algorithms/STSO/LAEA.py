"""
LLM-Assisted Evolutionary Algorithm (LAEA)

This module implements LAEA for expensive single-objective optimization
problems. The surrogate is not a trained model but a large language model
prompted with the archive as in-context examples: a regression prompt supplies
the value used to pick the single solution that is really evaluated, and a
classification prompt supplies the good/bad labels that steer the next
generation's estimation-of-distribution model.

The port follows the reference implementation (class ``LSEA`` in
https://github.com/hhyqhh/LAEA) rather than the paper, including its
variable-width histogram EDA, its quadratic local search, and its one-real-
evaluation-per-generation budget policy. The reference builds on pymoo; this
version is native and works in the platform's normalized [0, 1] space, where
the EDA bounds are 0 and 1.

Cost note: each generation issues ``2 * n_initial`` LLM calls -- one regression
and one classification query per offspring -- and consumes exactly one real
function evaluation. The default setting therefore spends about 25,000 calls
to reach ``max_nfes=300``. Set ``max_llm_calls`` to cap this, and keep the
cache file so the run can be replayed offline.

References
----------
    [1] Hao, H., Zhang, X., & Zhou, A. (2024). Large Language Models as Surrogate Models in Evolutionary Algorithms: A Preliminary Study. arXiv preprint arXiv:2406.10675.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.03
Version: 1.0
"""
import copy
import os
import time
import warnings

import numpy as np
from tqdm import tqdm

from ddmtolab.Methods.Algo_Methods.algo_utils import *
from ddmtolab.Methods.LLM_Methods.llm_budget import LLMBudget, LLMBudgetExceeded
from ddmtolab.Methods.LLM_Methods.llm_cache import LLMCache
from ddmtolab.Methods.LLM_Methods.llm_client import LLMCacheMiss, LLMClient
from ddmtolab.Methods.LLM_Methods.llm_surrogate import (
    LLM_Classification,
    LLM_Regression,
    load_prompt,
)

warnings.filterwarnings("ignore")


def local_search(Xs, ys):
    """
    Quadratic interpolation refinement along each coordinate.

    For every triple of consecutive solutions a parabola is fitted per
    dimension and the first solution of the triple is moved to its vertex.

    Parameters
    ----------
    Xs : ndarray
        Solutions sorted by fitness, shape (n, d).
    ys : ndarray
        Matching objective values, shape (n,).

    Returns
    -------
    ndarray
        Refined copy of ``Xs``.
    """
    Xs_new = copy.deepcopy(Xs)
    TINY = 1e-50
    No, Dim = Xs.shape
    for r0 in range(0, No - 2):
        r1 = r0 + 1
        r2 = r0 + 2
        for d in range(Dim):
            if (abs(Xs[r1, d] - Xs[r0, d]) > TINY and abs(Xs[r2, d] - Xs[r1, d]) > TINY
                    and abs(Xs[r0, d] - Xs[r2, d]) > TINY):
                a = ((ys[r1] - ys[r0]) / (Xs[r1, d] - Xs[r0, d])
                     - (ys[r0] - ys[r2]) / (Xs[r0, d] - Xs[r2, d])) / (Xs[r1, d] - Xs[r2, d])
                b = (ys[r1] - ys[r0]) / (Xs[r1, d] - Xs[r0, d]) - a * (Xs[r1, d] + Xs[r0, d])
                if abs(a) > TINY:
                    Xs_new[r0, d] = -b / (2.0 * a)
    return Xs_new


class VWH:
    """
    Variable-width histogram estimation-of-distribution model.

    Each dimension is modelled by an ``m``-bin histogram whose interior bin
    edges track the spread of the current population, with the two outer bins
    kept at a small fixed probability so that the model can still escape the
    region it has converged on.

    Parameters
    ----------
    m : int, optional
        Number of histogram bins per dimension (default: 15).
    """

    def __init__(self, m=15):
        self.M = m
        self.D = None
        self.LB = None
        self.UB = None
        self.originalLB = None
        self.originalUB = None
        self.Prob = None
        self.Range = None

    def init(self, D, LB, UB):
        """Initialize a uniform histogram over the box [LB, UB]."""
        self.D = D
        self.LB = LB
        self.UB = UB
        self.originalLB = LB
        self.originalUB = UB
        self.Prob = np.ones(shape=[self.M, self.D]) / self.M
        self.Range = (np.tile(self.LB, [self.M + 1, 1])
                      + np.tile(self.UB - self.LB, [self.M + 1, 1])
                      * np.tile(np.linspace(0, 1, self.M + 1).reshape(self.M + 1, -1), [1, self.D]))

    def update(self, Xs):
        """Re-fit the bin edges and bin probabilities on the solutions ``Xs``."""
        NP = Xs.shape[0]
        Xs_t = copy.deepcopy(Xs)
        Xs_t.sort(axis=0)

        LB = Xs_t[0, :] - 0.5 * (Xs_t[1, :] - Xs_t[0, :])
        UB = Xs_t[-1, :] + 0.5 * (Xs_t[-1, :] - Xs_t[-2, :])

        mask = LB < self.originalLB
        LB[mask] = self.originalLB[mask]
        mask = UB > self.originalUB
        UB[mask] = self.originalUB[mask]

        self.LB = LB
        self.UB = UB
        self.Range[1:self.M, :] = (np.tile(self.LB, [self.M - 1, 1])
                                   + np.tile(self.UB - self.LB, [self.M - 1, 1])
                                   * np.tile(np.linspace(0, 1, self.M - 1).reshape(self.M - 1, -1),
                                             [1, self.D]))

        epsilon = 1e-10
        index = np.floor((self.M - 2) * (Xs - np.tile(self.LB, [NP, 1]))
                         / (np.tile(self.UB - self.LB, [NP, 1]) + epsilon))

        if (index > self.M - 3).any() or np.isnan(index).any():
            index[index > self.M - 3] = self.M - 3
            index[np.isnan(index)] = 0

        Prob = np.zeros([self.M, self.D])
        Prob[1:self.M - 1, :] = self._update_umda(index)

        mask = self.Range[1, :] > self.Range[0, :]
        Prob[0, mask] = 0.1
        mask = self.Range[-1, :] > self.Range[-2, :]
        Prob[-1, mask] = 0.1

        self.Prob = Prob / np.tile(np.sum(Prob, axis=0), [self.M, 1])

    def _update_umda(self, index):
        """Count how many solutions fall into each interior bin."""
        NM = self.M - 2
        N = index.shape[0]
        Prob1 = np.ones([NM, self.D])
        for d in range(self.D):
            for k in range(N):
                Prob1[int(index[k, d]), d] += 1
        return Prob1

    def sample(self, N):
        """Draw ``N`` solutions from the current histogram."""
        probs = np.cumsum(self.Prob, axis=0)
        return self._sample_umda(
            probs,
            np.random.random([N, self.D]) * np.tile(probs[self.M - 1, :], [N, 1]),
            np.random.random([N, self.D]),
            self.Range
        )

    def _sample_umda(self, Probs, prob0, prob1, ranges):
        """Pick a bin by inverse-CDF, then a uniform point inside that bin."""
        NM = self.M
        N = prob0.shape[0]
        pop = np.zeros(shape=prob0.shape)

        for d in range(self.D):
            for k in range(N):
                for i in range(NM):
                    if i == NM - 1:
                        index = NM - 1
                    elif prob0[k, d] <= Probs[i, d]:
                        index = i
                        break
                pop[k, d] = ranges[index, d] + prob1[k, d] * (ranges[index + 1, d] - ranges[index, d])
        return pop


class LAEA:
    """
    LLM-Assisted Evolutionary Algorithm for expensive optimization problems.

    Each generation:

    1. take the ``tao`` best archive members as in-context examples;
    2. label them good/bad at quantile ``rate`` and fit both LLM surrogates;
    3. sample offspring from the VWH model, refined by quadratic local search
       and crossed with it;
    4. ask the regression surrogate to rank the offspring and the
       classification surrogate to label them;
    5. really evaluate only the top-ranked offspring, and carry the
       positively-labelled offspring forward as the unevaluated population that
       widens the next VWH update.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    llm_information : dict
        Dictionary describing how the algorithm uses a language model
    """

    #: Surrogate classes used for the two prompts, and how many query points
    #: each request covers. LAEA_light overrides them to batch; changing them
    #: here would change LAEA itself.
    regression_cls = LLM_Regression
    classification_cls = LLM_Classification
    llm_batch_size = 1

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

    llm_information = {
        'llm_role': 'surrogate',
        'backend': 'openai_compatible',
        'default_model': 'deepseek-chat',
        'max_llm_calls': '[1, L]',
        'prompt_version': 'laea_reg_v1 + laea_cla_v1',
        'calls_per_generation': '2 * n_initial',
        'offline_replay': 'True',
        'deterministic': 'cache_only'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, tao=50, rate=0.3, pb=0.2, pc=0.2, m=15,
                 llm_backend='replay', llm_model='deepseek-chat', llm_base_url='https://api.deepseek.com',
                 llm_api_key_env='DEEPSEEK_API_KEY', llm_temperature=0.0, llm_max_tokens=10,
                 llm_max_retries=3, llm_parallel=1, llm_beta=3, llm_seed=42, max_llm_calls=None,
                 llm_price_prompt=0.0, llm_price_completion=0.0, llm_cache_path='',
                 strict_source=False, save_data=True, save_path='./Data',
                 name='LAEA', disable_tqdm=True):
        """
        Initialize LAEA algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n_initial : int or List[int], optional
            Number of initial samples per task, also the population size and the
            number of offspring per generation (default: 50)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 300)
        tao : int, optional
            Number of best archive members shown to the LLM as examples (default: 50)
        rate : float, optional
            Quantile splitting good from bad solutions when labelling (default: 0.3)
        pb : float, optional
            Fraction of the population refined by local search (default: 0.2)
        pc : float, optional
            Per-variable probability of crossing an offspring with a
            locally-refined solution (default: 0.2)
        m : int, optional
            Number of histogram bins per dimension in the VWH model (default: 15)
        llm_backend : str, optional
            'openai_compatible' to call a real endpoint, 'replay' to serve only
            from the cache, or 'mock' for offline synthetic responses
            (default: 'replay')
        llm_model : str, optional
            Model identifier (default: 'deepseek-chat')
        llm_base_url : str, optional
            Endpoint root, passed through verbatim (default: 'https://api.deepseek.com')
        llm_api_key_env : str, optional
            Environment variable holding the API key (default: 'DEEPSEEK_API_KEY')
        llm_temperature : float, optional
            Sampling temperature (default: 0.0)
        llm_max_tokens : int, optional
            Completion token cap. The reference uses 10 (default: 10)
        llm_max_retries : int, optional
            Attempts per query before falling back to a random prediction (default: 3)
        llm_parallel : int, optional
            Worker threads used to issue the per-offspring queries (default: 1)
        llm_beta : int, optional
            Decimal places kept when rounding features in the prompt (default: 3)
        llm_seed : int, optional
            Seed for the fallback RNG of the surrogates (default: 42)
        max_llm_calls : int, optional
            Cap on LLM calls per run. The run stops early and returns partial
            results when it is reached (default: None, unlimited)
        llm_price_prompt : float, optional
            USD per million prompt tokens, used only to report an estimated
            spend (default: 0.0)
        llm_price_completion : float, optional
            USD per million completion tokens (default: 0.0)
        llm_cache_path : str, optional
            JSONL cache file. Empty means ``<save_path>/llm_cache/<name>.jsonl``
            (default: '')
        strict_source : bool, optional
            Reproduce the reference regression parser, which accepts only the
            'Value' JSON key even though its own prompt asks for 'Target'
            (default: False)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'LAEA')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n_initial = n_initial if n_initial is not None else 50
        self.max_nfes = max_nfes if max_nfes is not None else 300
        self.tao = tao
        self.rate = rate
        self.pb = pb
        self.pc = pc
        self.m = m

        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key_env = llm_api_key_env
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_max_retries = llm_max_retries
        self.llm_parallel = llm_parallel
        self.llm_beta = llm_beta
        self.llm_seed = llm_seed
        self.max_llm_calls = max_llm_calls
        self.llm_price_prompt = llm_price_prompt
        self.llm_price_completion = llm_price_completion
        self.llm_cache_path = llm_cache_path
        self.strict_source = strict_source

        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        self.llm_stats = None

    def optimize(self):
        """
        Execute the LAEA algorithm.

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

        for i in range(nt):
            n_local = int(np.floor(n_initial_per_task[i] * self.pb))
            if n_local < 3:
                raise ValueError(
                    f"task {i}: floor(n_initial * pb) = {n_local}, but the crossover step needs "
                    f"at least 3 locally-refined solutions; increase n_initial or pb"
                )

        budget = LLMBudget(max_llm_calls=self.max_llm_calls,
                           price_per_1m_prompt=self.llm_price_prompt,
                           price_per_1m_completion=self.llm_price_completion)
        cache_path = self.llm_cache_path
        if not cache_path:
            cache_path = os.path.join(self.save_path, 'llm_cache', f'{self.name}.jsonl')
        client = LLMClient(
            backend=self.llm_backend, model=self.llm_model, base_url=self.llm_base_url,
            api_key_env=self.llm_api_key_env, temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens, seed=self.llm_seed,
            cache=LLMCache(path=cache_path), budget=budget
        )
        surrogate_kwargs = dict(
            client=client, max_retries=self.llm_max_retries, beta=self.llm_beta,
            parallel=self.llm_parallel, show_progress=not self.disable_tqdm, seed=self.llm_seed,
            batch_size=self.llm_batch_size
        )

        # Fail before the initial design is evaluated rather than after it
        client.preflight()

        # Generate initial samples using Latin Hypercube Sampling
        decs = initialization(problem, self.n_initial, method='lhs')
        objs, _ = evaluation(problem, decs)
        nfes_per_task = n_initial_per_task.copy()

        # Per-task state: archive, survivor population, unevaluated population,
        # distribution model and the two LLM surrogates
        archive_decs = [decs[i].copy() for i in range(nt)]
        archive_objs = [objs[i].copy() for i in range(nt)]
        pop_decs = [decs[i].copy() for i in range(nt)]
        pop_objs = [objs[i].copy() for i in range(nt)]
        unevaluated_decs = [decs[i].copy() for i in range(nt)]

        eda = []
        for i in range(nt):
            model = VWH(m=self.m)
            model.init(D=dims[i], LB=np.zeros(dims[i]), UB=np.ones(dims[i]))
            eda.append(model)

        m1 = [self.regression_cls(introduction=load_prompt('laea_reg_v1.txt'),
                                  strict_source=self.strict_source, **surrogate_kwargs) for _ in range(nt)]
        m2 = [self.classification_cls(introduction=load_prompt('laea_cla_v1.txt'),
                                      **surrogate_kwargs) for _ in range(nt)]

        pbar = tqdm(total=sum(max_nfes_per_task), initial=sum(n_initial_per_task),
                    desc=f"{self.name}", disable=self.disable_tqdm)

        stopped_early = None
        while sum(nfes_per_task) < sum(max_nfes_per_task):
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                pop_size = n_initial_per_task[i]

                # Training data: the tao best archive members
                t_decs, t_objs = self._raw_training_data(archive_decs[i], archive_objs[i])

                # Label them and fit both surrogates on the same examples
                labels = self._get_label(t_objs, self.rate)
                m1[i].fit(t_decs, t_objs)
                m2[i].fit(t_decs, labels)

                # Sample offspring from the distribution model
                off_decs = self._reproduction(eda[i], pop_decs[i], pop_objs[i],
                                              unevaluated_decs[i], pop_size, dims[i])

                # Surrogate-assisted selection: 2 * pop_size LLM calls
                try:
                    objs_pre = m1[i].predict(off_decs)
                    labels_pre = m2[i].predict(off_decs)
                except (LLMBudgetExceeded, LLMCacheMiss) as exc:
                    stopped_early = f"{type(exc).__name__}: {exc}"
                    break

                best_dec = copy.deepcopy(off_decs[np.argsort(objs_pre.flatten())[0], :]).reshape(1, -1)

                selected_decs = off_decs[labels_pre == 1, :]
                if selected_decs.shape[0] > pop_size / 2:
                    r_index = np.random.permutation(selected_decs.shape[0])
                    selected_decs = selected_decs[r_index[:int(pop_size / 2)], :]
                unevaluated_decs[i] = selected_decs

                # Only the top-ranked offspring is really evaluated
                best_obj, _ = evaluation_single(problem, best_dec, i)
                archive_decs[i] = np.vstack([archive_decs[i], best_dec])
                archive_objs[i] = np.vstack([archive_objs[i], best_obj])

                nfes_per_task[i] += 1
                pbar.update(1)

                # Survival: keep the pop_size best archive members
                index = np.argsort(archive_objs[i].flatten())[:pop_size]
                pop_decs[i] = archive_decs[i][index, :]
                pop_objs[i] = archive_objs[i][index, :]

            if stopped_early is not None:
                break

        pbar.close()
        runtime = time.time() - start_time

        self.llm_stats = budget.report()
        if stopped_early is not None:
            warnings.warn(f"{self.name} stopped after {sum(nfes_per_task)} evaluations: {stopped_early}",
                          RuntimeWarning)

        all_decs, all_objs = build_staircase_history(archive_decs, archive_objs, k=1)

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=nfes_per_task, bounds=problem.bounds,
            save_path=self.save_path, filename=self.name,
            save_data=self.save_data, llm_stats=self.llm_stats
        )

        return results

    # ==================== Internal Methods ====================

    def _raw_training_data(self, archive_decs, archive_objs):
        """
        Select the ``tao`` best archive members as in-context examples.

        Returns
        -------
        tuple of ndarray
            Decision variables of shape (k, d) and flat objective values of
            shape (k,), with k = min(len(archive), tao).
        """
        if archive_decs.shape[0] <= self.tao:
            return archive_decs, archive_objs.flatten()
        index = archive_objs.flatten().argsort()
        return archive_decs[index[:self.tao], :], archive_objs[index[:self.tao], :].flatten()

    @staticmethod
    def _get_label(objs, rate):
        """
        Split objective values into +1 (better) and -1 (worse) at a quantile.

        Parameters
        ----------
        objs : ndarray
            Flat objective values.
        rate : float
            Quantile of the split boundary.

        Returns
        -------
        ndarray
            Labels in {+1, -1}, same shape as ``objs``.
        """
        objs_sorted = np.sort(copy.deepcopy(objs).flatten())
        split_bound = objs_sorted[int(len(objs_sorted) * rate)]

        labels = np.zeros_like(objs)
        labels[objs <= split_bound] = 1
        labels[objs > split_bound] = -1
        return labels

    def _reproduction(self, eda, pop_decs, pop_objs, unevaluated_decs, pop_size, dim):
        """
        Generate ``pop_size`` offspring from the VWH model.

        The model is updated on the survivors plus the better half of the
        unevaluated population, which widens the distribution with regions the
        classifier liked but that were never really evaluated. Sampled
        offspring are then crossed variable-wise with locally-refined elites.

        Returns
        -------
        ndarray
            Offspring of shape (pop_size, dim), inside [0, 1].
        """
        index = np.argsort(pop_objs.flatten())
        xs = pop_decs[index, :]
        ys = pop_objs[index, :].flatten()

        eda.update(np.concatenate([xs, unevaluated_decs[:int(pop_size / 2), :]], axis=0))
        off_decs = eda.sample(pop_size)

        n_local = int(np.floor(pop_size * self.pb))
        xs_ls = local_search(xs[:n_local, :], ys[:n_local])

        index = np.floor(np.random.random((pop_size, 1)) * (xs_ls.shape[0] - 2)).astype(int).flatten()
        xtmp = xs_ls[index, :]
        mask = np.random.random((pop_size, dim)) < self.pc
        off_decs[mask] = xtmp[mask]

        # Boundary repair, halfway back towards the matching parent
        pos = off_decs < 0.0
        off_decs[pos] = 0.5 * xs[pos]
        pos = off_decs > 1.0
        off_decs[pos] = 0.5 * (xs[pos] + 1.0)

        return off_decs

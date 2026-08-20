"""
Batched LLM-Assisted Evolutionary Algorithm (LAEA-light)

This module implements LAEA-light, a drop-in variant of
:class:`~ddmtolab.Algorithms.STSO.LAEA.LAEA` that asks the language model about
a whole generation of offspring in one request instead of one request per
candidate.

The search itself is untouched -- same variable-width histogram EDA, same
quadratic local search, same one-real-evaluation-per-generation policy, and the
same two questions asked of the same in-context examples. Only the transport
changes: the regression prompt and the classification prompt each carry every
candidate of the current generation and come back with one answer per candidate.

Why this is worth a separate class: in LAEA about 97% of every prompt is the
in-context archive, and the serial path resends it once per candidate. At
``n_initial=50`` a generation therefore costs ``2 * 50 = 100`` requests and, at
D=10 with ``tao=50``, roughly 149,000 prompt tokens -- to buy a single real
function evaluation. Batching the whole generation brings that down to two
requests and about 3,500 prompt tokens.

``llm_batch_size`` caps how many candidates share a request, so a generation
costs ``2 * ceil(n_initial / llm_batch_size)`` requests. The default packs the
whole generation into one request per surrogate.

Notes on fidelity
-----------------
The two surrogates stay independent, as in LAEA: the regression answer never
appears in the classification prompt or the other way round. What does change is
that a candidate is now judged alongside its siblings rather than alone, so
predictions are not bit-identical to the serial ones and existing serial
``llm_cache`` files do not carry over. A response that cannot be parsed, or that
does not return exactly one answer per candidate, is retried and then falls back
per candidate exactly as the serial path does.

References
----------
    [1] Hao, H., Zhang, X., & Zhou, A. (2024). Large Language Models as Surrogate Models in Evolutionary Algorithms: A Preliminary Study. arXiv preprint arXiv:2406.10675.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.15
Version: 1.0
"""
from ddmtolab.Algorithms.STSO.LAEA import LAEA
from ddmtolab.Methods.Algo_Methods.algo_utils import get_algorithm_information
from ddmtolab.Methods.LLM_Methods.llm_surrogate import (
    LLM_ClassificationBatch,
    LLM_RegressionBatch,
)


class LAEA_light(LAEA):
    """
    LAEA with one LLM request per surrogate per generation instead of one
    request per candidate.
    """

    regression_cls = LLM_RegressionBatch
    classification_cls = LLM_ClassificationBatch

    # Declared literally rather than copied from LAEA: the GUI reads these two
    # dicts out of the source with the AST, so a computed value would show up
    # as an empty capability panel
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
        'calls_per_generation': '2 * ceil(n_initial / llm_batch_size)',
        'offline_replay': 'True',
        'deterministic': 'cache_only'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n_initial=None, max_nfes=None, tao=50, rate=0.3, pb=0.2, pc=0.2, m=15,
                 llm_batch_size=None,
                 llm_backend='replay', llm_model='deepseek-chat', llm_base_url='https://api.deepseek.com',
                 llm_api_key_env='DEEPSEEK_API_KEY', llm_temperature=0.0, llm_max_tokens=2048,
                 llm_max_retries=3, llm_parallel=1, llm_beta=3, llm_seed=42, max_llm_calls=None,
                 llm_price_prompt=0.0, llm_price_completion=0.0, llm_cache_path='',
                 strict_source=False, save_data=True, save_path='./Data',
                 name='LAEA-light', disable_tqdm=True):
        """
        Initialize LAEA-light.

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
        llm_batch_size : int, optional
            Number of candidates packed into one request. ``None`` puts the whole
            generation into one request per surrogate, which is the point of this
            variant and costs two requests per real evaluation. A smaller value
            trades that back for shorter, easier to parse responses and costs
            ``2 * ceil(n_initial / llm_batch_size)`` (default: None)
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
            Completion cap. A batched answer carries one entry per candidate, so
            this is far larger than the serial default of 10 (default: 2048)
        llm_max_retries : int, optional
            Attempts per request before falling back (default: 3)
        llm_parallel : int, optional
            Worker threads issuing requests. Reduces wall-clock time, not the
            number of requests (default: 1)
        llm_beta : int, optional
            Decimal places kept when rounding normalized features (default: 3)
        llm_seed : int, optional
            Seed forwarded to the endpoint and used by the fallback RNG (default: 42)
        max_llm_calls : int, optional
            Hard cap on requests; the run stops early when it is reached
            (default: None)
        llm_price_prompt : float, optional
            USD per 1M prompt tokens, for the cost estimate (default: 0.0)
        llm_price_completion : float, optional
            USD per 1M completion tokens, for the cost estimate (default: 0.0)
        llm_cache_path : str, optional
            Cache file; defaults to ``<save_path>/llm_cache/<name>.jsonl``.
            Serial LAEA caches are not compatible because the prompts differ
            (default: '')
        strict_source : bool, optional
            Accept only the ``Value`` JSON key, reproducing the reference parser
            (default: False)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'LAEA-light')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        super().__init__(
            problem, n_initial=n_initial, max_nfes=max_nfes, tao=tao, rate=rate,
            pb=pb, pc=pc, m=m, llm_backend=llm_backend, llm_model=llm_model,
            llm_base_url=llm_base_url, llm_api_key_env=llm_api_key_env,
            llm_temperature=llm_temperature, llm_max_tokens=llm_max_tokens,
            llm_max_retries=llm_max_retries, llm_parallel=llm_parallel,
            llm_beta=llm_beta, llm_seed=llm_seed, max_llm_calls=max_llm_calls,
            llm_price_prompt=llm_price_prompt, llm_price_completion=llm_price_completion,
            llm_cache_path=llm_cache_path, strict_source=strict_source,
            save_data=save_data, save_path=save_path, name=name,
            disable_tqdm=disable_tqdm
        )
        # None means the whole generation in one request; n_initial is also the
        # number of offspring, so it is the largest batch that can occur
        self.llm_batch_size = int(llm_batch_size) if llm_batch_size else int(self.n_initial)

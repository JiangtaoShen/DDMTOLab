"""
Backend-agnostic chat client for LLM-assisted algorithms.

Three backends are provided:

``openai_compatible``
    Any endpoint that speaks the OpenAI chat-completions protocol. One code
    path therefore covers OpenAI, DeepSeek, vLLM, Ollama's ``/v1`` shim and the
    usual hosted providers.
``mock``
    Deterministic synthetic responses derived from a hash of the prompt. No
    network, no key, no cost. Intended for unit tests of the surrounding
    algorithm logic.
``replay``
    Serves only from the cache and raises on a miss. This is what makes an
    LLM-assisted experiment reproducible offline and runnable in CI.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.03
Version: 1.0
"""
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ddmtolab.Methods.LLM_Methods.llm_budget import LLMBudget
from ddmtolab.Methods.LLM_Methods.llm_cache import LLMCache, make_cache_key

BACKENDS = ('openai_compatible', 'mock', 'replay')


class LLMCacheMiss(Exception):
    """Raised by the ``replay`` backend when a prompt is not in the cache."""


@dataclass
class LLMResponse:
    """
    One model response.

    Attributes
    ----------
    text : str
        Raw response content, exactly as returned by the model.
    prompt_tokens : int
        Prompt tokens billed for this request.
    completion_tokens : int
        Completion tokens billed for this request.
    cached : bool
        True when the response was served from the cache.
    latency : float
        Wall-clock seconds spent on the request (0.0 for a cache hit).
    """
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency: float = 0.0


class LLMClient:
    """
    Uniform chat interface over the supported backends.

    Parameters
    ----------
    backend : str, optional
        One of ``'openai_compatible'``, ``'mock'``, ``'replay'``
        (default: 'replay').
    model : str, optional
        Model identifier (default: 'deepseek-chat').
    base_url : str, optional
        Endpoint root for ``openai_compatible``. Passed through verbatim, so it
        must already include any ``/v1`` suffix the provider requires
        (default: 'https://api.deepseek.com').
    api_key_env : str, optional
        Name of the environment variable holding the API key. The key itself is
        never accepted as an argument and never written to disk
        (default: 'DEEPSEEK_API_KEY').
    temperature : float, optional
        Sampling temperature (default: 0.0).
    max_tokens : int, optional
        Completion token cap (default: 10).
    seed : int, optional
        Seed forwarded to providers that support it (default: 42).
    timeout : float, optional
        Per-request timeout in seconds (default: 120.0).
    cache : LLMCache, optional
        Cache instance. When None, an in-memory cache is created (default: None).
    budget : LLMBudget, optional
        Budget instance. When None, an uncapped one is created (default: None).

    Raises
    ------
    ValueError
        If ``backend`` is not supported.
    """

    def __init__(self,
                 backend: str = 'replay',
                 model: str = 'deepseek-chat',
                 base_url: str = 'https://api.deepseek.com',
                 api_key_env: str = 'DEEPSEEK_API_KEY',
                 temperature: float = 0.0,
                 max_tokens: int = 10,
                 seed: int = 42,
                 timeout: float = 120.0,
                 cache: Optional[LLMCache] = None,
                 budget: Optional[LLMBudget] = None):
        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r} is not supported; expected one of {BACKENDS}")

        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout

        self.cache = cache if cache is not None else LLMCache(path=None)
        self.budget = budget if budget is not None else LLMBudget()

        self._client = None  # lazily constructed OpenAI SDK client

    @property
    def gen_params(self) -> Dict[str, Any]:
        """Generation parameters that participate in the cache key."""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'seed': self.seed,
        }

    def preflight(self) -> None:
        """
        Check that the backend can actually serve requests.

        Call this before spending real function evaluations. Without it the
        first failure surfaces only at the first prediction, by which time the
        initial design has already been evaluated -- expensive, since these
        algorithms exist precisely because evaluations are costly.

        Raises
        ------
        RuntimeError
            The API key environment variable is unset, or the replay cache is
            empty and could not serve anything.
        """
        if self.backend == 'openai_compatible':
            if not os.environ.get(self.api_key_env):
                raise RuntimeError(
                    f"environment variable {self.api_key_env} is not set; "
                    f"export the API key before running with backend='openai_compatible'"
                )
        elif self.backend == 'replay':
            if len(self.cache) == 0:
                path = self.cache.path or '<in-memory>'
                raise RuntimeError(
                    f"backend='replay' but the cache at {path} is empty, so every "
                    f"prediction would fail. Run once with "
                    f"backend='openai_compatible' to populate it, or point "
                    f"llm_cache_path at an archived cache file."
                )

    def chat(self, prompt: str, attempt: int = 0) -> LLMResponse:
        """
        Send a single-user-message chat request.

        Parameters
        ----------
        prompt : str
            Fully rendered prompt.
        attempt : int, optional
            Retry index, kept distinct in the cache so that a retry after an
            unparseable response is a real second request rather than a replay
            of the first failure (default: 0).

        Returns
        -------
        LLMResponse

        Raises
        ------
        LLMCacheMiss
            Backend is ``replay`` and the prompt is not cached.
        LLMBudgetExceeded
            The ``max_llm_calls`` cap has been reached.
        """
        key = make_cache_key(self.model, prompt, self.gen_params, attempt)

        record = self.cache.get(key)
        if record is not None:
            response = LLMResponse(
                text=record['text'],
                prompt_tokens=record.get('prompt_tokens', 0),
                completion_tokens=record.get('completion_tokens', 0),
                cached=True,
                latency=0.0,
            )
            self.budget.record(response.prompt_tokens, response.completion_tokens, cached=True)
            return response

        if self.backend == 'replay':
            raise LLMCacheMiss(
                "backend='replay' but this prompt is not in the cache. "
                "Run once with backend='openai_compatible' to populate it, "
                "or point llm_cache_path at an archived cache file."
            )

        # Reserve before spending; raises LLMBudgetExceeded when capped out.
        self.budget.reserve()

        start = time.time()
        if self.backend == 'mock':
            response = self._chat_mock(prompt)
        else:
            response = self._chat_openai_compatible(prompt)
        response.latency = time.time() - start

        self.budget.record(response.prompt_tokens, response.completion_tokens, cached=False)
        self.cache.put(key, {
            'model': self.model,
            'gen': self.gen_params,
            'attempt': attempt,
            'prompt': prompt,
            'text': response.text,
            'prompt_tokens': response.prompt_tokens,
            'completion_tokens': response.completion_tokens,
        })
        return response

    def _chat_openai_compatible(self, prompt: str) -> LLMResponse:
        """Issue the request through the OpenAI SDK."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise ImportError(
                    "backend='openai_compatible' requires the 'openai' package; "
                    "install it with: pip install ddmtolab[llm]"
                ) from exc

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"environment variable {self.api_key_env} is not set; "
                    f"export the API key before running with backend='openai_compatible'"
                )
            self._client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )
        usage = getattr(completion, 'usage', None)
        return LLMResponse(
            text=completion.choices[0].message.content or '',
            prompt_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            completion_tokens=getattr(usage, 'completion_tokens', 0) or 0,
            cached=False,
        )

    def _chat_mock(self, prompt: str) -> LLMResponse:
        """
        Return a deterministic synthetic response.

        The value is a pure function of the prompt digest, so a mock run is
        repeatable but carries no information about the objective function.
        The shape mirrors what the LAEA prompts ask for, which is what the
        surrogate parsers expect.
        """
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        bucket = int(digest[:8], 16)

        if 'better or worse?' in prompt:
            text = '{"Class": "better"}' if bucket % 2 == 0 else '{"Class": "worse"}'
        else:
            text = '{"Value": "%.5f"}' % ((bucket % 100000) / 100000.0)

        return LLMResponse(
            text=text,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=len(text) // 4,
            cached=False,
        )

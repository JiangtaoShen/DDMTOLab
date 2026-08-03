"""
Shared infrastructure for LLM-assisted optimization algorithms.

This package holds everything an ``Algorithms/*`` file needs in order to use a
large language model as part of the search loop, so that individual algorithm
files stay as thin as the conventional ones.

Modules
-------
llm_client
    Backend-agnostic chat client (``openai_compatible``, ``mock``, ``replay``).
llm_cache
    Content-addressed disk cache. Reproducibility of an LLM run rests on this,
    not on ``temperature=0``.
llm_budget
    Call / token / cost accounting and the ``max_llm_calls`` cap.
llm_surrogate
    LLM-as-surrogate models with a scikit-learn style ``fit`` / ``predict``.
"""
from ddmtolab.Methods.LLM_Methods.llm_budget import LLMBudget, LLMBudgetExceeded
from ddmtolab.Methods.LLM_Methods.llm_cache import LLMCache
from ddmtolab.Methods.LLM_Methods.llm_client import LLMClient, LLMResponse, LLMCacheMiss
from ddmtolab.Methods.LLM_Methods.llm_surrogate import (
    LLM_Base,
    LLM_Classification,
    LLM_Regression,
    load_prompt,
)

__all__ = [
    'LLMBudget',
    'LLMBudgetExceeded',
    'LLMCache',
    'LLMCacheMiss',
    'LLMClient',
    'LLMResponse',
    'LLM_Base',
    'LLM_Classification',
    'LLM_Regression',
    'load_prompt',
]

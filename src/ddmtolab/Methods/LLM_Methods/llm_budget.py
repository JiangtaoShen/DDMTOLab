"""
Call, token and cost accounting for LLM-assisted algorithms.

An LLM-assisted algorithm spends two separate budgets: real function
evaluations (``max_nfes``) and model inference calls (``max_llm_calls``).
Reporting only the first one is misleading, because the whole point of an
LLM surrogate is to trade the second for the first. This module tracks the
second one and enforces its cap.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.03
Version: 1.0
"""
import threading
from typing import Dict, Optional


class LLMBudgetExceeded(Exception):
    """Raised when the configured ``max_llm_calls`` cap is reached."""


class LLMBudget:
    """
    Thread-safe counter for LLM usage.

    Parameters
    ----------
    max_llm_calls : int, optional
        Hard cap on the number of logical LLM calls. ``None`` means unlimited
        (default: None). Cache hits are counted but never charged against the
        cap, because they cost neither money nor time.
    price_per_1m_prompt : float, optional
        Price in USD per one million prompt tokens, used only to report an
        estimated cost (default: 0.0).
    price_per_1m_completion : float, optional
        Price in USD per one million completion tokens (default: 0.0).
    """

    def __init__(self,
                 max_llm_calls: Optional[int] = None,
                 price_per_1m_prompt: float = 0.0,
                 price_per_1m_completion: float = 0.0):
        self.max_llm_calls = max_llm_calls
        self.price_per_1m_prompt = price_per_1m_prompt
        self.price_per_1m_completion = price_per_1m_completion

        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_cached = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_parse_failures = 0
        self.n_fallbacks = 0
        # Same counters split by surrogate role, because knowing that 50% of
        # predictions fell back is useless without knowing which model did it.
        self.by_role: Dict[str, Dict[str, int]] = {}

    def reserve(self) -> None:
        """
        Claim one slot of the call budget before issuing a request.

        Raises
        ------
        LLMBudgetExceeded
            If the cap has already been reached.
        """
        with self._lock:
            if self.max_llm_calls is not None and self.n_calls >= self.max_llm_calls:
                raise LLMBudgetExceeded(
                    f"max_llm_calls={self.max_llm_calls} reached")
            self.n_calls += 1

    def record(self, prompt_tokens: int = 0, completion_tokens: int = 0,
               cached: bool = False) -> None:
        """Record the token usage of a completed request."""
        with self._lock:
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            if cached:
                self.n_cached += 1
                # A cache hit is not charged against the cap.
                self.n_calls = max(0, self.n_calls - 1)

    def _role_slot(self, role: str) -> Dict[str, int]:
        """Return the counter dict for ``role``, creating it on first use."""
        slot = self.by_role.get(role)
        if slot is None:
            slot = {'n_predictions': 0, 'n_parse_failures': 0, 'n_fallbacks': 0}
            self.by_role[role] = slot
        return slot

    def record_prediction(self, role: str = 'unknown') -> None:
        """Record one prediction returned to the caller, however it was obtained."""
        with self._lock:
            self._role_slot(role)['n_predictions'] += 1

    def record_parse_failure(self, role: str = 'unknown') -> None:
        """Record one response that could not be parsed into a prediction."""
        with self._lock:
            self.n_parse_failures += 1
            self._role_slot(role)['n_parse_failures'] += 1

    def record_fallback(self, role: str = 'unknown') -> None:
        """Record one prediction that fell back to the random default."""
        with self._lock:
            self.n_fallbacks += 1
            self._role_slot(role)['n_fallbacks'] += 1

    def remaining(self) -> float:
        """Return the number of calls still available."""
        with self._lock:
            if self.max_llm_calls is None:
                return float('inf')
            return max(0, self.max_llm_calls - self.n_calls)

    def est_cost_usd(self) -> float:
        """Return the estimated spend in USD for the tokens seen so far."""
        with self._lock:
            return (self.prompt_tokens / 1e6 * self.price_per_1m_prompt
                    + self.completion_tokens / 1e6 * self.price_per_1m_completion)

    def report(self) -> Dict[str, float]:
        """
        Return a flat dictionary of usage statistics.

        Returns
        -------
        dict
            Keys: ``n_llm_calls``, ``n_cached``, ``prompt_tokens``,
            ``completion_tokens``, ``est_cost_usd``, ``n_parse_failures``,
            ``n_fallbacks``, ``fallback_rate``, and ``by_role`` holding the
            per-surrogate breakdown.
        """
        with self._lock:
            n_pred = sum(slot['n_predictions'] for slot in self.by_role.values())
            by_role = {}
            for role, slot in self.by_role.items():
                by_role[role] = dict(slot)
                by_role[role]['fallback_rate'] = (
                    slot['n_fallbacks'] / slot['n_predictions'] if slot['n_predictions'] else 0.0)
            return {
                'n_llm_calls': self.n_calls,
                'n_cached': self.n_cached,
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'est_cost_usd': (self.prompt_tokens / 1e6 * self.price_per_1m_prompt
                                 + self.completion_tokens / 1e6 * self.price_per_1m_completion),
                'n_parse_failures': self.n_parse_failures,
                'n_fallbacks': self.n_fallbacks,
                'fallback_rate': (self.n_fallbacks / n_pred) if n_pred > 0 else 0.0,
                'by_role': by_role,
            }

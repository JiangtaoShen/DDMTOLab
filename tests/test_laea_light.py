"""
Tests for LAEA-light, the batched variant of LAEA.

The variant exists for one reason -- to stop resending the in-context archive
once per candidate -- so the tests pin the request count, and they pin that LAEA
itself did not change when the dispatch hook was added.

The subtle failure mode is alignment: a batched answer carries one entry per
candidate, and a response with the wrong number of entries must be rejected
rather than lined up by position, which would attach one candidate's predicted
value or class to another.
"""

import math

import numpy as np
import pytest

from ddmtolab.Algorithms.STSO.LAEA import LAEA
from ddmtolab.Algorithms.STSO.LAEA_light import LAEA_light
from ddmtolab.Methods.Algo_Methods.algo_utils import set_seed
from ddmtolab.Methods.LLM_Methods.llm_budget import LLMBudget
from ddmtolab.Methods.LLM_Methods.llm_cache import LLMCache
from ddmtolab.Methods.LLM_Methods.llm_client import LLMClient
from ddmtolab.Methods.LLM_Methods.llm_surrogate import (
    LLM_ClassificationBatch,
    LLM_RegressionBatch,
    load_prompt,
)
from ddmtolab.Methods.mtop import MTOP

# LAEA needs floor(n_initial * pb) >= 3 locally-refined solutions, so with the
# default pb = 0.2 the population cannot go below 15
N_INITIAL = 15
MAX_NFES = 21
INFILLS = MAX_NFES - N_INITIAL


def sphere(x):
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def make_problem(dim=5):
    problem = MTOP()
    problem.add_task(sphere, dim=dim, lower_bound=-5, upper_bound=5)
    return problem


def run(cls, tmp_path, **kwargs):
    """Run one algorithm against the offline mock backend and return it."""
    set_seed(0)
    algorithm = cls(make_problem(), n_initial=N_INITIAL, max_nfes=MAX_NFES,
                    llm_backend='mock', save_data=False,
                    llm_cache_path=str(tmp_path / f'{cls.__name__}.jsonl'),
                    **kwargs)
    algorithm.optimize()
    return algorithm


# =============================================================================
# Request count, which is the whole point of the variant
# =============================================================================

class TestRequestCount:

    def test_two_requests_per_infill(self, tmp_path):
        """One request for the regression prompt, one for the classification."""
        stats = run(LAEA_light, tmp_path).llm_stats
        assert stats['n_llm_calls'] == 2 * INFILLS

    def test_batch_size_sets_the_count(self, tmp_path):
        """2 * ceil(n_initial / batch) requests per infill."""
        for batch in (3, 5, N_INITIAL):
            stats = run(LAEA_light, tmp_path, llm_batch_size=batch).llm_stats
            expected = 2 * math.ceil(N_INITIAL / batch) * INFILLS
            assert stats['n_llm_calls'] == expected, f'batch={batch}'

    def test_serial_laea_still_costs_two_per_candidate(self, tmp_path):
        """Adding the dispatch hook must not have changed LAEA itself."""
        stats = run(LAEA, tmp_path).llm_stats
        assert stats['n_llm_calls'] == 2 * N_INITIAL * INFILLS

    def test_far_cheaper_in_prompt_tokens(self, tmp_path):
        """The archive is sent once per batch instead of once per candidate."""
        serial = run(LAEA, tmp_path).llm_stats['prompt_tokens']
        batched = run(LAEA_light, tmp_path).llm_stats['prompt_tokens']
        assert batched * 4 < serial

    def test_every_candidate_is_still_predicted(self, tmp_path):
        """Batching changes the transport, not how many questions are asked."""
        by_role = run(LAEA_light, tmp_path).llm_stats['by_role']
        for role in ('regression', 'classification'):
            assert by_role[role]['n_predictions'] == N_INITIAL * INFILLS

    def test_nothing_falls_back_on_a_well_formed_backend(self, tmp_path):
        stats = run(LAEA_light, tmp_path).llm_stats
        assert stats['n_fallbacks'] == 0
        assert stats['n_parse_failures'] == 0


# =============================================================================
# Results stay well formed and the budget stays honest
# =============================================================================

class TestResults:

    def test_reported_count_matches_the_evaluations_performed(self, tmp_path):
        set_seed(0)
        problem = make_problem()
        counts = [0]
        original = problem.evaluate_task

        def counting(task_idx, X, *args, **kwargs):
            counts[task_idx] += np.atleast_2d(X).shape[0]
            return original(task_idx, X, *args, **kwargs)

        problem.evaluate_task = counting
        results = LAEA_light(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                             llm_backend='mock', save_data=False,
                             llm_cache_path=str(tmp_path / 'count.jsonl')).optimize()
        assert results.max_nfes == counts == [MAX_NFES]

    def test_shapes_and_bounds(self, tmp_path):
        set_seed(0)
        results = LAEA_light(make_problem(), n_initial=N_INITIAL, max_nfes=MAX_NFES,
                             llm_backend='mock', save_data=False,
                             llm_cache_path=str(tmp_path / 'shape.jsonl')).optimize()
        assert np.all(np.isfinite(results.best_objs[0]))
        assert results.best_decs[0].min() >= 0.0
        assert results.best_decs[0].max() <= 1.0

    def test_a_replay_costs_nothing(self, tmp_path):
        """A second identical run is served from the cache."""
        cache = str(tmp_path / 'replay.jsonl')
        first, second = [], []
        for sink in (first, second):
            set_seed(0)
            algorithm = LAEA_light(make_problem(), n_initial=N_INITIAL, max_nfes=MAX_NFES,
                                   llm_backend='mock', save_data=False, llm_cache_path=cache)
            algorithm.optimize()
            sink.append(algorithm.llm_stats)
        assert first[0]['n_llm_calls'] == 2 * INFILLS
        assert second[0]['n_llm_calls'] == 0
        assert second[0]['n_cached'] == 2 * INFILLS


# =============================================================================
# Alignment: a batched answer must line up with its questions
# =============================================================================

def make_surrogate(cls, tmp_path, prompt_file, batch_size=4):
    client = LLMClient(backend='mock', model='m',
                       cache=LLMCache(path=str(tmp_path / 'unit.jsonl')),
                       budget=LLMBudget())
    surrogate = cls(client=client, introduction=load_prompt(prompt_file),
                    batch_size=batch_size, max_retries=2)
    set_seed(1)
    X = np.random.rand(8, 3)
    y = np.random.rand(8) if cls is LLM_RegressionBatch else np.where(np.arange(8) < 4, 1, -1)
    return surrogate.fit(X, y)


class TestBatchAlignment:

    @pytest.mark.parametrize('cls, prompt_file', [
        (LLM_RegressionBatch, 'laea_reg_v1.txt'),
        (LLM_ClassificationBatch, 'laea_cla_v1.txt'),
    ])
    def test_one_prediction_per_query_point(self, cls, prompt_file, tmp_path):
        surrogate = make_surrogate(cls, tmp_path, prompt_file)
        set_seed(2)
        out = surrogate.predict(np.random.rand(10, 3))
        assert out.shape == (10,)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize('cls, prompt_file, bad', [
        (LLM_RegressionBatch, 'laea_reg_v1.txt', '{"Targets": [0.1, 0.2]}'),
        (LLM_ClassificationBatch, 'laea_cla_v1.txt', '{"Classes": ["better"]}'),
    ])
    def test_a_short_answer_is_rejected_not_padded(self, cls, prompt_file, bad, tmp_path):
        """
        Lining a short list up by position would attach one candidate's answer
        to another, so the batch must be treated as unparseable instead.
        """
        surrogate = make_surrogate(cls, tmp_path, prompt_file, batch_size=4)
        surrogate._query = lambda prompt, attempt=0: bad

        set_seed(3)
        out = surrogate.predict(np.random.rand(4, 3))
        assert out.shape == (4,)
        assert surrogate.client.budget.n_fallbacks == 4

    @pytest.mark.parametrize('cls, prompt_file', [
        (LLM_RegressionBatch, 'laea_reg_v1.txt'),
        (LLM_ClassificationBatch, 'laea_cla_v1.txt'),
    ])
    def test_the_fallback_is_replayable(self, cls, prompt_file, tmp_path):
        """Each row falls back from its own seed, so a rerun reproduces it."""
        runs = []
        for _ in range(2):
            surrogate = make_surrogate(cls, tmp_path, prompt_file, batch_size=4)
            surrogate._query = lambda prompt, attempt=0: 'not json at all'
            set_seed(4)
            runs.append(surrogate.predict(np.random.rand(4, 3)))
        assert np.array_equal(runs[0], runs[1])

    def test_classification_only_accepts_the_two_labels(self, tmp_path):
        surrogate = make_surrogate(LLM_ClassificationBatch, tmp_path, 'laea_cla_v1.txt')
        surrogate._query = lambda prompt, attempt=0: '{"Classes": ["good", "bad", "x", "y"]}'
        set_seed(5)
        out = surrogate.predict(np.random.rand(4, 3))
        assert set(np.unique(out)).issubset({-1, 1})
        assert surrogate.client.budget.n_fallbacks == 4

    def test_last_chunk_may_be_shorter(self, tmp_path):
        """A population that is not a multiple of the batch still lines up."""
        surrogate = make_surrogate(LLM_RegressionBatch, tmp_path, 'laea_reg_v1.txt', batch_size=4)
        set_seed(6)
        out = surrogate.predict(np.random.rand(10, 3))
        assert out.shape == (10,)
        assert surrogate.client.budget.n_fallbacks == 0

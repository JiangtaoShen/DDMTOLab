"""
LLM-as-surrogate models with a scikit-learn style interface.

Faithful port of ``model/surrogate.py`` from the reference LAEA implementation
(https://github.com/hhyqhh/LAEA). The model is not trained: ``fit`` only stores
the archive, and ``predict`` puts that archive into a prompt as in-context
examples and asks the LLM for the value or the class of each query point.

Deviations from the reference, all deliberate
---------------------------------------------
1. Backends go through :class:`~ddmtolab.Methods.LLM_Methods.llm_client.LLMClient`
   rather than being branched inline, so caching, budget accounting and the
   offline ``replay`` backend apply uniformly.
2. Random fallbacks are seeded from the query itself rather than drawn from the
   global ``random`` module, so the value a failed prediction falls back to
   depends only on what was asked -- not on call order or thread scheduling.
   Results are therefore independent of ``parallel``. The distributions are
   unchanged.
3. ``LLM_Regression`` accepts either ``Value`` or ``Target`` as the JSON key by
   default. The reference prompt is self-contradictory -- its preamble asks for
   ``{'Value':...}`` while the trailing note asks for ``{'Target':'result'}`` --
   but its parser reads ``Value`` only, so every response that follows the
   trailing note is discarded and silently replaced by ``random.random()``.
   Pass ``strict_source=True`` to reproduce that behaviour exactly.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.03
Version: 1.0
"""
import copy
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import tqdm

from ddmtolab.Methods.LLM_Methods.llm_client import LLMClient

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')


def load_prompt(name: str) -> str:
    """
    Load a versioned prompt asset verbatim.

    Prompts are kept as files rather than inline strings because the prompt is
    a hyperparameter: it goes into the cache key and results are only
    comparable across runs that used the same prompt version.

    Parameters
    ----------
    name : str
        File name inside ``LLM_Methods/prompts``, e.g. ``'laea_reg_v1.txt'``.

    Returns
    -------
    str
        File contents, unstripped.
    """
    with open(os.path.join(PROMPT_DIR, name), 'r', encoding='utf-8') as f:
        return f.read()


class LLM_Base:
    """
    Shared machinery for the LLM surrogates.

    Parameters
    ----------
    client : LLMClient
        Configured chat client.
    introduction : str
        Task preamble placed before the in-context examples.
    max_retries : int, optional
        Attempts per query point before falling back (default: 3).
    beta : int, optional
        Decimal places kept when rounding normalized features (default: 3).
    parallel : int, optional
        Number of worker threads for prediction. 1 means serial (default: 1).
    show_progress : bool, optional
        Show a per-prediction progress bar (default: False).
    show_prompt : bool, optional
        Print each rendered prompt (default: False).
    show_response : bool, optional
        Print each raw response (default: False).
    seed : int, optional
        Seed for the fallback RNG (default: 42).

    Attributes
    ----------
    role : str
        Label under which this surrogate's predictions are accounted for in the
        budget report.
    """

    role = 'unknown'

    def __init__(self,
                 client: LLMClient,
                 introduction: str,
                 max_retries: int = 3,
                 beta: int = 3,
                 parallel: int = 1,
                 show_progress: bool = False,
                 show_prompt: bool = False,
                 show_response: bool = False,
                 seed: int = 42):
        self.client = client
        self.introduction = introduction
        self.max_retries = max_retries
        self.beta = beta
        self.parallel = parallel
        self.show_progress = show_progress
        self.show_prompt = show_prompt
        self.show_response = show_response

        self.seed = seed
        self.fit_prompts: Optional[List[str]] = None
        self.Train_Xs: Optional[np.ndarray] = None
        self.Train_ys: Optional[np.ndarray] = None

    def fit(self, Xs, ys) -> 'LLM_Base':
        """
        Store the archive that will be shown to the model as examples.

        Parameters
        ----------
        Xs : array_like
            Decision variables, shape (n, d).
        ys : array_like
            Objective values or labels, shape (n,) or (n, 1).

        Returns
        -------
        LLM_Base
            self, so the call can be chained.
        """
        if not isinstance(Xs, np.ndarray):
            Xs = np.array(Xs)
        if Xs.ndim <= 1:
            Xs = Xs.reshape(1, -1)

        self.Train_Xs = Xs
        self.Train_ys = np.array(ys).flatten()
        return self

    def normalize(self, Train_Xs, Test_Xs, range_01=True):
        """
        Min-max scale train and test features jointly, then round to ``beta``.

        Scaling uses the range of the stacked train+test matrix, exactly as in
        the reference implementation, so the numbers a prompt shows depend on
        the query batch as well as the archive.
        """
        combined_Xs = np.vstack((Train_Xs, Test_Xs))
        max_Xs = np.max(combined_Xs, axis=0)
        min_Xs = np.min(combined_Xs, axis=0)

        if range_01:
            Norm_Train_Xs = (Train_Xs - min_Xs) / (max_Xs - min_Xs)
            Norm_Test_Xs = (Test_Xs - min_Xs) / (max_Xs - min_Xs)
        else:
            Norm_Train_Xs = 2 * ((Train_Xs - min_Xs) / (max_Xs - min_Xs)) - 1
            Norm_Test_Xs = 2 * ((Test_Xs - min_Xs) / (max_Xs - min_Xs)) - 1

        Norm_Train_Xs = np.round(Norm_Train_Xs, self.beta)
        Norm_Test_Xs = np.round(Norm_Test_Xs, self.beta)

        return Norm_Train_Xs, Norm_Test_Xs

    def predict(self, Test_Xs) -> np.ndarray:
        """
        Predict a value or class for every row of ``Test_Xs``.

        One LLM call is issued per row, so the cost of a prediction is linear
        in the size of the query batch.

        Returns
        -------
        ndarray
            Flat array of length ``len(Test_Xs)``.
        """
        if self.Train_Xs is None or self.Train_ys is None:
            raise Exception("Train_Xs and Train_ys are None. Please call fit() first.")

        if not isinstance(Test_Xs, np.ndarray):
            Test_Xs = np.array(Test_Xs)
        if Test_Xs.ndim <= 1:
            Test_Xs = Test_Xs.reshape(1, -1)

        Norm_Train_Xs, Norm_Test_Xs = self.normalize(self.Train_Xs, Test_Xs)

        if isinstance(self, LLM_Regression):
            Train_ys = self.normalize_ys(self.Train_ys)
        else:
            Train_ys = self.Train_ys

        self.fit_prompts = self.generate_fit_prompts(Norm_Train_Xs, Train_ys)

        if self.parallel and self.parallel > 1:
            res = self._predict_parallel(Norm_Test_Xs)
        else:
            res = self._predict_serial(Norm_Test_Xs)

        if isinstance(self, LLM_Regression):
            res = np.array(res) * (self.ys_max - self.ys_min) + self.ys_min
        return np.array(res).flatten()

    def _render(self, X) -> str:
        """Render the full prompt for one query point."""
        prompts = self.generate_predict_prompts(X)
        final_prompt = " ".join(prompts)
        if self.show_prompt:
            print(final_prompt)
        return final_prompt

    def _predict_serial(self, Norm_Test_Xs) -> list:
        res = []
        for X in tqdm.tqdm(Norm_Test_Xs, disable=not self.show_progress, leave=False):
            res.append(self.call_llm(self._render(X)))
        return res

    def _predict_parallel(self, Norm_Test_Xs) -> list:
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            res = list(tqdm.tqdm(
                executor.map(lambda X: self.call_llm(self._render(X)), Norm_Test_Xs),
                total=len(Norm_Test_Xs), disable=not self.show_progress, leave=False
            ))
        return res

    def _fallback_rng(self, prompt: str) -> random.Random:
        """
        Build the RNG used when every attempt at a query failed.

        Seeding from the query rather than sharing one sequential generator
        keeps a fallback value tied to the question that produced it. A shared
        generator would hand out values in thread-completion order, which makes
        a run with ``parallel > 1`` depend on scheduling and stops replay from
        being exact.
        """
        digest = hashlib.sha256(f"{self.seed}|{self.role}|{prompt}".encode('utf-8')).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _query(self, prompt: str, attempt: int = 0) -> str:
        """
        Issue one request and return the raw text.

        ``attempt`` is forwarded to the client so that each retry is a separate
        cache entry; otherwise a retry would be served the cached failure it is
        trying to get away from. Budget exhaustion and replay cache misses are
        control-flow signals and are allowed to propagate.
        """
        response = self.client.chat(prompt, attempt=attempt)
        if self.show_response:
            print(response.text)
        return response.text

    def call_llm(self, prompt: str):
        raise NotImplementedError("call_llm method not implemented")

    def generate_fit_prompts(self, Xs, ys) -> List[str]:
        raise NotImplementedError("generate_fit_prompts method not implemented")

    def generate_predict_prompts(self, X) -> List[str]:
        raise NotImplementedError("generate_predict_prompts method not implemented")


class LLM_Classification(LLM_Base):
    """
    Ask the LLM whether a candidate is ``better`` or ``worse``.

    Predictions are +1 for ``better`` and -1 for ``worse``; an unparseable
    response after ``max_retries`` attempts falls back to a uniform choice
    between the two.
    """

    role = 'classification'

    def generate_fit_prompts(self, Xs, ys) -> List[str]:
        prompts = [self.introduction]
        for row, label in zip(Xs, ys):
            if label == 1:
                p = f"Features: <{', '.join(map(str, row))}>, Class: better\n"
            else:
                p = f"Features: <{', '.join(map(str, row))}>, Class: worse\n"
            prompts.append(p)
        return prompts

    def generate_predict_prompts(self, X) -> List[str]:
        if self.fit_prompts is None:
            raise Exception("fit_prompts is None. Please call generate_fit_prompts() first.")

        prompts = copy.deepcopy(self.fit_prompts)
        prompts.append("\n\nNew Evaluation:\n")
        prompts.append(f"<{', '.join(map(str, X))}>  better or worse?")
        prompts.append("\n\nNote: Respond in Json with the format {'Class':'result'} only.")
        return prompts

    def call_llm(self, prompt: str) -> int:
        self.client.budget.record_prediction(self.role)
        for attempt in range(self.max_retries):
            raw_res = self._query(prompt, attempt)
            if not raw_res:
                continue

            json_str = raw_res.strip()
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'\s+', ' ', json_str)

            match = re.search(r'\{.*?\}', json_str)
            if match:
                json_part = match.group(0).replace("'", '"')
                try:
                    data = json.loads(json_part)
                    label = str(data['Class']).lower()
                except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                    self.client.budget.record_parse_failure(self.role)
                    continue
                if label == 'better':
                    return 1
                if label == 'worse':
                    return -1
            self.client.budget.record_parse_failure(self.role)

        self.client.budget.record_fallback(self.role)
        return self._fallback_rng(prompt).choice([-1, 1])


class LLM_Regression(LLM_Base):
    """
    Ask the LLM for the objective value of a candidate.

    Objective values shown as examples are min-max scaled to [0, 1] and the
    prediction is mapped back, so the model only ever sees numbers in a fixed
    range. An unparseable response after ``max_retries`` attempts falls back to
    a uniform draw in [0, 1) before that inverse mapping.

    Parameters
    ----------
    strict_source : bool, optional
        When True, accept only the ``Value`` JSON key, reproducing the
        reference parser (default: False, which also accepts ``Target``).
    """

    role = 'regression'

    def __init__(self, *args, strict_source: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.strict_source = strict_source
        self.ys_max = None
        self.ys_min = None

    @property
    def _keys(self):
        return ('Value',) if self.strict_source else ('Value', 'Target')

    def normalize_ys(self, ys):
        """Min-max scale objective values to [0, 1] and remember the range."""
        self.ys_max = np.max(ys)
        self.ys_min = np.min(ys)
        return (ys - self.ys_min) / (self.ys_max - self.ys_min)

    def generate_fit_prompts(self, Xs, ys) -> List[str]:
        prompts = [self.introduction]
        for row, value in zip(Xs, ys):
            value = round(float(value), 5)
            p = f"Features: <{', '.join(map(str, row))}> Value: {value}\n"
            prompts.append(p)
        return prompts

    def generate_predict_prompts(self, X) -> List[str]:
        if self.fit_prompts is None:
            raise Exception("fit_prompts is None. Please call generate_fit_prompts() first.")

        prompts = copy.deepcopy(self.fit_prompts)
        prompts.append("\n\nNew Evaluation:\n")
        prompts.append(f"<{', '.join(map(str, X))}>  Target?")
        prompts.append("\n\nNote: Respond in Json with the format {'Target':'result'} only.")
        return prompts

    def call_llm(self, prompt: str) -> float:
        self.client.budget.record_prediction(self.role)
        for attempt in range(self.max_retries):
            raw_res = self._query(prompt, attempt)
            if not raw_res:
                continue

            # The reference matches against the unprocessed response here,
            # unlike the classification branch which matches the cleaned string.
            match = re.search(r'\{.*?\}', raw_res, re.DOTALL)
            if match:
                json_part = match.group(0).replace("'", '"')
                try:
                    data = json.loads(json_part)
                except (json.JSONDecodeError, ValueError):
                    self.client.budget.record_parse_failure(self.role)
                    continue
                for key in self._keys:
                    if key in data:
                        try:
                            return float(data[key])
                        except (ValueError, TypeError):
                            break
            self.client.budget.record_parse_failure(self.role)

        self.client.budget.record_fallback(self.role)
        return self._fallback_rng(prompt).random()

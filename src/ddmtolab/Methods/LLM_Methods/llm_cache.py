"""
Content-addressed disk cache for LLM responses.

Setting ``temperature=0`` and a seed does not make a hosted LLM deterministic:
server-side batching, floating point non-associativity and silent model updates
all break bit-exactness. A cache is therefore the only practical guarantee that
an LLM-assisted run can be reproduced, which is why an experiment's cache file
should be archived next to its ``.pkl``.

The cache key covers everything that can change a response: model id, the
rendered prompt, and the generation parameters.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.08.03
Version: 1.0
"""
import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional


def make_cache_key(model: str, prompt: str, gen_params: Dict[str, Any],
                   attempt: int = 0) -> str:
    """
    Build the content-addressed key for one request.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``'deepseek-chat'``.
    prompt : str
        The fully rendered prompt sent to the model.
    gen_params : dict
        Generation parameters that affect the response (temperature,
        max_tokens, seed, ...).
    attempt : int, optional
        Retry index. It belongs in the key because a caller that retries an
        unparseable response resends an identical prompt; without it the retry
        would be served the cached failure and could never succeed
        (default: 0).

    Returns
    -------
    str
        Hex sha256 digest.
    """
    payload = json.dumps(
        {'model': model, 'prompt': prompt, 'gen': gen_params, 'attempt': attempt},
        sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class LLMCache:
    """
    Append-only JSONL cache, loaded into memory on construction.

    Parameters
    ----------
    path : str, optional
        JSONL file backing the cache. ``None`` disables persistence and keeps
        the cache in memory only (default: None).
    read_only : bool, optional
        If True, never append new records to disk (default: False).
    """

    def __init__(self, path: Optional[str] = None, read_only: bool = False):
        self.path = path
        self.read_only = read_only
        self._lock = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}

        if path is not None and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        """Read every record of the backing file into memory."""
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get('key')
                if key is not None:
                    self._store[key] = record

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the cached record for ``key``, or None on a miss."""
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, record: Dict[str, Any]) -> None:
        """Store ``record`` under ``key`` and append it to the backing file."""
        record = dict(record)
        record['key'] = key
        with self._lock:
            self._store[key] = record
            if self.path is None or self.read_only:
                return
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

"""Semantic cache — exact hash + cosine similarity lookup with LRU eviction."""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from typing import Any


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """LRU cache with exact hash lookup and cosine similarity fallback.

    Args:
        max_size: Maximum number of entries before LRU eviction.
        ttl_seconds: Time-to-live for each entry in seconds.
        similarity_threshold: Minimum cosine similarity for a cache hit.
    """

    def __init__(
        self,
        max_size: int = 256,
        ttl_seconds: float = 300.0,
        similarity_threshold: float = 0.95,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        # key: query_hash -> (result, embedding, timestamp)
        self._store: OrderedDict[str, tuple[Any, list[float] | None, float]] = OrderedDict()

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str, embedding: list[float] | None = None) -> Any | None:
        """Look up a cached result.

        1. Exact hash match (fast, no embedding needed).
        2. Cosine similarity scan over all entries (if embedding provided).

        Returns the cached result or None on miss.
        """
        now = time.time()
        qhash = self._hash(query)

        # 1. Exact match
        if qhash in self._store:
            result, _emb, ts = self._store[qhash]
            if now - ts <= self._ttl:
                self._store.move_to_end(qhash)
                return result
            else:
                del self._store[qhash]

        # 2. Similarity scan
        if embedding:
            for key, (result, cached_emb, ts) in list(self._store.items()):
                if now - ts > self._ttl:
                    del self._store[key]
                    continue
                if cached_emb and _cosine_similarity(embedding, cached_emb) >= self._threshold:
                    self._store.move_to_end(key)
                    return result

        return None

    def put(self, query: str, result: Any, embedding: list[float] | None = None) -> None:
        """Store a result in the cache with LRU eviction."""
        qhash = self._hash(query)

        if qhash in self._store:
            del self._store[qhash]

        self._store[qhash] = (result, embedding, time.time())

        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in cache."""
        return len(self._store)

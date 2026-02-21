"""Tests for SemanticCache."""

import time

from opensearch_graphrag.cache import SemanticCache
from opensearch_graphrag.utils import cosine_similarity

# ── Exact hash hit/miss ──────────────────────────────────────


def test_exact_hit():
    """Same query returns cached result."""
    cache = SemanticCache()
    cache.put("What is OpenSearch?", {"answer": "A search engine"})
    result = cache.get("What is OpenSearch?")
    assert result == {"answer": "A search engine"}


def test_exact_miss():
    """Different query returns None."""
    cache = SemanticCache()
    cache.put("What is OpenSearch?", {"answer": "A search engine"})
    result = cache.get("What is Neo4j?")
    assert result is None


# ── TTL expiry ────────────────────────────────────────────────


def test_ttl_expiry():
    """Expired entries are not returned."""
    cache = SemanticCache(ttl_seconds=0.1)
    cache.put("query", "result")
    time.sleep(0.15)
    assert cache.get("query") is None


# ── LRU eviction ──────────────────────────────────────────────


def test_lru_eviction():
    """Oldest entries are evicted when max_size is exceeded."""
    cache = SemanticCache(max_size=2)
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    cache.put("q3", "r3")  # Should evict q1
    assert cache.get("q1") is None
    assert cache.get("q2") == "r2"
    assert cache.get("q3") == "r3"
    assert cache.size == 2


# ── Similarity hit ────────────────────────────────────────────


def test_similarity_hit():
    """Similar embeddings return cached result."""
    cache = SemanticCache(similarity_threshold=0.99)
    emb = [1.0, 0.0, 0.0]
    cache.put("query", "result", embedding=emb)
    # Slightly different embedding but still very similar
    similar_emb = [0.999, 0.001, 0.0]
    result = cache.get("different query", embedding=similar_emb)
    assert result == "result"


def test_similarity_miss():
    """Dissimilar embeddings don't match."""
    cache = SemanticCache(similarity_threshold=0.99)
    emb = [1.0, 0.0, 0.0]
    cache.put("query", "result", embedding=emb)
    dissimilar_emb = [0.0, 1.0, 0.0]
    result = cache.get("different query", embedding=dissimilar_emb)
    assert result is None


# ── Clear ─────────────────────────────────────────────────────


def test_clear():
    """Clear removes all entries."""
    cache = SemanticCache()
    cache.put("q1", "r1")
    cache.put("q2", "r2")
    assert cache.size == 2
    cache.clear()
    assert cache.size == 0
    assert cache.get("q1") is None


# ── Cosine similarity ─────────────────────────────────────────


def test_cosine_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_orthogonal_vectors():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9

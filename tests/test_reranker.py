"""Tests for cosine reranker module."""

from opensearch_graphrag.models import SearchResult
from opensearch_graphrag.reranker import cosine_similarity, rerank


def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == 1.0


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_cosine_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == -1.0


def test_cosine_empty():
    assert cosine_similarity([], []) == 0.0


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_different_lengths():
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_rerank_empty():
    assert rerank([], [1.0], {}) == []


def test_rerank_no_embeddings_passthrough():
    results = [
        SearchResult(chunk_id="c1", text="a", score=0.9),
        SearchResult(chunk_id="c2", text="b", score=0.5),
    ]
    out = rerank(results, [1.0, 0.0], {}, top_k=2)
    assert len(out) == 2
    assert out[0].chunk_id == "c1"


def test_rerank_with_embeddings():
    results = [
        SearchResult(chunk_id="c1", text="a", score=0.5),
        SearchResult(chunk_id="c2", text="b", score=0.9),
    ]
    query_emb = [1.0, 0.0]
    chunk_embs = {
        "c1": [0.9, 0.1],  # close to query
        "c2": [0.0, 1.0],  # orthogonal to query
    }
    out = rerank(results, query_emb, chunk_embs, top_k=2, alpha=0.8)
    # c1 should rank higher due to cosine similarity despite lower original score
    assert out[0].chunk_id == "c1"


def test_rerank_top_k_truncation():
    results = [
        SearchResult(chunk_id=f"c{i}", text=f"t{i}", score=float(i))
        for i in range(10)
    ]
    out = rerank(results, [1.0], {"c0": [1.0]}, top_k=3)
    assert len(out) == 3

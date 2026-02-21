"""Cosine reranker — rerank search results by cosine similarity to query."""

from __future__ import annotations

import logging
import math

from opensearch_graphrag.models import SearchResult

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (no numpy dependency)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank(
    results: list[SearchResult],
    query_embedding: list[float],
    chunk_embeddings: dict[str, list[float]],
    top_k: int = 5,
    alpha: float = 0.6,
) -> list[SearchResult]:
    """Rerank results by blending cosine similarity with original score.

    final_score = alpha * cosine_sim + (1-alpha) * normalized_original_score

    Args:
        results: Search results to rerank.
        query_embedding: Query vector for cosine comparison.
        chunk_embeddings: Map chunk_id -> embedding vector.
        top_k: Number of results to return.
        alpha: Weight for cosine similarity (0..1).

    Returns:
        Reranked list of SearchResult, truncated to top_k.
    """
    if not results:
        return []

    # If no embeddings available, return as-is (graceful passthrough)
    if not query_embedding or not chunk_embeddings:
        return results[:top_k]

    # Normalize original scores
    max_score = max(r.score for r in results) if results else 1.0
    if max_score <= 0:
        max_score = 1.0

    scored: list[tuple[float, SearchResult]] = []
    for r in results:
        emb = chunk_embeddings.get(r.chunk_id)
        if emb:
            cos_sim = cosine_similarity(query_embedding, emb)
        else:
            cos_sim = 0.0

        norm_original = r.score / max_score
        final = alpha * cos_sim + (1 - alpha) * norm_original

        scored.append((
            final,
            SearchResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=final,
                source=r.source,
                metadata=r.metadata,
            ),
        ))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]

"""Shared utility functions — cosine similarity and RRF fusion."""

from __future__ import annotations

import math

from opensearch_graphrag.models import SearchResult


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


def rrf_fuse(
    *result_lists: list[SearchResult],
    k: int = 60,
    top_k: int | None = None,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of multiple ranked result lists.

    RRF score = sum(weight_i / (k + rank + 1)) across all lists.
    When weights is None, all lists are weighted equally at 1.0.
    """
    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    for i, results in enumerate(result_lists):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, r in enumerate(results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + w / (k + rank + 1)
            if r.chunk_id not in best_result or r.score > best_result[r.chunk_id].score:
                best_result[r.chunk_id] = r

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    final_k = top_k or 5
    fused: list[SearchResult] = []
    for cid in sorted_ids[:final_k]:
        r = best_result[cid]
        fused.append(SearchResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=scores[cid],
            source=r.source,
            metadata=r.metadata,
        ))

    return fused

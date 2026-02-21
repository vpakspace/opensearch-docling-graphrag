"""Hybrid retriever: OpenSearch (BM25 + vector) + Neo4j graph traversal + RRF fusion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.models import SearchResult

if TYPE_CHECKING:
    from neo4j import Driver

    from opensearch_graphrag.opensearch_store import OpenSearchStore

logger = logging.getLogger(__name__)

SearchMode = Literal["bm25", "vector", "graph", "hybrid"]


def rrf_fuse(
    *result_lists: list[SearchResult],
    k: int = 60,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of multiple ranked result lists.

    RRF score = sum(1 / (k + rank)) across all lists.
    """
    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank + 1)
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


class Retriever:
    """Multi-mode retriever combining OpenSearch and Neo4j."""

    def __init__(
        self,
        store: "OpenSearchStore",
        neo4j_driver: "Driver | None" = None,
        settings=None,
    ) -> None:
        self._store = store
        self._driver = neo4j_driver
        self._cfg = settings or get_settings()

    def search(
        self,
        query: str,
        embedding: list[float] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[SearchResult]:
        """Search using the specified mode."""
        if mode == "bm25":
            return self._store.search_bm25(query, top_k=self._cfg.retrieval.top_k_bm25)

        if mode == "vector":
            if not embedding:
                logger.warning("Vector search requires embedding, falling back to BM25")
                return self._store.search_bm25(query, top_k=self._cfg.retrieval.top_k_bm25)
            return self._store.search_vector(embedding, top_k=self._cfg.retrieval.top_k_vector)

        if mode == "graph":
            return self._graph_search(query)

        # hybrid: fuse all three
        bm25_results = self._store.search_bm25(query, top_k=self._cfg.retrieval.top_k_bm25)
        vector_results = (
            self._store.search_vector(embedding, top_k=self._cfg.retrieval.top_k_vector)
            if embedding
            else []
        )
        graph_results = self._graph_search(query)

        return rrf_fuse(
            bm25_results, vector_results, graph_results,
            top_k=self._cfg.retrieval.top_k_final,
        )

    def _graph_search(self, query: str) -> list[SearchResult]:
        """Search Neo4j: find entities matching query, then traverse to chunks."""
        if not self._driver:
            return []

        top_k = self._cfg.retrieval.top_k_graph
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($search_term)
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
        RETURN c.id AS chunk_id, c.text AS text, e.name AS entity_name
        LIMIT $limit
        """
        try:
            with self._driver.session() as session:
                result = session.run(cypher, search_term=query, limit=top_k)
                results: list[SearchResult] = []
                for record in result:
                    results.append(SearchResult(
                        chunk_id=record["chunk_id"] or "",
                        text=record["text"] or "",
                        score=0.8,  # fixed score for graph results
                        source="graph",
                        metadata={"entity": record["entity_name"] or ""},
                    ))
                return results
        except Exception as e:
            logger.error("Graph search failed: %s", e)
            return []

"""Hybrid retriever: OpenSearch (BM25 + vector) + Neo4j graph traversal + RRF fusion."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.models import SearchResult
from opensearch_graphrag.query_expander import build_expanded_query, expand_query
from opensearch_graphrag.reranker import rerank

if TYPE_CHECKING:
    from neo4j import Driver

    from opensearch_graphrag.opensearch_store import OpenSearchStore

logger = logging.getLogger(__name__)

SearchMode = Literal["bm25", "vector", "graph", "hybrid", "enhanced"]


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

        if mode == "enhanced":
            return self.enhanced_search(query, embedding)

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

    def enhanced_search(
        self,
        query: str,
        embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Enhanced search: query expansion + 3x candidates + RRF + cosine rerank.

        1. Expand query (themes, entities, related terms)
        2. Retrieve 3x candidates: BM25(expanded) + vector + graph(expanded)
        3. RRF fusion on all candidates
        4. Cosine rerank top results
        """
        expansion = expand_query(query, settings=self._cfg)
        expanded_query = build_expanded_query(query, expansion)
        triple_k = self._cfg.retrieval.top_k_bm25 * 3

        # BM25 with expanded query
        bm25_results = self._store.search_bm25(expanded_query, top_k=triple_k)

        # Vector search
        vector_results = (
            self._store.search_vector(embedding, top_k=triple_k)
            if embedding
            else []
        )

        # Graph search with expanded entities
        entities = expansion.get("entities", [])
        graph_query = " ".join(entities) if entities else query
        graph_results = self._graph_search(graph_query)

        # RRF fusion on all 3x candidate lists
        fused = rrf_fuse(
            bm25_results, vector_results, graph_results,
            top_k=self._cfg.retrieval.top_k_final * 2,
        )

        # Cosine rerank if we have embeddings
        if embedding and fused:
            chunk_ids = [r.chunk_id for r in fused]
            chunk_embeddings = self._store.get_embeddings(chunk_ids)
            fused = rerank(
                fused,
                query_embedding=embedding,
                chunk_embeddings=chunk_embeddings,
                top_k=self._cfg.retrieval.top_k_final,
            )

        return fused[:self._cfg.retrieval.top_k_final]

    @staticmethod
    def _extract_keywords(query: str, min_len: int = 3) -> list[str]:
        """Extract significant keywords from query for entity matching."""
        stop_words = {
            # Russian
            "что", "как", "какие", "какой", "какая", "какое", "где", "кто",
            "когда", "почему", "зачем", "для", "чего", "это", "его", "она",
            "они", "все", "при", "или", "так", "уже", "еще", "без",
            "между", "через", "после", "перед", "также", "которые", "который",
            "которая", "которое", "можно", "нужно", "есть", "был", "были",
            "быть", "будет", "если", "того", "этого", "этой", "этих",
            "более", "менее", "чем", "каких", "каким", "какими",
            "статье", "тексте", "документе", "опиши", "объясни",
            "перечисли", "резюмируй", "расскажи", "назови",
            # English
            "what", "how", "which", "where", "who", "when", "why",
            "the", "and", "for", "are", "was", "were", "been", "being",
            "have", "has", "had", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "with", "from", "into", "about", "between", "through", "after",
            "before", "also", "each", "other", "some", "such", "than",
            "its", "them", "then", "there", "their", "they", "not", "but",
            "describe", "explain", "list", "summarize", "article", "text",
        }
        words = re.findall(r"\b\w+\b", query.lower())
        return [w for w in words if len(w) >= min_len and w not in stop_words]

    def _graph_search(self, query: str) -> list[SearchResult]:
        """Search Neo4j: find entities matching query keywords, then traverse to chunks."""
        if not self._driver:
            return []

        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        top_k = self._cfg.retrieval.top_k_graph

        # Build regex pattern: match any keyword in entity name
        pattern = "|".join(re.escape(kw) for kw in keywords)
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) =~ $pattern
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
        RETURN DISTINCT c.id AS chunk_id, c.text AS text,
               collect(DISTINCT e.name) AS entities
        LIMIT $limit
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    cypher,
                    pattern=f".*({pattern}).*",
                    limit=top_k,
                )
                results: list[SearchResult] = []
                for record in result:
                    entities = record["entities"] or []
                    results.append(SearchResult(
                        chunk_id=record["chunk_id"] or "",
                        text=record["text"] or "",
                        score=min(0.5 + 0.1 * len(entities), 1.0),
                        source="graph",
                        metadata={"entities": ", ".join(entities)},
                    ))
                return results
        except Exception as e:
            logger.error("Graph search failed: %s", e)
            return []

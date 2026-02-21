"""Cognitive two-stage retriever inspired by Cog-RAG (AAAI 2026).

Stage 1 (Theme): expand query → theme keywords → BM25(themes) + vector → RRF
Stage 2 (Entity): entity keywords → BM25(entities) + graph_search → RRF
Merge: RRF(stage1, stage2) → cosine rerank → top_k_final
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.models import SearchResult
from opensearch_graphrag.query_expander import expand_query
from opensearch_graphrag.reranker import rerank
from opensearch_graphrag.retriever import Retriever
from opensearch_graphrag.utils import rrf_fuse

if TYPE_CHECKING:
    from neo4j import Driver

    from opensearch_graphrag.opensearch_store import OpenSearchStore

logger = logging.getLogger(__name__)


class CognitiveRetriever:
    """Two-stage cognitive retriever: Theme → Entity → Merge → Rerank."""

    def __init__(
        self,
        store: "OpenSearchStore",
        neo4j_driver: "Driver | None" = None,
        settings=None,
    ) -> None:
        self._store = store
        self._driver = neo4j_driver
        self._cfg = settings or get_settings()
        self._base_retriever = Retriever(
            store=store, neo4j_driver=neo4j_driver, settings=self._cfg,
        )

    def search(
        self,
        query: str,
        embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Two-stage cognitive retrieval.

        Stage 1: Theme-based (BM25 + vector)
        Stage 2: Entity-based (BM25 + graph)
        Merge via RRF, then cosine rerank.
        """
        expansion = expand_query(query, settings=self._cfg)
        top_k = self._cfg.retrieval.top_k_bm25

        # ── Stage 1: Theme retrieval ─────────────────────────────
        themes = expansion.get("themes", [])
        theme_query = " ".join(themes) if themes else query

        theme_bm25 = self._store.search_bm25(theme_query, top_k=top_k)
        theme_vector = (
            self._store.search_vector(embedding, top_k=top_k)
            if embedding
            else []
        )
        stage1 = rrf_fuse(theme_bm25, theme_vector, top_k=top_k)

        # ── Stage 2: Entity retrieval ────────────────────────────
        entities = expansion.get("entities", [])
        entity_query = " ".join(entities) if entities else query

        entity_bm25 = self._store.search_bm25(entity_query, top_k=top_k)
        entity_graph = self._entity_graph_search(entities) if entities else []
        stage2 = rrf_fuse(entity_bm25, entity_graph, top_k=top_k)

        # ── Merge stages via RRF ─────────────────────────────────
        merged = rrf_fuse(stage1, stage2, top_k=self._cfg.retrieval.top_k_final * 2)

        # ── Cosine rerank ────────────────────────────────────────
        if embedding and merged:
            chunk_ids = [r.chunk_id for r in merged]
            chunk_embeddings = self._store.get_embeddings(chunk_ids)
            merged = rerank(
                merged,
                query_embedding=embedding,
                chunk_embeddings=chunk_embeddings,
                top_k=self._cfg.retrieval.top_k_final,
            )

        return merged[:self._cfg.retrieval.top_k_final]

    def _entity_graph_search(self, entities: list[str]) -> list[SearchResult]:
        """Search Neo4j by entity names using regex match."""
        if not self._driver or not entities:
            return []

        pattern = "|".join(re.escape(e) for e in entities)
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
                    pattern=f".*({pattern.lower()}).*",
                    limit=self._cfg.retrieval.top_k_graph,
                )
                results: list[SearchResult] = []
                for record in result:
                    ent_list = record["entities"] or []
                    results.append(SearchResult(
                        chunk_id=record["chunk_id"] or "",
                        text=record["text"] or "",
                        score=min(0.5 + 0.1 * len(ent_list), 1.0),
                        source="graph",
                        metadata={"entities": ", ".join(ent_list)},
                    ))
                return results
        except Exception as e:
            logger.error("Cognitive graph search failed: %s", e)
            return []

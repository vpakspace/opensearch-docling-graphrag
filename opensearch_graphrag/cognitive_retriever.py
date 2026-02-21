"""Cognitive two-stage retriever inspired by Cog-RAG (AAAI 2026).

Stage 1 (Theme): expand query → theme keywords → BM25(themes) + vector → RRF
Stage 2 (Entity): entity keywords → BM25(entities) + graph_search → RRF
Merge: RRF(stage1, stage2) → cosine rerank → top_k_final

Iterative probing (ComoRAG-inspired): after initial retrieval, evaluate
evidence sufficiency; if insufficient, generate a refined probe query via
Ollama and search again (up to max_probes iterations).
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

# Stop-words excluded from keyword overlap computation
_STOP_WORDS = frozenset(
    "a an the is are was were be been being do does did will would shall "
    "should can could may might must have has had having in on at to for "
    "of with by from and or not no nor but if then so than that this "
    "what which who whom how when where why all each every".split()
)


def _content_words(text: str) -> set[str]:
    """Extract lowered content-words (no stop-words, len >= 2)."""
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
    return {w for w in tokens if len(w) >= 2 and w not in _STOP_WORDS}


class CognitiveRetriever:
    """Two-stage cognitive retriever: Theme → Entity → Merge → Rerank.

    After initial retrieval, iterative probing refines the search if
    evidence is insufficient (up to ``max_probes`` iterations).
    """

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
        self._last_probes_used: int = 0

    @property
    def last_probes_used(self) -> int:
        """Number of probes executed in the most recent ``search()`` call."""
        return self._last_probes_used

    def search(
        self,
        query: str,
        embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Two-stage cognitive retrieval with iterative probing.

        Stage 1: Theme-based (BM25 + vector)
        Stage 2: Entity-based (BM25 + graph)
        Merge via RRF, then iterative probing if evidence is weak,
        then cosine rerank.
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
        accumulated = rrf_fuse(
            stage1, stage2, top_k=self._cfg.retrieval.top_k_final * 2,
        )

        # ── Iterative probing ────────────────────────────────────
        max_probes = self._cfg.retrieval.max_probes
        threshold = self._cfg.retrieval.evidence_score_threshold
        probes_used = 0

        for _ in range(max_probes):
            score = self._check_evidence_sufficiency(query, accumulated)
            if score >= threshold:
                break
            probe = self._generate_probe_query(query, accumulated)
            if not probe:
                break
            probe_results = self._probe_search(probe, embedding)
            if not probe_results:
                break
            accumulated = rrf_fuse(
                accumulated, probe_results,
                top_k=self._cfg.retrieval.top_k_final * 2,
            )
            probes_used += 1

        self._last_probes_used = probes_used

        # ── Cosine rerank ────────────────────────────────────────
        if embedding and accumulated:
            chunk_ids = [r.chunk_id for r in accumulated]
            chunk_embeddings = self._store.get_embeddings(chunk_ids)
            accumulated = rerank(
                accumulated,
                query_embedding=embedding,
                chunk_embeddings=chunk_embeddings,
                top_k=self._cfg.retrieval.top_k_final,
            )

        return accumulated[:self._cfg.retrieval.top_k_final]

    # ── Evidence sufficiency heuristic ───────────────────────

    def _check_evidence_sufficiency(
        self,
        query: str,
        results: list[SearchResult],
    ) -> float:
        """Heuristic evidence score (0.0–1.0), no LLM call.

        Components (weighted sum):
        - Score quality (0.4): avg normalised score across results
        - Keyword coverage (0.4): query content-word overlap with result texts
        - Volume (0.2): len(results) / top_k_final
        """
        if not results:
            return 0.0

        top_k_final = self._cfg.retrieval.top_k_final

        # Score quality: normalize each result score to [0, 1]
        max_score = max(r.score for r in results)
        if max_score > 0:
            quality = sum(r.score / max_score for r in results) / len(results)
        else:
            quality = 0.0

        # Keyword coverage: fraction of query content-words found in results
        query_words = _content_words(query)
        if query_words:
            result_text = " ".join(r.text for r in results)
            result_words = _content_words(result_text)
            coverage = len(query_words & result_words) / len(query_words)
        else:
            coverage = 1.0  # no meaningful query words → assume covered

        # Volume: how close to expected number of results
        volume = min(len(results) / top_k_final, 1.0)

        return 0.4 * quality + 0.4 * coverage + 0.2 * volume

    # ── Probe query generation via Ollama ────────────────────

    def _generate_probe_query(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str | None:
        """Generate a refined probe query via Ollama /api/generate.

        Returns the probe string, or None on error / empty response.
        """
        snippets = "\n".join(
            f"- {r.text[:200]}" for r in results[:5]
        )
        prompt = (
            "You are a search refinement assistant. "
            "Given the original query and the snippets already retrieved, "
            "generate a SINGLE alternative search query that fills "
            "information gaps. Return ONLY the query, no explanation.\n\n"
            f"Original query: {query}\n\n"
            f"Retrieved snippets:\n{snippets}\n\n"
            "Refined query:"
        )

        try:
            from opensearch_graphrag.ollama_client import post_generate

            data = post_generate({
                "model": self._cfg.ollama.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            })
            response = data.get("response", "").strip()
            return response if response else None
        except Exception as e:
            logger.warning("Probe query generation failed: %s", e)
            return None

    # ── Probe search ─────────────────────────────────────────

    def _probe_search(
        self,
        probe_query: str,
        original_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Run BM25 + vector search using the probe query.

        Re-embeds the probe text for k-NN; falls back to BM25-only
        if embedding fails.
        """
        top_k = self._cfg.retrieval.top_k_bm25

        bm25_results = self._store.search_bm25(probe_query, top_k=top_k)

        vector_results: list[SearchResult] = []
        if original_embedding is not None:
            try:
                from opensearch_graphrag.embedder import embed_text

                probe_embedding = embed_text(probe_query, settings=self._cfg)
                vector_results = self._store.search_vector(
                    probe_embedding, top_k=top_k,
                )
            except Exception as e:
                logger.warning("Probe embedding failed, BM25-only: %s", e)

        return rrf_fuse(bm25_results, vector_results, top_k=top_k)

    # ── Entity graph search ──────────────────────────────────

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

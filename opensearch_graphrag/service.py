"""Pipeline service — typed contract for the RAG pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.embedder import embed_text
from opensearch_graphrag.generator import generate_answer
from opensearch_graphrag.models import QAResult, SearchResult
from opensearch_graphrag.retriever import Retriever

if TYPE_CHECKING:
    from neo4j import Driver

    from opensearch_graphrag.graph_builder import GraphBuilder
    from opensearch_graphrag.opensearch_store import OpenSearchStore

logger = logging.getLogger(__name__)

VALID_MODES = ("bm25", "vector", "graph", "hybrid")


class PipelineService:
    """Orchestrates the full RAG pipeline: embed → search → generate."""

    def __init__(
        self,
        store: "OpenSearchStore",
        neo4j_driver: "Driver | None" = None,
        graph_builder: "GraphBuilder | None" = None,
        settings=None,
    ) -> None:
        self._store = store
        self._driver = neo4j_driver
        self._graph_builder = graph_builder
        self._cfg = settings or get_settings()
        self._retriever = Retriever(
            store=store,
            neo4j_driver=neo4j_driver,
            settings=self._cfg,
        )

    def query(self, text: str, mode: str = "hybrid") -> QAResult:
        """Full RAG pipeline: embed query → retrieve → generate answer."""
        if mode not in VALID_MODES:
            mode = "hybrid"

        embedding = None
        if mode in ("vector", "hybrid"):
            try:
                embedding = embed_text(text, settings=self._cfg)
            except Exception as e:
                logger.warning("Embedding failed, falling back to BM25: %s", e)
                if mode == "vector":
                    mode = "bm25"

        results = self._retriever.search(text, embedding=embedding, mode=mode)
        return generate_answer(text, results, mode=mode, settings=self._cfg)

    def search(self, text: str, mode: str = "hybrid") -> list[SearchResult]:
        """Search only (no generation)."""
        if mode not in VALID_MODES:
            mode = "hybrid"

        embedding = None
        if mode in ("vector", "hybrid"):
            try:
                embedding = embed_text(text, settings=self._cfg)
            except Exception:
                if mode == "vector":
                    mode = "bm25"

        return self._retriever.search(text, embedding=embedding, mode=mode)

    def health(self) -> dict:
        """Check health of all components."""
        status = {"status": "ok", "opensearch": False, "neo4j": False, "ollama": False}

        try:
            status["opensearch"] = self._store.count() >= 0
        except Exception:
            pass

        if self._driver:
            try:
                with self._driver.session() as session:
                    session.run("RETURN 1")
                status["neo4j"] = True
            except Exception:
                pass

        try:
            import httpx

            resp = httpx.get(f"{self._cfg.ollama.base_url}/api/tags", timeout=5.0)
            status["ollama"] = resp.status_code == 200
        except Exception:
            pass

        if not all([status["opensearch"], status["ollama"]]):
            status["status"] = "degraded"

        return status

    def graph_stats(self) -> dict:
        """Get knowledge graph statistics."""
        if not self._driver:
            return {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}

        stats: dict[str, int] = {}
        try:
            with self._driver.session() as session:
                for label, key in [("Document", "documents"), ("Chunk", "chunks"), ("Entity", "entities")]:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                    record = result.single()
                    stats[key] = record["c"] if record else 0

                result = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
                record = result.single()
                stats["relationships"] = record["c"] if record else 0
        except Exception as e:
            logger.error("Failed to get graph stats: %s", e)
            stats = {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}

        return stats

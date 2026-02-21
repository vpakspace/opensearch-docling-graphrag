"""Pipeline service — typed contract for the RAG pipeline."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from opensearch_graphrag.cognitive_retriever import CognitiveRetriever
from opensearch_graphrag.config import get_settings
from opensearch_graphrag.embedder import embed_text
from opensearch_graphrag.exceptions import ValidationError
from opensearch_graphrag.generator import generate_answer
from opensearch_graphrag.models import QAResult, SearchResult
from opensearch_graphrag.retriever import Retriever

if TYPE_CHECKING:
    from neo4j import Driver

    from opensearch_graphrag.graph_builder import GraphBuilder
    from opensearch_graphrag.opensearch_store import OpenSearchStore

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 10000

VALID_MODES = ("bm25", "vector", "graph", "hybrid", "enhanced", "cognitive")


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
        self._cognitive = CognitiveRetriever(
            store=store,
            neo4j_driver=neo4j_driver,
            settings=self._cfg,
        )

    def _validate_text(self, text: str) -> None:
        """Validate query text input."""
        if not text or not text.strip():
            raise ValidationError("Query text must not be empty.")
        if len(text) > MAX_QUERY_LENGTH:
            raise ValidationError(
                f"Query text exceeds maximum length ({MAX_QUERY_LENGTH} chars)."
            )

    def query(self, text: str, mode: str = "hybrid") -> QAResult:
        """Full RAG pipeline: embed query → retrieve → generate answer."""
        self._validate_text(text)
        if mode not in VALID_MODES:
            mode = "hybrid"

        t0 = time.time()
        logger.info("query start mode=%s len=%d", mode, len(text))

        embedding = None
        if mode in ("vector", "hybrid", "enhanced", "cognitive"):
            try:
                embedding = embed_text(text, settings=self._cfg)
            except Exception as e:
                logger.warning("Embedding failed, falling back to BM25: %s", e)
                if mode == "vector":
                    mode = "bm25"

        if mode == "cognitive":
            results = self._cognitive.search(text, embedding=embedding)
        else:
            results = self._retriever.search(text, embedding=embedding, mode=mode)
        qa = generate_answer(text, results, mode=mode, settings=self._cfg)

        latency = time.time() - t0
        logger.info(
            "query end mode=%s results=%d confidence=%.2f latency=%.2fs",
            mode,
            len(results),
            qa.confidence,
            latency,
        )
        return qa

    def search(self, text: str, mode: str = "hybrid") -> list[SearchResult]:
        """Search only (no generation)."""
        self._validate_text(text)
        if mode not in VALID_MODES:
            mode = "hybrid"

        embedding = None
        if mode in ("vector", "hybrid", "enhanced", "cognitive"):
            try:
                embedding = embed_text(text, settings=self._cfg)
            except Exception:
                if mode == "vector":
                    mode = "bm25"

        if mode == "cognitive":
            return self._cognitive.search(text, embedding=embedding)
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

"""OpenSearch store with k-NN vector index and BM25 hybrid search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.models import Chunk, SearchResult

if TYPE_CHECKING:
    from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)


def _make_client(settings=None) -> "OpenSearch":
    """Create OpenSearch client."""
    from opensearchpy import OpenSearch

    cfg = settings or get_settings()
    return OpenSearch(
        hosts=[{"host": cfg.opensearch.host, "port": cfg.opensearch.port}],
        use_ssl=False,
        verify_certs=False,
    )


class OpenSearchStore:
    """OpenSearch k-NN vector index with BM25 hybrid search."""

    def __init__(self, client: "OpenSearch | None" = None, settings=None) -> None:
        self._cfg = settings or get_settings()
        self._client = client or _make_client(self._cfg)
        self._index = self._cfg.opensearch.index
        self._dims = self._cfg.ollama.embed_dimensions

    def init_index(self) -> None:
        """Create k-NN index with HNSW cosine if it doesn't exist."""
        if self._client.indices.exists(index=self._index):
            logger.info("Index '%s' already exists", self._index)
            return

        body: dict[str, Any] = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self._dims,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 128, "m": 16},
                        },
                    },
                    "source": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": False},
                }
            },
        }
        self._client.indices.create(index=self._index, body=body)
        logger.info("Created k-NN index '%s' (%d dims)", self._index, self._dims)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Bulk index chunks. Returns number indexed."""
        if not chunks:
            return 0

        actions: list[dict] = []
        for chunk in chunks:
            actions.append({"index": {"_index": self._index, "_id": chunk.id}})
            doc: dict[str, Any] = {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            if chunk.embedding:
                doc["embedding"] = chunk.embedding
            actions.append(doc)

        body = "\n".join(_serialize(a) for a in actions) + "\n"
        resp = self._client.bulk(body=body, refresh=True)

        errors = resp.get("errors", False)
        if errors:
            failed = [item for item in resp["items"] if "error" in item.get("index", {})]
            logger.warning("Bulk indexing had %d errors", len(failed))

        count = len(chunks)
        logger.info("Indexed %d chunks into '%s'", count, self._index)
        return count

    def search_vector(self, embedding: list[float], top_k: int | None = None) -> list[SearchResult]:
        """k-NN vector similarity search."""
        k = top_k or self._cfg.retrieval.top_k_vector
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": k,
                    }
                }
            },
        }
        return self._execute_search(body)

    def search_bm25(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """BM25 full-text search."""
        k = top_k or self._cfg.retrieval.top_k_bm25
        body = {
            "size": k,
            "query": {
                "match": {
                    "text": {"query": query, "operator": "or"},
                }
            },
        }
        return self._execute_search(body)

    def search_hybrid(
        self, query: str, embedding: list[float], top_k: int | None = None
    ) -> list[SearchResult]:
        """Hybrid search: BM25 + k-NN via bool query."""
        k = top_k or self._cfg.retrieval.top_k_final
        body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"text": {"query": query, "operator": "or"}}},
                        {"knn": {"embedding": {"vector": embedding, "k": k}}},
                    ]
                }
            },
        }
        return self._execute_search(body)

    def delete_all(self) -> None:
        """Delete all documents in the index."""
        if self._client.indices.exists(index=self._index):
            self._client.delete_by_query(
                index=self._index,
                body={"query": {"match_all": {}}},
                refresh=True,
            )
            logger.info("Deleted all documents from '%s'", self._index)

    def count(self) -> int:
        """Count documents in the index."""
        if not self._client.indices.exists(index=self._index):
            return 0
        resp = self._client.count(index=self._index)
        return resp.get("count", 0)

    def get_embeddings(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Fetch stored embeddings for given chunk IDs.

        Returns a dict mapping chunk_id -> embedding vector.
        Missing or error chunks are silently skipped.
        """
        if not chunk_ids:
            return {}

        try:
            body = {
                "size": len(chunk_ids),
                "query": {"ids": {"values": chunk_ids}},
                "_source": ["embedding"],
            }
            resp = self._client.search(index=self._index, body=body)
        except Exception as e:
            logger.error("Failed to fetch embeddings: %s", e)
            return {}

        result: dict[str, list[float]] = {}
        for hit in resp.get("hits", {}).get("hits", []):
            emb = hit.get("_source", {}).get("embedding")
            if emb:
                result[hit["_id"]] = emb
        return result

    def _execute_search(self, body: dict) -> list[SearchResult]:
        """Execute search and convert to SearchResult list."""
        try:
            resp = self._client.search(index=self._index, body=body)
        except Exception as e:
            logger.error("OpenSearch query failed: %s", e)
            return []

        results: list[SearchResult] = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            results.append(SearchResult(
                chunk_id=hit["_id"],
                text=src.get("text", ""),
                score=hit.get("_score", 0.0),
                source=src.get("source", ""),
                metadata=src.get("metadata", {}),
            ))
        return results


def _serialize(obj: dict) -> str:
    """Serialize dict to JSON string for bulk API."""
    import json
    return json.dumps(obj)

"""Ollama embeddings via REST API (POST /api/embed)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from opensearch_graphrag.models import Chunk
from opensearch_graphrag.retry import with_retry

if TYPE_CHECKING:
    from opensearch_graphrag.config import Settings

logger = logging.getLogger(__name__)


@with_retry(max_retries=2, backoff_base=1.0)
def _post_embed(base_url: str, payload: dict) -> dict:
    """POST /api/embed with retry on transient errors."""
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        response = client.post("/api/embed", json=payload)
        response.raise_for_status()
        return response.json()


def embed_text(text: str, settings: "Settings | None" = None) -> list[float]:
    """Embed a single text string using Ollama /api/embed.

    Args:
        text: The text to embed.
        settings: Optional settings instance; falls back to get_settings().

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-2xx status.
        ValueError: If the response payload has no embeddings.
    """
    from opensearch_graphrag.config import get_settings

    cfg = settings or get_settings()

    payload = {
        "model": cfg.ollama.embed_model,
        "input": text,
    }

    logger.debug("Embedding single text with model=%s", cfg.ollama.embed_model)

    data = _post_embed(cfg.ollama.base_url, payload)
    embeddings: list[list[float]] = data.get("embeddings", [])
    if not embeddings:
        raise ValueError(f"Ollama returned no embeddings for model={cfg.ollama.embed_model!r}")

    vector = embeddings[0]
    logger.debug("Received embedding vector of dimension %d", len(vector))
    return vector


def embed_chunks(
    chunks: list[Chunk],
    settings: "Settings | None" = None,
) -> list[Chunk]:
    """Batch-embed a list of Chunk objects using Ollama /api/embed.

    All chunk texts are sent in a single request. Each chunk's
    ``embedding`` field is populated in-place (Pydantic v2 model_copy).

    Args:
        chunks: Chunks to embed. Empty list is returned unchanged.
        settings: Optional settings instance; falls back to get_settings().

    Returns:
        The same list with each chunk's ``embedding`` field filled.

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-2xx status.
        ValueError: If the number of returned embeddings doesn't match the input.
    """
    if not chunks:
        logger.debug("embed_chunks called with empty list — returning immediately")
        return chunks

    from opensearch_graphrag.config import get_settings

    cfg = settings or get_settings()

    texts = [chunk.text for chunk in chunks]
    payload = {
        "model": cfg.ollama.embed_model,
        "input": texts,
    }

    logger.debug(
        "Batch-embedding %d chunks with model=%s",
        len(chunks),
        cfg.ollama.embed_model,
    )

    data = _post_embed(cfg.ollama.base_url, payload)
    embeddings: list[list[float]] = data.get("embeddings", [])

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} embeddings from Ollama, got {len(embeddings)}"
        )

    result: list[Chunk] = []
    for chunk, vector in zip(chunks, embeddings):
        result.append(chunk.model_copy(update={"embedding": vector}))

    logger.debug("All %d chunks embedded (dim=%d)", len(result), len(embeddings[0]))
    return result

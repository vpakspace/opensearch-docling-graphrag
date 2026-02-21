"""Shared Ollama REST API helpers — POST /api/generate, /api/chat, /api/embed."""

from __future__ import annotations

from opensearch_graphrag.retry import with_retry


@with_retry(max_retries=2, backoff_base=1.0)
def post_generate(body: dict) -> dict:
    """POST /api/generate with retry on transient errors."""
    from opensearch_graphrag.config import get_ollama_client

    client = get_ollama_client()
    resp = client.post("/api/generate", json=body)
    resp.raise_for_status()
    return resp.json()


@with_retry(max_retries=2, backoff_base=1.0)
def post_chat(body: dict) -> dict:
    """POST /api/chat with retry on transient errors."""
    from opensearch_graphrag.config import get_ollama_client

    client = get_ollama_client()
    resp = client.post("/api/chat", json=body)
    resp.raise_for_status()
    return resp.json()


@with_retry(max_retries=2, backoff_base=1.0)
def post_embed(payload: dict) -> dict:
    """POST /api/embed with retry on transient errors."""
    from opensearch_graphrag.config import get_ollama_client

    client = get_ollama_client()
    resp = client.post("/api/embed", json=payload)
    resp.raise_for_status()
    return resp.json()

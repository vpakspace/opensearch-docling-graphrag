"""Query expansion via Ollama LLM — extract themes, entities, and expanded terms."""

from __future__ import annotations

import json
import logging

import httpx

from opensearch_graphrag.retry import with_retry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a query expansion system for information retrieval. "
    "Given a search query, extract:\n"
    '1. "themes" — 2-4 key topic keywords\n'
    '2. "entities" — named entities mentioned (people, orgs, places)\n'
    '3. "expanded" — 2-3 related terms/synonyms that could help retrieval\n\n'
    "Return ONLY a JSON object with these three arrays. No extra text.\n"
    'Example: {"themes": ["machine learning", "neural"], '
    '"entities": ["OpenAI"], "expanded": ["deep learning", "AI"]}'
)


@with_retry(max_retries=2, backoff_base=1.0)
def _post_generate(base_url: str, body: dict) -> dict:
    """POST /api/generate with retry."""
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        resp = client.post("/api/generate", json=body)
        resp.raise_for_status()
        return resp.json()


def expand_query(query: str, settings=None) -> dict:
    """Expand query into themes, entities, and related terms.

    Returns dict with keys: themes, entities, expanded (each a list of str).
    On any error, returns empty dict (graceful fallback).
    """
    if not query or not query.strip():
        return {}

    from opensearch_graphrag.config import get_settings

    cfg = settings or get_settings()

    body = {
        "model": cfg.ollama.llm_model,
        "system": _SYSTEM_PROMPT,
        "prompt": query,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        raw = _post_generate(cfg.ollama.base_url, body)
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
        return {}

    text = raw.get("response", "")
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Query expansion JSON parse error: %r", text[:200])
        return {}

    result: dict[str, list[str]] = {}
    for key in ("themes", "entities", "expanded"):
        val = parsed.get(key, [])
        if isinstance(val, list):
            result[key] = [str(v) for v in val if v]
        else:
            result[key] = []
    return result


def build_expanded_query(query: str, expansion: dict) -> str:
    """Build an expanded BM25 query string from original query and expansion.

    Concatenates original query with themes and expanded terms.
    """
    parts = [query]
    for key in ("themes", "expanded"):
        terms = expansion.get(key, [])
        if terms:
            parts.extend(terms)
    return " ".join(parts)

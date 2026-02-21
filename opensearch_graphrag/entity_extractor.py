"""NER via Ollama LLM — extract named entities from text chunks."""

from __future__ import annotations

import json
import logging

import httpx

from opensearch_graphrag.models import Entity
from opensearch_graphrag.retry import with_retry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a named-entity recognition system. "
    "Given a text passage, extract all named entities and return them as JSON. "
    "Use exactly this format:\n"
    '{"entities": [{"name": "<entity name>", "type": "<entity type>"}]}\n'
    "Allowed entity types: Person, Organization, Location, Date, Other. "
    "Return only the JSON object, no extra text."
)

_USER_TEMPLATE = "Extract named entities from the following text:\n\n{text}"


@with_retry(max_retries=2, backoff_base=1.0)
def _post_generate(base_url: str, body: dict) -> dict:
    """POST /api/generate with retry on transient errors."""
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        response = client.post("/api/generate", json=body)
        response.raise_for_status()
        return response.json()


def extract_entities(
    text: str,
    chunk_id: str = "",
    settings=None,
) -> list[Entity]:
    """Extract named entities from *text* using the configured Ollama LLM.

    Parameters
    ----------
    text:
        The text passage to analyse.
    chunk_id:
        Identifier of the source chunk; stored on every returned Entity.
    settings:
        Optional ``Settings`` instance.  When *None* the module-level
        ``get_settings()`` cache is used so callers do not need to pass
        settings explicitly.

    Returns
    -------
    list[Entity]
        Entities extracted by the LLM.  Returns an empty list when the
        text is blank, the LLM is unreachable, or the response cannot be
        parsed as valid JSON.
    """
    if not text or not text.strip():
        return []

    # Defer import so the module is usable even without a running Ollama.
    from opensearch_graphrag.config import get_settings

    cfg = settings or get_settings()

    prompt = _USER_TEMPLATE.format(text=text)
    request_body = {
        "model": cfg.ollama.llm_model,
        "system": _SYSTEM_PROMPT,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }

    try:
        raw_response = _post_generate(cfg.ollama.base_url, request_body)
    except httpx.HTTPError as exc:
        logger.warning("Ollama request failed: %s", exc)
        return []

    # Ollama non-streaming response wraps the model output in "response".
    generated_text: str = raw_response.get("response", "")
    if not generated_text:
        logger.warning("Empty 'response' field from Ollama for chunk_id=%r", chunk_id)
        return []

    try:
        payload = json.loads(generated_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not parse JSON from Ollama output for chunk_id=%r: %s — raw=%r",
            chunk_id,
            exc,
            generated_text[:200],
        )
        return []

    raw_entities = payload.get("entities", [])
    if not isinstance(raw_entities, list):
        logger.warning(
            "Unexpected 'entities' value type %s for chunk_id=%r",
            type(raw_entities).__name__,
            chunk_id,
        )
        return []

    entities: list[Entity] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        entity_type = item.get("type", "Other").strip()
        if not name:
            continue
        entities.append(
            Entity(name=name, type=entity_type, source_chunk_id=chunk_id)
        )

    logger.debug(
        "Extracted %d entities from chunk_id=%r", len(entities), chunk_id
    )
    return entities

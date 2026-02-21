"""Answer generation via Ollama LLM (POST /api/chat)."""

from __future__ import annotations

import logging

import httpx

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.models import QAResult, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below. If the context doesn't contain enough information,
say so clearly. Be concise and accurate.

Context:
{context}"""


def generate_answer(
    query: str,
    results: list[SearchResult],
    mode: str = "hybrid",
    settings=None,
) -> QAResult:
    """Generate answer using Ollama chat API with retrieved context.

    Args:
        query: User question.
        results: Retrieved search results as context.
        mode: Search mode used (for metadata).
        settings: Optional Settings override.

    Returns:
        QAResult with answer, confidence, and sources.
    """
    cfg = settings or get_settings()

    if not results:
        return QAResult(
            answer="No relevant context found to answer the question.",
            confidence=0.0,
            sources=[],
            mode=mode,
        )

    context = "\n\n---\n\n".join(
        f"[Source: {r.source or 'unknown'}]\n{r.text}" for r in results
    )
    system_msg = SYSTEM_PROMPT.format(context=context)

    try:
        with httpx.Client(base_url=cfg.ollama.base_url, timeout=120.0) as client:
            resp = client.post(
                "/api/chat",
                json={
                    "model": cfg.ollama.llm_model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": query},
                    ],
                    "stream": False,
                    "options": {"temperature": cfg.ollama.temperature},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            answer = "The model returned an empty response."

    except Exception as e:
        logger.error("Ollama chat failed: %s", e)
        answer = f"Error generating answer: {e}"

    avg_score = sum(r.score for r in results) / len(results) if results else 0.0
    confidence = max(0.1, min(1.0, avg_score))

    return QAResult(
        answer=answer,
        confidence=confidence,
        sources=results,
        mode=mode,
    )

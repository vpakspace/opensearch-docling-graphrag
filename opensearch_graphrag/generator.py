"""Answer generation via Ollama LLM (POST /api/chat)."""

from __future__ import annotations

import logging
import re

from opensearch_graphrag.config import get_settings
from opensearch_graphrag.hallucination_detector import detect_hallucination
from opensearch_graphrag.models import QAResult, SearchResult
from opensearch_graphrag.retry import with_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You MUST answer in the SAME language as the user's question.
If the user writes in Russian — answer in Russian. If in English — answer in English.
NEVER switch to Chinese unless the user writes in Chinese.

You are a helpful RAG assistant. Answer using ONLY the provided context.
If the context lacks information, say so. Be concise and accurate.

Context:
{context}"""


@with_retry(max_retries=2, backoff_base=1.0)
def _post_chat(base_url: str, body: dict) -> dict:
    """POST /api/chat with retry on transient errors."""
    from opensearch_graphrag.config import get_ollama_client

    client = get_ollama_client()
    resp = client.post("/api/chat", json=body)
    resp.raise_for_status()
    return resp.json()


def _calibrate_confidence(
    query: str,
    answer: str,
    results: list[SearchResult],
) -> float:
    """Multi-signal confidence calibration.

    Three signals weighted:
    - Score consistency (0.4): how consistent retrieval scores are.
    - Token overlap (0.4): content word overlap between answer and context.
    - Source diversity (0.2): how many unique sources contributed.
    """
    if not results:
        return 0.0

    # 1. Normalized score consistency (0..1)
    scores = [r.score for r in results]
    max_score = max(scores) if scores else 1.0
    if max_score > 0:
        norm_scores = [s / max_score for s in scores]
    else:
        norm_scores = [0.0] * len(scores)
    avg_norm = sum(norm_scores) / len(norm_scores)

    # 2. Token overlap: content words (4+ chars) shared between answer and context
    def _content_words(text: str) -> set[str]:
        return {w for w in re.findall(r"\b\w{4,}\b", text.lower())}

    answer_words = _content_words(answer)
    context_text = " ".join(r.text for r in results)
    context_words = _content_words(context_text)
    if answer_words:
        overlap = len(answer_words & context_words) / len(answer_words)
    else:
        overlap = 0.0

    # 3. Source diversity (unique sources / total results)
    unique_sources = {r.source for r in results if r.source}
    diversity = min(len(unique_sources) / max(len(results), 1), 1.0)

    raw = 0.4 * avg_norm + 0.4 * overlap + 0.2 * diversity
    return max(0.1, min(1.0, raw))


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
        data = _post_chat(
            cfg.ollama.base_url,
            {
                "model": cfg.ollama.llm_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query},
                ],
                "stream": False,
                "options": {"temperature": cfg.ollama.temperature},
            },
        )

        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            answer = "The model returned an empty response."

    except Exception as e:
        logger.error("Ollama chat failed: %s", e)
        answer = f"Error generating answer: {e}"

    confidence = _calibrate_confidence(query, answer, results)

    # Hallucination detection
    context_texts = [r.text for r in results]
    hal = detect_hallucination(answer, context_texts)

    return QAResult(
        answer=answer,
        confidence=confidence,
        sources=results,
        mode=mode,
        grounded=hal["grounded"],
        grounding_score=hal["overlap"],
        warning=hal["warning"],
    )

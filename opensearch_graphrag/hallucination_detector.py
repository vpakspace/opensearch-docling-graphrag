"""Lightweight hallucination detection via token overlap."""

from __future__ import annotations

import re


def detect_hallucination(
    answer: str,
    context_texts: list[str],
    threshold: float = 0.3,
) -> dict:
    """Detect potential hallucination by measuring content word overlap.

    Compares content words (4+ chars) in the answer against the combined context.

    Args:
        answer: Generated answer text.
        context_texts: List of context passages used for generation.
        threshold: Minimum overlap ratio to consider grounded.

    Returns:
        dict with keys:
        - grounded (bool): True if overlap >= threshold.
        - overlap (float): Ratio of answer words found in context.
        - warning (str): Warning message if not grounded, empty otherwise.
    """
    if not answer or not context_texts:
        return {"grounded": False, "overlap": 0.0, "warning": "No context available"}

    answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
    if not answer_words:
        return {"grounded": True, "overlap": 1.0, "warning": ""}

    context_combined = " ".join(context_texts)
    context_words = set(re.findall(r"\b\w{4,}\b", context_combined.lower()))

    overlap = len(answer_words & context_words) / len(answer_words)
    grounded = overlap >= threshold

    warning = ""
    if not grounded:
        warning = (
            f"Low grounding score ({overlap:.0%}): "
            "answer may not be fully supported by the provided context."
        )

    return {"grounded": grounded, "overlap": round(overlap, 3), "warning": warning}

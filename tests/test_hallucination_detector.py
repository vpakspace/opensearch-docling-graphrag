"""Tests for hallucination detector."""

from opensearch_graphrag.hallucination_detector import detect_hallucination


def test_grounded_high_overlap():
    answer = "OpenSearch is a search engine for indexing documents"
    context = ["OpenSearch is a search engine", "It indexes documents efficiently"]
    result = detect_hallucination(answer, context)
    assert result["grounded"] is True
    assert result["overlap"] > 0.3
    assert result["warning"] == ""


def test_not_grounded_no_overlap():
    answer = "Quantum computing revolutionizes cryptography"
    context = ["OpenSearch is a search engine"]
    result = detect_hallucination(answer, context)
    assert result["grounded"] is False
    assert result["overlap"] < 0.3
    assert "Low grounding" in result["warning"]


def test_empty_answer():
    result = detect_hallucination("", ["context"])
    assert result["grounded"] is False


def test_empty_context():
    result = detect_hallucination("some answer", [])
    assert result["grounded"] is False


def test_short_words_ignored():
    """Words shorter than 4 chars should not count."""
    answer = "a b c is the not"
    context = ["a b c is the not"]
    result = detect_hallucination(answer, context)
    # No content words → grounded by default
    assert result["grounded"] is True

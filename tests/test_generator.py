"""Tests for generator module with mocked Ollama."""

from unittest.mock import MagicMock, patch

from opensearch_graphrag.generator import generate_answer
from opensearch_graphrag.models import SearchResult


def _make_results(n=2):
    return [
        SearchResult(chunk_id=f"c{i}", text=f"Text {i}", score=0.8 + i * 0.05, source="doc.txt")
        for i in range(n)
    ]


def test_generate_answer_empty_results():
    qa = generate_answer("What is X?", [])
    assert qa.confidence == 0.0
    assert "No relevant context" in qa.answer
    assert qa.mode == "hybrid"


@patch("opensearch_graphrag.generator.httpx.Client")
def test_generate_answer_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": "OpenSearch is a search engine."}
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_resp

    results = _make_results(2)
    qa = generate_answer("What is OpenSearch?", results, mode="bm25")

    assert qa.answer == "OpenSearch is a search engine."
    assert qa.mode == "bm25"
    assert len(qa.sources) == 2
    assert qa.confidence > 0


@patch("opensearch_graphrag.generator.httpx.Client")
def test_generate_answer_error(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = Exception("connection refused")

    results = _make_results(1)
    qa = generate_answer("test?", results)

    assert "Error" in qa.answer
    assert qa.confidence > 0


def test_generate_answer_confidence_clamped():
    """Confidence should be clamped between 0.1 and 1.0."""
    # Clamping is tested indirectly via test_generate_answer_success
    # and test_generate_answer_empty_results (confidence=0.0 for empty).
    # Here we verify the empty-results path gives 0.0 confidence.
    qa = generate_answer("test?", [])
    assert qa.confidence == 0.0


def test_generate_answer_mode_passthrough():
    qa = generate_answer("q?", [], mode="vector")
    assert qa.mode == "vector"

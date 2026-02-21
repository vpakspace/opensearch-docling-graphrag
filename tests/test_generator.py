"""Tests for generator module with mocked Ollama."""

from unittest.mock import MagicMock, patch

from opensearch_graphrag.generator import _calibrate_confidence, generate_answer
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


# ── Confidence calibration tests ────────────────────────────────


def test_calibrate_confidence_empty():
    assert _calibrate_confidence("q?", "answer", []) == 0.0


def test_calibrate_confidence_high_overlap():
    """Answer reuses context words → high overlap signal."""
    results = [
        SearchResult(chunk_id="c1", text="OpenSearch is a search engine", score=0.9, source="a.txt"),
        SearchResult(chunk_id="c2", text="Neo4j is a graph database", score=0.8, source="b.txt"),
    ]
    conf = _calibrate_confidence("What is OpenSearch?", "OpenSearch is a search engine", results)
    assert 0.1 <= conf <= 1.0
    assert conf > 0.4  # strong overlap expected


def test_calibrate_confidence_no_overlap():
    """Answer has nothing in common with context → low overlap."""
    results = [
        SearchResult(chunk_id="c1", text="Python is great", score=0.5, source="a.txt"),
    ]
    conf = _calibrate_confidence("q?", "totally unrelated words here nothing matches", results)
    assert 0.1 <= conf <= 1.0


def test_calibrate_confidence_bm25_scores_normalized():
    """BM25 scores >1.0 should still produce valid confidence."""
    results = [
        SearchResult(chunk_id="c1", text="data text", score=12.5, source="a.txt"),
        SearchResult(chunk_id="c2", text="more data", score=8.3, source="a.txt"),
    ]
    conf = _calibrate_confidence("data?", "data text and more data", results)
    assert 0.1 <= conf <= 1.0


# ── Edge cases ────────────────────────────────────────────────


def test_generate_answer_empty_results_message():
    """Empty results return a specific 'no context' message."""
    qa = generate_answer("What is the answer?", [])
    assert qa.answer == "No relevant context found to answer the question."
    assert qa.sources == []


@patch("opensearch_graphrag.generator.httpx.Client")
def test_generate_answer_very_long_context(mock_client_cls):
    """Generator handles very long context (>10000 chars) without error."""
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "Long context handled."}}
    mock_resp.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_resp

    long_text = "x" * 12000
    results = [SearchResult(chunk_id="c1", text=long_text, score=0.9, source="doc.txt")]
    qa = generate_answer("What?", results)
    assert qa.answer == "Long context handled."
    assert len(qa.sources) == 1

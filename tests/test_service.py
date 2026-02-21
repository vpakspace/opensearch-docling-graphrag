"""Tests for PipelineService."""

from unittest.mock import MagicMock, patch

from opensearch_graphrag.models import QAResult, SearchResult
from opensearch_graphrag.service import PipelineService


def _make_service():
    store = MagicMock()
    store.count.return_value = 10
    store.search_bm25.return_value = [
        SearchResult(chunk_id="c1", text="hello", score=0.9, source="test.txt"),
    ]
    store.search_vector.return_value = [
        SearchResult(chunk_id="c2", text="world", score=0.85, source="test.txt"),
    ]
    return PipelineService(store=store)


@patch("opensearch_graphrag.service.generate_answer")
@patch("opensearch_graphrag.service.embed_text")
def test_query_hybrid(mock_embed, mock_gen):
    mock_embed.return_value = [0.1] * 768
    mock_gen.return_value = QAResult(answer="Test answer", confidence=0.9, mode="hybrid")

    svc = _make_service()
    result = svc.query("What is this?", mode="hybrid")

    assert result.answer == "Test answer"
    mock_embed.assert_called_once()
    mock_gen.assert_called_once()


@patch("opensearch_graphrag.service.embed_text")
def test_query_bm25_no_embedding(mock_embed):
    svc = _make_service()

    with patch("opensearch_graphrag.service.generate_answer") as mock_gen:
        mock_gen.return_value = QAResult(answer="BM25 answer", mode="bm25")
        result = svc.query("test", mode="bm25")

    assert result.mode == "bm25"
    mock_embed.assert_not_called()


def test_query_invalid_mode_defaults_hybrid():
    svc = _make_service()
    with patch("opensearch_graphrag.service.embed_text") as mock_embed, \
         patch("opensearch_graphrag.service.generate_answer") as mock_gen:
        mock_embed.return_value = [0.1] * 768
        mock_gen.return_value = QAResult(answer="ok", mode="hybrid")
        result = svc.query("test", mode="invalid_mode")
        assert result.mode == "hybrid"


@patch("opensearch_graphrag.service.embed_text")
def test_search_bm25(mock_embed):
    svc = _make_service()
    results = svc.search("test", mode="bm25")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    mock_embed.assert_not_called()


def test_health():
    svc = _make_service()
    h = svc.health()
    assert "status" in h
    assert "opensearch" in h
    assert h["opensearch"] is True


def test_graph_stats_no_driver():
    store = MagicMock()
    svc = PipelineService(store=store, neo4j_driver=None)
    stats = svc.graph_stats()
    assert stats["documents"] == 0
    assert stats["chunks"] == 0
    assert stats["entities"] == 0

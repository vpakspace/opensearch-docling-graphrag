"""Tests for PipelineService."""

from unittest.mock import MagicMock, patch

from opensearch_graphrag.models import QAResult, SearchResult
from opensearch_graphrag.service import PipelineService


def _make_service(**kwargs):
    store = MagicMock()
    store.count.return_value = 10
    store.search_bm25.return_value = [
        SearchResult(chunk_id="c1", text="hello", score=0.9, source="test.txt"),
    ]
    store.search_vector.return_value = [
        SearchResult(chunk_id="c2", text="world", score=0.85, source="test.txt"),
    ]
    return PipelineService(store=store, **kwargs)


# ── query() tests ─────────────────────────────────────────


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


@patch("opensearch_graphrag.service.generate_answer")
@patch("opensearch_graphrag.service.embed_text")
def test_query_cache_hit_returns_early(mock_embed, mock_gen):
    """Second identical query returns cached result without calling embed/generate."""
    mock_embed.return_value = [0.1] * 768
    mock_gen.return_value = QAResult(answer="First", confidence=0.9, mode="hybrid")

    svc = _make_service()
    svc.query("What is this?", mode="hybrid")

    # Reset mocks, second call should hit cache
    mock_embed.reset_mock()
    mock_gen.reset_mock()

    result = svc.query("What is this?", mode="hybrid")
    assert result.answer == "First"
    mock_embed.assert_not_called()
    mock_gen.assert_not_called()


@patch("opensearch_graphrag.service.generate_answer")
@patch("opensearch_graphrag.service.embed_text")
def test_query_cognitive_mode(mock_embed, mock_gen):
    """Cognitive mode routes to CognitiveRetriever.search()."""
    mock_embed.return_value = [0.1] * 768
    mock_gen.return_value = QAResult(answer="Cognitive answer", mode="cognitive")

    svc = _make_service()
    svc._cognitive = MagicMock()
    svc._cognitive.search.return_value = [
        SearchResult(chunk_id="c3", text="cognitive", score=0.8),
    ]

    result = svc.query("test query?", mode="cognitive")
    assert result.answer == "Cognitive answer"
    svc._cognitive.search.assert_called_once()


@patch("opensearch_graphrag.service.generate_answer")
@patch("opensearch_graphrag.service.embed_text")
def test_query_embedding_failure_fallback(mock_embed, mock_gen):
    """Embedding failure for vector mode falls back to bm25."""
    mock_embed.side_effect = RuntimeError("Ollama down")
    mock_gen.return_value = QAResult(answer="fallback", mode="bm25")

    svc = _make_service()
    result = svc.query("test", mode="vector")

    assert result.mode == "bm25"
    mock_gen.assert_called_once()


@patch("opensearch_graphrag.service.generate_answer")
@patch("opensearch_graphrag.service.embed_text")
def test_query_embedding_failure_hybrid_continues(mock_embed, mock_gen):
    """Embedding failure for hybrid mode continues without embedding (doesn't switch to bm25)."""
    mock_embed.side_effect = RuntimeError("Ollama down")
    mock_gen.return_value = QAResult(answer="partial", mode="hybrid")

    svc = _make_service()
    result = svc.query("test", mode="hybrid")

    # Hybrid mode continues (only vector→bm25 fallback triggers mode change)
    assert result.mode == "hybrid"


# ── search() tests ────────────────────────────────────────


@patch("opensearch_graphrag.service.embed_text")
def test_search_bm25(mock_embed):
    svc = _make_service()
    results = svc.search("test", mode="bm25")
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    mock_embed.assert_not_called()


@patch("opensearch_graphrag.service.embed_text")
def test_search_cognitive_mode(mock_embed):
    """Cognitive mode in search() routes to CognitiveRetriever."""
    mock_embed.return_value = [0.1] * 768

    svc = _make_service()
    svc._cognitive = MagicMock()
    svc._cognitive.search.return_value = [
        SearchResult(chunk_id="c5", text="cog", score=0.7),
    ]

    results = svc.search("test?", mode="cognitive")
    assert len(results) == 1
    assert results[0].chunk_id == "c5"
    svc._cognitive.search.assert_called_once()


@patch("opensearch_graphrag.service.embed_text")
def test_search_embedding_failure_vector_fallback(mock_embed):
    """Embedding failure in search() for vector mode falls back to bm25."""
    mock_embed.side_effect = RuntimeError("Ollama down")

    svc = _make_service()
    results = svc.search("test", mode="vector")
    # Falls back to bm25 — should get bm25 results
    assert len(results) >= 1


# ── health() tests ────────────────────────────────────────


def test_health():
    svc = _make_service()
    h = svc.health()
    assert "status" in h
    assert "opensearch" in h
    assert h["opensearch"] is True


def test_health_with_neo4j_driver():
    """Health check with a mock Neo4j driver succeeding."""
    driver = MagicMock()
    mock_session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    svc = _make_service(neo4j_driver=driver)
    h = svc.health()
    assert h["neo4j"] is True
    mock_session.run.assert_called_with("RETURN 1")


def test_health_neo4j_failure():
    """Neo4j driver failure is handled gracefully."""
    driver = MagicMock()
    driver.session.side_effect = Exception("Neo4j down")

    svc = _make_service(neo4j_driver=driver)
    h = svc.health()
    assert h["neo4j"] is False


def test_health_degraded_when_opensearch_down():
    """Status is 'degraded' when OpenSearch is down."""
    store = MagicMock()
    store.count.side_effect = Exception("Connection refused")

    svc = PipelineService(store=store)
    h = svc.health()
    assert h["opensearch"] is False
    assert h["status"] == "degraded"


# ── graph_stats() tests ──────────────────────────────────


def test_graph_stats_no_driver():
    store = MagicMock()
    svc = PipelineService(store=store, neo4j_driver=None)
    stats = svc.graph_stats()
    assert stats["documents"] == 0
    assert stats["chunks"] == 0
    assert stats["entities"] == 0


def test_graph_stats_with_graph_builder():
    """graph_stats delegates to GraphBuilder.get_stats() when available."""
    builder = MagicMock()
    builder.get_stats.return_value = {
        "documents": 3, "chunks": 30, "entities": 15, "relationships": 20,
    }

    svc = _make_service(graph_builder=builder)
    stats = svc.graph_stats()
    assert stats["documents"] == 3
    assert stats["entities"] == 15
    builder.get_stats.assert_called_once()


def test_graph_stats_graph_builder_exception():
    """GraphBuilder exception returns zeros."""
    builder = MagicMock()
    builder.get_stats.side_effect = Exception("Neo4j error")

    svc = _make_service(graph_builder=builder)
    stats = svc.graph_stats()
    assert stats == {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}


def test_graph_stats_with_driver_fallback():
    """Direct Cypher query fallback when graph_builder is None but driver is set."""
    driver = MagicMock()
    mock_session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    # Mock sequential calls to session.run()
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: 5
    mock_result = MagicMock()
    mock_result.single.return_value = mock_record
    mock_session.run.return_value = mock_result

    svc = _make_service(neo4j_driver=driver)
    stats = svc.graph_stats()
    assert stats["documents"] == 5
    assert stats["entities"] == 5
    assert stats["relationships"] == 5


def test_graph_stats_driver_exception():
    """Driver exception in fallback returns zeros."""
    driver = MagicMock()
    driver.session.side_effect = Exception("Connection lost")

    svc = _make_service(neo4j_driver=driver)
    stats = svc.graph_stats()
    assert stats == {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}


# ── validation tests ──────────────────────────────────────


def test_query_empty_text_raises():
    """Empty query text raises ValidationError."""
    import pytest

    from opensearch_graphrag.exceptions import ValidationError

    svc = _make_service()
    with pytest.raises(ValidationError, match="empty"):
        svc.query("")


def test_query_too_long_raises():
    """Query exceeding max length raises ValidationError."""
    import pytest

    from opensearch_graphrag.exceptions import ValidationError

    svc = _make_service()
    with pytest.raises(ValidationError, match="maximum length"):
        svc.query("x" * 10001)

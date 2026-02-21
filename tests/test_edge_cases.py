"""Edge case tests for input handling and robustness."""

from unittest.mock import MagicMock, patch

import pytest

from opensearch_graphrag.exceptions import ValidationError
from opensearch_graphrag.hallucination_detector import detect_hallucination
from opensearch_graphrag.models import Chunk, SearchResult
from opensearch_graphrag.service import PipelineService


def _make_service():
    store = MagicMock()
    store.count.return_value = 0
    store.search_bm25.return_value = []
    store.search_vector.return_value = []
    return PipelineService(store=store)


def test_empty_query_raises_validation():
    svc = _make_service()
    with pytest.raises(ValidationError, match="empty"):
        svc.query("")


def test_whitespace_query_raises_validation():
    svc = _make_service()
    with pytest.raises(ValidationError, match="empty"):
        svc.query("   ")


def test_too_long_query_raises_validation():
    svc = _make_service()
    with pytest.raises(ValidationError, match="maximum length"):
        svc.query("x" * 10001)


def test_search_empty_raises_validation():
    svc = _make_service()
    with pytest.raises(ValidationError, match="empty"):
        svc.search("")


def test_unicode_query_accepted():
    svc = _make_service()
    with patch("opensearch_graphrag.service.generate_answer") as mock_gen:
        from opensearch_graphrag.models import QAResult
        mock_gen.return_value = QAResult(answer="ответ", mode="bm25")
        result = svc.query("Что такое OpenSearch? 日本語テスト", mode="bm25")
        assert result.answer == "ответ"


def test_special_chars_in_query():
    svc = _make_service()
    with patch("opensearch_graphrag.service.generate_answer") as mock_gen:
        from opensearch_graphrag.models import QAResult
        mock_gen.return_value = QAResult(answer="ok", mode="bm25")
        result = svc.query("test <script>alert('xss')</script>", mode="bm25")
        assert result.answer == "ok"


def test_regex_dangerous_query():
    svc = _make_service()
    with patch("opensearch_graphrag.service.generate_answer") as mock_gen:
        from opensearch_graphrag.models import QAResult
        mock_gen.return_value = QAResult(answer="ok", mode="bm25")
        result = svc.query("test (.*) [a-z]+ {3,} \\d+", mode="bm25")
        assert result.answer == "ok"


def test_hallucination_unicode_context():
    answer = "Документы содержат информацию"
    context = ["Документы содержат информацию о системе"]
    result = detect_hallucination(answer, context)
    assert result["grounded"] is True


def test_chunk_empty_text():
    c = Chunk(id="c1", text="")
    assert c.text == ""
    assert c.embedding == []


def test_search_result_zero_score():
    sr = SearchResult(chunk_id="c1", text="hello", score=0.0)
    assert sr.score == 0.0


def test_search_result_negative_score():
    sr = SearchResult(chunk_id="c1", text="hello", score=-1.5)
    assert sr.score == -1.5

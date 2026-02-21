"""Tests for query expander module."""

from unittest.mock import patch

from opensearch_graphrag.query_expander import build_expanded_query, expand_query


def test_expand_query_empty():
    assert expand_query("") == {}
    assert expand_query("   ") == {}


@patch("opensearch_graphrag.query_expander._post_generate")
def test_expand_query_success(mock_post):
    mock_post.return_value = {
        "response": '{"themes": ["search", "index"], "entities": ["OpenSearch"], "expanded": ["retrieval"]}'
    }
    result = expand_query("What is OpenSearch?")
    assert "themes" in result
    assert "OpenSearch" in result["entities"]
    assert len(result["expanded"]) >= 1


@patch("opensearch_graphrag.query_expander._post_generate")
def test_expand_query_invalid_json(mock_post):
    mock_post.return_value = {"response": "not json at all"}
    result = expand_query("test query")
    assert result == {}


@patch("opensearch_graphrag.query_expander._post_generate")
def test_expand_query_network_error(mock_post):
    mock_post.side_effect = Exception("connection refused")
    result = expand_query("test query")
    assert result == {}


def test_build_expanded_query_with_expansion():
    q = build_expanded_query("OpenSearch", {"themes": ["search", "index"], "expanded": ["retrieval"]})
    assert "OpenSearch" in q
    assert "search" in q
    assert "retrieval" in q


def test_build_expanded_query_empty_expansion():
    q = build_expanded_query("test", {})
    assert q == "test"

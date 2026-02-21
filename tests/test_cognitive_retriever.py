"""Tests for cognitive retriever."""

from unittest.mock import MagicMock, patch

from opensearch_graphrag.cognitive_retriever import CognitiveRetriever
from opensearch_graphrag.models import SearchResult


def _sr(cid, score=0.9, text="text"):
    return SearchResult(chunk_id=cid, text=text, score=score, source="test")


def _make_store():
    store = MagicMock()
    store.search_bm25.return_value = [_sr("c1", 8.5)]
    store.search_vector.return_value = [_sr("c2", 0.95)]
    store.get_embeddings.return_value = {}
    return store


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_cognitive_search_no_expansion(mock_expand):
    mock_expand.return_value = {}
    store = _make_store()
    retriever = CognitiveRetriever(store=store)

    results = retriever.search("test query", embedding=[0.1] * 768)

    assert len(results) > 0
    assert store.search_bm25.called
    assert store.search_vector.called


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_cognitive_search_with_entities(mock_expand):
    mock_expand.return_value = {
        "themes": ["search"],
        "entities": ["OpenSearch"],
        "expanded": ["retrieval"],
    }
    store = _make_store()
    retriever = CognitiveRetriever(store=store)

    retriever.search("What is OpenSearch?")

    assert store.search_bm25.called


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_cognitive_search_no_embedding(mock_expand):
    mock_expand.return_value = {"themes": ["test"]}
    store = _make_store()
    retriever = CognitiveRetriever(store=store)

    retriever.search("test")

    assert store.search_bm25.called
    store.search_vector.assert_not_called()


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_cognitive_entity_graph_search_no_driver(mock_expand):
    mock_expand.return_value = {"entities": ["Neo4j"]}
    store = _make_store()
    retriever = CognitiveRetriever(store=store, neo4j_driver=None)

    results = retriever._entity_graph_search(["Neo4j"])
    assert results == []

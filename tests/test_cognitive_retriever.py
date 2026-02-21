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


def _make_driver_with_records(records):
    """Create a mock Neo4j driver that returns given records."""
    driver = MagicMock()
    mock_session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = records
    return driver


# ── search() tests ────────────────────────────────────────


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


# ── _entity_graph_search() with driver ────────────────────


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_entity_graph_search_success(mock_expand):
    """Entity graph search with mock driver returns SearchResult list."""
    mock_expand.return_value = {}
    records = [
        {"chunk_id": "c10", "text": "Graph result", "entities": ["Neo4j", "Graph"]},
    ]
    driver = _make_driver_with_records(records)
    store = _make_store()
    retriever = CognitiveRetriever(store=store, neo4j_driver=driver)

    results = retriever._entity_graph_search(["Neo4j"])
    assert len(results) == 1
    assert results[0].chunk_id == "c10"
    assert results[0].source == "graph"
    assert "Neo4j" in results[0].metadata["entities"]


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_entity_graph_search_exception(mock_expand):
    """Driver exception in _entity_graph_search returns empty list."""
    mock_expand.return_value = {}
    driver = MagicMock()
    driver.session.side_effect = Exception("Neo4j unavailable")

    store = _make_store()
    retriever = CognitiveRetriever(store=store, neo4j_driver=driver)

    results = retriever._entity_graph_search(["Neo4j"])
    assert results == []


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_entity_graph_search_empty_entities(mock_expand):
    """Empty entity list returns empty results without calling driver."""
    mock_expand.return_value = {}
    driver = MagicMock()
    store = _make_store()
    retriever = CognitiveRetriever(store=store, neo4j_driver=driver)

    results = retriever._entity_graph_search([])
    assert results == []
    driver.session.assert_not_called()


# ── search() with rerank branch ───────────────────────────


@patch("opensearch_graphrag.cognitive_retriever.rerank")
@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_cognitive_search_with_rerank(mock_expand, mock_rerank):
    """When embedding and results are non-empty, rerank is called."""
    mock_expand.return_value = {"themes": ["test"], "entities": []}
    store = _make_store()
    # Provide non-empty chunk embeddings so rerank fires
    store.get_embeddings.return_value = {"c1": [0.1] * 768}
    mock_rerank.return_value = [_sr("c1", 0.95)]

    retriever = CognitiveRetriever(store=store)
    results = retriever.search("test?", embedding=[0.1] * 768)

    mock_rerank.assert_called_once()
    assert len(results) > 0


# ── Full two-stage with entities and driver ────────────────


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_cognitive_search_full_two_stage(mock_expand):
    """Full two-stage retrieval: themes + entities with graph search."""
    mock_expand.return_value = {
        "themes": ["knowledge graph"],
        "entities": ["Neo4j", "OpenSearch"],
    }
    records = [
        {"chunk_id": "c20", "text": "Neo4j graph", "entities": ["Neo4j"]},
    ]
    driver = _make_driver_with_records(records)
    store = _make_store()
    store.get_embeddings.return_value = {}

    retriever = CognitiveRetriever(store=store, neo4j_driver=driver)
    results = retriever.search("What is Neo4j?", embedding=[0.1] * 768)

    # Both BM25 and vector should be called (stage 1)
    assert store.search_bm25.called
    assert store.search_vector.called
    assert len(results) > 0

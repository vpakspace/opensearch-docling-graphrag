"""Tests for OpenSearch store with mocked client."""

from unittest.mock import MagicMock

from opensearch_graphrag.models import Chunk
from opensearch_graphrag.opensearch_store import OpenSearchStore


def _make_store(mock_client=None):
    """Create store with mocked client."""
    client = mock_client or MagicMock()
    return OpenSearchStore(client=client)


def test_init_index_creates_when_missing():
    client = MagicMock()
    client.indices.exists.return_value = False
    store = _make_store(client)

    store.init_index()

    client.indices.create.assert_called_once()
    call_kwargs = client.indices.create.call_args
    assert call_kwargs[1]["index"] == "rag_chunks"
    body = call_kwargs[1]["body"]
    assert body["settings"]["index"]["knn"] is True
    assert body["mappings"]["properties"]["embedding"]["type"] == "knn_vector"


def test_init_index_skips_when_exists():
    client = MagicMock()
    client.indices.exists.return_value = True
    store = _make_store(client)

    store.init_index()

    client.indices.create.assert_not_called()


def test_add_chunks():
    client = MagicMock()
    client.bulk.return_value = {"errors": False, "items": []}
    store = _make_store(client)

    chunks = [
        Chunk(id="c1", text="hello", embedding=[0.1] * 768, source="test.txt"),
        Chunk(id="c2", text="world", embedding=[0.2] * 768, source="test.txt"),
    ]
    count = store.add_chunks(chunks)

    assert count == 2
    client.bulk.assert_called_once()


def test_add_chunks_empty():
    client = MagicMock()
    store = _make_store(client)
    assert store.add_chunks([]) == 0
    client.bulk.assert_not_called()


def test_search_vector():
    client = MagicMock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "c1", "_score": 0.95, "_source": {"text": "hello", "source": "test.txt", "metadata": {}}},
            ]
        }
    }
    store = _make_store(client)

    results = store.search_vector([0.1] * 768, top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score == 0.95
    assert results[0].text == "hello"


def test_search_bm25():
    client = MagicMock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "c2", "_score": 8.5, "_source": {"text": "world", "source": "doc.txt", "metadata": {}}},
            ]
        }
    }
    store = _make_store(client)

    results = store.search_bm25("world", top_k=3)

    assert len(results) == 1
    assert results[0].chunk_id == "c2"


def test_search_hybrid():
    client = MagicMock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "c1", "_score": 5.0, "_source": {"text": "combined", "source": "f.txt", "metadata": {}}},
            ]
        }
    }
    store = _make_store(client)

    results = store.search_hybrid("test", [0.1] * 768, top_k=5)

    assert len(results) == 1
    call_body = client.search.call_args[1]["body"]
    assert "bool" in call_body["query"]


def test_delete_all():
    client = MagicMock()
    client.indices.exists.return_value = True
    store = _make_store(client)

    store.delete_all()

    client.delete_by_query.assert_called_once()


def test_count():
    client = MagicMock()
    client.indices.exists.return_value = True
    client.count.return_value = {"count": 42}
    store = _make_store(client)

    assert store.count() == 42


def test_count_no_index():
    client = MagicMock()
    client.indices.exists.return_value = False
    store = _make_store(client)

    assert store.count() == 0


def test_search_error_returns_empty():
    client = MagicMock()
    client.search.side_effect = Exception("connection refused")
    store = _make_store(client)

    results = store.search_bm25("test")
    assert results == []


def test_get_embeddings():
    client = MagicMock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "c1", "_source": {"embedding": [0.1, 0.2, 0.3]}},
                {"_id": "c2", "_source": {"embedding": [0.4, 0.5, 0.6]}},
            ]
        }
    }
    store = _make_store(client)

    embs = store.get_embeddings(["c1", "c2"])
    assert "c1" in embs
    assert "c2" in embs
    assert embs["c1"] == [0.1, 0.2, 0.3]


def test_get_embeddings_empty():
    store = _make_store()
    assert store.get_embeddings([]) == {}

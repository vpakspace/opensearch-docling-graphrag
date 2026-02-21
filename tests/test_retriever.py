"""Tests for retriever module."""

from unittest.mock import MagicMock

from opensearch_graphrag.models import SearchResult
from opensearch_graphrag.retriever import Retriever, rrf_fuse


def _sr(cid, score=0.9, text="text"):
    return SearchResult(chunk_id=cid, text=text, score=score, source="test")


# ── RRF Tests ──────────────────────────────────────────────────


def test_rrf_fuse_single_list():
    results = rrf_fuse([_sr("c1", 0.9), _sr("c2", 0.8)], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"


def test_rrf_fuse_two_lists_overlap():
    list1 = [_sr("c1", 0.9), _sr("c2", 0.8)]
    list2 = [_sr("c2", 0.95), _sr("c3", 0.7)]
    results = rrf_fuse(list1, list2, top_k=3)
    # c2 appears in both lists so should rank higher
    ids = [r.chunk_id for r in results]
    assert "c2" in ids
    assert len(results) == 3


def test_rrf_fuse_empty():
    results = rrf_fuse([], [], top_k=5)
    assert results == []


def test_rrf_fuse_top_k():
    big_list = [_sr(f"c{i}") for i in range(20)]
    results = rrf_fuse(big_list, top_k=3)
    assert len(results) == 3


# ── Retriever Tests ────────────────────────────────────────────


def _make_store():
    store = MagicMock()
    store.search_bm25.return_value = [_sr("c1", 8.5)]
    store.search_vector.return_value = [_sr("c2", 0.95)]
    return store


def test_retriever_bm25_mode():
    store = _make_store()
    retriever = Retriever(store=store)
    results = retriever.search("test", mode="bm25")
    assert len(results) == 1
    store.search_bm25.assert_called_once()
    store.search_vector.assert_not_called()


def test_retriever_vector_mode():
    store = _make_store()
    retriever = Retriever(store=store)
    emb = [0.1] * 768
    results = retriever.search("test", embedding=emb, mode="vector")
    assert len(results) == 1
    store.search_vector.assert_called_once()


def test_retriever_vector_mode_no_embedding_fallback():
    store = _make_store()
    retriever = Retriever(store=store)
    retriever.search("test", mode="vector")
    # Falls back to BM25
    store.search_bm25.assert_called()


def test_retriever_graph_mode_no_driver():
    store = _make_store()
    retriever = Retriever(store=store, neo4j_driver=None)
    results = retriever.search("test", mode="graph")
    assert results == []


def test_retriever_graph_mode_with_driver():
    store = _make_store()
    driver = MagicMock()
    mock_session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = [
        {"chunk_id": "c5", "text": "graph result", "entities": ["OpenSearch"]},
    ]

    retriever = Retriever(store=store, neo4j_driver=driver)
    results = retriever.search("OpenSearch query", mode="graph")

    assert len(results) == 1
    assert results[0].chunk_id == "c5"
    assert results[0].source == "graph"


def test_extract_keywords_russian():
    kw = Retriever._extract_keywords("Какие фреймворки для графовых баз знаний?")
    assert "фреймворки" in kw
    assert "графовых" in kw
    assert "знаний" in kw
    assert "какие" not in kw
    assert "для" not in kw


def test_extract_keywords_english():
    kw = Retriever._extract_keywords("What is the Semantic Companion Layer?")
    assert "semantic" in kw
    assert "companion" in kw
    assert "layer" in kw
    assert "what" not in kw
    assert "the" not in kw


def test_extract_keywords_empty():
    assert Retriever._extract_keywords("") == []
    assert Retriever._extract_keywords("a b") == []


def test_retriever_hybrid_mode():
    store = _make_store()
    retriever = Retriever(store=store)
    emb = [0.1] * 768
    results = retriever.search("test", embedding=emb, mode="hybrid")
    # Hybrid fuses BM25 + vector + graph
    store.search_bm25.assert_called_once()
    store.search_vector.assert_called_once()
    assert len(results) > 0

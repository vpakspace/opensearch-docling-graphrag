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


def _make_settings(**overrides):
    """Create a mock settings object with retrieval defaults."""
    retrieval = MagicMock()
    retrieval.top_k_vector = 10
    retrieval.top_k_bm25 = 10
    retrieval.top_k_graph = 10
    retrieval.top_k_final = 5
    retrieval.max_probes = overrides.get("max_probes", 2)
    retrieval.evidence_score_threshold = overrides.get(
        "evidence_score_threshold", 0.3,
    )
    settings = MagicMock()
    settings.retrieval = retrieval
    settings.ollama.llm_model = "llama3.1:8b"
    return settings


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


# ── _check_evidence_sufficiency() tests ──────────────────


def test_evidence_sufficiency_high_scores():
    """All results with high scores → evidence sufficient (>= 0.3)."""
    store = _make_store()
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    results = [
        _sr("c1", 0.9, text="OpenSearch vector database search engine"),
        _sr("c2", 0.85, text="OpenSearch supports vector search and BM25"),
        _sr("c3", 0.8, text="OpenSearch distributed search engine"),
    ]
    score = retriever._check_evidence_sufficiency("OpenSearch search", results)
    assert score >= 0.3


def test_evidence_sufficiency_empty_results():
    """No results → evidence score 0.0."""
    store = _make_store()
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    score = retriever._check_evidence_sufficiency("anything", [])
    assert score == 0.0


def test_evidence_sufficiency_low_overlap():
    """Answer words not in context → low keyword coverage score."""
    store = _make_store()
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    results = [
        _sr("c1", 0.0, text="weather forecast tomorrow sunny"),
    ]
    score = retriever._check_evidence_sufficiency(
        "OpenSearch vector database architecture", results,
    )
    # Zero score + zero overlap + low volume (1/5) → should be well below 0.3
    assert score < 0.3


# ── _generate_probe_query() tests ────────────────────────


@patch("opensearch_graphrag.ollama_client.post_generate")
def test_generate_probe_query_success(mock_generate):
    """Mock Ollama returns a probe string."""
    mock_generate.return_value = {
        "response": "How does OpenSearch handle distributed indexing?",
    }
    store = _make_store()
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    results = [_sr("c1", 0.5, text="OpenSearch basics")]
    probe = retriever._generate_probe_query("OpenSearch architecture", results)

    assert probe is not None
    assert "OpenSearch" in probe
    mock_generate.assert_called_once()


@patch("opensearch_graphrag.ollama_client.post_generate")
def test_generate_probe_query_empty_response(mock_generate):
    """Ollama returns empty → None."""
    mock_generate.return_value = {"response": ""}
    store = _make_store()
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    probe = retriever._generate_probe_query("test", [_sr("c1")])
    assert probe is None


@patch("opensearch_graphrag.ollama_client.post_generate")
def test_generate_probe_query_exception(mock_generate):
    """Ollama fails → None (graceful degradation)."""
    mock_generate.side_effect = Exception("Ollama unavailable")
    store = _make_store()
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    probe = retriever._generate_probe_query("test", [_sr("c1")])
    assert probe is None


# ── _probe_search() tests ────────────────────────────────


@patch("opensearch_graphrag.embedder.embed_text")
def test_probe_search_bm25_vector(mock_embed):
    """Probe search returns merged BM25 + vector results."""
    mock_embed.return_value = [0.2] * 768
    store = _make_store()
    store.search_bm25.return_value = [_sr("p1", 5.0, "probe bm25")]
    store.search_vector.return_value = [_sr("p2", 0.8, "probe vector")]
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    results = retriever._probe_search("refined query", original_embedding=[0.1] * 768)

    assert len(results) > 0
    mock_embed.assert_called_once()


def test_probe_search_no_embedding():
    """BM25-only when no original embedding provided."""
    store = _make_store()
    store.search_bm25.return_value = [_sr("p1", 5.0, "probe bm25")]
    settings = _make_settings()
    retriever = CognitiveRetriever(store=store, settings=settings)

    results = retriever._probe_search("refined query", original_embedding=None)

    assert len(results) > 0
    store.search_vector.assert_not_called()


# ── search() iterative probing integration ────────────────


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_search_no_probes_needed(mock_expand):
    """High evidence score → 0 probes used."""
    mock_expand.return_value = {"themes": ["OpenSearch"]}
    store = _make_store()
    # BM25 returns results with text matching the query → high evidence
    store.search_bm25.return_value = [
        _sr("c1", 8.5, text="OpenSearch vector search engine handles queries"),
        _sr("c2", 7.0, text="OpenSearch distributed search architecture"),
        _sr("c3", 6.0, text="OpenSearch index management and search"),
    ]
    store.search_vector.return_value = [
        _sr("c4", 0.9, text="OpenSearch search capabilities"),
    ]
    settings = _make_settings(evidence_score_threshold=0.3)
    retriever = CognitiveRetriever(store=store, settings=settings)

    results = retriever.search("OpenSearch search", embedding=[0.1] * 768)

    assert retriever.last_probes_used == 0
    assert len(results) > 0


@patch("opensearch_graphrag.embedder.embed_text")
@patch("opensearch_graphrag.ollama_client.post_generate")
@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_search_one_probe(mock_expand, mock_generate, mock_embed):
    """Low evidence → 1 probe fills gaps → sufficient."""
    mock_expand.return_value = {}
    mock_generate.return_value = {
        "response": "What are OpenSearch vector capabilities?",
    }
    mock_embed.return_value = [0.2] * 768

    store = _make_store()
    # Initial: weak results (no keyword overlap with query)
    store.search_bm25.return_value = [_sr("c1", 0.1, text="unrelated content xyz")]
    store.search_vector.return_value = []
    store.get_embeddings.return_value = {}

    # After probe: good results (matching keywords)
    call_count = {"n": 0}

    def bm25_side_effect(query, top_k=10):  # noqa: ARG001
        call_count["n"] += 1
        if call_count["n"] <= 2:
            # Initial calls (theme + entity stage)
            return [_sr("c1", 0.1, text="unrelated content xyz")]
        # Probe call → return relevant results
        return [
            _sr("c10", 8.0, text="OpenSearch vector search database architecture"),
            _sr("c11", 7.0, text="OpenSearch handles distributed search queries"),
            _sr("c12", 6.5, text="OpenSearch indexing and search performance"),
        ]

    store.search_bm25.side_effect = bm25_side_effect

    settings = _make_settings(max_probes=2, evidence_score_threshold=0.5)
    retriever = CognitiveRetriever(store=store, settings=settings)

    results = retriever.search("OpenSearch search architecture", embedding=[0.1] * 768)

    assert retriever.last_probes_used >= 1
    assert len(results) > 0


@patch("opensearch_graphrag.embedder.embed_text")
@patch("opensearch_graphrag.ollama_client.post_generate")
@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_search_max_probes_exhausted(mock_expand, mock_generate, mock_embed):
    """Low evidence persists → uses all max_probes iterations."""
    mock_expand.return_value = {}
    mock_generate.return_value = {"response": "some probe query"}
    mock_embed.return_value = [0.2] * 768

    store = _make_store()
    # Always return weak results (no keyword overlap with query)
    store.search_bm25.return_value = [_sr("c1", 0.1, text="xyz abc")]
    store.search_vector.return_value = []
    store.get_embeddings.return_value = {}

    settings = _make_settings(
        max_probes=3, evidence_score_threshold=0.99,  # unreachable threshold
    )
    retriever = CognitiveRetriever(store=store, settings=settings)

    retriever.search("OpenSearch distributed architecture", embedding=[0.1] * 768)

    assert retriever.last_probes_used == 3
    assert mock_generate.call_count == 3


@patch("opensearch_graphrag.cognitive_retriever.expand_query")
def test_last_probes_used_attribute(mock_expand):
    """Attribute correctly tracks probe count across calls."""
    mock_expand.return_value = {"themes": ["test"]}
    store = _make_store()
    store.search_bm25.return_value = [
        _sr("c1", 8.0, text="test content matching the query words"),
        _sr("c2", 7.0, text="more test content here for coverage"),
    ]
    settings = _make_settings(max_probes=0)  # probing disabled
    retriever = CognitiveRetriever(store=store, settings=settings)

    retriever.search("test content query")
    assert retriever.last_probes_used == 0

    # Subsequent call should reset
    retriever.search("another test query")
    assert retriever.last_probes_used == 0

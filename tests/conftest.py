"""Shared test fixtures for opensearch-docling-graphrag."""

from unittest.mock import MagicMock

import pytest

from opensearch_graphrag.models import SearchResult


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return (
        "OpenSearch is a search engine. "
        "Neo4j is a graph database. "
        "Ollama runs LLMs locally."
    )


@pytest.fixture
def sample_chunks():
    """Sample chunk dicts for testing."""
    return [
        {"id": "c1", "text": "OpenSearch is a search engine.", "metadata": {"source": "test.txt"}},
        {"id": "c2", "text": "Neo4j is a graph database.", "metadata": {"source": "test.txt"}},
    ]


@pytest.fixture
def mock_settings():
    """Mock Settings object with common defaults."""
    settings = MagicMock()
    settings.ollama.base_url = "http://localhost:11434"
    settings.ollama.llm_model = "llama3.1:8b"
    settings.ollama.embed_model = "nomic-embed-text-v2-moe"
    settings.ollama.temperature = 0.0
    settings.opensearch.url = "http://localhost:9200"
    settings.opensearch.index = "graphrag"
    settings.neo4j.uri = "bolt://localhost:7687"
    settings.neo4j.user = "neo4j"
    settings.neo4j.password = "password"
    settings.chunking.chunk_size = 512
    settings.chunking.chunk_overlap = 50
    settings.top_k.bm25 = 5
    settings.top_k.vector = 5
    settings.top_k.graph = 5
    settings.top_k.final = 5
    return settings


@pytest.fixture
def sample_search_results():
    """Sample SearchResult list for testing."""
    return [
        SearchResult(chunk_id="c1", text="OpenSearch is a search engine.", score=0.95, source="doc1.pdf"),
        SearchResult(chunk_id="c2", text="Neo4j is a graph database.", score=0.85, source="doc1.pdf"),
        SearchResult(chunk_id="c3", text="Ollama runs LLMs locally.", score=0.75, source="doc2.pdf"),
    ]


@pytest.fixture
def fake_embedding():
    """Fake 768-dimensional embedding vector."""
    return [0.01 * i for i in range(768)]

"""Shared test fixtures for opensearch-docling-graphrag."""

import pytest


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

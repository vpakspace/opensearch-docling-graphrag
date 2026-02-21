"""Tests for config module."""

from opensearch_graphrag.config import (
    ChunkingSettings,
    Neo4jSettings,
    OllamaSettings,
    OpenSearchSettings,
    RetrievalSettings,
    Settings,
)


def test_ollama_defaults():
    s = OllamaSettings()
    assert s.base_url == "http://localhost:11434"
    assert s.llm_model == "qwen2.5:7b"
    assert s.embed_model == "nomic-embed-text-v2-moe"
    assert s.embed_dimensions == 768
    assert s.temperature == 0.0


def test_opensearch_defaults():
    s = OpenSearchSettings()
    assert s.host == "localhost"
    assert s.port == 9200
    assert s.index == "rag_chunks"
    assert s.url == "http://localhost:9200"


def test_neo4j_defaults():
    s = Neo4jSettings()
    assert s.uri == "bolt://localhost:7687"
    assert s.user == "neo4j"
    assert s.password == "neo4j"


def test_chunking_defaults():
    s = ChunkingSettings()
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64


def test_retrieval_defaults():
    s = RetrievalSettings()
    assert s.top_k_vector == 10
    assert s.top_k_bm25 == 10
    assert s.top_k_graph == 10
    assert s.top_k_final == 5


def test_settings_composition():
    s = Settings()
    assert isinstance(s.ollama, OllamaSettings)
    assert isinstance(s.opensearch, OpenSearchSettings)
    assert isinstance(s.neo4j, Neo4jSettings)
    assert isinstance(s.chunking, ChunkingSettings)
    assert isinstance(s.retrieval, RetrievalSettings)
    assert s.log_level == "INFO"


def test_make_ollama_client():
    from opensearch_graphrag.config import make_ollama_client

    client = make_ollama_client()
    assert client is not None
    assert str(client.base_url).rstrip("/") == "http://localhost:11434"
    client.close()

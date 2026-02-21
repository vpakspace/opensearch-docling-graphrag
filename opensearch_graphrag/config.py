"""Configuration via Pydantic Settings — 100% local stack."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from pydantic import Field
from pydantic_settings import BaseSettings


class OllamaSettings(BaseSettings):
    base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text-v2-moe"
    embed_dimensions: int = Field(default=768, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    model_config = {"env_prefix": "OLLAMA_"}


class OpenSearchSettings(BaseSettings):
    host: str = "localhost"
    port: int = 9200
    index: str = "rag_chunks"

    model_config = {"env_prefix": "OPENSEARCH_"}

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "neo4j"

    model_config = {"env_prefix": "NEO4J_"}


class ChunkingSettings(BaseSettings):
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)

    model_config = {"env_prefix": "CHUNK_"}


class RetrievalSettings(BaseSettings):
    top_k_vector: int = Field(default=10, gt=0)
    top_k_bm25: int = Field(default=10, gt=0)
    top_k_graph: int = Field(default=10, gt=0)
    top_k_final: int = Field(default=5, gt=0)

    model_config = {"env_prefix": "TOP_K_"}


class Settings(BaseSettings):
    ollama: OllamaSettings = OllamaSettings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()

    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create and cache settings instance loading from environment."""
    return Settings()


def make_ollama_client(settings: Settings | None = None) -> "httpx.Client":
    """Create httpx client for Ollama REST API."""
    import httpx

    cfg = settings or get_settings()
    return httpx.Client(base_url=cfg.ollama.base_url, timeout=120.0)

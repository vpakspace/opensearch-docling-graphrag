"""Domain models for the RAG pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk with embedding and metadata."""

    id: str
    text: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    source: str = ""
    chunk_index: int = 0


class Entity(BaseModel):
    """A named entity extracted from text."""

    name: str
    type: str  # Person, Organization, Location, Date, Other
    source_chunk_id: str = ""


class Relationship(BaseModel):
    """A relationship between two entities."""

    source: str
    target: str
    type: str = "RELATED_TO"


class SearchResult(BaseModel):
    """A single search result with score and source info."""

    chunk_id: str
    text: str
    score: float = 0.0
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class QAResult(BaseModel):
    """Question-answering result with answer and supporting context."""

    answer: str
    confidence: float = 0.0
    sources: list[SearchResult] = Field(default_factory=list)
    mode: str = "hybrid"
    grounded: bool = True
    grounding_score: float = 1.0
    warning: str = ""

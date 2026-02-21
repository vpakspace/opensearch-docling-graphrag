"""Custom exceptions for opensearch-docling-graphrag."""


class GraphRAGError(Exception):
    """Base exception for the GraphRAG pipeline."""


class EmbeddingError(GraphRAGError):
    """Error during embedding generation."""


class GenerationError(GraphRAGError):
    """Error during LLM answer generation."""


class StoreError(GraphRAGError):
    """Error interacting with OpenSearch store."""


class GraphError(GraphRAGError):
    """Error interacting with Neo4j graph."""


class ValidationError(GraphRAGError):
    """Input validation error."""

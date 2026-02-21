"""Tests for opensearch_graphrag.entity_extractor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from opensearch_graphrag.entity_extractor import extract_entities
from opensearch_graphrag.models import Entity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_response(entities: list[dict]) -> MagicMock:
    """Return a mock httpx Response carrying a valid Ollama /api/generate reply."""
    payload = json.dumps({"entities": entities})
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": payload}
    return mock_response


def _make_settings(
    base_url: str = "http://localhost:11434",
    llm_model: str = "llama3.1:8b",
) -> MagicMock:
    settings = MagicMock()
    settings.ollama.base_url = base_url
    settings.ollama.llm_model = llm_model
    return settings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractEntities:
    """Unit tests for extract_entities()."""

    def test_extract_entities(self):
        """Mock Ollama returning 3 entities — all should be parsed correctly."""
        raw_entities = [
            {"name": "OpenSearch", "type": "Organization"},
            {"name": "Berlin", "type": "Location"},
            {"name": "2024-01-15", "type": "Date"},
        ]
        mock_response = _make_ollama_response(raw_entities)
        settings = _make_settings()

        with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.post.return_value = mock_response

            result = extract_entities(
                text="OpenSearch conference in Berlin on 2024-01-15.",
                chunk_id="chunk-42",
                settings=settings,
            )

        assert len(result) == 3

        names = [e.name for e in result]
        assert "OpenSearch" in names
        assert "Berlin" in names
        assert "2024-01-15" in names

        types = {e.name: e.type for e in result}
        assert types["OpenSearch"] == "Organization"
        assert types["Berlin"] == "Location"
        assert types["2024-01-15"] == "Date"

        # Every entity must carry the caller-supplied chunk_id.
        assert all(e.source_chunk_id == "chunk-42" for e in result)

    def test_extract_entities_empty_text(self):
        """Empty or whitespace-only text must return an empty list immediately."""
        # No HTTP call should be made for blank text.
        with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
            result_empty = extract_entities(text="", chunk_id="c0")
            result_whitespace = extract_entities(text="   \t\n", chunk_id="c0")

        assert result_empty == []
        assert result_whitespace == []
        mock_get_client.assert_not_called()

    def test_extract_entities_invalid_json(self):
        """A non-JSON Ollama response must yield an empty list, not an exception."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # Ollama returns plain text instead of JSON.
        mock_response.json.return_value = {"response": "Sorry, I cannot help."}
        settings = _make_settings()

        with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.post.return_value = mock_response

            result = extract_entities(
                text="Some text that confuses the model.",
                chunk_id="c99",
                settings=settings,
            )

        assert result == []

    def test_extract_entities_sets_chunk_id(self):
        """The source_chunk_id on every returned Entity must match the argument."""
        raw_entities = [
            {"name": "Alice", "type": "Person"},
            {"name": "Acme Corp", "type": "Organization"},
        ]
        mock_response = _make_ollama_response(raw_entities)
        settings = _make_settings()

        with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.post.return_value = mock_response

            result = extract_entities(
                text="Alice works at Acme Corp.",
                chunk_id="special-chunk-id",
                settings=settings,
            )

        assert len(result) == 2
        for entity in result:
            assert entity.source_chunk_id == "special-chunk-id", (
                f"Expected 'special-chunk-id', got {entity.source_chunk_id!r}"
            )

    def test_extract_entities_returns_entity_objects(self):
        """Returned items must be proper Entity instances."""
        raw_entities = [{"name": "London", "type": "Location"}]
        mock_response = _make_ollama_response(raw_entities)
        settings = _make_settings()

        with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.post.return_value = mock_response

            result = extract_entities(
                text="The meeting is in London.",
                chunk_id="c1",
                settings=settings,
            )

        assert len(result) == 1
        assert isinstance(result[0], Entity)

    def test_extract_entities_http_error_returns_empty(self):
        """An httpx.HTTPError during the Ollama call must yield an empty list."""
        import httpx

        settings = _make_settings()

        with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("connection refused")

            result = extract_entities(
                text="OpenSearch is amazing.",
                chunk_id="c5",
                settings=settings,
            )

        assert result == []

    def test_extract_entities_skips_items_with_empty_name(self):
        """Entities with a blank name must be silently skipped."""
        raw_entities = [
            {"name": "", "type": "Organization"},
            {"name": "  ", "type": "Person"},
            {"name": "Neo4j", "type": "Organization"},
        ]
        mock_response = _make_ollama_response(raw_entities)
        settings = _make_settings()

        with patch("opensearch_graphrag.config.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.post.return_value = mock_response

            result = extract_entities(
                text="Neo4j is a graph database.",
                chunk_id="c2",
                settings=settings,
            )

        assert len(result) == 1
        assert result[0].name == "Neo4j"

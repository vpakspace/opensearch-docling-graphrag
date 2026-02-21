"""Tests for opensearch_graphrag.graph_builder."""

from __future__ import annotations

from unittest.mock import MagicMock

from opensearch_graphrag.graph_builder import GraphBuilder
from opensearch_graphrag.models import Entity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_driver() -> MagicMock:
    """Return a mock neo4j.Driver with a usable session() context manager."""
    driver = MagicMock()
    session = MagicMock()
    # session() must work as a context manager: `with driver.session() as s:`
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


def _get_session(driver: MagicMock) -> MagicMock:
    """Extract the mock session object from a mock driver."""
    return driver.session.return_value.__enter__.return_value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAddDocument:
    def test_add_document_calls_merge_cypher(self):
        """add_document should issue a MERGE Cypher statement via session.run."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.add_document(doc_id="doc-1", source="report.pdf")

        # session.run must have been called at least once.
        assert session.run.called

        # The first positional argument to run() must contain MERGE.
        cypher_arg: str = session.run.call_args[0][0]
        assert "MERGE" in cypher_arg
        assert "Document" in cypher_arg

    def test_add_document_passes_id_and_source(self):
        """add_document must forward doc_id and source as keyword arguments."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.add_document(doc_id="doc-42", source="whitepaper.txt")

        kwargs = session.run.call_args[1]
        assert kwargs.get("id") == "doc-42"
        assert kwargs.get("source") == "whitepaper.txt"

    def test_add_document_with_metadata(self):
        """add_document should accept and forward optional metadata."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.add_document(
            doc_id="doc-meta",
            source="meta.txt",
            metadata={"author": "Alice", "year": 2024},
        )

        assert session.run.called


class TestAddChunk:
    def test_add_chunk_creates_chunk_node(self):
        """add_chunk should MERGE a Chunk node."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.add_chunk(
            chunk_id="c1",
            text="OpenSearch is fast.",
            doc_id="doc-1",
            chunk_index=0,
        )

        assert session.run.called
        cypher_arg: str = session.run.call_args[0][0]
        assert "Chunk" in cypher_arg

    def test_add_chunk_creates_has_chunk_relationship(self):
        """The Cypher issued by add_chunk must include a HAS_CHUNK relationship."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.add_chunk(
            chunk_id="c2",
            text="Neo4j stores graphs.",
            doc_id="doc-1",
            chunk_index=1,
        )

        cypher_arg: str = session.run.call_args[0][0]
        assert "HAS_CHUNK" in cypher_arg

    def test_add_chunk_passes_correct_kwargs(self):
        """All parameters must arrive as keyword arguments to session.run."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.add_chunk(
            chunk_id="c3",
            text="Ollama runs locally.",
            doc_id="doc-2",
            chunk_index=2,
        )

        kwargs = session.run.call_args[1]
        assert kwargs.get("chunk_id") == "c3"
        assert kwargs.get("doc_id") == "doc-2"
        assert kwargs.get("chunk_index") == 2


class TestAddEntity:
    def test_add_entity_issues_merge(self):
        """add_entity should MERGE an Entity node by name and type."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        entity = Entity(name="IBM", type="Organization", source_chunk_id="c1")
        builder.add_entity(entity)

        assert session.run.called
        cypher_arg: str = session.run.call_args[0][0]
        assert "MERGE" in cypher_arg
        assert "Entity" in cypher_arg

    def test_add_entity_merges_by_name_and_type(self):
        """Entity MERGE must use both name and type as discriminators."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        entity = Entity(name="Paris", type="Location", source_chunk_id="c2")
        builder.add_entity(entity)

        kwargs = session.run.call_args[1]
        assert kwargs.get("name") == "Paris"
        assert kwargs.get("type") == "Location"


class TestLinkEntityToChunk:
    def test_link_entity_to_chunk_mentioned_in(self):
        """link_entity_to_chunk should create a MENTIONED_IN relationship."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.link_entity_to_chunk(
            entity_name="Neo4j",
            entity_type="Organization",
            chunk_id="c5",
        )

        assert session.run.called
        cypher_arg: str = session.run.call_args[0][0]
        assert "MENTIONED_IN" in cypher_arg

    def test_link_entity_to_chunk_kwargs(self):
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.link_entity_to_chunk(
            entity_name="Ollama",
            entity_type="Organization",
            chunk_id="c10",
        )

        kwargs = session.run.call_args[1]
        assert kwargs.get("name") == "Ollama"
        assert kwargs.get("type") == "Organization"
        assert kwargs.get("chunk_id") == "c10"


class TestLinkEntities:
    def test_link_entities_related_to(self):
        """link_entities with default rel_type should produce RELATED_TO."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.link_entities("IBM", "Granite")

        cypher_arg: str = session.run.call_args[0][0]
        assert "RELATED_TO" in cypher_arg

    def test_link_entities_custom_rel_type(self):
        """A custom rel_type must appear in the generated Cypher."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.link_entities("Alice", "Acme", rel_type="WORKS_FOR")

        cypher_arg: str = session.run.call_args[0][0]
        assert "WORKS_FOR" in cypher_arg


class TestGetStats:
    def test_get_stats_returns_expected_keys(self):
        """get_stats must return a dict with the four standard keys."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        # For each query, session.run returns a result whose .single() gives a record.
        def _run_side_effect(cypher, **_kwargs):
            mock_result = MagicMock()
            if "Document" in cypher:
                mock_result.single.return_value = {"c": 3}
            elif "Chunk" in cypher:
                mock_result.single.return_value = {"c": 10}
            elif "Entity" in cypher:
                mock_result.single.return_value = {"c": 7}
            else:
                # relationships query
                mock_result.single.return_value = {"c": 20}
            return mock_result

        session.run.side_effect = _run_side_effect

        stats = builder.get_stats()

        assert set(stats.keys()) == {"documents", "chunks", "entities", "relationships"}

    def test_get_stats_returns_correct_counts(self):
        """get_stats values should reflect what Neo4j returns."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        counts = {"documents": 2, "chunks": 8, "entities": 5, "relationships": 15}
        count_iter = iter(counts.values())

        def _run_side_effect(cypher, **_kwargs):
            mock_result = MagicMock()
            mock_result.single.return_value = {"c": next(count_iter)}
            return mock_result

        session.run.side_effect = _run_side_effect

        stats = builder.get_stats()

        assert stats["documents"] == 2
        assert stats["chunks"] == 8
        assert stats["entities"] == 5
        assert stats["relationships"] == 15

    def test_get_stats_none_single_returns_zero(self):
        """If .single() returns None (empty result), the stat should be 0."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        mock_result = MagicMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result

        stats = builder.get_stats()

        assert all(v == 0 for v in stats.values())


class TestClear:
    def test_clear_issues_detach_delete(self):
        """clear() must run a DETACH DELETE statement."""
        driver = _make_driver()
        session = _get_session(driver)
        builder = GraphBuilder(driver)

        builder.clear()

        assert session.run.called
        cypher_arg: str = session.run.call_args[0][0]
        assert "DELETE" in cypher_arg.upper()


class TestBuildFromChunks:
    def test_build_from_chunks_orchestrates_calls(self):
        """build_from_chunks should call add_document, add_chunk, add_entity,
        and link_entity_to_chunk for each chunk/entity pair."""
        driver = _make_driver()
        builder = GraphBuilder(driver)

        # Patch individual methods so we can track calls without Neo4j.
        builder.add_document = MagicMock()
        builder.add_chunk = MagicMock()
        builder.add_entity = MagicMock()
        builder.link_entity_to_chunk = MagicMock()

        chunks = [
            {"id": "c1", "text": "OpenSearch is a search engine.", "chunk_index": 0},
            {"id": "c2", "text": "Neo4j is a graph database.", "chunk_index": 1},
        ]
        entity_c1 = Entity(name="OpenSearch", type="Organization", source_chunk_id="c1")
        entity_c2 = Entity(name="Neo4j", type="Organization", source_chunk_id="c2")
        entities_per_chunk = {"c1": [entity_c1], "c2": [entity_c2]}

        builder.build_from_chunks(
            chunks=chunks,
            entities_per_chunk=entities_per_chunk,
            doc_id="doc-1",
            source="test.txt",
        )

        builder.add_document.assert_called_once_with(doc_id="doc-1", source="test.txt")
        assert builder.add_chunk.call_count == 2
        assert builder.add_entity.call_count == 2
        assert builder.link_entity_to_chunk.call_count == 2

    def test_build_from_chunks_no_entities(self):
        """build_from_chunks with an empty entities_per_chunk must not call
        add_entity or link_entity_to_chunk."""
        driver = _make_driver()
        builder = GraphBuilder(driver)

        builder.add_document = MagicMock()
        builder.add_chunk = MagicMock()
        builder.add_entity = MagicMock()
        builder.link_entity_to_chunk = MagicMock()

        chunks = [{"id": "c1", "text": "Hello world.", "chunk_index": 0}]

        builder.build_from_chunks(
            chunks=chunks,
            entities_per_chunk={},
            doc_id="doc-empty",
            source="empty.txt",
        )

        builder.add_entity.assert_not_called()
        builder.link_entity_to_chunk.assert_not_called()

"""Neo4j knowledge graph builder for the RAG pipeline.

Schema
------
(:Document {id, source, **metadata})
    -[:HAS_CHUNK]->
(:Chunk {id, text, chunk_index, doc_id})
    <-[:MENTIONED_IN]-
(:Entity {name, type})
    -[:RELATED_TO]->
(:Entity {name, type})
"""

from __future__ import annotations

import logging
from typing import Any

from opensearch_graphrag.models import Entity

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build and query a Neo4j knowledge graph from documents and chunks.

    Parameters
    ----------
    driver:
        A ``neo4j.Driver`` instance (or any compatible object that exposes
        a ``session()`` context manager).
    """

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    # ------------------------------------------------------------------
    # Node / relationship creation helpers
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        source: str,
        metadata: dict | None = None,
    ) -> None:
        """Create or update a Document node.

        Parameters
        ----------
        doc_id:
            Unique document identifier (used as the ``id`` property).
        source:
            Human-readable source description (file path, URL, …).
        metadata:
            Optional extra properties stored on the node.
        """
        props: dict = {"id": doc_id, "source": source}
        if metadata:
            props.update(metadata)

        cypher = (
            "MERGE (d:Document {id: $id}) "
            "SET d.source = $source, d += $extra"
        )
        with self._driver.session() as session:
            session.run(cypher, id=doc_id, source=source, extra=metadata or {})

        logger.debug("Added/updated Document node id=%r", doc_id)

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        doc_id: str,
        chunk_index: int = 0,
    ) -> None:
        """Create or update a Chunk node and link it to its parent Document.

        Parameters
        ----------
        chunk_id:
            Unique chunk identifier.
        text:
            Raw text content of the chunk.
        doc_id:
            Identifier of the parent Document node.
        chunk_index:
            Zero-based position of the chunk within the document.
        """
        cypher = (
            "MERGE (c:Chunk {id: $chunk_id}) "
            "SET c.text = $text, c.chunk_index = $chunk_index, c.doc_id = $doc_id "
            "WITH c "
            "MATCH (d:Document {id: $doc_id}) "
            "MERGE (d)-[:HAS_CHUNK]->(c)"
        )
        with self._driver.session() as session:
            session.run(
                cypher,
                chunk_id=chunk_id,
                text=text,
                chunk_index=chunk_index,
                doc_id=doc_id,
            )

        logger.debug("Added/updated Chunk node id=%r for doc_id=%r", chunk_id, doc_id)

    def add_entity(self, entity: Entity) -> None:
        """Merge an Entity node into the graph (deduplicated by name+type).

        Parameters
        ----------
        entity:
            An :class:`~opensearch_graphrag.models.Entity` instance.
        """
        cypher = "MERGE (e:Entity {name: $name, type: $type})"
        with self._driver.session() as session:
            session.run(cypher, name=entity.name, type=entity.type)

        logger.debug("Merged Entity name=%r type=%r", entity.name, entity.type)

    def link_entity_to_chunk(
        self,
        entity_name: str,
        entity_type: str,
        chunk_id: str,
    ) -> None:
        """Create a MENTIONED_IN relationship from an Entity to a Chunk.

        Both the Entity and the Chunk must already exist in the graph
        (or will be silently skipped if the MATCH finds nothing).

        Parameters
        ----------
        entity_name:
            ``name`` property of the Entity node.
        entity_type:
            ``type`` property of the Entity node.
        chunk_id:
            ``id`` property of the target Chunk node.
        """
        cypher = (
            "MATCH (e:Entity {name: $name, type: $type}) "
            "MATCH (c:Chunk {id: $chunk_id}) "
            "MERGE (e)-[:MENTIONED_IN]->(c)"
        )
        with self._driver.session() as session:
            session.run(
                cypher,
                name=entity_name,
                type=entity_type,
                chunk_id=chunk_id,
            )

        logger.debug(
            "Linked Entity name=%r → Chunk id=%r", entity_name, chunk_id
        )

    def link_entities(
        self,
        source_name: str,
        target_name: str,
        rel_type: str = "RELATED_TO",
    ) -> None:
        """Create a directed relationship between two Entity nodes.

        Parameters
        ----------
        source_name:
            ``name`` property of the source Entity.
        target_name:
            ``name`` property of the target Entity.
        rel_type:
            Relationship type label.  Defaults to ``"RELATED_TO"``.
        """
        # Cypher does not allow parameterised relationship type labels, so
        # we interpolate the validated label directly.  Callers pass the
        # rel_type from controlled pipeline code, not from raw user input.
        cypher = (
            f"MATCH (a:Entity {{name: $source_name}}) "
            f"MATCH (b:Entity {{name: $target_name}}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )
        with self._driver.session() as session:
            session.run(cypher, source_name=source_name, target_name=target_name)

        logger.debug(
            "Linked Entity %r -[:%s]-> %r", source_name, rel_type, target_name
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def build_from_chunks(
        self,
        chunks: list,
        entities_per_chunk: dict[str, list[Entity]],
        doc_id: str,
        source: str,
    ) -> None:
        """Orchestrate a full graph build for a document.

        For each chunk the method:

        1. Creates / updates the Chunk node.
        2. Creates / updates each Entity referenced by that chunk.
        3. Creates MENTIONED_IN edges between entities and the chunk.

        Parameters
        ----------
        chunks:
            List of chunk objects.  Each element must have ``id``,
            ``text``, and ``chunk_index`` attributes *or* be a ``dict``
            with the same keys.
        entities_per_chunk:
            Mapping of ``chunk_id`` → list of :class:`Entity` objects
            produced by the entity extractor.
        doc_id:
            Identifier for the parent Document node (created here).
        source:
            Source description forwarded to :meth:`add_document`.
        """
        self.add_document(doc_id=doc_id, source=source)

        for chunk in chunks:
            # Support both object-style and dict-style chunks.
            if isinstance(chunk, dict):
                chunk_id = chunk["id"]
                text = chunk["text"]
                chunk_index = chunk.get("chunk_index", 0)
            else:
                chunk_id = chunk.id
                text = chunk.text
                chunk_index = getattr(chunk, "chunk_index", 0)

            self.add_chunk(
                chunk_id=chunk_id,
                text=text,
                doc_id=doc_id,
                chunk_index=chunk_index,
            )

            for entity in entities_per_chunk.get(chunk_id, []):
                self.add_entity(entity)
                self.link_entity_to_chunk(
                    entity_name=entity.name,
                    entity_type=entity.type,
                    chunk_id=chunk_id,
                )

        logger.info(
            "Graph built for doc_id=%r source=%r (%d chunks)",
            doc_id,
            source,
            len(chunks),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return node and relationship counts for the current graph.

        Returns
        -------
        dict
            Keys: ``documents``, ``chunks``, ``entities``, ``relationships``.
        """
        queries = {
            "documents": "MATCH (n:Document) RETURN count(n) AS c",
            "chunks": "MATCH (n:Chunk) RETURN count(n) AS c",
            "entities": "MATCH (n:Entity) RETURN count(n) AS c",
            "relationships": "MATCH ()-[r]->() RETURN count(r) AS c",
        }
        stats: dict = {}
        with self._driver.session() as session:
            for key, cypher in queries.items():
                result = session.run(cypher)
                record = result.single()
                stats[key] = record["c"] if record else 0

        logger.debug("Graph stats: %s", stats)
        return stats

    def clear(self) -> None:
        """Delete every node and relationship in the database.

        .. warning::
            This is destructive and irreversible.  Use with care.
        """
        cypher = "MATCH (n) DETACH DELETE n"
        with self._driver.session() as session:
            session.run(cypher)

        logger.info("Graph cleared (all nodes and relationships deleted)")

#!/usr/bin/env python3
"""Ingest documents into the OpenSearch Docling GraphRAG pipeline.

Usage:
    python scripts/ingest.py <file_or_directory> [options]

Options:
    --skip-ner      Skip NER entity extraction (no graph building)
    --use-gpu       Enable GPU acceleration for Docling document parsing

Examples:
    python scripts/ingest.py data/sample_graphrag.txt
    python scripts/ingest.py data/sample_graphrag.txt --skip-ner
    python scripts/ingest.py ~/documents/ --use-gpu
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ingest")


def ingest_file(
    file_path: str,
    *,
    skip_ner: bool = False,
    use_gpu: bool = False,
) -> None:
    """Ingest a single file through the full pipeline."""
    from neo4j import GraphDatabase

    from opensearch_graphrag.chunker import chunk_text
    from opensearch_graphrag.config import get_settings
    from opensearch_graphrag.embedder import embed_chunks
    from opensearch_graphrag.entity_extractor import extract_entities
    from opensearch_graphrag.graph_builder import GraphBuilder
    from opensearch_graphrag.loader import load_file
    from opensearch_graphrag.opensearch_store import OpenSearchStore

    cfg = get_settings()

    # 1. Load document
    logger.info("Loading: %s (GPU=%s)", file_path, use_gpu)
    text = load_file(file_path, use_gpu=use_gpu)
    logger.info("Loaded %d characters", len(text))

    if not text.strip():
        logger.warning("Document is empty, skipping: %s", file_path)
        return

    # 2. Chunk
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        chunk.source = os.path.basename(file_path)
        chunk.chunk_index = i
    logger.info("Created %d chunks", len(chunks))

    # 3. Embed
    logger.info("Embedding %d chunks...", len(chunks))
    chunks = embed_chunks(chunks)
    logger.info("Embedding complete")

    # 4. Store in OpenSearch
    store = OpenSearchStore(settings=cfg)
    try:
        store.init_index()
    except Exception as e:
        logger.error("OpenSearch init failed: %s. Is it running? Try: docker compose up -d", e)
        sys.exit(1)

    stored = store.add_chunks(chunks)
    logger.info("Stored %d chunks in OpenSearch", stored)

    # 5. NER + Graph building (optional)
    if not skip_ner:
        driver = GraphDatabase.driver(
            cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        try:
            builder = GraphBuilder(driver)
            doc_id = os.path.basename(file_path).replace(".", "_")
            builder.add_document(doc_id, file_path)

            entities_per_chunk: dict[str, list] = {}
            for chunk in chunks:
                builder.add_chunk(chunk.id, chunk.text, doc_id, chunk.chunk_index)

                entities = extract_entities(chunk.text, chunk_id=chunk.id, settings=cfg)
                entities_per_chunk[chunk.id] = entities
                for entity in entities:
                    builder.add_entity(entity)
                    builder.link_entity_to_chunk(entity.name, entity.type, chunk.id)

                logger.info("Chunk %s: %d entities", chunk.id, len(entities))

            # Link co-occurring entities
            for chunk_id, entities in entities_per_chunk.items():
                for i, e1 in enumerate(entities):
                    for e2 in entities[i + 1:]:
                        builder.link_entities(e1.name, e2.name)

            stats = builder.get_stats()
            logger.info("Graph: %s", stats)

        finally:
            driver.close()
    else:
        logger.info("Skipping NER (--skip-ner)")

    logger.info("Done: %s", file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into OpenSearch Docling GraphRAG",
    )
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--skip-ner", action="store_true", help="Skip NER entity extraction")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU for Docling")
    args = parser.parse_args()

    target = os.path.abspath(args.path)
    if not os.path.exists(target):
        logger.error("Path does not exist: %s", target)
        sys.exit(1)

    files: list[str] = []
    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            if os.path.isfile(full) and not name.startswith("."):
                files.append(full)

    if not files:
        logger.error("No files found at: %s", target)
        sys.exit(1)

    logger.info("Ingesting %d file(s)...", len(files))
    for f in files:
        ingest_file(f, skip_ner=args.skip_ner, use_gpu=args.use_gpu)

    logger.info("All done. %d file(s) ingested.", len(files))


if __name__ == "__main__":
    main()

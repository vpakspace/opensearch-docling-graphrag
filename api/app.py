"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from api.deps import set_service
from api.routes import router

if TYPE_CHECKING:
    from opensearch_graphrag.service import PipelineService

logger = logging.getLogger(__name__)


def create_app(service: "PipelineService | None" = None) -> FastAPI:
    """Create FastAPI app with lifespan management."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            from neo4j import GraphDatabase

            from opensearch_graphrag.config import get_settings
            from opensearch_graphrag.graph_builder import GraphBuilder
            from opensearch_graphrag.opensearch_store import OpenSearchStore
            from opensearch_graphrag.service import PipelineService

            cfg = get_settings()

            store = OpenSearchStore(settings=cfg)
            try:
                store.init_index()
            except Exception as e:
                logger.warning("OpenSearch index init failed: %s", e)

            driver = GraphDatabase.driver(
                cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
            )
            graph_builder = GraphBuilder(driver)

            svc = PipelineService(
                store=store,
                neo4j_driver=driver,
                graph_builder=graph_builder,
                settings=cfg,
            )
            set_service(svc)
            yield
            driver.close()
        else:
            set_service(service)
            yield

    app = FastAPI(
        title="OpenSearch Docling GraphRAG API",
        version="0.1.0",
        description="Fully local RAG: OpenSearch + Neo4j + Ollama + Docling",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app

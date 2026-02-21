"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.deps import set_service
from api.limiter import limiter
from api.routes import router
from opensearch_graphrag.exceptions import GraphRAGError

if TYPE_CHECKING:
    from opensearch_graphrag.service import PipelineService

logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "")


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

    # Rate limiting
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    # API key auth middleware (skip /health and /docs; disabled when API_KEY is empty)
    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if API_KEY and request.url.path not in (
            "/api/v1/health", "/docs", "/openapi.json", "/redoc",
        ):
            key = request.headers.get("X-API-Key", "")
            if key != API_KEY:
                return JSONResponse(status_code=401, content={"error": "Invalid or missing API key"})
        return await call_next(request)

    app.include_router(router)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    @app.exception_handler(PydanticValidationError)
    async def pydantic_validation_handler(request: Request, exc: PydanticValidationError):
        return JSONResponse(status_code=422, content={"error": "Validation error", "detail": str(exc)})

    @app.exception_handler(GraphRAGError)
    async def graphrag_error_handler(request: Request, exc: GraphRAGError):
        logger.error("GraphRAGError: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.error("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    return app

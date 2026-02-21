"""FastAPI route handlers."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from api.deps import get_service
from api.limiter import limiter
from opensearch_graphrag.models import QAResult, SearchResult

router = APIRouter(prefix="/api/v1")

VALID_MODES = Literal["bm25", "vector", "graph", "hybrid", "enhanced", "cognitive"]


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: VALID_MODES = "hybrid"


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: VALID_MODES = "hybrid"


class HealthResponse(BaseModel):
    opensearch: bool = False
    neo4j: bool = False
    ollama: bool = False


class GraphStatsResponse(BaseModel):
    documents: int = 0
    chunks: int = 0
    entities: int = 0
    relationships: int = 0


@router.get("/health", response_model=HealthResponse)
def health():
    svc = get_service()
    return svc.health()


@router.post("/query", response_model=QAResult)
@limiter.limit("60/minute")
def query(req: QueryRequest, request: Request):
    svc = get_service()
    return svc.query(req.text, mode=req.mode)


@router.post("/search", response_model=list[SearchResult])
@limiter.limit("60/minute")
def search(req: SearchRequest, request: Request):
    svc = get_service()
    return svc.search(req.text, mode=req.mode)


@router.get("/graph/stats", response_model=GraphStatsResponse)
def graph_stats():
    svc = get_service()
    return svc.graph_stats()

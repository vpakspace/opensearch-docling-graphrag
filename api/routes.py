"""FastAPI route handlers."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from api.deps import get_service
from api.limiter import limiter

router = APIRouter(prefix="/api/v1")

VALID_MODES = Literal["bm25", "vector", "graph", "hybrid", "enhanced", "cognitive"]


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: VALID_MODES = "hybrid"


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: VALID_MODES = "hybrid"


@router.get("/health")
def health():
    svc = get_service()
    return svc.health()


@router.post("/query")
@limiter.limit("60/minute")
def query(req: QueryRequest, request: Request):
    svc = get_service()
    qa = svc.query(req.text, mode=req.mode)
    return qa.model_dump()


@router.post("/search")
@limiter.limit("60/minute")
def search(req: SearchRequest, request: Request):
    svc = get_service()
    results = svc.search(req.text, mode=req.mode)
    return [r.model_dump() for r in results]


@router.get("/graph/stats")
def graph_stats():
    svc = get_service()
    return svc.graph_stats()

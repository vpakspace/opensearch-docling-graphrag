"""FastAPI route handlers."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.deps import get_service

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
def query(req: QueryRequest):
    svc = get_service()
    qa = svc.query(req.text, mode=req.mode)
    return qa.model_dump()


@router.post("/search")
def search(req: SearchRequest):
    svc = get_service()
    results = svc.search(req.text, mode=req.mode)
    return [r.model_dump() for r in results]


@router.get("/graph/stats")
def graph_stats():
    svc = get_service()
    return svc.graph_stats()

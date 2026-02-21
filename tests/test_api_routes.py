"""Tests for FastAPI routes."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opensearch_graphrag.models import QAResult, SearchResult


@pytest.fixture
def client_and_svc():
    from api.app import create_app

    svc = MagicMock()
    svc.health.return_value = {"status": "ok", "opensearch": True, "neo4j": True, "ollama": True}
    svc.query.return_value = QAResult(answer="Test", confidence=0.9, mode="hybrid")
    svc.search.return_value = [SearchResult(chunk_id="c1", text="hello", score=0.9)]
    svc.graph_stats.return_value = {"documents": 1, "chunks": 5, "entities": 10, "relationships": 8}

    app = create_app(service=svc)
    with TestClient(app) as client:
        yield client, svc


def test_health(client_and_svc):
    client, svc = client_and_svc
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    svc.health.assert_called_once()


def test_query(client_and_svc):
    client, svc = client_and_svc
    resp = client.post("/api/v1/query", json={"text": "What is OpenSearch?", "mode": "hybrid"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test"
    assert data["mode"] == "hybrid"


def test_query_empty_text(client_and_svc):
    client, _ = client_and_svc
    resp = client.post("/api/v1/query", json={"text": "", "mode": "hybrid"})
    assert resp.status_code == 422


def test_search(client_and_svc):
    client, svc = client_and_svc
    resp = client.post("/api/v1/search", json={"text": "test query", "mode": "bm25"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1


def test_graph_stats(client_and_svc):
    client, svc = client_and_svc
    resp = client.get("/api/v1/graph/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entities"] == 10
    assert data["documents"] == 1

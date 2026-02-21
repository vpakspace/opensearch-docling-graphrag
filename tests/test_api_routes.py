"""Tests for FastAPI routes."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opensearch_graphrag.exceptions import GraphRAGError
from opensearch_graphrag.models import QAResult, SearchResult


def _make_svc():
    svc = MagicMock()
    svc.health.return_value = {"status": "ok", "opensearch": True, "neo4j": True, "ollama": True}
    svc.query.return_value = QAResult(answer="Test", confidence=0.9, mode="hybrid")
    svc.search.return_value = [SearchResult(chunk_id="c1", text="hello", score=0.9)]
    svc.graph_stats.return_value = {"documents": 1, "chunks": 5, "entities": 10, "relationships": 8}
    return svc


@pytest.fixture
def client_and_svc():
    from api.app import create_app

    svc = _make_svc()
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


# ── API key auth tests ────────────────────────────────────────


@pytest.fixture
def auth_client():
    """Client with API_KEY enabled."""
    import api.app as app_module

    original_key = app_module.API_KEY
    app_module.API_KEY = "test-secret-key"
    try:
        from api.app import create_app

        svc = _make_svc()
        app = create_app(service=svc)
        with TestClient(app) as client:
            yield client
    finally:
        app_module.API_KEY = original_key


def test_auth_no_key_returns_401(auth_client):
    resp = auth_client.post("/api/v1/query", json={"text": "test", "mode": "hybrid"})
    assert resp.status_code == 401
    assert "API key" in resp.json()["error"]


def test_auth_correct_key_returns_200(auth_client):
    resp = auth_client.post(
        "/api/v1/query",
        json={"text": "test", "mode": "hybrid"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 200


def test_auth_wrong_key_returns_401(auth_client):
    resp = auth_client.post(
        "/api/v1/query",
        json={"text": "test", "mode": "hybrid"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_auth_health_no_key_returns_200(auth_client):
    """Health endpoint should always be accessible without auth."""
    resp = auth_client.get("/api/v1/health")
    assert resp.status_code == 200


# ── Exception handler tests ──────────────────────────────────


def test_graphrag_error_returns_json_500(client_and_svc):
    """GraphRAGError should return JSON 500 without traceback."""
    client, svc = client_and_svc
    svc.query.side_effect = GraphRAGError("Pipeline failed")
    resp = client.post("/api/v1/query", json={"text": "test", "mode": "hybrid"})
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
    assert "Traceback" not in data["error"]

"""Unit and integration tests for FastAPI search endpoints."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Fixture providing a FastAPI TestClient with initialized lifespan models."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_serves_html(client: TestClient) -> None:
    """Verify GET / returns the frontend HTML page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Gallica" in response.text
    assert "<!DOCTYPE html>" in response.text


def test_health_check(client: TestClient) -> None:
    """Verify GET /api/health returns operational status and index count."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "total_indexed" in data
    assert isinstance(data["total_indexed"], int)


def test_search_multilingual_query(client: TestClient) -> None:
    """Verify GET /api/search returns ranked results for a query."""
    response = client.get("/api/search", params={"q": "soldat 1914", "top_k": 3})
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "soldat 1914"
    assert "latency_ms" in data
    assert isinstance(data["results"], list)

    if data["count"] > 0:
        top_hit = data["results"][0]
        assert "id" in top_hit
        assert "similarity" in top_hit
        assert "calibrated_score" in top_hit
        assert "is_confident" in top_hit
        assert "image_url" in top_hit
        assert top_hit["image_url"].startswith("/static/crops/")


def test_search_min_similarity_filter(client: TestClient) -> None:
    """Verify min_similarity filters out results below the given threshold."""
    # With a high threshold (e.g. 0.99), no result should be returned
    response = client.get("/api/search", params={"q": "soldat", "min_similarity": 0.99})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert len(data["results"]) == 0


def test_search_missing_query_validation(client: TestClient) -> None:
    """Verify searching without query parameter returns 422 Unprocessable Entity."""
    response = client.get("/api/search")
    assert response.status_code == 422

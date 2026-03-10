"""API integration tests using httpx AsyncClient."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest_asyncio.fixture
async def client():
    """Create an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Health endpoint should return 200 with status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "message" in data


@pytest.mark.asyncio
async def test_health_has_request_id(client: AsyncClient):
    """All responses should include X-Request-ID header."""
    response = await client.get("/health")
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_search_endpoint(client: AsyncClient):
    """POST /search with valid query returns results."""
    response = await client.post(
        "/search",
        json={"query": "running shoes", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 5
    assert len(data["results"]) == 5
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_search_empty_query_returns_422(client: AsyncClient):
    """POST /search with empty query returns 422 validation error."""
    response = await client.post(
        "/search",
        json={"query": "", "top_k": 5},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_invalid_top_k_returns_422(client: AsyncClient):
    """POST /search with top_k=0 returns 422."""
    response = await client.post(
        "/search",
        json={"query": "shoes", "top_k": 0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_assistant_endpoint(client: AsyncClient):
    """POST /assistant/ask returns answer with source IDs."""
    response = await client.post(
        "/assistant/ask",
        json={"question": "What laptops do you have?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert isinstance(data["source_product_ids"], list)
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_assistant_empty_question_returns_422(client: AsyncClient):
    """POST /assistant/ask with empty question returns 422."""
    response = await client.post(
        "/assistant/ask",
        json={"question": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_similar_products_endpoint(client: AsyncClient):
    """GET /recommendations/similar/{product_id} returns results."""
    response = await client.get("/recommendations/similar/P0001?top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] == 3


@pytest.mark.asyncio
async def test_similar_products_not_found(client: AsyncClient):
    """GET /recommendations/similar with invalid ID returns 404."""
    response = await client.get("/recommendations/similar/P9999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_user_recommendations_endpoint(client: AsyncClient):
    """GET /recommendations/user/{user_id} returns results."""
    response = await client.get("/recommendations/user/U001?top_k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0


@pytest.mark.asyncio
async def test_cold_start_endpoint(client: AsyncClient):
    """GET /recommendations/cold-start returns results."""
    response = await client.get("/recommendations/cold-start")
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0


@pytest.mark.asyncio
async def test_cold_start_with_category(client: AsyncClient):
    """GET /recommendations/cold-start?category=Electronics returns filtered results."""
    response = await client.get("/recommendations/cold-start?category=Electronics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_results"] > 0

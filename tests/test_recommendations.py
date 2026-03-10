"""Tests for recommendation engine."""

import pytest

from recommendations.recommender import (
    get_cold_start_recommendations,
    get_similar_products,
    get_user_recommendations,
)


@pytest.mark.asyncio
async def test_similar_products_returns_correct_count():
    """Should return exactly top_k similar products."""
    results = await get_similar_products("P0001", top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_similar_products_excludes_self():
    """The input product should not appear in results."""
    results = await get_similar_products("P0001", top_k=5)
    result_ids = [r["product_id"] for r in results]
    assert "P0001" not in result_ids


@pytest.mark.asyncio
async def test_similar_products_has_required_fields():
    """Each result should have required fields."""
    results = await get_similar_products("P0001", top_k=3)
    for r in results:
        assert "product_id" in r
        assert "title" in r
        assert "category" in r
        assert "price" in r
        assert "brand" in r
        assert "similarity_score" in r


@pytest.mark.asyncio
async def test_similar_products_invalid_id_raises():
    """Nonexistent product_id should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await get_similar_products("P9999")


@pytest.mark.asyncio
async def test_user_recommendations_with_history():
    """User with interactions should get personalized recommendations."""
    results = await get_user_recommendations("U001", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5


@pytest.mark.asyncio
async def test_user_recommendations_no_history():
    """User with no interactions should get cold-start recommendations."""
    results = await get_user_recommendations("U999", top_k=5)
    assert len(results) > 0  # Should fall back to cold start


@pytest.mark.asyncio
async def test_cold_start_no_category():
    """Cold start without category returns most purchased products."""
    results = await get_cold_start_recommendations()
    assert len(results) == 5


@pytest.mark.asyncio
async def test_cold_start_with_category():
    """Cold start with category returns products from that category."""
    results = await get_cold_start_recommendations(category="Electronics")
    assert len(results) > 0
    for r in results:
        assert r["category"] == "Electronics"


@pytest.mark.asyncio
async def test_cold_start_invalid_category_returns_empty():
    """Cold start with nonexistent category returns empty list."""
    results = await get_cold_start_recommendations(category="NonexistentCategory")
    assert results == []

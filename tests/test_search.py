"""Tests for hybrid search engine."""

import pytest

from search.hybrid_search import hybrid_search


@pytest.mark.asyncio
async def test_hybrid_search_returns_correct_count():
    """Search returns exactly top_k results with correct structure."""
    results = await hybrid_search("running shoes", top_k=5)
    assert len(results) == 5
    for r in results:
        assert "product_id" in r
        assert "title" in r
        assert "final_score" in r


@pytest.mark.asyncio
async def test_hybrid_search_scores_in_range():
    """All normalized scores should be in [0, 1]."""
    results = await hybrid_search("wireless headphones", top_k=5)
    for r in results:
        assert 0.0 <= r["semantic_score"] <= 1.0
        assert 0.0 <= r["bm25_score"] <= 1.0
        assert 0.0 <= r["final_score"] <= 1.0


@pytest.mark.asyncio
async def test_hybrid_search_sorted_by_final_score():
    """Results should be sorted by final_score descending."""
    results = await hybrid_search("laptop for machine learning", top_k=10)
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_hybrid_search_top_k_one():
    """top_k=1 returns exactly 1 result."""
    results = await hybrid_search("yoga mat", top_k=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_hybrid_search_empty_query_raises():
    """Empty query should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        await hybrid_search("")


@pytest.mark.asyncio
async def test_hybrid_search_long_query_raises():
    """Query exceeding max length should raise ValueError."""
    long_query = "x" * 501
    with pytest.raises(ValueError, match="maximum length"):
        await hybrid_search(long_query)


@pytest.mark.asyncio
async def test_hybrid_search_results_have_required_fields():
    """Each result should contain all expected fields."""
    results = await hybrid_search("summer dress", top_k=3)
    required_fields = {"product_id", "title", "category", "price", "brand", "description",
                       "semantic_score", "bm25_score", "final_score"}
    for r in results:
        assert required_fields.issubset(r.keys())

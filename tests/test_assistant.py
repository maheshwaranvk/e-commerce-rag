"""Tests for RAG shopping assistant."""

import pytest

from assistant.rag_assistant import ask_assistant


@pytest.mark.asyncio
async def test_ask_assistant_returns_answer():
    """Assistant returns a non-empty answer with source products."""
    result = await ask_assistant("What laptops do you have?")
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert isinstance(result["source_products"], list)


@pytest.mark.asyncio
async def test_ask_assistant_source_products_are_unique():
    """Source product IDs should be deduplicated."""
    result = await ask_assistant("Compare running shoes under 5000 rupees")
    source_ids = result["source_products"]
    assert len(source_ids) == len(set(source_ids))


@pytest.mark.asyncio
async def test_ask_assistant_empty_question_raises():
    """Empty question should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        await ask_assistant("")


@pytest.mark.asyncio
async def test_ask_assistant_long_question_raises():
    """Question exceeding max length should raise ValueError."""
    long_question = "Tell me about " * 100
    with pytest.raises(ValueError, match="maximum length"):
        await ask_assistant(long_question)


@pytest.mark.asyncio
async def test_ask_assistant_out_of_catalog_question():
    """Question about unavailable products should get a polite decline."""
    result = await ask_assistant("Do you sell helicopters?")
    answer = result["answer"].lower()
    # Should contain some variation of "don't have" or "not in catalog"
    assert any(phrase in answer for phrase in [
        "don't have", "not in", "don't have information", "no information",
        "don't carry", "not available", "cannot find",
    ])

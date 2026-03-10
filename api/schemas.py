"""
Pydantic v2 request/response schemas for the API with input validation.
"""

from pydantic import BaseModel, Field


# --- Requests ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")


class AssistantRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question for the assistant")


# --- Response items ---

class ProductResult(BaseModel):
    product_id: str
    title: str
    category: str
    price: float
    brand: str
    description: str | None = None
    semantic_score: float | None = None
    bm25_score: float | None = None
    final_score: float | None = None


class RecommendationResult(BaseModel):
    product_id: str
    title: str
    category: str
    price: float
    brand: str
    similarity_score: float


# --- Responses ---

class SearchResponse(BaseModel):
    query: str
    results: list[ProductResult]
    total_results: int
    latency_ms: float


class AssistantResponse(BaseModel):
    question: str
    answer: str
    source_product_ids: list[str]
    latency_ms: float


class RecommendationResponse(BaseModel):
    results: list[RecommendationResult]
    total_results: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    message: str
    redis_connected: bool = False


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None

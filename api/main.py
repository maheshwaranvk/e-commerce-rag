"""
FastAPI application for the AI Commerce Platform.

Endpoints:
  GET  /health                              — Health check
  POST /search                              — Hybrid product search
  POST /assistant/ask                       — RAG shopping assistant
  GET  /recommendations/similar/{product_id} — Similar products
  GET  /recommendations/user/{user_id}       — User-based recommendations
  GET  /recommendations/cold-start           — Cold-start recommendations
"""

import logging
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import lifespan
from api.schemas import (
    AssistantRequest,
    AssistantResponse,
    ErrorResponse,
    HealthResponse,
    RecommendationResponse,
    SearchRequest,
    SearchResponse,
)
from cache.redis_cache import (
    TTL_ASSISTANT,
    TTL_RECOMMENDATIONS,
    TTL_SEARCH,
    get_cached,
    is_connected,
    make_cache_key,
    set_cached,
)

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_commerce")


# --- App ---
app = FastAPI(
    title="AI Commerce Platform",
    description="Hybrid search, RAG assistant, and recommendation engine for e-commerce",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request ID + Logging Middleware ---
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    response: Response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s | %d | %.1fms | %s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
    )
    return response


# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled error [%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error. Please try again later.",
            request_id=request_id,
        ).model_dump(),
    )


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    redis_ok = await is_connected()
    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        message="AI Commerce Platform is running"
        + ("" if redis_ok else " (Redis unavailable)"),
        redis_connected=redis_ok,
    )


@app.post("/search", response_model=SearchResponse)
async def search_products(body: SearchRequest):
    """Hybrid semantic + keyword product search."""
    # Check cache
    cache_key = make_cache_key("search", query=body.query.lower().strip(), top_k=body.top_k)
    cached = await get_cached(cache_key)
    if cached:
        logger.info("Cache HIT for search: %s", body.query[:100])
        return SearchResponse(**cached)

    try:
        from search.hybrid_search import hybrid_search_with_latency

        results, latency_ms = await hybrid_search_with_latency(body.query, body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = SearchResponse(
        query=body.query,
        results=results,
        total_results=len(results),
        latency_ms=round(latency_ms, 1),
    )

    # Cache the result
    await set_cached(cache_key, response.model_dump(), ttl=TTL_SEARCH)

    return response


@app.post("/assistant/ask", response_model=AssistantResponse)
async def ask_assistant_endpoint(body: AssistantRequest):
    """RAG-based shopping assistant."""
    cache_key = make_cache_key("assistant", question=body.question.lower().strip())
    cached = await get_cached(cache_key)
    if cached:
        logger.info("Cache HIT for assistant: %s", body.question[:100])
        return AssistantResponse(**cached)

    try:
        from assistant.rag_assistant import ask_assistant

        start = time.perf_counter()
        result = await ask_assistant(body.question)
        latency_ms = (time.perf_counter() - start) * 1000
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = AssistantResponse(
        question=body.question,
        answer=result["answer"],
        source_product_ids=result["source_products"],
        latency_ms=round(latency_ms, 1),
    )

    await set_cached(cache_key, response.model_dump(), ttl=TTL_ASSISTANT)

    return response


@app.get("/recommendations/similar/{product_id}", response_model=RecommendationResponse)
async def similar_products(
    product_id: str,
    top_k: int = Query(default=5, ge=1, le=50),
):
    """Get products similar to a given product."""
    cache_key = make_cache_key("similar", product_id=product_id, top_k=top_k)
    cached = await get_cached(cache_key)
    if cached:
        return RecommendationResponse(**cached)

    try:
        from recommendations.recommender import get_similar_products

        start = time.perf_counter()
        results = await get_similar_products(product_id, top_k)
        latency_ms = (time.perf_counter() - start) * 1000
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = RecommendationResponse(
        results=results,
        total_results=len(results),
        latency_ms=round(latency_ms, 1),
    )

    await set_cached(cache_key, response.model_dump(), ttl=TTL_RECOMMENDATIONS)

    return response


@app.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
async def user_recommendations(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=50),
):
    """Get personalized recommendations for a user."""
    cache_key = make_cache_key("user_rec", user_id=user_id, top_k=top_k)
    cached = await get_cached(cache_key)
    if cached:
        return RecommendationResponse(**cached)

    try:
        from recommendations.recommender import get_user_recommendations

        start = time.perf_counter()
        results = await get_user_recommendations(user_id, top_k)
        latency_ms = (time.perf_counter() - start) * 1000
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = RecommendationResponse(
        results=results,
        total_results=len(results),
        latency_ms=round(latency_ms, 1),
    )

    await set_cached(cache_key, response.model_dump(), ttl=TTL_RECOMMENDATIONS)

    return response


@app.get("/recommendations/cold-start", response_model=RecommendationResponse)
async def cold_start_recommendations(
    category: str | None = Query(default=None),
):
    """Get cold-start recommendations (popular or by category)."""
    cache_key = make_cache_key("cold_start", category=category or "all")
    cached = await get_cached(cache_key)
    if cached:
        return RecommendationResponse(**cached)

    try:
        from recommendations.recommender import get_cold_start_recommendations

        start = time.perf_counter()
        results = await get_cold_start_recommendations(category)
        latency_ms = (time.perf_counter() - start) * 1000
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    response = RecommendationResponse(
        results=results,
        total_results=len(results),
        latency_ms=round(latency_ms, 1),
    )

    await set_cached(cache_key, response.model_dump(), ttl=TTL_RECOMMENDATIONS)

    return response


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

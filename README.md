# AI Commerce Platform (E-commerce RAG)

Hybrid search + RAG shopping assistant + recommendations, exposed via a FastAPI API with Redis caching, rate limiting, and Prometheus metrics.

## What this service does

- **Hybrid product search**: combines semantic similarity (FAISS + OpenAI embeddings) with keyword relevance (BM25) to return ranked products.
- **RAG shopping assistant**: answers catalog questions using retrieved product context (LangChain + FAISS vectorstore) and an LLM.
- **Recommendations**:
  - similar-products (content-based, FAISS neighbor search)
  - user-based recommendations (based on interaction history)
  - cold-start recommendations (popular or category-based)
- **Production-minded API concerns**: request IDs, structured logging, per-endpoint rate limits, Redis caching with graceful degradation.
- **Monitoring**: Prometheus metrics on `/metrics` for latency, errors, cache hit-rate, and LLM token usage.

## Architecture (high level)

1. **Search** (`POST /search`)
  - Embed query with OpenAI embeddings
  - Semantic top-N from FAISS + keyword scores from BM25
  - Normalize and combine into a final score
2. **Assistant** (`POST /assistant/ask`)
  - Retrieve candidate products from a LangChain FAISS vectorstore
  - Apply lightweight heuristics (category + budget extraction) to pick the best context
  - Ask an LLM to answer strictly from that context
3. **Recommendations** (`/recommendations/*`)
  - Similar items via FAISS neighbors
  - User recommendations via interaction-weighted history + merged neighbor searches

## Tech stack

**Runtime**
- Python 3.11
- FastAPI + Uvicorn
- Pydantic v2

**Search**
- FAISS (`faiss-cpu`) for vector similarity search
- OpenAI embeddings (`text-embedding-3-small`) for semantic search
- BM25 (`rank_bm25`) for keyword relevance
- Hybrid scoring: $0.6 \cdot \text{semantic} + 0.4 \cdot \text{bm25}$

**Assistant (RAG)**
- LangChain (`langchain`, `langchain-openai`, `langchain-community`)
- LangChain FAISS vectorstore persisted under `assistant/vectorstore/`
- LLM: `gpt-4o-mini` (configured in code)

**Recommendations**
- Content-based similarities from the same FAISS index used by search
- Simple user-personalization from interaction history (`data/user_interactions.csv`)

**Caching & reliability**
- Redis (async client via `redis>=5`)
- Graceful degradation: if Redis is down, the API continues without caching

**Rate limiting**
- `slowapi` (per-IP)

**Monitoring**
- `prometheus-client`

**Testing**
- `pytest`, `pytest-asyncio`, `httpx`

## Repository layout

- `api/`: FastAPI app and schemas
- `search/`: embedding + BM25 index build scripts, hybrid retrieval
- `assistant/`: RAG assistant + persisted LangChain vectorstore
- `recommendations/`: recommendation engine
- `cache/`: Redis caching layer
- `monitoring/`: Prometheus metrics helpers
- `data/`: synthetic dataset generator + CSV data
- `tests/`: unit/integration tests

## Configuration

Create `.env` (or copy from `.env.example`):

```bash
OPENAI_API_KEY=... 
REDIS_URL=redis://localhost:6379
```

Notes:
- `OPENAI_API_KEY` is required for **search** and **assistant** (embeddings + LLM calls).
- `REDIS_URL` is optional. If Redis is unreachable, the API runs with caching disabled.

## Setup (local)

### 1) Install dependencies

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Ensure required data and indexes exist

On startup, the API verifies these artifacts exist (see `api/dependencies.py`):
- `data/products.csv`
- `data/user_interactions.csv`
- `search/faiss_index.bin`
- `search/product_id_map.json`
- `search/bm25_index.pkl`
- `assistant/vectorstore/`

This repo already contains these artifacts. If you regenerate `data/products.csv`, rebuild indexes:

```bash
python data/generate_products.py
python search/embed_products.py
python search/bm25_index.py
python assistant/rag_assistant.py
```

### 3) Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:
- http://localhost:8000
- Swagger UI: http://localhost:8000/docs

## Setup (Docker)

### Run API + Redis

```bash
docker compose up --build
```

Services:
- API: http://localhost:8000
- Redis: localhost:6379

The container health check hits `GET /health`.

## API Endpoints

All endpoints:
- include an `X-Request-ID` response header
- are rate limited (per-IP)
- emit Prometheus metrics

CORS is configured as `allow_origins=["*"]` for convenience.

### `GET /health`

Health check and Redis status.

**Rate limit**: `1000/minute`

**Response**

```json
{
  "status": "ok",
  "message": "AI Commerce Platform is running",
  "redis_connected": true
}
```

If Redis is unavailable, `status` becomes `degraded`.

---

### `GET /metrics`

Prometheus metrics endpoint.

**Rate limit**: `60/minute`

**Response**: Prometheus text format.

---

### `POST /search`

Hybrid semantic + keyword product search.

**Rate limit**: `30/minute`

**Caching**: enabled (Redis). TTL: 1 hour.

**Request body**

```json
{
  "query": "wireless headphones for gym",
  "top_k": 5
}
```

Constraints:
- `query`: 1..500 chars
- `top_k`: 1..50

**Response**

```json
{
  "query": "wireless headphones for gym",
  "results": [
    {
      "product_id": "P0001",
      "title": "...",
      "category": "Electronics",
      "price": 1999.0,
      "brand": "...",
      "description": "...",
      "semantic_score": 0.87,
      "bm25_score": 0.54,
      "final_score": 0.74
    }
  ],
  "total_results": 5,
  "latency_ms": 123.4
}
```

**Errors**
- `422`: schema validation (e.g., empty query, invalid `top_k`)
- `400`: invalid query (e.g., whitespace-only) / query too long
- `503`: missing index/data files

---

### `POST /assistant/ask`

RAG-based shopping assistant: retrieves relevant products and answers using those as context.

Retrieval details (from the implementation):
- Retrieves up to 10 candidate products by semantic similarity.
- Attempts to infer **categories** and a **budget/price range** from the question.
- Filters candidates using those constraints (with a fallback if filtering yields no results).
- Sends the final context (up to 5 products) to the LLM.

**Rate limit**: `5/minute` (expensive endpoint)

**Caching**: enabled (Redis). TTL: 30 minutes.

**Request body**

```json
{
  "question": "Compare running shoes under 5000 rupees"
}
```

Constraints:
- `question`: 1..1000 chars

**Response**

```json
{
  "question": "Compare running shoes under 5000 rupees",
  "answer": "...",
  "source_product_ids": ["P0042", "P0310"],
  "latency_ms": 456.7
}
```

**Errors**
- `422`: schema validation (e.g., empty question)
- `400`: invalid question (e.g., whitespace-only) / too long
- `503`: missing vectorstore/data

---

### `GET /recommendations/similar/{product_id}`

Content-based similar products from FAISS neighbors.

**Rate limit**: `20/minute`

**Caching**: enabled (Redis). TTL: 2 hours.

**Query params**
- `top_k` (default 5, range 1..50)

**Response**

```json
{
  "results": [
    {
      "product_id": "P0002",
      "title": "...",
      "category": "Footwear",
      "price": 3499.0,
      "brand": "...",
      "similarity_score": 0.8123
    }
  ],
  "total_results": 5,
  "latency_ms": 12.3
}
```

**Errors**
- `404`: product not found in index
- `422`: invalid `top_k`
- `503`: missing index/data files

---

### `GET /recommendations/user/{user_id}`

Personalized recommendations based on the user’s interaction history.

**Rate limit**: `20/minute`

**Caching**: enabled (Redis). TTL: 2 hours.

**Query params**
- `top_k` (default 5, range 1..50)

Behavior:
- If the user has interactions, the engine selects up to 3 most-interacted products (weighted purchase > click > view), searches neighbors per product, and merges scores.
- If the user has no interactions, it falls back to cold-start.

**Errors**
- `422`: invalid `top_k`
- `503`: missing index/data files

---

### `GET /recommendations/cold-start`

Recommendations for new users.

**Rate limit**: `50/minute`

**Caching**: enabled (Redis). TTL: 2 hours.

**Query params**
- `category` (optional): if provided, returns products from that category; otherwise returns most purchased products

**Errors**
- `503`: missing index/data files

---

### Common API error formats

**Rate limiting (`429`)** returns:

```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "request_id": "..."
}
```

**Unhandled errors (`500`)** return:

```json
{
  "detail": "Internal server error. Please try again later.",
  "request_id": "..."
}
```

## Rate limits

Configured in `api/dependencies.py`:

- `GET /health`: `1000/minute`
- `POST /search`: `30/minute`
- `POST /assistant/ask`: `5/minute`
- `GET /recommendations/similar/{product_id}`: `20/minute`
- `GET /recommendations/user/{user_id}`: `20/minute`
- `GET /recommendations/cold-start`: `50/minute`
- `GET /metrics`: `60/minute`

Operational note: rate limiting uses the client IP from FastAPI/Starlette request info. If you run behind a reverse proxy/load balancer, you may need to ensure `X-Forwarded-For` handling is configured appropriately.

## Caching (Redis)

Caching is implemented in `cache/redis_cache.py` and is best-effort:
- If Redis is down, cache operations are skipped and requests still succeed.
- Cache keys are deterministic and hashed.

TTL defaults:
- search: 3600s
- assistant: 1800s
- recommendations: 7200s

Endpoints using caching:
- `POST /search`
- `POST /assistant/ask`
- `GET /recommendations/similar/{product_id}`
- `GET /recommendations/user/{user_id}`
- `GET /recommendations/cold-start`

## Monitoring

### Prometheus metrics

Metrics are defined in `monitoring/metrics.py` and exported on `GET /metrics`.

Available metrics:
- `api_requests_total{method,path,status}`
- `api_request_latency_ms_bucket{method,path,le}` (+ `_sum`, `_count`)
- `api_errors_total{method,path,status}` (HTTP status >= 400)
- `api_cache_hits_total{endpoint}`
- `api_cache_misses_total{endpoint}`
- `llm_tokens_total{model,type}` where `type` is `prompt|completion|total`

### Example Prometheus scrape config

```yaml
scrape_configs:
  - job_name: ai-commerce-api
    static_configs:
      - targets: ["localhost:8000"]
```

### Logging and request IDs

The API logs one line per request including latency and cache hit state.
Every response includes `X-Request-ID` to correlate logs and clients.

## Postman

Import the collection in `postman_collection.json`.

- It defines a `base_url` variable (default: `http://localhost:8000`).
- Includes example requests for health, metrics, search, assistant, and recommendations.

## Tests

```bash
pytest -q
```

Notes:
- Tests import and exercise real modules; the assistant and hybrid search may call OpenAI APIs.
- Ensure `OPENAI_API_KEY` is set and artifacts exist, or tests may fail.

## Troubleshooting

- API fails on startup with “Missing required files…”
  - Build/rebuild artifacts using the commands in “Ensure required data and indexes exist”.
- `GET /health` returns `degraded`
  - Redis is unreachable; requests still work but caching is disabled.
- Frequent `429` responses
  - You’re hitting per-IP rate limits; see “Rate limits”.
- Search/assistant errors related to OpenAI
  - Confirm `OPENAI_API_KEY` is set and valid; embeddings/LLM calls require outbound network access.

## Build/rebuild notes

- If `data/products.csv` changes, rebuild:
  - `search/faiss_index.bin` + `search/product_id_map.json` + `search/index_metadata.json` via `python search/embed_products.py`
  - `search/bm25_index.pkl` via `python search/bm25_index.py`
  - `assistant/vectorstore/` via `python assistant/rag_assistant.py`
- `search/hybrid_search.py` warns if `products.csv` hash does not match `search/index_metadata.json`.

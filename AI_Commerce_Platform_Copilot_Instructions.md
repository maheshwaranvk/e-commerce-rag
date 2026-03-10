# 🛒 AI Commerce Platform — GitHub Copilot Implementation Guide
> **For:** Sorim.AI Technical Assessment | **Stack:** Python, FastAPI, OpenAI, FAISS, LangChain, Docker

---

## 📁 Project Folder Structure to Create

```
ai-commerce-platform/
├── data/
│   └── generate_products.py        # Script to generate synthetic product data
├── search/
│   ├── embed_products.py           # Embed all products and store in FAISS
│   ├── bm25_index.py               # BM25 keyword index builder
│   └── hybrid_search.py            # Combines semantic + keyword search
├── assistant/
│   └── rag_assistant.py            # RAG-based shopping assistant using LangChain
├── recommendations/
│   └── recommender.py              # Content-based recommendation engine
├── api/
│   └── main.py                     # FastAPI app with all endpoints
├── tests/
│   └── test_all.py                 # Basic tests for each component
├── .env                            # API keys (never commit this)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚙️ STEP 0 — Environment Setup

### `requirements.txt`
```
openai>=1.0.0
faiss-cpu
rank_bm25
langchain
langchain-openai
langchain-community
fastapi
uvicorn
python-dotenv
pandas
numpy
pydantic
faker
scikit-learn
tiktoken
```

### `.env` file
```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📦 STEP 1 — Generate Synthetic Product Data

### File: `data/generate_products.py`

**Copilot Prompt:**
> Write a Python script using the `Faker` library and `random` module to generate a CSV file called `products.csv` with 1000 products. Each product must have these fields:
> - `product_id` (string: "P0001" to "P1000")
> - `title` (realistic product name, e.g. "Nike Air Max Running Shoes")
> - `description` (2–3 sentence product description, 50–80 words)
> - `category` (one of: Electronics, Clothing, Footwear, Books, Sports, Home, Beauty, Toys)
> - `price` (float between 199 and 49999, representing Indian Rupees)
> - `brand` (realistic brand name)
> - `attributes` (JSON string with keys like color, size, material, warranty)
>
> Also generate a second CSV file `user_interactions.csv` with 5000 rows simulating user behavior:
> - `user_id` (string: "U001" to "U200")
> - `product_id` (random product from products.csv)
> - `interaction_type` (one of: view, click, purchase — weighted 70%, 20%, 10%)
> - `timestamp` (random datetime in last 90 days)
>
> Save both files to the `data/` folder. Print a confirmation when done.

---

## 🔍 STEP 2 — Hybrid Search Engine

### STEP 2A — Embed Products into FAISS

### File: `search/embed_products.py`

**Copilot Prompt:**
> Write a Python script that:
> 1. Loads `data/products.csv` using pandas
> 2. For each product, creates a combined text string: `f"{title}. {description}. Category: {category}. Brand: {brand}."`
> 3. Uses OpenAI `text-embedding-3-small` model to embed all products in batches of 100
> 4. Stores all embeddings as a numpy array
> 5. Saves the FAISS index to `search/faiss_index.bin` using `faiss.write_index()`
> 6. Saves a mapping file `search/product_id_map.json` — a list of product_ids in the same order as the FAISS index
> 7. Loads OPENAI_API_KEY from `.env` using `python-dotenv`
> 8. Prints progress every 100 products and total time taken at end
>
> Use `openai.OpenAI()` client style (not the old `openai.Embedding.create` style).

---

### STEP 2B — BM25 Keyword Index

### File: `search/bm25_index.py`

**Copilot Prompt:**
> Write a Python module `bm25_index.py` that:
> 1. Loads `data/products.csv`
> 2. Creates a combined text for each product: title + description + category + brand (all lowercased)
> 3. Tokenizes each text by splitting on spaces (simple tokenization)
> 4. Builds a BM25 index using the `rank_bm25` library (`BM25Okapi`)
> 5. Saves the BM25 index to `search/bm25_index.pkl` using `pickle`
> 6. Exposes a function `get_bm25_scores(query: str) -> list[float]` that:
>    - Tokenizes the query
>    - Returns BM25 scores for all documents as a list
>
> Make the module importable (use `if __name__ == "__main__":` guard for the build step).

---

### STEP 2C — Hybrid Search Logic

### File: `search/hybrid_search.py`

**Copilot Prompt:**
> Write a Python module `hybrid_search.py` that combines semantic and keyword search. It should:
>
> **Imports and setup:**
> - Load the FAISS index from `search/faiss_index.bin`
> - Load `search/product_id_map.json`
> - Load `data/products.csv` as a pandas DataFrame
> - Load the BM25 index from `search/bm25_index.pkl`
> - Load OPENAI_API_KEY from `.env`
>
> **Main function: `hybrid_search(query: str, top_k: int = 10) -> list[dict]`**
>
> Inside this function:
> 1. **Semantic Search:**
>    - Embed the query using OpenAI `text-embedding-3-small`
>    - Search FAISS index for top 50 results
>    - Normalize semantic scores to range [0, 1] using min-max normalization
>
> 2. **Keyword Search (BM25):**
>    - Get BM25 scores for all products using the query
>    - Normalize BM25 scores to range [0, 1] using min-max normalization
>
> 3. **Hybrid Scoring:**
>    - For each product, compute: `final_score = 0.6 * semantic_score + 0.4 * bm25_score`
>    - Sort all products by `final_score` descending
>    - Return top_k results
>
> 4. **Return format** — each result should be a dict:
> ```python
> {
>   "product_id": "P0042",
>   "title": "...",
>   "category": "...",
>   "price": 1999.0,
>   "brand": "...",
>   "description": "...",
>   "semantic_score": 0.87,
>   "bm25_score": 0.54,
>   "final_score": 0.74
> }
> ```
>
> Also add a `measure_latency()` wrapper that times the search and prints milliseconds taken.

---

## 🤖 STEP 3 — RAG Shopping Assistant

### File: `assistant/rag_assistant.py`

**Copilot Prompt:**
> Write a Python module `rag_assistant.py` using LangChain that builds a RAG-based shopping assistant.
>
> **Setup:**
> - Load products from `data/products.csv`
> - For each product, create a LangChain `Document` object where:
>   - `page_content` = `f"Product: {title}\nBrand: {brand}\nCategory: {category}\nPrice: ₹{price}\nDescription: {description}\nAttributes: {attributes}"`
>   - `metadata` = `{"product_id": ..., "category": ..., "price": ...}`
> - Use `OpenAIEmbeddings(model="text-embedding-3-small")` for embeddings
> - Use `FAISS.from_documents()` to create a LangChain FAISS vectorstore
> - Save the vectorstore locally using `vectorstore.save_local("assistant/vectorstore")`
>
> **RAG Chain:**
> - Use `ChatOpenAI(model="gpt-4o-mini", temperature=0.3)`
> - Build a retriever: `vectorstore.as_retriever(search_kwargs={"k": 5})`
> - Use this exact system prompt:
>
> ```
> You are a helpful shopping assistant for an e-commerce platform.
> Your job is to help users find products and answer product-related questions.
>
> RULES:
> - Only answer based on the product context provided below.
> - If the context does not contain relevant products, say: "I don't have information about that in our catalog."
> - Never make up product names, prices, or specifications.
> - When comparing products, be factual and list pros/cons based on the description.
> - Always mention the price in Indian Rupees (₹).
>
> CONTEXT:
> {context}
> ```
>
> - Build a `RetrievalQA` chain or LCEL chain that takes a `question` and returns an `answer`
>
> **Expose a function:**
> ```python
> def ask_assistant(question: str) -> dict:
>     # Returns: {"answer": str, "source_products": list[str]}
> ```
> where `source_products` is a list of product_ids from the retrieved documents' metadata.
>
> Also load the vectorstore from disk if it already exists (don't re-embed every time).

---

## 🎯 STEP 4 — Recommendation Engine

### File: `recommendations/recommender.py`

**Copilot Prompt:**
> Write a Python module `recommender.py` for a content-based recommendation engine using product embeddings.
>
> **Setup:**
> - Load the FAISS index from `search/faiss_index.bin`
> - Load `search/product_id_map.json`
> - Load `data/products.csv`
> - Load `data/user_interactions.csv`
>
> **Function 1: `get_similar_products(product_id: str, top_k: int = 5) -> list[dict]`**
> - Find the embedding of the given product_id from the FAISS index
> - Search FAISS for top_k+1 similar products (exclude itself)
> - Return list of similar products with similarity scores
>
> **Function 2: `get_user_recommendations(user_id: str, top_k: int = 5) -> list[dict]`**
> - Load user's interaction history from `user_interactions.csv`
> - Filter for this user_id, prioritize `purchase` > `click` > `view` interactions
> - If user has interactions: get the top 3 most interacted products, average their embeddings, search FAISS for nearest neighbors, exclude already-seen products
> - If user has NO interactions (cold start): return top 5 products from the most popular category (by purchase count across all users)
> - Return list of recommended products with scores
>
> **Function 3: `get_cold_start_recommendations(category: str = None) -> list[dict]`**
> - If category given: return 5 random products from that category
> - If no category: return 5 most purchased products across the entire catalog
>
> Each returned product dict should include: `product_id`, `title`, `category`, `price`, `brand`, `similarity_score`.

---

## 🌐 STEP 5 — FastAPI Application

### File: `api/main.py`

**Copilot Prompt:**
> Write a FastAPI application in `api/main.py` that exposes the following endpoints. Import functions from the modules already built (search, assistant, recommendations).
>
> **Setup:**
> ```python
> from fastapi import FastAPI, HTTPException, Query
> from pydantic import BaseModel
> import time
> import logging
> ```
>
> Enable basic logging to print each request with timestamp and latency.
>
> ---
>
> **Endpoint 1: GET `/health`**
> - Returns: `{"status": "ok", "message": "AI Commerce Platform is running"}`
>
> ---
>
> **Endpoint 2: POST `/search`**
> - Request body:
> ```json
> { "query": "affordable running shoes under 4000", "top_k": 10 }
> ```
> - Calls `hybrid_search(query, top_k)` from `search/hybrid_search.py`
> - Returns:
> ```json
> {
>   "query": "...",
>   "results": [...],
>   "total_results": 10,
>   "latency_ms": 143.5
> }
> ```
> - On error: raise HTTP 500 with error message
>
> ---
>
> **Endpoint 3: POST `/assistant/ask`**
> - Request body:
> ```json
> { "question": "Is the Dell laptop good for AI model training?" }
> ```
> - Calls `ask_assistant(question)` from `assistant/rag_assistant.py`
> - Returns:
> ```json
> {
>   "question": "...",
>   "answer": "...",
>   "source_product_ids": ["P0042", "P0108"],
>   "latency_ms": 980.2
> }
> ```
>
> ---
>
> **Endpoint 4: GET `/recommendations/similar/{product_id}`**
> - Path param: `product_id`
> - Query param: `top_k` (default 5)
> - Calls `get_similar_products(product_id, top_k)`
> - Returns list of similar products with scores
> - If product not found: raise HTTP 404
>
> ---
>
> **Endpoint 5: GET `/recommendations/user/{user_id}`**
> - Path param: `user_id`
> - Query param: `top_k` (default 5)
> - Calls `get_user_recommendations(user_id, top_k)`
> - Returns personalized recommendations
>
> ---
>
> **Endpoint 6: GET `/recommendations/cold-start`**
> - Query param: `category` (optional string)
> - Calls `get_cold_start_recommendations(category)`
> - Returns popular/category-based recommendations
>
> ---
>
> Run with: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
> Add `if __name__ == "__main__": uvicorn.run(...)` at bottom.

---

## 🐳 STEP 6 — Docker Setup

### `Dockerfile`

**Copilot Prompt:**
> Write a Dockerfile for a Python 3.11 FastAPI application:
> - Base image: `python:3.11-slim`
> - Set working directory to `/app`
> - Copy `requirements.txt` and install dependencies
> - Copy the entire project
> - Expose port 8000
> - CMD: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

### `docker-compose.yml`

**Copilot Prompt:**
> Write a `docker-compose.yml` for a single service called `ai-commerce-api`:
> - Build from current directory (`.`)
> - Map port 8000:8000
> - Mount a `.env` file as environment
> - Add a volume for `./data:/app/data` so data files persist

---

## 🧪 STEP 7 — Basic Tests

### File: `tests/test_all.py`

**Copilot Prompt:**
> Write a Python test file using only `assert` statements (no pytest decorators needed) that tests:
> 1. `test_hybrid_search()` — calls `hybrid_search("running shoes", top_k=5)`, asserts result is a list of 5 items, each with keys `title`, `final_score`, `product_id`
> 2. `test_rag_assistant()` — calls `ask_assistant("What laptops do you have?")`, asserts `answer` is a non-empty string
> 3. `test_recommendations()` — calls `get_similar_products("P0001", top_k=3)`, asserts 3 results returned
> 4. `test_cold_start()` — calls `get_cold_start_recommendations()`, asserts 5 results returned
>
> Wrap each in try/except and print PASS or FAIL with the test name.
> Run all tests in `if __name__ == "__main__":` block.

---

## 🚀 Build Order (Run These In Sequence)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python data/generate_products.py

# 3. Build search indexes (run ONCE — takes ~2 minutes due to OpenAI API calls)
python search/embed_products.py
python search/bm25_index.py

# 4. Build RAG vectorstore (run ONCE)
python assistant/rag_assistant.py

# 5. Start the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 6. Run tests
python tests/test_all.py

# 7. Docker (optional)
docker-compose up --build
```

---

## 💬 Live Demo Test Queries (Practice These)

### Search Queries to Test
```
"affordable running shoes under 4000"
"wireless headphones for gym"
"laptop for machine learning under 60000"
"summer dress for women"
```

### RAG Assistant Queries to Test
```
"Compare the best laptops for AI model training"
"What are the best products for home workout?"
"Is there any smartwatch with long battery life?"
```

### Recommendation Queries to Test
```
GET /recommendations/user/U001
GET /recommendations/user/U002
GET /recommendations/similar/P0001
GET /recommendations/cold-start?category=Electronics
```

---

## 💰 Cost Model — Memorize This for Presentation

```
Scenario: 100K daily users, 8 prompts/user, 1K tokens/request

Daily tokens  = 100,000 × 8 × 1,000 = 800,000,000 tokens (800M)
Model         = GPT-4o-mini = $0.15 per 1M input tokens

Daily cost    = 800M / 1M × $0.15 = $120/day
Monthly cost  = $120 × 30 = ~$3,600/month

With 40% query caching → ~$2,160/month
```

**Key talking points:**
- Cache repeated/similar queries using Redis (saves 30–40%)
- Use `text-embedding-3-small` (not large) — 5x cheaper, 80% as good
- Switch to **Llama 3 on self-hosted GPU** when monthly cost exceeds ~$5K
- RAG beats fine-tuning here because product catalog changes daily

---

## 🏗️ Architecture Diagram Description (For Presentation)

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT / USER                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Request
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI API Gateway                        │
│              (Rate Limiting + Logging)                       │
└────┬──────────────────┬──────────────────┬──────────────────┘
     │                  │                  │
┌────▼────┐      ┌──────▼──────┐    ┌──────▼──────┐
│  Hybrid │      │     RAG     │    │  Recommend  │
│  Search │      │  Assistant  │    │   Engine    │
│  Engine │      │ (LangChain) │    │  (FAISS)    │
└────┬────┘      └──────┬──────┘    └──────┬──────┘
     │                  │                  │
┌────▼──────────────────▼──────────────────▼──────┐
│            FAISS Vector Index + BM25             │
│              (Product Embeddings)                │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│              OpenAI API                           │
│   (text-embedding-3-small + gpt-4o-mini)         │
└───────────────────────────────────────────────────┘
```

---

## ❓ Key Questions & Answers for Interview

| Question | Short Answer |
|---|---|
| Why RAG not fine-tuning? | Product catalog changes daily. RAG updates by re-indexing. Fine-tuning needs full retraining. |
| How reduce hallucinations? | Strict system prompt: only answer from context. If context empty → "not in catalog" |
| Why hybrid over pure semantic? | Keyword search catches exact matches (brand names, model numbers). Semantic catches meaning. |
| How scale to 1M users? | Shard FAISS by category, cache in Redis, async FastAPI workers, move to Pinecone |
| When switch to open-source LLM? | When monthly cost > $5K or data privacy requirements apply (e.g., healthcare) |
| Cold start problem? | New user → show popular products by category. No history needed. |

---

*Generated for Sorim.AI Technical Assessment — AI Engineer Role*

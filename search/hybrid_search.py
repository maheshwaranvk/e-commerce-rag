"""
Hybrid search engine combining FAISS semantic search with BM25 keyword search.

Exposes:
  - hybrid_search(query, top_k) -> list[dict]  (async)
"""

import asyncio
import hashlib
import json
import logging
import os
import time

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from search.bm25_index import get_bm25_scores, load_bm25, tokenize

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
PRODUCT_ID_MAP_PATH = os.path.join(BASE_DIR, "product_id_map.json")
INDEX_METADATA_PATH = os.path.join(BASE_DIR, "index_metadata.json")
BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")

MAX_QUERY_LENGTH = 500
SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# --- Module-level loading with fail-fast checks ---

_required_files = {
    "FAISS index": FAISS_INDEX_PATH,
    "Product ID map": PRODUCT_ID_MAP_PATH,
    "BM25 index": BM25_INDEX_PATH,
    "Products CSV": PRODUCTS_CSV,
}
for name, path in _required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found at {path}. "
            "Run 'python search/embed_products.py' and 'python search/bm25_index.py' first."
        )

# Check index/CSV consistency
if os.path.exists(INDEX_METADATA_PATH):
    with open(INDEX_METADATA_PATH, "r", encoding="utf-8") as _f:
        _metadata = json.load(_f)
    _current_hash = hashlib.sha256(open(PRODUCTS_CSV, "rb").read()).hexdigest()
    if _metadata.get("source_csv_hash") != _current_hash:
        logger.warning(
            "products.csv has changed since the FAISS index was built. "
            "Consider re-running 'python search/embed_products.py'."
        )

# Load resources
_faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(PRODUCT_ID_MAP_PATH, "r", encoding="utf-8") as _f:
    _product_id_map: list[str] = json.load(_f)
_products_df = pd.read_csv(PRODUCTS_CSV)
_bm25 = load_bm25()
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]. Handles all-zero case."""
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def _faiss_search(query_embedding: np.ndarray, top_n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS index synchronously. Returns (scores, indices)."""
    query_vec = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = _faiss_index.search(query_vec, top_n)
    return scores[0], indices[0]


async def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Combine semantic (FAISS) and keyword (BM25) search.

    Args:
        query: Search query string (1-500 chars).
        top_k: Number of results to return (1-50).

    Returns:
        List of product dicts with scores, sorted by final_score descending.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.")

    query = query.strip()
    num_products = len(_product_id_map)

    # Run semantic embedding + BM25 scoring in parallel
    async def _semantic_task():
        query_embedding = await _embeddings.aembed_query(query)
        scores, indices = await asyncio.to_thread(_faiss_search, np.array(query_embedding), min(50, num_products))
        return scores, indices

    async def _bm25_task():
        scores = await asyncio.to_thread(get_bm25_scores, _bm25, query)
        return np.array(scores)

    (sem_scores, sem_indices), bm25_all_scores = await asyncio.gather(
        _semantic_task(), _bm25_task()
    )

    # Build semantic score map (product index -> raw score)
    semantic_score_map = {}
    for score, idx in zip(sem_scores, sem_indices):
        if idx >= 0:  # FAISS returns -1 for unfilled slots
            semantic_score_map[int(idx)] = float(score)

    # Normalize BM25 scores
    bm25_normalized = _normalize_scores(bm25_all_scores)

    # Normalize semantic scores across the top-50 retrieved
    if semantic_score_map:
        sem_values = np.array(list(semantic_score_map.values()))
        sem_min, sem_max = sem_values.min(), sem_values.max()
        if sem_max - sem_min < 1e-9:
            sem_normalized_map = {idx: 0.0 for idx in semantic_score_map}
        else:
            sem_normalized_map = {
                idx: float((score - sem_min) / (sem_max - sem_min))
                for idx, score in semantic_score_map.items()
            }
    else:
        sem_normalized_map = {}

    # Combine scores for all products
    combined = []
    for i in range(num_products):
        sem_score = sem_normalized_map.get(i, 0.0)
        bm25_score = float(bm25_normalized[i])
        final_score = SEMANTIC_WEIGHT * sem_score + BM25_WEIGHT * bm25_score
        combined.append((i, sem_score, bm25_score, final_score))

    # Sort by final score descending, take top_k
    combined.sort(key=lambda x: x[3], reverse=True)
    top_results = combined[:top_k]

    # Build result dicts
    results = []
    for idx, sem_score, bm25_score, final_score in top_results:
        product_id = _product_id_map[idx]
        row = _products_df[_products_df["product_id"] == product_id].iloc[0]
        results.append({
            "product_id": product_id,
            "title": row["title"],
            "category": row["category"],
            "price": float(row["price"]),
            "brand": row["brand"],
            "description": row["description"],
            "semantic_score": round(sem_score, 4),
            "bm25_score": round(bm25_score, 4),
            "final_score": round(final_score, 4),
        })

    return results


async def hybrid_search_with_latency(query: str, top_k: int = 10) -> tuple[list[dict], float]:
    """Run hybrid search and return (results, latency_ms)."""
    start = time.perf_counter()
    results = await hybrid_search(query, top_k)
    latency_ms = (time.perf_counter() - start) * 1000
    return results, latency_ms

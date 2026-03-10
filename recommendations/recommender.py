"""
Content-based recommendation engine using FAISS product embeddings.

Exposes (all async):
  - get_similar_products(product_id, top_k) -> list[dict]
  - get_user_recommendations(user_id, top_k) -> list[dict]
  - get_cold_start_recommendations(category) -> list[dict]
"""

import asyncio
import json
import os
from collections import Counter

import faiss
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SEARCH_DIR = os.path.join(PROJECT_DIR, "search")

FAISS_INDEX_PATH = os.path.join(SEARCH_DIR, "faiss_index.bin")
PRODUCT_ID_MAP_PATH = os.path.join(SEARCH_DIR, "product_id_map.json")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
USER_INTERACTIONS_CSV = os.path.join(DATA_DIR, "user_interactions.csv")

# Interaction weights for prioritization
INTERACTION_WEIGHTS = {"purchase": 3, "click": 2, "view": 1}

# Lazy-loaded module state
_faiss_index = None
_product_id_map: list[str] = []
_products_df: pd.DataFrame = None
_interactions_df: pd.DataFrame = None
_id_to_index: dict[str, int] = {}


def _ensure_loaded():
    """Lazy-load all data files on first use."""
    global _faiss_index, _product_id_map, _products_df, _interactions_df, _id_to_index

    if _faiss_index is not None:
        return

    for name, path in [
        ("FAISS index", FAISS_INDEX_PATH),
        ("Product ID map", PRODUCT_ID_MAP_PATH),
        ("Products CSV", PRODUCTS_CSV),
        ("User interactions CSV", USER_INTERACTIONS_CSV),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}.")

    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(PRODUCT_ID_MAP_PATH, "r", encoding="utf-8") as f:
        _product_id_map = json.load(f)
    _products_df = pd.read_csv(PRODUCTS_CSV)
    _interactions_df = pd.read_csv(USER_INTERACTIONS_CSV)
    _id_to_index = {pid: i for i, pid in enumerate(_product_id_map)}


def _product_dict(product_id: str, similarity_score: float = 0.0) -> dict:
    """Build a product result dict."""
    _ensure_loaded()
    row = _products_df[_products_df["product_id"] == product_id].iloc[0]
    return {
        "product_id": product_id,
        "title": row["title"],
        "category": row["category"],
        "price": float(row["price"]),
        "brand": row["brand"],
        "similarity_score": round(similarity_score, 4),
    }


def _faiss_search_by_index(idx: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS by vector index. Returns (scores, indices)."""
    _ensure_loaded()
    vec = _faiss_index.reconstruct(idx).reshape(1, -1)
    faiss.normalize_L2(vec)
    scores, indices = _faiss_index.search(vec, top_k)
    return scores[0], indices[0]


def _faiss_search_by_vector(vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS by raw vector. Returns (scores, indices)."""
    _ensure_loaded()
    query = vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)
    scores, indices = _faiss_index.search(query, top_k)
    return scores[0], indices[0]


async def get_similar_products(product_id: str, top_k: int = 5) -> list[dict]:
    """
    Find products similar to a given product using FAISS embeddings.

    Args:
        product_id: The product to find similarities for.
        top_k: Number of similar products to return.

    Returns:
        List of similar product dicts (excludes the input product).

    Raises:
        ValueError: If product_id is not found.
    """
    _ensure_loaded()

    if product_id not in _id_to_index:
        raise ValueError(f"Product '{product_id}' not found in index.")

    idx = _id_to_index[product_id]
    scores, indices = await asyncio.to_thread(
        _faiss_search_by_index, idx, top_k + 1
    )

    results = []
    for score, result_idx in zip(scores, indices):
        if result_idx < 0:
            continue
        pid = _product_id_map[int(result_idx)]
        if pid == product_id:
            continue
        results.append(_product_dict(pid, float(score)))
        if len(results) >= top_k:
            break

    return results


async def get_user_recommendations(user_id: str, top_k: int = 5) -> list[dict]:
    """
    Get personalized recommendations for a user based on interaction history.

    Uses per-product search + merge strategy:
      1. Find top 3 most-interacted products (weighted by interaction type)
      2. For each, search FAISS for neighbors
      3. Merge results: sum similarity scores for products appearing in multiple searches
      4. Exclude already-seen products

    Falls back to cold-start if user has no interactions.

    Args:
        user_id: The user to recommend for.
        top_k: Number of recommendations to return.

    Returns:
        List of recommended product dicts with scores.
    """
    _ensure_loaded()

    user_interactions = _interactions_df[_interactions_df["user_id"] == user_id]

    if user_interactions.empty:
        return await get_cold_start_recommendations()

    # Compute weighted interaction scores per product
    product_scores: Counter = Counter()
    for _, row in user_interactions.iterrows():
        weight = INTERACTION_WEIGHTS.get(row["interaction_type"], 1)
        product_scores[row["product_id"]] += weight

    # Get top 3 most-interacted products
    top_interacted = [pid for pid, _ in product_scores.most_common(3)]
    seen_products = set(user_interactions["product_id"].unique())

    # Per-product FAISS search and merge results
    merged_scores: Counter = Counter()
    search_k = top_k + len(seen_products) + 5  # fetch extra to account for exclusions

    search_tasks = []
    for pid in top_interacted:
        if pid in _id_to_index:
            idx = _id_to_index[pid]
            search_tasks.append(asyncio.to_thread(_faiss_search_by_index, idx, search_k))

    if not search_tasks:
        return await get_cold_start_recommendations()

    search_results = await asyncio.gather(*search_tasks)

    for scores, indices in search_results:
        for score, result_idx in zip(scores, indices):
            if result_idx < 0:
                continue
            pid = _product_id_map[int(result_idx)]
            if pid not in seen_products:
                merged_scores[pid] += float(score)

    # Sort by merged score, return top_k
    top_products = merged_scores.most_common(top_k)
    return [_product_dict(pid, score) for pid, score in top_products]


async def get_cold_start_recommendations(category: str = None) -> list[dict]:
    """
    Get recommendations for new users or when no history is available.

    Args:
        category: Optional category to filter by. If None, returns most purchased products.

    Returns:
        List of 5 recommended product dicts.
    """
    _ensure_loaded()

    if category:
        # Return 5 random products from the specified category
        category_products = _products_df[
            _products_df["category"].str.lower() == category.lower()
        ]
        if category_products.empty:
            return []
        sampled = category_products.sample(n=min(5, len(category_products)), random_state=42)
        return [
            _product_dict(row["product_id"])
            for _, row in sampled.iterrows()
        ]

    # No category: return 5 most purchased products
    purchases = _interactions_df[_interactions_df["interaction_type"] == "purchase"]
    if purchases.empty:
        # Fallback: return first 5 products
        return [
            _product_dict(row["product_id"])
            for _, row in _products_df.head(5).iterrows()
        ]

    top_purchased = (
        purchases.groupby("product_id")
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    return [_product_dict(pid) for pid in top_purchased]

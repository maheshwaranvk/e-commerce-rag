"""
FastAPI application lifecycle and shared dependencies.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cache.redis_cache import close_redis, connect_redis

logger = logging.getLogger(__name__)

SEARCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search")
ASSISTANT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assistant")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

REQUIRED_FILES = {
    "FAISS index": os.path.join(SEARCH_DIR, "faiss_index.bin"),
    "Product ID map": os.path.join(SEARCH_DIR, "product_id_map.json"),
    "BM25 index": os.path.join(SEARCH_DIR, "bm25_index.pkl"),
    "Products CSV": os.path.join(DATA_DIR, "products.csv"),
    "User interactions CSV": os.path.join(DATA_DIR, "user_interactions.csv"),
    "RAG vectorstore": os.path.join(ASSISTANT_DIR, "vectorstore"),
}


def verify_required_files():
    """Check all required data/index files exist. Raises if any are missing."""
    missing = []
    for name, path in REQUIRED_FILES.items():
        if not os.path.exists(path):
            missing.append(f"  - {name}: {path}")
    if missing:
        msg = "Missing required files:\n" + "\n".join(missing)
        msg += "\n\nRun the build steps first (see README)."
        raise FileNotFoundError(msg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # --- Startup ---
    logger.info("Starting AI Commerce Platform...")

    # Verify all required files exist
    verify_required_files()
    logger.info("All required data/index files verified.")

    # Connect Redis (non-fatal if unavailable)
    redis_ok = await connect_redis()
    if redis_ok:
        logger.info("Redis cache connected.")
    else:
        logger.warning("Redis unavailable — running without cache.")

    logger.info("AI Commerce Platform ready.")

    yield

    # --- Shutdown ---
    logger.info("Shutting down AI Commerce Platform...")
    await close_redis()
    logger.info("Shutdown complete.")

"""
Async Redis caching layer with graceful degradation.

If Redis is unavailable, all operations silently return None / do nothing.
"""

import hashlib
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Cache TTLs in seconds
TTL_SEARCH = 3600       # 1 hour
TTL_ASSISTANT = 1800    # 30 minutes
TTL_RECOMMENDATIONS = 7200  # 2 hours

_redis_client = None


async def connect_redis():
    """Initialize the async Redis connection. Safe to call multiple times."""
    global _redis_client
    try:
        from redis.asyncio import Redis
        _redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
        await _redis_client.ping()
        logger.info("Redis connected at %s", REDIS_URL)
        return True
    except Exception as e:
        logger.warning("Redis connection failed (%s). Caching disabled.", e)
        _redis_client = None
        return False


async def close_redis():
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed.")


async def is_connected() -> bool:
    """Check if Redis is currently reachable."""
    if _redis_client is None:
        return False
    try:
        await _redis_client.ping()
        return True
    except Exception:
        return False


def make_cache_key(prefix: str, **params) -> str:
    """Generate a deterministic cache key from prefix and params."""
    # Sort params for determinism, then hash
    raw = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"aicommerce:{prefix}:{digest}"


async def get_cached(key: str) -> dict | None:
    """Get a cached value. Returns None on miss or Redis failure."""
    if _redis_client is None:
        return None
    try:
        data = await _redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.warning("Redis GET failed for key %s: %s", key, e)
        return None


async def set_cached(key: str, value: dict, ttl: int = 3600) -> None:
    """Set a cached value with TTL. Silently fails on Redis error."""
    if _redis_client is None:
        return
    try:
        await _redis_client.set(key, json.dumps(value, default=str), ex=ttl)
    except Exception as e:
        logger.warning("Redis SET failed for key %s: %s", key, e)

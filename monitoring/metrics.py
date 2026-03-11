"""Local metrics (no external service required).

Exports Prometheus metrics on /metrics.
Tracks:
  - Request count and latency by endpoint
  - Cache hit/miss rate
  - Error rate
  - LLM token usage

If prometheus-client is not installed, metrics become no-ops.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram

    _PROM_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning("prometheus-client not available, metrics disabled: %s", e)
    _PROM_AVAILABLE = False


if _PROM_AVAILABLE:
    REQUEST_COUNT = Counter(
        "api_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )

    REQUEST_LATENCY_MS = Histogram(
        "api_request_latency_ms",
        "Request latency in milliseconds",
        ["method", "path"],
        buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
    )

    CACHE_HITS = Counter(
        "api_cache_hits_total",
        "Total cache hits",
        ["endpoint"],
    )

    CACHE_MISSES = Counter(
        "api_cache_misses_total",
        "Total cache misses",
        ["endpoint"],
    )

    ERROR_COUNT = Counter(
        "api_errors_total",
        "Total error responses (HTTP status >= 400)",
        ["method", "path", "status"],
    )

    LLM_TOKENS = Counter(
        "llm_tokens_total",
        "Total LLM tokens used",
        ["model", "type"],  # type in {prompt, completion, total}
    )
else:  # pragma: no cover
    REQUEST_COUNT = None
    REQUEST_LATENCY_MS = None
    CACHE_HITS = None
    CACHE_MISSES = None
    ERROR_COUNT = None
    LLM_TOKENS = None


def observe_request(method: str, path: str, status: int, latency_ms: float) -> None:
    if not _PROM_AVAILABLE:
        return
    REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY_MS.labels(method=method, path=path).observe(latency_ms)
    if status >= 400:
        ERROR_COUNT.labels(method=method, path=path, status=str(status)).inc()


def observe_cache(endpoint: str, hit: bool) -> None:
    if not _PROM_AVAILABLE:
        return
    if hit:
        CACHE_HITS.labels(endpoint=endpoint).inc()
    else:
        CACHE_MISSES.labels(endpoint=endpoint).inc()


def observe_llm_tokens(model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    if not _PROM_AVAILABLE:
        return
    LLM_TOKENS.labels(model=model, type="prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(model=model, type="completion").inc(completion_tokens)
    LLM_TOKENS.labels(model=model, type="total").inc(total_tokens)

"""Prometheus metrics for the LLM Gateway."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "llmgw_requests_total",
    "Total number of requests processed",
    ["method", "path", "status_code", "provider"],
)

PROVIDER_ERRORS = Counter(
    "llmgw_provider_errors_total",
    "Total provider-level errors",
    ["provider", "error_type"],
)

CACHE_HITS = Counter(
    "llmgw_cache_hits_total",
    "Total cache hits",
)

CACHE_MISSES = Counter(
    "llmgw_cache_misses_total",
    "Total cache misses",
)

TOKEN_USAGE = Counter(
    "llmgw_token_usage_total",
    "Total tokens consumed",
    ["provider", "model", "direction"],  # direction: input | output
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

REQUEST_LATENCY = Histogram(
    "llmgw_request_duration_seconds",
    "Request latency in seconds",
    ["method", "path", "provider"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

ACTIVE_REQUESTS = Gauge(
    "llmgw_active_requests",
    "Number of requests currently being processed",
    ["provider"],
)

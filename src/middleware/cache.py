"""Response caching middleware using Redis or in-memory backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

# Paths eligible for caching (non-streaming POST endpoints)
_CACHEABLE_PATHS = {"/v1/chat/completions", "/v1/embeddings"}


def cache_key(body: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest for the request body.

    The key is stable across Python process restarts because we serialize
    with ``sort_keys=True``.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class CacheMiddleware(BaseHTTPMiddleware):
    """Cache non-streaming responses in Redis (or compatible store).

    Parameters
    ----------
    enabled:
        Master switch.  When ``False`` the middleware is a no-op.
    redis:
        An object exposing async ``get(key)`` / ``set(key, value, ex=ttl)``
        (e.g. ``redis.asyncio.Redis``).
    ttl_seconds:
        Time-to-live for cached entries.
    """

    def __init__(
        self,
        app,
        *,
        enabled: bool = False,
        redis: Any = None,
        ttl_seconds: int = 3600,
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled or self.redis is None:
            return await call_next(request)

        if request.url.path not in _CACHEABLE_PATHS:
            return await call_next(request)

        # Read body -- need to buffer it for hashing
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await call_next(request)

        # Skip streaming requests
        if body_json.get("stream", False):
            return await call_next(request)

        key = f"llmgw:{cache_key(body_json)}"

        # Cache lookup
        cached = await self.redis.get(key)
        if cached is not None:
            logger.debug("cache.hit", path=request.url.path)
            return JSONResponse(
                content=json.loads(cached),
                headers={"X-Cache": "HIT"},
            )

        # Cache miss -- forward request
        response = await call_next(request)

        # Only cache successful JSON responses
        if response.status_code == 200:
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk if isinstance(chunk, bytes) else chunk.encode()
            await self.redis.set(key, resp_body.decode(), ex=self.ttl_seconds)
            logger.debug("cache.stored", path=request.url.path)
            return Response(
                content=resp_body,
                status_code=200,
                media_type="application/json",
                headers={"X-Cache": "MISS"},
            )

        return response

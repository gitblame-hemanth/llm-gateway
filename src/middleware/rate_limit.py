"""Sliding-window rate limiter with Redis sorted sets and in-memory fallback."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

_PREFIX = "llmgw:rl"


# ---------------------------------------------------------------------------
# In-memory sliding window (fallback)
# ---------------------------------------------------------------------------


@dataclass
class _MemWindow:
    timestamps: list[float] = field(default_factory=list)

    def add(self, ts: float) -> None:
        self.timestamps.append(ts)

    def count_since(self, since: float) -> int:
        self.timestamps = [t for t in self.timestamps if t > since]
        return len(self.timestamps)


class _InMemoryBackend:
    def __init__(self) -> None:
        self._rpm: dict[str, _MemWindow] = defaultdict(_MemWindow)
        self._tpm: dict[str, _MemWindow] = defaultdict(_MemWindow)

    def check_rpm(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.time()
        window = self._rpm[key]
        count = window.count_since(now - 60)
        if count >= limit:
            retry_after = int(60 - (now - window.timestamps[0])) + 1 if window.timestamps else 60
            return False, max(retry_after, 1)
        window.add(now)
        return True, 0

    def check_tpm(self, key: str, tokens: int, limit: int) -> tuple[bool, int]:
        now = time.time()
        window = self._tpm[key]
        count = window.count_since(now - 60)
        if count + tokens > limit:
            retry_after = int(60 - (now - window.timestamps[0])) + 1 if window.timestamps else 60
            return False, max(retry_after, 1)
        for _ in range(tokens):
            window.add(now)
        return True, 0


class RateLimiter:
    """Per-API-key sliding-window rate limiter.

    Uses Redis sorted sets when available; falls back to in-memory.
    """

    def __init__(
        self,
        rpm: int = 60,
        tpm: int = 100_000,
        redis_url: str | None = None,
    ) -> None:
        self._rpm_limit = rpm
        self._tpm_limit = tpm
        self._redis: Any | None = None
        self._mem = _InMemoryBackend()

        if redis_url:
            try:
                import redis as _redis

                self._redis = _redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("rate_limiter_redis_connected", url=redis_url)
            except Exception:
                logger.warning("rate_limiter_redis_unavailable_falling_back_to_memory")
                self._redis = None

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def check(self, api_key: str, estimated_tokens: int = 1) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``.

        ``retry_after_seconds`` is 0 when allowed.
        """
        if self._redis:
            try:
                return self._check_redis(api_key, estimated_tokens)
            except Exception:
                logger.warning("rate_limiter_redis_check_failed_falling_back")

        # In-memory path
        allowed, retry = self._mem.check_rpm(api_key, self._rpm_limit)
        if not allowed:
            return False, retry

        allowed, retry = self._mem.check_tpm(api_key, estimated_tokens, self._tpm_limit)
        return allowed, retry

    def _check_redis(self, api_key: str, estimated_tokens: int) -> tuple[bool, int]:
        now = time.time()
        window_start = now - 60

        pipe = self._redis.pipeline()

        rpm_key = f"{_PREFIX}:rpm:{api_key}"
        pipe.zremrangebyscore(rpm_key, "-inf", window_start)
        pipe.zcard(rpm_key)
        pipe.zadd(rpm_key, {str(now): now})
        pipe.expire(rpm_key, 120)

        tpm_key = f"{_PREFIX}:tpm:{api_key}"
        pipe.zremrangebyscore(tpm_key, "-inf", window_start)
        pipe.zcard(tpm_key)

        results = pipe.execute()
        rpm_count: int = results[1]
        tpm_count: int = results[5]

        if rpm_count >= self._rpm_limit:
            # Remove the optimistic add
            self._redis.zrem(rpm_key, str(now))
            oldest = self._redis.zrange(rpm_key, 0, 0, withscores=True)
            retry = int(60 - (now - oldest[0][1])) + 1 if oldest else 60
            logger.warning("rate_limit_rpm_exceeded", api_key=api_key[:8])
            return False, max(retry, 1)

        if tpm_count + estimated_tokens > self._tpm_limit:
            oldest = self._redis.zrange(tpm_key, 0, 0, withscores=True)
            retry = int(60 - (now - oldest[0][1])) + 1 if oldest else 60
            logger.warning("rate_limit_tpm_exceeded", api_key=api_key[:8])
            return False, max(retry, 1)

        # Record tokens
        if estimated_tokens > 0:
            pipe2 = self._redis.pipeline()
            for i in range(estimated_tokens):
                pipe2.zadd(tpm_key, {f"{now}:{i}": now})
            pipe2.expire(tpm_key, 120)
            pipe2.execute()

        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware wrapping :class:`RateLimiter`."""

    def __init__(self, app: object, limiter: RateLimiter) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        api_key = request.headers.get("X-API-Key", request.client.host if request.client else "anon")
        allowed, retry_after = self._limiter.check(api_key)
        if not allowed:
            return JSONResponse(
                {"error": "Rate limit exceeded", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)

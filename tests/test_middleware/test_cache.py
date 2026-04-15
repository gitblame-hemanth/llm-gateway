"""Tests for src.middleware.cache — CacheMiddleware and cache_key."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from src.middleware.cache import CacheMiddleware, cache_key


def _build_app(*, enabled: bool = True, redis: MagicMock | None = None) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        return JSONResponse(content={"id": "test", "model": body.get("model", "m")})

    app.add_middleware(CacheMiddleware, enabled=enabled, redis=redis, ttl_seconds=60)
    return app


class TestCacheKey:
    """cache_key determinism."""

    def test_deterministic_same_input(self) -> None:
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        assert cache_key(body) == cache_key(body)

    def test_deterministic_key_order_irrelevant(self) -> None:
        a = {"model": "gpt-4", "messages": []}
        b = {"messages": [], "model": "gpt-4"}
        assert cache_key(a) == cache_key(b)

    def test_different_input_different_key(self) -> None:
        a = {"model": "gpt-4", "messages": []}
        b = {"model": "gpt-3.5-turbo", "messages": []}
        assert cache_key(a) != cache_key(b)


class TestCacheMiddleware:
    """CacheMiddleware hit / miss / skip behavior."""

    def test_cache_miss_forwards_request(self) -> None:
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        app = _build_app(redis=redis)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": False},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-cache") == "MISS"

    def test_cache_hit_returns_cached(self) -> None:
        cached_body = json.dumps({"id": "cached", "model": "test"})
        redis = MagicMock()
        redis.get = AsyncMock(return_value=cached_body)

        app = _build_app(redis=redis)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": False},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "cached"
        assert resp.headers.get("x-cache") == "HIT"

    def test_skip_streaming_requests(self) -> None:
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        app = _build_app(redis=redis)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [], "stream": True},
        )
        assert resp.status_code == 200
        # Should not have X-Cache header since streaming bypasses cache
        assert resp.headers.get("x-cache") is None

    def test_disabled_cache_passes_through(self) -> None:
        app = _build_app(enabled=False)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-cache") is None

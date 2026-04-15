"""Tests for src.middleware.auth.AuthMiddleware."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.middleware.auth import AuthMiddleware


def _build_app(*, enabled: bool = True, api_keys: list[str] | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"data": []}

    app.add_middleware(AuthMiddleware, enabled=enabled, api_keys=api_keys)
    return app


class TestAuthMiddleware:
    """Authentication middleware behavior."""

    def test_valid_key_passes(self) -> None:
        app = _build_app(api_keys=["test-key-123"])
        client = TestClient(app)
        resp = client.get("/v1/models", headers={"Authorization": "Bearer test-key-123"})
        assert resp.status_code == 200

    def test_invalid_key_returns_401(self) -> None:
        app = _build_app(api_keys=["correct-key"])
        client = TestClient(app)
        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["detail"]

    def test_missing_key_returns_401(self) -> None:
        app = _build_app(api_keys=["some-key"])
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["detail"]

    def test_auth_disabled_passes_without_key(self) -> None:
        app = _build_app(enabled=False)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_public_paths_bypass_auth(self) -> None:
        app = _build_app(api_keys=["key"])
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

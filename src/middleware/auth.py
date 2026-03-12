"""API-key authentication middleware."""

from __future__ import annotations

import os
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)

_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate ``Authorization: Bearer <key>`` against a set of allowed keys.

    When *enabled* is ``False`` all requests pass through.
    """

    def __init__(self, app, *, enabled: bool = True, api_keys: list[str] | None = None) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.api_keys: set[str] = set(api_keys or [])
        # Also accept keys from the environment
        env_keys = os.environ.get("GATEWAY_API_KEYS", "")
        if env_keys:
            self.api_keys.update(k.strip() for k in env_keys.split(",") if k.strip())

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key"},
            )

        token = auth_header[7:]
        if token not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)

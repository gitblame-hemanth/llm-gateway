"""Structured JSON request/response logging middleware."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request and response as structured JSON via structlog."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        log = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )
        log.info("request_started")

        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            log.exception("request_failed", duration_ms=round(elapsed * 1000, 2))
            raise

        elapsed = time.perf_counter() - start
        log.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        response.headers["X-Request-ID"] = request_id
        return response

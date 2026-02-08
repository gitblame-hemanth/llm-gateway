"""FastAPI dependency injection functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from src.core.tracker import UsageTracker
    from src.middleware.cache import ResponseCache
    from src.middleware.rate_limit import RateLimiter
    from src.providers.registry import ProviderRegistry
    from src.routing.router import Router


def get_router(request: Request) -> Router:
    """Retrieve the request router from app state."""
    return request.app.state.router


def get_registry(request: Request) -> ProviderRegistry:
    """Retrieve the provider registry from app state."""
    return request.app.state.registry


def get_cache(request: Request) -> ResponseCache | None:
    """Retrieve the response cache from app state (may be None if disabled)."""
    return getattr(request.app.state, "cache", None)


def get_rate_limiter(request: Request) -> RateLimiter | None:
    """Retrieve the rate limiter from app state (may be None if disabled)."""
    return getattr(request.app.state, "rate_limiter", None)


def get_tracker(request: Request) -> UsageTracker | None:
    """Retrieve the usage tracker from app state (may be None)."""
    return getattr(request.app.state, "tracker", None)

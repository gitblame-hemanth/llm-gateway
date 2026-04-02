"""FastAPI application entry-point for the LLM Gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app as prometheus_asgi_app

from src import __version__
from src.core.config import get_config, reload_config
from src.middleware.auth import APIKeyAuth
from src.middleware.cache import ResponseCache
from src.middleware.logging_mw import RequestLoggingMiddleware
from src.middleware.rate_limit import RateLimiter
from src.providers.registry import ProviderRegistry
from src.routing.router import Router

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider factory — maps config names to concrete classes
# ---------------------------------------------------------------------------


def _build_provider_factory() -> dict[str, type]:
    """Import available provider classes; skip missing optional deps."""
    factory: dict[str, type] = {}
    try:
        from src.providers.openai_provider import OpenAIProvider

        factory["openai"] = OpenAIProvider
    except ImportError:
        pass
    try:
        from src.providers.anthropic_provider import AnthropicProvider

        factory["anthropic"] = AnthropicProvider
    except ImportError:
        pass
    try:
        from src.providers.azure_provider import AzureProvider

        factory["azure"] = AzureProvider
    except ImportError:
        pass
    try:
        from src.providers.bedrock_provider import BedrockProvider

        factory["bedrock"] = BedrockProvider
    except ImportError:
        pass
    try:
        from src.providers.vertex_provider import VertexProvider

        factory["vertex"] = VertexProvider
    except ImportError:
        pass
    try:
        from src.providers.ollama_provider import OllamaProvider

        factory["ollama"] = OllamaProvider
    except ImportError:
        pass
    return factory


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup / shutdown."""
    cfg = reload_config()
    logger.info("gateway.starting", version=__version__, providers=list(cfg.providers.keys()))

    # --- Provider registry ---
    factory = _build_provider_factory()
    registry = ProviderRegistry.from_config(cfg, factory=factory)
    app.state.registry = registry

    # --- Router ---
    app.state.router = Router(registry=registry, routing_config=cfg.routing)

    # --- Response cache ---
    cache = None
    if cfg.caching.enabled:
        cache = ResponseCache(redis_url=cfg.caching.redis_url, default_ttl=cfg.caching.ttl_seconds)
        logger.info("cache.enabled", ttl=cfg.caching.ttl_seconds)
    app.state.cache = cache

    # --- Rate limiter ---
    rate_limiter = None
    if cfg.rate_limit.enabled:
        redis_url = cfg.rate_limit.redis_url if cfg.rate_limit.backend == "redis" else None
        rate_limiter = RateLimiter(
            rpm=cfg.rate_limit.requests_per_minute,
            tpm=cfg.rate_limit.tokens_per_minute,
            redis_url=redis_url,
        )
        logger.info("rate_limiter.enabled", rpm=cfg.rate_limit.requests_per_minute)
    app.state.rate_limiter = rate_limiter

    # --- Usage tracker ---
    from src.core.tracker import UsageTracker

    redis_url_for_tracker = cfg.caching.redis_url if cfg.caching.enabled else None
    tracker = UsageTracker(redis_url=redis_url_for_tracker)
    app.state.tracker = tracker

    # --- MCP server ---
    from src.mcp.server import MCPServer

    app.state.mcp_server = MCPServer(
        router=app.state.router,
        registry=registry,
        tracker=tracker,
    )

    logger.info("gateway.ready", providers=registry.list_providers())

    yield

    logger.info("gateway.shutting_down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="LLM Gateway",
        description="Unified multi-provider LLM proxy with OpenAI-compatible API",
        version=__version__,
        lifespan=lifespan,
    )

    cfg = get_config()

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Custom middleware (order: outermost first) ---
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(APIKeyAuth)

    # --- Prometheus /metrics ---
    metrics_app = prometheus_asgi_app()
    app.mount("/metrics", metrics_app)

    # --- API routes ---
    from src.api.routes import router as api_router

    app.include_router(api_router)

    # --- MCP routes ---
    from src.mcp.routes import router as mcp_router

    app.include_router(mcp_router)

    return app


app = create_app()

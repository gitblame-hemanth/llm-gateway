"""Shared fixtures for the LLM Gateway test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.models import ChatCompletionRequest, Message
from src.api.routes import router
from src.core.config import FallbackChain, ModelMapping, RoutingConfig
from src.providers.base import (
    EmbeddingResponse,
    LLMProvider,
    ProviderResponse,
    UsageStats,
)
from src.providers.registry import ProviderRegistry
from src.routing.router import Router

# ---------------------------------------------------------------------------
# Provider fixtures
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    """Concrete mock provider for testing."""

    provider_name = "mock"

    def __init__(self, models: list[str] | None = None, **kwargs: Any) -> None:
        self._models = models or ["mock-model", "mock-embed"]

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs: Any,
    ) -> ProviderResponse | AsyncIterator[str]:
        return ProviderResponse(
            content="Hello from mock provider!",
            model=model,
            usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
            provider_name=self.provider_name,
            latency_ms=42.0,
        )

    async def embed(
        self,
        texts: list[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            model=model,
            usage=UsageStats(input_tokens=5, output_tokens=0, total_tokens=5),
        )

    async def list_models(self) -> list[str]:
        return list(self._models)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the singleton ProviderRegistry between tests."""
    ProviderRegistry.reset()
    yield
    ProviderRegistry.reset()


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def mock_registry(mock_provider: MockProvider) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("mock", mock_provider)
    return registry


# ---------------------------------------------------------------------------
# Request fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def sample_chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mock-model",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
        ],
    )


# ---------------------------------------------------------------------------
# Redis fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> MagicMock:
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    return redis


# ---------------------------------------------------------------------------
# FastAPI test app
# ---------------------------------------------------------------------------


@pytest.fixture
def app(mock_provider: MockProvider, mock_registry: ProviderRegistry) -> FastAPI:
    """Create a FastAPI test app with mocked providers."""
    application = FastAPI()
    application.include_router(router)

    routing_config = RoutingConfig(
        default_provider="mock",
        model_mapping={
            "gpt-4": ModelMapping(alias="gpt-4", provider="mock", model="mock-model"),
        },
        fallback_chains={
            "default": FallbackChain(providers=["mock"]),
        },
    )

    application.state.registry = mock_registry
    application.state.router = Router(mock_registry, routing_config)

    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)

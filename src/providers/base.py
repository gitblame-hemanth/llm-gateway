"""Abstract base for all LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    content: str
    model: str
    usage: UsageStats
    provider_name: str
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    model: str
    usage: UsageStats


class LLMProvider(ABC):
    """Abstract interface every provider must implement."""

    provider_name: str = "base"

    @abstractmethod
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
        """Send a chat completion request.

        When *stream=True*, returns an async iterator yielding content chunks.
        Otherwise returns a full ``ProviderResponse``.
        """

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings for the given texts."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Return the list of available model identifiers."""

    # Convenience timer -------------------------------------------------

    @staticmethod
    def _start_timer() -> float:
        return time.perf_counter()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)

"""Anthropic direct API provider."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from src.core.config import ProviderConfig
from src.core.exceptions import (
    AuthenticationError,
    ModelNotFound,
    ProviderError,
    ProviderUnavailable,
    RateLimitExceeded,
)
from src.providers.base import EmbeddingResponse, LLMProvider, ProviderResponse, UsageStats


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = AsyncAnthropic(
            api_key=config.api_key or None,
            base_url=config.api_base or None,
            timeout=float(config.timeout),
            max_retries=0,
        )
        self._max_retries = config.max_retries

    async def _retry(self, coro_factory, *, retries: int | None = None):
        retries = retries if retries is not None else self._max_retries
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await coro_factory()
            except anthropic.AuthenticationError as exc:
                raise AuthenticationError(str(exc), provider=self.provider_name) from exc
            except anthropic.NotFoundError as exc:
                raise ModelNotFound(str(exc), provider=self.provider_name) from exc
            except anthropic.RateLimitError as exc:
                retry_after = float(exc.response.headers.get("retry-after", 1)) if exc.response else 1
                if attempt == retries:
                    raise RateLimitExceeded(str(exc), provider=self.provider_name, retry_after=retry_after) from exc
                await asyncio.sleep(min(retry_after, 2**attempt))
                last_exc = exc
            except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError) as exc:
                if attempt == retries:
                    raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc
                await asyncio.sleep(2**attempt)
                last_exc = exc
            except anthropic.APIStatusError as exc:
                raise ProviderError(
                    str(exc),
                    provider=self.provider_name,
                    status_code=exc.status_code,
                ) from exc
        raise ProviderUnavailable(str(last_exc), provider=self.provider_name)

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        """Extract system prompt and convert to Anthropic message format."""
        system: str | None = None
        converted: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                converted.append({"role": msg["role"], "content": msg["content"]})
        return system, converted

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
        if stream:
            return self._stream(messages, model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        return await self._complete(messages, model, temperature=temperature, max_tokens=max_tokens, **kwargs)

    async def _complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> ProviderResponse:
        start = self._start_timer()
        system, msgs = self._to_anthropic_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system:
            create_kwargs["system"] = system

        resp = await self._retry(lambda: self._client.messages.create(**create_kwargs))

        content = ""
        for block in resp.content:
            if block.type == "text":
                content += block.text

        return ProviderResponse(
            content=content,
            model=resp.model,
            usage=UsageStats(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
            ),
            provider_name=self.provider_name,
            latency_ms=self._elapsed_ms(start),
            raw=resp.model_dump(),
        )

    async def _stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        system, msgs = self._to_anthropic_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system:
            create_kwargs["system"] = system

        stream = await self._retry(lambda: self._client.messages.create(**create_kwargs, stream=True))

        async def _iter():
            async for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield event.delta.text

        return _iter()

    async def embed(self, texts: list[str], model: str, **kwargs: Any) -> EmbeddingResponse:
        raise ProviderError(
            "Anthropic does not support embeddings",
            provider=self.provider_name,
        )

    async def list_models(self) -> list[str]:
        return [m.name for m in self._config.models if m.enabled]

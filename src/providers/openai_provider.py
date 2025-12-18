"""OpenAI provider with retry and streaming support."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from src.core.config import ProviderConfig
from src.core.exceptions import (
    AuthenticationError,
    ModelNotFound,
    ProviderError,
    ProviderUnavailable,
    RateLimitExceeded,
)
from src.providers.base import EmbeddingResponse, LLMProvider, ProviderResponse, UsageStats


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or None,
            base_url=config.api_base or None,
            timeout=float(config.timeout),
            max_retries=0,  # we handle retries ourselves
        )
        self._max_retries = config.max_retries

    async def _retry(self, coro_factory, *, retries: int | None = None):
        retries = retries if retries is not None else self._max_retries
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await coro_factory()
            except openai.AuthenticationError as exc:
                raise AuthenticationError(str(exc), provider=self.provider_name) from exc
            except openai.NotFoundError as exc:
                raise ModelNotFound(str(exc), provider=self.provider_name) from exc
            except openai.RateLimitError as exc:
                retry_after = float(exc.response.headers.get("retry-after", 1)) if exc.response else 1
                if attempt == retries:
                    raise RateLimitExceeded(str(exc), provider=self.provider_name, retry_after=retry_after) from exc
                await asyncio.sleep(min(retry_after, 2**attempt))
                last_exc = exc
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
                if attempt == retries:
                    raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc
                await asyncio.sleep(2**attempt)
                last_exc = exc
            except openai.APIStatusError as exc:
                raise ProviderError(
                    str(exc),
                    provider=self.provider_name,
                    status_code=exc.status_code,
                ) from exc
        raise ProviderUnavailable(str(last_exc), provider=self.provider_name)  # pragma: no cover

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

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

        resp = await self._retry(
            lambda: self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        )

        choice = resp.choices[0]
        usage = resp.usage
        return ProviderResponse(
            content=choice.message.content or "",
            model=resp.model,
            usage=UsageStats(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
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
        stream = await self._retry(
            lambda: self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
        )

        async def _iter():
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return _iter()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str], model: str, **kwargs: Any) -> EmbeddingResponse:
        resp = await self._retry(lambda: self._client.embeddings.create(model=model, input=texts, **kwargs))
        vectors = [item.embedding for item in resp.data]
        usage = resp.usage
        return EmbeddingResponse(
            embeddings=vectors,
            model=resp.model,
            usage=UsageStats(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )

    async def list_models(self) -> list[str]:
        resp = await self._retry(lambda: self._client.models.list())
        return sorted(m.id for m in resp.data)

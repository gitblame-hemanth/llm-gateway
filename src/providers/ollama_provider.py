"""Ollama local provider — communicates via HTTP to localhost:11434."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.core.config import ProviderConfig
from src.core.exceptions import (
    ModelNotFound,
    ProviderError,
    ProviderUnavailable,
)
from src.providers.base import EmbeddingResponse, LLMProvider, ProviderResponse, UsageStats


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._base_url = (config.api_base or "http://localhost:11434").rstrip("/")
        self._timeout = config.timeout
        self._max_retries = config.max_retries

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )

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
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with self._client() as client:
            try:
                resp = await client.post("/api/chat", json=payload)
            except httpx.ConnectError as exc:
                raise ProviderUnavailable(
                    f"Cannot connect to Ollama at {self._base_url}: {exc}",
                    provider=self.provider_name,
                    model=model,
                ) from exc
            except httpx.TimeoutException as exc:
                raise ProviderUnavailable(
                    f"Ollama request timed out: {exc}",
                    provider=self.provider_name,
                    model=model,
                ) from exc

        if resp.status_code == 404:
            raise ModelNotFound(
                f"Model '{model}' not found in Ollama",
                provider=self.provider_name,
                model=model,
            )
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama returned {resp.status_code}: {resp.text}",
                provider=self.provider_name,
                model=model,
                status_code=resp.status_code,
            )

        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")

        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return ProviderResponse(
            content=content,
            model=model,
            usage=UsageStats(
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
                total_tokens=prompt_eval_count + eval_count,
            ),
            provider_name=self.provider_name,
            latency_ms=self._elapsed_ms(start),
            raw=data,
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
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        client = self._client()

        async def _iter():
            try:
                async with client.stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code == 404:
                        raise ModelNotFound(
                            f"Model '{model}' not found in Ollama",
                            provider=self.provider_name,
                            model=model,
                        )
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise ProviderError(
                            f"Ollama returned {resp.status_code}: {body.decode()}",
                            provider=self.provider_name,
                            model=model,
                            status_code=resp.status_code,
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        msg = data.get("message", {})
                        chunk = msg.get("content", "")
                        if chunk:
                            yield chunk
            except httpx.ConnectError as exc:
                raise ProviderUnavailable(
                    f"Cannot connect to Ollama at {self._base_url}: {exc}",
                    provider=self.provider_name,
                    model=model,
                ) from exc
            finally:
                await client.aclose()

        return _iter()

    async def embed(self, texts: list[str], model: str, **kwargs: Any) -> EmbeddingResponse:
        all_embeddings: list[list[float]] = []

        async with self._client() as client:
            for text in texts:
                payload = {"model": model, "input": text}
                try:
                    resp = await client.post("/api/embed", json=payload)
                except httpx.ConnectError as exc:
                    raise ProviderUnavailable(
                        f"Cannot connect to Ollama at {self._base_url}: {exc}",
                        provider=self.provider_name,
                        model=model,
                    ) from exc

                if resp.status_code == 404:
                    raise ModelNotFound(
                        f"Embedding model '{model}' not found in Ollama",
                        provider=self.provider_name,
                        model=model,
                    )
                if resp.status_code >= 400:
                    raise ProviderError(
                        f"Ollama returned {resp.status_code}: {resp.text}",
                        provider=self.provider_name,
                        model=model,
                        status_code=resp.status_code,
                    )

                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    all_embeddings.append(embeddings[0])

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=model,
            usage=UsageStats(input_tokens=0, output_tokens=0, total_tokens=0),
        )

    async def list_models(self) -> list[str]:
        async with self._client() as client:
            try:
                resp = await client.get("/api/tags")
            except httpx.ConnectError as exc:
                raise ProviderUnavailable(
                    f"Cannot connect to Ollama at {self._base_url}: {exc}",
                    provider=self.provider_name,
                ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama returned {resp.status_code}: {resp.text}",
                provider=self.provider_name,
                status_code=resp.status_code,
            )

        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

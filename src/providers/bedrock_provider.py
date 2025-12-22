"""AWS Bedrock provider using the Claude Messages API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from src.core.config import ProviderConfig
from src.core.exceptions import (
    AuthenticationError,
    ModelNotFound,
    ProviderError,
    ProviderUnavailable,
    RateLimitExceeded,
)
from src.providers.base import EmbeddingResponse, LLMProvider, ProviderResponse, UsageStats


class BedrockProvider(LLMProvider):
    provider_name = "bedrock"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._region = config.region or "us-east-1"
        self._max_retries = config.max_retries
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            import boto3

            session = boto3.Session(region_name=self._region)
            self._client = session.client("bedrock-runtime")
        return self._client

    async def _invoke(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    client.invoke_model,
                    modelId=model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                return json.loads(response["body"].read())
            except client.exceptions.AccessDeniedException as exc:
                raise AuthenticationError(str(exc), provider=self.provider_name, model=model) from exc
            except client.exceptions.ResourceNotFoundException as exc:
                raise ModelNotFound(str(exc), provider=self.provider_name, model=model) from exc
            except client.exceptions.ThrottlingException as exc:
                if attempt == self._max_retries:
                    raise RateLimitExceeded(str(exc), provider=self.provider_name, model=model) from exc
                await asyncio.sleep(2**attempt)
                last_exc = exc
            except client.exceptions.ModelTimeoutException as exc:
                if attempt == self._max_retries:
                    raise ProviderUnavailable(str(exc), provider=self.provider_name, model=model) from exc
                await asyncio.sleep(2**attempt)
                last_exc = exc
            except Exception as exc:
                if attempt == self._max_retries:
                    raise ProviderError(str(exc), provider=self.provider_name, model=model) from exc
                await asyncio.sleep(2**attempt)
                last_exc = exc

        raise ProviderUnavailable(str(last_exc), provider=self.provider_name, model=model)

    async def _invoke_stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[str]:
        client = self._get_client()

        response = await asyncio.to_thread(
            client.invoke_model_with_response_stream,
            modelId=model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        stream = response["body"]

        async def _iter():
            for event in stream:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

        return _iter()

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
        # Separate system message for Claude Messages API
        system_parts: list[str] = []
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        if stream:
            return await self._invoke_stream(model, body)

        start = self._start_timer()
        resp = await self._invoke(model, body)

        content = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = resp.get("usage", {})
        return ProviderResponse(
            content=content,
            model=model,
            usage=UsageStats(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            ),
            provider_name=self.provider_name,
            latency_ms=self._elapsed_ms(start),
            raw=resp,
        )

    async def embed(self, texts: list[str], model: str, **kwargs: Any) -> EmbeddingResponse:
        # Bedrock supports Titan embeddings
        client = self._get_client()
        all_embeddings: list[list[float]] = []
        total_input = 0

        for text in texts:
            body = {"inputText": text}
            resp = await asyncio.to_thread(
                client.invoke_model,
                modelId=model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(resp["body"].read())
            all_embeddings.append(result["embedding"])
            total_input += result.get("inputTextTokenCount", 0)

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=model,
            usage=UsageStats(input_tokens=total_input, output_tokens=0, total_tokens=total_input),
        )

    async def list_models(self) -> list[str]:
        return [m.name for m in self._config.models if m.enabled]

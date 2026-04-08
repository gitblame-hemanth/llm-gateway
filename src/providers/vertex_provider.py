"""GCP Vertex AI provider with lazy import of google-cloud dependencies."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from src.core.config import ProviderConfig
from src.core.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderUnavailable,
    RateLimitExceeded,
)
from src.providers.base import EmbeddingResponse, LLMProvider, ProviderResponse, UsageStats


class VertexProvider(LLMProvider):
    provider_name = "vertex"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._project = config.project_id
        self._region = config.region or "us-central1"
        self._max_retries = config.max_retries
        self._model_client = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        try:
            import vertexai

            vertexai.init(project=self._project, location=self._region)
        except ImportError as exc:
            raise ProviderError(
                "google-cloud-aiplatform is not installed. Install it with: pip install google-cloud-aiplatform",
                provider=self.provider_name,
            ) from exc
        except Exception as exc:
            raise AuthenticationError(
                f"Failed to initialize Vertex AI: {exc}",
                provider=self.provider_name,
            ) from exc
        self._initialized = True

    def _get_generative_model(self, model: str):
        self._ensure_init()
        from vertexai.generative_models import GenerativeModel

        return GenerativeModel(model)

    async def _retry(self, coro_factory, *, retries: int | None = None):
        retries = retries if retries is not None else self._max_retries
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                exc_str = str(exc).lower()
                if "403" in exc_str or "permission" in exc_str:
                    raise AuthenticationError(str(exc), provider=self.provider_name) from exc
                if "429" in exc_str or "quota" in exc_str:
                    if attempt == retries:
                        raise RateLimitExceeded(str(exc), provider=self.provider_name) from exc
                    await asyncio.sleep(2**attempt)
                    last_exc = exc
                    continue
                if "503" in exc_str or "unavailable" in exc_str:
                    if attempt == retries:
                        raise ProviderUnavailable(str(exc), provider=self.provider_name) from exc
                    await asyncio.sleep(2**attempt)
                    last_exc = exc
                    continue
                if attempt == retries:
                    raise ProviderError(str(exc), provider=self.provider_name) from exc
                await asyncio.sleep(2**attempt)
                last_exc = exc
        raise ProviderUnavailable(str(last_exc), provider=self.provider_name)

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
            return await self._stream(messages, model, temperature=temperature, max_tokens=max_tokens, **kwargs)
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
        from vertexai.generative_models import GenerationConfig

        start = self._start_timer()
        gen_model = self._get_generative_model(model)

        # Convert OpenAI-style messages to Vertex contents
        contents = self._convert_messages(messages)
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        resp = await self._retry(
            lambda: asyncio.to_thread(
                gen_model.generate_content,
                contents,
                generation_config=generation_config,
            )
        )

        text = resp.text if resp.text else ""
        usage_meta = resp.usage_metadata
        input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        return ProviderResponse(
            content=text,
            model=model,
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            provider_name=self.provider_name,
            latency_ms=self._elapsed_ms(start),
            raw={"text": text},
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
        from vertexai.generative_models import GenerationConfig

        gen_model = self._get_generative_model(model)
        contents = self._convert_messages(messages)
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        resp_iter = await asyncio.to_thread(
            gen_model.generate_content,
            contents,
            generation_config=generation_config,
            stream=True,
        )

        async def _iter():
            for chunk in resp_iter:
                if chunk.text:
                    yield chunk.text

        return _iter()

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Vertex AI Content format."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # Vertex doesn't have a system role; prepend as user context
                contents.insert(0, {"role": "user", "parts": [{"text": f"[System] {msg['content']}"}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
            else:
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
        return contents

    async def embed(self, texts: list[str], model: str, **kwargs: Any) -> EmbeddingResponse:
        self._ensure_init()
        from vertexai.language_models import TextEmbeddingModel

        embed_model = TextEmbeddingModel.from_pretrained(model)
        embeddings_list = await asyncio.to_thread(embed_model.get_embeddings, texts)
        vectors = [e.values for e in embeddings_list]

        return EmbeddingResponse(
            embeddings=vectors,
            model=model,
            usage=UsageStats(input_tokens=0, output_tokens=0, total_tokens=0),
        )

    async def list_models(self) -> list[str]:
        return [m.name for m in self._config.models if m.enabled]

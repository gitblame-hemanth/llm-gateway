"""FastAPI route definitions -- OpenAI-compatible REST API."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src import __version__
from src.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    EmbeddingData,
    EmbeddingRequest,
    HealthResponse,
    Message,
    ModelInfo,
    ModelListResponse,
    Usage,
)
from src.api.models import (
    EmbeddingResponse as EmbeddingResponseModel,
)
from src.core.exceptions import ModelNotFound, ProviderError
from src.providers.base import LLMProvider, ProviderResponse
from src.providers.registry import ProviderRegistry
from src.routing.router import Router

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_router(request: Request) -> Router:
    return request.app.state.router


def _get_registry(request: Request) -> ProviderRegistry:
    return request.app.state.registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_key(messages: list[dict], model: str) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


async def _stream_sse(
    provider: LLMProvider,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[str]:
    """Yield SSE-formatted chat completion chunks from a provider stream."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    stream: AsyncIterator[str] = await provider.chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    async for token in stream:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with finish_reason
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


@router.get("/usage")
async def usage_stats(request: Request) -> dict:
    """Return usage statistics for the requesting API key."""
    api_key = request.headers.get("X-API-Key", "anonymous")
    tracker = getattr(request.app.state, "tracker", None)
    if tracker is None:
        return {"api_key": api_key, "usage": {}, "message": "Tracking disabled"}
    stats = tracker.get_usage(api_key)
    return {"api_key": api_key, "usage": stats}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(registry: ProviderRegistry = Depends(_get_registry)) -> ModelListResponse:
    models: list[ModelInfo] = []
    for provider_name in registry.list_providers():
        provider = registry.get(provider_name)
        try:
            provider_models = await provider.list_models()
            for m in provider_models:
                models.append(ModelInfo(id=m, owned_by=provider_name))
        except Exception:
            logger.warning("models.list_failed", provider=provider_name)
    return ModelListResponse(data=models)


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completion(
    body: ChatCompletionRequest,
    request: Request,
    app_router: Router = Depends(_get_router),
):
    api_key = request.headers.get("X-API-Key", "anonymous")

    # --- Rate limit check ---
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is not None:
        allowed, retry_after = rate_limiter.check(api_key)
        if not allowed:
            return JSONResponse(
                {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

    try:
        provider, resolved_model = app_router.resolve(body.model)
    except ModelNotFound:
        raise HTTPException(status_code=404, detail=f"Model {body.model!r} not found")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    # --- Cache check (skip for streaming) ---
    cache = getattr(request.app.state, "cache", None)
    if not body.stream and cache is not None:
        key = _cache_key(messages, resolved_model)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("cache.hit", model=resolved_model)
            return JSONResponse(content=cached)

    # --- Streaming ---
    if body.stream:
        return StreamingResponse(
            _stream_sse(provider, messages, resolved_model, body.temperature, body.max_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Non-streaming ---
    try:
        result: ProviderResponse = await provider.chat_completion(
            messages=messages,
            model=resolved_model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            stream=False,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    response = ChatCompletionResponse(
        model=result.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=result.content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        provider=result.provider_name,
    )

    # --- Cache store ---
    if cache is not None:
        key = _cache_key(messages, resolved_model)
        cache.set(key, response.model_dump())

    # --- Usage tracking ---
    tracker = getattr(request.app.state, "tracker", None)
    if tracker is not None:
        tracker.record(
            api_key=api_key,
            model=resolved_model,
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
        )

    return response


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@router.post("/v1/embeddings", response_model=EmbeddingResponseModel)
async def create_embedding(
    body: EmbeddingRequest,
    request: Request,
    app_router: Router = Depends(_get_router),
) -> EmbeddingResponseModel:
    api_key = request.headers.get("X-API-Key", "anonymous")

    # --- Rate limit check ---
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is not None:
        allowed, retry_after = rate_limiter.check(api_key)
        if not allowed:
            return JSONResponse(
                {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

    texts = body.input if isinstance(body.input, list) else [body.input]
    try:
        provider, resolved_model = app_router.resolve(body.model)
    except ModelNotFound:
        raise HTTPException(status_code=404, detail=f"Model {body.model!r} not found")

    try:
        result = await provider.embed(texts=texts, model=resolved_model)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    response = EmbeddingResponseModel(
        model=result.model,
        data=[EmbeddingData(index=i, embedding=emb) for i, emb in enumerate(result.embeddings)],
        usage=Usage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        ),
    )

    # --- Usage tracking ---
    tracker = getattr(request.app.state, "tracker", None)
    if tracker is not None:
        tracker.record(
            api_key=api_key,
            model=resolved_model,
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=0,
        )

    return response

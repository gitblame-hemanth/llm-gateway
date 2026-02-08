"""Pydantic request / response models -- OpenAI-compatible API surface."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str
    name: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Chat Completion
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 1.0
    max_tokens: int = 4096
    stream: bool = False
    top_p: float = 1.0
    n: int = 1
    stop: list[str] | str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    provider: str = ""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] | str


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "llm-gateway"
    created: int = Field(default_factory=lambda: int(time.time()))


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""

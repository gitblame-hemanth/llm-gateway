"""Provider pricing data for LLM models."""

from __future__ import annotations

from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)


class ModelPricing(NamedTuple):
    """Cost per 1 000 tokens for input and output."""

    input_per_1k: float
    output_per_1k: float


# ---------------------------------------------------------------------------
# Pricing table — keyed by (provider, model).  A bare model key (no provider)
# acts as a default when no provider-specific entry exists.
# ---------------------------------------------------------------------------

PRICING: dict[tuple[str, str] | str, ModelPricing] = {
    # OpenAI -----------------------------------------------------------------
    ("openai", "gpt-4"): ModelPricing(0.03, 0.06),
    ("openai", "gpt-4-turbo"): ModelPricing(0.01, 0.03),
    ("openai", "gpt-4o"): ModelPricing(0.005, 0.015),
    ("openai", "gpt-4o-mini"): ModelPricing(0.00015, 0.0006),
    ("openai", "gpt-4.1"): ModelPricing(0.002, 0.008),
    ("openai", "gpt-4.1-mini"): ModelPricing(0.0004, 0.0016),
    ("openai", "gpt-4.1-nano"): ModelPricing(0.0001, 0.0004),
    ("openai", "gpt-3.5-turbo"): ModelPricing(0.0005, 0.0015),
    ("openai", "o3"): ModelPricing(0.01, 0.04),
    ("openai", "o3-mini"): ModelPricing(0.0011, 0.0044),
    ("openai", "o4-mini"): ModelPricing(0.0011, 0.0044),
    # Anthropic --------------------------------------------------------------
    ("anthropic", "claude-3-opus"): ModelPricing(0.015, 0.075),
    ("anthropic", "claude-3.5-sonnet"): ModelPricing(0.003, 0.015),
    ("anthropic", "claude-3.5-haiku"): ModelPricing(0.0008, 0.004),
    ("anthropic", "claude-3-haiku"): ModelPricing(0.00025, 0.00125),
    ("anthropic", "claude-4-opus"): ModelPricing(0.015, 0.075),
    ("anthropic", "claude-4-sonnet"): ModelPricing(0.003, 0.015),
    # Google -----------------------------------------------------------------
    ("google", "gemini-2.0-flash"): ModelPricing(0.0001, 0.0004),
    ("google", "gemini-2.5-pro"): ModelPricing(0.00125, 0.01),
    ("google", "gemini-2.5-flash"): ModelPricing(0.00015, 0.0006),
    ("google", "gemini-1.5-pro"): ModelPricing(0.00125, 0.005),
    ("google", "gemini-1.5-flash"): ModelPricing(0.000075, 0.0003),
    # AWS Bedrock / Titan ----------------------------------------------------
    ("aws", "titan-text-express"): ModelPricing(0.0002, 0.0006),
    ("aws", "titan-text-lite"): ModelPricing(0.00015, 0.0002),
    # Fallback defaults (no provider prefix) ---------------------------------
    "gpt-4": ModelPricing(0.03, 0.06),
    "gpt-4-turbo": ModelPricing(0.01, 0.03),
    "gpt-4o": ModelPricing(0.005, 0.015),
    "gpt-3.5-turbo": ModelPricing(0.0005, 0.0015),
    "claude-3-opus": ModelPricing(0.015, 0.075),
    "claude-3.5-sonnet": ModelPricing(0.003, 0.015),
    "claude-3-haiku": ModelPricing(0.00025, 0.00125),
    "gemini-1.5-pro": ModelPricing(0.00125, 0.005),
}


def get_price(model: str, provider: str | None = None) -> tuple[float, float]:
    """Return ``(input_cost_per_1k, output_cost_per_1k)`` for *model*.

    Looks up ``(provider, model)`` first, then falls back to bare *model*.
    Raises ``KeyError`` when no pricing is found.
    """
    if provider:
        key: tuple[str, str] | str = (provider, model)
        if key in PRICING:
            p = PRICING[key]
            return p.input_per_1k, p.output_per_1k

    if model in PRICING:
        p = PRICING[model]  # type: ignore[assignment]
        return p.input_per_1k, p.output_per_1k

    raise KeyError(f"No pricing data for model={model!r}, provider={provider!r}")

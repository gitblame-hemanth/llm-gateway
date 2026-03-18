"""Cost calculation from usage stats and pricing data."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from src.cost.pricing import get_price
from src.providers.base import UsageStats

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    provider: str


class CostCalculator:
    """Calculate dollar costs from token usage."""

    def calculate(
        self,
        usage: UsageStats,
        model: str,
        provider: str | None = None,
    ) -> CostBreakdown:
        """Return a :class:`CostBreakdown` for the given *usage*.

        Returns zero costs when pricing data is unavailable.
        """
        try:
            input_per_1k, output_per_1k = get_price(model, provider)
        except KeyError:
            logger.warning("cost.no_pricing", model=model, provider=provider)
            return CostBreakdown(
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                model=model,
                provider=provider or "",
            )

        input_cost = (usage.input_tokens / 1000) * input_per_1k
        output_cost = (usage.output_tokens / 1000) * output_per_1k
        return CostBreakdown(
            input_cost=round(input_cost, 8),
            output_cost=round(output_cost, 8),
            total_cost=round(input_cost + output_cost, 8),
            model=model,
            provider=provider or "",
        )

    def estimate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str | None = None,
    ) -> CostBreakdown:
        """Estimate cost without a full :class:`UsageStats`."""
        usage = UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return self.calculate(usage, model, provider)

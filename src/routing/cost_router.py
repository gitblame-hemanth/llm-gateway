"""Cost-optimised routing — pick the cheapest provider that supports a model."""

from __future__ import annotations

from typing import Any

import structlog

from src.core.config import GatewayConfig, get_config
from src.cost.pricing import get_price

logger = structlog.get_logger(__name__)


class CostRouter:
    """Select the cheapest provider for a given model request.

    Scans all enabled providers that list the requested model (or all providers
    if none explicitly list it) and picks the one with the lowest combined
    input + output rate from the pricing table.
    """

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self._cfg = config or get_config()

    def route(self, model: str, messages: list[dict[str, Any]] | None = None) -> str:
        """Return the cheapest provider name for *model*.

        Falls back to ``routing.default_provider`` when no pricing data is
        available for any candidate.
        """
        candidates = self._candidates_for(model)

        if not candidates:
            logger.debug("cost_router_no_candidates_default", model=model)
            return self._cfg.routing.default_provider

        best_provider: str | None = None
        best_cost = float("inf")

        for provider in candidates:
            try:
                input_rate, output_rate = get_price(model, provider)
                combined = input_rate + output_rate
                if combined < best_cost:
                    best_cost = combined
                    best_provider = provider
            except KeyError:
                continue

        if best_provider is None:
            logger.warning("cost_router_no_pricing_fallback", model=model)
            return self._cfg.routing.default_provider

        logger.info(
            "cost_router_selected",
            model=model,
            provider=best_provider,
            combined_rate=best_cost,
        )
        return best_provider

    def _candidates_for(self, model: str) -> list[str]:
        """Return providers that support *model* (enabled only)."""
        candidates: list[str] = []
        for pname, pcfg in self._cfg.providers.items():
            if not pcfg.enabled:
                continue
            # If provider explicitly lists models, check membership
            if pcfg.models:
                if any(m.name == model and m.enabled for m in pcfg.models):
                    candidates.append(pname)
            else:
                # Provider has no explicit model list — assume it might support it
                candidates.append(pname)
        return candidates

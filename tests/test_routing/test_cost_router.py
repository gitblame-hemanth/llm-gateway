"""Tests for cost-based routing logic."""

from __future__ import annotations

from src.cost.calculator import CostCalculator
from src.cost.pricing import get_price
from src.providers.base import UsageStats


class TestCostBasedRouting:
    """Verify cheapest provider selection via pricing data."""

    def test_cheapest_provider_selection(self) -> None:
        """Given multiple providers for the same model, pick the cheapest."""
        calc = CostCalculator()
        usage = UsageStats(input_tokens=1000, output_tokens=500, total_tokens=1500)

        # gpt-4 with openai pricing
        openai_cost = calc.calculate(usage, "gpt-4", provider="openai")
        # claude-3-haiku with anthropic pricing (should be cheaper than gpt-4)
        haiku_cost = calc.calculate(usage, "claude-3-haiku", provider="anthropic")

        assert haiku_cost.total_cost < openai_cost.total_cost

    def test_select_cheapest_across_providers(self) -> None:
        """Compare costs across providers and verify ordering."""
        calc = CostCalculator()
        usage = UsageStats(input_tokens=1000, output_tokens=1000, total_tokens=2000)

        candidates = [
            ("openai", "gpt-4"),
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-haiku"),
        ]

        costs = []
        for prov, model in candidates:
            breakdown = calc.calculate(usage, model, provider=prov)
            costs.append((prov, model, breakdown.total_cost))

        costs.sort(key=lambda x: x[2])
        # gpt-4 is the most expensive
        assert costs[-1][1] == "gpt-4"

    def test_missing_model_returns_zero_cost(self) -> None:
        """Unknown model should return zero cost, not raise."""
        calc = CostCalculator()
        usage = UsageStats(input_tokens=100, output_tokens=100, total_tokens=200)
        result = calc.calculate(usage, "nonexistent-model-xyz", provider="unknown")
        assert result.total_cost == 0.0
        assert result.input_cost == 0.0
        assert result.output_cost == 0.0

    def test_pricing_lookup_with_provider(self) -> None:
        """get_price should prefer (provider, model) key."""
        inp, out = get_price("gpt-4o", provider="openai")
        assert inp == 0.005
        assert out == 0.015

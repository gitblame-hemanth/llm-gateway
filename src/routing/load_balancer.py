"""Load balancing strategies for distributing requests across providers."""

from __future__ import annotations

import abc
import threading
from collections import defaultdict
from collections.abc import Sequence

import structlog

from src.cost.pricing import get_price

logger = structlog.get_logger(__name__)


class LoadBalancer(abc.ABC):
    """Base class for load-balancing strategies."""

    @abc.abstractmethod
    def pick(self, providers: Sequence[str], model: str) -> str:
        """Select one provider from *providers* for the given *model*."""

    def record_latency(self, provider: str, latency_ms: float) -> None:  # noqa: B027
        """Optional hook to record observed latency (used by latency-based)."""

    def record_cost(self, provider: str, cost: float) -> None:  # noqa: B027
        """Optional hook to record observed cost."""


# ---------------------------------------------------------------------------
# Round-robin
# ---------------------------------------------------------------------------


class RoundRobinBalancer(LoadBalancer):
    """Rotates through providers sequentially, per model."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def pick(self, providers: Sequence[str], model: str) -> str:
        if not providers:
            raise ValueError("No providers available")
        with self._lock:
            idx = self._counters[model] % len(providers)
            self._counters[model] += 1
        choice = providers[idx]
        logger.debug("round_robin_pick", model=model, provider=choice, index=idx)
        return choice


# ---------------------------------------------------------------------------
# Latency-based (exponential moving average)
# ---------------------------------------------------------------------------

_DEFAULT_EMA_ALPHA = 0.3
_DEFAULT_INITIAL_MS = 500.0


class LatencyBasedBalancer(LoadBalancer):
    """Routes to the provider with the lowest moving-average latency."""

    def __init__(self, alpha: float = _DEFAULT_EMA_ALPHA, initial_ms: float = _DEFAULT_INITIAL_MS) -> None:
        self._alpha = alpha
        self._initial = initial_ms
        self._ema: dict[str, float] = {}
        self._lock = threading.Lock()

    def pick(self, providers: Sequence[str], model: str) -> str:
        if not providers:
            raise ValueError("No providers available")
        with self._lock:
            best = min(providers, key=lambda p: self._ema.get(p, self._initial))
        logger.debug(
            "latency_pick",
            model=model,
            provider=best,
            latency_ema_ms=round(self._ema.get(best, self._initial), 2),
        )
        return best

    def record_latency(self, provider: str, latency_ms: float) -> None:
        with self._lock:
            prev = self._ema.get(provider, self._initial)
            self._ema[provider] = self._alpha * latency_ms + (1 - self._alpha) * prev
        logger.debug("latency_recorded", provider=provider, latency_ms=round(latency_ms, 2))


# ---------------------------------------------------------------------------
# Cost-based
# ---------------------------------------------------------------------------


class CostBasedBalancer(LoadBalancer):
    """Routes to the cheapest provider for the requested model (by pricing table)."""

    def pick(self, providers: Sequence[str], model: str) -> str:
        if not providers:
            raise ValueError("No providers available")

        cheapest: str | None = None
        lowest_cost = float("inf")

        for provider in providers:
            try:
                input_rate, output_rate = get_price(model, provider)
                # Simple combined rate as proxy
                combined = input_rate + output_rate
                if combined < lowest_cost:
                    lowest_cost = combined
                    cheapest = provider
            except KeyError:
                logger.debug("cost_balancer_no_pricing", provider=provider, model=model)
                continue

        if cheapest is None:
            # No pricing data — fall back to first provider
            cheapest = providers[0]
            logger.warning("cost_balancer_no_pricing_data_fallback", model=model, provider=cheapest)
        else:
            logger.debug("cost_pick", model=model, provider=cheapest, cost=lowest_cost)

        return cheapest


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, type[LoadBalancer]] = {
    "round_robin": RoundRobinBalancer,
    "latency": LatencyBasedBalancer,
    "cost": CostBasedBalancer,
}


def get_balancer(strategy: str = "round_robin") -> LoadBalancer:
    """Instantiate a load balancer by strategy name.

    Supported strategies: ``round_robin``, ``latency``, ``cost``.
    """
    cls = _STRATEGIES.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown balancing strategy {strategy!r}. Choose from {list(_STRATEGIES)}")
    return cls()

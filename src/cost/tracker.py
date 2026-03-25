"""Per-API-key usage tracking backed by Redis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.cost.calculator import CostCalculator

logger = structlog.get_logger(__name__)

_PREFIX = "llmgw:usage"


@dataclass
class _InMemoryStore:
    """Fallback when Redis is unavailable."""

    _data: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _key(self, api_key: str) -> dict[str, Any]:
        if api_key not in self._data:
            self._data[api_key] = {
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "request_count": 0,
                "records": [],
            }
        return self._data[api_key]

    def record(self, api_key: str, record: dict) -> None:
        d = self._key(api_key)
        d["total_cost"] += record["cost_usd"]
        d["total_input_tokens"] += record["input_tokens"]
        d["total_output_tokens"] += record["output_tokens"]
        d["request_count"] += 1
        d["records"].append(record)

    def get_totals(self, api_key: str) -> dict[str, Any]:
        d = self._key(api_key)
        return {
            "total_cost": d["total_cost"],
            "total_input_tokens": d["total_input_tokens"],
            "total_output_tokens": d["total_output_tokens"],
            "request_count": d["request_count"],
        }

    def check_budget(self, api_key: str, budget: float) -> bool:
        return self._key(api_key)["total_cost"] < budget


class UsageTracker:
    """Track per-API-key token and cost usage.

    Uses Redis when available, falls back to in-memory.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis: Any | None = None
        self._mem = _InMemoryStore()

        if redis_url:
            try:
                import redis as _redis

                self._redis = _redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("usage_tracker_redis_connected", url=redis_url)
            except Exception:
                logger.warning("usage_tracker_redis_unavailable_falling_back_to_memory")
                self._redis = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_usage(
        self,
        api_key: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: str | None = None,
    ) -> float:
        """Record a request's token usage. Returns estimated cost in USD."""
        _calc = CostCalculator()
        breakdown = _calc.estimate(model, input_tokens, output_tokens, provider)
        cost = breakdown.total_cost
        record = {
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "timestamp": time.time(),
        }

        if self._redis:
            try:
                rk = f"{_PREFIX}:{api_key}"
                self._redis.hincrbyfloat(rk, "total_cost", cost)
                self._redis.hincrby(rk, "total_input_tokens", input_tokens)
                self._redis.hincrby(rk, "total_output_tokens", output_tokens)
                self._redis.hincrby(rk, "request_count", 1)
                self._redis.rpush(f"{rk}:records", json.dumps(record))
            except Exception:
                logger.warning("usage_tracker_redis_write_failed")
                self._mem.record(api_key, record)
        else:
            self._mem.record(api_key, record)

        logger.info(
            "usage_recorded",
            api_key=api_key[:8] + "…",
            model=model,
            cost_usd=round(cost, 8),
        )
        return cost

    def get_totals(self, api_key: str) -> dict[str, Any]:
        """Return aggregate totals for the given API key."""
        if self._redis:
            try:
                rk = f"{_PREFIX}:{api_key}"
                raw = self._redis.hgetall(rk)
                if raw:
                    return {
                        "total_cost": float(raw.get("total_cost", 0)),
                        "total_input_tokens": int(raw.get("total_input_tokens", 0)),
                        "total_output_tokens": int(raw.get("total_output_tokens", 0)),
                        "request_count": int(raw.get("request_count", 0)),
                    }
            except Exception:
                logger.warning("usage_tracker_redis_read_failed")

        return self._mem.get_totals(api_key)

    def check_budget(self, api_key: str, budget: float) -> bool:
        """Return ``True`` if the key's total spend is below *budget* (USD)."""
        totals = self.get_totals(api_key)
        within = totals["total_cost"] < budget
        if not within:
            logger.warning(
                "budget_exceeded",
                api_key=api_key[:8] + "…",
                spent=totals["total_cost"],
                budget=budget,
            )
        return within

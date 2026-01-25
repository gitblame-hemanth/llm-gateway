"""Fallback execution — try providers in sequence until one succeeds."""

from __future__ import annotations

from typing import Any

import structlog

from src.core.exceptions import ProviderError, ProviderUnavailable
from src.providers.base import ProviderResponse
from src.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


class FallbackExecutor:
    """Execute a request against an ordered list of providers, falling back on failure."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        provider_names: list[str],
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Try each provider in *provider_names* until one succeeds.

        Raises the last :class:`ProviderError` / :class:`ProviderUnavailable`
        if all providers fail.
        """
        last_error: Exception | None = None
        for name in provider_names:
            if not self._registry.has(name):
                logger.warning("fallback.skip_unknown", provider=name)
                continue
            provider = self._registry.get(name)
            try:
                result = await provider.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                logger.info("fallback.success", provider=name, model=model)
                return result  # type: ignore[return-value]
            except (ProviderError, ProviderUnavailable) as exc:
                logger.warning("fallback.provider_failed", provider=name, error=str(exc))
                last_error = exc

        if last_error is not None:
            raise last_error
        raise ProviderError("No providers available in fallback chain")

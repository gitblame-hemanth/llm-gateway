"""Request router — resolve model/alias to a provider."""

from __future__ import annotations

import structlog

from src.core.config import RoutingConfig
from src.core.exceptions import ModelNotFound
from src.providers.base import LLMProvider
from src.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


class Router:
    """Route incoming requests to the correct :class:`LLMProvider`."""

    def __init__(self, registry: ProviderRegistry, routing_config: RoutingConfig) -> None:
        self._registry = registry
        self._config = routing_config

    # -- public API ----------------------------------------------------------

    def resolve(self, model: str) -> tuple[LLMProvider, str]:
        """Return ``(provider_instance, resolved_model_name)`` for *model*.

        Resolution order:
        1. Exact match in ``model_mapping``.
        2. Default provider (if it hosts the model).
        3. :class:`ModelNotFound` is raised.
        """
        # 1. Model mapping
        mapping = self._config.model_mapping.get(model)
        if mapping:
            provider = self._registry.get(mapping.provider)
            resolved_model = mapping.model or model
            logger.debug("router.mapped", alias=model, provider=mapping.provider, model=resolved_model)
            return provider, resolved_model

        # 2. Default provider
        default_name = self._config.default_provider
        if self._registry.has(default_name):
            logger.debug("router.default", model=model, provider=default_name)
            return self._registry.get(default_name), model

        raise ModelNotFound(
            f"No provider found for model {model!r}",
            model=model,
        )

    def get_fallback_chain(self, chain_name: str) -> list[str]:
        """Return the ordered list of provider names for *chain_name*."""
        chain = self._config.fallback_chains.get(chain_name)
        if chain is None:
            return []
        return list(chain.providers)

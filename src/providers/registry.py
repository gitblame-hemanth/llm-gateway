"""Provider registry — singleton that manages provider lifecycle."""

from __future__ import annotations

import importlib
import threading

import structlog

from src.core.config import GatewayConfig
from src.core.exceptions import ModelNotFound, ProviderUnavailable
from src.providers.base import LLMProvider

logger = structlog.get_logger(__name__)

# Provider class mapping — lazy imports happen at registration time
_PROVIDER_CLASSES: dict[str, str] = {
    "openai": "src.providers.openai_provider.OpenAIProvider",
    "anthropic": "src.providers.anthropic_provider.AnthropicProvider",
    "azure": "src.providers.azure_provider.AzureOpenAIProvider",
    "bedrock": "src.providers.bedrock_provider.BedrockProvider",
    "vertex": "src.providers.vertex_provider.VertexProvider",
    "ollama": "src.providers.ollama_provider.OllamaProvider",
}


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ProviderRegistry:
    """Thread-safe singleton registry for LLM providers."""

    _instance: ProviderRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ProviderRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._providers: dict[str, LLMProvider] = {}
                    cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str, provider: LLMProvider) -> None:
        """Register a provider instance under the given name."""
        self._providers[name] = provider
        logger.info("provider_registered", provider=name)

    def get(self, name: str) -> LLMProvider:
        """Retrieve a registered provider by name."""
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderUnavailable(
                f"Provider '{name}' is not registered",
                provider=name,
            )
        return provider

    def list_all(self) -> dict[str, LLMProvider]:
        """Return a copy of all registered providers."""
        return dict(self._providers)

    def list_providers(self) -> list[str]:
        """Return a sorted list of registered provider names."""
        return sorted(self._providers)

    def has(self, name: str) -> bool:
        return name in self._providers

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)
        logger.info("provider_unregistered", provider=name)

    def clear(self) -> None:
        self._providers.clear()

    # ------------------------------------------------------------------
    # Bulk init from config
    # ------------------------------------------------------------------

    def init_from_config(self, config: GatewayConfig) -> None:
        """Instantiate and register all enabled providers from configuration."""
        for name, provider_cfg in config.providers.items():
            if not provider_cfg.enabled:
                logger.info("provider_skipped", provider=name, reason="disabled")
                continue

            dotted = _PROVIDER_CLASSES.get(name)
            if dotted is None:
                logger.warning("provider_unknown", provider=name)
                continue

            try:
                cls = _import_class(dotted)
                instance = cls(provider_cfg)
                self.register(name, instance)
            except Exception:
                logger.exception("provider_init_failed", provider=name)

    @classmethod
    def from_config(
        cls,
        config: GatewayConfig,
        factory: dict[str, type[LLMProvider]] | None = None,
    ) -> ProviderRegistry:
        """Build a registry from config. If *factory* is provided, use it for class lookup."""
        registry = cls()
        if factory:
            for name, provider_cfg in config.providers.items():
                if not provider_cfg.enabled:
                    continue
                provider_cls = factory.get(name)
                if provider_cls is None:
                    continue
                try:
                    instance = provider_cls(provider_cfg)
                    registry.register(name, instance)
                except Exception:
                    logger.exception("provider_init_failed", provider=name)
        else:
            registry.init_from_config(config)
        return registry

    # ------------------------------------------------------------------
    # Convenience: resolve model alias -> (provider, model)
    # ------------------------------------------------------------------

    def resolve_model(self, model_alias: str, config: GatewayConfig) -> tuple[LLMProvider, str]:
        """Resolve a model alias to a (provider_instance, real_model_name) pair."""
        mapping = config.routing.model_mapping.get(model_alias)
        if mapping and mapping.provider:
            provider = self.get(mapping.provider)
            return provider, mapping.model or model_alias

        # Try to find any registered provider that has this model
        for name, provider in self._providers.items():
            provider_cfg = config.providers.get(name)
            if provider_cfg:
                for m in provider_cfg.models:
                    if m.name == model_alias and m.enabled:
                        return provider, model_alias

        raise ModelNotFound(
            f"No provider found for model '{model_alias}'",
            model=model_alias,
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._providers.clear()
                cls._instance = None

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(self, name: str) -> bool:
        return name in self._providers

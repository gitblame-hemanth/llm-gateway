"""Tests for src.providers.registry.ProviderRegistry."""

from __future__ import annotations

import pytest

from src.core.config import GatewayConfig, ProviderConfig
from src.core.exceptions import ProviderUnavailable
from src.providers.registry import ProviderRegistry
from tests.conftest import MockProvider


class TestProviderRegistry:
    """ProviderRegistry core operations."""

    def test_register_and_get(self) -> None:
        registry = ProviderRegistry()
        provider = MockProvider()
        registry.register("openai", provider)
        assert registry.get("openai") is provider

    def test_list_providers_sorted(self) -> None:
        registry = ProviderRegistry()
        registry.register("zeta", MockProvider())
        registry.register("alpha", MockProvider())
        registry.register("mid", MockProvider())
        assert registry.list_providers() == ["alpha", "mid", "zeta"]

    def test_unknown_provider_raises(self) -> None:
        registry = ProviderRegistry()
        with pytest.raises(ProviderUnavailable, match="not registered"):
            registry.get("nonexistent")

    def test_has_and_contains(self) -> None:
        registry = ProviderRegistry()
        registry.register("test", MockProvider())
        assert registry.has("test")
        assert "test" in registry
        assert not registry.has("missing")
        assert "missing" not in registry

    def test_unregister(self) -> None:
        registry = ProviderRegistry()
        registry.register("temp", MockProvider())
        assert registry.has("temp")
        registry.unregister("temp")
        assert not registry.has("temp")

    def test_len(self) -> None:
        registry = ProviderRegistry()
        assert len(registry) == 0
        registry.register("a", MockProvider())
        registry.register("b", MockProvider())
        assert len(registry) == 2

    def test_list_all(self) -> None:
        registry = ProviderRegistry()
        p1 = MockProvider()
        p2 = MockProvider()
        registry.register("x", p1)
        registry.register("y", p2)
        all_providers = registry.list_all()
        assert set(all_providers.keys()) == {"x", "y"}

    def test_from_config_skips_disabled(self) -> None:
        config = GatewayConfig(
            providers={
                "enabled_one": ProviderConfig(name="enabled_one", enabled=True),
                "disabled_one": ProviderConfig(name="disabled_one", enabled=False),
            }
        )

        class FactoryProvider(MockProvider):
            pass

        registry = ProviderRegistry.from_config(
            config,
            factory={"enabled_one": FactoryProvider},
        )
        assert registry.has("enabled_one")
        assert not registry.has("disabled_one")

    def test_from_config_missing_factory(self) -> None:
        config = GatewayConfig(
            providers={
                "unknown_provider": ProviderConfig(name="unknown_provider", enabled=True),
            }
        )
        registry = ProviderRegistry.from_config(config, factory={})
        assert not registry.has("unknown_provider")

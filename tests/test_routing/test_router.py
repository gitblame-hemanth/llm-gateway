"""Tests for src.routing.router.Router."""

from __future__ import annotations

import pytest

from src.core.config import FallbackChain, ModelMapping, RoutingConfig
from src.core.exceptions import ModelNotFound
from src.providers.registry import ProviderRegistry
from src.routing.router import Router
from tests.conftest import MockProvider


@pytest.fixture
def routing_config() -> RoutingConfig:
    return RoutingConfig(
        default_provider="mock",
        model_mapping={
            "gpt-4": ModelMapping(alias="gpt-4", provider="mock", model="mock-model"),
            "claude": ModelMapping(alias="claude", provider="anthropic", model="claude-3"),
        },
        fallback_chains={
            "high_quality": FallbackChain(providers=["anthropic", "mock"]),
        },
    )


@pytest.fixture
def registry_with_providers() -> ProviderRegistry:
    reg = ProviderRegistry()
    mock = MockProvider(models=["mock-model"])
    mock.provider_name = "mock"
    anth = MockProvider(models=["claude-3"])
    anth.provider_name = "anthropic"
    reg.register("mock", mock)
    reg.register("anthropic", anth)
    return reg


class TestRouter:
    """Router resolution logic."""

    def test_resolve_by_model_mapping(
        self, registry_with_providers: ProviderRegistry, routing_config: RoutingConfig
    ) -> None:
        r = Router(registry_with_providers, routing_config)
        provider, model = r.resolve("gpt-4")
        assert model == "mock-model"
        assert provider.provider_name == "mock"

    def test_resolve_mapping_different_provider(
        self, registry_with_providers: ProviderRegistry, routing_config: RoutingConfig
    ) -> None:
        r = Router(registry_with_providers, routing_config)
        provider, model = r.resolve("claude")
        assert model == "claude-3"
        assert provider.provider_name == "anthropic"

    def test_resolve_default_provider(
        self, registry_with_providers: ProviderRegistry, routing_config: RoutingConfig
    ) -> None:
        r = Router(registry_with_providers, routing_config)
        provider, model = r.resolve("some-unmapped-model")
        assert model == "some-unmapped-model"
        assert provider.provider_name == "mock"

    def test_resolve_unknown_model_no_default(self) -> None:
        reg = ProviderRegistry()
        config = RoutingConfig(default_provider="nonexistent")
        r = Router(reg, config)
        with pytest.raises(ModelNotFound):
            r.resolve("anything")

    def test_get_fallback_chain(self, registry_with_providers: ProviderRegistry, routing_config: RoutingConfig) -> None:
        r = Router(registry_with_providers, routing_config)
        chain = r.get_fallback_chain("high_quality")
        assert chain == ["anthropic", "mock"]

    def test_get_fallback_chain_missing(
        self, registry_with_providers: ProviderRegistry, routing_config: RoutingConfig
    ) -> None:
        r = Router(registry_with_providers, routing_config)
        assert r.get_fallback_chain("nonexistent") == []

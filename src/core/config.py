"""Configuration loading from YAML files with environment variable overrides."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR} and ${VAR:default} placeholders in config values."""
    if isinstance(value, str):

        def _replacer(m: re.Match) -> str:
            var_name, default = m.group(1), m.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_env_vars(raw)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    name: str
    enabled: bool = True
    max_tokens: int = 4096
    context_window: int = 128_000


class ProviderConfig(BaseModel):
    name: str
    enabled: bool = True
    api_key: str = ""
    api_base: str = ""
    region: str = ""
    project_id: str = ""
    endpoint: str = ""
    deployment: str = ""
    api_version: str = ""
    models: list[ModelConfig] = Field(default_factory=list)
    timeout: int = 60
    max_retries: int = 3
    extra: dict[str, Any] = Field(default_factory=dict)


class FallbackChain(BaseModel):
    providers: list[str] = Field(default_factory=list)


class ModelMapping(BaseModel):
    alias: str = ""
    provider: str = ""
    model: str = ""


class RoutingConfig(BaseModel):
    default_provider: str = "openai"
    fallback_chains: dict[str, FallbackChain] = Field(default_factory=dict)
    model_mapping: dict[str, ModelMapping] = Field(default_factory=dict)


class CachingConfig(BaseModel):
    enabled: bool = False
    backend: str = "redis"
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 3600
    max_size: int = 10_000


class RateLimitConfig(BaseModel):
    enabled: bool = False
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000
    backend: str = "memory"
    redis_url: str = "redis://localhost:6379/1"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class GatewayConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


def _build_config() -> GatewayConfig:
    main_cfg = _load_yaml("config.yaml")
    providers_cfg = _load_yaml("providers.yaml")
    routing_cfg = _load_yaml("routing.yaml")

    server = ServerConfig(**main_cfg.get("server", {}))

    # Build provider configs
    providers: dict[str, ProviderConfig] = {}
    for name, raw in providers_cfg.get("providers", {}).items():
        models_raw = raw.pop("models", [])
        models = [ModelConfig(**m) if isinstance(m, dict) else ModelConfig(name=m) for m in models_raw]
        providers[name] = ProviderConfig(name=name, models=models, **raw)

    # Routing
    routing_raw = routing_cfg.get("routing", {})
    fallback_chains = {
        k: FallbackChain(**v) if isinstance(v, dict) else FallbackChain(providers=v)
        for k, v in routing_raw.get("fallback_chains", {}).items()
    }
    model_mapping = {k: ModelMapping(**v) for k, v in routing_raw.get("model_mapping", {}).items()}
    routing = RoutingConfig(
        default_provider=routing_raw.get("default_provider", "openai"),
        fallback_chains=fallback_chains,
        model_mapping=model_mapping,
    )

    caching = CachingConfig(**routing_cfg.get("caching", {}))
    rate_limit = RateLimitConfig(**routing_cfg.get("rate_limit", {}))

    return GatewayConfig(
        server=server,
        providers=providers,
        routing=routing,
        caching=caching,
        rate_limit=rate_limit,
    )


@lru_cache(maxsize=1)
def get_config() -> GatewayConfig:
    """Return the singleton gateway configuration."""
    return _build_config()


def reload_config() -> GatewayConfig:
    """Clear cache and reload configuration from disk."""
    get_config.cache_clear()
    return get_config()

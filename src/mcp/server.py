"""MCP server — exposes LLM Gateway capabilities as MCP-compatible tools."""

from __future__ import annotations

import structlog

from src.providers.registry import ProviderRegistry
from src.routing.router import Router

logger = structlog.get_logger(__name__)


class MCPServer:
    """MCP server exposing gateway tools: chat_completion, list_models, list_providers, get_usage."""

    def __init__(self, router: Router, registry: ProviderRegistry, tracker=None) -> None:
        self._router = router
        self._registry = registry
        self._tracker = tracker
        self._tools = self._register_tools()

    def _register_tools(self) -> dict:
        return {
            "chat_completion": {
                "description": "Send a chat completion request through the LLM gateway",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "description": "Model to use (e.g. gpt-4o, claude-3-sonnet)"},
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                            "description": "Chat messages",
                        },
                        "temperature": {"type": "number", "default": 0.7},
                        "max_tokens": {"type": "integer", "default": 4096},
                    },
                    "required": ["model", "messages"],
                },
            },
            "list_models": {
                "description": "List all available models across configured providers",
                "input_schema": {"type": "object", "properties": {}},
            },
            "list_providers": {
                "description": "List configured LLM providers and their status",
                "input_schema": {"type": "object", "properties": {}},
            },
            "get_usage": {
                "description": "Get token usage and cost statistics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "description": "API key to query usage for"},
                        "period": {"type": "string", "enum": ["day", "week", "month"], "default": "day"},
                    },
                },
            },
        }

    def get_tool_definitions(self) -> list[dict]:
        """Return MCP-compatible tool definitions."""
        return [
            {"name": name, "description": tool["description"], "input_schema": tool["input_schema"]}
            for name, tool in self._tools.items()
        ]

    async def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call and return the result."""
        logger.info("mcp.tool_call", tool=tool_name)

        handlers = {
            "chat_completion": self._handle_chat_completion,
            "list_models": self._handle_list_models,
            "list_providers": self._handle_list_providers,
            "get_usage": self._handle_get_usage,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"Unknown MCP tool: {tool_name}")

        return await handler(arguments)

    async def _handle_chat_completion(self, arguments: dict) -> dict:
        model = arguments["model"]
        messages = arguments["messages"]
        temperature = arguments.get("temperature", 0.7)
        max_tokens = arguments.get("max_tokens", 4096)

        provider, resolved_model = self._router.resolve(model)

        response = await provider.chat_completion(
            messages=messages,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider_name,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "latency_ms": response.latency_ms,
        }

    async def _handle_list_models(self, _arguments: dict | None = None) -> dict:
        models = []
        for name, provider in self._registry.list_all().items():
            try:
                provider_models = await provider.list_models()
                for m in provider_models:
                    models.append({"id": m, "provider": name})
            except Exception:
                logger.warning("mcp.list_models.provider_failed", provider=name)
        return {"models": models}

    async def _handle_list_providers(self, _arguments: dict | None = None) -> dict:
        providers = []
        for name in self._registry.list_providers():
            providers.append({"name": name, "enabled": True})
        return {"providers": providers}

    async def _handle_get_usage(self, arguments: dict) -> dict:
        if self._tracker is None:
            return {"error": "Usage tracking not available"}

        api_key = arguments.get("api_key", "default")
        totals = self._tracker.get_totals(api_key)
        return {
            "api_key": api_key,
            "period": arguments.get("period", "day"),
            "tokens": {
                "input": totals.get("total_input_tokens", 0),
                "output": totals.get("total_output_tokens", 0),
            },
            "cost_usd": totals.get("total_cost", 0.0),
        }

"""Tests for MCP server and routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp.server import MCPServer


@pytest.fixture()
def mock_router():
    r = MagicMock()
    r.resolve.return_value = (AsyncMock(), "gpt-4o")
    return r


@pytest.fixture()
def mock_registry():
    reg = MagicMock()
    reg.list_all.return_value = {}
    reg.list_providers.return_value = ["openai", "bedrock"]
    return reg


@pytest.fixture()
def mcp_server(mock_router, mock_registry):
    return MCPServer(router=mock_router, registry=mock_registry, tracker=None)


class TestMCPToolDefinitions:
    def test_returns_tool_list(self, mcp_server):
        tools = mcp_server.get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) == 4
        names = {t["name"] for t in tools}
        assert names == {"chat_completion", "list_models", "list_providers", "get_usage"}

    def test_tool_has_schema(self, mcp_server):
        tools = mcp_server.get_tool_definitions()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool


class TestMCPListProviders:
    @pytest.mark.asyncio()
    async def test_list_providers(self, mcp_server):
        result = await mcp_server.handle_tool_call("list_providers", {})
        assert "providers" in result
        assert len(result["providers"]) == 2
        names = {p["name"] for p in result["providers"]}
        assert names == {"openai", "bedrock"}


class TestMCPGetUsage:
    @pytest.mark.asyncio()
    async def test_get_usage_no_tracker(self, mcp_server):
        result = await mcp_server.handle_tool_call("get_usage", {"api_key": "test"})
        assert "error" in result


class TestMCPUnknownTool:
    @pytest.mark.asyncio()
    async def test_unknown_tool_raises(self, mcp_server):
        with pytest.raises(ValueError, match="Unknown MCP tool"):
            await mcp_server.handle_tool_call("nonexistent", {})

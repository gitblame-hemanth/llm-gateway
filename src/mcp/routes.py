"""MCP API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.mcp.server import MCPServer
from src.mcp.transport import MCPTransport

router = APIRouter(prefix="/mcp", tags=["MCP"])


class ToolCallRequest(BaseModel):
    """Request body for MCP tool invocation."""

    tool: str
    arguments: dict = {}


def _get_mcp(request: Request) -> tuple[MCPServer, MCPTransport]:
    """Resolve MCP server and transport from app state."""
    server: MCPServer = request.app.state.mcp_server
    transport = MCPTransport(server)
    return server, transport


@router.get("/sse")
async def mcp_sse(request: Request) -> StreamingResponse:
    """MCP SSE endpoint — connect to discover available tools."""
    _, transport = _get_mcp(request)
    return await transport.handle_sse(request)


@router.post("/tools/call")
async def call_tool(body: ToolCallRequest, request: Request) -> dict:
    """Execute an MCP tool call."""
    _, transport = _get_mcp(request)
    return await transport.handle_tool_call(body.tool, body.arguments)


@router.get("/tools")
async def list_tools(request: Request) -> dict:
    """List all available MCP tools."""
    server, _ = _get_mcp(request)
    return {"tools": server.get_tool_definitions()}

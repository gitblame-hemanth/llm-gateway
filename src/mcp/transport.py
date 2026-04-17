"""MCP transport layer — SSE (Server-Sent Events) implementation."""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import Request
from fastapi.responses import StreamingResponse

from src.mcp.server import MCPServer

logger = structlog.get_logger(__name__)


class MCPTransport:
    """SSE-based transport for MCP protocol."""

    def __init__(self, mcp_server: MCPServer) -> None:
        self._server = mcp_server

    async def handle_sse(self, request: Request) -> StreamingResponse:
        """Handle SSE connection for MCP tool discovery."""
        return StreamingResponse(
            self._sse_generator(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _sse_generator(self, request: Request):
        """Yield SSE events: capabilities on connect, then keepalive pings."""
        yield _format_sse({"type": "capabilities", "tools": self._server.get_tool_definitions()})

        try:
            while not await request.is_disconnected():
                await asyncio.sleep(30)
                yield _format_sse({"type": "ping"})
        except asyncio.CancelledError:
            logger.debug("mcp.sse.disconnected")

    async def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """Execute an MCP tool call and wrap the result."""
        result = await self._server.handle_tool_call(tool_name, arguments)
        return {"type": "tool_result", "tool": tool_name, "result": result}


def _format_sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"

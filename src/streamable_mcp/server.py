from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

PROTOCOL_VERSION = "2025-03-26"

METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

TOOLS_CALL_METHOD = "tools/call"

StreamingTool = Callable[[dict[str, Any]], Iterable[dict[str, Any]]]


@dataclass(frozen=True)
class ServerInfo:
    name: str
    version: str


@dataclass
class MCPServer:
    info: ServerInfo
    capabilities: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, StreamingTool] = field(default_factory=dict)

    def register_stream_tool(self, name: str, tool: StreamingTool) -> None:
        self.tools[name] = tool

    def is_streaming(self, message: dict[str, Any]) -> bool:
        if message.get("method") != TOOLS_CALL_METHOD:
            return False
        tool_name = _tool_name(message)
        return tool_name is not None and tool_name in self.tools

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        if method == "initialize":
            return self._initialize(message)
        if method == "notifications/initialized":
            return None
        if method == TOOLS_CALL_METHOD:
            return self._reject_tools_call(message)
        return _error(message.get("id"), METHOD_NOT_FOUND, f"method not found: {method}")

    def stream_messages(self, message: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if message.get("method") != TOOLS_CALL_METHOD:
            reply = self.handle(message)
            if reply is not None:
                yield reply
            return
        tool_name = _tool_name(message)
        tool = self.tools.get(tool_name) if tool_name else None
        if tool is None:
            yield _error(
                message.get("id"),
                METHOD_NOT_FOUND,
                f"unknown tool: {tool_name}" if tool_name else "missing tool name",
            )
            return
        yield from tool(message)

    def _reject_tools_call(self, message: dict[str, Any]) -> dict[str, Any]:
        tool_name = _tool_name(message)
        if tool_name and tool_name in self.tools:
            return _error(
                message.get("id"),
                INVALID_PARAMS,
                "tool streams over SSE — set Accept: text/event-stream",
            )
        return _error(
            message.get("id"),
            METHOD_NOT_FOUND,
            f"unknown tool: {tool_name}" if tool_name else "missing tool name",
        )

    def _initialize(self, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": self.capabilities,
                "serverInfo": {"name": self.info.name, "version": self.info.version},
            },
        }


def _tool_name(message: dict[str, Any]) -> str | None:
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    name = params.get("name")
    return name if isinstance(name, str) else None


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }

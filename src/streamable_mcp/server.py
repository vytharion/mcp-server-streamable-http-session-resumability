from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "2025-03-26"

METHOD_NOT_FOUND = -32601


@dataclass(frozen=True)
class ServerInfo:
    name: str
    version: str


@dataclass
class MCPServer:
    info: ServerInfo
    capabilities: dict[str, Any] = field(default_factory=dict)

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        if method == "initialize":
            return self._initialize(message)
        if method == "notifications/initialized":
            return None
        return _error(message.get("id"), METHOD_NOT_FOUND, f"method not found: {method}")

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


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }

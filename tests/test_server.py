from __future__ import annotations

from typing import Any, Iterator

from streamable_mcp.server import PROTOCOL_VERSION, MCPServer, ServerInfo


def _make_server() -> MCPServer:
    return MCPServer(info=ServerInfo(name="demo", version="0.1.0"))


def _echo_tool(message: dict[str, Any]) -> Iterator[dict[str, Any]]:
    params = message.get("params") or {}
    text = (params.get("arguments") or {}).get("text", "")
    yield {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progress": 1, "total": 1},
    }
    yield {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {"content": [{"type": "text", "text": text}]},
    }


def test_initialize_returns_server_info() -> None:
    server = _make_server()

    response = server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )

    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert response["result"]["serverInfo"] == {"name": "demo", "version": "0.1.0"}
    assert response["result"]["capabilities"] == {}


def test_initialized_notification_is_absorbed() -> None:
    server = _make_server()

    response = server.handle(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}
    )

    assert response is None


def test_unknown_method_returns_method_not_found_error() -> None:
    server = _make_server()

    response = server.handle(
        {"jsonrpc": "2.0", "id": 42, "method": "does/not/exist"}
    )

    assert response is not None
    assert response["id"] == 42
    assert response["error"]["code"] == -32601
    assert "does/not/exist" in response["error"]["message"]


def test_custom_capabilities_are_advertised() -> None:
    server = MCPServer(
        info=ServerInfo(name="demo", version="0.1.0"),
        capabilities={"tools": {"listChanged": False}},
    )

    response = server.handle({"jsonrpc": "2.0", "id": 7, "method": "initialize"})

    assert response is not None
    assert response["result"]["capabilities"] == {"tools": {"listChanged": False}}


def test_is_streaming_true_only_for_registered_tools_call() -> None:
    server = _make_server()
    server.register_stream_tool("echo", _echo_tool)

    tool_message = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "echo", "arguments": {"text": "hi"}},
    }
    assert server.is_streaming(tool_message) is True
    assert server.is_streaming({"jsonrpc": "2.0", "id": 3, "method": "initialize"}) is False


def test_is_streaming_false_for_unknown_tool_name() -> None:
    server = _make_server()

    message = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "not-registered"},
    }

    assert server.is_streaming(message) is False


def test_stream_messages_yields_progress_then_final_result() -> None:
    server = _make_server()
    server.register_stream_tool("echo", _echo_tool)

    frames = list(
        server.stream_messages(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "hello"}},
            }
        )
    )

    assert len(frames) == 2
    assert frames[0]["method"] == "notifications/progress"
    assert frames[1]["id"] == 9
    assert frames[1]["result"]["content"][0]["text"] == "hello"


def test_stream_messages_yields_error_for_unknown_tool() -> None:
    server = _make_server()

    frames = list(
        server.stream_messages(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "missing"},
            }
        )
    )

    assert len(frames) == 1
    assert frames[0]["id"] == 5
    assert frames[0]["error"]["code"] == -32601
    assert "missing" in frames[0]["error"]["message"]


def test_handle_rejects_tools_call_when_tool_registered_asking_for_sse() -> None:
    server = _make_server()
    server.register_stream_tool("echo", _echo_tool)

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "echo"},
        }
    )

    assert response is not None
    assert response["error"]["code"] == -32602
    assert "text/event-stream" in response["error"]["message"]

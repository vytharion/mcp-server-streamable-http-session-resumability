from streamable_mcp.server import PROTOCOL_VERSION, MCPServer, ServerInfo


def _make_server() -> MCPServer:
    return MCPServer(info=ServerInfo(name="demo", version="0.1.0"))


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

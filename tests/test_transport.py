from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from streamable_mcp.server import PROTOCOL_VERSION, MCPServer, ServerInfo
from streamable_mcp.transport import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    server = MCPServer(info=ServerInfo(name="demo", version="0.1.0"))
    with TestClient(create_app(server)) as test_client:
        yield test_client


def test_post_initialize_returns_json_response(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        headers={"Accept": "application/json, text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["id"] == 1
    assert body["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert body["result"]["serverInfo"] == {"name": "demo", "version": "0.1.0"}


def test_post_notification_returns_202_with_empty_body(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )

    assert response.status_code == 202
    assert response.content == b""


def test_post_batch_with_only_notifications_returns_202(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ],
    )

    assert response.status_code == 202
    assert response.content == b""


def test_post_batch_with_requests_returns_array_of_replies(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "does/not/exist"},
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    by_id = {item["id"]: item for item in body}
    assert by_id[1]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert by_id[2]["error"]["code"] == -32601


def test_get_on_mcp_endpoint_is_method_not_allowed(client: TestClient) -> None:
    response = client.get("/mcp")

    assert response.status_code == 405


def test_invalid_json_body_returns_parse_error(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32700


def test_non_json_content_type_returns_415(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        content=b"{}",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 415
    body = response.json()
    assert body["error"]["code"] == -32700


def test_content_type_with_charset_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        content=b'{"jsonrpc":"2.0","id":9,"method":"initialize"}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == 9


def test_scalar_payload_is_rejected_as_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        content=b"42",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600

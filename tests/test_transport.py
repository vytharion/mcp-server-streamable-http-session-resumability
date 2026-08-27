from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from streamable_mcp.server import PROTOCOL_VERSION, MCPServer, ServerInfo
from streamable_mcp.sessions import SESSION_HEADER, InMemorySessionStore
from streamable_mcp.transport import create_app


@pytest.fixture()
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture()
def client(store: InMemorySessionStore) -> Iterator[TestClient]:
    server = MCPServer(info=ServerInfo(name="demo", version="0.1.0"))
    with TestClient(create_app(server, store=store)) as test_client:
        yield test_client


def _initialize(client: TestClient, request_id: int = 1) -> str:
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": request_id, "method": "initialize"},
    )
    assert response.status_code == 200
    session_id = response.headers[SESSION_HEADER]
    assert session_id
    return session_id


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


def test_initialize_issues_session_header(client: TestClient, store: InMemorySessionStore) -> None:
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
    )

    assert response.status_code == 200
    session_id = response.headers[SESSION_HEADER]
    assert session_id
    assert store.get(session_id) is not None


def test_two_initializes_produce_distinct_sessions(client: TestClient) -> None:
    first = _initialize(client, request_id=1)
    second = _initialize(client, request_id=2)

    assert first != second


def test_post_notification_returns_202_with_session_header(client: TestClient) -> None:
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={SESSION_HEADER: session_id},
    )

    assert response.status_code == 202
    assert response.content == b""


def test_post_without_session_header_after_initialize_is_rejected(client: TestClient) -> None:
    _initialize(client)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600
    assert SESSION_HEADER in body["error"]["message"]


def test_post_with_unknown_session_header_returns_404(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 5, "method": "does/not/exist"},
        headers={SESSION_HEADER: "totally-made-up"},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == -32600


def test_post_batch_with_only_notifications_returns_202(client: TestClient) -> None:
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ],
        headers={SESSION_HEADER: session_id},
    )

    assert response.status_code == 202
    assert response.content == b""


def test_post_batch_with_requests_returns_array_of_replies(client: TestClient) -> None:
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "id": 2, "method": "does/not/exist"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 3, "method": "does/not/exist"},
        ],
        headers={SESSION_HEADER: session_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    by_id = {item["id"]: item for item in body}
    assert by_id[2]["error"]["code"] == -32601
    assert by_id[3]["error"]["code"] == -32601


def test_batch_containing_initialize_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json=[
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ],
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600
    assert "initialize" in body["error"]["message"]


def test_get_on_mcp_endpoint_is_method_not_allowed(client: TestClient) -> None:
    response = client.get("/mcp")

    assert response.status_code == 405


def test_delete_terminates_session(client: TestClient, store: InMemorySessionStore) -> None:
    session_id = _initialize(client)
    assert store.get(session_id) is not None

    response = client.delete("/mcp", headers={SESSION_HEADER: session_id})

    assert response.status_code == 204
    assert store.get(session_id) is None


def test_delete_without_header_returns_400(client: TestClient) -> None:
    response = client.delete("/mcp")

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600


def test_delete_with_unknown_session_returns_404(client: TestClient) -> None:
    response = client.delete("/mcp", headers={SESSION_HEADER: "nope"})

    assert response.status_code == 404


def test_touch_updates_last_seen_on_authenticated_post() -> None:
    times = iter([1.0, 2.0, 3.0, 4.0])
    fixed_store = InMemorySessionStore(clock=lambda: next(times))
    server = MCPServer(info=ServerInfo(name="demo", version="0.1.0"))
    with TestClient(create_app(server, store=fixed_store)) as client:
        session_id = _initialize(client)
        created_at = fixed_store.get(session_id).created_at  # type: ignore[union-attr]

        client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={SESSION_HEADER: session_id},
        )

        session = fixed_store.get(session_id)
        assert session is not None
        assert session.last_seen_at > created_at


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
    assert response.headers[SESSION_HEADER]


def test_scalar_payload_is_rejected_as_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        content=b"42",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600

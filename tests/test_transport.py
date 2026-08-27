from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from streamable_mcp.server import PROTOCOL_VERSION, MCPServer, ServerInfo
from streamable_mcp.sessions import SESSION_HEADER, InMemorySessionStore
from streamable_mcp.streams import EventLogRegistry
from streamable_mcp.transport import LAST_EVENT_ID_HEADER, create_app


def _echo_tool(message: dict[str, Any]) -> Iterator[dict[str, Any]]:
    params = message.get("params") or {}
    text = (params.get("arguments") or {}).get("text", "")
    yield {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progress": 1, "total": 2},
    }
    yield {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progress": 2, "total": 2},
    }
    yield {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {"content": [{"type": "text", "text": text}]},
    }


@pytest.fixture()
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture()
def events() -> EventLogRegistry:
    return EventLogRegistry()


@pytest.fixture()
def server() -> MCPServer:
    instance = MCPServer(info=ServerInfo(name="demo", version="0.1.0"))
    instance.register_stream_tool("echo", _echo_tool)
    return instance


@pytest.fixture()
def client(
    server: MCPServer,
    store: InMemorySessionStore,
    events: EventLogRegistry,
) -> Iterator[TestClient]:
    with TestClient(create_app(server, store=store, events=events)) as test_client:
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


def test_get_without_session_header_returns_400(client: TestClient) -> None:
    response = client.get("/mcp", headers={"Accept": "text/event-stream"})

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600
    assert SESSION_HEADER in body["error"]["message"]


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


def _parse_sse_frames(raw: bytes) -> list[dict[str, str]]:
    text = raw.decode("utf-8")
    frames: list[dict[str, str]] = []
    for chunk in text.split("\n\n"):
        stripped = chunk.strip("\n")
        if not stripped:
            continue
        parsed: dict[str, str] = {}
        for line in stripped.splitlines():
            key, _, value = line.partition(": ")
            parsed[key] = value
        frames.append(parsed)
    return frames


def test_tools_call_streams_sse_with_monotonic_event_ids(
    client: TestClient,
    events: EventLogRegistry,
) -> None:
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        },
        headers={
            SESSION_HEADER: session_id,
            "Accept": "application/json, text/event-stream",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers[SESSION_HEADER] == session_id

    frames = _parse_sse_frames(response.content)
    assert [frame["id"] for frame in frames] == ["1", "2", "3"]

    log = events.get(session_id)
    assert log is not None
    assert len(log) == 3
    assert log.snapshot()[-1].data["id"] == 11
    assert log.snapshot()[-1].data["result"]["content"][0]["text"] == "hi"


def test_streaming_tool_call_without_sse_accept_returns_406(client: TestClient) -> None:
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        },
        headers={SESSION_HEADER: session_id, "Accept": "application/json"},
    )

    assert response.status_code == 406
    body = response.json()
    assert body["error"]["code"] == -32600
    assert "text/event-stream" in body["error"]["message"]


def test_streaming_tool_call_without_session_returns_400(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        },
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600
    assert SESSION_HEADER in body["error"]["message"]


def test_streaming_tool_call_with_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        },
        headers={SESSION_HEADER: "not-a-session", "Accept": "text/event-stream"},
    )

    assert response.status_code == 404


def test_unknown_tool_still_streams_error_frame(client: TestClient) -> None:
    session_id = _initialize(client)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "ghost"},
        },
        headers={SESSION_HEADER: session_id, "Accept": "application/json"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["code"] == -32601


def test_event_ids_stay_monotonic_across_two_streams(
    client: TestClient,
    events: EventLogRegistry,
) -> None:
    session_id = _initialize(client)

    for request_id in (21, 22):
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"text": "x"}},
            },
            headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
        )
        assert response.status_code == 200

    log = events.get(session_id)
    assert log is not None
    assert [event.id for event in log.snapshot()] == [1, 2, 3, 4, 5, 6]


def test_delete_evicts_event_log(
    client: TestClient,
    events: EventLogRegistry,
) -> None:
    session_id = _initialize(client)
    client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 33,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        },
        headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
    )
    assert events.get(session_id) is not None

    response = client.delete("/mcp", headers={SESSION_HEADER: session_id})

    assert response.status_code == 204
    assert events.get(session_id) is None


def _stream_tool_call(
    client: TestClient, session_id: str, request_id: int, text: str = "hi"
) -> None:
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": text}},
        },
        headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
    )
    assert response.status_code == 200


def test_get_with_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get(
        "/mcp",
        headers={SESSION_HEADER: "totally-made-up", "Accept": "text/event-stream"},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == -32600


def test_get_without_sse_accept_returns_406(client: TestClient) -> None:
    session_id = _initialize(client)

    response = client.get(
        "/mcp",
        headers={SESSION_HEADER: session_id, "Accept": "application/json"},
    )

    assert response.status_code == 406
    body = response.json()
    assert body["error"]["code"] == -32600
    assert "text/event-stream" in body["error"]["message"]


def test_get_without_last_event_id_replays_full_history(client: TestClient) -> None:
    session_id = _initialize(client)
    _stream_tool_call(client, session_id, request_id=100, text="first")
    _stream_tool_call(client, session_id, request_id=101, text="second")

    response = client.get(
        "/mcp",
        headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers[SESSION_HEADER] == session_id

    frames = _parse_sse_frames(response.content)
    assert [frame["id"] for frame in frames] == ["1", "2", "3", "4", "5", "6"]


def test_get_with_last_event_id_replays_only_events_after_cursor(
    client: TestClient,
) -> None:
    session_id = _initialize(client)
    _stream_tool_call(client, session_id, request_id=200, text="alpha")
    _stream_tool_call(client, session_id, request_id=201, text="beta")

    response = client.get(
        "/mcp",
        headers={
            SESSION_HEADER: session_id,
            "Accept": "text/event-stream",
            LAST_EVENT_ID_HEADER: "3",
        },
    )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    assert [frame["id"] for frame in frames] == ["4", "5", "6"]


def test_get_replays_final_result_payload_after_cursor(client: TestClient) -> None:
    session_id = _initialize(client)
    _stream_tool_call(client, session_id, request_id=42, text="resumed")

    response = client.get(
        "/mcp",
        headers={
            SESSION_HEADER: session_id,
            "Accept": "text/event-stream",
            LAST_EVENT_ID_HEADER: "2",
        },
    )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    assert [frame["id"] for frame in frames] == ["3"]
    payload = json.loads(frames[0]["data"])
    assert payload["id"] == 42
    assert payload["result"]["content"][0]["text"] == "resumed"


def test_get_with_cursor_at_end_returns_empty_stream(client: TestClient) -> None:
    session_id = _initialize(client)
    _stream_tool_call(client, session_id, request_id=300)

    response = client.get(
        "/mcp",
        headers={
            SESSION_HEADER: session_id,
            "Accept": "text/event-stream",
            LAST_EVENT_ID_HEADER: "3",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse_frames(response.content) == []


def test_get_with_zero_cursor_replays_full_history(client: TestClient) -> None:
    session_id = _initialize(client)
    _stream_tool_call(client, session_id, request_id=400)

    response = client.get(
        "/mcp",
        headers={
            SESSION_HEADER: session_id,
            "Accept": "text/event-stream",
            LAST_EVENT_ID_HEADER: "0",
        },
    )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    assert [frame["id"] for frame in frames] == ["1", "2", "3"]


def test_get_with_malformed_last_event_id_falls_back_to_full_replay(
    client: TestClient,
) -> None:
    session_id = _initialize(client)
    _stream_tool_call(client, session_id, request_id=500)

    response = client.get(
        "/mcp",
        headers={
            SESSION_HEADER: session_id,
            "Accept": "text/event-stream",
            LAST_EVENT_ID_HEADER: "not-a-number",
        },
    )

    assert response.status_code == 200
    frames = _parse_sse_frames(response.content)
    assert [frame["id"] for frame in frames] == ["1", "2", "3"]


def test_get_on_session_that_never_streamed_returns_empty_body(
    client: TestClient,
) -> None:
    session_id = _initialize(client)

    response = client.get(
        "/mcp",
        headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse_frames(response.content) == []


def test_get_does_not_mutate_event_log(
    client: TestClient,
    events: EventLogRegistry,
) -> None:
    session_id = _initialize(client)
    _stream_tool_call(client, session_id, request_id=600)

    before = [event.id for event in events.for_session(session_id).snapshot()]
    client.get(
        "/mcp",
        headers={SESSION_HEADER: session_id, "Accept": "text/event-stream"},
    )
    client.get(
        "/mcp",
        headers={
            SESSION_HEADER: session_id,
            "Accept": "text/event-stream",
            LAST_EVENT_ID_HEADER: "1",
        },
    )
    after = [event.id for event in events.for_session(session_id).snapshot()]

    assert before == after == [1, 2, 3]

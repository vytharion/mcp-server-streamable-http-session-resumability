from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from streamable_mcp.client import (
    ScriptedClient,
    SSEFrame,
    run_drop_and_resume,
)
from streamable_mcp.server import MCPServer, ServerInfo
from streamable_mcp.sessions import SESSION_HEADER, InMemorySessionStore
from streamable_mcp.streams import EventLogRegistry
from streamable_mcp.tools import make_slow_counter_tool
from streamable_mcp.transport import create_app


def _noop(_: float) -> None:
    return None


def _build_env() -> tuple[TestClient, EventLogRegistry, InMemorySessionStore]:
    server = MCPServer(info=ServerInfo(name="demo", version="0.1.0"))
    server.register_stream_tool("slow_counter", make_slow_counter_tool(sleep=_noop))
    store = InMemorySessionStore()
    events = EventLogRegistry()
    return TestClient(create_app(server, store=store, events=events)), events, store


@pytest.fixture()
def env() -> Iterator[tuple[TestClient, EventLogRegistry, InMemorySessionStore]]:
    test_client, events, store = _build_env()
    with test_client as opened:
        yield opened, events, store


def test_client_initialize_stores_session_id_from_header(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, store = env
    client = ScriptedClient(client_http)

    result = client.initialize()

    assert client.session_id is not None
    assert store.get(client.session_id) is not None
    assert result["result"]["serverInfo"] == {"name": "demo", "version": "0.1.0"}


def test_client_stream_tool_call_yields_every_frame_and_tracks_last_event_id(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, events, _store = env
    client = ScriptedClient(client_http)
    client.initialize()

    frames = list(
        client.stream_tool_call(
            "slow_counter",
            arguments={"steps": 3, "delay": 0},
            request_id=500,
        )
    )

    assert [frame.id for frame in frames] == [1, 2, 3, 4]
    assert [frame.progress for frame in frames[:3]] == [1, 2, 3]
    assert frames[-1].is_result
    assert frames[-1].data["id"] == 500
    assert client.last_event_id == 4
    assert len(events.for_session(client.session_id or "")) == 4


def test_client_stream_tool_call_forwards_progress_token(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)
    client.initialize()

    frames = list(
        client.stream_tool_call(
            "slow_counter",
            arguments={"steps": 2, "delay": 0},
            progress_token="run-77",
        )
    )

    assert frames[0].data["params"]["progressToken"] == "run-77"
    assert frames[1].data["params"]["progressToken"] == "run-77"


def test_client_resume_with_cursor_replays_only_frames_after_last_event_id(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)
    client.initialize()
    list(
        client.stream_tool_call(
            "slow_counter", arguments={"steps": 3, "delay": 0}, request_id=88
        )
    )
    assert client.last_event_id == 4

    client.set_last_event_id(2)

    replayed = list(client.resume())
    assert [frame.id for frame in replayed] == [3, 4]
    assert replayed[-1].data["id"] == 88
    assert client.last_event_id == 4


def test_client_resume_after_full_stream_returns_no_new_frames(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)
    client.initialize()
    list(
        client.stream_tool_call(
            "slow_counter", arguments={"steps": 2, "delay": 0}, request_id=1
        )
    )

    replayed = list(client.resume())

    assert replayed == []


def test_client_resume_without_cursor_replays_full_history(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)
    client.initialize()
    list(
        client.stream_tool_call(
            "slow_counter", arguments={"steps": 2, "delay": 0}, request_id=17
        )
    )

    client.set_last_event_id(0)
    replayed = list(client.resume())

    assert [frame.id for frame in replayed] == [1, 2, 3]


def test_client_requires_initialize_before_streaming(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)

    with pytest.raises(RuntimeError):
        list(client.stream_tool_call("slow_counter"))


def test_client_requires_initialize_before_resuming(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)

    with pytest.raises(RuntimeError):
        list(client.resume())


def test_sse_frame_progress_returns_none_when_data_is_not_progress_notification() -> None:
    frame = SSEFrame(id=1, data={"jsonrpc": "2.0", "id": 5, "result": {"content": []}})
    assert frame.progress is None
    assert frame.is_result is True
    assert frame.is_progress is False


def test_run_drop_and_resume_stitches_full_transcript_from_scripted_flow(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, events, _store = env
    client = ScriptedClient(client_http)

    transcript = run_drop_and_resume(
        client,
        "slow_counter",
        arguments={"steps": 5, "delay": 0},
        drop_after=2,
        request_id=42,
    )

    assert transcript.session_id == client.session_id
    assert len(transcript.before_drop) == 2
    assert [frame.id for frame in transcript.before_drop] == [1, 2]

    assert [frame.id for frame in transcript.after_drop] == [3, 4, 5, 6]
    final_frame = transcript.after_drop[-1]
    assert final_frame.is_result
    assert final_frame.data["id"] == 42
    assert final_frame.data["result"]["content"][0]["text"] == "counted to 5"

    assert transcript.event_ids == [1, 2, 3, 4, 5, 6]
    server_log = events.get(client.session_id or "")
    assert server_log is not None
    assert [event.id for event in server_log.snapshot()] == [1, 2, 3, 4, 5, 6]


def test_run_drop_and_resume_yields_no_after_drop_frames_when_dropping_past_end(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)

    transcript = run_drop_and_resume(
        client,
        "slow_counter",
        arguments={"steps": 2, "delay": 0},
        drop_after=99,
        request_id=1,
    )

    assert len(transcript.before_drop) == 3
    assert transcript.after_drop == []
    assert transcript.event_ids == [1, 2, 3]


def test_run_drop_and_resume_captures_initialize_result(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)

    transcript = run_drop_and_resume(
        client,
        "slow_counter",
        arguments={"steps": 3, "delay": 0},
        drop_after=1,
        request_id=9,
    )

    assert transcript.initialize["result"]["serverInfo"] == {
        "name": "demo",
        "version": "0.1.0",
    }
    assert transcript.session_id
    assert transcript.before_drop[0].id == 1
    assert transcript.after_drop[-1].data["id"] == 9


def test_scripted_client_last_event_id_is_monotonic_across_operations(
    env: tuple[TestClient, EventLogRegistry, InMemorySessionStore],
) -> None:
    client_http, _events, _store = env
    client = ScriptedClient(client_http)
    client.initialize()

    stream = client.stream_tool_call(
        "slow_counter", arguments={"steps": 4, "delay": 0}, request_id=3
    )
    seen: list[int] = []
    for frame in stream:
        assert frame.id is not None
        seen.append(frame.id)
        if len(seen) >= 2:
            break
    stream.close()

    assert client.last_event_id == 2

    tail = list(client.resume())
    assert [frame.id for frame in tail] == [3, 4, 5]
    assert client.last_event_id == 5

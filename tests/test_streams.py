from __future__ import annotations

import json

from streamable_mcp.streams import EventLogRegistry, SessionEventLog, SSEvent


def test_sseevent_encodes_id_and_data_line() -> None:
    event = SSEvent(id=7, data={"jsonrpc": "2.0", "method": "notifications/progress"})

    encoded = event.encode()

    assert encoded.startswith(b"id: 7\n")
    assert b"\ndata: " in encoded
    assert encoded.endswith(b"\n\n")


def test_sseevent_data_is_compact_json() -> None:
    event = SSEvent(id=1, data={"a": 1, "b": 2})

    encoded = event.encode().decode("utf-8")

    data_line = [line for line in encoded.splitlines() if line.startswith("data: ")][0]
    assert data_line == 'data: {"a":1,"b":2}'
    assert json.loads(data_line[len("data: "):]) == {"a": 1, "b": 2}


def test_session_event_log_assigns_monotonic_ids_starting_at_one() -> None:
    log = SessionEventLog()

    first = log.append({"n": 1})
    second = log.append({"n": 2})
    third = log.append({"n": 3})

    assert (first.id, second.id, third.id) == (1, 2, 3)
    assert log.next_id() == 4
    assert len(log) == 3


def test_session_event_log_since_returns_events_after_cursor() -> None:
    log = SessionEventLog()
    for value in range(1, 6):
        log.append({"n": value})

    after_two = log.since(2)

    assert [event.id for event in after_two] == [3, 4, 5]
    assert log.since(5) == []
    assert log.since(0) == log.snapshot()


def test_event_log_registry_creates_log_lazily_and_reuses_it() -> None:
    registry = EventLogRegistry()

    first = registry.for_session("sid-1")
    second = registry.for_session("sid-1")

    assert first is second
    assert len(registry) == 1


def test_event_log_registry_isolates_sessions() -> None:
    registry = EventLogRegistry()

    log_a = registry.for_session("sid-a")
    log_b = registry.for_session("sid-b")
    log_a.append({"who": "a"})

    assert len(log_a) == 1
    assert len(log_b) == 0
    assert len(registry) == 2


def test_event_log_registry_discard_removes_log() -> None:
    registry = EventLogRegistry()
    registry.for_session("sid-1").append({"n": 1})

    assert registry.discard("sid-1") is True
    assert registry.discard("sid-1") is False
    assert registry.get("sid-1") is None

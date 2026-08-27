from __future__ import annotations

from itertools import count

from streamable_mcp.sessions import InMemorySessionStore, SessionStore


def _fixed_id_factory() -> "type[_Counter]":
    return _Counter


class _Counter:
    _seq = count(1)

    def __call__(self) -> str:
        return f"sid-{next(self._seq)}"


def test_in_memory_store_satisfies_protocol() -> None:
    store = InMemorySessionStore()
    assert isinstance(store, SessionStore)


def test_create_returns_unique_ids() -> None:
    store = InMemorySessionStore()

    first = store.create()
    second = store.create()

    assert first.id != second.id
    assert store.get(first.id) is first
    assert store.get(second.id) is second


def test_create_uses_injected_id_factory_and_clock() -> None:
    ids = iter(["alpha", "beta"])
    times = iter([100.0, 200.0])
    store = InMemorySessionStore(id_factory=lambda: next(ids), clock=lambda: next(times))

    session = store.create()

    assert session.id == "alpha"
    assert session.created_at == 100.0
    assert session.last_seen_at == 100.0


def test_touch_updates_last_seen_at_and_returns_session() -> None:
    times = iter([10.0, 25.0])
    store = InMemorySessionStore(clock=lambda: next(times))
    session = store.create()

    touched = store.touch(session.id)

    assert touched is session
    assert session.created_at == 10.0
    assert session.last_seen_at == 25.0


def test_touch_returns_none_for_unknown_session() -> None:
    store = InMemorySessionStore()
    assert store.touch("does-not-exist") is None


def test_get_returns_none_for_unknown_session() -> None:
    store = InMemorySessionStore()
    assert store.get("does-not-exist") is None


def test_delete_removes_session_and_reports_success() -> None:
    store = InMemorySessionStore()
    session = store.create()

    assert store.delete(session.id) is True
    assert store.delete(session.id) is False
    assert store.get(session.id) is None

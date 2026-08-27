from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

SESSION_HEADER = "Mcp-Session-Id"


@dataclass
class Session:
    id: str
    created_at: float
    last_seen_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SessionStore(Protocol):
    def create(self) -> Session: ...

    def get(self, session_id: str) -> Session | None: ...

    def touch(self, session_id: str) -> Session | None: ...

    def delete(self, session_id: str) -> bool: ...


def _default_id_factory() -> str:
    return secrets.token_urlsafe(24)


class InMemorySessionStore:
    def __init__(
        self,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._id_factory: Callable[[], str] = id_factory or _default_id_factory
        self._clock: Callable[[], float] = clock or time.time

    def create(self) -> Session:
        session_id = self._id_factory()
        now = self._clock()
        session = Session(id=session_id, created_at=now, last_seen_at=now)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def touch(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.last_seen_at = self._clock()
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)

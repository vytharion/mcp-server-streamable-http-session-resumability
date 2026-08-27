from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SSEvent:
    id: int
    data: dict[str, Any]

    def encode(self) -> bytes:
        payload = json.dumps(self.data, separators=(",", ":"), ensure_ascii=False)
        frame = f"id: {self.id}\ndata: {payload}\n\n"
        return frame.encode("utf-8")


@dataclass
class SessionEventLog:
    _events: list[SSEvent] = field(default_factory=list)
    _next_id: int = 1

    def append(self, data: dict[str, Any]) -> SSEvent:
        event = SSEvent(id=self._next_id, data=data)
        self._events.append(event)
        self._next_id += 1
        return event

    def since(self, last_event_id: int) -> list[SSEvent]:
        return [event for event in self._events if event.id > last_event_id]

    def snapshot(self) -> list[SSEvent]:
        return list(self._events)

    def next_id(self) -> int:
        return self._next_id

    def __len__(self) -> int:
        return len(self._events)


class EventLogRegistry:
    def __init__(self) -> None:
        self._logs: dict[str, SessionEventLog] = {}

    def for_session(self, session_id: str) -> SessionEventLog:
        log = self._logs.get(session_id)
        if log is None:
            log = SessionEventLog()
            self._logs[session_id] = log
        return log

    def get(self, session_id: str) -> SessionEventLog | None:
        return self._logs.get(session_id)

    def discard(self, session_id: str) -> bool:
        return self._logs.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._logs)

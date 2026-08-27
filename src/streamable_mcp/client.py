from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from streamable_mcp.sessions import SESSION_HEADER
from streamable_mcp.transport import LAST_EVENT_ID_HEADER, MCP_PATH

DEFAULT_INITIALIZE_ID = 1
DEFAULT_TOOL_CALL_ID = 42


@dataclass(frozen=True)
class SSEFrame:
    id: int | None
    data: dict[str, Any]

    @property
    def is_progress(self) -> bool:
        return self.data.get("method") == "notifications/progress"

    @property
    def is_result(self) -> bool:
        return "result" in self.data

    @property
    def progress(self) -> int | None:
        params = self.data.get("params")
        if not isinstance(params, dict):
            return None
        value = params.get("progress")
        return value if isinstance(value, int) else None


@dataclass
class DropResumeTranscript:
    session_id: str
    initialize: dict[str, Any]
    before_drop: list[SSEFrame] = field(default_factory=list)
    after_drop: list[SSEFrame] = field(default_factory=list)

    @property
    def all_frames(self) -> list[SSEFrame]:
        return [*self.before_drop, *self.after_drop]

    @property
    def event_ids(self) -> list[int]:
        return [frame.id for frame in self.all_frames if frame.id is not None]


class HttpBackend(Protocol):
    def post(
        self,
        url: str,
        *,
        json: Any = ...,
        headers: dict[str, str] | None = ...,
    ) -> Any: ...

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Any = ...,
        headers: dict[str, str] | None = ...,
    ) -> Any: ...


class ScriptedClient:
    def __init__(self, http: HttpBackend, path: str = MCP_PATH) -> None:
        self._http = http
        self._path = path
        self._session_id: str | None = None
        self._last_event_id: int = 0

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def last_event_id(self) -> int:
        return self._last_event_id

    def set_last_event_id(self, value: int) -> None:
        self._last_event_id = value if value > 0 else 0

    def initialize(self, request_id: int = DEFAULT_INITIALIZE_ID) -> dict[str, Any]:
        response = self._http.post(
            self._path,
            json={"jsonrpc": "2.0", "id": request_id, "method": "initialize"},
        )
        response.raise_for_status()
        self._session_id = response.headers[SESSION_HEADER]
        return response.json()

    def stream_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        request_id: int = DEFAULT_TOOL_CALL_ID,
        progress_token: str | int | None = None,
    ) -> Iterator[SSEFrame]:
        session_id = self._require_session()
        payload = _tool_call_payload(request_id, tool_name, arguments or {}, progress_token)
        headers = {SESSION_HEADER: session_id, "Accept": "text/event-stream"}
        with self._http.stream("POST", self._path, json=payload, headers=headers) as response:
            response.raise_for_status()
            yield from self._read_frames(response)

    def resume(self) -> Iterator[SSEFrame]:
        session_id = self._require_session()
        headers = {SESSION_HEADER: session_id, "Accept": "text/event-stream"}
        if self._last_event_id > 0:
            headers[LAST_EVENT_ID_HEADER] = str(self._last_event_id)
        with self._http.stream("GET", self._path, headers=headers) as response:
            response.raise_for_status()
            yield from self._read_frames(response)

    def _read_frames(self, response: Any) -> Iterator[SSEFrame]:
        buffer = b""
        for chunk in response.iter_bytes():
            buffer += chunk
            while b"\n\n" in buffer:
                raw, buffer = buffer.split(b"\n\n", 1)
                frame = _parse_frame(raw)
                if frame is None:
                    continue
                self._record(frame)
                yield frame
        remainder = buffer.strip(b"\n")
        if not remainder:
            return
        frame = _parse_frame(remainder)
        if frame is None:
            return
        self._record(frame)
        yield frame

    def _record(self, frame: SSEFrame) -> None:
        if frame.id is not None and frame.id > self._last_event_id:
            self._last_event_id = frame.id

    def _require_session(self) -> str:
        if self._session_id is None:
            raise RuntimeError("call initialize() before opening streams")
        return self._session_id


def run_drop_and_resume(
    client: ScriptedClient,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    drop_after: int = 2,
    request_id: int = DEFAULT_TOOL_CALL_ID,
    progress_token: str | int | None = None,
) -> DropResumeTranscript:
    initialize_result = client.initialize()
    transcript = DropResumeTranscript(
        session_id=client.session_id or "",
        initialize=initialize_result,
    )
    _consume_until_drop(
        client, tool_name, arguments, drop_after, request_id, progress_token, transcript
    )
    transcript.after_drop.extend(client.resume())
    return transcript


def _consume_until_drop(
    client: ScriptedClient,
    tool_name: str,
    arguments: dict[str, Any] | None,
    drop_after: int,
    request_id: int,
    progress_token: str | int | None,
    transcript: DropResumeTranscript,
) -> None:
    stream = client.stream_tool_call(
        tool_name,
        arguments=arguments,
        request_id=request_id,
        progress_token=progress_token,
    )
    try:
        for frame in stream:
            transcript.before_drop.append(frame)
            if len(transcript.before_drop) >= drop_after:
                return
    finally:
        stream.close()


def _tool_call_payload(
    request_id: int,
    name: str,
    arguments: dict[str, Any],
    progress_token: str | int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"name": name, "arguments": arguments}
    if progress_token is not None:
        params["_meta"] = {"progressToken": progress_token}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": params,
    }


def _parse_frame(raw: bytes) -> SSEFrame | None:
    text = raw.decode("utf-8").strip("\n")
    if not text:
        return None
    frame_id: int | None = None
    data_lines: list[str] = []
    for line in text.splitlines():
        key, _, value = line.partition(": ")
        if key == "id":
            frame_id = _parse_id(value)
        elif key == "data":
            data_lines.append(value)
    if not data_lines:
        return None
    payload = json.loads("\n".join(data_lines))
    if not isinstance(payload, dict):
        return None
    return SSEFrame(id=frame_id, data=payload)


def _parse_id(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None



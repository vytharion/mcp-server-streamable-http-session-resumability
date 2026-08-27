from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from streamable_mcp.server import MCPServer
from streamable_mcp.sessions import (
    SESSION_HEADER,
    InMemorySessionStore,
    Session,
    SessionStore,
)

MCP_PATH = "/mcp"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600

INITIALIZE_METHOD = "initialize"


def create_app(
    server: MCPServer,
    store: SessionStore | None = None,
    path: str = MCP_PATH,
) -> Starlette:
    resolved_store: SessionStore = store if store is not None else InMemorySessionStore()

    async def endpoint(request: Request) -> Response:
        if request.method == "POST":
            return await _handle_post(server, resolved_store, request)
        return await _handle_delete(resolved_store, request)

    return Starlette(routes=[Route(path, endpoint, methods=["POST", "DELETE"])])


async def _handle_delete(store: SessionStore, request: Request) -> Response:
    session_id = request.headers.get(SESSION_HEADER)
    if not session_id:
        return _http_error(400, INVALID_REQUEST, f"missing {SESSION_HEADER} header")
    if not store.delete(session_id):
        return _http_error(404, INVALID_REQUEST, "unknown session")
    return Response(status_code=204)


async def _handle_post(server: MCPServer, store: SessionStore, request: Request) -> Response:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/json":
        return _http_error(415, PARSE_ERROR, "content-type must be application/json")

    raw = await request.body()
    payload = _try_parse(raw)
    if payload is _PARSE_FAILURE:
        return _http_error(400, PARSE_ERROR, "invalid JSON body")

    session_id = request.headers.get(SESSION_HEADER)

    if isinstance(payload, dict):
        return _dispatch_single(server, store, payload, session_id)
    if isinstance(payload, list):
        return _dispatch_batch(server, store, payload, session_id)
    return _http_error(400, INVALID_REQUEST, "payload must be an object or array")


def _dispatch_single(
    server: MCPServer,
    store: SessionStore,
    message: dict[str, Any],
    session_id: str | None,
) -> Response:
    if _is_initialize(message):
        session = store.create()
        reply = server.handle(message)
        return _reply_with_session(reply, session)

    validated = _resolve_session(store, session_id)
    if validated is None:
        return _missing_session_error(session_id)

    reply = server.handle(message)
    if reply is None:
        return Response(status_code=202)
    return JSONResponse(reply)


def _dispatch_batch(
    server: MCPServer,
    store: SessionStore,
    messages: list[Any],
    session_id: str | None,
) -> Response:
    if _batch_contains_initialize(messages):
        return _http_error(400, INVALID_REQUEST, "initialize must not be batched")

    validated = _resolve_session(store, session_id)
    if validated is None:
        return _missing_session_error(session_id)

    replies: list[dict[str, Any]] = []
    for message in messages:
        replies.extend(_reply_for_batch_item(server, message))
    if not replies:
        return Response(status_code=202)
    return JSONResponse(replies)


def _reply_for_batch_item(server: MCPServer, message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return [_error_envelope(None, INVALID_REQUEST, "batch item must be an object")]
    reply = server.handle(message)
    if reply is None:
        return []
    return [reply]


def _reply_with_session(reply: dict[str, Any] | None, session: Session) -> Response:
    headers = {SESSION_HEADER: session.id}
    if reply is None:
        return Response(status_code=202, headers=headers)
    return JSONResponse(reply, headers=headers)


def _resolve_session(store: SessionStore, session_id: str | None) -> Session | None:
    if not session_id:
        return None
    return store.touch(session_id)


def _missing_session_error(session_id: str | None) -> JSONResponse:
    if not session_id:
        return _http_error(400, INVALID_REQUEST, f"missing {SESSION_HEADER} header")
    return _http_error(404, INVALID_REQUEST, "unknown session")


def _is_initialize(message: dict[str, Any]) -> bool:
    return message.get("method") == INITIALIZE_METHOD


def _batch_contains_initialize(messages: list[Any]) -> bool:
    for message in messages:
        if isinstance(message, dict) and _is_initialize(message):
            return True
    return False


_PARSE_FAILURE: Any = object()


def _try_parse(raw: bytes) -> Any:
    try:
        return json.loads(raw or b"null")
    except json.JSONDecodeError:
        return _PARSE_FAILURE


def _http_error(status: int, code: int, message: str) -> JSONResponse:
    return JSONResponse(_error_envelope(None, code, message), status_code=status)


def _error_envelope(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }

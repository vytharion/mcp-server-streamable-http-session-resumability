from __future__ import annotations

import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from streamable_mcp.server import MCPServer

MCP_PATH = "/mcp"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600


def create_app(server: MCPServer, path: str = MCP_PATH) -> Starlette:
    async def endpoint(request: Request) -> Response:
        return await _handle_post(server, request)

    return Starlette(routes=[Route(path, endpoint, methods=["POST"])])


async def _handle_post(server: MCPServer, request: Request) -> Response:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != "application/json":
        return _http_error(415, PARSE_ERROR, "content-type must be application/json")

    raw = await request.body()
    payload = _try_parse(raw)
    if payload is _PARSE_FAILURE:
        return _http_error(400, PARSE_ERROR, "invalid JSON body")

    if isinstance(payload, list):
        return _dispatch_batch(server, payload)
    if isinstance(payload, dict):
        return _dispatch_single(server, payload)
    return _http_error(400, INVALID_REQUEST, "payload must be an object or array")


def _dispatch_single(server: MCPServer, message: dict[str, Any]) -> Response:
    reply = server.handle(message)
    if reply is None:
        return Response(status_code=202)
    return JSONResponse(reply)


def _dispatch_batch(server: MCPServer, messages: list[Any]) -> Response:
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

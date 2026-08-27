from __future__ import annotations

import time
from typing import Any, Callable, Iterator

from streamable_mcp.server import StreamingTool

DEFAULT_STEPS = 5
DEFAULT_DELAY_SECONDS = 0.2
MAX_STEPS = 100

SleepFn = Callable[[float], None]


def make_slow_counter_tool(sleep: SleepFn | None = None) -> StreamingTool:
    sleeper: SleepFn = sleep if sleep is not None else time.sleep

    def tool(message: dict[str, Any]) -> Iterator[dict[str, Any]]:
        request_id = message.get("id")
        params = message.get("params") or {}
        arguments = params.get("arguments") or {}
        steps = _coerce_steps(arguments.get("steps"))
        delay = _coerce_delay(arguments.get("delay"))
        progress_token = _extract_progress_token(params)

        for tick in range(1, steps + 1):
            if tick > 1:
                sleeper(delay)
            yield _progress_frame(progress_token, tick, steps)

        yield _final_frame(request_id, steps)

    return tool


def _progress_frame(
    progress_token: str | int | None, progress: int, total: int
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "progress": progress,
        "total": total,
        "message": f"tick {progress}/{total}",
    }
    if progress_token is not None:
        params["progressToken"] = progress_token
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": params,
    }


def _final_frame(request_id: Any, steps: int) -> dict[str, Any]:
    text = f"counted to {steps}"
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": text}]},
    }


def _extract_progress_token(params: dict[str, Any]) -> str | int | None:
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    token = meta.get("progressToken")
    if isinstance(token, (str, int)) and not isinstance(token, bool):
        return token
    return None


def _coerce_steps(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return DEFAULT_STEPS
    if raw < 1:
        return 1
    if raw > MAX_STEPS:
        return MAX_STEPS
    return raw


def _coerce_delay(raw: Any) -> float:
    if isinstance(raw, bool):
        return DEFAULT_DELAY_SECONDS
    if isinstance(raw, (int, float)) and raw >= 0:
        return float(raw)
    return DEFAULT_DELAY_SECONDS

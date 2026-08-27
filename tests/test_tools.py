from __future__ import annotations

from typing import Any

from streamable_mcp.tools import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_STEPS,
    MAX_STEPS,
    make_slow_counter_tool,
)


def _noop_sleep(_seconds: float) -> None:
    return None


def _make_message(**arguments: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {"name": "slow_counter", "arguments": arguments},
    }


def test_tool_yields_progress_frames_then_final_result() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    frames = list(tool(_make_message(steps=3)))

    assert len(frames) == 4
    assert all(frame["method"] == "notifications/progress" for frame in frames[:3])
    assert "method" not in frames[3]
    assert frames[3]["id"] == 42
    assert frames[3]["result"]["content"][0]["text"] == "counted to 3"


def test_progress_frames_report_monotonic_tick_and_matching_total() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    frames = list(tool(_make_message(steps=4)))

    params = [frame["params"] for frame in frames[:4]]
    assert [item["progress"] for item in params] == [1, 2, 3, 4]
    assert all(item["total"] == 4 for item in params)
    assert [item["message"] for item in params] == [
        "tick 1/4",
        "tick 2/4",
        "tick 3/4",
        "tick 4/4",
    ]


def test_progress_token_from_meta_is_echoed_on_every_notification() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "slow_counter",
            "arguments": {"steps": 2},
            "_meta": {"progressToken": "run-123"},
        },
    }

    frames = list(tool(message))

    assert frames[0]["params"]["progressToken"] == "run-123"
    assert frames[1]["params"]["progressToken"] == "run-123"
    assert "progressToken" not in frames[2].get("params", {})


def test_progress_token_omitted_when_meta_is_absent() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    frames = list(tool(_make_message(steps=1)))

    assert "progressToken" not in frames[0]["params"]


def test_integer_progress_token_is_preserved() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "slow_counter",
            "arguments": {"steps": 1},
            "_meta": {"progressToken": 77},
        },
    }

    frames = list(tool(message))

    assert frames[0]["params"]["progressToken"] == 77


def test_sleep_runs_between_ticks_not_before_first_or_after_last() -> None:
    calls: list[float] = []
    tool = make_slow_counter_tool(sleep=calls.append)

    list(tool(_make_message(steps=3, delay=0.05)))

    assert calls == [0.05, 0.05]


def test_sleep_uses_default_delay_when_argument_missing() -> None:
    calls: list[float] = []
    tool = make_slow_counter_tool(sleep=calls.append)

    list(tool(_make_message(steps=2)))

    assert calls == [DEFAULT_DELAY_SECONDS]


def test_default_steps_used_when_argument_missing() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    frames = list(tool(_make_message()))

    assert len(frames) == DEFAULT_STEPS + 1


def test_zero_or_negative_steps_clamps_to_one_tick() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    zero_frames = list(tool(_make_message(steps=0)))
    negative_frames = list(tool(_make_message(steps=-5)))

    assert len(zero_frames) == 2
    assert len(negative_frames) == 2
    assert zero_frames[0]["params"]["progress"] == 1
    assert zero_frames[1]["result"]["content"][0]["text"] == "counted to 1"


def test_steps_above_ceiling_are_clamped() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    frames = list(tool(_make_message(steps=MAX_STEPS + 50)))

    assert len(frames) == MAX_STEPS + 1
    assert frames[-1]["result"]["content"][0]["text"] == f"counted to {MAX_STEPS}"


def test_non_integer_steps_falls_back_to_default() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    frames = list(tool(_make_message(steps="lots")))

    assert len(frames) == DEFAULT_STEPS + 1


def test_boolean_steps_falls_back_to_default() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    frames = list(tool(_make_message(steps=True)))

    assert len(frames) == DEFAULT_STEPS + 1


def test_negative_delay_falls_back_to_default() -> None:
    calls: list[float] = []
    tool = make_slow_counter_tool(sleep=calls.append)

    list(tool(_make_message(steps=2, delay=-1)))

    assert calls == [DEFAULT_DELAY_SECONDS]


def test_generator_yields_lazily_so_readers_see_ticks_incrementally() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)

    generator = tool(_make_message(steps=3))
    first = next(generator)
    second = next(generator)

    assert first["params"]["progress"] == 1
    assert second["params"]["progress"] == 2


def test_sleep_pacing_is_interleaved_between_yields_not_batched_up_front() -> None:
    calls: list[float] = []
    tool = make_slow_counter_tool(sleep=calls.append)

    generator = tool(_make_message(steps=3, delay=0.1))
    next(generator)
    assert calls == []
    next(generator)
    assert calls == [0.1]
    next(generator)
    assert calls == [0.1, 0.1]


def test_missing_arguments_dict_uses_all_defaults() -> None:
    tool = make_slow_counter_tool(sleep=_noop_sleep)
    message = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "slow_counter"},
    }

    frames = list(tool(message))

    assert len(frames) == DEFAULT_STEPS + 1
    assert frames[-1]["id"] == 7


def test_defaults_can_be_used_without_injecting_sleep() -> None:
    tool = make_slow_counter_tool()

    generator = tool(_make_message(steps=1))
    first = next(generator)
    final = next(generator)

    assert first["params"]["progress"] == 1
    assert final["result"]["content"][0]["text"] == "counted to 1"

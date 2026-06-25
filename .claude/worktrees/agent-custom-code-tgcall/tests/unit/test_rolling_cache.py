"""Rolling incremental prompt cache tests.

Verifies the new request shape and cache breakpoint placement:

  [ static-system (cached, breakpoint 1) ]
  → tools (cached by provider)
  → [ conversation history (rolling breakpoints 2-3) ]
  → [ dynamic-context tail (uncached) ]

Checks:
A. Static system block contains NO dynamic flow-state (project_id / flow stats).
B. Dynamic context appears as the LAST message in the assembled request.
C. Anthropic path: static-system breakpoint + rolling history breakpoints.
   Dynamic tail carries NO cache_control.  Total breakpoints <= 4.
D. OpenAI path: same guarantees.
E. Empty history guard: no crash / no out-of-bounds on first turn.
"""

from __future__ import annotations

from agent.llm.anthropic import AnthropicClient
from agent.llm.base import Message, ToolUsePart
from agent.llm.openai import OpenAIClient
from agent.prompts.system import (
    STATIC_SYSTEM_BLOCK,
    build_dynamic_context_message,
    build_system_message,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_request(history: list[Message], state: dict | None = None) -> list[Message]:
    """Build the full request list: [system, *history, dynamic_tail]."""
    state = state or {"project_id": "proj-test"}
    return [build_system_message(state), *history, build_dynamic_context_message(state)]


def _count_cache_controls_anthropic(request: list[Message]) -> int:
    """Count total cache_control breakpoints in an Anthropic-converted request."""
    system_payload, out = AnthropicClient._to_anthropic_messages(request)
    count = 0
    # System blocks
    if isinstance(system_payload, list):
        for block in system_payload:
            if isinstance(block, dict) and "cache_control" in block:
                count += 1
    # Message content blocks
    for msg in out:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1
    return count


def _count_cache_controls_openai(request: list[Message]) -> int:
    """Count total cache_control breakpoints in an OpenAI-converted request."""
    oai = OpenAIClient._to_openai_messages(request)
    OpenAIClient._inject_cache_markers(oai)
    count = 0
    for msg in oai:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1
        # str content never carries cache_control directly
    return count


# ── A. Static system prefix contains no dynamic content ──────────────────────


def test_static_block_no_project_id():
    """STATIC_SYSTEM_BLOCK must not contain project_id or user_id."""
    assert "project_id:" not in STATIC_SYSTEM_BLOCK
    assert "user_id:" not in STATIC_SYSTEM_BLOCK


def test_static_block_no_flow_stats():
    """STATIC_SYSTEM_BLOCK must not contain per-session flow statistics."""
    assert "Joriy saqlangan flow:" not in STATIC_SYSTEM_BLOCK
    assert "saqlanmagan flow" not in STATIC_SYSTEM_BLOCK


def test_build_system_message_ignores_state():
    """build_system_message must return identical content regardless of state."""
    msg_no_state = build_system_message()
    msg_with_state = build_system_message({"project_id": "proj-abc", "user_id": "u-1"})
    assert msg_no_state.content[0].text == msg_with_state.content[0].text


# ── B. Dynamic context is the LAST message ────────────────────────────────────


def test_dynamic_context_is_last_in_request():
    history = [
        Message.user_text("hello"),
        Message.assistant_text("hi there"),
        Message.user_text("create a bot"),
    ]
    request = _make_request(history, {"project_id": "p-last-check"})
    assert request[-1].role == "user"
    assert "p-last-check" in request[-1].text
    # None of the preceding messages should contain the project_id value
    for m in request[:-1]:
        if m.role == "system":
            assert "p-last-check" not in m.text


def test_dynamic_context_last_with_tool_results():
    """Dynamic tail must remain last even when history ends with tool_result."""
    history = [
        Message.user_text("do X"),
        Message(role="assistant", content=[ToolUsePart(id="tc1", name="plan.set", arguments={})]),
        Message.tool_result(tool_use_id="tc1", content="done"),
    ]
    request = _make_request(history, {"project_id": "p-tool-check"})
    assert "p-tool-check" in request[-1].text
    assert request[-1].role == "user"


# ── C. Anthropic: system breakpoint + rolling history breakpoints ─────────────


def test_anthropic_system_block_has_cache_control():
    request = _make_request([Message.user_text("hi")], {"project_id": "p1"})
    system_payload, _ = AnthropicClient._to_anthropic_messages(request)
    assert isinstance(system_payload, list)
    assert system_payload[0].get("cache_control") == {"type": "ephemeral"}


def test_anthropic_dynamic_tail_has_no_cache_control():
    """The dynamic-context tail (out[-1]) must NOT carry cache_control."""
    history = [
        Message.user_text("turn 1"),
        Message.assistant_text("ok 1"),
        Message.user_text("turn 2"),
    ]
    request = _make_request(history, {"project_id": "p-tail"})
    _, out = AnthropicClient._to_anthropic_messages(request)
    tail = out[-1]
    for block in tail.get("content", []):
        assert "cache_control" not in block, (
            f"Dynamic tail must not have cache_control, but found it: {block}"
        )


def test_anthropic_rolling_breakpoints_on_last_history_message():
    """cache_control must be placed on out[-2] (last history message)."""
    history = [
        Message.user_text("step 1"),
        Message.assistant_text("done 1"),
        Message.user_text("step 2"),
    ]
    request = _make_request(history, {"project_id": "p-rolling"})
    _, out = AnthropicClient._to_anthropic_messages(request)
    # out[-1] = dynamic tail (not marked)
    # out[-2] = last history message (should be marked)
    last_hist = out[-2]
    last_hist_content = last_hist.get("content", [])
    assert any(
        isinstance(b, dict) and b.get("cache_control") == {"type": "ephemeral"}
        for b in last_hist_content
    ), f"Last history message (out[-2]) must have cache_control: {last_hist_content}"


def test_anthropic_rolling_two_breakpoints_with_sufficient_history():
    """With 3+ history messages, both out[-2] and out[-3] should be marked."""
    history = [
        Message.user_text("msg a"),
        Message.assistant_text("reply a"),
        Message.user_text("msg b"),
        Message.assistant_text("reply b"),
        Message.user_text("msg c"),
    ]
    request = _make_request(history, {"project_id": "p-two-bp"})
    _, out = AnthropicClient._to_anthropic_messages(request)

    def _has_cache(msg_dict: dict) -> bool:
        return any(
            isinstance(b, dict) and "cache_control" in b
            for b in msg_dict.get("content", [])
        )

    assert _has_cache(out[-2]), "out[-2] must have cache_control"
    assert _has_cache(out[-3]), "out[-3] must have cache_control"
    # Tail must NOT be marked
    assert not _has_cache(out[-1]), "dynamic tail (out[-1]) must NOT have cache_control"


def test_anthropic_total_breakpoints_within_budget():
    """Total cache_control breakpoints must not exceed 4."""
    history = [
        Message.user_text("a"),
        Message.assistant_text("b"),
        Message.user_text("c"),
        Message.assistant_text("d"),
        Message.user_text("e"),
    ]
    request = _make_request(history, {"project_id": "p-budget"})
    count = _count_cache_controls_anthropic(request)
    assert count <= 4, f"Too many cache_control breakpoints: {count} (max 4)"


def test_anthropic_empty_history_no_crash():
    """First turn (no history): no crash, still a valid request."""
    request = _make_request([], {"project_id": "p-first"})
    # Only system + dynamic tail
    system_payload, out = AnthropicClient._to_anthropic_messages(request)
    assert isinstance(system_payload, list)
    assert system_payload[0].get("cache_control") == {"type": "ephemeral"}
    # tail must not have cache_control
    if out:
        tail = out[-1]
        for block in tail.get("content", []):
            assert "cache_control" not in block


def test_anthropic_single_history_message_no_crash():
    """With only one history message, second rolling breakpoint index (-3) is out of range."""
    request = _make_request([Message.user_text("just one")], {"project_id": "p-one"})
    count = _count_cache_controls_anthropic(request)
    assert count <= 4


# ── D. OpenAI path: system breakpoint + rolling history breakpoints ───────────


def _build_oai(request: list[Message]) -> list[dict]:
    oai = OpenAIClient._to_openai_messages(request)
    OpenAIClient._inject_cache_markers(oai)
    return oai


def test_openai_system_block_has_cache_control():
    request = _make_request([Message.user_text("hi")], {"project_id": "p-oai"})
    oai = _build_oai(request)
    sys_msg = next(m for m in oai if m["role"] == "system")
    content = sys_msg["content"]
    assert isinstance(content, list)
    assert content[0].get("cache_control") == {"type": "ephemeral"}


def test_openai_dynamic_tail_has_no_cache_control():
    """Dynamic tail (last non-system message) must NOT carry cache_control."""
    history = [
        Message.user_text("turn a"),
        Message.assistant_text("reply a"),
        Message.user_text("turn b"),
    ]
    request = _make_request(history, {"project_id": "p-oai-tail"})
    oai = _build_oai(request)
    non_sys = [m for m in oai if m["role"] != "system"]
    tail = non_sys[-1]
    content = tail.get("content", "")
    if isinstance(content, list):
        for block in content:
            assert "cache_control" not in block, (
                f"OpenAI dynamic tail must not have cache_control: {block}"
            )


def test_openai_rolling_breakpoint_on_last_history_message():
    """non_sys[-2] (last history message) must carry cache_control."""
    history = [
        Message.user_text("s1"),
        Message.assistant_text("r1"),
        Message.user_text("s2"),
    ]
    request = _make_request(history, {"project_id": "p-oai-rolling"})
    oai = _build_oai(request)
    non_sys = [m for m in oai if m["role"] != "system"]
    last_hist = non_sys[-2]
    content = last_hist.get("content", "")
    if isinstance(content, list):
        assert any(
            isinstance(b, dict) and b.get("cache_control") == {"type": "ephemeral"}
            for b in content
        ), f"OpenAI: last history message must have cache_control: {content}"
    else:
        # string content was converted to block form by _oai_mark_message
        raise AssertionError(f"Expected list content after mark, got: {content!r}")


def test_openai_total_breakpoints_within_budget():
    """Total cache_control breakpoints must not exceed 4."""
    history = [
        Message.user_text("a"),
        Message.assistant_text("b"),
        Message.user_text("c"),
        Message.assistant_text("d"),
        Message.user_text("e"),
    ]
    request = _make_request(history, {"project_id": "p-oai-budget"})
    count = _count_cache_controls_openai(request)
    assert count <= 4, f"Too many OpenAI cache_control breakpoints: {count} (max 4)"


def test_openai_empty_history_no_crash():
    """First turn (no history): valid request, system marked, no crash."""
    request = _make_request([], {"project_id": "p-oai-first"})
    oai = _build_oai(request)
    sys_msg = next(m for m in oai if m["role"] == "system")
    content = sys_msg["content"]
    assert isinstance(content, list)
    assert content[0].get("cache_control") == {"type": "ephemeral"}

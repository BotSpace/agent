"""Anthropic Messages API message conversion — strict alternation + toolResult pairing.

Mirrors test_bedrock_message_conversion.py: both providers share the same
normalize_tool_turns() helper so the conversion guarantees are identical.

Additional cases cover rolling-cache breakpoints and the system-block format.
"""

from itertools import pairwise

from agent.llm.anthropic import AnthropicClient
from agent.llm.base import (
    Message,
    TextPart,
    ThinkingPart,
    ToolUsePart,
)


def _roles(out):
    return [m["role"] for m in out]


def _block_types(content):
    return [b["type"] for b in content]


# ── parity with bedrock cases ────────────────────────────────────────────────


def test_parallel_tool_results_merge_into_single_user_turn():
    """Assistant with 2 parallel toolUse → 2 separate tool messages + a trailing
    dynamic-context user message must collapse into ONE user turn with exactly
    2 tool_result blocks (matching ids) plus the context text, roles alternating,
    first message is user.
    """
    msgs = [
        Message.user_text("build me a flow"),
        Message(
            role="assistant",
            content=[
                TextPart(text="working on it"),
                ToolUsePart(id="tu1", name="flow.read", arguments={}),
                ToolUsePart(id="tu2", name="plugin.list", arguments={}),
            ],
        ),
        Message.tool_result(tool_use_id="tu1", content="flow ok"),
        Message.tool_result(tool_use_id="tu2", content="plugins ok"),
        # Rolling-cache trailing dynamic-context user message.
        Message.user_text("<context>current flow state</context>"),
    ]

    system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=False)

    assert system is None
    assert _roles(out) == ["user", "assistant", "user"]

    # Assistant turn keeps the text + both tool_use blocks.
    assistant_types = _block_types(out[1]["content"])
    assert assistant_types.count("tool_use") == 2

    # Merged user turn: 2 tool_results + 1 text.
    user_turn = out[2]["content"]
    result_ids = [b["tool_use_id"] for b in user_turn if b["type"] == "tool_result"]
    assert result_ids == ["tu1", "tu2"]
    assert sum(1 for b in user_turn if b["type"] == "text") == 1


def test_alternation_no_consecutive_user_turns():
    """No two consecutive user messages may survive conversion."""
    msgs = [
        Message.user_text("hi"),
        Message.user_text("more context"),
        Message.assistant_text("hello"),
    ]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=False)
    roles = _roles(out)
    assert roles == ["user", "assistant"]
    assert sum(1 for b in out[0]["content"] if b["type"] == "text") == 2
    for a, b in pairwise(roles):
        assert a != b


def test_orphan_tool_result_is_dropped():
    """A toolResult with no matching toolUse in the preceding assistant turn
    must be dropped.
    """
    msgs = [
        Message.user_text("go"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="tu1", name="flow.read", arguments={})],
        ),
        Message.tool_result(tool_use_id="tu1", content="ok"),
        Message.tool_result(tool_use_id="ghost", content="stale"),
    ]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=False)
    assert _roles(out) == ["user", "assistant", "user"]
    result_ids = [
        b["tool_use_id"] for b in out[2]["content"] if b["type"] == "tool_result"
    ]
    assert result_ids == ["tu1"]  # ghost dropped


def test_orphan_only_user_turn_is_removed_and_resequenced():
    """If a user turn contained ONLY an orphan toolResult, the now-empty turn is
    removed and surrounding assistant turns re-merge without breaking alternation.
    """
    msgs = [
        Message.user_text("start"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="tu1", name="x.do", arguments={})],
        ),
        Message.tool_result(tool_use_id="ghost", content="stale"),
        Message.assistant_text("continuing"),
    ]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=False)
    roles = _roles(out)
    assert roles == ["user", "assistant"]
    assert roles[0] == "user"
    for a, b in pairwise(roles):
        assert a != b
    # Merged assistant turn keeps the toolUse and the follow-up text.
    types = _block_types(out[1]["content"])
    assert "tool_use" in types and "text" in types


def test_system_blocks_preserved():
    msgs = [
        Message.system_text("you are an agent"),
        Message.user_text("hi"),
    ]
    system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=False)
    assert isinstance(system, list)
    assert system[0]["text"] == "you are an agent"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert _roles(out) == ["user"]


# ── Anthropic-specific cases ─────────────────────────────────────────────────


def test_rolling_cache_breakpoints_on_history():
    """Rolling-cache breakpoints are placed on out[-2] and out[-3] (history),
    never on out[-1] (the dynamic-context tail).
    """
    msgs = [
        Message.user_text("turn 1"),
        Message.assistant_text("reply 1"),
        Message.user_text("turn 2"),
        Message.assistant_text("reply 2"),
        # dynamic-context tail (out[-1]) — must NOT be marked.
        Message.user_text("<context>tail</context>"),
    ]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=True)

    # tail (out[-1]) must not have cache_control on its last block
    tail_last = out[-1]["content"][-1]
    assert "cache_control" not in tail_last

    # out[-2] (second "reply 2" assistant) must have cache_control
    hist_last = out[-2]["content"][-1]
    assert hist_last.get("cache_control") == {"type": "ephemeral"}

    # out[-3] ("turn 2" user) must have cache_control
    hist_prev = out[-3]["content"][-1]
    assert hist_prev.get("cache_control") == {"type": "ephemeral"}


def test_rolling_cache_not_placed_when_only_tail():
    """With only one message (the dynamic tail), no rolling breakpoints are
    placed (nothing to cache in history).
    """
    msgs = [Message.user_text("<context>tail only</context>")]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=True)
    assert len(out) == 1
    tail_last = out[-1]["content"][-1]
    assert "cache_control" not in tail_last


def test_tool_role_becomes_user_in_output():
    """tool-role messages must appear as role='user' in output."""
    msgs = [
        Message.user_text("hi"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="u1", name="x.do", arguments={"a": 1})],
        ),
        Message.tool_result(tool_use_id="u1", content="done"),
    ]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=False)
    assert _roles(out) == ["user", "assistant", "user"]
    assert any(b["type"] == "tool_result" for b in out[2]["content"])


def test_thinking_part_included():
    """ThinkingPart blocks are serialized as type='thinking'."""
    msgs = [
        Message.user_text("think"),
        Message(
            role="assistant",
            content=[
                ThinkingPart(text="internal reasoning"),
                TextPart(text="answer"),
            ],
        ),
        Message.user_text("<context>ctx</context>"),
    ]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=False)
    assert _roles(out) == ["user", "assistant", "user"]
    types = _block_types(out[1]["content"])
    assert "thinking" in types
    assert "text" in types


def test_parallel_tools_rolling_cache_intact():
    """Parallel tool round-trip with rolling cache enabled — breakpoints must
    still land on the correct history positions after normalization merges the
    tool-result turns.
    """
    msgs = [
        Message.user_text("go"),
        Message(
            role="assistant",
            content=[
                ToolUsePart(id="tu1", name="a.do", arguments={}),
                ToolUsePart(id="tu2", name="b.do", arguments={}),
            ],
        ),
        Message.tool_result(tool_use_id="tu1", content="r1"),
        Message.tool_result(tool_use_id="tu2", content="r2"),
        # dynamic-context tail
        Message.user_text("<context>state</context>"),
    ]
    _system, out = AnthropicClient._to_anthropic_messages(msgs, inject_rolling_cache=True)

    # After normalization: user, assistant, user(merged results+tail) → 3 turns
    assert _roles(out) == ["user", "assistant", "user"]

    # tail is out[-1]; only 2 history items → only out[-2] gets a breakpoint,
    # out[-3] would be IndexError (only 1 history turn) → gracefully skipped.
    tail_last = out[-1]["content"][-1]
    assert "cache_control" not in tail_last

    hist_last = out[-2]["content"][-1]
    assert hist_last.get("cache_control") == {"type": "ephemeral"}

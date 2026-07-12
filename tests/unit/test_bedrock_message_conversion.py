"""Bedrock Converse message conversion — strict alternation + toolResult pairing.

Converse rejects (ValidationException) when consecutive same-role messages
appear or when a user turn carries more toolResult blocks than the preceding
assistant turn had toolUse blocks. ``_to_bedrock_messages`` must merge adjacent
same-role messages and prune orphan toolResults so the emitted structure is
always valid.
"""

from itertools import pairwise

from agent.llm.base import (
    Message,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    normalize_tool_turns,
)
from agent.llm.bedrock import BedrockClient


def _roles(out):
    return [m["role"] for m in out]


def _block_types(content):
    types = []
    for b in content:
        if "text" in b:
            types.append("text")
        elif "toolUse" in b:
            types.append("toolUse")
        elif "toolResult" in b:
            types.append("toolResult")
    return types


def test_parallel_tool_results_merge_into_single_user_turn():
    """Assistant with 2 parallel toolUse → 2 separate tool messages + a trailing
    dynamic-context user message must collapse into ONE user turn with exactly
    2 toolResult blocks (matching ids) plus the context text, roles alternating,
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
        # Parallel tool calls → two separate `tool`-role messages.
        Message.tool_result(tool_use_id="tu1", content="flow ok"),
        Message.tool_result(tool_use_id="tu2", content="plugins ok"),
        # Rolling-cache trailing dynamic-context user message.
        Message.user_text("<context>current flow state</context>"),
    ]

    system, out = BedrockClient._to_bedrock_messages(msgs)

    assert system == []
    # user, assistant, user — strictly alternating, first is user.
    assert _roles(out) == ["user", "assistant", "user"]

    # Assistant turn keeps both toolUse blocks.
    assistant_types = _block_types(out[1]["content"])
    assert assistant_types.count("toolUse") == 2

    # The merged user turn holds exactly the 2 toolResults + the context text.
    user_turn = out[2]["content"]
    result_ids = [
        b["toolResult"]["toolUseId"] for b in user_turn if "toolResult" in b
    ]
    assert result_ids == ["tu1", "tu2"]  # matching ids, in order
    assert sum(1 for b in user_turn if "text" in b) == 1  # context text folded in


def test_alternation_no_consecutive_user_turns():
    """No two consecutive user messages may survive conversion."""
    msgs = [
        Message.user_text("hi"),
        Message.user_text("more context"),  # would be a 2nd consecutive user
        Message.assistant_text("hello"),
    ]
    _system, out = BedrockClient._to_bedrock_messages(msgs)
    roles = _roles(out)
    assert roles == ["user", "assistant"]
    # The two user texts merged into a single user turn with two text blocks.
    assert sum(1 for b in out[0]["content"] if "text" in b) == 2
    # No same-role adjacency anywhere.
    for a, b in pairwise(roles):
        assert a != b


def test_orphan_tool_result_is_dropped():
    """A toolResult with no matching toolUse in the preceding assistant turn
    (e.g. left over after an ask_user/flow.clear interrupt) must be dropped so
    the user turn's toolResult count never exceeds the prior toolUse count.
    """
    msgs = [
        Message.user_text("go"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="tu1", name="flow.read", arguments={})],
        ),
        # Valid result for tu1 + an ORPHAN result for an id that was never used.
        Message.tool_result(tool_use_id="tu1", content="ok"),
        Message.tool_result(tool_use_id="ghost", content="stale"),
    ]
    _system, out = BedrockClient._to_bedrock_messages(msgs)
    assert _roles(out) == ["user", "assistant", "user"]
    result_ids = [
        b["toolResult"]["toolUseId"]
        for b in out[2]["content"]
        if "toolResult" in b
    ]
    assert result_ids == ["tu1"]  # ghost dropped


def test_orphan_only_user_turn_is_removed_and_resequenced():
    """If a user turn contained ONLY an orphan toolResult, the now-empty turn is
    removed and surrounding assistant turns re-merge without breaking
    alternation or leading-user invariants.
    """
    msgs = [
        Message.user_text("start"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="tu1", name="x.do", arguments={})],
        ),
        # Orphan-only result turn (no matching toolUse → whole turn drops out).
        Message.tool_result(tool_use_id="ghost", content="stale"),
        Message.assistant_text("continuing"),
    ]
    _system, out = BedrockClient._to_bedrock_messages(msgs)
    roles = _roles(out)
    # The empty orphan user turn is gone; the two assistant turns merge.
    assert roles == ["user", "assistant"]
    assert roles[0] == "user"
    for a, b in pairwise(roles):
        assert a != b
    # Merged assistant turn keeps the toolUse and the follow-up text.
    types = _block_types(out[1]["content"])
    assert "toolUse" in types and "text" in types


def test_duplicate_tool_use_id_deduped_keep_last():
    """The interrupt/resume flow can append a SECOND toolResult for a
    tool_call_id that already has one (an "awaiting approval"/placeholder result
    followed by the post-resume real result). After merge into one user turn
    that is two ToolResultParts sharing an id → Converse rejects with
    "contain duplicate Ids". normalize_tool_turns must keep only ONE — the LAST
    (the resolved/final result).
    """
    msgs = [
        Message.user_text("clarify please"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="tu1", name="ask_user", arguments={})],
        ),
        # placeholder (pre-resume) then real (post-resume) — same id, same turn.
        Message.tool_result(tool_use_id="tu1", content="Awaiting user input..."),
        Message.tool_result(tool_use_id="tu1", content="User clarification: yes"),
    ]
    out = normalize_tool_turns(msgs)
    # Exactly one ToolResultPart for tu1, and it is the LAST (resolved) one.
    results = [
        p
        for m in out
        for p in m.content
        if isinstance(p, ToolResultPart) and p.tool_use_id == "tu1"
    ]
    assert len(results) == 1
    assert results[0].content == "User clarification: yes"


def test_duplicate_tool_use_id_converse_no_validation_error():
    """End-to-end through _to_bedrock_messages: a duplicate-id toolResult must
    collapse to a single Converse toolResult block for that id."""
    msgs = [
        Message.user_text("clarify please"),
        Message(
            role="assistant",
            content=[ToolUsePart(id="tu1", name="ask_user", arguments={})],
        ),
        Message.tool_result(tool_use_id="tu1", content="Awaiting user input..."),
        Message.tool_result(tool_use_id="tu1", content="User clarification: yes"),
    ]
    _system, out = BedrockClient._to_bedrock_messages(msgs)
    assert _roles(out) == ["user", "assistant", "user"]
    result_ids = [
        b["toolResult"]["toolUseId"]
        for b in out[2]["content"]
        if "toolResult" in b
    ]
    assert result_ids == ["tu1"]  # no duplicate id


def test_system_blocks_preserved():
    msgs = [
        Message.system_text("you are an agent"),
        Message.user_text("hi"),
    ]
    system, out = BedrockClient._to_bedrock_messages(msgs)
    assert system == [{"text": "you are an agent"}]
    assert _roles(out) == ["user"]


def test_empty_text_blocks_dropped():
    """Blank text parts must not become empty Converse content blocks."""
    msgs = [
        Message(role="user", content=[TextPart(text=""), TextPart(text="real")]),
    ]
    _system, out = BedrockClient._to_bedrock_messages(msgs)
    assert sum(1 for b in out[0]["content"] if "text" in b) == 1


# ── prompt caching (cachePoint) ────────────────────────────────────────────────
from agent.llm.base import GenerateOptions, ToolSpec


def _has_cache_point(blocks):
    return any(isinstance(b, dict) and "cachePoint" in b for b in blocks)


def _opts(model: str):
    return GenerateOptions(
        model=model,
        max_tokens=128,
        temperature=0.4,
        tools=[ToolSpec(name="flow.add", description="d", input_schema={"type": "object"})],
    )


def test_claude_model_gets_cache_points_on_system_tools_and_last_message():
    client = BedrockClient(region="us-west-2")
    msgs = [Message.system_text("SYSTEM CATALOG"), Message.user_text("salom")]
    req = client._build_request(msgs, _opts("claude-sonnet-4-6"))

    assert _has_cache_point(req["system"]), "system must carry a cachePoint"
    assert _has_cache_point(req["toolConfig"]["tools"]), "tools must carry a cachePoint"
    assert _has_cache_point(req["messages"][-1]["content"]), "rolling cachePoint on last msg"


def test_non_claude_model_gets_no_cache_points():
    # Qwen / Llama / etc. reject cachePoint — must NOT be inserted.
    client = BedrockClient(region="us-west-2")
    msgs = [Message.system_text("SYSTEM CATALOG"), Message.user_text("salom")]
    req = client._build_request(msgs, _opts("qwen.qwen3-coder-480b-a35b-v1:0"))

    assert not _has_cache_point(req["system"])
    assert not _has_cache_point(req["toolConfig"]["tools"])
    assert not _has_cache_point(req["messages"][-1]["content"])

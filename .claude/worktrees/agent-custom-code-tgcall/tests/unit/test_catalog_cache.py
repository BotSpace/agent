"""Catalog-in-system-prompt cache correctness tests.

Verifies:
1. render_compact_catalog() is deterministic and contains all node types.
2. STATIC_SYSTEM_BLOCK does NOT contain per-type field-schema dumps.
3. build_system_message() returns a SINGLE-BLOCK static-only system message.
   cache_control is placed on that block in both the Anthropic and OpenAI paths.
4. The dynamic flow-state context appears as a TRAILING USER message (not in system).
5. Token size estimate for the static block is in the expected range.
"""

from __future__ import annotations

from agent.flow.catalog import NODE_TYPES, render_compact_catalog
from agent.llm.anthropic import AnthropicClient
from agent.llm.base import Message, TextPart
from agent.llm.openai import OpenAIClient
from agent.prompts.system import (
    STATIC_SYSTEM_BLOCK,
    build_dynamic_block,
    build_dynamic_context_message,
    build_system_message,
)

# ── 1. render_compact_catalog determinism & completeness ─────────────────────


def test_compact_catalog_is_deterministic():
    a = render_compact_catalog()
    b = render_compact_catalog()
    assert a == b, "render_compact_catalog() must be byte-identical across calls"


def test_compact_catalog_contains_all_node_types():
    catalog = render_compact_catalog()
    for type_name in NODE_TYPES:
        assert type_name in catalog, f"Node type '{type_name}' missing from compact catalog"


def test_compact_catalog_no_field_schema_dumps():
    """The compact catalog must NOT contain detailed field-schema content
    such as 'input_schema', 'properties', 'additionalProperties', or
    JSON-schema keywords — those stay in flow.list_node_types."""
    catalog = render_compact_catalog()
    forbidden = ["input_schema", "additionalProperties", '"properties"', "validators"]
    for kw in forbidden:
        assert kw not in catalog, f"Compact catalog should not contain field-schema keyword '{kw}'"


def test_compact_catalog_token_estimate():
    """Rough token estimate (chars / 3.5) should be between 2K and 12K tokens.

    The compact catalog intentionally omits per-type field schemas, so it is
    smaller than the full catalog but still covers all node types with their
    descriptions.  The lower bound ensures meaningful content is present.
    """
    catalog = render_compact_catalog()
    estimated_tokens = len(catalog) / 3.5
    assert estimated_tokens >= 2_000, f"Catalog too small ({estimated_tokens:.0f} tokens)"
    assert estimated_tokens <= 12_000, f"Catalog too large ({estimated_tokens:.0f} tokens)"


# ── 2. STATIC_SYSTEM_BLOCK content ───────────────────────────────────────────


def test_static_block_contains_core_identity():
    assert "dispatcher" in STATIC_SYSTEM_BLOCK.lower()
    assert "flow-builder" in STATIC_SYSTEM_BLOCK


def test_static_block_contains_catalog():
    for type_name in NODE_TYPES:
        assert type_name in STATIC_SYSTEM_BLOCK, (
            f"STATIC_SYSTEM_BLOCK missing node type '{type_name}'"
        )


def test_static_block_no_dynamic_content():
    """The static block must not contain session-level placeholders."""
    assert "project_id:" not in STATIC_SYSTEM_BLOCK
    assert "user_id:" not in STATIC_SYSTEM_BLOCK
    assert "Joriy saqlangan flow:" not in STATIC_SYSTEM_BLOCK
    assert "saqlanmagan flow" not in STATIC_SYSTEM_BLOCK


# ── 3a. build_system_message → single static block ───────────────────────────


def test_build_system_message_is_single_static_block():
    """build_system_message must return a SINGLE-BLOCK static-only system message.

    The dynamic context has been moved to a trailing user message.
    Passing state must not leak into the system message.
    """
    msg = build_system_message({"project_id": "proj-1", "user_id": "u-99"})
    assert msg.role == "system"
    assert len(msg.content) == 1, "system message must have exactly 1 block (static only)"
    assert isinstance(msg.content[0], TextPart)
    assert "proj-1" not in msg.content[0].text, "project_id must NOT appear in the static system block"
    assert "u-99" not in msg.content[0].text, "user_id must NOT appear in the static system block"


def test_build_dynamic_context_message_is_trailing_user():
    """build_dynamic_context_message must return a user-role message with session context."""
    dyn = build_dynamic_context_message({"project_id": "proj-trail", "user_id": "u-77"})
    assert dyn.role == "user", "dynamic context must be a user-role message"
    assert "proj-trail" in dyn.text, "project_id must appear in the dynamic context message"


def test_dynamic_context_is_last_message_in_request():
    """The dynamic-context message must be the LAST message in the assembled request."""
    state = {"project_id": "p-order-check", "user_id": "u-1"}
    system = build_system_message(state)
    history = [Message.user_text("hello"), Message.assistant_text("hi")]
    dynamic_ctx = build_dynamic_context_message(state)
    request = [system, *history, dynamic_ctx]
    assert request[-1] is dynamic_ctx, "dynamic_ctx must be the last message"
    assert "p-order-check" in request[-1].text
    # dynamic context must NOT appear in the static system prefix
    assert "p-order-check" not in request[0].text


# ── 3b. Anthropic path: system cache_control + no dynamic in system ───────────


def test_anthropic_cache_control_on_static_block():
    """Anthropic _to_anthropic_messages must place cache_control on the static system block."""
    state = {"project_id": "proj-1", "user_id": "u-99"}
    system_msg = build_system_message(state)
    dynamic_msg = build_dynamic_context_message(state)
    # Assemble full request: [system, history_user, dynamic_tail]
    history_user = Message.user_text("hello")
    request = [system_msg, history_user, dynamic_msg]
    system_payload, _out = AnthropicClient._to_anthropic_messages(request)

    # System payload: single cached block (list with 1 dict)
    assert isinstance(system_payload, list), "Expected list[dict] for system with cache_control"
    assert len(system_payload) == 1
    block0 = system_payload[0]
    assert block0.get("cache_control") == {"type": "ephemeral"}, (
        "cache_control must be on the static system block"
    )
    # Dynamic context must NOT appear in the system payload
    assert "proj-1" not in block0.get("text", ""), (
        "project_id must NOT appear in the cached system block"
    )


def test_anthropic_dynamic_block_is_last_message_not_in_system():
    """Dynamic context must appear in out[-1] (last user message), not in system."""
    state = {"project_id": "proj-xyz", "user_id": "u-42"}
    system_msg = build_system_message(state)
    history_user = Message.user_text("what can you do?")
    dynamic_msg = build_dynamic_context_message(state)
    request = [system_msg, history_user, dynamic_msg]
    system_payload, out = AnthropicClient._to_anthropic_messages(request)

    # system does NOT contain project_id
    system_text = system_payload[0]["text"] if isinstance(system_payload, list) else (system_payload or "")
    assert "proj-xyz" not in system_text, "project_id must NOT appear in the static cached block"

    # last message in out contains project_id
    assert out, "out must not be empty"
    last_out = out[-1]
    assert last_out["role"] == "user"
    last_text = " ".join(
        b.get("text", "") for b in last_out["content"] if isinstance(b, dict)
    )
    assert "proj-xyz" in last_text, "project_id must appear in the last (dynamic tail) message"


def test_anthropic_flow_dirty_in_dynamic_tail_not_system():
    state = {"project_id": "p1", "flow_dirty": True, "baseline_flow": {"nodes": [{"id": "x"}], "edges": []}}
    system_msg = build_system_message(state)
    dynamic_msg = build_dynamic_context_message(state)
    request = [system_msg, Message.user_text("fix"), dynamic_msg]
    system_payload, out = AnthropicClient._to_anthropic_messages(request)

    system_text = system_payload[0]["text"] if isinstance(system_payload, list) else (system_payload or "")
    assert "saqlanmagan" not in system_text, "dirty-flag text must NOT be in static cached system block"

    last_text = " ".join(
        b.get("text", "") for b in out[-1]["content"] if isinstance(b, dict)
    )
    assert "saqlanmagan" in last_text, "dirty-flag text must be in the dynamic tail message"


# ── 3c. OpenAI/OpenRouter path: system cache_control ────────────────────────


def _build_openai_full_request(state: dict, history: list | None = None) -> list[dict]:
    """Build a full OpenAI-format request with system + history + dynamic tail."""
    system_msg = build_system_message(state)
    dynamic_msg = build_dynamic_context_message(state)
    msgs = [system_msg] + (history or []) + [dynamic_msg]
    oai_messages = OpenAIClient._to_openai_messages(msgs)
    OpenAIClient._inject_cache_markers(oai_messages)
    return oai_messages


def _get_system_blocks(oai_messages: list[dict]) -> list[dict]:
    sys_msg = next(m for m in oai_messages if m["role"] == "system")
    content = sys_msg["content"]
    assert isinstance(content, list)
    return content


def test_openai_cache_control_on_static_block():
    oai_messages = _build_openai_full_request({"project_id": "proj-1"})
    blocks = _get_system_blocks(oai_messages)
    # Single block (static only)
    assert len(blocks) == 1, f"Expected 1 system block (static only), got {len(blocks)}"
    assert blocks[0].get("cache_control") == {"type": "ephemeral"}, (
        "OpenAI path: cache_control must be on the static system block"
    )


def test_openai_dynamic_block_is_last_message_not_in_system():
    oai_messages = _build_openai_full_request({"project_id": "proj-abc"})
    blocks = _get_system_blocks(oai_messages)
    # project_id must NOT be in system block
    assert "proj-abc" not in blocks[0].get("text", ""), (
        "project_id must NOT appear in the static cached block (OpenAI path)"
    )
    # project_id must appear in the last (dynamic tail) non-system message
    non_sys = [m for m in oai_messages if m["role"] != "system"]
    assert non_sys, "must have at least 1 non-system message"
    last = non_sys[-1]
    content = last.get("content", "")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    assert "proj-abc" in str(content), (
        "project_id must appear in the last (dynamic tail) message (OpenAI path)"
    )


# ── 4. build_dynamic_block ───────────────────────────────────────────────────


def test_dynamic_block_flow_stats():
    state = {
        "project_id": "p1",
        "baseline_flow": {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": [{"id": "e1"}]},
    }
    dyn = build_dynamic_block(state)
    assert "2 node" in dyn
    assert "1 edge" in dyn


def test_dynamic_block_no_catalog_types():
    """Dynamic block should not contain node-type catalog entries."""
    dyn = build_dynamic_block({"project_id": "p2"})
    # A few catalog types that would only appear in the catalog, not in dynamic context
    assert "CommandTriggerNode" not in dyn
    assert "SendTextMessageNode" not in dyn


# ── 5. Single-block system path returns a cached block ───────────────────────


def test_anthropic_single_block_returns_cached_list():
    """When a plain system_text message is used, _to_anthropic_messages
    should return a list[dict] with cache_control on the single block.

    This covers the new static-only system message shape from build_system_message().
    """
    msg = Message.system_text("plain system text")
    system_payload, _ = AnthropicClient._to_anthropic_messages([msg])
    assert isinstance(system_payload, list), "Single-block system must return list[dict]"
    assert len(system_payload) == 1
    assert system_payload[0]["text"] == "plain system text"
    assert system_payload[0].get("cache_control") == {"type": "ephemeral"}, (
        "Single-block system must carry cache_control"
    )

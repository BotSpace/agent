"""Catalog-in-system-prompt cache correctness tests.

Verifies:
1. render_compact_catalog() is deterministic and contains all node types.
2. STATIC_SYSTEM_BLOCK does NOT contain per-type field-schema dumps.
3. build_system_message() places cache_control on the STATIC block (index 0)
   in both the Anthropic and OpenAI (OpenRouter) client paths.
4. The dynamic flow-state text is NOT inside the cached block.
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


# ── 3a. build_system_message → Anthropic path cache_control placement ────────


def test_anthropic_cache_control_on_static_block():
    """Anthropic _to_anthropic_messages must place cache_control on block 0."""
    msg = build_system_message({"project_id": "proj-1", "user_id": "u-99"})
    # Confirm two-block message
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextPart)
    assert isinstance(msg.content[1], TextPart)

    system_payload, _ = AnthropicClient._to_anthropic_messages([msg])

    assert isinstance(system_payload, list), "Expected list[dict] for two-block system"
    assert len(system_payload) == 2

    block0 = system_payload[0]
    block1 = system_payload[1]

    # Cache breakpoint on static block
    assert block0.get("cache_control") == {"type": "ephemeral"}, (
        "cache_control must be on block 0 (static)"
    )
    # Dynamic block must NOT carry cache_control
    assert "cache_control" not in block1, (
        "cache_control must NOT be on block 1 (dynamic)"
    )


def test_anthropic_dynamic_block_contains_project_id():
    """Dynamic block text must include session context, not static block."""
    msg = build_system_message({"project_id": "proj-xyz", "user_id": "u-42"})
    system_payload, _ = AnthropicClient._to_anthropic_messages([msg])

    assert isinstance(system_payload, list)
    block0_text = system_payload[0]["text"]
    block1_text = system_payload[1]["text"]

    assert "proj-xyz" not in block0_text, "project_id must NOT appear in the static cached block"
    assert "proj-xyz" in block1_text, "project_id must appear in the dynamic (uncached) block"


def test_anthropic_flow_dirty_in_dynamic_block_only():
    state = {"project_id": "p1", "flow_dirty": True, "baseline_flow": {"nodes": [{"id": "x"}], "edges": []}}
    msg = build_system_message(state)
    system_payload, _ = AnthropicClient._to_anthropic_messages([msg])

    assert isinstance(system_payload, list)
    block0_text = system_payload[0]["text"]
    block1_text = system_payload[1]["text"]

    assert "saqlanmagan" not in block0_text, "dirty-flag text must NOT be in static cached block"
    assert "saqlanmagan" in block1_text, "dirty-flag text must be in dynamic block"


# ── 3b. build_system_message → OpenAI/OpenRouter path cache_control placement ─


def _build_openai_system_blocks(state: dict) -> list[dict]:
    """Helper: apply _inject_cache_markers and return the system message content."""
    msg = build_system_message(state)
    oai_messages = OpenAIClient._to_openai_messages([msg])
    OpenAIClient._inject_cache_markers(oai_messages)
    sys_blocks = next(m for m in oai_messages if m["role"] == "system")
    content = sys_blocks["content"]
    assert isinstance(content, list)
    return content


def test_openai_cache_control_on_static_block():
    blocks = _build_openai_system_blocks({"project_id": "proj-1"})
    # Two blocks expected
    assert len(blocks) == 2, f"Expected 2 system blocks, got {len(blocks)}"

    block0 = blocks[0]
    block1 = blocks[1]

    assert block0.get("cache_control") == {"type": "ephemeral"}, (
        "OpenAI path: cache_control must be on block 0 (static)"
    )
    assert "cache_control" not in block1, (
        "OpenAI path: cache_control must NOT be on block 1 (dynamic)"
    )


def test_openai_dynamic_block_contains_project_id():
    blocks = _build_openai_system_blocks({"project_id": "proj-abc"})
    assert "proj-abc" not in blocks[0]["text"], (
        "project_id must NOT appear in the static cached block (OpenAI path)"
    )
    assert "proj-abc" in blocks[1]["text"], (
        "project_id must appear in the dynamic block (OpenAI path)"
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


# ── 5. Single-block legacy path (system_text) still works ─────────────────────


def test_anthropic_single_block_returns_plain_string():
    """When a plain system_text message is used, _to_anthropic_messages
    should return a plain str (backward-compatible, no cache_control)."""
    msg = Message.system_text("plain system text")
    system_payload, _ = AnthropicClient._to_anthropic_messages([msg])
    assert isinstance(system_payload, str)
    assert system_payload == "plain system text"

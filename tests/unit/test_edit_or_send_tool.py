"""Typed flow.add_* tools were removed (lazy schema loading) — the node types
they covered now go through the generic flow.add_node."""

from agent.tools.flow import build_flow_tools


def test_typed_message_add_tools_removed():
    tools = {t.name: t for t in build_flow_tools()}
    assert "flow.add_node" in tools
    assert "flow.add_edit_or_send" not in tools
    assert "flow.add_send_text_message" not in tools

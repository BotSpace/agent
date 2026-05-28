"""Tests for state-aware tool scope picking in the graph builder."""
from agent.graph.builder import _FLOW_LIGHT_TOOLS, _is_flow_active, _pick_tool_scope
from agent.llm.base import Message, ToolUsePart
from agent.tools import build_default_registry


def _all_names() -> list[str]:
    return [t.name for t in build_default_registry().all()]


def test_chat_state_drops_heavy_flow_tools():
    """No flow context, no plan, no tool history → heavy flow.* tools removed."""
    state = {"messages": []}
    scope = _pick_tool_scope(state, _all_names())
    assert scope is not None
    # Read-only flow tools stay
    for name in _FLOW_LIGHT_TOOLS:
        assert name in scope, f"{name} must remain in chat scope"
    # Heavy flow tools dropped
    for heavy in ("flow.add_send_text_message", "flow.add_command_trigger", "flow.connect_button"):
        assert heavy not in scope, f"{heavy} must be dropped in chat scope"
    # Non-flow tools stay
    for keep in ("plan.set", "project.get", "release.list", "subscription.current"):
        assert keep in scope, f"{keep} must remain in chat scope"


def test_working_flow_keeps_full_scope():
    state = {"working_flow": {"nodes": [], "edges": []}, "messages": []}
    assert _is_flow_active(state)
    scope = _pick_tool_scope(state, _all_names())
    assert scope is None  # None = full registry exposed


def test_flow_dirty_keeps_full_scope():
    state = {"flow_dirty": True, "messages": []}
    assert _is_flow_active(state)
    assert _pick_tool_scope(state, _all_names()) is None


def test_prior_flow_tool_keeps_full_scope():
    """If a flow.* tool was already used, expand to full scope."""
    state = {
        "messages": [
            Message(role="assistant", content=[ToolUsePart(id="t1", name="flow.upsert_node", arguments={})]),
        ]
    }
    assert _is_flow_active(state)
    assert _pick_tool_scope(state, _all_names()) is None


def test_incomplete_plan_keeps_full_scope():
    state = {
        "current_plan": [
            {"id": "1", "title": "step", "status": "in_progress"},
        ],
        "messages": [],
    }
    assert _is_flow_active(state)
    assert _pick_tool_scope(state, _all_names()) is None

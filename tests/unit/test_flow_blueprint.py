"""Tests for flow.plan/bulk_apply blueprint workflow."""
import pytest

from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools
from agent.tools.flow.blueprint import FlowBlueprint, parse_blueprint


def _tools():
    return {t.name: t for t in build_flow_tools()}


def test_parse_blueprint_from_fenced_json():
    text = """
    Here it is:
    ```json
    {"summary": "test", "nodes": [{"id": "x", "type": "CommandTriggerNode", "data": {"command": "/start"}}], "edges": []}
    ```
    """
    bp = parse_blueprint(text)
    assert isinstance(bp, FlowBlueprint)
    assert len(bp.nodes) == 1
    assert bp.nodes[0].type == "CommandTriggerNode"
    assert bp.summary == "test"


def test_parse_blueprint_from_raw_json():
    text = '{"nodes": [], "edges": []}'
    bp = parse_blueprint(text)
    assert bp.nodes == []
    assert bp.edges == []


def test_parse_blueprint_rejects_bad_json():
    with pytest.raises(ValueError):
        parse_blueprint("not json at all")


@pytest.mark.asyncio
async def test_bulk_apply_validates_each_node():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "a", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "b", "type": "SendTextMessageNode", "data": {"BAD": "field"}},
            ],
            "edges": [],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    assert res.is_error
    assert "BAD" in res.content or "messageText" in res.content


@pytest.mark.asyncio
async def test_bulk_apply_applies_full_blueprint():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "trig", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "msg", "type": "SendTextMessageNode", "data": {"messageText": "Salom"}},
            ],
            "edges": [{"source": "trig", "target": "msg"}],
            "validate_flow": True,
            "layout": True,
            "save": False,
        },
        ctx,
    )
    assert not res.is_error, res.content
    assert "2 node" in res.content
    assert "1 edge" in res.content
    assert "validate OK" in res.content


@pytest.mark.asyncio
async def test_bulk_apply_rejects_orphan_edge():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "a", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            ],
            "edges": [{"source": "a", "target": "ghost"}],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    # Edge orphan is a warning, not a hard error — node was applied
    assert not res.is_error
    assert "Edge ogohlantirish" in res.content


@pytest.mark.asyncio
async def test_bulk_apply_uses_state_blueprint_when_no_args():
    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id="42",
        user_id="u",
        state={
            "flow_blueprint": {
                "nodes": [
                    {"id": "trig", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                ],
                "edges": [],
            },
        },
    )
    res = await tools["flow.bulk_apply"].run(
        {"validate_flow": False, "layout": False, "save": False}, ctx,
    )
    assert not res.is_error, res.content
    assert "1 node" in res.content

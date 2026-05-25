import pytest

from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools


def _tools():
    return {t.name: t for t in build_flow_tools()}


@pytest.mark.asyncio
async def test_build_simple_start_flow():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")

    res = await tools["flow.upsert_node"].run(
        {"id": "t1", "type": "CommandTriggerNode", "data": {"command": "/start"}}, ctx
    )
    assert not res.is_error
    ctx.state.update(res.state_updates)

    res = await tools["flow.upsert_node"].run(
        {
            "id": "m1",
            "type": "SendTextMessageNode",
            "data": {"messageText": "Salom, {{first_name}}!"},
        },
        ctx,
    )
    assert not res.is_error
    ctx.state.update(res.state_updates)

    res = await tools["flow.connect"].run({"source": "t1", "target": "m1"}, ctx)
    assert not res.is_error
    ctx.state.update(res.state_updates)

    res = await tools["flow.validate"].run({}, ctx)
    assert not res.is_error
    assert res.data["ok"] is True


@pytest.mark.asyncio
async def test_flow_save_blocks_on_validation_errors():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")

    # node with missing required field
    res = await tools["flow.upsert_node"].run(
        {"id": "bad", "type": "SendTextMessageNode", "data": {}}, ctx
    )
    # upsert_node validates required fields itself → must error
    assert res.is_error


def test_flow_ai_build_is_not_registered():
    assert "flow.ai_build" not in _tools()

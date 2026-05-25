import pytest

from agent.builder.commands import AddTextCommand, CreateBotCommand
from agent.builder.session import build_from_commands
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


@pytest.mark.asyncio
async def test_flow_ai_build_sets_working_flow(monkeypatch):
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    compiled = build_from_commands(
        [
            CreateBotCommand(action="create_bot", name="Demo", language="uz"),
            AddTextCommand(action="add_text", key="hello", text="Salom", after="start"),
        ]
    )

    class _FakeResponse:
        def __init__(self) -> None:
            self.ok = True
            self.attempts = 1
            self.commands = []
            self.result = compiled
            self.input_tokens = 11
            self.output_tokens = 7

    async def _fake_build(*args, **kwargs):
        return _FakeResponse()

    monkeypatch.setattr("agent.tools.flow.build_bot_with_ai_stream", _fake_build)

    res = await tools["flow.ai_build"].run({"prompt": "salom bot yarat"}, ctx)

    assert not res.is_error
    assert res.data["input_tokens"] == 11
    assert res.data["output_tokens"] == 7
    assert res.state_updates["flow_dirty"] is True
    assert len(res.state_updates["working_flow"]["nodes"]) == 2

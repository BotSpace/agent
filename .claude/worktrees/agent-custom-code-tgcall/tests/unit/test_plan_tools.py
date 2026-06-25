import pytest

from agent.tools.base import ToolContext
from agent.tools.plan import build_plan_tools, has_incomplete_plan


@pytest.mark.asyncio
async def test_plan_set_and_update_cycle():
    tools = {t.name: t for t in build_plan_tools()}
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)

    # set
    res = await tools["plan.set"].run(
        {"items": [{"title": "first"}, {"title": "second"}]}, ctx
    )
    assert not res.is_error
    plan = res.state_updates["current_plan"]
    assert len(plan) == 2
    assert has_incomplete_plan(plan)

    # update first → in_progress
    ctx.state["current_plan"] = plan
    res = await tools["plan.update"].run({"id": plan[0]["id"], "status": "in_progress"}, ctx)
    assert not res.is_error
    ctx.state["current_plan"] = res.state_updates["current_plan"]
    assert ctx.state["current_plan"][0]["status"] == "in_progress"

    # complete both
    for item in plan:
        res = await tools["plan.complete"].run({"id": item["id"]}, ctx)
        ctx.state["current_plan"] = res.state_updates["current_plan"]

    assert not has_incomplete_plan(ctx.state["current_plan"])


@pytest.mark.asyncio
async def test_plan_advance_completes_one_and_starts_next_in_a_single_call():
    tools = {t.name: t for t in build_plan_tools()}
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)

    res = await tools["plan.set"].run(
        {"items": [{"id": "a", "title": "first"}, {"id": "b", "title": "second"}]}, ctx
    )
    ctx.state["current_plan"] = res.state_updates["current_plan"]
    # Mark `a` in_progress first
    await tools["plan.update"].run({"id": "a", "status": "in_progress"}, ctx)

    res = await tools["plan.advance"].run({"complete": ["a"], "start": ["b"]}, ctx)
    assert not res.is_error
    plan = res.state_updates["current_plan"]
    assert plan[0]["status"] == "completed"
    assert plan[1]["status"] == "in_progress"
    assert res.data["completed"] == ["a"]
    assert res.data["started"] == ["b"]


@pytest.mark.asyncio
async def test_plan_advance_reports_missing_ids_but_keeps_real_changes():
    tools = {t.name: t for t in build_plan_tools()}
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    await tools["plan.set"].run({"items": [{"id": "a", "title": "first"}]}, ctx)
    ctx.state["current_plan"] = (await tools["plan.set"].run(
        {"items": [{"id": "a", "title": "first"}]}, ctx
    )).state_updates["current_plan"]

    res = await tools["plan.advance"].run({"complete": ["a"], "start": ["ghost"]}, ctx)
    assert not res.is_error
    assert res.data["completed"] == ["a"]
    assert res.data["started"] == []
    assert "ghost" in res.data["not_found"]


@pytest.mark.asyncio
async def test_plan_advance_all_missing_is_error():
    tools = {t.name: t for t in build_plan_tools()}
    ctx = ToolContext(run_id="r", project_id=None, user_id=None)
    await tools["plan.set"].run({"items": [{"id": "a", "title": "first"}]}, ctx)
    ctx.state["current_plan"] = [{"id": "a", "title": "first", "status": "pending"}]

    res = await tools["plan.advance"].run({"complete": ["x"], "start": ["y"]}, ctx)
    assert res.is_error

import pytest

from agentv3.tools.base import ToolContext
from agentv3.tools.plan import build_plan_tools, has_incomplete_plan


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

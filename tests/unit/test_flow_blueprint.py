"""Tests for flow.design/bulk_apply blueprint workflow."""
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
async def test_bulk_apply_partial_recovery_for_invalid_nodes():
    """Mix of valid + invalid nodes — apply the valid ones, report the failed."""
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
    # Partial mode: not a hard error — one node applied, one failed
    assert not res.is_error
    assert "Qisman apply" in res.content
    assert res.data["nodes_applied"] == 1
    assert len(res.data["failed_nodes"]) == 1
    assert res.data["failed_nodes"][0]["id"] == "b"
    assert res.data["save_pending"] is True


@pytest.mark.asyncio
async def test_bulk_apply_all_invalid_is_hard_error():
    """If no node passes, return a hard error."""
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
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
    assert "Birorta ham node qo'llanmadi" in res.content


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
async def test_flow_clear_empties_working_flow_and_blueprint():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    apply_res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "trig", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "msg", "type": "SendTextMessageNode", "data": {"messageText": "Salom"}},
            ],
            "edges": [{"source": "trig", "target": "msg"}],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    ctx.state.update(apply_res.state_updates or {})
    ctx.state["flow_blueprint"] = {"nodes": [{"id": "stale"}], "edges": []}

    res = await tools["flow.clear"].run({"confirm": True}, ctx)
    assert not res.is_error
    assert "tozalandi" in res.content.lower()
    assert "2 node" in res.content
    ctx.state.update(res.state_updates or {})
    assert ctx.state["working_flow"]["nodes"] == []
    assert ctx.state["working_flow"]["edges"] == []
    assert ctx.state["flow_blueprint"] is None
    assert ctx.state["flow_dirty"] is True


@pytest.mark.asyncio
async def test_flow_clear_without_confirm_is_error():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.clear"].run({"confirm": False}, ctx)
    assert res.is_error
    assert "confirm" in res.content.lower()


@pytest.mark.asyncio
async def test_flow_clear_on_empty_flow_is_idempotent():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.clear"].run({"confirm": True}, ctx)
    assert not res.is_error
    assert "bo'sh" in res.content


@pytest.mark.asyncio
async def test_bulk_apply_removes_old_nodes_before_adding_new():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    # Birinchi: 3 ta node yozamiz
    res1 = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "old-a", "type": "CommandTriggerNode", "data": {"command": "/old"}},
                {"id": "old-b", "type": "SendTextMessageNode", "data": {"messageText": "old"}},
                {"id": "keep", "type": "SendTextMessageNode", "data": {"messageText": "keep me"}},
            ],
            "edges": [{"source": "old-a", "target": "old-b"}, {"source": "old-b", "target": "keep"}],
            "validate_flow": False, "layout": False, "save": False,
        },
        ctx,
    )
    assert not res1.is_error
    ctx.state.update(res1.state_updates or {})

    # Endi: 2 ta eski o'chiriladi, yangi qo'shiladi
    res2 = await tools["flow.bulk_apply"].run(
        {
            "remove_node_ids": ["old-a", "old-b"],
            "nodes": [
                {"id": "new-a", "type": "CommandTriggerNode", "data": {"command": "/new"}},
                {"id": "new-b", "type": "SendTextMessageNode", "data": {"messageText": "new"}},
            ],
            "edges": [{"source": "new-a", "target": "new-b"}, {"source": "new-b", "target": "keep"}],
            "validate_flow": False, "layout": False, "save": False,
        },
        ctx,
    )
    assert not res2.is_error
    assert "2 eski node o'chirildi" in res2.content
    assert res2.data["nodes_removed"] == 2
    ctx.state.update(res2.state_updates or {})

    # Flow tekshirish: faqat keep + 2 ta yangi
    final_ids = {n["id"] for n in ctx.state["working_flow"]["nodes"]}
    assert final_ids == {"keep", "new-a", "new-b"}


@pytest.mark.asyncio
async def test_bulk_apply_remove_missing_node_warns_but_does_not_fail():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.bulk_apply"].run(
        {
            "remove_node_ids": ["nonexistent-id"],
            "nodes": [
                {"id": "trig", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            ],
            "edges": [],
            "validate_flow": False, "layout": False, "save": False,
        },
        ctx,
    )
    assert not res.is_error
    assert res.data["nodes_removed"] == 0
    assert "nonexistent-id" in res.data["removed_missing"]


@pytest.mark.asyncio
async def test_bulk_apply_reads_remove_node_ids_from_blueprint_state():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    # Avval bir node yozamiz
    res0 = await tools["flow.bulk_apply"].run(
        {
            "nodes": [{"id": "old", "type": "CommandTriggerNode", "data": {"command": "/old"}}],
            "edges": [], "validate_flow": False, "layout": False, "save": False,
        },
        ctx,
    )
    ctx.state.update(res0.state_updates or {})
    # Endi blueprint state'da remove_node_ids bilan
    ctx.state["flow_blueprint"] = {
        "nodes": [{"id": "new", "type": "CommandTriggerNode", "data": {"command": "/new"}}],
        "edges": [],
        "remove_node_ids": ["old"],
    }
    res = await tools["flow.bulk_apply"].run(
        {"validate_flow": False, "layout": False, "save": False}, ctx,
    )
    assert not res.is_error
    assert res.data["nodes_removed"] == 1
    assert res.data["nodes_applied"] == 1


@pytest.mark.asyncio
async def test_flow_get_returns_tree_by_default():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    apply_res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "trig", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "msg", "type": "SendTextMessageNode", "data": {"messageText": "Salom"}},
            ],
            "edges": [{"source": "trig", "target": "msg"}],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    ctx.state.update(apply_res.state_updates or {})
    res = await tools["flow.get"].run({}, ctx)
    assert not res.is_error
    assert "Flow arxitekturasi" in res.content
    assert "Trigger" in res.content
    assert "trig" in res.content and "msg" in res.content
    # Not JSON
    assert not res.content.lstrip().startswith("{")


@pytest.mark.asyncio
async def test_flow_get_as_json_returns_json():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    apply_res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "trig", "type": "CommandTriggerNode", "data": {"command": "/start"}},
            ],
            "edges": [],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    ctx.state.update(apply_res.state_updates or {})
    res = await tools["flow.get"].run({"as_json": True}, ctx)
    assert not res.is_error
    assert res.content.lstrip().startswith("{")
    assert "nodes" in res.content


@pytest.mark.asyncio
async def test_flow_get_tree_button_label_keeps_dashed_id_and_text():
    """Button handle `target-handler-{nodeId}-{text}` — kebab-case id va
    dash'li matn to'liq saqlanishi kerak (split bilan kesilmasin)."""
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    apply_res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "trig", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "menu-1", "type": "SendTextMessageNode", "data": {"messageText": "Menu"}},
                {"id": "next", "type": "SendTextMessageNode", "data": {"messageText": "Ok"}},
            ],
            "edges": [
                {"source": "trig", "target": "menu-1"},
                {"source": "menu-1", "target": "next",
                 "source_handle": "target-handler-menu-1-Buy-now"},
            ],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    ctx.state.update(apply_res.state_updates or {})
    res = await tools["flow.get"].run({}, ctx)
    assert not res.is_error
    assert 'button: "Buy-now"' in res.content


@pytest.mark.asyncio
async def test_flow_get_tree_diamond_is_merge_not_loop():
    """Diamond (T->A, T->B, A->C, B->C): C merge nuqtasi sikl emas."""
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    apply_res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "a", "type": "SendTextMessageNode", "data": {"messageText": "A"}},
                {"id": "b", "type": "SendTextMessageNode", "data": {"messageText": "B"}},
                {"id": "c", "type": "SendTextMessageNode", "data": {"messageText": "C"}},
            ],
            "edges": [
                {"source": "t", "target": "a"},
                {"source": "t", "target": "b"},
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    ctx.state.update(apply_res.state_updates or {})
    res = await tools["flow.get"].run({}, ctx)
    assert not res.is_error
    assert "yuqorida" in res.content  # merge marker
    assert "(loop)" not in res.content  # diamond ≠ cycle


@pytest.mark.asyncio
async def test_flow_get_tree_real_cycle_marked_loop():
    """Haqiqiy sikl (A->B->A) `(loop)` deb belgilanadi."""
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    apply_res = await tools["flow.bulk_apply"].run(
        {
            "nodes": [
                {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}},
                {"id": "a", "type": "SendTextMessageNode", "data": {"messageText": "A"}},
                {"id": "b", "type": "SendTextMessageNode", "data": {"messageText": "B"}},
            ],
            "edges": [
                {"source": "t", "target": "a"},
                {"source": "a", "target": "b"},
                {"source": "b", "target": "a"},
            ],
            "validate_flow": False,
            "layout": False,
            "save": False,
        },
        ctx,
    )
    ctx.state.update(apply_res.state_updates or {})
    res = await tools["flow.get"].run({}, ctx)
    assert not res.is_error
    assert "(loop)" in res.content


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

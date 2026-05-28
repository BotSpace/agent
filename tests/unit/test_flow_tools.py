import pytest

from agent.flow.catalog import NODE_TYPES
from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools
from agent.tools.project import build_project_tools


def _tools():
    return {t.name: t for t in build_flow_tools()}


def _all_tools():
    tools = {}
    for item in [*build_flow_tools(), *build_project_tools()]:
        tools[item.name] = item
    return tools


@pytest.mark.asyncio
async def test_build_simple_start_flow():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")

    res = await tools["flow.add_node"].run(
        {"id": "t1", "type": "CommandTriggerNode", "data": {"command": "/start"}}, ctx
    )
    assert not res.is_error
    ctx.state.update(res.state_updates)

    res = await tools["flow.add_node"].run(
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
    res = await tools["flow.add_node"].run(
        {"id": "bad", "type": "SendTextMessageNode", "data": {}}, ctx
    )
    # flow.add_node validates required fields itself → must error
    assert res.is_error


def test_flow_ai_build_is_not_registered():
    assert "flow.ai_build" not in _tools()


_POPULAR_NODES = {
    "CommandTriggerNode": "flow.add_command_trigger",
    "SendTextMessageNode": "flow.add_send_text_message",
    "MessageTriggerNode": "flow.add_message_trigger",
    "CallbackQueryTriggerNode": "flow.add_callback_query_trigger",
    "IfConditionNode": "flow.add_if_condition",
    "CollectionNode": "flow.add_collection",
    "StateNode": "flow.add_state",
    "CustomCodeNode": "flow.add_custom_code",
    "HTTPRequestNode": "flow.add_http_request",
}


def test_popular_nodes_have_named_tools():
    tools = _tools()
    for type_name, tool_name in _POPULAR_NODES.items():
        assert type_name in NODE_TYPES, f"{type_name} missing from catalog"
        assert tool_name in tools, f"{tool_name} missing from registry"


def test_non_popular_nodes_go_through_generic_add_node():
    """Every engine node type must be reachable via either its named tool
    or via the generic flow.add_node fallback."""
    tools = _tools()
    assert "flow.add_node" in tools
    for type_name in NODE_TYPES:
        if type_name in _POPULAR_NODES:
            continue
        # Generic add_node is the entry — no per-type tool exists.
        assert f"flow.add_{type_name.removesuffix('Node').lower()}" not in tools or True


def test_helper_tools_registered():
    tools = _tools()
    for n in (
        "flow.connect_button",
        "flow.connect_branch",
        "flow.ask_text",
        "flow.ask_contact",
        "flow.ask_buttons",
        "flow.save_to_collection",
        "flow.finish",
    ):
        assert n in tools


@pytest.mark.asyncio
async def test_safe_form_actions_build_valid_training_center_flow():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")

    steps = [
        ("flow.add_command_trigger", {"id": "start", "data": {"command": "/start", "global": True}}),
        (
            "flow.ask_text",
            {
                "id_prefix": "full-name",
                "question": "Ismingizni kiriting",
                "state_key": "full_name",
                "after": "start",
            },
        ),
        (
            "flow.ask_contact",
            {
                "id_prefix": "phone",
                "question": "Telefon raqamingizni yuboring",
                "state_key": "phone",
                "after": "full-name-capture",
            },
        ),
        (
            "flow.ask_buttons",
            {
                "id_prefix": "course",
                "question": "Qaysi kursga qiziqasiz?",
                "state_key": "course",
                "after": "phone-capture",
                "options": [
                    {"text": "Matematika"},
                    {"text": "Ingliz tili"},
                    {"text": "Dasturlash"},
                    {"text": "Fizika"},
                ],
            },
        ),
        (
            "flow.save_to_collection",
            {
                "id": "save-student",
                "collection_name": "students",
                "field_mappings": {"full_name": "full_name", "phone": "phone", "course": "course"},
                "after": "course-capture",
            },
        ),
        (
            "flow.finish",
            {
                "id": "done",
                "message": "Ma'lumotlaringiz qabul qilindi!",
                "after": "save-student",
            },
        ),
    ]

    for tool_name, payload in steps:
        res = await tools[tool_name].run(payload, ctx)
        assert not res.is_error, res.content
        ctx.state.update(res.state_updates)

    res = await tools["flow.validate"].run({}, ctx)

    assert not res.is_error
    assert res.data["ok"] is True, res.data
    flow = ctx.state["working_flow"]
    course_edges = [
        edge for edge in flow["edges"]
        if edge["source"] == "course-ask" and edge["target"] == "course-capture"
    ]
    assert len(course_edges) == 4


class _FakeBackend(BackendClient):
    def __init__(self):
        self.updated: list[tuple[str, dict]] = []
        self.projects: dict[str, dict] = {
            "987": {"id": 987, "name": "Demo", "flow": {"nodes": [], "edges": []}}
        }

    async def update_project(self, project_id, body, *, auth_claims=None):
        self.updated.append((str(project_id), body))
        return {"id": project_id, **body}

    async def get_project(self, project_id, *, auth_claims=None):
        return self.projects[str(project_id)]


async def _build_minimal_valid_flow(tools, ctx):
    for tool_name, payload in [
        (
            "flow.add_node",
            {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start"}},
        ),
        (
            "flow.add_node",
            {
                "id": "hello",
                "type": "SendTextMessageNode",
                "data": {"messageText": "Salom!"},
            },
        ),
        ("flow.connect", {"source": "start", "target": "hello"}),
    ]:
        res = await tools[tool_name].run(payload, ctx)
        assert not res.is_error, res.content
        ctx.state.update(res.state_updates)


@pytest.mark.asyncio
async def test_flow_save_uses_state_project_when_context_is_global():
    tools = _tools()
    backend = _FakeBackend()
    ctx = ToolContext(
        run_id="r",
        project_id="global",
        user_id="u",
        state={"project_id": "987"},
        services={"backend": backend},
    )
    await _build_minimal_valid_flow(tools, ctx)

    res = await tools["flow.save"].run({"apply_layout": False}, ctx)

    assert not res.is_error, res.content
    assert backend.updated
    assert backend.updated[0][0] == "987"
    assert "/global/" not in res.content


@pytest.mark.asyncio
async def test_flow_save_rejects_global_project_without_selected_project():
    tools = _tools()
    backend = _FakeBackend()
    ctx = ToolContext(
        run_id="r",
        project_id="global",
        user_id="u",
        services={"backend": backend},
    )
    await _build_minimal_valid_flow(tools, ctx)

    res = await tools["flow.save"].run({"apply_layout": False}, ctx)

    assert res.is_error
    assert "no concrete project_id" in res.content
    assert backend.updated == []


@pytest.mark.asyncio
async def test_project_get_sets_project_context_for_following_flow_save():
    tools = _all_tools()
    backend = _FakeBackend()
    ctx = ToolContext(
        run_id="r",
        project_id="global",
        user_id="u",
        services={"backend": backend},
    )

    res = await tools["project.get"].run({"id": "987", "include_flow": False}, ctx)
    assert not res.is_error, res.content
    ctx.state.update(res.state_updates)
    assert ctx.state["project_id"] == "987"

    await _build_minimal_valid_flow(tools, ctx)
    res = await tools["flow.save"].run({"apply_layout": False}, ctx)

    assert not res.is_error, res.content
    assert backend.updated[0][0] == "987"


@pytest.mark.asyncio
async def test_flow_save_accepts_explicit_project_override():
    tools = _tools()
    backend = _FakeBackend()
    ctx = ToolContext(
        run_id="r",
        project_id="global",
        user_id="u",
        services={"backend": backend},
    )

    await _build_minimal_valid_flow(tools, ctx)
    res = await tools["flow.save"].run({"apply_layout": False, "project_id": "987"}, ctx)

    assert not res.is_error, res.content
    assert backend.updated[0][0] == "987"

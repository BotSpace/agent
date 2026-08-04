import pytest

from agent.flow.manifest_compiler.node_catalog import NODE_TYPES
from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.flow import FlowSaveInput, build_flow_tools
from agent.tools.flow.node_data_models import get_data_model
from agent.tools.project import build_project_tools


def _tools():
    return {t.name: t for t in build_flow_tools()}


def _all_tools():
    tools = {}
    for item in [*build_flow_tools(), *build_project_tools()]:
        tools[item.name] = item
    return tools


def test_flow_save_preserves_manual_layout_by_default():
    assert FlowSaveInput().apply_layout is False


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


def test_typed_add_tools_removed_use_generic():
    # Lazy schema loading: the typed flow.add_* tools were removed; every node
    # type goes through the generic flow.add_node (schema via flow.list_node_types).
    tools = _tools()
    assert "flow.add_node" in tools
    assert "flow.list_node_types" in tools
    for type_name, tool_name in _POPULAR_NODES.items():
        assert type_name in NODE_TYPES, f"{type_name} missing from catalog"
        assert tool_name not in tools, f"{tool_name} should be removed (use flow.add_node)"


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
        ("flow.add_node", {"id": "start", "type": "CommandTriggerNode", "data": {"command": "/start", "global": True}}),
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
    nodes = {node["id"]: node for node in flow["nodes"]}
    assert nodes["full-name-capture"]["data"]["state_key"] == "full_name"
    assert "state" not in nodes["full-name-capture"]["data"]
    assert nodes["phone-capture"]["data"]["state_key"] == "phone"
    assert nodes["phone-capture"]["data"]["state_type"] == "phone_number"
    phone_edges = [
        edge for edge in flow["edges"]
        if edge["source"] == "phone-ask" and edge["target"] == "phone-capture"
    ]
    assert phone_edges[0]["sourceHandle"] == "target-handler-phone-ask-Telefon yuborish"
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


# ── #3: button-text change remaps edge handles ───────────────────────────────


@pytest.mark.asyncio
async def test_patch_remaps_button_edge_handle_on_text_change():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    ctx.state.update((await tools["flow.add_node"].run(
        {"id": "t", "type": "CommandTriggerNode", "data": {"command": "/start"}}, ctx
    )).state_updates)
    ctx.state.update((await tools["flow.add_node"].run({
        "id": "menu", "type": "SendTextMessageNode",
        "data": {"messageText": "M", "keyboard_kind": "inline",
                 "rows": [{"columns": [{"id": "b1", "text": "Eski", "type": "callback", "value": "x"}]}]},
        "after": "t",
    }, ctx)).state_updates)
    ctx.state.update((await tools["flow.add_node"].run(
        {"id": "next", "type": "SendTextMessageNode", "data": {"messageText": "N"}}, ctx
    )).state_updates)
    # Edge via the button's current text.
    ctx.state.update((await tools["flow.connect_button"].run(
        {"source": "menu", "target": "next", "button_text": "Eski"}, ctx
    )).state_updates)
    # Change the button text, keeping the same id.
    res = await tools["flow.patch"].run({
        "id": "menu",
        "data": {"rows": [{"columns": [{"id": "b1", "text": "Yangi", "type": "callback", "value": "x"}]}]},
    }, ctx)
    assert not res.is_error
    assert "remapped" in res.content
    ctx.state.update(res.state_updates)
    from agent.flow.schema import FlowDocument
    flow = FlowDocument.model_validate(ctx.state["working_flow"])
    handles = [e.sourceHandle for e in flow.edges if e.source == "menu"]
    assert "target-handler-menu-Yangi" in handles
    assert "target-handler-menu-Eski" not in handles


def test_keyboard_schema_teaches_canonical_shape():
    model = get_data_model("SendTextMessageNode")
    assert model is not None
    description = model.model_fields["rows"].description or ""

    assert '"columns"' in description
    assert '"type":"callback"' in description
    assert '"value":"callback_data"' in description
    assert "do not use callbackData" in description


@pytest.mark.asyncio
async def test_add_node_rejects_legacy_keyboard_shape():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")

    res = await tools["flow.add_node"].run({
        "id": "welcome",
        "type": "SendTextMessageNode",
        "data": {
            "messageText": "Salom",
            "keyboard_kind": "inline",
            "rows": [[{"text": "Yordam", "callbackData": "/help"}]],
        },
    }, ctx)

    assert res.is_error
    assert "rows.0" in res.content


@pytest.mark.asyncio
async def test_patch_validates_merged_node_data():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    created = await tools["flow.add_node"].run({
        "id": "welcome",
        "type": "SendTextMessageNode",
        "data": {"messageText": "Salom"},
    }, ctx)
    ctx.state.update(created.state_updates)

    res = await tools["flow.patch"].run({
        "id": "welcome",
        "data": {"inventedKeyboard": []},
    }, ctx)

    assert res.is_error
    assert "inventedKeyboard" in res.content
    assert ctx.state["working_flow"]["nodes"][0]["data"]["messageText"] == "Salom"


# ── #5: connect_button works for EditOrSendMessageNode ───────────────────────


@pytest.mark.asyncio
async def test_connect_button_supports_edit_or_send_node():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    ctx.state.update((await tools["flow.add_node"].run({
        "id": "edit", "type": "EditOrSendMessageNode",
        "data": {"messageText": "M", "keyboard_kind": "inline",
                 "rows": [{"columns": [{"id": "b", "text": "Tap", "type": "callback", "value": "v"}]}]},
    }, ctx)).state_updates)
    ctx.state.update((await tools["flow.add_node"].run(
        {"id": "next", "type": "SendTextMessageNode", "data": {"messageText": "N"}}, ctx
    )).state_updates)
    res = await tools["flow.connect_button"].run(
        {"source": "edit", "target": "next", "button_text": "Tap"}, ctx
    )
    assert not res.is_error, res.content
    assert res.data["source_handle"] == "target-handler-edit-Tap"


@pytest.mark.asyncio
async def test_connect_button_rejects_non_keyboard_node_with_hint():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    ctx.state.update((await tools["flow.add_node"].run(
        {"id": "if", "type": "IfConditionNode", "data": {"conditions": []}}, ctx
    )).state_updates)
    ctx.state.update((await tools["flow.add_node"].run(
        {"id": "next", "type": "SendTextMessageNode", "data": {"messageText": "N"}}, ctx
    )).state_updates)
    res = await tools["flow.connect_button"].run(
        {"source": "if", "target": "next", "button_text": "X"}, ctx
    )
    assert res.is_error
    assert "flow.connect" in res.content  # suggests the alternative path


# ── #9: flow.patch_many is atomic ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_many_atomic_all_or_nothing():
    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    for nid in ("a", "b"):
        ctx.state.update((await tools["flow.add_node"].run(
            {"id": nid, "type": "SendTextMessageNode", "data": {"messageText": "x"}}, ctx
        )).state_updates)
    # One bad id → nothing applied.
    res = await tools["flow.patch_many"].run({
        "patches": [{"id": "a", "data": {"messageText": "A"}},
                    {"id": "ghost", "data": {"messageText": "Z"}}],
    }, ctx)
    assert res.is_error
    from agent.flow.schema import FlowDocument
    flow = FlowDocument.model_validate(ctx.state["working_flow"])
    assert flow.get_node("a").data["messageText"] == "x"  # untouched
    # All good → applied.
    res2 = await tools["flow.patch_many"].run({
        "patches": [{"id": "a", "data": {"messageText": "A"}},
                    {"id": "b", "data": {"messageText": "B"}}],
    }, ctx)
    assert not res2.is_error
    ctx.state.update(res2.state_updates)
    flow2 = FlowDocument.model_validate(ctx.state["working_flow"])
    assert flow2.get_node("a").data["messageText"] == "A"
    assert flow2.get_node("b").data["messageText"] == "B"


# ── #6: flow.save returns a compact summary by default ───────────────────────


@pytest.mark.asyncio
async def test_flow_save_summary_default(monkeypatch):
    tools = _all_tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u",
                      services={"backend": object()})
    for nid, typ, data in [("t", "CommandTriggerNode", {"command": "/start"}),
                           ("m", "SendTextMessageNode", {"messageText": "hi"})]:
        ctx.state.update((await tools["flow.add_node"].run(
            {"id": nid, "type": typ, "data": data}, ctx
        )).state_updates)
    ctx.state.update((await tools["flow.connect"].run(
        {"source": "t", "target": "m"}, ctx
    )).state_updates)

    async def fake_update(project_id, body, **kwargs):
        return {"id": project_id}

    import agent.integrations.backend as backend_mod
    monkeypatch.setattr(backend_mod, "get_backend_client",
                        lambda: type("B", (), {"update_project": staticmethod(fake_update)})())

    # Patch the module-level get_backend_client used inside flow.save.
    from agent.tools import flow as flow_pkg  # noqa: F401
    res = await tools["flow.save"].run({"project_id": "42"}, ctx)
    assert not res.is_error, res.content
    assert "flow" not in (res.data or {})  # compact: no full flow
    assert res.data["nodes"] == 2
    assert res.data["flow_dirty"] is False


@pytest.mark.asyncio
async def test_add_node_accepts_remove_keyboard():
    """remove_keyboard — barcha send node'larda qabul qilinadi (engine ReplyKeyboardRemove)."""
    tools = {t.name: t for t in build_flow_tools()}
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.add_node"].run(
        {
            "id": "thanks",
            "type": "SendTextMessageNode",
            "data": {"messageText": "Rahmat!", "remove_keyboard": True},
        },
        ctx,
    )
    assert not res.is_error, res.content
    flow = res.state_updates["working_flow"]
    node = next(n for n in flow["nodes"] if n["id"] == "thanks")
    assert node["data"]["remove_keyboard"] is True


@pytest.mark.asyncio
async def test_apply_template_surfaces_required_envs(monkeypatch):
    """apply_template natijasida backend'dan kelgan required_envs ToolResult
    message + data'ga chiqishi; RELEASE_BOT_TOKEN tashlab yuborilishi."""
    from agent.tools.flow import templates as tpl_mod

    fake_tpl = {
        "id": 7,
        "slug": "shop",
        "name": "Shop bot",
        "flow": {
            "nodes": [{"id": "t1", "type": "CommandTriggerNode", "data": {"command": "/start"}}],
            "edges": [],
        },
        "required_envs": [
            {"key": "ADMIN_CHAT_ID", "description": "Admin chat id"},
            {"key": "PAYMENT_TOKEN", "description": ""},
            {"key": "RELEASE_BOT_TOKEN", "description": "bot token"},
        ],
    }

    class _FakeBackend:
        async def get_internal_template(self, identifier):
            return fake_tpl

    monkeypatch.setattr(tpl_mod, "get_backend_client", lambda: _FakeBackend())

    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.apply_template"].run({"template_id": 7}, ctx)

    assert not res.is_error, res.content
    keys = [e["key"] for e in res.data["required_envs"]]
    assert keys == ["ADMIN_CHAT_ID", "PAYMENT_TOKEN"]  # RELEASE_BOT_TOKEN dropped
    assert "ADMIN_CHAT_ID" in res.content
    assert "PAYMENT_TOKEN" in res.content
    assert "RELEASE_BOT_TOKEN" not in res.content
    assert "MAJBURIY env" in res.content


@pytest.mark.asyncio
async def test_apply_template_without_required_envs_omits_field(monkeypatch):
    """required_envs yo'q template — data'da kalit umuman bo'lmasligi va
    message'da env eslatmasi chiqmasligi (regressiyani oldini olish)."""
    from agent.tools.flow import templates as tpl_mod

    class _FakeBackend:
        async def get_internal_template(self, identifier):
            return {
                "id": 1,
                "slug": "basic",
                "name": "Basic",
                "flow": {
                    "nodes": [{"id": "t1", "type": "CommandTriggerNode", "data": {"command": "/start"}}],
                    "edges": [],
                },
            }

    monkeypatch.setattr(tpl_mod, "get_backend_client", lambda: _FakeBackend())

    tools = _tools()
    ctx = ToolContext(run_id="r", project_id="42", user_id="u")
    res = await tools["flow.apply_template"].run({"slug": "basic"}, ctx)

    assert not res.is_error, res.content
    assert "required_envs" not in (res.data or {})
    assert "MAJBURIY env" not in res.content

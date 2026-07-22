import pytest
from pydantic import BaseModel, Field, ValidationError

import agent.flow.blueprint as blueprint_module
from agent.flow.blueprint import (
    BlueprintDefinition,
    BlueprintField,
    BotBlueprint,
    compile_blueprint,
    list_blueprint_definitions,
    register_blueprint,
    regression_steps,
)
from agent.flow.validator import validate
from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.flow.blueprints import (
    BuildBlueprintInput,
    DiscardBlueprintInput,
    DraftBlueprintInput,
    blueprint_draft,
    build_blueprint,
    discard_blueprint,
)


def _field(key: str = "full_name", kind: str = "text") -> BlueprintField:
    return BlueprintField(key=key, label=key, prompt=f"{key} ni kiriting", kind=kind)


class _BoardBackend(BackendClient):
    def __init__(self, board: dict):
        super().__init__("http://backend.invalid")
        self.board = board

    async def get_kanban_board(self, board_id, **kwargs):
        return self.board


def test_blueprint_registry_auto_discovers_extension_modules():
    patterns = {definition.pattern for definition in list_blueprint_definitions()}
    assert {
        "info",
        "lead",
        "registration",
        "catalog",
        "order",
        "support",
        "feedback",
        "appointment",
    } <= patterns


def test_unknown_blueprint_is_rejected_with_available_patterns():
    with pytest.raises(ValidationError, match="unknown blueprint"):
        BotBlueprint(pattern="invented", title="Bad")


def test_registration_compiler_produces_semantically_valid_dynamic_flow():
    blueprint = BotBlueprint(
        pattern="registration",
        title="Kursga yozilish",
        intro="Kursni tanlang",
        destination_collection="registrations",
        catalog_collection="courses",
        relation_field="course_id",
        fields=[_field(), _field("phone", "contact")],
    )
    flow = compile_blueprint(blueprint)
    result = validate(
        flow,
        known_collections=["courses", "registrations"],
        collection_schemas={
            "courses": {"name": "text", "price": "float"},
            "registrations": {
                "full_name": "text",
                "phone": "phone",
                "course_id": "relation",
            },
        },
    )
    assert result.ok, result.render()
    assert result.issues == []
    menu = flow.get_node("catalog-menu")
    assert menu is not None
    dynamic = menu.data["rows"][0]["columns"][0]["collectionData"]
    assert dynamic["callbackDataPattern"] == "collection:courses:{id}"
    selected = flow.get_node("catalog-selected")
    assert selected is not None
    assert selected.data["global"] is True
    assert selected.data["state_type"] == "full_data"
    save = flow.get_node("save-result")
    assert save is not None
    assert save.data["fieldMappings"]["course_id"] == "{{selected_item.id}}"


def test_registration_can_write_operational_result_to_kanban():
    blueprint = BotBlueprint(
        pattern="registration",
        title="Kursga yozilish",
        intro="Kursni tanlang",
        catalog_collection="courses",
        relation_field="course_id",
        kanban_board_id="board-1",
        kanban_column_id="column-1",
        fields=[_field(), _field("phone", "contact")],
    )

    flow = compile_blueprint(blueprint)
    save = flow.get_node("save-result")
    assert save is not None
    assert save.type == "KanbanCardNode"
    assert save.data == {
        "boardId": "board-1",
        "columnId": "column-1",
        "operation": "upsert",
        "kind": "registration",
        "title": "{{full_name}} — {{selected_item.name}}",
        "description": "Tanlov: {{selected_item.name}}\nfull_name: {{full_name}}\nphone: {{phone}}",
        "priority": "normal",
        "externalKey": "{{chat_id}}:registration:{{selected_item.id}}",
        "outputKey": "kanban_card",
    }
    assert all(node.type != "CollectionNode" for node in flow.nodes)


@pytest.mark.asyncio
async def test_build_blueprint_verifies_board_link_and_real_column():
    blueprint = BotBlueprint(
        pattern="lead",
        title="Lead",
        kanban_board_id="board-1",
        kanban_column_id="column-1",
        fields=[_field()],
    )
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={"working_flow": {"nodes": [], "edges": []}, "collections": []},
        services={
            "backend": _BoardBackend(
                {
                    "id": "board-1",
                    "project_ids": ["project-1"],
                    "columns": [{"id": "column-1", "name": "Yangi"}],
                }
            )
        },
    )

    result = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )

    assert not result.is_error, result.content
    assert result.state_updates["working_flow"]["nodes"][-2]["type"] == "KanbanCardNode"

    blueprint.kanban_column_id = "invented-column"
    invalid = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )
    assert invalid.is_error
    assert "real column ID" in invalid.content


def test_catalog_blueprint_rejects_capture_or_destination_fields():
    with pytest.raises(ValidationError, match="browse-only"):
        BotBlueprint(
            pattern="catalog",
            title="Noto'g'ri katalog",
            catalog_collection="courses",
            destination_collection="registrations",
            fields=[_field()],
        )


def test_registration_regression_uses_real_catalog_fixture_for_full_journey():
    blueprint = BotBlueprint(
        pattern="registration",
        title="Kursga yozilish",
        intro="Kursni tanlang",
        destination_collection="registrations",
        catalog_collection="courses",
        relation_field="course_id",
        fields=[_field(), _field("phone", "contact")],
    )
    steps = regression_steps(
        blueprint,
        {"_id": "course-1", "name": "Ingliz tili", "price": 100000},
    )
    assert [step["type"] for step in steps] == [
        "command",
        "click",
        "text",
        "contact",
        "click",
    ]
    assert steps[0]["expect"]["keyboard_contains"] == "Ingliz tili"
    assert steps[1]["data"] == "collection:courses:course-1"


def test_feedback_blueprint_supplies_typed_rating_field_by_default():
    blueprint = BotBlueprint(
        pattern="feedback",
        title="Baho",
        destination_collection="feedback",
    )
    assert [field.key for field in blueprint.fields] == ["rating"]


class _ExtensionOptions(BaseModel):
    minimum_age: int = Field(default=18, ge=1)


def test_extension_can_own_version_presets_and_typed_options(monkeypatch):
    monkeypatch.setattr(blueprint_module, "_BLUEPRINTS", dict(blueprint_module._BLUEPRINTS))
    register_blueprint(
        BlueprintDefinition(
            pattern="typed_extension",
            title="Typed extension",
            description="Test",
            compiler=lambda blueprint: compile_blueprint(
                blueprint.model_copy(update={"pattern": "info"})
            ),
            test_generator=regression_steps,
            version=2,
            field_presets=(
                {"key": "full_name", "label": "Ism", "prompt": "Ismingiz:"},
            ),
            options_model=_ExtensionOptions,
        )
    )
    blueprint = BotBlueprint(
        pattern="typed_extension",
        title="Typed",
        options={"minimum_age": 21},
    )
    assert blueprint.options == {"minimum_age": 21}
    with pytest.raises(ValidationError, match="greater than or equal"):
        BotBlueprint(pattern="typed_extension", title="Typed", options={"minimum_age": 0})


@pytest.mark.asyncio
async def test_blueprint_draft_uses_registered_presets_and_reports_cms_requirements():
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={"collections": []},
    )
    result = await blueprint_draft.run(
        DraftBlueprintInput(pattern="appointment", title="Klinika qabuli").model_dump(),
        ctx,
    )
    assert not result.is_error, result.content
    assert result.data["blueprint"]["destination_collection"] == "appointments"
    assert [field["key"] for field in result.data["blueprint"]["fields"]] == [
        "full_name",
        "phone",
        "preferred_time",
    ]
    assert result.data["ready"] is False
    assert "`appointments` collection yarating" in result.data["requirements"]
    fields = result.data["cms_schema_plan"]["destination"]["fields"]
    assert {field["name"]: field["kind"] for field in fields} == {
        "full_name": "text",
        "phone": "phone",
        "preferred_time": "text",
    }


@pytest.mark.asyncio
async def test_blueprint_draft_uses_kanban_instead_of_destination_collection():
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={
            "collections": [
                {
                    "name": "courses",
                    "fields": ["name"],
                    "field_kinds": {"name": "text"},
                }
            ]
        },
    )
    result = await blueprint_draft.run(
        DraftBlueprintInput(
            pattern="registration",
            title="Kursga yozilish",
            kanban_board_id="board-1",
            kanban_column_id="column-1",
        ).model_dump(),
        ctx,
    )

    assert not result.is_error, result.content
    assert result.data["ready"] is True
    assert result.data["blueprint"]["destination_collection"] is None
    assert result.data["blueprint"]["kanban_board_id"] == "board-1"
    assert result.data["cms_schema_plan"]["destination"] is None


@pytest.mark.asyncio
async def test_blueprint_draft_reports_incompatible_capture_field_kind():
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={
            "collections": [
                {
                    "name": "leads",
                    "fields": ["full_name", "phone"],
                    "field_kinds": {"full_name": "text", "phone": "text"},
                }
            ]
        },
    )
    result = await blueprint_draft.run(
        DraftBlueprintInput(pattern="lead", title="Lead").model_dump(), ctx
    )

    assert result.data["ready"] is False
    assert "phone:text→phone" in result.data["requirements"][0]


@pytest.mark.asyncio
async def test_build_blueprint_is_atomic_and_restorable():
    previous = {
        "nodes": [
            {
                "id": "old",
                "type": "CommandTriggerNode",
                "data": {"command": "/old", "global": True},
                "position": {"x": 0, "y": 0},
            }
        ],
        "edges": [],
    }
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={
            "working_flow": previous,
            "collections": [
                {
                    "name": "leads",
                    "fields": ["name", "phone"],
                    "field_kinds": {"name": "text", "phone": "phone"},
                }
            ],
        },
    )
    blueprint = BotBlueprint(
        pattern="lead",
        title="Lead",
        intro="Salom",
        destination_collection="leads",
        fields=[_field("name"), _field("phone", "contact")],
    )
    blocked = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            replace_existing=False,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )
    assert blocked.is_error
    assert blocked.state_updates == {}

    built = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            replace_existing=True,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )
    assert not built.is_error, built.content
    assert built.state_updates["blueprint_previous_flow"] == previous
    restore_ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={**ctx.state, **built.state_updates},
    )
    restored = await discard_blueprint.run(DiscardBlueprintInput().model_dump(), restore_ctx)
    assert not restored.is_error
    assert restored.state_updates["working_flow"] == previous


@pytest.mark.asyncio
async def test_build_blueprint_rejects_missing_collection_without_mutation():
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={"working_flow": {"nodes": [], "edges": []}, "collections": [{"name": "other"}]},
    )
    blueprint = BotBlueprint(
        pattern="lead",
        title="Lead",
        destination_collection="leads",
        fields=[_field("name")],
    )
    result = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )
    assert result.is_error
    assert result.state_updates == {}
    assert "`leads`" in result.content


@pytest.mark.asyncio
async def test_build_blueprint_rejects_incompatible_capture_field_kind():
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={
            "working_flow": {"nodes": [], "edges": []},
            "collections": [
                {
                    "name": "leads",
                    "fields": ["full_name", "phone"],
                    "field_kinds": {"full_name": "text", "phone": "text"},
                }
            ],
        },
    )
    blueprint = BotBlueprint(
        pattern="lead",
        title="Lead",
        destination_collection="leads",
        fields=[_field(), _field("phone", "contact")],
    )

    result = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )

    assert result.is_error
    assert "`leads.phone` text emas, `phone`" in result.content
    assert result.state_updates == {}


@pytest.mark.asyncio
async def test_build_blueprint_treats_loaded_empty_collection_list_as_authoritative():
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={"working_flow": {"nodes": [], "edges": []}, "collections": []},
    )
    blueprint = BotBlueprint(
        pattern="lead",
        title="Lead",
        destination_collection="leads",
        fields=[_field("name")],
    )
    result = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )
    assert result.is_error
    assert "`leads`" in result.content


def test_blueprint_phone_field_requires_contact_kind():
    with pytest.raises(ValidationError, match="must be `contact`"):
        BlueprintField(key="phone", label="Telefon", prompt="Telefonni kiriting", kind="text")


@pytest.mark.asyncio
async def test_build_blueprint_rejects_unavailable_cms_contract_without_mutation():
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={"working_flow": {"nodes": [], "edges": []}},
    )
    blueprint = BotBlueprint(
        pattern="lead",
        title="Lead",
        destination_collection="leads",
        fields=[_field("full_name")],
    )
    result = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )
    assert result.is_error
    assert result.state_updates == {}
    assert "CMS contract yuklanmagan" in result.content


@pytest.mark.asyncio
async def test_build_blueprint_replaces_new_project_clean_scaffold_without_force_flag():
    scaffold = {
        "nodes": [
            {
                "id": "start",
                "type": "CommandTriggerNode",
                "data": {"command": "/start", "global": True},
                "position": {"x": 0, "y": 0},
            },
            {
                "id": "welcome",
                "type": "SendTextMessageNode",
                "data": {"messageText": "Assalomu alaykum! 👋"},
                "position": {"x": 320, "y": 0},
            },
        ],
        "edges": [{"id": "e", "source": "start", "target": "welcome"}],
    }
    ctx = ToolContext(
        run_id="test",
        project_id="project-1",
        user_id="user-1",
        state={
            "working_flow": scaffold,
            "collections": [
                {
                    "name": "leads",
                    "fields": ["full_name", "phone"],
                    "field_kinds": {"full_name": "text", "phone": "phone"},
                }
            ],
        },
    )
    blueprint = BotBlueprint(
        pattern="lead",
        title="Lead",
        destination_collection="leads",
        fields=[_field(), _field("phone", "contact")],
    )
    result = await build_blueprint.run(
        BuildBlueprintInput(
            blueprint=blueprint,
            verify=False,
            save_regression_test=False,
        ).model_dump(),
        ctx,
    )
    assert not result.is_error, result.content
    assert result.state_updates["blueprint_previous_flow"] == scaffold

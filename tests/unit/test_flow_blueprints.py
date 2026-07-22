import pytest
from pydantic import ValidationError

from agent.flow.blueprint import (
    BlueprintField,
    BotBlueprint,
    compile_blueprint,
    list_blueprint_definitions,
    regression_steps,
)
from agent.flow.validator import validate
from agent.tools.base import ToolContext
from agent.tools.flow.blueprints import (
    BuildBlueprintInput,
    DiscardBlueprintInput,
    build_blueprint,
    discard_blueprint,
)


def _field(key: str = "full_name", kind: str = "text") -> BlueprintField:
    return BlueprintField(key=key, label=key, prompt=f"{key} ni kiriting", kind=kind)


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

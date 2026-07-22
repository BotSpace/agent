from __future__ import annotations

import pytest

from agent.flow import FlowDocument, validate
from agent.tools.base import ToolContext
from agent.tools.flow import build_flow_tools


def _tools():
    return {tool.name: tool for tool in build_flow_tools()}


def _ctx() -> ToolContext:
    return ToolContext(
        run_id="r",
        project_id="p",
        user_id="u",
        state={"working_flow": {"nodes": [], "edges": []}},
        services={},
    )


@pytest.mark.asyncio
async def test_ask_buttons_builds_canonical_collection_callback() -> None:
    result = await _tools()["flow.ask_buttons"].run(
        {
            "id_prefix": "course",
            "question": "Kursni tanlang",
            "state_key": "course_id",
            "style": "inline",
            "state_type": "collection_id",
            "collection_name": "courses",
            "options": [{"text": "Matematika", "value": "507f1f77bcf86cd799439011"}],
        },
        _ctx(),
    )

    assert not result.is_error, result.content
    flow = result.state_updates["working_flow"]
    ask = next(node for node in flow["nodes"] if node["id"] == "course-ask")
    capture = next(node for node in flow["nodes"] if node["id"] == "course-capture")
    assert ask["data"]["keyboard"]["inline"][0][0]["value"] == ("collection:courses:507f1f77bcf86cd799439011")
    assert capture["data"]["type"] == "collection_id"


@pytest.mark.asyncio
async def test_save_to_collection_normalizes_runtime_context_mappings() -> None:
    result = await _tools()["flow.save_to_collection"].run(
        {
            "id": "save",
            "collection_name": "enrollments",
            "field_mappings": {
                "full_name": "full_name",
                "user_id": "user_id",
                "chat_id": "chat_id",
            },
        },
        _ctx(),
    )

    assert not result.is_error, result.content
    save = next(node for node in result.state_updates["working_flow"]["nodes"] if node["id"] == "save")
    assert save["data"]["fieldMappings"] == {
        "full_name": "full_name",
        "user_id": "{{user_id}}",
        "chat_id": "{{chat_id}}",
    }


def test_collection_mapping_is_a_consumer_not_a_state_producer() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "start",
                    "type": "CommandTriggerNode",
                    "data": {"command": "/start"},
                },
                {
                    "id": "save",
                    "type": "CollectionNode",
                    "data": {
                        "collection_name": "enrollments",
                        "fieldMappings": {"name": "missing_name"},
                    },
                },
            ],
            "edges": [{"id": "e", "source": "start", "target": "save"}],
        }
    )

    result = validate(flow)
    assert any(issue.code == "unresolved_collection_mapping_source" for issue in result.issues)


def test_unknown_template_mapping_is_an_error() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "start",
                    "type": "CommandTriggerNode",
                    "data": {"command": "/start"},
                },
                {
                    "id": "save",
                    "type": "CollectionNode",
                    "data": {
                        "collection_name": "enrollments",
                        "fieldMappings": {"name": "{{missing_name}}"},
                    },
                },
            ],
            "edges": [{"id": "e", "source": "start", "target": "save"}],
        }
    )

    result = validate(flow)
    assert not result.ok
    assert any(issue.code == "unknown_collection_mapping_variable" for issue in result.errors)


def test_custom_code_rejects_async_await() -> None:
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "start",
                    "type": "CommandTriggerNode",
                    "data": {"command": "/start"},
                },
                {
                    "id": "load",
                    "type": "CustomCodeNode",
                    "data": {"jsCode": "const rows = await db_find('courses', {});"},
                },
            ],
            "edges": [{"id": "e", "source": "start", "target": "load"}],
        }
    )

    result = validate(flow)
    assert any(issue.code == "custom_code_async_unsupported" for issue in result.errors)

from agent.flow.repair import repair_flow
from agent.flow.schema import FlowDocument


def test_repair_normalizes_typed_capture_state_templates_and_nested_mapping():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "capture",
                    "type": "MessageTriggerNode",
                    "data": {
                        "type": "any",
                        "filter": "any",
                        "global": False,
                        "state": {"key": "phone", "type": "contact_phone_number"},
                    },
                },
                {
                    "id": "selected",
                    "type": "CallbackQueryTriggerNode",
                    "data": {
                        "filter": "starts_with",
                        "context": {"value": "collection:courses:"},
                        "global": True,
                        "state_key": "course",
                        "state_type": "full_data",
                    },
                },
                {
                    "id": "save",
                    "type": "CollectionNode",
                    "data": {
                        "collection_name": "registrations",
                        "fieldMappings": {"phone": "phone", "course_id": "course._id"},
                    },
                },
                {
                    "id": "message",
                    "type": "EditOrSendMessageNode",
                    "data": {"messageText": "Telefon: {{state_phone}}"},
                },
            ],
            "edges": [],
        }
    )
    repaired, changes = repair_flow(flow)
    capture = repaired.get_node("capture")
    assert capture is not None
    assert capture.data["type"] == "contact"
    assert capture.data["state_key"] == "phone"
    assert capture.data["state_type"] == "phone_number"
    assert "state" not in capture.data
    save = repaired.get_node("save")
    assert save is not None
    assert save.data["fieldMappings"]["course_id"] == "{{course.id}}"
    message = repaired.get_node("message")
    assert message is not None
    assert message.data["messageText"] == "Telefon: {{phone}}"
    assert changes


def test_repair_canonicalizes_collection_keyboard_and_trigger():
    flow = FlowDocument.model_validate(
        {
            "nodes": [
                {
                    "id": "menu",
                    "type": "EditOrSendMessageNode",
                    "data": {
                        "messageText": "Tanlang",
                        "rows": [
                            {
                                "columns": [
                                    {
                                        "collectionData": {
                                            "collection": "courses",
                                            "field": "name",
                                            "text": "{{name}}",
                                            "callbackDataPattern": "course_{id}",
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                },
                {
                    "id": "selected",
                    "type": "CallbackQueryTriggerNode",
                    "data": {
                        "filter": "starts_with",
                        "context": {"value": "course_courses_"},
                        "global": False,
                        "state_key": "course",
                        "state_type": "full_data",
                    },
                },
            ],
            "edges": [],
        }
    )
    repaired, _ = repair_flow(flow)
    menu = repaired.get_node("menu")
    assert menu is not None
    dynamic = menu.data["rows"][0]["columns"][0]["collectionData"]
    assert dynamic["callbackDataPattern"] == "collection:courses:{id}"
    selected = repaired.get_node("selected")
    assert selected is not None
    assert selected.data["context"]["value"] == "collection:courses:"
    assert selected.data["global"] is True

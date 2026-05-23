from agent.flow.schema import FlowDocument, FlowEdge, FlowNode
from agent.flow.validator import validate


def test_valid_flow_passes():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="m", type="SendTextMessageNode", data={"messageText": "Hi!"}),
        ],
        edges=[FlowEdge(id="e1", source="t", target="m")],
    )
    result = validate(flow)
    assert result.ok, result.render()


def test_missing_required_field_is_error():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={}),  # missing `command`
        ]
    )
    result = validate(flow)
    assert not result.ok
    assert any(i.code == "missing_required_field" for i in result.errors)


def test_unreachable_node_warning():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/x"}),
            FlowNode(id="orphan", type="SendTextMessageNode", data={"messageText": "lost"}),
        ]
    )
    result = validate(flow)
    assert any(i.code == "unreachable" for i in result.warnings)

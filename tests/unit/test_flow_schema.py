from agentv3.flow.schema import FlowDocument, FlowEdge, FlowNode


def test_flow_document_upsert_replaces_existing_node():
    flow = FlowDocument(nodes=[FlowNode(id="a", type="CommandTriggerNode", data={"command": "/start"})])
    flow.upsert_node(FlowNode(id="a", type="MessageTriggerNode", data={}))
    assert len(flow.nodes) == 1
    assert flow.nodes[0].type == "MessageTriggerNode"


def test_remove_node_drops_edges():
    flow = FlowDocument(
        nodes=[FlowNode(id="a", type="X", data={}), FlowNode(id="b", type="Y", data={})],
        edges=[FlowEdge(id="e1", source="a", target="b")],
    )
    flow.remove_node("a")
    assert all(n.id != "a" for n in flow.nodes)
    assert flow.edges == []


def test_triggers_detected_when_no_incoming_edge():
    flow = FlowDocument(
        nodes=[FlowNode(id="root", type="CommandTriggerNode", data={"command": "/x"})]
    )
    triggers = flow.triggers()
    assert len(triggers) == 1
    assert triggers[0].id == "root"

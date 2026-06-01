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


def test_button_source_handle_must_match_exact_button_text():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="menu",
                type="SendTextMessageNode",
                data={
                    "messageText": "Kurs tanlang",
                    "keyboard": {
                        "active": "reply",
                        "reply": [[{"id": "math", "text": "Matematika", "type": "reply"}]],
                    },
                },
            ),
            FlowNode(id="math", type="SendTextMessageNode", data={"messageText": "Matematika"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="menu"),
            FlowEdge(id="e2", source="menu", target="math", sourceHandle="target-handler-menu-Matematika"),
        ],
    )

    assert validate(flow).ok

    flow.edges[1].sourceHandle = "target-handler-menu-Ingliz tili"
    result = validate(flow)

    assert not result.ok
    assert any(i.code == "unsupported_source_handle" for i in result.errors)


def test_branch_source_handles_are_validated():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="check", type="CheckMembershipNode", data={"chatId": "-100123"}),
            FlowNode(id="ok", type="SendTextMessageNode", data={"messageText": "OK"}),
            FlowNode(id="no", type="SendTextMessageNode", data={"messageText": "NO"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="check"),
            FlowEdge(id="e2", source="check", target="ok", sourceHandle="is-member"),
            FlowEdge(id="e3", source="check", target="no", sourceHandle="not_member"),
        ],
    )

    result = validate(flow)

    assert not result.ok
    assert any(i.code == "unsupported_source_handle" for i in result.errors)


def test_training_center_form_flow_is_valid():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="start", type="CommandTriggerNode", data={"command": "/start", "global": True}),
            FlowNode(id="ask-name", type="SendTextMessageNode", data={"messageText": "Ismingizni kiriting"}),
            FlowNode(
                id="name-input",
                type="MessageTriggerNode",
                data={"type": "text", "filter": "any", "global": False, "state": {"key": "ism", "type": "text"}},
            ),
            FlowNode(
                id="ask-phone",
                type="SendTextMessageNode",
                data={
                    "messageText": "Telefon raqamingizni yuboring",
                    "keyboard": {
                        "active": "reply",
                        "reply": [[{"id": "phone", "text": "Telefon yuborish", "type": "request_contact"}]],
                    },
                },
            ),
            FlowNode(
                id="phone-input",
                type="MessageTriggerNode",
                data={
                    "type": "contact",
                    "filter": "any",
                    "global": False,
                    "state": {"key": "telefon", "type": "phone_number"},
                },
            ),
            FlowNode(id="ask-age", type="SendTextMessageNode", data={"messageText": "Yoshingizni kiriting"}),
            FlowNode(
                id="age-input",
                type="MessageTriggerNode",
                data={"type": "text", "filter": "any", "global": False, "state": {"key": "yosh", "type": "text"}},
            ),
            FlowNode(
                id="course-menu",
                type="SendTextMessageNode",
                data={
                    "messageText": "Qaysi kursga qiziqasiz?",
                    "keyboard": {
                        "active": "reply",
                        "reply": [
                            [
                                {"id": "math", "text": "Matematika", "type": "reply"},
                                {"id": "english", "text": "Ingliz tili", "type": "reply"},
                            ],
                            [
                                {"id": "coding", "text": "Dasturlash", "type": "reply"},
                                {"id": "physics", "text": "Fizika", "type": "reply"},
                            ],
                        ],
                    },
                },
            ),
            FlowNode(
                id="save",
                type="CollectionNode",
                data={
                    "collection_name": "students",
                    "fieldMappings": {"ism": "ism", "telefon": "telefon", "yosh": "yosh", "kurs": "kurs"},
                },
            ),
            FlowNode(id="done", type="SendTextMessageNode", data={"messageText": "Ma'lumotlaringiz qabul qilindi!"}),
        ],
        edges=[
            FlowEdge(id="e1", source="start", target="ask-name"),
            FlowEdge(id="e2", source="ask-name", target="name-input"),
            FlowEdge(id="e3", source="name-input", target="ask-phone"),
            FlowEdge(id="e4", source="ask-phone", target="phone-input"),
            FlowEdge(id="e5", source="phone-input", target="ask-age"),
            FlowEdge(id="e6", source="ask-age", target="age-input"),
            FlowEdge(id="e7", source="age-input", target="course-menu"),
            FlowEdge(id="e8", source="course-menu", target="save", sourceHandle="target-handler-course-menu-Matematika"),
            FlowEdge(id="e9", source="course-menu", target="save", sourceHandle="target-handler-course-menu-Ingliz tili"),
            FlowEdge(id="e10", source="course-menu", target="save", sourceHandle="target-handler-course-menu-Dasturlash"),
            FlowEdge(id="e11", source="course-menu", target="save", sourceHandle="target-handler-course-menu-Fizika"),
            FlowEdge(id="e12", source="save", target="done"),
        ],
    )

    result = validate(flow)

    assert result.ok, result.render()


def test_editorsend_supports_button_edges_directly():
    # EditOrSendMessageNode supports button-edges just like SendTextMessageNode;
    # the engine resolves "target-handler-{nodeId}-{text}" edges type-agnostically
    # (controller.go pause list includes EditOrSendMessageNode). Before the fix
    # the matching edge was wrongly flagged as unsupported_source_handle.
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="menu",
                type="EditOrSendMessageNode",
                data={
                    "messageText": "Kurs tanlang",
                    "keyboard": {
                        "active": "reply",
                        "reply": [[{"id": "math", "text": "Matematika", "type": "reply"}]],
                    },
                },
            ),
            FlowNode(id="math", type="SendTextMessageNode", data={"messageText": "Matematika"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="menu"),
            FlowEdge(
                id="e2",
                source="menu",
                target="math",
                sourceHandle="target-handler-menu-Matematika",
            ),
        ],
    )

    result = validate(flow)

    assert result.ok, result.render()


def test_customcode_declared_outputs_are_valid_source_handles():
    # CustomCodeNode with data.outputs=[{name}] exposes a dynamic
    # `output-{name}` source handle per output (engine routes exit('name')
    # via sourceHandle == "output-"+name). The validator must accept it.
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="code",
                type="CustomCodeNode",
                data={
                    "jsCode": "if (x) exit('ok'); else exit('fail');",
                    "outputs": [{"name": "ok"}, {"name": "fail"}],
                },
            ),
            FlowNode(id="yes", type="SendTextMessageNode", data={"messageText": "OK"}),
            FlowNode(id="no", type="SendTextMessageNode", data={"messageText": "Fail"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="code"),
            FlowEdge(id="e2", source="code", target="yes", sourceHandle="output-ok"),
            FlowEdge(id="e3", source="code", target="no", sourceHandle="output-fail"),
        ],
    )

    result = validate(flow)

    assert result.ok, result.render()

    # An output handle that was NOT declared is still rejected.
    flow.edges[1].sourceHandle = "output-undeclared"
    bad = validate(flow)
    assert not bad.ok
    assert any(i.code == "unsupported_source_handle" for i in bad.errors)

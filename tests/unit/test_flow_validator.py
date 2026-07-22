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


def test_full_data_relation_mapping_requires_public_id_not_mongo_id():
    flow = FlowDocument(
        nodes=[
            FlowNode(
                id="selected",
                type="CallbackQueryTriggerNode",
                data={
                    "filter": "starts_with",
                    "context": {"value": "collection:courses:"},
                    "global": True,
                    "state_key": "course",
                    "state_type": "full_data",
                },
            ),
            FlowNode(
                id="save",
                type="CollectionNode",
                data={
                    "collection_name": "registrations",
                    "fieldMappings": {"course_id": "{{course._id}}"},
                },
            ),
        ],
        edges=[FlowEdge(id="e1", source="selected", target="save")],
    )
    result = validate(flow, known_collections=["courses", "registrations"])
    assert not result.ok
    assert any(issue.code == "full_data_uses_public_id" for issue in result.errors)


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


def test_switch_node_dynamic_case_handles_are_valid():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="sw",
                type="SwitchNode",
                data={"value": "{{status}}", "cases": [{"match": "paid", "label": "Paid"}], "default": True},
            ),
            FlowNode(id="paid", type="SendTextMessageNode", data={"messageText": "Paid"}),
            FlowNode(id="other", type="SendTextMessageNode", data={"messageText": "Other"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="sw"),
            FlowEdge(id="e2", source="sw", target="paid", sourceHandle="output-case_0"),
            FlowEdge(id="e3", source="sw", target="other", sourceHandle="output-default"),
        ],
    )

    result = validate(flow)

    assert result.ok, result.render()


def test_training_center_form_flow_is_valid():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="start", type="CommandTriggerNode", data={"command": "/start", "global": True}),
            FlowNode(id="ask-name", type="SendTextMessageNode", data={"messageText": "Ismingizni kiriting"}),
            FlowNode(
                id="name-input",
                type="MessageTriggerNode",
                data={"type": "text", "filter": "any", "global": False, "state_key": "ism", "state_type": "text"},
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
                    "state_key": "telefon",
                    "state_type": "phone_number",
                },
            ),
            FlowNode(id="ask-age", type="SendTextMessageNode", data={"messageText": "Yoshingizni kiriting"}),
            FlowNode(
                id="age-input",
                type="MessageTriggerNode",
                data={"type": "text", "filter": "any", "global": False, "state_key": "yosh", "state_type": "text"},
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


def test_knowledge_tool_node_accepted_on_acc_ai_tool():
    """AIKnowledgeToolNode is a valid source for acc:ai_tool (no acc_wrong_node_type error)."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="agent", type="AIAgentNode", data={}),
            FlowNode(id="prov", type="AIOpenAIProviderNode", data={}),
            FlowNode(
                id="knowledge",
                type="AIKnowledgeToolNode",
                data={"collection": "products"},
            ),
            FlowNode(id="reply", type="SendTextMessageNode", data={"messageText": "{{agent_output}}"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="agent"),
            FlowEdge(id="e2", source="prov", target="agent", sourceHandle="source-default", targetHandle="acc:ai_model"),
            FlowEdge(id="e3", source="knowledge", target="agent", sourceHandle="source-default", targetHandle="acc:ai_tool"),
            FlowEdge(id="e4", source="agent", target="reply"),
        ],
    )
    result = validate(flow)
    # No acc_wrong_node_type error for the knowledge tool
    assert not any(i.code == "acc_wrong_node_type" for i in result.issues)


def test_unknown_node_type_rejected_on_acc_ai_tool():
    """A non-tool node type connected to acc:ai_tool raises acc_wrong_node_type."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="agent", type="AIAgentNode", data={}),
            FlowNode(id="prov", type="AIOpenAIProviderNode", data={}),
            FlowNode(id="wrong", type="SendTextMessageNode", data={"messageText": "bad"}),
            FlowNode(id="reply", type="SendTextMessageNode", data={"messageText": "{{agent_output}}"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="agent"),
            FlowEdge(id="e2", source="prov", target="agent", sourceHandle="source-default", targetHandle="acc:ai_model"),
            FlowEdge(id="e3", source="wrong", target="agent", sourceHandle="source-default", targetHandle="acc:ai_tool"),
            FlowEdge(id="e4", source="agent", target="reply"),
        ],
    )
    result = validate(flow)
    assert any(i.code == "acc_wrong_node_type" for i in result.issues)


def _provider_flow(prov_type: str, prov_data: dict) -> FlowDocument:
    return FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="agent", type="AIAgentNode", data={}),
            FlowNode(id="prov", type=prov_type, data=prov_data),
            FlowNode(id="reply", type="SendTextMessageNode", data={"messageText": "{{agent_output}}"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="agent"),
            FlowEdge(id="e2", source="prov", target="agent", sourceHandle="source-default", targetHandle="acc:ai_model"),
            FlowEdge(id="e3", source="agent", target="reply"),
        ],
    )


def test_ai_provider_without_credential_or_api_key_is_error():
    """A cloud AI provider node with neither credential_id nor apiKey must fail —
    else the LLM call returns empty and the bot sends nothing."""
    flow = _provider_flow(
        "AIOpenAIProviderNode",
        {"model": "gpt-4o-mini", "baseURL": "https://api.openai.com/v1", "apiKey": ""},
    )
    result = validate(flow)
    assert any(i.code == "ai_provider_missing_credential" for i in result.errors)


def test_ai_provider_with_credential_id_passes():
    """The PREFERRED path: a stored credential_id satisfies the requirement."""
    flow = _provider_flow(
        "AIOpenAIProviderNode",
        {"model": "gpt-4o-mini", "credential_id": 7},
    )
    assert not any(i.code == "ai_provider_missing_credential" for i in validate(flow).issues)


def test_ai_provider_with_api_key_passes():
    """Legacy fallback: a real apiKey still satisfies the requirement."""
    flow = _provider_flow(
        "AIOllamaProviderNode",
        {"model": "gemma4:31b-cloud", "baseURL": "https://ollama.com/v1", "apiKey": "sk-xxx"},
    )
    assert not any(i.code == "ai_provider_missing_credential" for i in validate(flow).issues)


def test_local_ollama_without_api_key_is_exempt():
    """Local Ollama (localhost) does not need a key or credential."""
    flow = _provider_flow(
        "AIOllamaProviderNode",
        {"model": "llama3.1", "baseURL": "http://localhost:11434", "apiKey": ""},
    )
    assert not any(i.code == "ai_provider_missing_credential" for i in validate(flow).issues)


def test_ai_provider_with_env_placeholder_key_is_error():
    """An {{env.*}} placeholder is NOT a real key — the user must be asked.
    The validator must reject it just like an empty key (no credential_id either)."""
    flow = _provider_flow(
        "AIOpenAIProviderNode",
        {"model": "gpt-4o-mini", "apiKey": "{{env.OPENAI_KEY}}"},
    )
    assert any(i.code == "ai_provider_missing_credential" for i in validate(flow).errors)


def test_anthropic_provider_without_api_key_is_error():
    """Anthropic has no credential type — only apiKey works; empty must fail."""
    flow = _provider_flow(
        "AIAnthropicProviderNode",
        {"model": "claude-3-5-haiku-latest", "apiKey": ""},
    )
    assert any(i.code == "ai_provider_missing_credential" for i in validate(flow).errors)


def test_state_node_without_type_is_error():
    """A StateNode that saves a value MUST declare `data.type`."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="s", type="StateNode", data={"key": "phone"}),  # no type
        ],
        edges=[FlowEdge(id="e1", source="t", target="s")],
    )
    result = validate(flow)
    assert not result.ok
    assert any(i.code == "state_type_required" for i in result.errors)


def test_state_node_with_blank_type_is_error():
    """An empty/blank `type` is as bad as a missing one."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="s", type="StateNode", data={"key": "phone", "type": "   "}),
        ],
        edges=[FlowEdge(id="e1", source="t", target="s")],
    )
    result = validate(flow)
    assert not result.ok
    assert any(i.code == "state_type_required" for i in result.errors)


def test_state_node_with_type_passes():
    """StateNode with a valid `type` does not raise state_type_required."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="s",
                type="StateNode",
                data={"key": "phone", "type": "contact_phone_number"},
            ),
        ],
        edges=[FlowEdge(id="e1", source="t", target="s")],
    )
    result = validate(flow)
    assert result.ok, result.render()
    assert not any(i.code == "state_type_required" for i in result.issues)


def test_callback_trigger_rejects_message_state_type():
    flow = FlowDocument(
        nodes=[FlowNode(
            id="course-callback", type="CallbackQueryTriggerNode",
            data={
                "filter": "starts_with", "context": {"value": "collection:"},
                "state_key": "selected_course", "state_type": "text",
            },
        )],
        edges=[],
    )
    result = validate(flow)
    assert any(i.code == "incompatible_trigger_state_type" for i in result.errors)


def test_callback_trigger_accepts_full_collection_data():
    flow = FlowDocument(
        nodes=[FlowNode(
            id="course-callback", type="CallbackQueryTriggerNode",
            data={
                "filter": "starts_with", "context": {"value": "collection:"},
                "state_key": "selected_course", "state_type": "full_data",
            },
        )],
        edges=[],
    )
    result = validate(flow)
    assert not any(i.code == "incompatible_trigger_state_type" for i in result.errors)


def test_contact_trigger_rejects_text_capture():
    flow = FlowDocument(
        nodes=[FlowNode(
            id="phone", type="MessageTriggerNode",
            data={
                "type": "contact", "filter": "any", "state_key": "phone",
                "state_type": "text",
            },
        )],
        edges=[],
    )
    result = validate(flow)
    assert any(i.code == "incompatible_trigger_state_type" for i in result.errors)


def test_message_trigger_rejects_unknown_select_value():
    flow = FlowDocument(nodes=[FlowNode(
        id="message", type="MessageTriggerNode",
        data={"type": "Matn", "filter": "any", "global": False},
    )])
    result = validate(flow)
    assert any(i.code == "invalid_select_value" for i in result.errors)


def test_collection_select_must_be_string():
    flow = FlowDocument(nodes=[
        FlowNode(id="start", type="CommandTriggerNode", data={"command": "/start"}),
        FlowNode(
            id="save", type="CollectionNode",
            data={"collection_name": {"name": "courses"}, "fieldMappings": {}},
        ),
    ], edges=[FlowEdge(id="e", source="start", target="save")])
    result = validate(flow)
    assert any(i.code == "invalid_collection_select_value" for i in result.errors)


# ── #8a: global CallbackQueryTrigger suppresses "button has no edge" warning ──


def _menu_node(buttons):
    return FlowNode(
        id="menu",
        type="SendTextMessageNode",
        data={
            "messageText": "Menyu",
            "keyboard": {"active": "inline", "inline": [buttons]},
        },
    )


def test_callback_button_with_global_trigger_no_warning():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            _menu_node([{"text": "Buy", "type": "callback", "value": "buy:1"}]),
            FlowNode(
                id="cb",
                type="CallbackQueryTriggerNode",
                data={"global": True, "filter": "starts_with", "context": {"value": "buy:"}},
            ),
        ],
        edges=[FlowEdge(id="e1", source="t", target="menu")],
    )
    result = validate(flow)
    assert not any(i.code == "missing_button_branch" for i in result.warnings), result.render()


def test_callback_button_without_matching_trigger_warns():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            _menu_node([{"text": "Buy", "type": "callback", "value": "buy:1"}]),
        ],
        edges=[FlowEdge(id="e1", source="t", target="menu")],
    )
    result = validate(flow)
    assert any(i.code == "missing_button_branch" for i in result.warnings)


# ── #8b: url/contact buttons never require an edge ───────────────────────────


def test_url_button_requires_no_edge():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            _menu_node([{"text": "Site", "type": "url", "value": "https://x.uz"}]),
        ],
        edges=[FlowEdge(id="e1", source="t", target="menu")],
    )
    result = validate(flow)
    assert not any(i.code == "missing_button_branch" for i in result.warnings), result.render()


def test_request_contact_button_requires_no_edge():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="menu",
                type="SendTextMessageNode",
                data={
                    "messageText": "Telefon",
                    "keyboard": {"active": "reply",
                                 "reply": [[{"id": "c", "text": "Telefon", "type": "request_contact"}]]},
                },
            ),
        ],
        edges=[FlowEdge(id="e1", source="t", target="menu")],
    )
    result = validate(flow)
    assert not any(i.code == "missing_button_branch" for i in result.warnings), result.render()


def test_plain_reply_keyboard_buttons_use_base_capture_edge():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="ask", type="SendTextMessageNode", data={
                "keyboard_kind": "reply",
                "rows": [{"columns": [{"text": "Bugun"}, {"text": "Ertaga"}]}],
            }),
            FlowNode(id="capture", type="MessageTriggerNode", data={
                "type": "text", "global": False,
                "state_key": "day", "state_type": "text",
            }),
        ],
        edges=[FlowEdge(id="e1", source="ask", target="capture")],
    )
    result = validate(flow)
    assert not any(i.code == "missing_button_branch" for i in result.warnings), result.render()


def test_contact_alias_is_rejected_in_favor_of_request_contact():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="menu",
                type="SendTextMessageNode",
                data={
                    "messageText": "Telefon",
                    "rows": [{"columns": [{"text": "Kontakt", "type": "contact"}]}],
                },
            ),
        ],
        edges=[FlowEdge(id="e1", source="t", target="menu")],
    )
    result = validate(flow)
    assert any(i.code == "invalid_reply_button_type" for i in result.errors), result.render()


# ── #12: collection cross-check ──────────────────────────────────────────────


def test_unknown_collection_blocks_save_when_list_known():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="save", type="CollectionNode",
                     data={"collection_name": "ghosts", "fieldMappings": {"a": "b"}}),
        ],
        edges=[FlowEdge(id="e1", source="t", target="save")],
    )
    result = validate(flow, known_collections=["orders", "users"])
    assert any(i.code == "unknown_collection" for i in result.errors)
    assert not result.ok
    # Known collection → no error.
    result2 = validate(flow, known_collections=["ghosts"])
    assert not any(i.code == "unknown_collection" for i in result2.errors)
    # No list provided → check skipped entirely.
    result3 = validate(flow)
    assert not any(i.code == "unknown_collection" for i in result3.issues)


# ── #8 — set_context('key', ...) in CustomCodeNode jsCode whitelists {{key}} ──


def test_set_context_keys_are_known_variables():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="code",
                type="CustomCodeNode",
                data={"jsCode": "set_context('enroll_id', 42); set_context(\"course\", 'math');"},
            ),
            FlowNode(
                id="m",
                type="SendTextMessageNode",
                data={"messageText": "Ro'yxat: {{enroll_id}} / {{course}}"},
            ),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="code"),
            FlowEdge(id="e2", source="code", target="m"),
        ],
    )
    result = validate(flow)
    # No unknown_variable warning for bare keys written via set_context.
    unknown = [i for i in result.warnings if i.code == "unknown_variable"]
    assert unknown == [], [i.message for i in unknown]


def test_each_block_fields_are_not_treated_as_flow_variables():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(id="m", type="SendTextMessageNode", data={
                "messageText": "{{#each orders}}{{title}} — {{price}}{{/each}}",
            }),
        ],
        edges=[FlowEdge(id="e1", source="t", target="m")],
    )
    result = validate(flow)
    assert not any(i.code == "unknown_variable" for i in result.warnings), result.render()


def test_underscore_state_form_is_flagged():
    """The dead `{{state_key}}` underscore form never resolves in the engine
    (saved state lives under `public_key`), so the validator must surface it as
    a hard validation error instead of silently accepting it — this is the
    guardrail that prevents blank values from being persisted."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="code",
                type="CustomCodeNode",
                data={"jsCode": "set_context('enroll_id', 42);"},
            ),
            FlowNode(
                id="m",
                type="SendTextMessageNode",
                data={"messageText": "Ro'yxat: {{state_enroll_id}}"},
            ),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="code"),
            FlowEdge(id="e2", source="code", target="m"),
        ],
    )
    result = validate(flow)
    legacy = [i for i in result.errors if i.code == "legacy_state_reference"]
    assert any("{{enroll_id}}" in i.message for i in legacy), [
        i.message for i in legacy
    ]


def test_unset_variable_still_warns():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="m",
                type="SendTextMessageNode",
                data={"messageText": "Salom {{never_set}}"},
            ),
        ],
        edges=[FlowEdge(id="e1", source="t", target="m")],
    )
    result = validate(flow)
    assert any(i.code == "unknown_variable" for i in result.warnings)


def _btn_screen(node_id, value, text="📚 Kurslar"):
    return FlowNode(id=node_id, type="EditOrSendMessageNode", data={
        "messageText": "x", "keyboard_kind": "inline",
        "rows": [{"columns": [{"text": text, "type": "callback", "value": value}]}]})


def test_single_screen_button_global_trigger_warns():
    """Bitta ekrandagi tugmani ushlovchi GLOBAL callback trigger → button-edge tavsiya."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="s", type="CommandTriggerNode", data={"command": "/start"}),
            _btn_screen("menu", "courses"),
            FlowNode(id="cb", type="CallbackQueryTriggerNode",
                     data={"global": True, "filter": "equals", "context": {"value": "courses"}}),
        ],
        edges=[FlowEdge(id="e", source="s", target="menu")],
    )
    res = validate(flow)
    assert any(i.code == "global_callback_should_be_button_edge" for i in res.issues)


def test_reusable_back_button_global_trigger_ok():
    """Ko'p ekranda takrorlanadigan (Ortga) tugma → global trigger to'g'ri, ogohlantirma."""
    flow = FlowDocument(
        nodes=[
            FlowNode(id="s", type="CommandTriggerNode", data={"command": "/start"}),
            _btn_screen("scr1", "back", "⬅️ Ortga"),
            _btn_screen("scr2", "back", "⬅️ Ortga"),
            FlowNode(id="cb", type="CallbackQueryTriggerNode",
                     data={"global": True, "filter": "equals", "context": {"value": "back"}}),
        ],
        edges=[FlowEdge(id="e", source="s", target="scr1")],
    )
    res = validate(flow)
    assert not any(i.code == "global_callback_should_be_button_edge" for i in res.issues)


def test_explicit_terminal_output_is_not_reported_as_unused():
    flow = FlowDocument(
        nodes=[
            FlowNode(id="t", type="CommandTriggerNode", data={"command": "/start"}),
            FlowNode(
                id="code",
                type="CustomCodeNode",
                data={
                    "jsCode": "exit('done');",
                    "outputs": [{"name": "done"}, {"name": "next"}],
                    "terminalOutputs": ["done"],
                },
            ),
            FlowNode(id="m", type="SendTextMessageNode", data={"messageText": "ok"}),
        ],
        edges=[
            FlowEdge(id="e1", source="t", target="code"),
            FlowEdge(id="e2", source="code", target="m", sourceHandle="output-next"),
        ],
    )
    result = validate(flow)
    assert not any(i.code == "unused_custom_output" for i in result.warnings), result.render()

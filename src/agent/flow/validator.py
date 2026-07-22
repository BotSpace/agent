"""Flow validation — schema, connectivity, edge handles, variables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .manifest_compiler.loader import load_node_manifests
from .manifest_compiler.node_catalog import (
    NODE_TYPES,
    acc_port_accepts,
    button_edge_types,
    get_node_spec,
    is_trigger,
    state_saving_types,
)
from .schema import FlowDocument, FlowNode


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def render(self) -> str:
        if self.ok and not self.issues:
            return "OK"
        lines = []
        for issue in self.issues:
            loc = ""
            if issue.node_id:
                loc = f" (node {issue.node_id})"
            elif issue.edge_id:
                loc = f" (edge {issue.edge_id})"
            lines.append(f"[{issue.severity}:{issue.code}]{loc} {issue.message}")
        return "\n".join(lines)


VARIABLE_RE = re.compile(r"\{\{\s*[\w\.\[\]]+\s*\}\}|\$[\w\.\[\]]+")

_CALLBACK_STATE_TYPES = {"callback", "data", "collection_id", "full_data", "static"}
_MESSAGE_STATE_TYPES_BY_KIND = {
    "text": {"text", "static"},
    "phone": {"text", "static"},
    "contact": {
        "phone_number",
        "contact_phone_number",
        "contact_first_name",
        "contact_last_name",
        "static",
    },
    "location": {
        "location",
        "latitude",
        "location_latitude",
        "longitude",
        "location_longitude",
        "static",
    },
    "photo": {"file_id", "caption", "static"},
    "video": {"file_id", "caption", "static"},
    "document": {"file_id", "caption", "static"},
    "audio": {"file_id", "caption", "static"},
    "voice": {"file_id", "caption", "static"},
    "video_note": {"file_id", "static"},
    "animation": {"file_id", "caption", "static"},
    "sticker": {"file_id", "static"},
}
# `type:any` can receive text, contact, location or media. Capturing a
# type-specific value from it is unsafe: the trigger may match a different
# update and persist nothing. Use an explicit message type for those values.
_ANY_MESSAGE_STATE_TYPES = {"text", "static"}

# Accessory targetHandle prefix (contract §1.1)
_ACC_PREFIX = "acc:"

# Nodes whose sole job is to persist a value and therefore MUST carry
# `data.type` (the value kind: text / contact_phone_number / *_file_id / user_id /
# Custom / …). Triggers that auto-save through `state_key` + `state_type` are
# intentionally NOT here — their capture type lives in `data.state_type`.
# State-saqlovchi node'lar (data.type majburiy) va accessory port qabul-ro'yxati
# endi MANIFESTDAN derive qilinadi (node_catalog helper'lari) — qotirilgan emas.


def _is_accessory_node(node_type: str) -> bool:
    spec = NODE_TYPES.get(node_type)
    return bool(spec and spec.is_accessory)


def _referenced_collection_name(node) -> str | None:
    """Best-effort extraction of a node's referenced collection name across the
    flat (`collection_name`) and nested (`collection: {name}`) shapes. Returns
    None if no concrete name (templates / `{{vars}}` are ignored)."""
    data = node.data if isinstance(node.data, dict) else {}
    name = data.get("collection_name")
    if not isinstance(name, str) or not name:
        legacy = data.get("collection")
        if isinstance(legacy, dict):
            name = legacy.get("name")
        elif isinstance(legacy, str):
            name = legacy
    if isinstance(name, str) and name and "{{" not in name and "{%" not in name:
        return name
    return None


def _nested_value(data: dict, dotted_key: str):
    value = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def validate(
    flow: FlowDocument,
    *,
    known_collections: list[str] | None = None,
    collection_schemas: dict[str, dict[str, str]] | None = None,
    module_nodes: list[dict] | None = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    # External MODULE node types (namespaced `moduleId.NodeName`), loaded per
    # project at session_use_project. Not in the static catalog, so a node whose
    # type matches one of these must NOT be flagged `unknown_node_type`. Map
    # type → normalized manifest so we can also check the module node's required
    # fields (the engine routes the rest to node.execute).
    module_map: dict[str, dict] = {
        str(m.get("type")): m for m in (module_nodes or []) if isinstance(m, dict) and m.get("type")
    }
    builtin_manifests = load_node_manifests()

    def _node_is_trigger(node_type: str) -> bool:
        """Trigger if a static-catalog trigger OR a module trigger node
        (manifest `is_trigger`). Module triggers count for presence/reachability."""
        if is_trigger(node_type):
            return True
        mod = module_map.get(node_type)
        return bool(mod and mod.get("is_trigger"))

    # #12 — known collection names (from the workspace's loaded CMS list). When
    # provided, a collection-select value must be one of these exact API values.
    # The constructor uses the same value as the Radix SelectItem value; allowing
    # a translated label or an invented name here makes the UI render the select
    # as empty and lets runtime target a collection that does not exist.
    known_set: set[str] | None = (
        {c for c in known_collections if isinstance(c, str) and c} if known_collections is not None else None
    )
    # #4 — {collection_name: {field: kind}} so filter values can be type-checked
    # against the loaded collection schema (warning only).
    schemas: dict[str, dict[str, str]] = collection_schemas or {}

    # ── 1. Node-level checks ───────────────────────────────────────────
    node_ids: set[str] = set()
    for node in flow.nodes:
        if not node.id:
            issues.append(ValidationIssue("error", "node_id_missing", "Node has empty id", node_id=node.id))
            continue
        if node.id in node_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_node_id",
                    f"Duplicate node id `{node.id}`",
                    node_id=node.id,
                )
            )
        node_ids.add(node.id)

        spec = get_node_spec(node.type)
        if spec is None:
            mod = module_map.get(node.type)
            if mod is not None:
                # External module node — validate against its manifest, then skip
                # the static-catalog-specific checks below (handles, state, etc.).
                for required in mod.get("required_fields") or []:
                    if required not in node.data or node.data.get(required) in (None, ""):
                        issues.append(
                            ValidationIssue(
                                "error",
                                "missing_required_field",
                                f"module node `{node.type}` requires `data.{required}`",
                                node_id=node.id,
                            )
                        )
                for cf in mod.get("credential_fields") or []:
                    key = cf.get("key")
                    if key and not node.data.get(key):
                        issues.append(
                            ValidationIssue(
                                "warning",
                                "missing_module_credential",
                                f"module node `{node.type}` field `{key}` has no "
                                f"credential (type_key `{cf.get('type_key')}`)",
                                node_id=node.id,
                            )
                        )
                continue
            issues.append(
                ValidationIssue(
                    "error",
                    "unknown_node_type",
                    f"Unknown node type `{node.type}` (not in catalog or enabled modules)",
                    node_id=node.id,
                )
            )
            continue
        for required in spec.required_data:
            if required not in node.data:
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_required_field",
                        f"`{node.type}` requires `data.{required}`",
                        node_id=node.id,
                    )
                )

        # Validate every manifest-backed select, including nodes changed via
        # flow.patch/import (paths that do not pass through flow.add_node's
        # Pydantic model). Unknown option values render as an empty select in
        # the constructor and often change runtime semantics silently.
        option_sets: dict[str, set[object]] = {}
        for manifest_field in builtin_manifests.get(node.type, {}).get("content") or []:
            if not isinstance(manifest_field, dict):
                continue
            key = manifest_field.get("key")
            if manifest_field.get("type") == "select" and isinstance(key, str):
                if key in {"state_type", "public_type"}:
                    # Trigger state compatibility is runtime-dependent and is
                    # validated by the matrix below; old backend manifests had
                    # stale state options, so do not use their list here.
                    continue
                option_sets.setdefault(key, set()).update(
                    option.get("value")
                    for option in (manifest_field.get("options") or [])
                    if isinstance(option, dict) and "value" in option
                )
            if manifest_field.get("type") == "collection-select" and isinstance(key, str):
                selected = _nested_value(node.data, key)
                if selected is not None and not isinstance(selected, str):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "invalid_collection_select_value",
                            f"`{key}` must be a collection slug/name string, not "
                            f"{type(selected).__name__}. Use a value returned by collections.list.",
                            node_id=node.id,
                        )
                    )
        for key, allowed in option_sets.items():
            selected = _nested_value(node.data, key)
            if selected is not None and selected not in allowed:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_select_value",
                        f"`{node.type}.{key}` value `{selected}` is not a manifest option. "
                        f"Use one of: {', '.join(sorted(map(str, allowed)))}.",
                        node_id=node.id,
                    )
                )

        # A trigger can only persist values that exist on its incoming Telegram
        # update.  The engine intentionally has separate callback and message
        # extraction branches; accepting `text` on a callback silently stores
        # nothing and lets the flow continue with an undefined variable.
        state_key = node.data.get("public_key") or node.data.get("state_key")
        if (
            isinstance(state_key, str)
            and state_key.strip()
            and node.type
            in {
                "CallbackQueryTriggerNode",
                "MessageTriggerNode",
            }
        ):
            state_type = node.data.get("public_type") or node.data.get("state_type")
            if not isinstance(state_type, str) or not state_type.strip():
                issues.append(
                    ValidationIssue(
                        "error",
                        "trigger_state_type_required",
                        f"`{node.type}` saves `{state_key}` but has no state type.",
                        node_id=node.id,
                    )
                )
            else:
                if node.type == "CallbackQueryTriggerNode":
                    allowed = _CALLBACK_STATE_TYPES
                    source = "callback query"
                else:
                    message_kind = str(node.data.get("type") or "any")
                    allowed = _MESSAGE_STATE_TYPES_BY_KIND.get(
                        message_kind,
                        _ANY_MESSAGE_STATE_TYPES,
                    )
                    source = f"{message_kind} message"
                if state_type not in allowed:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "incompatible_trigger_state_type",
                            f"`{state_type}` cannot be extracted from a {source}; "
                            f"use one of: {', '.join(sorted(allowed))}.",
                            node_id=node.id,
                        )
                    )

                if node.type == "CallbackQueryTriggerNode" and state_type in {"collection_id", "full_data"}:
                    context = node.data.get("context")
                    callback_value = context.get("value") if isinstance(context, dict) else None
                    if not (
                        isinstance(callback_value, str)
                        and re.fullmatch(r"collection:[A-Za-z0-9_-]+:", callback_value)
                    ):
                        issues.append(
                            ValidationIssue(
                                "error",
                                "invalid_collection_callback_prefix",
                                f"`{state_type}` only understands canonical callbacks "
                                "`collection:<collection_slug>:<document_id>`. Set this "
                                "trigger to filter='starts_with' and context.value to "
                                "`collection:<collection_slug>:`.",
                                node_id=node.id,
                            )
                        )

        # State-saving nodes MUST declare `data.type` (what kind of value is
        # captured — text/phone_number/file_id/Custom/…). Without it the engine
        # cannot resolve what to store and saves nothing → broken form. The
        # generic check above only catches a fully-absent key; here we also
        # reject an empty/blank `type` and emit an actionable message.
        if node.type in state_saving_types():
            state_type = node.data.get("type")
            if not (isinstance(state_type, str) and state_type.strip()):
                issues.append(
                    ValidationIssue(
                        "error",
                        "state_type_required",
                        f"`{node.type}` `{node.id}` must set `data.type` "
                        "(e.g. 'text', 'contact_phone_number', 'photo_file_id', "
                        "'user_id', 'Custom', …) so the engine knows WHAT value to "
                        "save it. For data arriving via a trigger "
                        "(MessageTrigger / callback), configure the trigger's "
                        "`state_key` + `state_type` fields.",
                        node_id=node.id,
                    )
                )

        # AI provider talking to a CLOUD endpoint MUST have credentials. The
        # PREFERRED path is a stored credential (data.credential_id) — the engine
        # resolves it to api_key/base_url at runtime. apiKey is a legacy fallback
        # (still accepted; required for AIAnthropicProviderNode, which has no
        # credential type). Without either, the engine's LLM call yields empty
        # output → the bot sends nothing → dead bot. Local Ollama is exempt.
        if node.type in acc_port_accepts()["acc:ai_model"]:
            cred_id = node.data.get("credential_id")
            has_credential = isinstance(cred_id, (int, str)) and str(cred_id).strip() not in (
                "",
                "0",
                "none",
                "null",
                "None",
            )
            api_key = str(node.data.get("apiKey") or "").strip()
            base_url = str(node.data.get("baseURL") or "").lower()
            is_local = "localhost" in base_url or "127.0.0.1" in base_url or "://ollama:" in base_url
            # A `{{...}}` template (e.g. {{env.OPENAI_KEY}}) is NOT a real key —
            # the user wants to be ASKED for the key, not have env auto-used.
            # Treat any placeholder as missing so the agent must call ask_user.
            is_placeholder = api_key.startswith("{{") or api_key.startswith("{%")
            has_api_key = bool(api_key) and not is_placeholder
            # Anthropic has no credential type — it can ONLY use apiKey.
            supports_credential = node.type != "AIAnthropicProviderNode"
            if not has_credential and not has_api_key and not is_local:
                if supports_credential:
                    hint = (
                        "Set data.credential_id to a stored credential id "
                        "(credentials.list(type_key=...) — openai→'openai', "
                        "ollama→'ollama', gemini→'ai_studio'); if the user has none, "
                        "they must create one in the panel/dashboard first. (Legacy: a "
                        "real data.apiKey also works, but NOT a {{...}} placeholder.)"
                    )
                else:
                    hint = (
                        "AIAnthropicProviderNode has no credential type — ASK THE USER "
                        "(ask_user) and set data.apiKey to the real value; do NOT use "
                        "{{env.*}} or any placeholder."
                    )
                issues.append(
                    ValidationIssue(
                        "error",
                        "ai_provider_missing_credential",
                        f"`{node.type}` `{node.id}` has no credential_id and no real "
                        "apiKey"
                        + (" (a `{{...}}` placeholder is not allowed)" if is_placeholder else "")
                        + f". A cloud LLM provider needs credentials. {hint} "
                        "Otherwise the AI node returns empty output and the bot sends nothing.",
                        node_id=node.id,
                    )
                )

        # #12 — cross-check referenced collection names against the known list.
        if known_set is not None:
            ref = _referenced_collection_name(node)
            if ref and ref not in known_set:
                issues.append(
                    ValidationIssue(
                        "error",
                        "unknown_collection",
                        f"`{node.type}` `{node.id}` references collection `{ref}` which is "
                        f"not in this project's collections ({sorted(known_set) or '—'}). "
                        "Use one of those exact values (not the UI label), or create it "
                        "with cms.* first and reload the project collection list.",
                        node_id=node.id,
                    )
                )

        # #4 — filter value type vs loaded collection field kind (warning).
        if schemas:
            issues.extend(_filter_kind_warnings(node, schemas))

        # #9a — CustomCodeNode declares output handles (data.outputs) that no
        # edge uses. exit('NAME') without a wired edge is a dead branch.
        if node.type == "CustomCodeNode":
            js_code = node.data.get("jsCode")
            if isinstance(js_code, str) and (
                re.search(r"\bawait\b", js_code) or re.search(r"\basync\s+(?:function|\()", js_code)
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "custom_code_async_unsupported",
                        "CustomCode runs synchronously in Goja: do not use async/await. "
                        "Call db_find/db_insert/send_message directly and use the returned value.",
                        node_id=node.id,
                    )
                )
            issues.extend(_unused_output_warnings(node, flow))

    # ── 2. Edge-level checks ───────────────────────────────────────────
    nodes_by_id = {node.id: node for node in flow.nodes}
    edge_pairs: set[tuple[str, str, str | None]] = set()
    # Track acc:ai_model connections per AIAgentNode (must be exactly 1)
    ai_model_connections: dict[str, int] = {}

    for edge in flow.edges:
        if not edge.id:
            issues.append(ValidationIssue("error", "edge_id_missing", "Edge has empty id", edge_id=edge.id))
        if edge.source not in node_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "edge_source_missing",
                    f"Edge `{edge.id}` source `{edge.source}` does not exist",
                    edge_id=edge.id,
                )
            )
        if edge.target not in node_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "edge_target_missing",
                    f"Edge `{edge.id}` target `{edge.target}` does not exist",
                    edge_id=edge.id,
                )
            )

        target_handle = edge.targetHandle or ""
        is_acc_edge = target_handle.startswith(_ACC_PREFIX)

        if is_acc_edge:
            # Validate that the accessory edge connects accepted node types.
            source_node = nodes_by_id.get(edge.source)
            target_node = nodes_by_id.get(edge.target)
            accepted = acc_port_accepts().get(target_handle)
            if accepted is not None and source_node is not None:
                if source_node.type not in accepted:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "acc_wrong_node_type",
                            (
                                f"Accessory edge `{edge.id}`: source `{source_node.type}` is not "
                                f"accepted on port `{target_handle}`. Accepted: {sorted(accepted)}."
                            ),
                            edge_id=edge.id,
                        )
                    )
            # Track acc:ai_model multiplicity (max 1 per AIAgentNode)
            if target_handle == "acc:ai_model" and target_node is not None:
                ai_model_connections[target_node.id] = ai_model_connections.get(target_node.id, 0) + 1
            # Accessory source nodes must use source-default
            if edge.sourceHandle not in (None, "source-default"):
                issues.append(
                    ValidationIssue(
                        "warning",
                        "acc_wrong_source_handle",
                        (
                            f"Accessory edge `{edge.id}` should use sourceHandle='source-default', "
                            f"got '{edge.sourceHandle}'."
                        ),
                        edge_id=edge.id,
                    )
                )
        else:
            # Only validate source handles for non-accessory edges.
            source_node = nodes_by_id.get(edge.source)
            if source_node is not None:
                unsupported = _unsupported_source_handle(source_node, edge.sourceHandle, module_map)
                if unsupported is not None:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "unsupported_source_handle",
                            unsupported,
                            node_id=source_node.id,
                            edge_id=edge.id,
                        )
                    )

        key = (edge.source, edge.target, edge.sourceHandle)
        if key in edge_pairs:
            issues.append(
                ValidationIssue(
                    "warning",
                    "duplicate_edge",
                    f"Duplicate edge `{edge.source}` → `{edge.target}` on `{edge.sourceHandle}`",
                    edge_id=edge.id,
                )
            )
        edge_pairs.add(key)

    # acc:ai_model multiplicity: AIAgentNode should have exactly one provider.
    for node in flow.nodes:
        if node.type == "AIAgentNode":
            count = ai_model_connections.get(node.id, 0)
            if count == 0:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "ai_agent_no_provider",
                        f"AIAgentNode `{node.id}` has no provider attached on acc:ai_model. "
                        "Add one provider sub-node (e.g. AIOpenAIProviderNode).",
                        node_id=node.id,
                    )
                )
            elif count > 1:
                issues.append(
                    ValidationIssue(
                        "error",
                        "ai_agent_multiple_providers",
                        f"AIAgentNode `{node.id}` has {count} providers on acc:ai_model (max 1).",
                        node_id=node.id,
                    )
                )

    # ── 3. Trigger presence ────────────────────────────────────────────
    trigger_nodes = [n for n in flow.nodes if _node_is_trigger(n.type)]
    if flow.nodes and not trigger_nodes:
        issues.append(
            ValidationIssue("warning", "no_trigger", "Flow has no trigger node — it can never start")
        )

    # ── 4. Reachability (BFS from triggers) ────────────────────────────
    # Accessory nodes are intentionally not reachable from triggers — exclude from warning.
    reachable: set[str] = set()
    queue = [t.id for t in trigger_nodes]
    while queue:
        nid = queue.pop(0)
        if nid in reachable:
            continue
        reachable.add(nid)
        for e in flow.edges:
            # Do not follow accessory edges during BFS (they are not flow connections).
            if e.source == nid and e.target not in reachable:
                if not (e.targetHandle or "").startswith(_ACC_PREFIX):
                    queue.append(e.target)
    for node in flow.nodes:
        if node.id not in reachable and not _node_is_trigger(node.type):
            # Accessory nodes are never reachable from triggers by design — skip warning.
            if _is_accessory_node(node.type):
                continue
            issues.append(
                ValidationIssue(
                    "warning",
                    "unreachable",
                    f"Node `{node.id}` ({node.type}) is unreachable from any trigger",
                    node_id=node.id,
                )
            )

    # ── 5. Required source handles for branching nodes ─────────────────
    branching_handles = {
        "IfConditionNode": {"true", "false"},
        "LoadCollectionItemNode": {"found", "not_found"},
        "CheckMembershipNode": {"is-member", "not-member"},
        "ForLoopNode": {"loop-body", "no-items"},
        "PaymentNode": {"paid", "failed"},
    }
    edges_by_source: dict[str, set[str]] = {}
    for e in flow.edges:
        edges_by_source.setdefault(e.source, set()).add(e.sourceHandle or "source-handler")
    # #8a — global CallbackQueryTriggerNodes that can handle a callback button's
    # value (equals/starts_with/contains/ends_with). If one matches a button,
    # that button does NOT need its own outgoing edge.
    global_callback_matchers = _global_callback_matchers(flow)
    for node in flow.nodes:
        present = edges_by_source.get(node.id, set())
        for _handle, button in _button_source_handles_with_buttons(node):
            button_type = str(button.get("type") or "").strip().lower()
            if button_type in {"contact", "location"}:
                canonical = "request_contact" if button_type == "contact" else "request_location"
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_reply_button_type",
                        f"`{node.id}` button type `{button_type}` must be `{canonical}` "
                        "to request Telegram native data.",
                        node_id=node.id,
                    )
                )
        if node.type in branching_handles:
            for handle in branching_handles[node.type]:
                if handle not in present:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "missing_branch",
                            f"`{node.type}` `{node.id}` has no edge on `{handle}` handle",
                            node_id=node.id,
                        )
                    )
        for handle, button in _missing_button_handles(node, present):
            # #8b — url/contact/location buttons never route via an edge.
            if not _button_needs_edge(button, keyboard_kind=node.data.get("keyboard_kind")):
                continue
            # #8a — a matching global CallbackQueryTriggerNode handles it.
            if _callback_handled_globally(button, global_callback_matchers):
                continue
            issues.append(
                ValidationIssue(
                    "warning",
                    "missing_button_branch",
                    f"`{node.id}` button has no edge on `{handle}` handle",
                    node_id=node.id,
                )
            )

    # ── 6. Unknown {{var}} references (#9b, conservative) ──────────────
    issues.extend(_collection_mapping_issues(flow))
    issues.extend(_button_template_issues(flow))
    issues.extend(_collection_data_issues(flow))
    issues.extend(_unknown_variable_warnings(flow, module_map))

    # ── 7. Screen-specific button → global trigger over-use ────────────
    # AI tez-tez har inline tugma uchun root GLOBAL CallbackQueryTriggerNode
    # yaratadi; ekranga xos (bitta node'dagi) tugma button-edge bilan ulanishi
    # kerak. Faqat ko'p ekranda takrorlanadigan (Ortga/menyu) tugma global bo'lsin.
    issues.extend(_screen_specific_global_callback_warnings(flow))

    return ValidationResult(ok=not any(i.severity == "error" for i in issues), issues=issues)


def _screen_specific_global_callback_warnings(flow: FlowDocument) -> list[ValidationIssue]:
    """Global CallbackQueryTriggerNode FAQAT BITTA ekrandagi statik tugmaga mos
    kelsa — button-edge afzal (tugma o'sha node'ga bog'liq). Ko'p ekranda
    takrorlanadigan (Ortga/menyu) yoki dinamik tugma uchun global to'g'ri."""
    # callback value → o'sha qiymatli tugma turgan node id'lari (statik tugmalar)
    value_nodes: dict[str, set[str]] = {}
    value_text: dict[str, str] = {}
    for node in flow.nodes:
        for _handle, button in _button_source_handles_with_buttons(node):
            btype = str(button.get("type") or "").strip().lower()
            if btype not in ("callback", ""):
                continue
            val = str(button.get("value") or button.get("text") or "")
            if not val:
                continue
            value_nodes.setdefault(val, set()).add(node.id)
            value_text[val] = str(button.get("text") or val)

    out: list[ValidationIssue] = []
    incoming = {edge.target for edge in flow.edges}
    for n in flow.nodes:
        if n.type != "CallbackQueryTriggerNode":
            continue
        # Engine checks root triggers for every update too. A root callback is
        # therefore global in behaviour even if an older flow omitted `global`.
        if not bool(n.data.get("global")) and n.id in incoming:
            continue
        if str(n.data.get("filter") or "").strip().lower() != "equals":
            continue  # faqat aniq-mos (bitta tugma); starts_with = dinamik, tegmaymiz
        ctx = n.data.get("context")
        val = str(ctx.get("value") or "") if isinstance(ctx, dict) else ""
        owners = value_nodes.get(val)
        if owners and len(owners) == 1:
            owner = next(iter(owners))
            label = value_text.get(val, val)
            out.append(
                ValidationIssue(
                    "warning",
                    "global_callback_should_be_button_edge",
                    f"global CallbackQueryTriggerNode `{n.id}` `{label}` tugmasini "
                    f"ushlaydi, lekin u FAQAT bitta ekranda (node `{owner}`). Root "
                    f"global trigger O'RNIGA button-edge ishlat: "
                    f"flow.connect_button(source='{owner}', target=<keyingi-node>, "
                    f"button_text='{label}'). Global trigger faqat KO'P ekranda "
                    f"takrorlanadigan (⬅️Ortga/🏠Menyu) yoki dinamik tugma uchun.",
                    node_id=n.id,
                )
            )
    return out


# ── #4: filter value type vs collection field kind ───────────────────────────


# Map FieldKind → a predicate that accepts a concrete (non-template) filter value.
def _kind_accepts(kind: str, value) -> bool:
    kind = (kind or "").lower()
    if isinstance(value, str) and ("{{" in value or "{%" in value):
        return True  # template — can't type-check
    if kind == "bool":
        return isinstance(value, bool)
    if kind in ("int",):
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in ("float",):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    # text/email/url/phone/enum/datetime/etc — accept strings (and most scalars).
    return True


def _iter_filter_conditions(node):
    """Yield (field_name, value) pairs from the various filter shapes a node may
    carry: LoadCollectionItem `filters:[{field, value|stateKey}]`, and the flat
    `filter:{field: value}` / `where:{field: value}` Mongo-style maps."""
    data = node.data if isinstance(node.data, dict) else {}
    filters = data.get("filters")
    if isinstance(filters, list):
        for f in filters:
            if not isinstance(f, dict):
                continue
            field = f.get("field")
            # A state-sourced value is resolved at runtime — skip type-check.
            if f.get("valueSource") == "state" or "stateKey" in f:
                continue
            if isinstance(field, str) and field and "value" in f:
                yield field, f.get("value")
    for key in ("filter", "where"):
        flat = data.get(key)
        coll_data = data.get("collectionData")
        if isinstance(coll_data, dict) and isinstance(coll_data.get(key), dict):
            flat = coll_data.get(key)
        if isinstance(flat, dict):
            for fld, value in flat.items():
                if not isinstance(fld, str) or fld.startswith("$"):
                    continue
                # Skip Mongo operator objects ({$gte: ...}) — too ambiguous.
                if isinstance(value, dict):
                    continue
                yield fld, value


def _filter_kind_warnings(node, schemas: dict[str, dict[str, str]]) -> list[ValidationIssue]:
    name = _referenced_collection_name(node)
    if not name or name not in schemas:
        return []
    kinds = schemas[name]
    if not kinds:
        return []
    out: list[ValidationIssue] = []
    for fld, value in _iter_filter_conditions(node):
        kind = kinds.get(fld)
        if not kind:
            continue
        if not _kind_accepts(kind, value):
            out.append(
                ValidationIssue(
                    "warning",
                    "filter_kind_mismatch",
                    f"`{node.type}` `{node.id}` filters `{name}.{fld}` (kind={kind}) "
                    f"with value {value!r} of an incompatible type. "
                    "Check the filter value type.",
                    node_id=node.id,
                )
            )
    return out


# ── #9a: CustomCode declared outputs not wired by any edge ────────────────────


def _unused_output_warnings(node, flow: FlowDocument) -> list[ValidationIssue]:
    data = node.data if isinstance(node.data, dict) else {}
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        return []
    declared: list[str] = [
        out["name"]
        for out in outputs
        if isinstance(out, dict) and isinstance(out.get("name"), str) and out["name"]
    ]
    if not declared:
        return []
    terminal_outputs = {
        str(name) for name in data.get("terminalOutputs") or [] if isinstance(name, str) and name
    }

    # #7 — terminal CustomCodeNode: a node that produces user-facing output
    # itself (send_*/edit_* in its jsCode) and then exit()s is a legitimate
    # leaf — it has NO downstream node by design, so its declared outputs being
    # unwired is NOT a bug. Two ways to mark this so we don't emit a misleading
    # "dead-end" warning:
    #   1. data.terminal == true  (explicit, preferred), OR
    #   2. heuristic: jsCode both sends a message AND calls exit(...).
    if data.get("terminal") is True:
        return []
    used = {e.sourceHandle for e in flow.edges if e.source == node.id and isinstance(e.sourceHandle, str)}
    declared_handles = {f"output-{name}" for name in declared}
    js = data.get("jsCode")
    if isinstance(js, str):
        sends = any(
            tok in js
            for tok in (
                "send_message",
                "edit_or_send",
                "edit_message_text",
                "send_photo",
                "answer_callback_query",
            )
        )
        exits = "exit(" in js
        if sends and exits and not (used & declared_handles):
            # Treat as a terminal leaf — surface at low severity as a tip, not a
            # warning, so it doesn't look like a flow error.
            return [
                ValidationIssue(
                    "info",
                    "terminal_custom_output",
                    f"CustomCodeNode `{node.id}` looks terminal (it sends output "
                    "then exits). Its declared outputs have no downstream edges — "
                    "that is fine for a leaf. To silence this, set "
                    "`data.terminal: true` (or drop the unused outputs).",
                    node_id=node.id,
                )
            ]

    out: list[ValidationIssue] = []
    for name in declared:
        if name in terminal_outputs:
            continue
        handle = f"output-{name}"
        if handle not in used:
            out.append(
                ValidationIssue(
                    "warning",
                    "unused_custom_output",
                    f"CustomCodeNode `{node.id}` declares output `{name}` "
                    f"(handle `{handle}`) but no edge uses it — exit('{name}') "
                    "would dead-end. Wire an edge, set data.terminal:true if this "
                    "is a leaf, or drop the output.",
                    node_id=node.id,
                )
            )
    return out


# ── #9b: unknown {{var}} references (conservative) ────────────────────────────

# Standard runtime context keys always available to templates.
_STANDARD_CONTEXT_VARS = frozenset(
    {
        "first_name",
        "last_name",
        "username",
        "user_id",
        "language_code",
        "chat_id",
        "message_id",
        "message_text",
        "text",
        "phone",
        "phone_number",
        "latitude",
        "longitude",
        "inserted_id",
        "agent_output",
        "callback_data",
        "now",
        "today",
        "bot_username",
        "user_first_name",
        "user_last_name",
        "user_username",
    }
)

# Prefixes/roots that are always allowed (namespaced runtime data).
# Variable model: `public.`/`vars.`/`state.` → public_<key> (legacy `state.` kept
# for back-compat); `private.` → private_<key>; `callback_query.` → raw callback
# query fields; `user.` → user_<key>.
_ALLOWED_VAR_PREFIXES = (
    "state.",
    "public.",
    "private.",
    "vars.",
    "env.",
    "collection_",
    "item.",
    "doc.",
    "loop.",
    "response.",
    "http_",
    "context.",
    "callback_query.",
    "user.",
)

# Only scan these node fields for {{var}} usage — text/code carrying templates.
_TEMPLATE_TEXT_FIELDS = ("messageText", "text", "caption")

# `{{ var }}` with a simple root token (letters/digits/_), optional .path / [i].
_SIMPLE_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][\w]*)\b[^}]*\}\}")
_EACH_BLOCK_RE = re.compile(r"\{\{#each\b[^}]*\}\}.*?\{\{/each\}\}", re.DOTALL)

# #8 — CustomCodeNode jsCode writes vars via set_context('key', ...) /
# set_state('key', ...) / vars.key = ... / private.key = ... . Capture every
# literal key it sets so {{key}} / {{public_key}} don't warn.
# Matches single- or double-quoted first args of set_context/set_state, and
# object-access assignments vars.x = / private.x = .
_SET_CONTEXT_RE = re.compile(r"""(?:set_context|set_state)\(\s*['"]([A-Za-z_]\w*)['"]""")
_OBJ_ASSIGN_RE = re.compile(r"""\b(?:vars|private|state)\.([A-Za-z_]\w*)\s*=""")


def _collect_state_keys(flow: FlowDocument) -> set[str]:
    """Best-effort set of variable keys a flow writes (so they're known to {{var}}).

    Each written key is registered under every prefix the engine can expose it as:
    the bare key (preferred `{{key}}` form) plus `public_<key>` / `private_<key>`.
    The underscore `state_<key>` form is intentionally NOT registered — the engine
    never resolves it (saved state lives under `public_<key>`), so leaving it out
    lets `{{state_key}}` surface as an `unknown_variable` warning instead of being
    silently accepted.
    """
    keys: set[str] = set()

    def _register(key: str) -> None:
        if not key:
            return
        keys.add(key)
        keys.add(f"public_{key}")
        keys.add(f"private_{key}")

    for n in flow.nodes:
        data = n.data if isinstance(n.data, dict) else {}
        # StateNode / capture: data.key
        k = data.get("key")
        if isinstance(k, str) and k:
            _register(k)
        # Canonical flat trigger capture fields.
        for field_name in ("public_key", "state_key"):
            flat_key = data.get(field_name)
            if isinstance(flat_key, str) and flat_key:
                _register(flat_key)
        # Backward-compatible nested trigger capture shape.
        st = data.get("state")
        if isinstance(st, dict) and isinstance(st.get("key"), str):
            _register(st["key"])
        # HTTPRequest saveAs / Download saveAs / Google* responseVariable
        for f in ("saveAs", "contextKey", "outputKey", "responseVariable"):
            v = data.get(f)
            if isinstance(v, str) and v:
                _register(v)
                if n.type == "LoadCollectionListNode" and f == "contextKey":
                    _register(f"{v}_found")
                    _register(f"{v}_count")
        if n.type == "LoadCollectionListNode":
            result_key = data.get("resultKey")
            if isinstance(result_key, str) and result_key:
                _register(result_key)
                _register(f"{result_key}_found")
                _register(f"{result_key}_count")
        # CreateInviteLinkNode writes the generated link to data.resultVar
        # (default `invite_link`) — so {{invite_link}} downstream is valid.
        if n.type == "CreateInviteLinkNode":
            rv = data.get("resultVar")
            _register(rv if isinstance(rv, str) and rv else "invite_link")
        # CollectionNode.fieldMappings CONSUME state; they do not create it.
        # Registering mapping values here used to make a missing source validate
        # itself as "known", hiding literal values such as user_id in Mongo.
        # #8 — CustomCodeNode jsCode sets vars via set_context/set_state or
        # vars.x = / private.x = . Each written key is valid downstream as
        # {{key}} / {{public_key}} / {{private_key}}.
        if n.type == "CustomCodeNode":
            code = data.get("jsCode")
            if isinstance(code, str) and code:
                for m in _SET_CONTEXT_RE.finditer(code):
                    _register(m.group(1))
                for m in _OBJ_ASSIGN_RE.finditer(code):
                    _register(m.group(1))
    return keys


def _collection_mapping_issues(flow: FlowDocument) -> list[ValidationIssue]:
    """Check CollectionNode mapping sources without pretending they produce state."""
    state_keys = _collect_state_keys(flow)
    allowed = state_keys | set(_STANDARD_CONTEXT_VARS)
    out: list[ValidationIssue] = []
    for node in flow.nodes:
        if node.type != "CollectionNode":
            continue
        mappings = node.data.get("fieldMappings")
        if not isinstance(mappings, dict):
            continue
        for field_name, source in mappings.items():
            if not isinstance(source, str) or not source.strip():
                out.append(
                    ValidationIssue(
                        "error",
                        "invalid_collection_mapping",
                        f"Collection field `{field_name}` has an empty/non-string mapping.",
                        node_id=node.id,
                    )
                )
                continue
            template_vars = list(_SIMPLE_VAR_RE.finditer(source))
            for match in template_vars:
                root = match.group(1)
                namespaced = any(
                    (root == prefix[:-1] if prefix.endswith(".") else root.startswith(prefix))
                    for prefix in _ALLOWED_VAR_PREFIXES
                )
                if root not in allowed and not namespaced:
                    out.append(
                        ValidationIssue(
                            "error",
                            "unknown_collection_mapping_variable",
                            f"Collection field `{field_name}` maps from unknown variable "
                            f"`{{{{{root}}}}}`. Capture it before this node or fix the name.",
                            node_id=node.id,
                        )
                    )
            if not template_vars and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", source
            ):
                root = source.split(".", 1)[0]
                if root in allowed:
                    out.append(
                        ValidationIssue(
                            "error",
                            "unwrapped_collection_mapping_source",
                            f"Collection field `{field_name}` maps from nested state "
                            f"`{source}` as literal text. Use `{{{{{source}}}}}` so the "
                            "runtime resolves the nested value.",
                            node_id=node.id,
                        )
                    )
            # Per tool contract, an identifier-shaped bare value is normally a
            # state key. Keep constants backward-compatible, but surface likely
            # typos instead of silently certifying them as valid.
            if not template_vars and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source):
                if source not in allowed:
                    out.append(
                        ValidationIssue(
                            "warning",
                            "unresolved_collection_mapping_source",
                            f"Collection field `{field_name}` maps from `{source}`, but no "
                            "preceding state/runtime variable with that name is known. The "
                            "engine will store it as literal text.",
                            node_id=node.id,
                        )
                    )
    return out


def _button_template_issues(flow: FlowDocument) -> list[ValidationIssue]:
    """Reject row-local templates in static keyboard definitions."""
    state_keys = _collect_state_keys(flow)
    allowed = state_keys | set(_STANDARD_CONTEXT_VARS)
    out: list[ValidationIssue] = []

    def scan_rows(node: FlowNode, rows: object) -> None:
        def walk(value: object) -> None:
            if isinstance(value, list):
                for item in value:
                    walk(item)
                return
            if not isinstance(value, dict):
                return

            # collectionData is the one keyboard subtree where row-document
            # templates ({{name}}, callback {id}, …) are intentionally valid.
            for field_name in ("text", "value"):
                field_value = value.get(field_name)
                if not isinstance(field_value, str) or "{{" not in field_value:
                    continue
                roots = {match.group(1) for match in _SIMPLE_VAR_RE.finditer(field_value)}
                unsupported = sorted(root for root in roots if root not in allowed)
                if "{{#each" in field_value or unsupported:
                    bad = ", ".join(unsupported) or "#each"
                    out.append(
                        ValidationIssue(
                            "error",
                            "dynamic_keyboard_template_unsupported",
                            f"Static keyboard `{field_name}` uses row-local template "
                            f"`{bad}`. Keyboard rows do not iterate collection data. "
                            "Use rows[].columns[].collectionData, build rows in synchronous "
                            "CustomCode, or use flow.ask_buttons with concrete IDs.",
                            node_id=node.id,
                        )
                    )
            for key, child in value.items():
                if key != "collectionData":
                    walk(child)

        walk(rows)

    for node in flow.nodes:
        data = node.data if isinstance(node.data, dict) else {}
        scan_rows(node, data.get("rows"))
        keyboard = data.get("keyboard")
        if isinstance(keyboard, dict):
            scan_rows(node, keyboard.get("inline"))
            scan_rows(node, keyboard.get("reply"))
    return out


def _collection_data_issues(flow: FlowDocument) -> list[ValidationIssue]:
    """Validate runtime collection-backed keyboard definitions."""
    out: list[ValidationIssue] = []
    callback_prefixes: dict[str, str] = {}

    def walk(node: FlowNode, value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(node, item)
            return
        if not isinstance(value, dict):
            return
        collection_data = value.get("collectionData")
        if isinstance(collection_data, dict):
            collection = collection_data.get("collection")
            field = collection_data.get("field")
            text = collection_data.get("text")
            pattern = collection_data.get("callbackDataPattern")
            has_template = isinstance(text, str) and bool(_SIMPLE_VAR_RE.search(text))
            if not isinstance(collection, str) or not collection.strip():
                out.append(
                    ValidationIssue(
                        "error",
                        "collection_data_collection_required",
                        "collectionData.collection must be a CMS collection slug.",
                        node_id=node.id,
                    )
                )
            if not (isinstance(field, str) and field.strip()) and not has_template:
                out.append(
                    ValidationIssue(
                        "error",
                        "invalid_collection_data_text",
                        "collectionData needs `field` or a double-brace text template "
                        "such as `{{name}} — {{price}}`. Single braces like `{name}` "
                        "are not resolved by the engine.",
                        node_id=node.id,
                    )
                )
            if isinstance(pattern, str) and pattern and isinstance(collection, str):
                expected = f"collection:{collection}:{{id}}"
                if pattern != expected:
                    out.append(
                        ValidationIssue(
                            "error",
                            "invalid_collection_data_callback",
                            f"collectionData callback must be `{expected}` (or omitted "
                            "to use the engine default), so collection_id/full_data can resolve it.",
                            node_id=node.id,
                        )
                    )
                else:
                    callback_prefixes[f"collection:{collection}:"] = node.id
        for child in value.values():
            walk(node, child)

    matched_prefixes: set[str] = set()
    for node in flow.nodes:
        data = node.data if isinstance(node.data, dict) else {}
        walk(node, data.get("rows"))
        walk(node, data.get("keyboard"))
    for node in flow.nodes:
        if node.type != "CallbackQueryTriggerNode":
            continue
        data = node.data if isinstance(node.data, dict) else {}
        context = data.get("context")
        prefix = context.get("value") if isinstance(context, dict) else None
        if prefix not in callback_prefixes:
            continue
        matched_prefixes.add(prefix)
        if not bool(data.get("global")):
            out.append(
                ValidationIssue(
                    "error",
                    "collection_data_trigger_must_be_global",
                    f"collectionData creates `{prefix}<id>` buttons at runtime, so the "
                    "matching CallbackQueryTriggerNode must set global=true; there is no "
                    "static button edge to activate a non-global trigger.",
                    node_id=node.id,
                )
            )
    for prefix, source_node_id in callback_prefixes.items():
        if prefix not in matched_prefixes:
            out.append(
                ValidationIssue(
                    "error",
                    "collection_data_trigger_missing",
                    f"collectionData emits `{prefix}<id>` but no matching global "
                    "CallbackQueryTriggerNode captures that prefix.",
                    node_id=source_node_id,
                )
            )
    return out


def _unknown_variable_warnings(flow: FlowDocument, module_map: dict | None = None) -> list[ValidationIssue]:
    state_keys = _collect_state_keys(flow)
    # Module nodes write their own state vars (manifest `produces_state`, e.g.
    # youtube.Download → yt_file/yt_title). Register those (and their public_
    # variants) so a downstream `{{yt_title}}` isn't flagged as unknown.
    for mod in (module_map or {}).values():
        for key in mod.get("produces_state") or []:
            if isinstance(key, str) and key:
                state_keys.add(key)
                state_keys.add(f"public_{key}")
    allowed_roots = state_keys | set(_STANDARD_CONTEXT_VARS)
    out: list[ValidationIssue] = []
    for node in flow.nodes:
        data = node.data if isinstance(node.data, dict) else {}
        for fld in _TEMPLATE_TEXT_FIELDS:
            text = data.get(fld)
            if not isinstance(text, str) or "{{" not in text:
                continue
            # Bare names inside an each block are document fields of the
            # current row, not flow variables (e.g. {{title}}, {{price}}).
            text = _EACH_BLOCK_RE.sub("", text)
            for m in _SIMPLE_VAR_RE.finditer(text):
                root = m.group(1)
                whole = m.group(0)
                inner = whole.strip("{} ").strip()
                if root in allowed_roots:
                    continue
                # Trigger/State values are stored as public_<key> and exposed to
                # templates as {{key}}. The historical {{state_key}} spelling is
                # silently rendered as an empty string by the engine. Treat this
                # as a hard error when we can identify the real written key: it
                # causes irreversible blank CMS fields and must never be dismissed
                # as a harmless validator warning.
                if root.startswith("state_") and root.removeprefix("state_") in state_keys:
                    preferred = root.removeprefix("state_")
                    out.append(
                        ValidationIssue(
                            "error",
                            "legacy_state_reference",
                            f"`{node.id}` uses dead variable `{{{{{root}}}}}`. "
                            f"Use `{{{{{preferred}}}}}`; otherwise runtime writes an empty value.",
                            node_id=node.id,
                        )
                    )
                    continue
                if any(inner.startswith(p) or root + "." == p for p in _ALLOWED_VAR_PREFIXES):
                    continue
                if any(inner.startswith(p) for p in _ALLOWED_VAR_PREFIXES):
                    continue
                # Conservative: only warn on a bare unknown root (no namespace).
                if "." in inner or "[" in inner:
                    continue
                out.append(
                    ValidationIssue(
                        "warning",
                        "unknown_variable",
                        f"`{node.id}` references `{{{{{root}}}}}` which is not a known "
                        "state key, standard context var (first_name, user_id, …) or "
                        "collection_* value. Check the variable name.",
                        node_id=node.id,
                    )
                )
    return out


# Button `type` values that route via a flow edge and therefore can warn when
# unconnected. url / request_contact / request_location buttons perform a
# Telegram-native action and never need an outgoing edge.
_EDGE_BUTTON_TYPES = frozenset({"callback", "reply", ""})
_NO_EDGE_BUTTON_TYPES = frozenset(
    {
        "url",
        "request_contact",
        "request_location",
        "switch_inline",
        "login",
        "web_app",
        "pay",
    }
)


def _button_needs_edge(button: dict | None, *, keyboard_kind: object = None) -> bool:
    if not isinstance(button, dict):
        return True
    btype = str(button.get("type") or "").strip().lower()
    # Plain buttons in a reply keyboard send their text as the next message;
    # the base edge to a MessageTriggerNode captures it. They are not callback
    # branches and must not each have a button edge.
    if str(keyboard_kind or "").strip().lower() == "reply" and btype in ("", "reply"):
        return False
    if btype in _NO_EDGE_BUTTON_TYPES:
        return False
    return True


def _global_callback_matchers(flow: FlowDocument) -> list[tuple[str, str]]:
    """Return (filter, value) pairs for every GLOBAL CallbackQueryTriggerNode.

    A global callback trigger with global=true acts as a catch-all router for
    matching callback_data anywhere in the flow, so callback buttons whose value
    it matches do not each need their own edge.
    """
    matchers: list[tuple[str, str]] = []
    incoming = {edge.target for edge in flow.edges}
    for n in flow.nodes:
        if n.type != "CallbackQueryTriggerNode":
            continue
        # The engine evaluates root callback triggers globally even when the
        # serialized node does not explicitly contain `global: true`.
        if not bool(n.data.get("global")) and n.id in incoming:
            continue
        flt = str(n.data.get("filter") or "any").strip().lower()
        ctx = n.data.get("context")
        value = ""
        if isinstance(ctx, dict):
            value = str(ctx.get("value") or "")
        matchers.append((flt, value))
    return matchers


def _callback_handled_globally(button: dict | None, matchers: list[tuple[str, str]]) -> bool:
    if not matchers or not isinstance(button, dict):
        return False
    btype = str(button.get("type") or "").strip().lower()
    if btype not in ("callback", ""):
        return False
    cb_value = str(button.get("value") or button.get("text") or "")
    for flt, value in matchers:
        if flt == "any":
            return True
        if not value:
            continue
        if flt == "equals" and cb_value == value:
            return True
        if flt in ("starts_with", "prefix") and cb_value.startswith(value):
            return True
        if flt == "ends_with" and cb_value.endswith(value):
            return True
        if flt == "contains" and value in cb_value:
            return True
    return False


def _unsupported_source_handle(node, source_handle: str | None, module_map: dict | None = None) -> str | None:
    handle = source_handle or "source-handler"
    if handle == "source-handler":
        return None

    button_handles = set(_button_source_handles(node))
    if button_handles:
        if handle in button_handles:
            return None
        if handle.startswith(f"target-handler-{node.id}-"):
            return (
                f"`{node.id}` has no keyboard button matching source handle `{handle}`. "
                "Use `target-handler-{nodeId}-{button.text}` with the exact button text."
            )

    allowed = set(_allowed_source_handles(node, module_map))
    if handle not in allowed:
        return (
            f"`{node.type}` `{node.id}` does not support source handle `{handle}`. "
            f"Allowed handles: {', '.join(sorted(allowed)) or 'source-handler'}."
        )
    return None


def _allowed_source_handles(node, module_map: dict | None = None) -> set[str]:
    # MODULE nodes (moduleId.NodeName) aren't in the static catalog — their real
    # source handles (e.g. output-success/output-error) come from the per-project
    # manifest carried in module_map. Without this the validator wrongly allows
    # only "source-handler" and rejects every module branch edge.
    mod = (module_map or {}).get(node.type)
    if mod is not None:
        mod_handles = {
            str(h.get("id")) for h in (mod.get("source_handles") or []) if isinstance(h, dict) and h.get("id")
        }
        allowed = mod_handles or {"source-handler"}
        # module nodes may still declare dynamic outputs in data.outputs below
        spec = None
    else:
        spec = get_node_spec(node.type)
        allowed = set(spec.source_handles if spec else ["source-handler"])
    if node.type == "IfConditionNode" and isinstance(node.data.get("branches"), list):
        allowed.update(f"branch_{index}" for index, _ in enumerate(node.data["branches"]))
    if node.type == "RandomNode":
        weights = node.data.get("weights")
        if isinstance(weights, list):
            allowed.update(f"option_{index}" for index, _ in enumerate(weights))
        else:
            allowed.update({"option_0", "option_1"})
    if node.type == "ForLoopContinueNode":
        allowed.update({"loop-continue", "loop-done"})
    if node.type == "SwitchNode":
        cases = node.data.get("cases")
        if isinstance(cases, list):
            for index, case in enumerate(cases):
                case_id = ""
                if isinstance(case, dict):
                    case_id = str(case.get("id") or "")
                allowed.add(f"output-{case_id or f'case_{index}'}")
        if node.data.get("default") is True:
            allowed.add("output-default")
    # Nodes that declare named outputs (CustomCodeNode via exit("name"),
    # SubFlowNode, plugins) expose a dynamic `output-{name}` source handle per
    # declared output in data.outputs. The engine routes these via
    # sourceHandle == "output-" + name.
    outputs = node.data.get("outputs")
    if isinstance(outputs, list):
        for out in outputs:
            if isinstance(out, dict) and isinstance(out.get("name"), str) and out["name"]:
                allowed.add(f"output-{out['name']}")
    allowed.update(_button_source_handles(node))
    return allowed


def _missing_button_handles(node, present: set[str]) -> list[tuple[str, dict]]:
    return [
        (handle, button)
        for handle, button in _button_source_handles_with_buttons(node)
        if handle not in present
    ]


# Button-edge node turlari (inline tugma → target-handler-{nodeId}-{text}) endi
# MANIFESTDAN derive qilinadi — `button_edge_types()` (manifest `buttonEdge` +
# legacy MessageNode). Engine controller.go pause ro'yxatini aks ettiradi.


def _button_source_handles(node) -> list[str]:
    return [handle for handle, _ in _button_source_handles_with_buttons(node)]


def _button_source_handles_with_buttons(node) -> list[tuple[str, dict]]:
    """(`target-handler-{nodeId}-{text}`, button_dict) for each keyboard button."""
    if node.type not in button_edge_types():
        return []

    pairs: list[tuple[str, dict]] = []

    def _add(button: object) -> None:
        # collectionData (dynamic) columns have no fixed text — skip them.
        if isinstance(button, dict) and isinstance(button.get("text"), str) and button["text"]:
            pairs.append((f"target-handler-{node.id}-{button['text']}", button))

    # New format: keyboard_kind + rows:[{columns:[btn,...]}, ...]
    rows_new = node.data.get("rows")
    if isinstance(rows_new, list):
        for row in rows_new:
            if isinstance(row, dict):
                for button in row.get("columns") or []:
                    _add(button)
            elif isinstance(row, list):  # tolerate bare 2-D rows too
                for button in row:
                    _add(button)

    # Legacy format: keyboard:{active, inline:[[btn]], reply:[[btn]]}
    keyboard = node.data.get("keyboard")
    if isinstance(keyboard, dict):
        for group_name in ("reply", "inline"):
            for row in keyboard.get(group_name) or []:
                if isinstance(row, list):
                    for button in row:
                        _add(button)

    return pairs

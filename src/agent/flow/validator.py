"""Flow validation — schema, connectivity, edge handles, variables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import NODE_TYPES, get_node_spec, is_trigger
from .schema import FlowDocument


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

# Accessory targetHandle prefix (contract §1.1)
_ACC_PREFIX = "acc:"

# Nodes whose sole job is to persist a value into state and therefore MUST carry
# `data.type` (the value kind: text / contact_phone_number / *_file_id / user_id /
# Custom / …). Triggers that auto-save via an optional `data.state` object are
# intentionally NOT here — there `type` lives inside `data.state`, not `data`.
_STATE_SAVING_NODE_TYPES = frozenset({"StateNode"})

# Mapping: accessory targetHandle → accepted source node types (contract §1.1)
_ACC_PORT_ACCEPTS: dict[str, frozenset[str]] = {
    "acc:ai_model": frozenset({
        "AIOpenAIProviderNode", "AIOllamaProviderNode",
        "AIGeminiProviderNode", "AIAnthropicProviderNode",
    }),
    "acc:ai_tool": frozenset({
        "AIHTTPToolNode", "AICollectionToolNode", "AICustomCodeToolNode",
        "AIKnowledgeToolNode",
    }),
}


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


def validate(
    flow: FlowDocument,
    *,
    known_collections: list[str] | None = None,
    collection_schemas: dict[str, dict[str, str]] | None = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    # #12 — known collection names (from the workspace's loaded CMS list). When
    # provided, flow nodes referencing an unknown collection get a WARNING.
    known_set: set[str] | None = (
        {c for c in known_collections if isinstance(c, str) and c}
        if known_collections is not None
        else None
    )
    # #4 — {collection_name: {field: kind}} so filter values can be type-checked
    # against the loaded collection schema (warning only).
    schemas: dict[str, dict[str, str]] = collection_schemas or {}

    # ── 1. Node-level checks ───────────────────────────────────────────
    node_ids: set[str] = set()
    for node in flow.nodes:
        if not node.id:
            issues.append(
                ValidationIssue("error", "node_id_missing", "Node has empty id", node_id=node.id)
            )
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
            issues.append(
                ValidationIssue(
                    "error",
                    "unknown_node_type",
                    f"Unknown node type `{node.type}` (not in catalog)",
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

        # State-saving nodes MUST declare `data.type` (what kind of value is
        # captured — text/phone_number/file_id/Custom/…). Without it the engine
        # cannot resolve what to store and saves nothing → broken form. The
        # generic check above only catches a fully-absent key; here we also
        # reject an empty/blank `type` and emit an actionable message.
        if node.type in _STATE_SAVING_NODE_TYPES:
            state_type = node.data.get("type")
            if not (isinstance(state_type, str) and state_type.strip()):
                issues.append(
                    ValidationIssue(
                        "error",
                        "state_type_required",
                        f"`{node.type}` `{node.id}` must set `data.type` "
                        "(e.g. 'text', 'contact_phone_number', 'photo_file_id', "
                        "'user_id', 'Custom', …) so the engine knows WHAT value to "
                        "save into state. Note: for data arriving via a trigger "
                        "(MessageTrigger / inline button), prefer the trigger's own "
                        "`data.state = {key, type}` field instead of a separate "
                        "StateNode.",
                        node_id=node.id,
                    )
                )

        # AI provider talking to a CLOUD endpoint MUST have credentials. The
        # PREFERRED path is a stored credential (data.credential_id) — the engine
        # resolves it to api_key/base_url at runtime. apiKey is a legacy fallback
        # (still accepted; required for AIAnthropicProviderNode, which has no
        # credential type). Without either, the engine's LLM call yields empty
        # output → the bot sends nothing → dead bot. Local Ollama is exempt.
        if node.type in _ACC_PORT_ACCEPTS["acc:ai_model"]:
            cred_id = node.data.get("credential_id")
            has_credential = isinstance(cred_id, (int, str)) and str(cred_id).strip() not in (
                "", "0", "none", "null", "None",
            )
            api_key = str(node.data.get("apiKey") or "").strip()
            base_url = str(node.data.get("baseURL") or "").lower()
            is_local = (
                "localhost" in base_url
                or "127.0.0.1" in base_url
                or "://ollama:" in base_url
            )
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
                        "warning",
                        "unknown_collection",
                        f"`{node.type}` `{node.id}` references collection `{ref}` which is "
                        f"not in this project's collections ({sorted(known_set) or '—'}). "
                        "Create it (cms.* tools) or fix the name.",
                        node_id=node.id,
                    )
                )

        # #4 — filter value type vs loaded collection field kind (warning).
        if schemas:
            issues.extend(_filter_kind_warnings(node, schemas))

        # #9a — CustomCodeNode declares output handles (data.outputs) that no
        # edge uses. exit('NAME') without a wired edge is a dead branch.
        if node.type == "CustomCodeNode":
            issues.extend(_unused_output_warnings(node, flow))

    # ── 2. Edge-level checks ───────────────────────────────────────────
    nodes_by_id = {node.id: node for node in flow.nodes}
    edge_pairs: set[tuple[str, str, str | None]] = set()
    # Track acc:ai_model connections per AIAgentNode (must be exactly 1)
    ai_model_connections: dict[str, int] = {}

    for edge in flow.edges:
        if not edge.id:
            issues.append(
                ValidationIssue("error", "edge_id_missing", "Edge has empty id", edge_id=edge.id)
            )
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
            accepted = _ACC_PORT_ACCEPTS.get(target_handle)
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
                unsupported = _unsupported_source_handle(source_node, edge.sourceHandle)
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
    trigger_nodes = [n for n in flow.nodes if is_trigger(n.type)]
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
        if node.id not in reachable and not is_trigger(node.type):
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
            if not _button_needs_edge(button):
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
    issues.extend(_unknown_variable_warnings(flow))

    return ValidationResult(ok=not any(i.severity == "error" for i in issues), issues=issues)


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

    # #7 — terminal CustomCodeNode: a node that produces user-facing output
    # itself (send_*/edit_* in its jsCode) and then exit()s is a legitimate
    # leaf — it has NO downstream node by design, so its declared outputs being
    # unwired is NOT a bug. Two ways to mark this so we don't emit a misleading
    # "dead-end" warning:
    #   1. data.terminal == true  (explicit, preferred), OR
    #   2. heuristic: jsCode both sends a message AND calls exit(...).
    if data.get("terminal") is True:
        return []
    js = data.get("jsCode")
    if isinstance(js, str):
        sends = any(tok in js for tok in ("send_message", "edit_or_send", "edit_message_text", "send_photo", "answer_callback_query"))
        exits = "exit(" in js
        if sends and exits:
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

    used = {
        e.sourceHandle
        for e in flow.edges
        if e.source == node.id and isinstance(e.sourceHandle, str)
    }
    out: list[ValidationIssue] = []
    for name in declared:
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
        "first_name", "last_name", "username", "user_id", "language_code",
        "chat_id", "message_id", "message_text", "text", "phone", "phone_number",
        "latitude", "longitude", "inserted_id", "agent_output", "callback_data",
        "now", "today", "bot_username",
    }
)

# Prefixes/roots that are always allowed (namespaced runtime data).
_ALLOWED_VAR_PREFIXES = ("state.", "env.", "collection_", "item.", "doc.", "loop.",
                         "response.", "http_", "context.")

# Only scan these node fields for {{var}} usage — text/code carrying templates.
_TEMPLATE_TEXT_FIELDS = ("messageText", "text", "caption")

# `{{ var }}` with a simple root token (letters/digits/_), optional .path / [i].
_SIMPLE_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][\w]*)\b[^}]*\}\}")

# #8 — CustomCodeNode jsCode writes state via set_context('key', value). Capture
# every literal key it sets so {{key}} / {{state_key}} don't warn. Matches both
# single- and double-quoted first arguments: set_context('enroll_id', ...).
_SET_CONTEXT_RE = re.compile(r"""set_context\(\s*['"]([A-Za-z_]\w*)['"]""")


def _collect_state_keys(flow: FlowDocument) -> set[str]:
    """Best-effort set of state keys a flow writes (so they're known to {{var}})."""
    keys: set[str] = set()
    for n in flow.nodes:
        data = n.data if isinstance(n.data, dict) else {}
        # StateNode / capture: data.key
        k = data.get("key")
        if isinstance(k, str) and k:
            keys.add(k)
        # Triggers auto-saving via data.state = {key, type}
        st = data.get("state")
        if isinstance(st, dict) and isinstance(st.get("key"), str):
            keys.add(st["key"])
        # HTTPRequest saveAs / Download saveAs / Google* responseVariable
        for f in ("saveAs", "contextKey", "outputKey", "responseVariable"):
            v = data.get(f)
            if isinstance(v, str) and v:
                keys.add(v)
        # CreateInviteLinkNode writes the generated link to data.resultVar
        # (default `invite_link`) — so {{invite_link}} downstream is valid.
        if n.type == "CreateInviteLinkNode":
            rv = data.get("resultVar")
            keys.add(rv if isinstance(rv, str) and rv else "invite_link")
        # CollectionNode fieldMappings values are state keys; CollectionData too.
        fm = data.get("fieldMappings")
        if isinstance(fm, dict):
            keys.update(v for v in fm.values() if isinstance(v, str) and v)
        # #8 — CustomCodeNode jsCode sets state via set_context('key', ...). Each
        # key it writes is then valid as {{key}} AND {{state_key}} downstream.
        if n.type == "CustomCodeNode":
            code = data.get("jsCode")
            if isinstance(code, str) and code:
                for m in _SET_CONTEXT_RE.finditer(code):
                    key = m.group(1)
                    keys.add(key)
                    keys.add(f"state_{key}")
    return keys


def _unknown_variable_warnings(flow: FlowDocument) -> list[ValidationIssue]:
    state_keys = _collect_state_keys(flow)
    allowed_roots = state_keys | set(_STANDARD_CONTEXT_VARS)
    out: list[ValidationIssue] = []
    for node in flow.nodes:
        data = node.data if isinstance(node.data, dict) else {}
        for fld in _TEMPLATE_TEXT_FIELDS:
            text = data.get(fld)
            if not isinstance(text, str) or "{{" not in text:
                continue
            for m in _SIMPLE_VAR_RE.finditer(text):
                root = m.group(1)
                whole = m.group(0)
                inner = whole.strip("{} ").strip()
                if root in allowed_roots:
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
    {"url", "request_contact", "request_location", "switch_inline", "login", "web_app", "pay"}
)


def _button_needs_edge(button: dict | None) -> bool:
    if not isinstance(button, dict):
        return True
    btype = str(button.get("type") or "").strip().lower()
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
    for n in flow.nodes:
        if n.type != "CallbackQueryTriggerNode":
            continue
        if not bool(n.data.get("global")):
            continue
        flt = str(n.data.get("filter") or "any").strip().lower()
        ctx = n.data.get("context")
        value = ""
        if isinstance(ctx, dict):
            value = str(ctx.get("value") or "")
        matchers.append((flt, value))
    return matchers


def _callback_handled_globally(
    button: dict | None, matchers: list[tuple[str, str]]
) -> bool:
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


def _unsupported_source_handle(node, source_handle: str | None) -> str | None:
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

    allowed = set(_allowed_source_handles(node))
    if handle not in allowed:
        return (
            f"`{node.type}` `{node.id}` does not support source handle `{handle}`. "
            f"Allowed handles: {', '.join(sorted(allowed)) or 'source-handler'}."
        )
    return None


def _allowed_source_handles(node) -> set[str]:
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


# Node types whose inline-keyboard buttons can route via button-edges
# ("target-handler-{nodeId}-{text}"). Mirrors the engine's controller.go pause
# list (MessageNode / EditMessageNode / EditOrSendMessageNode /
# SendTextMessageNode); the engine resolves these edges type-agnostically, so
# EditOrSendMessageNode supports button-edges directly — no global
# CallbackQueryTriggerNode workaround is needed.
_BUTTON_EDGE_NODE_TYPES = frozenset(
    {
        "SendTextMessageNode",
        "EditOrSendMessageNode",
        "EditMessageNode",
        "MessageNode",
    }
)


def _button_source_handles(node) -> list[str]:
    return [handle for handle, _ in _button_source_handles_with_buttons(node)]


def _button_source_handles_with_buttons(node) -> list[tuple[str, dict]]:
    """(`target-handler-{nodeId}-{text}`, button_dict) for each keyboard button."""
    if node.type not in _BUTTON_EDGE_NODE_TYPES:
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

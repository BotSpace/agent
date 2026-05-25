"""Flow validation — schema, connectivity, edge handles, variables."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import get_node_spec, is_trigger
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


def validate(flow: FlowDocument) -> ValidationResult:
    issues: list[ValidationIssue] = []

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

    # ── 2. Edge-level checks ───────────────────────────────────────────
    nodes_by_id = {node.id: node for node in flow.nodes}
    edge_pairs: set[tuple[str, str, str | None]] = set()
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

    # ── 3. Trigger presence ────────────────────────────────────────────
    trigger_nodes = [n for n in flow.nodes if is_trigger(n.type)]
    if flow.nodes and not trigger_nodes:
        issues.append(
            ValidationIssue("warning", "no_trigger", "Flow has no trigger node — it can never start")
        )

    # ── 4. Reachability (BFS from triggers) ────────────────────────────
    reachable: set[str] = set()
    queue = [t.id for t in trigger_nodes]
    while queue:
        nid = queue.pop(0)
        if nid in reachable:
            continue
        reachable.add(nid)
        for e in flow.edges:
            if e.source == nid and e.target not in reachable:
                queue.append(e.target)
    for node in flow.nodes:
        if node.id not in reachable and not is_trigger(node.type):
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
        for handle in _missing_button_handles(node, present):
            issues.append(
                ValidationIssue(
                    "warning",
                    "missing_button_branch",
                    f"`{node.id}` button has no edge on `{handle}` handle",
                    node_id=node.id,
                )
            )

    return ValidationResult(ok=not any(i.severity == "error" for i in issues), issues=issues)


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
    allowed.update(_button_source_handles(node))
    return allowed


def _missing_button_handles(node, present: set[str]) -> list[str]:
    return [handle for handle in _button_source_handles(node) if handle not in present]


def _button_source_handles(node) -> list[str]:
    keyboard = node.data.get("keyboard")
    if node.type != "SendTextMessageNode" or not isinstance(keyboard, dict):
        return []

    handles: list[str] = []
    for group_name in ("reply", "inline"):
        rows = keyboard.get(group_name)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, list):
                continue
            for button in row:
                if not isinstance(button, dict):
                    continue
                text = button.get("text")
                if isinstance(text, str) and text:
                    handles.append(f"target-handler-{node.id}-{text}")
    return handles

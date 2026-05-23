"""Flow diff — agent o'zgartirishlarni inson o'qiy oladigan ko'rinishda chiqaradi."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import FlowDocument


@dataclass
class FlowDiff:
    added_nodes: list[str]
    removed_nodes: list[str]
    modified_nodes: list[str]
    added_edges: list[str]
    removed_edges: list[str]

    def render(self) -> str:
        lines: list[str] = []
        if self.added_nodes:
            lines.append(f"+ Nodes: {', '.join(self.added_nodes)}")
        if self.removed_nodes:
            lines.append(f"- Nodes: {', '.join(self.removed_nodes)}")
        if self.modified_nodes:
            lines.append(f"~ Nodes: {', '.join(self.modified_nodes)}")
        if self.added_edges:
            lines.append(f"+ Edges: {', '.join(self.added_edges)}")
        if self.removed_edges:
            lines.append(f"- Edges: {', '.join(self.removed_edges)}")
        return "\n".join(lines) if lines else "(no changes)"

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.added_nodes,
                self.removed_nodes,
                self.modified_nodes,
                self.added_edges,
                self.removed_edges,
            )
        )


def diff(before: FlowDocument, after: FlowDocument) -> FlowDiff:
    before_nodes = {n.id: n for n in before.nodes}
    after_nodes = {n.id: n for n in after.nodes}
    before_edges = {e.id: e for e in before.edges}
    after_edges = {e.id: e for e in after.edges}

    added_nodes = [nid for nid in after_nodes if nid not in before_nodes]
    removed_nodes = [nid for nid in before_nodes if nid not in after_nodes]
    modified_nodes = []
    for nid in after_nodes:
        if nid in before_nodes:
            if (
                before_nodes[nid].type != after_nodes[nid].type
                or before_nodes[nid].data != after_nodes[nid].data
            ):
                modified_nodes.append(nid)

    added_edges = [eid for eid in after_edges if eid not in before_edges]
    removed_edges = [eid for eid in before_edges if eid not in after_edges]
    return FlowDiff(
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        modified_nodes=modified_nodes,
        added_edges=added_edges,
        removed_edges=removed_edges,
    )

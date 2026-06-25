"""Flow domain models — engine-golang serialisation bilan to'g'ridan-to'g'ri mos.

Engine schema (loader/loader.go):
    Node  = {id, type, data, position}
    Edge  = {id, source, target, sourceHandle?, targetHandle?, type?}
    Flow  = {nodes: [...], edges: [...]}
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Position(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: float = 0.0
    y: float = 0.0


class FlowNode(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)


class FlowEdge(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    source: str
    target: str
    sourceHandle: str | None = "source-handler"
    targetHandle: str | None = "target-handler"
    type: str | None = "smart-edge"


class FlowDocument(BaseModel):
    model_config = ConfigDict(extra="allow")
    nodes: list[FlowNode] = Field(default_factory=list)
    edges: list[FlowEdge] = Field(default_factory=list)

    @classmethod
    def from_json(cls, raw: str | dict[str, Any]) -> FlowDocument:
        if isinstance(raw, str):
            payload = json.loads(raw or "{}")
        else:
            payload = raw
        return cls.model_validate(payload or {})

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=indent)

    def get_node(self, node_id: str) -> FlowNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def upsert_node(self, node: FlowNode) -> None:
        for i, existing in enumerate(self.nodes):
            if existing.id == node.id:
                self.nodes[i] = node
                return
        self.nodes.append(node)

    def remove_node(self, node_id: str) -> None:
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]

    def upsert_edge(self, edge: FlowEdge) -> None:
        for i, existing in enumerate(self.edges):
            if existing.id == edge.id:
                self.edges[i] = edge
                return
            if (
                existing.source == edge.source
                and existing.target == edge.target
                and existing.sourceHandle == edge.sourceHandle
                and existing.targetHandle == edge.targetHandle
            ):
                self.edges[i] = edge
                return
        self.edges.append(edge)

    def remove_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if e.id != edge_id]

    def triggers(self) -> list[FlowNode]:
        """Nodes with no incoming edges OR explicit triggers — entry points."""
        incoming = {e.target for e in self.edges}
        return [n for n in self.nodes if n.id not in incoming or "Trigger" in n.type]

    def downstream(self, node_id: str) -> list[FlowNode]:
        targets = [e.target for e in self.edges if e.source == node_id]
        return [n for n in self.nodes if n.id in targets]

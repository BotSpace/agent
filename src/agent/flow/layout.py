"""Auto-positioning — flow node'larini gridda joylashtirish.

Foydalanuvchi flow yaratganda position'larni qo'lda berishi shart emas.
Layered approach: BFS levels from triggers → x = level * 280, y = sibling * 160.
"""

from __future__ import annotations

from .schema import FlowDocument, Position


def auto_layout(flow: FlowDocument, *, x_step: int = 280, y_step: int = 160) -> FlowDocument:
    if not flow.nodes:
        return flow

    incoming: dict[str, list[str]] = {n.id: [] for n in flow.nodes}
    for e in flow.edges:
        if e.target in incoming:
            incoming[e.target].append(e.source)

    levels: dict[str, int] = {}
    queue = [n.id for n in flow.nodes if not incoming[n.id]]
    if not queue:
        queue = [flow.nodes[0].id]
    for nid in queue:
        levels[nid] = 0

    while queue:
        nid = queue.pop(0)
        current_level = levels[nid]
        for e in flow.edges:
            if e.source == nid and e.target not in levels:
                levels[e.target] = current_level + 1
                queue.append(e.target)

    # Unreached → after max + 1
    if levels:
        max_level = max(levels.values())
        for n in flow.nodes:
            if n.id not in levels:
                levels[n.id] = max_level + 1

    by_level: dict[int, list[str]] = {}
    for nid, lvl in levels.items():
        by_level.setdefault(lvl, []).append(nid)

    for lvl, ids in by_level.items():
        ids.sort()
        for idx, nid in enumerate(ids):
            node = flow.get_node(nid)
            if node:
                node.position = Position(x=lvl * x_step, y=idx * y_step)

    return flow

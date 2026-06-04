"""project.snapshot (#11) — one call, full project situational overview.

Aggregates, in ONE tool call, everything the model needs to orient itself on a
project: name/id, a flow summary (node/edge counts, triggers, type breakdown),
collections (name + document count), debug + release status, the last few saved
flow versions, and whether there are unsaved working changes.

Each backend lookup runs in parallel (asyncio.gather) and is INDIVIDUALLY error
-isolated — one failing section (e.g. CMS down) never blanks the rest; it just
reports that section as unavailable. readOnlyHint=true.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from ...integrations.backend import get_backend_client
from ..base import Tool, ToolContext, ToolResult, tool


def _backend(ctx: ToolContext):
    backend = ctx.services.get("backend")
    if backend is not None:
        return backend
    return get_backend_client()


def _concrete_project_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "global", "none", "null"}:
        return None
    return text


def _project_id(ctx: ToolContext, override: Any) -> str | None:
    return (
        _concrete_project_id(override)
        or _concrete_project_id(ctx.state.get("project_id"))
        or _concrete_project_id(ctx.project_id)
    )


async def _safe(coro: Any) -> Any:
    """Await a coroutine, returning the exception instead of raising so one bad
    section can't sink asyncio.gather (return_exceptions also used as backstop)."""
    try:
        return await coro
    except Exception as exc:  # pragma: no cover — exercised via partial-failure test
        return exc


class ProjectSnapshotInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Optional project id override; defaults to the active project.",
    )


def _flow_summary(flow: Any) -> dict[str, Any]:
    if not isinstance(flow, dict):
        return {"nodes": 0, "edges": 0, "triggers": [], "type_counts": {}}
    nodes = flow.get("nodes") if isinstance(flow.get("nodes"), list) else []
    edges = flow.get("edges") if isinstance(flow.get("edges"), list) else []
    type_counts: dict[str, int] = {}
    triggers: list[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        ntype = str(n.get("type") or "?")
        type_counts[ntype] = type_counts.get(ntype, 0) + 1
        if ntype.endswith("TriggerNode") or ntype == "CommandTriggerNode":
            data = n.get("data") if isinstance(n.get("data"), dict) else {}
            label = data.get("command") or data.get("filter") or n.get("id")
            triggers.append(f"{ntype}:{label}")
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "triggers": triggers,
        "type_counts": type_counts,
    }


async def _collections_section(backend: Any, pid: str, claims: Any) -> Any:
    """[{name, count}] — collection names with a best-effort document count.
    Counts are fetched in parallel and individually error-isolated."""
    cols = await backend.cms_list_collections(pid, auth_claims=claims)
    names = [
        c.get("name")
        for c in (cols if isinstance(cols, list) else [])
        if isinstance(c, dict) and c.get("name")
    ]

    async def _count(name: str) -> dict[str, Any]:
        try:
            res = await backend.cms_list_documents(
                pid, name, limit=1, auth_claims=claims
            )
            count = res.get("count") if isinstance(res, dict) else None
            if count is None and isinstance(res, dict):
                rows = res.get("results")
                count = len(rows) if isinstance(rows, list) else None
            return {"name": name, "count": count}
        except Exception:
            return {"name": name, "count": None}

    return await asyncio.gather(*[_count(n) for n in names]) if names else []


@tool(
    name="project.snapshot",
    description=(
        "Loyihaning TO'LIQ holatini BITTA chaqiruvda qaytaradi: nom/id, flow "
        "xulosasi (node/edge soni, triggerlar, type breakdown), kolleksiyalar "
        "(nom + hujjat soni), debug holati, release holati, oxirgi 3 flow "
        "versiyasi va saqlanmagan o'zgarish bor-yo'qligi. Loyiha ustida ishni "
        "boshlashda yoki vaziyatni tezda baholashda ishlat — bir nechta alohida "
        "tool chaqirishdan ko'ra arzon."
    ),
    input_model=ProjectSnapshotInput,
)
async def project_snapshot(input: ProjectSnapshotInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if not pid:
        return ToolResult.error(
            "Aktiv loyiha yo'q — session_use_project(project_id) yoki project_id bering."
        )
    backend = _backend(ctx)
    claims = ctx.state.get("auth_claims")

    async def _versions() -> Any:
        from ...mcp.flow_versions import get_flow_version_store

        rows = await get_flow_version_store().list(project_id=str(pid))
        return rows[:3]

    project, collections, releases, versions = await asyncio.gather(
        _safe(backend.get_project(pid, auth_claims=claims)),
        _safe(_collections_section(backend, str(pid), claims)),
        _safe(backend.list_releases(pid, auth_claims=claims)),
        _safe(_versions()),
        return_exceptions=True,
    )

    data: dict[str, Any] = {"project_id": str(pid)}
    lines: list[str] = []

    # ── project + flow ────────────────────────────────────────────────────
    name = str(pid)
    debug_status = "noma'lum"
    if isinstance(project, dict):
        name = str(project.get("name") or pid)
        data["name"] = name
        debug_uuid = project.get("debug_uuid") or project.get("debug_webhook_uuid")
        debug_running = bool(project.get("debug_running") or project.get("is_debug"))
        if debug_running:
            debug_status = "ishlayapti"
        elif debug_uuid:
            debug_status = "havola bor (ishlamayotgan bo'lishi mumkin)"
        else:
            debug_status = "yo'q"
        flow_sum = _flow_summary(project.get("flow"))
        data["flow"] = flow_sum
        lines.append(f"Loyiha: {name} (id={pid})")
        lines.append(
            f"Flow (saqlangan): {flow_sum['nodes']} node, {flow_sum['edges']} edge"
        )
        if flow_sum["triggers"]:
            lines.append("  Triggerlar: " + ", ".join(flow_sum["triggers"]))
        if flow_sum["type_counts"]:
            tb = ", ".join(
                f"{t}×{c}"  # noqa: RUF001 — established node-count separator glyph
                for t, c in sorted(flow_sum["type_counts"].items())
            )
            lines.append("  Node turlari: " + tb)
    else:
        lines.append(f"Loyiha: id={pid} (ma'lumot olinmadi: {project})")

    # ── unsaved working changes (from workspace state) ────────────────────
    flow_dirty = bool(ctx.state.get("flow_dirty"))
    working = ctx.state.get("working_flow")
    if isinstance(working, dict):
        w = _flow_summary(working)
        data["working"] = {"nodes": w["nodes"], "edges": w["edges"], "dirty": flow_dirty}
        if flow_dirty:
            lines.append(
                f"Saqlanmagan o'zgarishlar: BOR (working {w['nodes']} node, {w['edges']} edge)"
            )
        else:
            lines.append("Saqlanmagan o'zgarishlar: yo'q")
    data["flow_dirty"] = flow_dirty

    lines.append(f"Debug: {debug_status}")
    data["debug_status"] = debug_status

    # ── collections ───────────────────────────────────────────────────────
    if isinstance(collections, list):
        data["collections"] = collections
        if collections:
            parts = [
                f"{c['name']}({c['count'] if c['count'] is not None else '?'})"
                for c in collections
                if isinstance(c, dict)
            ]
            lines.append(f"Kolleksiyalar ({len(collections)}): " + ", ".join(parts))
        else:
            lines.append("Kolleksiyalar: yo'q")
    else:
        lines.append(f"Kolleksiyalar: olinmadi ({collections})")
        data["collections"] = None

    # ── releases ──────────────────────────────────────────────────────────
    if isinstance(releases, list):
        rel = [
            {"id": r.get("id"), "status": r.get("status")}
            for r in releases
            if isinstance(r, dict)
        ]
        data["releases"] = rel
        if rel:
            latest = rel[0]
            lines.append(
                f"Release: {len(rel)} ta (oxirgisi id={latest.get('id')}, "
                f"status={latest.get('status')})"
            )
        else:
            lines.append("Release: yo'q (deploy qilinmagan)")
    else:
        lines.append(f"Release: olinmadi ({releases})")
        data["releases"] = None

    # ── flow versions ─────────────────────────────────────────────────────
    if isinstance(versions, list):
        data["versions"] = versions
        if versions:
            vparts = [
                f"v{v.get('version')}({v.get('node_count')}n)"
                for v in versions
                if isinstance(v, dict)
            ]
            lines.append("Oxirgi versiyalar: " + ", ".join(vparts))
        else:
            lines.append("Flow versiyalari: yo'q")
    else:
        lines.append(f"Flow versiyalari: olinmadi ({versions})")
        data["versions"] = None

    return ToolResult.ok("\n".join(lines), data=data)


def build_project_snapshot_tools() -> list[Tool]:
    return [project_snapshot]

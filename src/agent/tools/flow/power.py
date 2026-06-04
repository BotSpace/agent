"""flow.* power tools — simulate / export / import / history / rollback.

BITTA implementatsiya, ikkala interfeys:
  - LangGraph agent (dashboard chat) — ToolContext.state ichidagi working_flow,
    backend client services'da, auth_claims state'da.
  - MCP — `server._execute_registry_tool` shu tool'larni workspace state bilan
    ishga tushiradi (state_updates merge + draft write-through), shuning uchun
    MCP'da AVTOMATIK to'g'ri ishlaydi (workspace working_flow bilan).

Tashqi servislar:
  - simulator klient (`integrations.simulator.simulate`) — engine HTTP servisi.
  - flow versiyalar store (`mcp.flow_versions`) — Postgres, graceful degrade.
    history/rollback va flow.save versiyalash shu store'dan foydalanadi.
"""

from __future__ import annotations

import json as _json
from typing import Any

from pydantic import BaseModel, Field

from ...flow import FlowDocument, validate
from ..base import ToolContext, ToolResult, tool

# ── shared state helpers ──────────────────────────────────────────────────────


def _working_flow(ctx: ToolContext) -> dict[str, Any]:
    raw = ctx.state.get("working_flow")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return FlowDocument.from_json(raw).model_dump()
        except Exception:
            return {"nodes": [], "edges": []}
    return {"nodes": [], "edges": []}


def _concrete_project_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "global", "none", "null"}:
        return None
    return text


def _power_project_id(ctx: ToolContext) -> str | None:
    return _concrete_project_id(ctx.state.get("project_id")) or _concrete_project_id(
        ctx.project_id
    )


def _backend(ctx: ToolContext):
    from ...integrations.backend import get_backend_client

    client = ctx.services.get("backend")
    if client is not None:
        return client
    return get_backend_client()


def _validate_flow_dict(flow: dict) -> tuple[bool, str, dict]:
    """Run the flow validator on a raw flow dict. Returns (ok, rendered, summary).
    `ok` is False on validation ERRORS (warnings don't block)."""
    try:
        doc = FlowDocument.model_validate(flow or {})
    except Exception as exc:
        return False, f"Flow JSON noto'g'ri: {exc}", {"errors": [str(exc)], "warnings": []}
    result = validate(doc)
    summary = {
        "ok": result.ok,
        "errors": [i.message for i in result.errors],
        "warnings": [i.message for i in result.warnings],
    }
    return result.ok, result.render(), summary


# ── flow.simulate ─────────────────────────────────────────────────────────────


class FlowSimulateStep(BaseModel):
    type: str = Field(
        description=(
            "Qadam turi: 'command' (text=/start), 'text' (text='salom'), "
            "'contact' (phone, first_name?), 'location' (latitude, longitude), "
            "'click' (callback button: data=callback_data), 'click_text' "
            "(reply/inline tugma matnini bosish: text='Tugma')."
        )
    )
    text: str | None = None
    data: str | None = None
    phone: str | None = None
    first_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class FlowSimulateInput(BaseModel):
    steps: list[FlowSimulateStep] = Field(
        min_length=1,
        description=(
            "Foydalanuvchi kiritmalari ketma-ketligi (kamida bitta). Har element "
            "{type, ...}."
        ),
    )
    seed_collections: bool = Field(
        default=True,
        description=(
            "True (default) — loyiha kolleksiyalarining HOZIRGI hujjatlarini "
            "backend'dan o'qib simulyatorga beradi (har kolleksiyadan max 50). "
            "False — bo'sh kolleksiyalar."
        ),
    )
    use_saved: bool = Field(
        default=False,
        description=(
            "False (default) — workspace'dagi working (saqlanmagan) flow'ni "
            "sinaydi. True — backend'dagi SAQLANGAN flow'ni."
        ),
    )


async def _seed_collections(ctx: ToolContext) -> dict[str, list[dict]]:
    """Read the CURRENT documents of the project's collections (max 50 each) so
    the simulator runs against realistic data. Per-collection failures isolated."""
    out: dict[str, list[dict]] = {}
    project_id = _power_project_id(ctx)
    if not project_id:
        return out
    raw = ctx.state.get("collections")
    names = [
        c.get("name")
        for c in (raw if isinstance(raw, list) else [])
        if isinstance(c, dict) and c.get("name")
    ]
    if not names:
        return out
    backend = _backend(ctx)
    claims = ctx.state.get("auth_claims")
    for name in names:
        try:
            res = await backend.cms_list_documents(
                project_id, str(name), limit=50, auth_claims=claims
            )
            rows = res.get("results") if isinstance(res, dict) else None
            out[str(name)] = rows if isinstance(rows, list) else []
        except Exception:  # pragma: no cover — isolate one bad collection
            out[str(name)] = []
    return out


def _fmt_simulate_step(idx: int, step: dict) -> list[str]:
    """One simulated step → human-readable lines for the model."""
    inp = step.get("input") if isinstance(step, dict) else None
    in_label = ""
    if isinstance(inp, dict):
        itype = inp.get("type") or "?"
        val = (
            inp.get("text")
            or inp.get("data")
            or inp.get("phone")
            or (
                f"{inp.get('latitude')},{inp.get('longitude')}"
                if inp.get("latitude") is not None
                else ""
            )
            or ""
        )
        in_label = f"{itype}:{val}".rstrip(":")
    msgs = step.get("bot_messages") if isinstance(step, dict) else None
    n_msgs = len(msgs) if isinstance(msgs, list) else 0
    nodes = step.get("nodes") if isinstance(step, dict) else None
    node_ids = [
        n.get("id")
        for n in (nodes if isinstance(nodes, list) else [])
        if isinstance(n, dict) and n.get("id")
    ]
    path = " → ".join(node_ids) if node_ids else "—"
    edits = len(step.get("edits") or []) if isinstance(step, dict) else 0
    deletes = len(step.get("deletes") or []) if isinstance(step, dict) else 0
    extra = ""
    if edits:
        extra += f", ✏️ {edits} edit"
    if deletes:
        extra += f", 🗑️ {deletes} delete"
    lines = [f"{idx}. ➡️ {in_label or '—'} → 🤖 {n_msgs} xabar [{path}]{extra}"]
    for m in msgs if isinstance(msgs, list) else []:
        if not isinstance(m, dict):
            continue
        txt = str(m.get("text") or "").replace("\n", " ")
        if len(txt) > 120:
            txt = txt[:117] + "…"
        if txt:
            lines.append(f"     🤖 {txt}")
    errs = step.get("errors") if isinstance(step, dict) else None
    for e in errs if isinstance(errs, list) else []:
        lines.append(f"     ❌ XATO: {e}")
    return lines


@tool(
    name="flow.simulate",
    description=(
        "Joriy flow'ni engine simulyatorida sinab ko'radi — REAL bot / "
        "Telegram'siz. Foydalanuvchi qadamlari ketma-ketligini (steps) "
        "yuborasiz, har qadam uchun bot QANDAY javob berishini (xabarlar, "
        "klaviatura, o'zgartirish/o'chirish, kechgan node'lar, xatolar) "
        "qaytaradi. flow.save'dan keyin asosiy stsenariylarni O'ZINGIZ "
        "tekshiring — foydalanuvchidan so'rashdan oldin verify qiling. "
        "use_saved=false (default) SAQLANMAGAN working draft'ni ham sinaydi."
    ),
    input_model=FlowSimulateInput,
)
async def flow_simulate(input: FlowSimulateInput, ctx: ToolContext) -> ToolResult:
    from ...config import get_settings
    from ...integrations.backend import load_saved_flow
    from ...integrations.simulator import SimulatorUnavailableError, simulate

    project_id = _power_project_id(ctx)
    if not project_id:
        return ToolResult.error(
            "Aktiv loyiha yo'q — avval loyihani tanlang (session_use_project / "
            "project.create)."
        )

    claims = ctx.state.get("auth_claims")
    if input.use_saved:
        try:
            flow = await load_saved_flow(
                project_id, auth_claims=claims, client=_backend(ctx)
            )
            flow = flow if isinstance(flow, dict) else {"nodes": [], "edges": []}
        except Exception as exc:
            return ToolResult.error(f"Saqlangan flow'ni yuklab bo'lmadi: {exc}")
    else:
        flow = _working_flow(ctx)

    collections: dict[str, list[dict]] = {}
    if input.seed_collections:
        collections = await _seed_collections(ctx)

    claims_dict = claims if isinstance(claims, dict) else {}
    chat = {
        "id": 1,
        "first_name": str(claims_dict.get("first_name") or "Test"),
        "username": str(claims_dict.get("username") or "test_user"),
    }
    steps = [s.model_dump(exclude_none=True) for s in input.steps]

    try:
        resp = await simulate(
            simulator_url=get_settings().simulator_url,
            flow=flow,
            steps=steps,
            collections=collections,
            chat=chat,
        )
    except SimulatorUnavailableError:
        return ToolResult.error(
            "Simulyator servisi hozircha ishga tushmagan — flow.simulate'ni "
            "keyinroq qayta urinib ko'ring (yoki release_debug bilan debug "
            "botda sinab ko'ring)."
        )

    sim_steps = resp.get("steps") if isinstance(resp, dict) else None
    top_errors = resp.get("errors") if isinstance(resp, dict) else None
    flow_src = "saqlangan flow" if input.use_saved else "working draft (saqlanmagan)"
    lines = [
        f"Simulyatsiya — {flow_src}, {len(steps)} qadam "
        f"({len(collections)} kolleksiya seed):"
    ]
    if isinstance(sim_steps, list):
        for i, st in enumerate(sim_steps, 1):
            lines.extend(_fmt_simulate_step(i, st if isinstance(st, dict) else {}))
    else:
        lines.append("(simulyator qadam natijalarini qaytarmadi)")
    if isinstance(top_errors, list) and top_errors:
        lines.append("")
        lines.append("Umumiy xatolar:")
        lines.extend(f"  ❌ {e}" for e in top_errors)
    ok = bool(resp.get("ok")) if isinstance(resp, dict) else False
    lines.append("")
    lines.append(
        "✅ Simulyatsiya muvaffaqiyatli"
        if ok
        else "⚠️ Simulyatsiyada xatolar bor — yuqoriga qarang"
    )
    return ToolResult(
        content="\n".join(lines),
        is_error=not ok,
        data=resp if isinstance(resp, dict) else None,
    )


# ── flow.export ───────────────────────────────────────────────────────────────


class FlowExportInput(BaseModel):
    pass


@tool(
    name="flow.export",
    description=(
        "Joriy working flow'ni TO'LIQ JSON ko'rinishida qaytaradi (data + matn). "
        "Zaxira nusxa olish yoki boshqa loyihaga ko'chirish (flow.import) uchun."
    ),
    input_model=FlowExportInput,
)
async def flow_export(_: FlowExportInput, ctx: ToolContext) -> ToolResult:
    flow = _working_flow(ctx)
    nodes = len(flow.get("nodes") or [])
    edges = len(flow.get("edges") or [])
    body = _json.dumps(flow, ensure_ascii=False, indent=2)
    return ToolResult.ok(
        f"Working flow ({nodes} node, {edges} edge):\n```json\n{body}\n```",
        data=flow,
    )


# ── flow.import (DESTRUCTIVE) ─────────────────────────────────────────────────


class FlowImportInput(BaseModel):
    flow_json: dict[str, Any] = Field(
        description="To'liq flow obyekti: {nodes:[...], edges:[...]}."
    )


@tool(
    name="flow.import",
    description=(
        "To'liq flow JSON'ini working flow sifatida O'RNATADI (mavjudini "
        "ALMASHTIRADI). Avval validatsiyadan o'tkaziladi — xato bo'lsa import "
        "QILINMAYDI. Persist EMAS — flow.save bilan tasdiqlang."
    ),
    input_model=FlowImportInput,
    destructive=True,
)
async def flow_import(input: FlowImportInput, ctx: ToolContext) -> ToolResult:
    if not isinstance(input.flow_json, dict):
        return ToolResult.error(
            "flow_json majburiy — to'liq flow obyekti {nodes, edges} bering."
        )
    ok, rendered, summary = _validate_flow_dict(input.flow_json)
    if not ok:
        return ToolResult.error(
            "Import RAD ETILDI — flow validatsiyadan o'tmadi:\n" + rendered,
            data=summary,
        )
    normalized = FlowDocument.model_validate(input.flow_json).model_dump()
    nodes = len(normalized.get("nodes") or [])
    edges = len(normalized.get("edges") or [])
    warn = ""
    if summary.get("warnings"):
        warn = "\nOgohlantirishlar:\n" + "\n".join(
            f"  ⚠️ {w}" for w in summary["warnings"]
        )
    return ToolResult.ok(
        f"Flow import qilindi: {nodes} node, {edges} edge (flow_dirty=true). "
        "Persist EMAS — saqlash uchun flow.save chaqiring." + warn,
        state_updates={"working_flow": normalized, "flow_dirty": True},
        data={"nodes": nodes, "edges": edges, "warnings": summary.get("warnings", [])},
    )


# ── flow.history ──────────────────────────────────────────────────────────────


class FlowHistoryInput(BaseModel):
    pass


@tool(
    name="flow.history",
    description=(
        "Loyihaning saqlangan flow versiyalari ro'yxati (oxirgi 20). Har "
        "muvaffaqiyatli flow.save yangi versiya yozadi. flow.rollback bilan "
        "istalgan versiyaga qaytish mumkin."
    ),
    input_model=FlowHistoryInput,
)
async def flow_history(_: FlowHistoryInput, ctx: ToolContext) -> ToolResult:
    from ...mcp.flow_versions import get_flow_version_store

    project_id = _power_project_id(ctx)
    if not project_id:
        return ToolResult.error(
            "Aktiv loyiha yo'q — avval loyihani tanlang."
        )
    versions = await get_flow_version_store().list(project_id=project_id)
    if not versions:
        return ToolResult.ok(
            "Hali saqlangan versiyalar yo'q — flow.save qilingach versiyalar "
            "paydo bo'ladi.",
            data={"versions": []},
        )
    lines = [
        f"Loyiha {project_id} — saqlangan flow versiyalari (oxirgi {len(versions)}):"
    ]
    for v in versions:
        lines.append(
            f"  v{v['version']} — {v.get('saved_at') or '?'} "
            f"({v['node_count']} node, {v['edge_count']} edge)"
        )
    lines.append("flow.rollback(version) bilan istalganiga qayting.")
    return ToolResult.ok("\n".join(lines), data={"versions": versions})


# ── flow.rollback (DESTRUCTIVE) ───────────────────────────────────────────────


class FlowRollbackInput(BaseModel):
    version: int = Field(description="flow.history qaytargan version raqami.")


@tool(
    name="flow.rollback",
    description=(
        "flow.history'dagi tanlangan VERSION flow'ini working flow'ga yuklaydi "
        "(working_flow, flow_dirty=true). Persist EMAS — o'zgarishni tasdiqlash "
        "uchun keyin flow.save chaqiring."
    ),
    input_model=FlowRollbackInput,
    destructive=True,
)
async def flow_rollback(input: FlowRollbackInput, ctx: ToolContext) -> ToolResult:
    from ...mcp.flow_versions import get_flow_version_store

    project_id = _power_project_id(ctx)
    if not project_id:
        return ToolResult.error(
            "Aktiv loyiha yo'q — avval loyihani tanlang."
        )
    record = await get_flow_version_store().get(
        project_id=project_id, version=input.version
    )
    if record is None:
        return ToolResult.error(
            f"v{input.version} topilmadi. flow.history bilan mavjud versiyalarni "
            "ko'ring."
        )
    flow = record.get("flow") or {"nodes": [], "edges": []}
    normalized = FlowDocument.model_validate(flow).model_dump()
    nodes = len(normalized.get("nodes") or [])
    edges = len(normalized.get("edges") or [])
    return ToolResult.ok(
        f"v{input.version} working flow'ga yuklandi: {nodes} node, {edges} edge "
        "(flow_dirty=true). Persist EMAS — flow.save bilan tasdiqlang.",
        state_updates={"working_flow": normalized, "flow_dirty": True},
        data={"version": input.version, "nodes": nodes, "edges": edges},
    )


def build_flow_power_tools() -> list:
    return [
        flow_simulate,
        flow_export,
        flow_import,
        flow_history,
        flow_rollback,
    ]

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
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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


def _power_project_id(ctx: ToolContext, explicit: Any = None) -> str | None:
    """Resolve the project id. Priority: explicit tool arg → state → ctx.

    The explicit arg matters in the dashboard agent: a sub-agent may know the
    project from its prompt but not have it set in ctx (only the working flow is
    propagated), so tools like flow.simulate/save MUST accept project_id directly
    — exactly like flow.save already does."""
    return (
        _concrete_project_id(explicit)
        or _concrete_project_id(ctx.state.get("project_id"))
        or _concrete_project_id(ctx.project_id)
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


class FlowSimulateChat(BaseModel):
    """Per-step user identity (simulator v2) — multi-user stsenariy uchun
    (masalan referral: user A user B'ni taklif qiladi)."""

    id: int = Field(description="Telegram chat/user id (har user uchun farqli).")
    first_name: str | None = None
    username: str | None = None


class FlowSimulateExpect(BaseModel):
    """Qadamdan keyin bot javobiga qo'yiladigan tekshiruvlar (simulator v2).

    Hamma maydonlar ixtiyoriy; berilganlari tekshiriladi. Bittasi ham bajarilmasa
    qadam 'failed' deb belgilanadi."""

    contains: str | None = Field(
        default=None, description="Bot xabarlaridan birortasi shu matnni o'z ichiga olishi shart."
    )
    not_contains: str | None = Field(
        default=None, description="Hech bir bot xabarida shu matn BO'LMASLIGI kerak."
    )
    keyboard_contains: str | None = Field(
        default=None, description="Klaviatura tugmalaridan biri shu matnni o'z ichiga olishi shart."
    )
    message_count: int | None = Field(
        default=None, description="Shu qadamda bot AYNAN shuncha xabar yuborishi kerak."
    )
    no_errors: bool | None = Field(
        default=None, description="True — shu qadamda hech qanday engine xatosi bo'lmasligi kerak."
    )


# AI klientlar tabiiy ravishda yozadigan muqobil nomlar → canonical tur.
_STEP_TYPE_ALIASES = {
    "callback": "click",
    "callback_query": "click",
    "callbackquery": "click",
    "callback_data": "click",
    "callback_query_data": "click",
    "button": "click",
    "inline": "click",
    "press": "click",
    "message": "text",
    "cmd": "command",
}

_STEP_TYPES = ("command", "text", "contact", "location", "click", "click_text", "advance_time")

# Maydon nomlarining tabiiy variantlari → kanonik. command:"/start" yozsa text'ga,
# lat/lng → latitude/longitude, phone_number → phone, callback_data → data.
_STEP_FIELD_ALIASES = {
    "command": "text",
    "message": "text",
    "callback_data": "data",
    "callbackData": "data",
    "phone_number": "phone",
    "phoneNumber": "phone",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "firstName": "first_name",
}


class FlowSimulateStep(BaseModel):
    type: Literal[
        "command", "text", "contact", "location", "click", "click_text", "advance_time"
    ] = Field(
        description=(
            "Qadam turi: 'command' (text=/start), 'text' (text='salom'), "
            "'contact' (phone, first_name?), 'location' (latitude, longitude), "
            "'click' (INLINE tugma bosish: data=callback_data — 'callback'/'button' "
            "emas, aynan 'click'!), 'click_text' (tugma MATNI bo'yicha bosish: "
            "text='Tugma' — callback_data noma'lum bo'lsa), 'advance_time' "
            "(vaqtni seconds soniyaga oldinga suradi — CronTriggerNode'larni sinash)."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_type(cls, values: Any) -> Any:
        """Tip va maydon nomlarini kanonik shaklga keltiradi — AI klient tabiiy
        nomlar yozsa ham round-trip bo'lmasin ('command'→text, lat→latitude...)."""
        if isinstance(values, dict):
            raw = str(values.get("type") or "").strip().lower()
            if raw in _STEP_TYPE_ALIASES:
                values["type"] = _STEP_TYPE_ALIASES[raw]
                raw = values["type"]
            elif raw and raw not in _STEP_TYPES:
                raise ValueError(
                    f"Noma'lum step type {raw!r}. Ruxsat etilganlar: {', '.join(_STEP_TYPES)}. "
                    "Inline tugma bosish uchun 'click' (data=callback_data) yoki "
                    "'click_text' (text=tugma matni) ishlating."
                )
            # Maydon aliaslari — agent docs o'qimasa ham to'g'ri o'qiladi.
            for alias, canonical in _STEP_FIELD_ALIASES.items():
                if alias in values and not values.get(canonical):
                    values[canonical] = values.pop(alias)
        return values

    text: str | None = None
    data: str | None = None
    phone: str | None = None
    first_name: str | None = None
    latitude: float | None = Field(default=None, serialization_alias="lat")
    longitude: float | None = Field(default=None, serialization_alias="lon")
    seconds: int | None = Field(
        default=None,
        description="advance_time uchun: vaqtni necha soniya oldinga surish.",
    )
    chat: FlowSimulateChat | None = Field(
        default=None,
        description=(
            "Shu qadamni BOSHQA foydalanuvchi nomidan yuborish (multi-user). "
            "Berilmasa — default test foydalanuvchisi. Referral/2-user oqimlarda ishlat."
        ),
    )
    expect: FlowSimulateExpect | None = Field(
        default=None,
        description=(
            "Shu qadamdan keyin bot javobini avtomatik tekshirish "
            "(contains/not_contains/keyboard_contains/message_count/no_errors). "
            "Ko'p stsenariy tekshirayotganda mode='assert' bilan birga ishlat."
        ),
    )


class FlowSimulateInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Loyiha id (ixtiyoriy). Dashboard agentida ctx'da loyiha "
        "o'rnatilmagan bo'lsa SHU bilan bering — promptdagi project_id'ni uzating.",
    )
    steps: list[FlowSimulateStep] = Field(
        min_length=1,
        description=(
            "Foydalanuvchi kiritmalari ketma-ketligi (kamida bitta). Har element "
            "{type, ...}. Har qadamga ixtiyoriy `chat` (boshqa user) va `expect` "
            "(javob tekshiruvi) qo'shsa bo'ladi."
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
    mode: str = Field(
        default="full",
        description=(
            "'full' (default) — har qadam batafsil ko'rsatiladi (xabarlar, node yo'li). "
            "'assert' — KOMPAKT natija: '✅ 9/10 o'tdi' + faqat FAILED qadamlar batafsil. "
            "Ko'p `expect` bilan stsenariy tekshirayotganda mode='assert' ishlat — "
            "token sezilarli tejaydi."
        ),
    )


async def _seed_collections(ctx: ToolContext, project_id: str | None = None) -> dict[str, list[dict]]:
    """Read the CURRENT documents of the project's collections (max 50 each) so
    the simulator runs against realistic data. Per-collection failures isolated."""
    out: dict[str, list[dict]] = {}
    project_id = _concrete_project_id(project_id) or _power_project_id(ctx)
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
    # Edit'larni ham KO'RSAT — pagination/edit-in-place o'zgargan matnni ko'rsin
    # (faqat soni emas). Aks holda agent "0 xabar" ko'rib edit ishladimi bilmaydi.
    for ed in (step.get("edits") or []) if isinstance(step, dict) else []:
        if not isinstance(ed, dict):
            continue
        etxt = str(ed.get("text") or ed.get("caption") or "").replace("\n", " ")
        if len(etxt) > 120:
            etxt = etxt[:117] + "…"
        if etxt:
            lines.append(f"     ✏️ (edit) {etxt}")
    errs = step.get("errors") if isinstance(step, dict) else None
    for e in errs if isinstance(errs, list) else []:
        lines.append(f"     ❌ XATO: {e}")
    # simulator v2 — per-step expect verdict.
    if isinstance(step, dict) and step.get("expect_passed") is not None:
        if step.get("expect_passed"):
            lines.append("     ✅ expect: o'tdi")
        else:
            fails = step.get("expect_failures")
            detail = "; ".join(str(f) for f in fails) if isinstance(fails, list) and fails else "?"
            lines.append(f"     ⚠️ expect: o'tmadi — {detail}")
    return lines


def _step_input_label(inp: Any) -> str:
    """Short '{type}:{value}' label for an assert-mode failed step."""
    if not isinstance(inp, dict):
        return "—"
    itype = inp.get("type") or "?"
    val = (
        inp.get("text")
        or inp.get("data")
        or inp.get("phone")
        or (str(inp.get("seconds")) if inp.get("seconds") is not None else "")
        or ""
    )
    return f"{itype}:{val}".rstrip(":")


def _expect_failures(step: dict) -> list[str]:
    """Collect failure reasons for one step: explicit expect failures + engine
    errors (so an assert failure always carries actionable detail)."""
    reasons: list[str] = []
    fails = step.get("expect_failures")
    if isinstance(fails, list):
        reasons.extend(str(f) for f in fails if f)
    return reasons


def _format_assert(resp: dict, n_steps: int) -> tuple[str, dict]:
    """Compact assert-mode render: '✅ 9/10 o'tdi' + only failed steps detailed.

    A step counts as FAILED if expect_passed is False OR it carries engine errors.
    Returns (text, structured) where structured matches the #1 assert contract:
      {"ok", "passed", "failed":[{step,input,reason,errors}], "final_state"}.
    """
    # Engine assert rejimida KOMPAKT javob qaytaradi: {ok, passed, failed,
    # final_state} — "steps" YO'Q. Avval shu native shaklni ishlatamiz; aks
    # holda (mode=full yoki eski simulyator) steps'dan qayta hisoblaymiz.
    if isinstance(resp.get("passed"), int) and "steps" not in resp:
        passed = resp["passed"]
        failed = []
        for f in resp.get("failed") or []:
            f = f if isinstance(f, dict) else {}
            failed.append(
                {
                    # engine 0-based step indeksini 1-based ko'rsatamiz
                    "step": (f.get("step") or 0) + 1,
                    "input": _step_input_label(f.get("input")),
                    "reason": str(f.get("reason") or "?"),
                    "errors": [str(e) for e in (f.get("errors") or [])],
                }
            )
        total = passed + len(failed) or n_steps
        top_errors = resp.get("errors") if isinstance(resp.get("errors"), list) else []
        ok = (not failed) and (not top_errors) and bool(resp.get("ok", True))
        icon = "✅" if ok else "❌"
        lines = [f"{icon} {passed}/{total} o'tdi"]
        for f in failed:
            line = f"  ❌ step {f['step']} ({f['input']}) — {f['reason']}"
            if f["errors"]:
                line += " | " + "; ".join(f["errors"])
            lines.append(line)
        if top_errors:
            lines.append("Umumiy xatolar: " + "; ".join(str(e) for e in top_errors))
        structured = {
            "ok": ok,
            "passed": passed,
            "failed": failed,
            "final_state": resp.get("final_state"),
        }
        return "\n".join(lines), structured

    sim_steps = resp.get("steps") if isinstance(resp.get("steps"), list) else []
    failed: list[dict] = []
    passed = 0
    for i, st in enumerate(sim_steps, 1):
        st = st if isinstance(st, dict) else {}
        errors = st.get("errors") if isinstance(st.get("errors"), list) else []
        expect_passed = st.get("expect_passed")
        is_fail = (expect_passed is False) or bool(errors)
        if is_fail:
            reasons = _expect_failures(st)
            if not reasons and errors:
                reasons = ["engine xatosi"]
            failed.append(
                {
                    "step": i,
                    "input": _step_input_label(st.get("input")),
                    "reason": "; ".join(reasons) or "?",
                    "errors": [str(e) for e in errors],
                }
            )
        else:
            passed += 1
    total = len(sim_steps) or n_steps
    top_errors = resp.get("errors") if isinstance(resp.get("errors"), list) else []
    ok = (not failed) and (not top_errors) and bool(resp.get("ok", True))
    icon = "✅" if ok else "❌"
    lines = [f"{icon} {passed}/{total} o'tdi"]
    for f in failed:
        line = f"  ❌ step {f['step']} ({f['input']}) — {f['reason']}"
        if f["errors"]:
            line += " | " + "; ".join(f["errors"])
        lines.append(line)
    if top_errors:
        lines.append("Umumiy xatolar: " + "; ".join(str(e) for e in top_errors))
    structured = {
        "ok": ok,
        "passed": passed,
        "failed": failed,
        "final_state": resp.get("final_state"),
    }
    return "\n".join(lines), structured


def _default_chat(ctx: ToolContext) -> dict[str, Any]:
    claims = ctx.state.get("auth_claims")
    claims_dict = claims if isinstance(claims, dict) else {}
    return {
        "id": 1,
        "first_name": str(claims_dict.get("first_name") or "Test"),
        "username": str(claims_dict.get("username") or "test_user"),
    }


async def _resolve_sim_flow(ctx: ToolContext, *, use_saved: bool) -> dict[str, Any]:
    """The flow simulator runs against — saved (backend) or working draft."""
    if use_saved:
        from ...integrations.backend import load_saved_flow

        flow = await load_saved_flow(
            _power_project_id(ctx), auth_claims=ctx.state.get("auth_claims"),
            client=_backend(ctx),
        )
        return flow if isinstance(flow, dict) else {"nodes": [], "edges": []}
    return _working_flow(ctx)


async def _run_simulation(
    ctx: ToolContext,
    *,
    flow: dict[str, Any],
    steps: list[dict[str, Any]],
    seed_collections: bool,
    mode: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Thin wrapper around the simulator transport: seeds collections, fills the
    default chat, forwards `mode`. Raises SimulatorUnavailableError on outage."""
    from ...config import get_settings
    from ...integrations.simulator import simulate

    collections: dict[str, list[dict]] = {}
    if seed_collections:
        collections = await _seed_collections(ctx, project_id)
    return await simulate(
        simulator_url=get_settings().simulator_url,
        flow=flow,
        steps=steps,
        collections=collections,
        chat=_default_chat(ctx),
        mode=mode,
    )


@tool(
    name="flow.simulate",
    description=(
        "Joriy flow'ni engine simulyatorida sinab ko'radi — REAL bot / "
        "Telegram'siz. Foydalanuvchi qadamlari ketma-ketligini (steps) "
        "yuborasiz, har qadam uchun bot QANDAY javob berishini (xabarlar, "
        "klaviatura, o'zgartirish/o'chirish, kechgan node'lar, xatolar) "
        "qaytaradi. Har qadamga `expect` qo'shib javobni AVTOMATIK tekshiring "
        "(contains/keyboard_contains/...); `chat` bilan boshqa user nomidan "
        "yuboring (referral/multi-user). KO'P stsenariy tekshirayotganda "
        "mode='assert' ishlat — kompakt '✅ 9/10 o'tdi' natija qaytaradi va "
        "TOKEN tejaydi. flow.save'dan keyin asosiy stsenariylarni O'ZINGIZ "
        "verify qiling. use_saved=false (default) SAQLANMAGAN working draft'ni sinaydi."
    ),
    input_model=FlowSimulateInput,
)
async def flow_simulate(input: FlowSimulateInput, ctx: ToolContext) -> ToolResult:
    from ...integrations.simulator import SimulatorUnavailableError

    project_id = _power_project_id(ctx, getattr(input, "project_id", None))
    if not project_id:
        return ToolResult.error(
            "Aktiv loyiha yo'q — avval loyihani tanlang (session_use_project / "
            "project.create)."
        )

    if input.use_saved:
        try:
            flow = await _resolve_sim_flow(ctx, use_saved=True)
        except Exception as exc:
            return ToolResult.error(f"Saqlangan flow'ni yuklab bo'lmadi: {exc}")
    else:
        flow = _working_flow(ctx)

    steps = [s.model_dump(exclude_none=True, by_alias=True) for s in input.steps]
    mode = "assert" if str(input.mode).lower() == "assert" else "full"

    try:
        resp = await _run_simulation(
            ctx,
            flow=flow,
            steps=steps,
            seed_collections=input.seed_collections,
            mode=mode,
            project_id=project_id,
        )
    except SimulatorUnavailableError:
        return ToolResult.error(
            "Simulyator servisi hozircha ishga tushmagan — flow.simulate'ni "
            "keyinroq qayta urinib ko'ring (yoki release_debug bilan debug "
            "botda sinab ko'ring)."
        )

    resp = resp if isinstance(resp, dict) else {"ok": False, "steps": []}

    # mode=assert — compact pass/fail summary (token-saving).
    if mode == "assert":
        text, structured = _format_assert(resp, len(steps))
        return ToolResult(
            content=text, is_error=not structured["ok"], data=structured
        )

    # mode=full — per-step detail + expect verdict markers.
    sim_steps = resp.get("steps")
    top_errors = resp.get("errors")
    flow_src = "saqlangan flow" if input.use_saved else "working draft (saqlanmagan)"
    lines = [
        f"Simulyatsiya — {flow_src}, {len(steps)} qadam:"
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
    ok = bool(resp.get("ok"))
    lines.append("")
    lines.append(
        "✅ Simulyatsiya muvaffaqiyatli"
        if ok
        else "⚠️ Simulyatsiyada xatolar bor — yuqoriga qarang"
    )
    return ToolResult(content="\n".join(lines), is_error=not ok, data=resp)


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
    project_id: str | int | None = Field(
        default=None,
        description="Loyiha id (ixtiyoriy). Dashboard agentida ctx'da loyiha "
        "o'rnatilmagan bo'lsa SHU bilan bering — promptdagi project_id'ni uzating.",
    )
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
async def flow_history(input: FlowHistoryInput, ctx: ToolContext) -> ToolResult:
    from ...mcp.flow_versions import get_flow_version_store

    project_id = _power_project_id(ctx, getattr(input, "project_id", None))
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
    project_id: str | int | None = Field(
        default=None,
        description="Loyiha id (ixtiyoriy). Dashboard agentida ctx'da loyiha "
        "o'rnatilmagan bo'lsa SHU bilan bering — promptdagi project_id'ni uzating.",
    )
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

    project_id = _power_project_id(ctx, getattr(input, "project_id", None))
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


# ── flow.test_* — saved regression tests (#10) ───────────────────────────────


def _validate_test_steps(raw_steps: list[Any]) -> tuple[list[dict] | None, str | None]:
    """Validate steps against the simulate step schema. Returns (steps, error)."""
    if not isinstance(raw_steps, list) or not raw_steps:
        return None, "steps majburiy — kamida bitta qadam bering (simulate formati)."
    out: list[dict] = []
    for i, s in enumerate(raw_steps):
        try:
            model = FlowSimulateStep.model_validate(s)
        except Exception as exc:
            return None, f"steps[{i}] noto'g'ri: {exc}"
        out.append(model.model_dump(exclude_none=True, by_alias=True))
    return out, None


async def _run_saved_tests(
    ctx: ToolContext,
    *,
    flow: dict[str, Any],
    tests: list[dict[str, Any]],
    seed_collections: bool = True,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Run each saved test in mode=assert against `flow`. Returns aggregate:
      {"ran": bool, "passed": N, "total": M, "results": [{name, ok, passed, total,
        failed:[...]}]}. `ran` is False ONLY if the simulator was unavailable —
    callers degrade gracefully (e.g. flow.save must not break)."""
    from ...integrations.simulator import SimulatorUnavailableError

    results: list[dict[str, Any]] = []
    passed = 0
    for t in tests:
        steps = t.get("steps") if isinstance(t.get("steps"), list) else []
        normalized, err = _validate_test_steps(steps)
        if err or normalized is None:
            results.append(
                {"name": t.get("name"), "ok": False, "error": err or "bad steps"}
            )
            continue
        try:
            resp = await _run_simulation(
                ctx,
                flow=flow,
                steps=normalized,
                seed_collections=seed_collections,
                project_id=project_id,
                mode="assert",
            )
        except SimulatorUnavailableError:
            return {"ran": False, "passed": 0, "total": len(tests), "results": []}
        resp = resp if isinstance(resp, dict) else {"ok": False, "steps": []}
        _, structured = _format_assert(resp, len(normalized))
        ok = bool(structured["ok"])
        if ok:
            passed += 1
        results.append(
            {
                "name": t.get("name"),
                "ok": ok,
                "passed": structured["passed"],
                "total": len(normalized),
                "failed": structured["failed"],
            }
        )
    return {"ran": True, "passed": passed, "total": len(tests), "results": results}


def _fmt_test_results(agg: dict[str, Any]) -> str:
    """'✅ 3/4 test o'tdi; ❌ 'referral': step 2 — ...'"""
    total = agg.get("total", 0)
    passed = agg.get("passed", 0)
    icon = "✅" if passed == total and total else ("❌" if total else "ℹ️")  # noqa: RUF001
    parts = [f"{icon} {passed}/{total} test o'tdi"]
    for r in agg.get("results", []):
        if r.get("ok"):
            continue
        if r.get("error"):
            parts.append(f"❌ '{r.get('name')}': {r['error']}")
            continue
        failed = r.get("failed") or []
        if failed:
            details = []
            for f in failed[:3]:
                d = f"step {f.get('step')} ({f.get('input', '?')}) — {f.get('reason')}"
                if f.get("errors"):
                    d += " | " + "; ".join(str(e) for e in f["errors"])
                details.append(d)
            more = f" (+{len(failed) - 3} yana)" if len(failed) > 3 else ""
            parts.append(f"❌ '{r.get('name')}': " + "; ".join(details) + more)
        else:
            parts.append(
                f"❌ '{r.get('name')}': o'tmadi "
                f"({r.get('passed', '?')}/{r.get('total', '?')} step)"
            )
    return "; ".join(parts)


class FlowTestSaveInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Loyiha id (ixtiyoriy). Dashboard agentida ctx'da loyiha "
        "o'rnatilmagan bo'lsa SHU bilan bering — promptdagi project_id'ni uzating.",
    )
    name: str = Field(description="Test nomi (loyiha ichida unikal — qayta saqlasa almashadi).")
    steps: list[FlowSimulateStep] = Field(
        min_length=1,
        description=(
            "Simulate qadamlari (expect'lar bilan) — bot qurib bo'lgach asosiy "
            "stsenariyni shu yerda saqlang. Har qadamga `expect` qo'shing "
            "(contains/keyboard_contains/...) — aks holda test faqat xato yo'qligini "
            "tekshiradi."
        ),
    )
    enabled: bool = Field(default=True, description="False — testni o'chirmasdan vaqtincha o'chirib qo'yadi.")


@tool(
    name="flow.test_save",
    description=(
        "Asosiy stsenariyni SAQLANADIGAN regression test sifatida yozadi (simulate "
        "qadamlari + expect'lar). Bot qurib bo'lgach yoki muhim oqim qo'shgach "
        "saqlang — keyin flow.save har safar bu testlarni avtomatik ishga tushiradi "
        "va regressiyani ushlaydi. Nom loyiha ichida unikal (upsert)."
    ),
    input_model=FlowTestSaveInput,
)
async def flow_test_save(input: FlowTestSaveInput, ctx: ToolContext) -> ToolResult:
    from ...mcp.flow_tests import get_flow_test_store

    project_id = _power_project_id(ctx, getattr(input, "project_id", None))
    if not project_id:
        return ToolResult.error("Aktiv loyiha yo'q — avval loyihani tanlang.")
    name = input.name.strip()
    if not name:
        return ToolResult.error("name bo'sh bo'lmasligi kerak.")
    steps = [s.model_dump(exclude_none=True, by_alias=True) for s in input.steps]
    ok = await get_flow_test_store().save(
        project_id=project_id,
        user_id=str(ctx.user_id) if ctx.user_id is not None else None,
        name=name,
        steps=steps,
        enabled=input.enabled,
    )
    if not ok:
        return ToolResult.error(
            "Testni saqlab bo'lmadi (test ombori hozircha mavjud emas). Keyinroq "
            "qayta urinib ko'ring."
        )
    return ToolResult.ok(
        f"Regression test `{name}` saqlandi ({len(steps)} qadam, "
        f"enabled={input.enabled}). flow.save endi uni avtomatik ishga tushiradi.",
        data={"name": name, "steps": len(steps), "enabled": input.enabled},
    )


class FlowTestListInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Loyiha id (ixtiyoriy). Dashboard agentida ctx'da loyiha "
        "o'rnatilmagan bo'lsa SHU bilan bering — promptdagi project_id'ni uzating.",
    )
    pass


@tool(
    name="flow.test_list",
    description="Loyihaning saqlangan regression testlari ro'yxati (nom, qadam soni, enabled).",
    input_model=FlowTestListInput,
)
async def flow_test_list(input: FlowTestListInput, ctx: ToolContext) -> ToolResult:
    from ...mcp.flow_tests import get_flow_test_store

    project_id = _power_project_id(ctx, getattr(input, "project_id", None))
    if not project_id:
        return ToolResult.error("Aktiv loyiha yo'q — avval loyihani tanlang.")
    tests = await get_flow_test_store().list(project_id=project_id)
    if not tests:
        return ToolResult.ok(
            "Saqlangan test yo'q — flow.test_save bilan asosiy stsenariylarni saqlang.",
            data={"tests": []},
        )
    lines = [f"Loyiha {project_id} — {len(tests)} saqlangan test:"]
    summary = []
    for t in tests:
        n_steps = len(t.get("steps") or [])
        mark = "" if t.get("enabled") else " (o'chirilgan)"
        lines.append(f"  • {t['name']} — {n_steps} qadam{mark}")
        summary.append({"name": t["name"], "steps": n_steps, "enabled": t["enabled"]})
    lines.append("flow.test_run bilan barchasini ishga tushiring.")
    return ToolResult.ok("\n".join(lines), data={"tests": summary})


class FlowTestDeleteInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Loyiha id (ixtiyoriy). Dashboard agentida ctx'da loyiha "
        "o'rnatilmagan bo'lsa SHU bilan bering — promptdagi project_id'ni uzating.",
    )
    name: str = Field(description="O'chiriladigan test nomi.")
    confirm: bool = Field(
        default=False,
        description="Tasdiqlash — false bo'lsa o'chirilmaydi (qaytarib bo'lmaydi).",
    )


@tool(
    name="flow.test_delete",
    description=(
        "Saqlangan regression testni o'chiradi (qaytarib bo'lmaydi). "
        "confirm=true talab qiladi."
    ),
    input_model=FlowTestDeleteInput,
    destructive=True,
)
async def flow_test_delete(input: FlowTestDeleteInput, ctx: ToolContext) -> ToolResult:
    from ...mcp.flow_tests import get_flow_test_store

    project_id = _power_project_id(ctx, getattr(input, "project_id", None))
    if not project_id:
        return ToolResult.error("Aktiv loyiha yo'q — avval loyihani tanlang.")
    name = input.name.strip()
    if not input.confirm:
        return ToolResult.error(
            f"`{name}` testini o'chirish qaytarib bo'lmaydi. Tasdiqlash uchun "
            "confirm=true bilan qayta chaqiring."
        )
    deleted = await get_flow_test_store().delete(project_id=project_id, name=name)
    if not deleted:
        return ToolResult.error(
            f"`{name}` topilmadi yoki o'chirib bo'lmadi (flow.test_list bilan tekshiring)."
        )
    return ToolResult.ok(f"Regression test `{name}` o'chirildi.")


class FlowTestRunInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Loyiha id (ixtiyoriy). Dashboard agentida ctx'da loyiha "
        "o'rnatilmagan bo'lsa SHU bilan bering — promptdagi project_id'ni uzating.",
    )
    name: str | None = Field(
        default=None,
        description=(
            "Bitta testni ishga tushirish uchun nom. Berilmasa — BARCHA enabled "
            "testlar mode=assert bilan ishga tushadi va jamlangan natija qaytadi."
        ),
    )
    use_saved: bool = Field(
        default=False,
        description="True — backend'dagi SAQLANGAN flow'da, False (default) — working draft'da.",
    )


@tool(
    name="flow.test_run",
    description=(
        "Saqlangan regression testlarni ishga tushiradi (mode=assert). name "
        "berilmasa BARCHA enabled testlar ishga tushadi: '✅ 3/4 test o'tdi; "
        "❌ \"referral\": step 2 — ...'. O'zgarish kiritgach yoki flow.save'dan "
        "oldin chaqirib regressiyani tekshiring."
    ),
    input_model=FlowTestRunInput,
)
async def flow_test_run(input: FlowTestRunInput, ctx: ToolContext) -> ToolResult:
    from ...mcp.flow_tests import get_flow_test_store

    project_id = _power_project_id(ctx, getattr(input, "project_id", None))
    if not project_id:
        return ToolResult.error("Aktiv loyiha yo'q — avval loyihani tanlang.")
    store = get_flow_test_store()
    if input.name:
        one = await store.get(project_id=project_id, name=input.name.strip())
        if one is None:
            return ToolResult.error(
                f"`{input.name}` topilmadi. flow.test_list bilan mavjudlarini ko'ring."
            )
        tests = [one]
    else:
        tests = await store.list(project_id=project_id, only_enabled=True)
        if not tests:
            return ToolResult.ok(
                "Ishga tushiriladigan enabled test yo'q — flow.test_save bilan saqlang.",
                data={"ran": False, "total": 0, "results": []},
            )

    try:
        flow = await _resolve_sim_flow(ctx, use_saved=input.use_saved)
    except Exception as exc:
        return ToolResult.error(f"Flow'ni yuklab bo'lmadi: {exc}")

    agg = await _run_saved_tests(ctx, flow=flow, tests=tests, project_id=project_id)
    if not agg.get("ran"):
        return ToolResult.error(
            "Simulyator servisi mavjud emas — testlar ishga tushmadi. Keyinroq "
            "qayta urinib ko'ring (yoki release_debug bilan debug botda sinang)."
        )
    text = _fmt_test_results(agg)
    return ToolResult(content=text, is_error=agg["passed"] != agg["total"], data=agg)


def build_flow_power_tools() -> list:
    return [
        flow_simulate,
        flow_export,
        flow_import,
        flow_history,
        flow_rollback,
        flow_test_save,
        flow_test_list,
        flow_test_delete,
        flow_test_run,
    ]

"""release.* tools — bot deploy lifecycle."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...integrations.backend import BackendClient, BackendError, get_backend_client, pagination_summary
from ..base import Tool, ToolContext, ToolResult, tool

_HINTS_BY_CODE = {
    "paid_plan_required": (
        "Bu loyihaga PULLI tarif kerak (siz boshqa loyihangizda FREE'ni ishlatgansiz). "
        "`plans.list`'dan start yoki pro plan id'sini olib `release.deploy(plan_id=<paid_id>)` bilan qayta chaqir."
    ),
    "plan_required": (
        "Birinchi release uchun `plan_id` majburiy — `plans.list` chaqir va plan id bilan qayta urinib ko'r."
    ),
    "plan_inactive": "plan_id noto'g'ri yoki nofaol. `plans.list` natijasidan faol plan tanla.",
    "invalid_billing_months": "`months` faqat 1, 3, 6 yoki 12 bo'lishi mumkin.",
    "insufficient_balance": (
        "Balans yetarli emas. Foydalanuvchini /app/billing sahifasiga yo'naltir va "
        "to'ldirgandan keyin qaytarib chaqir."
    ),
}


def _deploy_hint(err: BackendError) -> str | None:
    """Translate known backend failures into an actionable next step.

    Prefers stable error codes (e.g. ``paid_plan_required``) shipped by the
    backend's exception handler. Falls back to substring matching when the
    code is missing (old backend, non-DRF endpoint).
    """
    for code in err.codes:
        if code in _HINTS_BY_CODE:
            hint = _HINTS_BY_CODE[code]
            if code == "insufficient_balance":
                amounts = re.findall(r"\d[\d_ ]*", err.human_message())
                if amounts:
                    return f"{hint} (Kerakli: {amounts[0]} so'm)"
            return hint

    msg = err.human_message().lower()
    if "balansingiz yetarli emas" in msg:
        return _HINTS_BY_CODE["insufficient_balance"]
    if "bepul tarifdan foydalangansiz" in msg or "pulli tarif tanlash majburiy" in msg:
        return _HINTS_BY_CODE["paid_plan_required"]
    if "tarif tanlash majburiy" in msg or "birinchi release" in msg:
        return _HINTS_BY_CODE["plan_required"]
    if "noto'g'ri yoki nofaol tarif" in msg:
        return _HINTS_BY_CODE["plan_inactive"]
    if "project is required" in msg or "bu maydon to'ldirilishi shart" in msg:
        return "project_id kontekstda yo'q — foydalanuvchidan loyihani aniqlash so'ra yoki `project.list` orqali tanla."
    if err.status == 401:
        return "Sessiya muddati tugagan ko'rinadi — foydalanuvchidan qayta kirishni so'ra."
    if err.status == 403:
        return "Bu operatsiyaga ruxsat yo'q (ehtimol loyiha boshqa foydalanuvchiga tegishli)."
    if err.status == 429:
        return "Server release chastotasini cheklab qo'ydi (rate-limit) — bir necha daqiqadan so'ng qayta urin."
    return None


def _wrap_backend_err(action: str, err: BackendError) -> str:
    """Format a Backend error so the LLM and the chat strip both stay useful."""
    parts = [f"{action} muvaffaqiyatsiz (HTTP {err.status or '?'}):", err.human_message()]
    hint = _deploy_hint(err)
    if hint:
        parts.append(f"Keyingi qadam: {hint}")
    return "\n".join(parts)


def _backend(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if backend is not None:
        return backend
    return get_backend_client()


def _project_id(ctx: ToolContext, override: str | int | None) -> str | int | None:
    # Release calls must be scoped to a concrete project, never global context.
    value = override or ctx.state.get("project_id") or ctx.project_id
    if value is None:
        return None
    if str(value).strip().lower() in {"", "global", "none", "null"}:
        return None
    return value


class ReleaseListInput(BaseModel):
    project_id: str | int | None = None
    limit: int = Field(default=10, ge=1, le=50)
    page: int | None = Field(
        default=None,
        description="Page number. Paginated; omit for page 1, increment if total_pages > 1.",
    )
    page_size: int | None = Field(
        default=None,
        description="Items per page (default 10). Increase (e.g. 100) to fetch more at once.",
    )


@tool(
    name="release.list",
    description=(
        "List recent releases (deploys) for the current project. Paginated (10/page by default) — "
        "the result reports total_items/total_pages; pass `page`/`page_size` for more."
    ),
    input_model=ReleaseListInput,
)
async def release_list(input: ReleaseListInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id available.")
    try:
        items = await _backend(ctx).list_releases(
            pid, page=input.page, page_size=input.page_size, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Releaselarni olish", e))
    total = getattr(items, "total_items", len(items))
    note = pagination_summary(items)
    shown = list(items)[: input.limit]
    payload = [
        {
            "id": r.get("id"),
            "version": r.get("version"),
            "status": r.get("status"),
            "created_at": r.get("created_at"),
        }
        for r in shown
    ]
    summary = ", ".join(
        f"id={r['id']} v{r['version']}({r['status']})"
        for r in payload if r.get("id")
    ) or "(none)"
    return ToolResult.ok(
        f"Showing {len(payload)} of {total} release(s): {summary}.{note} "
        f"Note: use the `id` field (NOT the version number) as release_id "
        f"for release.status / release.logs.",
        data={
            "releases": payload,
            "total_items": total,
            "total_pages": getattr(items, "total_pages", 1),
            "current_page": getattr(items, "current_page", 1),
        },
    )


class ReleaseDeployInput(BaseModel):
    project_id: str | int | None = None
    # str because the Go backend's plan ids are UUIDs (plans.list returns
    # UUID strings, not Django's legacy integer ids). Kept optional: the Go
    # release-create endpoint no longer accepts/uses plan_id at all (project
    # plans were dropped from release creation server-side) — this field is
    # currently a no-op, retained only for backward-compatible tool inputs.
    plan_id: str | int | None = Field(
        default=None,
        description=(
            "Plan id (from plans.list). NOTE: currently ignored by the backend — "
            "release creation no longer requires/uses a plan."
        ),
    )
    months: int | None = Field(
        default=None, description="Billing period in months (1, 3, 6, 12). Default: 1."
    )


@tool(
    name="release.deploy",
    description=(
        "Deploy the saved flow of the project to production. This creates a new release "
        "(version) and brings up the engine container. For the first release of a "
        "project, plan_id is required — call plans.list first."
    ),
    input_model=ReleaseDeployInput,
    destructive=True,
)
async def release_deploy(input: ReleaseDeployInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id available.")
    try:
        release = await _backend(ctx).create_release(
            pid,
            plan_id=input.plan_id,
            months=input.months,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Deploy", e))
    return ToolResult.ok(
        f"Deploy queued (release id={release.get('id')}, version={release.get('version')}).",
        data={"release": release},
    )


# ── release.debug ─────────────────────────────────────────────────────────────


_DEBUG_BOT_USERNAME = "botmother_test_robot"


class ReleaseDebugInput(BaseModel):
    project_id: str | int | None = None


@tool(
    name="release.debug",
    description=(
        "Run the bot in DEBUG mode (like the constructor's 'Debug Run'): brings up a "
        "temporary debug container from the SAVED flow — NO bot token required — and "
        "returns a Telegram link to test it. The debug container auto-expires after "
        "~30 minutes. Use this so the user can try the bot immediately without "
        "deploying to production or setting a token."
    ),
    input_model=ReleaseDebugInput,
)
async def release_debug(input: ReleaseDebugInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id available.")
    auth = ctx.state.get("auth_claims")
    try:
        await _backend(ctx).start_debug(pid, auth_claims=auth)
        project = await _backend(ctx).get_project(pid, auth_claims=auth)
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Debug run", e))
    debug_uuid = project.get("debug_uuid") or project.get("debug_webhook_uuid")
    if not debug_uuid:
        return ToolResult.error(
            "Debug container started, but the project has no debug_uuid to build the test link."
        )
    link = f"https://t.me/{_DEBUG_BOT_USERNAME}?start={debug_uuid}"
    return ToolResult.ok(
        "Debug bot ishga tushdi (token shart emas, ~30 daqiqadan keyin avtomatik "
        f"o'chadi). Sinab ko'rish uchun: {link}",
        data={"debug_link": link, "debug_uuid": str(debug_uuid)},
    )


# ── plans.list ──────────────────────────────────────────────────────────────


class PlansListInput(BaseModel):
    pass


@tool(
    name="plans.list",
    description=(
        "List available pricing plans. Use this BEFORE the first release of a "
        "project to discover the FREE plan id (and pricing of paid plans)."
    ),
    input_model=PlansListInput,
)
async def plans_list(_: PlansListInput, ctx: ToolContext) -> ToolResult:
    try:
        items = await _backend(ctx).list_plans(auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Tariflarni olish", e))
    rows = [
        {
            "id": p.get("id"),
            "name": p.get("name"),
            "price": p.get("price"),
            "currency": p.get("currency"),
            "is_default": p.get("is_default"),
        }
        for p in items
    ]
    summary = ", ".join(
        f"{r['name']}#{r['id']}({r['price']})" for r in rows if r.get("name")
    ) or "(none)"
    return ToolResult.ok(f"{len(rows)} plan(s): {summary}", data={"plans": rows})


class SubscriptionCurrentInput(BaseModel):
    pass


@tool(
    name="subscription.current",
    description="Get the current user's active subscription/plan.",
    input_model=SubscriptionCurrentInput,
)
async def subscription_current(_: SubscriptionCurrentInput, ctx: ToolContext) -> ToolResult:
    try:
        sub = await _backend(ctx).subscription_current(auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Joriy tarifni olish", e))

    sub_dict: dict[str, Any] = sub if isinstance(sub, dict) else {}
    plan_raw = sub_dict.get("plan")
    plan = plan_raw if isinstance(plan_raw, dict) else {}
    plan_name = (
        plan.get("name")
        or plan.get("display")
        or sub_dict.get("plan_name")
        or (str(plan_raw) if plan_raw not in (None, "") else "unknown")
    )
    status = sub_dict.get("status", "unknown")
    return ToolResult.ok(
        f"Current plan: {plan_name} (status={status}).",
        data={"subscription": sub_dict},
    )


class SubscriptionUpgradeInput(BaseModel):
    # str because the Go backend's plan ids are UUIDs, not integers.
    plan_id: str | int = Field(description="Target plan id (from plans.list).")
    months: int = Field(default=1, description="Billing months: 1, 3, 6, or 12.")


@tool(
    name="subscription.upgrade",
    description="Upgrade or change the current user's subscription plan.",
    input_model=SubscriptionUpgradeInput,
    destructive=True,
)
async def subscription_upgrade(input: SubscriptionUpgradeInput, ctx: ToolContext) -> ToolResult:
    try:
        res = await _backend(ctx).subscription_upgrade(
            plan_id=input.plan_id,
            months=input.months,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Tarifni o'zgartirish", e))

    new_plan = res.get("new_plan") or res.get("plan_name") or "unknown"
    return ToolResult.ok(
        f"Subscription updated to `{new_plan}`.",
        data={"subscription_upgrade": res},
    )


class PaymentLinkCreateInput(BaseModel):
    method: Literal["click", "payme"] = Field(
        description="Payment provider."
    )
    amount: int = Field(ge=1, description="Top-up amount in UZS.")


@tool(
    name="payment.link.create",
    description="Create a payment top-up link for the current user.",
    input_model=PaymentLinkCreateInput,
)
async def payment_link_create(input: PaymentLinkCreateInput, ctx: ToolContext) -> ToolResult:
    try:
        res = await _backend(ctx).create_payment_link(
            method=input.method,
            amount=input.amount,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("To'lov havolasini yaratish", e))

    link = res.get("link")
    if not isinstance(link, str) or not link.strip():
        return ToolResult.error("Payment link yaratildi, lekin bo'sh link qaytdi.")

    return ToolResult.ok(
        f"Payment link created via {input.method}: {link}",
        data={"payment": {"method": input.method, "amount": input.amount, "link": link}},
    )


class ReleaseStatusInput(BaseModel):
    release_id: str | int = Field(
        description=(
            "Release DB id (the `id` field from release.list — NOT the version "
            "number)."
        )
    )


@tool(
    name="release.status",
    description="Get the status of a specific release (running/stopped/failed/pending).",
    input_model=ReleaseStatusInput,
)
async def release_status(input: ReleaseStatusInput, ctx: ToolContext) -> ToolResult:
    try:
        target = await _backend(ctx).get_release(
            input.release_id,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Release statusini olish", e))
    if target is None:
        return ToolResult.error(f"Release `{input.release_id}` not found.")
    return ToolResult.ok(f"Release `{input.release_id}` status: {target.get('status')}", data={"release": target})


class ReleaseLogsInput(BaseModel):
    release_id: str | int = Field(
        description=(
            "Release DB id (the `id` field from release.list — NOT the version "
            "number like 4 or 'v4'). Example: 4521."
        )
    )
    tail: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Number of trailing log lines to return.",
    )
    since_seconds: int | None = Field(
        default=None,
        description="Only return lines from the last N seconds (optional).",
    )


@tool(
    name="release.logs",
    description=(
        "Fetch a snapshot of the deployed bot's recent pod logs. Use this when a "
        "release status is failed/crashed or the user asks 'why isn't my bot "
        "working'. Returns the last N lines (default 200) — NOT a live stream."
    ),
    input_model=ReleaseLogsInput,
)
async def release_logs(input: ReleaseLogsInput, ctx: ToolContext) -> ToolResult:
    try:
        snap = await _backend(ctx).get_release_logs(
            input.release_id,
            tail=input.tail,
            since_seconds=input.since_seconds,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Release loglarini olish", e))

    lines = snap.get("lines") if isinstance(snap, dict) else None
    if not isinstance(lines, list):
        lines = []
    pod = snap.get("pod_name") if isinstance(snap, dict) else None
    truncated = bool(snap.get("truncated")) if isinstance(snap, dict) else False

    if not lines:
        detail = snap.get("detail") if isinstance(snap, dict) else None
        msg = detail or "No log lines available (pod may not be running yet)."
        return ToolResult.ok(msg, data={"logs": snap})

    head = "\n".join(lines[-50:])
    summary = (
        f"{len(lines)} line(s) from pod `{pod or '?'}`"
        f"{' (truncated to tail)' if truncated else ''}.\n```\n{head}\n```"
    )
    return ToolResult.ok(summary, data={"logs": snap})


# ── release.debug_logs ────────────────────────────────────────────────────────


def _clean_log_lines(raw_lines: Any) -> list[str]:
    """Normalise backend log payload into clean text lines (#8).

    Handles: bytes / bytes-repr (`b'...'`), single blobs containing escaped or
    real newlines, and already-split lists. Returns a flat list of trimmed,
    non-empty strings with surrounding quotes/`b` prefixes stripped.
    """
    out: list[str] = []
    items = raw_lines if isinstance(raw_lines, list) else [raw_lines]
    for item in items:
        if item is None:
            continue
        if isinstance(item, bytes):
            text = item.decode("utf-8", errors="replace")
        else:
            text = str(item)
        # Strip a leading bytes-literal wrapper: b'...' / b"..."
        if len(text) >= 3 and text[0] == "b" and text[1] in ("'", '"') and text[-1] == text[1]:
            text = text[2:-1]
        # Turn escaped newlines/tabs from a repr'd blob into real ones.
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        for line in text.split("\n"):
            line = line.rstrip("\r").strip()
            if line:
                out.append(line)
    return out


def _filter_log_lines(
    lines: list[str], *, level: str | None, node_id: str | None
) -> list[str]:
    """Filter by JSON `"level"` field substring and/or `node_id` substring."""
    if level:
        lvl = level.lower()
        # Match common JSON shapes: "level":"error" or level=error.
        needles = (f'"level":"{lvl}"', f'"level": "{lvl}"', f"level={lvl}")
        lines = [ln for ln in lines if any(n in ln.lower() for n in needles)]
    if node_id:
        lines = [ln for ln in lines if node_id in ln]
    return lines


class ReleaseDebugLogsInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Project id (defaults to the active project).",
    )
    tail: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Number of trailing log lines to return.",
    )
    since_seconds: int | None = Field(
        default=None,
        description="Only return lines from the last N seconds (optional).",
    )
    level: Literal["info", "warn", "error"] | None = Field(
        default=None,
        description="Keep only lines whose JSON `level` field matches (info/warn/error).",
    )
    node_id: str | None = Field(
        default=None,
        description="Keep only lines containing this node_id substring.",
    )


@tool(
    name="release.debug_logs",
    description=(
        "Fetch a snapshot of the DEBUG bot container's recent pod logs (the one "
        "started by release.debug). Use this when the debug bot misbehaves or "
        "doesn't respond — node errors, crashes, webhook problems show up here. "
        "Returns the last N lines (default 200) — NOT a live stream."
    ),
    input_model=ReleaseDebugLogsInput,
)
async def release_debug_logs(input: ReleaseDebugLogsInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error(
            "No project_id available. session_use_project yoki project_id bering."
        )
    try:
        snap = await _backend(ctx).get_debug_logs(
            pid,
            tail=input.tail,
            since_seconds=input.since_seconds,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Debug loglarini olish", e))

    raw_lines = snap.get("lines") if isinstance(snap, dict) else None
    pod = snap.get("pod_name") if isinstance(snap, dict) else None
    truncated = bool(snap.get("truncated")) if isinstance(snap, dict) else False

    # #8 — decode/clean noisy bytes & escaped newlines into plain text lines.
    lines = _clean_log_lines(raw_lines)
    total_before_filter = len(lines)
    lines = _filter_log_lines(lines, level=input.level, node_id=input.node_id)

    if not lines:
        detail = snap.get("detail") if isinstance(snap, dict) else None
        if total_before_filter and (input.level or input.node_id):
            msg = (
                f"{total_before_filter} qator bor edi, lekin filtr "
                f"(level={input.level}, node_id={input.node_id}) bo'yicha mos kelmadi."
            )
        else:
            msg = detail or (
                "Log qatorlari yo'q — debug konteyner ishlamayotgan bo'lishi mumkin. "
                "Avval release_debug bilan ishga tushiring."
            )
        return ToolResult.ok(msg, data={"lines": [], "pod_name": pod, "truncated": truncated})

    head = "\n".join(lines[-50:])
    flt = ""
    if input.level or input.node_id:
        flt = f" (filter: level={input.level}, node_id={input.node_id})"
    summary = (
        f"{len(lines)} line(s) from debug pod `{pod or '?'}`"
        f"{' (truncated to tail)' if truncated else ''}{flt}.\n```\n{head}\n```"
    )
    return ToolResult.ok(
        summary,
        data={"lines": lines, "pod_name": pod, "truncated": truncated},
    )


def build_release_tools() -> list[Tool]:
    return [
        release_list,
        release_deploy,
        release_debug,
        release_status,
        release_logs,
        release_debug_logs,
        plans_list,
        subscription_current,
        subscription_upgrade,
        payment_link_create,
    ]

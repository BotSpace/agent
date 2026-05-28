"""release.* tools — bot deploy lifecycle."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...integrations.backend import BackendClient, BackendError, get_backend_client
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


def _bearer(ctx: ToolContext) -> str | None:
    claims = ctx.state.get("auth_claims") or {}
    return claims.get("access_token") if isinstance(claims, dict) else None


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


@tool(
    name="release.list",
    description="List recent releases (deploys) for the current project.",
    input_model=ReleaseListInput,
)
async def release_list(input: ReleaseListInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id available.")
    try:
        items = await _backend(ctx).list_releases(pid, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(_wrap_backend_err("Releaselarni olish", e))
    items = items[: input.limit]
    payload = [
        {
            "id": r.get("id"),
            "version": r.get("version"),
            "status": r.get("status"),
            "created_at": r.get("created_at"),
        }
        for r in items
    ]
    summary = ", ".join(
        f"v{r['version']}({r['status']})" for r in payload if r.get("version")
    ) or "(none)"
    return ToolResult.ok(
        f"{len(payload)} release(s): {summary}",
        data={"releases": payload},
    )


class ReleaseDeployInput(BaseModel):
    project_id: str | int | None = None
    plan_id: int | None = Field(
        default=None,
        description=(
            "REQUIRED for the FIRST release of a project. Use plans.list to discover "
            "available plan ids. Subsequent releases do not need plan_id."
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
    plan_id: int = Field(description="Target plan id (from plans.list).")
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
    release_id: str | int


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


def build_release_tools() -> list[Tool]:
    return [
        release_list,
        release_deploy,
        release_status,
        plans_list,
        subscription_current,
        subscription_upgrade,
        payment_link_create,
    ]

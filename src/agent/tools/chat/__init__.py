"""chat.* tools — bot foydalanuvchilari va ularning xabar tarixi."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ...integrations.backend import (
    BackendClient,
    BackendError,
    format_backend_error,
    pagination_summary,
)
from ..base import Tool, ToolContext, ToolResult, tool


from .._common import backend_from_ctx as _backend, resolve_project_id


class ChatListUsersInput(BaseModel):
    project_id: str | int | None = None
    limit: int = Field(default=50, ge=1, le=500)
    page: int | None = Field(
        default=None,
        description="Page number. Paginated; omit for page 1, increment if total_pages > 1.",
    )
    page_size: int | None = Field(
        default=None,
        description="Items per page (default 10). Increase (e.g. 100) to fetch more at once.",
    )


@tool(
    name="chat.list_users",
    description=(
        "List Telegram users who interacted with the bot (with message stats). "
        "Paginated (10/page by default) — the result reports total_items/total_pages; "
        "pass `page`/`page_size` for more."
    ),
    input_model=ChatListUsersInput,
)
async def chat_list_users(input: ChatListUsersInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        users = await _backend(ctx).list_chat_users(
            pid, page=input.page, page_size=input.page_size, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    total = getattr(users, "total_items", len(users))
    note = pagination_summary(users)
    shown = list(users)[: input.limit]
    return ToolResult.ok(
        f"Showing {len(shown)} of {total} user(s).{note}",
        data={
            "users": shown,
            "total_items": total,
            "total_pages": getattr(users, "total_pages", 1),
            "current_page": getattr(users, "current_page", 1),
        },
    )


class ChatUserHistoryInput(BaseModel):
    chat_id: str | int = Field(description="Telegram chat (user) id")
    project_id: str | int | None = None


@tool(
    name="chat.user_history",
    description="Get full message history with a single user.",
    input_model=ChatUserHistoryInput,
)
async def chat_user_history(input: ChatUserHistoryInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        history = await _backend(ctx).user_chat_history(pid, input.chat_id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"{len(history)} message(s).", data={"history": history})


# ── chat.user_stats ───────────────────────────────────────────────────────────
#
# Bir LLM tool chaqiruvida loyiha foydalanuvchi statistikasini yig'adi. Ichida
# bir nechta backend chaqiruv qiladi (chat users + har bir collection count), lekin
# agent uchun bu BITTA tool. Avval CRM Manager bu statistikani 40+ cms.find/aggregate
# bilan qo'lda hisoblardi — bu tool o'sha samarasizlikni bartaraf etadi.

# Chat-user dict'da uchrashi mumkin bo'lgan "yangi" (ro'yxatdan o'tgan) sana maydonlari.
_CREATED_FIELDS = ("created", "created_at", "first_seen", "first_seen_at", "joined_at", "date_joined")
# Chat-user dict'da uchrashi mumkin bo'lgan "faollik" (oxirgi xabar) sana maydonlari.
_ACTIVE_FIELDS = (
    "last_message",
    "last_message_at",
    "last_seen",
    "last_seen_at",
    "last_activity",
    "updated_at",
    "date",
    "timestamp",
)


def _parse_dt(value: Any) -> datetime | None:
    """ISO string yoki unix timestamp (sec/ms) ni timezone-aware datetime ga aylantiradi.

    Tanib bo'lmasa None (best-effort — crash qilmaydi)."""
    if value is None or isinstance(value, bool):
        return None
    # Unix timestamp (int/float yoki raqamli string)
    if isinstance(value, (int, float)):
        ts = float(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            ts = float(s)
        except ValueError:
            # ISO 8601 ('Z' suffiks bilan ham)
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
    else:
        return None
    # Millisekund epoch bo'lsa sekundga keltir
    if ts > 1e11:
        ts /= 1000.0
    try:
        return datetime.fromtimestamp(ts, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _first_dt(user: dict[str, Any], fields: tuple[str, ...]) -> datetime | None:
    for f in fields:
        if f in user:
            dt = _parse_dt(user.get(f))
            if dt is not None:
                return dt
    return None


def _summarize_users(
    users: list[dict[str, Any]], days: int
) -> tuple[int, int | None, int | None, bool]:
    """(total, new_users, active_users, has_date_fields) qaytaradi.

    Sana maydonlari topilmasa new/active = None va has_date_fields = False."""
    total = len(users)
    cutoff = datetime.now(tz=UTC) - timedelta(days=days)

    has_created = any(any(f in u for f in _CREATED_FIELDS) for u in users if isinstance(u, dict))
    has_active = any(any(f in u for f in _ACTIVE_FIELDS) for u in users if isinstance(u, dict))

    new_users: int | None = 0 if has_created else None
    active_users: int | None = 0 if has_active else None

    for u in users:
        if not isinstance(u, dict):
            continue
        if has_created:
            cdt = _first_dt(u, _CREATED_FIELDS)
            if cdt is not None and cdt >= cutoff:
                new_users += 1  # type: ignore[operator]
        if has_active:
            adt = _first_dt(u, _ACTIVE_FIELDS)
            if adt is not None and adt >= cutoff:
                active_users += 1  # type: ignore[operator]

    return total, new_users, active_users, (has_created or has_active)


async def _collection_count(
    backend: BackendClient,
    pid: str | int,
    name: str,
    *,
    auth_claims: dict | None,
) -> int | None:
    """Bitta collection document sonini eng arzon usulda oladi.

    Avval server-side aggregate `$count`, u ishlamasa `find` total. Hech biri
    bo'lmasa None (crash QILMAYDI)."""
    try:
        agg = await backend.cms_aggregate(
            pid, name, [{"$count": "n"}], limit=1, auth_claims=auth_claims
        )
        results = agg.get("results") if isinstance(agg, dict) else None
        if isinstance(results, list) and results and isinstance(results[0], dict):
            n = results[0].get("n")
            if isinstance(n, (int, float)):
                return int(n)
        # Bo'sh collection — aggregate $count hech narsa qaytarmaydi → 0
        if isinstance(results, list):
            return 0
    except BackendError:
        pass
    # Fallback: find total (limit=1, faqat metadata kerak)
    try:
        res = await backend.cms_list_documents(
            pid, name, limit=1, skip=0, auth_claims=auth_claims
        )
        total = res.get("total") if isinstance(res, dict) else None
        if isinstance(total, (int, float)):
            return int(total)
    except BackendError:
        pass
    return None


class ChatUserStatsInput(BaseModel):
    project_id: str | int | None = None
    days: int = Field(
        default=30,
        ge=1,
        le=3650,
        description="Window (kun) — new_users_<days>d / active_users_<days>d uchun.",
    )


@tool(
    name="chat.user_stats",
    description=(
        "Loyiha foydalanuvchi statistikasini BITTA chaqiruvda yig'adi: jami Telegram "
        "chat foydalanuvchilari, oxirgi N kundagi yangi/faol foydalanuvchilar, va har bir "
        "CMS collection'dagi document soni. Foydalanuvchi statistikasi so'ralganda buni "
        "ishlat — cms.find/cms.aggregate bilan QO'LDA hisoblama."
    ),
    input_model=ChatUserStatsInput,
)
async def chat_user_stats(input: ChatUserStatsInput, ctx: ToolContext) -> ToolResult:
    pid = resolve_project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    backend = _backend(ctx)
    auth_claims = ctx.state.get("auth_claims")
    days = input.days

    # 1) Telegram chat foydalanuvchilari ────────────────────────────────────
    users: list[dict[str, Any]] = []
    users_error: str | None = None
    try:
        raw = await backend.list_chat_users(pid, auth_claims=auth_claims)
        users = [u for u in raw if isinstance(u, dict)] if isinstance(raw, list) else []
    except BackendError as e:
        users_error = format_backend_error("chat.list_users", e)

    total_users, new_users, active_users, has_date_fields = _summarize_users(users, days)

    # 2) CMS collection'lar + document soni ──────────────────────────────────
    collections: list[dict[str, Any]] = []
    collections_error: str | None = None
    try:
        cols = await backend.cms_list_collections(pid, auth_claims=auth_claims)
    except BackendError as e:
        collections_error = format_backend_error("cms.list_collections", e)
        cols = []

    for c in cols:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name:
            continue
        count = await _collection_count(backend, pid, name, auth_claims=auth_claims)
        collections.append({"name": name, "count": count})

    total_documents = sum(
        c["count"] for c in collections if isinstance(c.get("count"), int)
    )

    # 3) Strukturali natija ──────────────────────────────────────────────────
    data: dict[str, Any] = {
        "project_id": pid,
        "days": days,
        "total_chat_users": total_users,
        f"new_users_{days}d": new_users,
        f"active_users_{days}d": active_users,
        "collections": collections,
        "total_documents": total_documents,
    }
    if users_error:
        data["chat_users_error"] = users_error
    if collections_error:
        data["collections_error"] = collections_error

    # 4) Markdown xulosa ──────────────────────────────────────────────────────
    lines: list[str] = ["## Foydalanuvchi statistikasi", ""]
    if users_error:
        lines.append(f"- Telegram users: olinmadi ({users_error})")
    else:
        lines.append(f"- Jami Telegram users: **{total_users}**")
        new_txt = "n/a" if new_users is None else str(new_users)
        active_txt = "n/a" if active_users is None else str(active_users)
        lines.append(f"- Yangi (oxirgi {days} kun): **{new_txt}**")
        lines.append(f"- Faol (oxirgi {days} kun): **{active_txt}**")
        if not has_date_fields:
            lines.append(
                "  - (chat user ma'lumotida sana maydoni yo'q — new/active hisoblanmadi)"
            )

    lines.append("")
    if collections_error:
        lines.append(f"Collection'lar olinmadi: {collections_error}")
    elif collections:
        lines.append(f"### Collection'lar ({len(collections)})")
        lines.append("")
        lines.append("| Collection | Documents |")
        lines.append("| --- | --- |")
        for c in collections:
            cnt = c["count"]
            lines.append(f"| {c['name']} | {'n/a' if cnt is None else cnt} |")
        lines.append("")
        lines.append(f"Jami documents: **{total_documents}**")
    else:
        lines.append("Collection topilmadi.")

    return ToolResult.ok("\n".join(lines), data=data)


def build_chat_tools() -> list[Tool]:
    return [chat_list_users, chat_user_history, chat_user_stats]

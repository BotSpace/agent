"""broadcast.* tools — segmentlangan xabar yuborish."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...integrations.backend import (
    BackendError,
    format_backend_error,
    pagination_summary,
)
from ..base import Tool, ToolContext, ToolResult, tool


from .._backend import backend_from_ctx as _backend  # ponytail: umumiy resolver


def _project_id(ctx: ToolContext, override: str | int | None) -> str | int | None:
    return override or ctx.state.get("project_id")


class BroadcastPreviewInput(BaseModel):
    text: str = Field(description="Broadcast message text (variables like {{first_name}} allowed)")
    project_id: str | int | None = None
    target: str = Field(
        default="all",
        description="`all`, `segment` (uses last crm.segment result), or `custom`",
    )
    telegram_ids: list[int] | None = Field(
        default=None,
        description="Explicit list (for target=custom)",
    )


@tool(
    name="broadcast.preview",
    description=(
        "Preview a broadcast — shows the text, recipient count and a sample. "
        "Recommended before broadcast.send so the user can confirm (not enforced — "
        "send can also be called directly with an explicit telegram_ids list)."
    ),
    input_model=BroadcastPreviewInput,
)
async def broadcast_preview(input: BroadcastPreviewInput, ctx: ToolContext) -> ToolResult:
    if input.target == "segment":
        seg = ctx.state.get("last_segment") or {}
        ids = list(seg.get("ids") or [])
    elif input.target == "custom":
        ids = list(input.telegram_ids or [])
    else:
        # `all` — we don't enumerate; counts come from backend later
        ids = []

    sample = ids[:5]
    return ToolResult.ok(
        f"Broadcast preview: target={input.target}, recipients={len(ids) if ids else 'all'}",
        data={
            "text": input.text,
            "target": input.target,
            "recipient_count": len(ids) if ids else None,
            "sample": sample,
        },
        state_updates={
            "pending_broadcast": {
                "text": input.text,
                "target": input.target,
                "telegram_ids": ids if input.target != "all" else None,
                "project_id": _project_id(ctx, input.project_id),
            }
        },
    )


class BroadcastSendInput(BaseModel):
    """If `text` is omitted, the previously previewed broadcast is sent."""

    text: str | None = None
    project_id: str | int | None = None
    telegram_ids: list[int] | None = None


@tool(
    name="broadcast.send",
    description=(
        "Send a broadcast. Recipients come from an explicit `telegram_ids` list if "
        "given, otherwise the previously previewed segment, otherwise all users. "
        "Requires approval. Prefer broadcast.preview first to confirm recipients."
    ),
    input_model=BroadcastSendInput,
    destructive=True,
)
async def broadcast_send(input: BroadcastSendInput, ctx: ToolContext) -> ToolResult:
    pending = ctx.state.get("pending_broadcast") or {}
    text = input.text or pending.get("text")
    pid = _project_id(ctx, input.project_id) or pending.get("project_id")
    ids = input.telegram_ids if input.telegram_ids is not None else pending.get("telegram_ids")
    if not text:
        return ToolResult.error("No broadcast text — call broadcast.preview first.")
    if pid is None:
        return ToolResult.error("No project_id available.")

    body: dict[str, Any] = {"project": pid, "caption": text, "type": "text"}
    if ids:
        body["target_telegram_ids"] = ids
    try:
        msg = await _backend(ctx).create_message(body, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(
        f"Broadcast queued (message id={msg.get('id')}). Total recipients: {msg.get('total') or 'pending'}.",
        data={"message": msg},
        state_updates={"pending_broadcast": None, "last_broadcast_id": msg.get("id")},
    )


class BroadcastStatusInput(BaseModel):
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
    name="broadcast.status",
    description=(
        "List recent broadcasts and their delivery status. Paginated (10/page by default) — "
        "the result reports total_items/total_pages; pass `page`/`page_size` for more."
    ),
    input_model=BroadcastStatusInput,
)
async def broadcast_status(input: BroadcastStatusInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    try:
        items = await _backend(ctx).list_messages(
            pid, page=input.page, page_size=input.page_size, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    list_total = getattr(items, "total_items", len(items))
    note = pagination_summary(items)
    shown = list(items)[: input.limit]
    payload = [
        {
            "id": m.get("id"),
            "status": m.get("status"),
            "total": m.get("total"),
            "done": m.get("done"),
            "fail": m.get("fail"),
        }
        for m in shown
    ]
    lines = [f"Showing {len(payload)} of {list_total} broadcast(s):{note}"]
    for b in payload:
        done = b.get("done", 0)
        total = b.get("total", 0)
        fail = b.get("fail", 0)
        pct = f"{round(done / total * 100)}%" if total else "?"
        lines.append(f"  id={b['id']}  status={b.get('status')}  sent={done}/{total}({pct})  fail={fail}")
    return ToolResult.ok(
        "\n".join(lines) if payload else f"0 of {list_total} broadcast(s).{note}",
        data={
            "broadcasts": payload,
            "total_items": list_total,
            "total_pages": getattr(items, "total_pages", 1),
            "current_page": getattr(items, "current_page", 1),
        },
    )


def build_broadcast_tools() -> list[Tool]:
    return [broadcast_preview, broadcast_send, broadcast_status]

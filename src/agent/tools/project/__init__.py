"""project.* tools — bot loyihalarini boshqarish (Django backend orqali)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ...integrations.backend import (
    BackendError,
    format_backend_error,
    pagination_summary,
)
from ..base import Tool, ToolContext, ToolResult, tool


from .._common import backend_from_ctx as _backend


# ── project.list ─────────────────────────────────────────────────────────────


class ProjectListInput(BaseModel):
    search: str | None = Field(default=None, description="Filter by name")
    page: int | None = Field(
        default=None,
        description="Page number. Results are paginated; omit for page 1, "
        "then increment to fetch the rest when total_pages > 1.",
    )
    page_size: int | None = Field(
        default=None,
        description="Items per page (default 10). Increase (e.g. 100) to fetch more at once.",
    )


@tool(
    name="project.list",
    description=(
        "List bot projects belonging to the current user. Paginated (10/page by default) — "
        "the result reports total_items/total_pages; pass `page`/`page_size` to get more."
    ),
    input_model=ProjectListInput,
)
async def project_list(input: ProjectListInput, ctx: ToolContext) -> ToolResult:
    try:
        items = await _backend(ctx).list_projects(
            search=input.search,
            page=input.page,
            page_size=input.page_size,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    summary = [{"id": i.get("id"), "name": i.get("name"), "status": i.get("status")} for i in items]
    names = ", ".join(f"{s['name']} (id={s['id']})" for s in summary) or "(none)"
    total = getattr(items, "total_items", len(items))
    return ToolResult.ok(
        f"Showing {len(items)} of {total} project(s): {names}{pagination_summary(items)}",
        data={
            "projects": summary,
            "total_items": total,
            "total_pages": getattr(items, "total_pages", 1),
            "current_page": getattr(items, "current_page", 1),
        },
    )


# ── project.get ──────────────────────────────────────────────────────────────


class ProjectGetInput(BaseModel):
    id: str | int = Field(description="Project id")
    include_flow: bool = Field(default=False, description="Include the current flow JSON")


@tool(
    name="project.get",
    description=(
        "Fetch a single project's details. Set `include_flow=true` to also load the "
        "saved flow into the agent — this resets BOTH `baseline_flow` and "
        "`working_flow` to the project's saved flow (discards unsaved working changes)."
    ),
    input_model=ProjectGetInput,
)
async def project_get(input: ProjectGetInput, ctx: ToolContext) -> ToolResult:
    try:
        project = await _backend(ctx).get_project(input.id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))

    updates: dict[str, Any] = {"project_id": str(input.id)}
    if input.include_flow and isinstance(project.get("flow"), dict):
        updates["baseline_flow"] = project["flow"]
        updates["working_flow"] = project["flow"]
        updates["flow_dirty"] = False

    return ToolResult.ok(
        f"Project `{project.get('name', input.id)}` loaded.",
        data={"project": project},
        state_updates=updates,
    )


# ── project.create ───────────────────────────────────────────────────────────


# Minimal clean default flow: /start → welcome. Replaces any backend-injected
# promotional/demo flow (#13).
def _clean_default_flow() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "start",
                "type": "CommandTriggerNode",
                "data": {"command": "/start", "global": True},
                "position": {"x": 0, "y": 0},
            },
            {
                "id": "welcome",
                "type": "SendTextMessageNode",
                "data": {"messageText": "Assalomu alaykum! 👋"},
                "position": {"x": 320, "y": 0},
            },
        ],
        "edges": [
            {
                "id": "e-start-welcome",
                "source": "start",
                "target": "welcome",
                "sourceHandle": "source-handler",
                "targetHandle": "target-handler",
                "type": "smart-edge",
            }
        ],
    }


class ProjectCreateInput(BaseModel):
    name: str = Field(description="Bot name (3-50 chars)")
    description: str | None = None
    token: str | None = Field(default=None, description="Telegram bot token (BotFather)")
    default_flow: Literal["clean", "keep"] = Field(
        default="clean",
        description="'clean' (default) overwrites whatever flow the backend "
        "seeds with a minimal /start→welcome flow (drops any promo/demo content). "
        "'keep' leaves the backend's default flow untouched.",
    )


@tool(
    name="project.create",
    description=(
        "Create a new bot project. Returns the project id. By default replaces "
        "the backend's seeded flow with a clean /start→welcome flow "
        "(default_flow='keep' to skip)."
    ),
    input_model=ProjectCreateInput,
)
async def project_create(input: ProjectCreateInput, ctx: ToolContext) -> ToolResult:
    body = input.model_dump(exclude_none=True, exclude={"default_flow"})
    backend = _backend(ctx)
    try:
        project = await backend.create_project(body, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))

    project_id = str(project.get("id") or "")
    state_updates: dict[str, Any] = {"project_id": project_id}
    note = ""
    if input.default_flow == "clean" and project_id:
        clean = _clean_default_flow()
        try:
            await backend.update_project(
                project_id, {"flow": clean}, auth_claims=ctx.state.get("auth_claims")
            )
            # Seed agent working state so flow.* tools edit the clean flow.
            state_updates.update(
                {"working_flow": clean, "baseline_flow": clean, "flow_dirty": False}
            )
            note = " Boshlang'ich flow toza /start→welcome bilan almashtirildi."
        except BackendError:
            note = " (Diqqat: default flow'ni tozalab bo'lmadi — flow.clear bilan tozalashingiz mumkin.)"

    # #10 — single source of truth: do NOT echo the backend flow JSON here.
    # The flow that session_use_project loads (after the clean overwrite) is
    # authoritative. Return a compact summary + a pointer to load the session.
    summary: dict[str, Any] = {
        "id": project_id,
        "name": project.get("name"),
    }
    for key in ("debug_link", "debug_uuid", "status", "username"):
        if project.get(key):
            summary[key] = project[key]
    return ToolResult.ok(
        f"Project `{project.get('name')}` created with id `{project_id}`.{note} "
        "Endi `session_use_project(project_id)` chaqirib flow ustida ishlang.",
        data={"project": summary},
        state_updates=state_updates,
    )


# ── project.update ───────────────────────────────────────────────────────────


class ProjectUpdateInput(BaseModel):
    id: str | int
    name: str | None = None
    description: str | None = None
    token: str | None = None
    flow: dict[str, Any] | None = Field(
        default=None,
        description="Provide to overwrite the saved flow (usually you use flow.save instead).",
    )


@tool(
    name="project.update",
    description="Update a project's name, description, or token. To save a flow use flow.save.",
    input_model=ProjectUpdateInput,
)
async def project_update(input: ProjectUpdateInput, ctx: ToolContext) -> ToolResult:
    body = input.model_dump(exclude={"id"}, exclude_none=True)
    if not body:
        return ToolResult.error("Nothing to update.")
    try:
        project = await _backend(ctx).update_project(input.id, body, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"Project `{input.id}` updated.", data={"project": project})


# ── project.delete (destructive) ─────────────────────────────────────────────


class ProjectDeleteInput(BaseModel):
    id: str | int
    confirm: bool = Field(
        default=False,
        description=(
            "Tasdiqlash. False (default) — o'chirilMAYDI, nima o'chishi haqida "
            "xulosa qaytaradi. Faqat foydalanuvchi aniq tasdiqlagach confirm=true."
        ),
    )


@tool(
    name="project.delete",
    description=(
        "Delete a bot project permanently (flow, kolleksiyalar, releaselar bilan). "
        "QAYTARIB BO'LMAYDI. confirm=true talab qiladi — confirm berilmasa nima "
        "o'chishini qaytaradi, o'chirmaydi."
    ),
    input_model=ProjectDeleteInput,
    destructive=True,
)
async def project_delete(input: ProjectDeleteInput, ctx: ToolContext) -> ToolResult:
    if not input.confirm:
        name = input.id
        try:
            project = await _backend(ctx).get_project(
                input.id, auth_claims=ctx.state.get("auth_claims")
            )
            if isinstance(project, dict) and project.get("name"):
                name = f"{project['name']} (id={input.id})"
        except BackendError:
            pass
        return ToolResult.error(
            f"DIQQAT: `{name}` loyihasi butunlay o'chadi — flow, barcha "
            "kolleksiyalar va releaselar bilan, QAYTARIB BO'LMAYDI. Tasdiqlash "
            "uchun confirm=true bilan qayta chaqiring."
        )
    try:
        await _backend(ctx).delete_project(input.id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"Project `{input.id}` deleted.")


# ── project.errors (read-only — debug flow errors) ──────────────────────────


def _resolve_project_id(input_pid: str | int | None, ctx: ToolContext) -> str | None:
    """Explicit `project_id` arg wins; else fall back to the active session.

    Placeholders ('', 'global', 'none', 'null') count as "not given" — same list
    the MCP/flow layer uses — so the active-project pointer takes over.
    """
    for raw in (input_pid, ctx.project_id):
        if raw is None:
            continue
        text = str(raw).strip()
        if text and text.lower() not in {"global", "none", "null"}:
            return text
    return None


def _truncate(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


class ProjectErrorsInput(BaseModel):
    project_id: str | int | None = Field(
        default=None,
        description="Project id. Omit to use the active session's project.",
    )
    limit: int | None = Field(
        default=10,
        description="Max number of recent errors to return (newest first).",
    )
    include_detail: bool = Field(
        default=False,
        description=(
            "Include the raw `detail` (stack/raw error) per row. Off by default to "
            "save tokens — turn on only when the short message isn't enough to fix it."
        ),
    )


@tool(
    name="project.errors",
    description=(
        "Show recent runtime errors from the project's DEBUG flow run — which node "
        "failed and why (node_id, node_type, message, time). Use this FIRST when the "
        "bot doesn't work or throws an error in debug, to locate the broken node and "
        "fix the flow. Read-only. Newest first. Pass include_detail=true only if the "
        "short message isn't enough."
    ),
    input_model=ProjectErrorsInput,
)
async def project_errors(input: ProjectErrorsInput, ctx: ToolContext) -> ToolResult:
    pid = _resolve_project_id(input.project_id, ctx)
    if not pid:
        return ToolResult.error(
            "Qaysi loyiha? project_id bering yoki avval loyihani tanlang "
            "(session_use_project / project.get)."
        )
    try:
        rows = await _backend(ctx).list_project_errors(
            pid, limit=input.limit, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error("project.errors", e))

    if not rows:
        return ToolResult.ok(
            f"Loyiha {pid} uchun debug flow xatosi topilmadi — flow xatosiz "
            "ishlagan yoki hali debug rejimda ishlatilmagan.",
            data={"errors": [], "total": 0},
        )

    lines = [f"Loyiha {pid} — {len(rows)} ta oxirgi debug flow xatosi (yangi birinchi):"]
    payload: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        node_id = r.get("node_id")
        node_type = r.get("node_type")
        message = r.get("message") or r.get("detail")
        ts = r.get("ts")
        item: dict[str, Any] = {
            "node_id": node_id,
            "node_type": node_type,
            "message": message,
            "ts": ts,
        }
        node_label = f"{node_id}" + (f" ({node_type})" if node_type else "")
        line = f"  • node={node_label or '?'}: {_truncate(message, 240)}"
        if ts:
            line += f"  [{ts}]"
        lines.append(line)
        if input.include_detail and r.get("detail") and r.get("detail") != message:
            detail = _truncate(r.get("detail"), 600)
            lines.append(f"      detail: {detail}")
            item["detail"] = detail
        payload.append(item)

    lines.append(
        "Yuqoridagi node(lar)ni flow'da topib (flow.get) sababini tuzating, "
        "keyin flow.validate → flow.save."
    )
    return ToolResult.ok(
        "\n".join(lines),
        data={"errors": payload, "total": len(payload), "project_id": pid},
    )


def build_project_tools() -> list[Tool]:
    from .snapshot import build_project_snapshot_tools

    return [
        project_list,
        project_get,
        project_create,
        project_update,
        project_delete,
        project_errors,
        *build_project_snapshot_tools(),
    ]

"""project.* tools — bot loyihalarini boshqarish (Django backend orqali)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...integrations.backend import BackendClient, BackendError, get_backend_client, format_backend_error
from ..base import Tool, ToolContext, ToolResult, tool


def _backend(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()


def _bearer(ctx: ToolContext) -> str | None:
    claims = ctx.state.get("auth_claims") or {}
    return claims.get("access_token") if isinstance(claims, dict) else None


# ── project.list ─────────────────────────────────────────────────────────────


class ProjectListInput(BaseModel):
    search: str | None = Field(default=None, description="Filter by name")


@tool(
    name="project.list",
    description="List all bot projects belonging to the current user.",
    input_model=ProjectListInput,
)
async def project_list(input: ProjectListInput, ctx: ToolContext) -> ToolResult:
    try:
        items = await _backend(ctx).list_projects(search=input.search, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    summary = [{"id": i.get("id"), "name": i.get("name"), "status": i.get("status")} for i in items]
    names = ", ".join(f"{s['name']} (id={s['id']})" for s in summary) or "(none)"
    return ToolResult.ok(
        f"Found {len(items)} project(s): {names}",
        data={"projects": summary},
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


class ProjectCreateInput(BaseModel):
    name: str = Field(description="Bot name (3-50 chars)")
    description: str | None = None
    token: str | None = Field(default=None, description="Telegram bot token (BotFather)")


@tool(
    name="project.create",
    description="Create a new bot project. Returns the project id.",
    input_model=ProjectCreateInput,
)
async def project_create(input: ProjectCreateInput, ctx: ToolContext) -> ToolResult:
    body = input.model_dump(exclude_none=True)
    try:
        project = await _backend(ctx).create_project(body, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(
        f"Project `{project.get('name')}` created with id `{project.get('id')}`",
        data={"project": project},
        state_updates={"project_id": str(project.get("id") or "")},
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


@tool(
    name="project.delete",
    description="Delete a bot project permanently. Requires approval.",
    input_model=ProjectDeleteInput,
    destructive=True,
)
async def project_delete(input: ProjectDeleteInput, ctx: ToolContext) -> ToolResult:
    try:
        await _backend(ctx).delete_project(input.id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"Project `{input.id}` deleted.")


def build_project_tools() -> list[Tool]:
    return [project_list, project_get, project_create, project_update, project_delete]

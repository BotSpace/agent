"""release.* tools — bot deploy lifecycle."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...integrations.backend import BackendClient, BackendError, get_backend_client
from ..base import Tool, ToolContext, ToolResult, tool


def _backend(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()


def _bearer(ctx: ToolContext) -> str | None:
    claims = ctx.state.get("auth_claims") or {}
    return claims.get("access_token") if isinstance(claims, dict) else None


def _project_id(ctx: ToolContext, override: str | int | None) -> str | int | None:
    return override or ctx.state.get("project_id")


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
        return ToolResult.error(f"Backend error: {e}")
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


@tool(
    name="release.deploy",
    description=(
        "Deploy the saved flow of the project to production. This creates a new release "
        "(version) and brings up the engine container. Requires approval."
    ),
    input_model=ReleaseDeployInput,
    destructive=True,
)
async def release_deploy(input: ReleaseDeployInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id available.")
    try:
        release = await _backend(ctx).create_release(pid, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(f"Backend error: {e}")
    return ToolResult.ok(
        f"Deploy queued (release id={release.get('id')}, version={release.get('version')}).",
        data={"release": release},
    )


class ReleaseStatusInput(BaseModel):
    release_id: str | int


@tool(
    name="release.status",
    description="Get the status of a specific release (running/stopped/failed/pending).",
    input_model=ReleaseStatusInput,
)
async def release_status(input: ReleaseStatusInput, ctx: ToolContext) -> ToolResult:
    # Backend exposes status on the list view; we approximate by re-listing.
    pid = ctx.state.get("project_id")
    if pid is None:
        return ToolResult.error("No project_id in context.")
    try:
        items = await _backend(ctx).list_releases(pid, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(f"Backend error: {e}")
    target = next((r for r in items if str(r.get("id")) == str(input.release_id)), None)
    if target is None:
        return ToolResult.error(f"Release `{input.release_id}` not found.")
    return ToolResult.ok(f"Release `{input.release_id}` status: {target.get('status')}", data={"release": target})


def build_release_tools() -> list[Tool]:
    return [release_list, release_deploy, release_status]

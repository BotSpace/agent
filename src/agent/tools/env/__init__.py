"""env.* tools — environment variables for a project (Django backend)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...integrations.backend import (
    BackendClient,
    BackendError,
    format_backend_error,
    get_backend_client,
    pagination_summary,
)
from ..base import Tool, ToolContext, ToolResult, tool


def _backend(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()


def _bearer(ctx: ToolContext) -> str | None:
    claims = ctx.state.get("auth_claims") or {}
    return claims.get("access_token") if isinstance(claims, dict) else None


def _project_id(ctx: ToolContext, override: str | int | None = None) -> str | int | None:
    return override or ctx.state.get("project_id")


class EnvListInput(BaseModel):
    project_id: str | int | None = Field(default=None, description="Project id (defaults to current)")
    page: int | None = Field(
        default=None,
        description="Page number (10 per page). Paginated; omit for page 1, increment if total_pages > 1.",
    )


@tool(
    name="env.list",
    description=(
        "List environment variables for the current (or given) project. Paginated (10 per page) — "
        "the result reports total_items/total_pages; pass `page` for more."
    ),
    input_model=EnvListInput,
)
async def env_list(input: EnvListInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id in context or input.")
    try:
        items = await _backend(ctx).list_env(pid, page=input.page, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    safe = [{"id": i.get("id"), "key": i.get("key")} for i in items]
    total = getattr(items, "total_items", len(safe))
    note = pagination_summary(items)
    lines = [f"Showing {len(safe)} of {total} env var(s):{note}"]
    for s in safe:
        lines.append(f"  id={s['id']}  key={s['key']}")
    return ToolResult.ok(
        "\n".join(lines) if safe else f"0 of {total} env var(s).{note}",
        data={
            "env": safe,
            "total_items": total,
            "total_pages": getattr(items, "total_pages", 1),
            "current_page": getattr(items, "current_page", 1),
        },
    )


class EnvSetInput(BaseModel):
    key: str = Field(description="Env var name (UPPER_SNAKE_CASE recommended)")
    value: str = Field(description="Env var value (kept secret — never logged)")
    project_id: str | int | None = None


@tool(
    name="env.set",
    description=(
        "Create or replace an environment variable for the current project. "
        "Treat the value as secret — never echo it back."
    ),
    input_model=EnvSetInput,
)
async def env_set(input: EnvSetInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id in context or input.")
    try:
        await _backend(ctx).create_env(
            {"project": pid, "key": input.key, "value": input.value},
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"Env var `{input.key}` set for project `{pid}`.")


class EnvUnsetInput(BaseModel):
    env_id: str | int


@tool(
    name="env.unset",
    description=(
        "Remove an environment variable by its globally-unique `env_id` "
        "(no project_id needed — unlike env.set/env.list which are project-scoped)."
    ),
    input_model=EnvUnsetInput,
    destructive=True,
)
async def env_unset(input: EnvUnsetInput, ctx: ToolContext) -> ToolResult:
    try:
        await _backend(ctx).delete_env(input.env_id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"Env var `{input.env_id}` removed.")


def build_env_tools() -> list[Tool]:
    return [env_list, env_set, env_unset]

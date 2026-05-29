"""env.* tools — environment variables for a project (Django backend)."""

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


def _project_id(ctx: ToolContext, override: str | int | None = None) -> str | int | None:
    return override or ctx.state.get("project_id")


class EnvListInput(BaseModel):
    project_id: str | int | None = Field(default=None, description="Project id (defaults to current)")


@tool(
    name="env.list",
    description="List environment variables for the current (or given) project.",
    input_model=EnvListInput,
)
async def env_list(input: EnvListInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id in context or input.")
    try:
        items = await _backend(ctx).list_env(pid, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(f"Backend error: {e}")
    safe = [{"id": i.get("id"), "key": i.get("key")} for i in items]
    keys = ", ".join(str(s["key"]) for s in safe if s.get("key")) or "(none)"
    return ToolResult.ok(
        f"{len(items)} env var(s). Keys: {keys}",
        data={"env": safe},
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
        return ToolResult.error(f"Backend error: {e}")
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
        return ToolResult.error(f"Backend error: {e}")
    return ToolResult.ok(f"Env var `{input.env_id}` removed.")


def build_env_tools() -> list[Tool]:
    return [env_list, env_set, env_unset]

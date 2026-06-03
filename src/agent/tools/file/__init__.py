"""file.* tools — project files (Django backend FileModel)."""

from __future__ import annotations

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


class FileListInput(BaseModel):
    project_id: str | int | None = None
    limit: int = Field(default=50, ge=1, le=200)


@tool(
    name="file.list",
    description="List files belonging to a project (or all user files when project omitted).",
    input_model=FileListInput,
)
async def file_list(input: FileListInput, ctx: ToolContext) -> ToolResult:
    pid = input.project_id or ctx.state.get("project_id")
    try:
        items = await _backend(ctx).list_files(pid, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    items = items[: input.limit]
    payload = [
        {"id": f.get("id"), "name": f.get("name"), "url": f.get("url"), "size": f.get("size")}
        for f in items
    ]
    lines = [f"{len(payload)} file(s):"]
    for f in payload:
        size_kb = f"{round(f['size'] / 1024, 1)} KB" if f.get("size") else "?"
        lines.append(f"  id={f['id']}  name={f['name']}  size={size_kb}  url={f.get('url', '')}")
    return ToolResult.ok("\n".join(lines) if payload else "0 file(s).", data={"files": payload})


class FileDeleteInput(BaseModel):
    file_id: str | int


@tool(
    name="file.delete",
    description="Delete a file by id.",
    input_model=FileDeleteInput,
    destructive=True,
)
async def file_delete(input: FileDeleteInput, ctx: ToolContext) -> ToolResult:
    try:
        await _backend(ctx).delete_file(input.file_id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"File `{input.file_id}` deleted.")


def build_file_tools() -> list[Tool]:
    return [file_list, file_delete]

"""file.* tools — project files (Django backend FileModel)."""

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
        return ToolResult.error(f"Backend error: {e}")
    items = items[: input.limit]
    payload = [
        {"id": f.get("id"), "name": f.get("name"), "url": f.get("url"), "size": f.get("size")}
        for f in items
    ]
    names = ", ".join(str(f.get("name") or f.get("id")) for f in payload) or "(none)"
    return ToolResult.ok(
        f"{len(payload)} file(s): {names}",
        data={"files": payload},
    )


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
        return ToolResult.error(f"Backend error: {e}")
    return ToolResult.ok(f"File `{input.file_id}` deleted.")


def build_file_tools() -> list[Tool]:
    return [file_list, file_delete]

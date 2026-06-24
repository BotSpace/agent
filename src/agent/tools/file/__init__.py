"""file.* tools — project files (Django backend FileModel)."""

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


class FileListInput(BaseModel):
    project_id: str | int | None = None
    limit: int = Field(default=50, ge=1, le=200)
    page: int | None = Field(
        default=None,
        description="Page number. Paginated; omit for page 1, increment if total_pages > 1.",
    )
    page_size: int | None = Field(
        default=None,
        description="Items per page (default 10). Increase (e.g. 100) to fetch more at once.",
    )


@tool(
    name="file.list",
    description=(
        "List files belonging to a project (or all user files when project omitted). "
        "Paginated (10/page by default) — the result reports total_items/total_pages; "
        "pass `page`/`page_size` for more."
    ),
    input_model=FileListInput,
)
async def file_list(input: FileListInput, ctx: ToolContext) -> ToolResult:
    pid = input.project_id or ctx.state.get("project_id")
    try:
        items = await _backend(ctx).list_files(
            pid, page=input.page, page_size=input.page_size, auth_claims=ctx.state.get("auth_claims")
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error(None, e))
    total = getattr(items, "total_items", len(items))
    note = pagination_summary(items)
    shown = list(items)[: input.limit]
    payload = [
        {
            "id": f.get("id"),
            "name": f.get("name"),
            "url": f.get("url"),
            "size": f.get("size"),
            # expires_at=None -> stored permanently; otherwise the file is auto-deleted then (TTL).
            "expires_at": f.get("expires_at"),
        }
        for f in shown
    ]
    lines = [f"Showing {len(payload)} of {total} file(s):{note}"]
    for f in payload:
        size_kb = f"{round(f['size'] / 1024, 1)} KB" if f.get("size") else "?"
        ttl = "permanent" if not f.get("expires_at") else f"expires {f['expires_at']}"
        lines.append(
            f"  id={f['id']}  name={f['name']}  size={size_kb}  {ttl}  url={f.get('url', '')}"
        )
    return ToolResult.ok(
        "\n".join(lines) if payload else f"0 of {total} file(s).{note}",
        data={
            "files": payload,
            "total_items": total,
            "total_pages": getattr(items, "total_pages", 1),
            "current_page": getattr(items, "current_page", 1),
        },
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
        return ToolResult.error(format_backend_error(None, e))
    return ToolResult.ok(f"File `{input.file_id}` deleted.")


def build_file_tools() -> list[Tool]:
    return [file_list, file_delete]

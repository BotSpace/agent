"""file.* tools — project files (Django backend FileModel)."""

from __future__ import annotations

import mimetypes

import httpx
from pydantic import BaseModel, Field, model_validator

from ...integrations.backend import (
    BackendError,
    format_backend_error,
    pagination_summary,
)
from ..base import Tool, ToolContext, ToolResult, tool


from .._backend import backend_from_ctx as _backend  # ponytail: umumiy resolver


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


class FileUploadInput(BaseModel):
    filename: str = Field(description="Name of the file to upload (e.g. report.pdf).")
    content: str | None = Field(
        default=None,
        description="Text content of the file. Mutually exclusive with `url`.",
    )
    url: str | None = Field(
        default=None,
        description="Public URL to download the file from. Mutually exclusive with `content`.",
    )
    project_id: str | int | None = Field(
        default=None,
        description="Project id/uuid. Defaults to the current session project.",
    )
    ttl: int | None = Field(
        default=None,
        ge=1,
        description="Time-to-live in seconds. Omit (or pass null) for permanent storage.",
    )

    @model_validator(mode="after")
    def _check_source(self) -> "FileUploadInput":
        if self.content is None and self.url is None:
            raise ValueError("Provide either `content` (text) or `url` (download source).")
        if self.content is not None and self.url is not None:
            raise ValueError("`content` and `url` are mutually exclusive.")
        return self


@tool(
    name="file.upload",
    description=(
        "Upload a file to the project (optional ttl in seconds; omit for permanent). "
        "Supply either `content` (plain text) or `url` (file will be downloaded first)."
    ),
    input_model=FileUploadInput,
)
async def file_upload(input: FileUploadInput, ctx: ToolContext) -> ToolResult:
    pid = input.project_id or ctx.state.get("project_id")
    if not pid:
        return ToolResult.error("project_id talab qilinadi — ctx.state da ham yo'q.")

    if input.url:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
                resp = await http.get(input.url)
                resp.raise_for_status()
            content_bytes = resp.content
            content_type = (
                resp.headers.get("content-type", "").split(";")[0].strip()
                or _mime_from_name(input.filename)
            )
        except httpx.HTTPError as exc:
            return ToolResult.error(f"URL dan yuklab olishda xato: {exc}")
    else:
        assert input.content is not None
        content_bytes = input.content.encode("utf-8")
        content_type = _mime_from_name(input.filename)

    try:
        result = await _backend(ctx).upload_file(
            input.filename,
            content_bytes,
            content_type,
            project=pid,
            ttl=input.ttl or None,
            auth_claims=ctx.state.get("auth_claims"),
        )
    except BackendError as e:
        return ToolResult.error(format_backend_error("file.upload", e))

    file_id = result.get("id")
    name = result.get("name") or input.filename
    url = result.get("url", "")
    expires_at = result.get("expires_at")
    ttl_note = "permanent" if not expires_at else f"expires {expires_at}"
    size = result.get("size")
    size_note = f"  size={round(size / 1024, 1)} KB" if size else ""

    return ToolResult.ok(
        f"Uploaded: id={file_id}  name={name}{size_note}  {ttl_note}\n  url={url}",
        data={
            "id": file_id,
            "name": name,
            "url": url,
            "expires_at": expires_at,
            "size": size,
            "mime": result.get("mime"),
            "uuid": result.get("uuid"),
        },
    )


def _mime_from_name(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def build_file_tools() -> list[Tool]:
    return [file_list, file_delete, file_upload]

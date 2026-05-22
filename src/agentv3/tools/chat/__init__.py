"""chat.* tools — bot foydalanuvchilari va ularning xabar tarixi."""

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


class ChatListUsersInput(BaseModel):
    project_id: str | int | None = None
    limit: int = Field(default=50, ge=1, le=500)


@tool(
    name="chat.list_users",
    description="List Telegram users who interacted with the bot (with message stats).",
    input_model=ChatListUsersInput,
)
async def chat_list_users(input: ChatListUsersInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        users = await _backend(ctx).list_chat_users(pid, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(f"Backend error: {e}")
    users = users[: input.limit]
    return ToolResult.ok(f"{len(users)} user(s).", data={"users": users})


class ChatUserHistoryInput(BaseModel):
    chat_id: str | int = Field(description="Telegram chat (user) id")
    project_id: str | int | None = None


@tool(
    name="chat.user_history",
    description="Get full message history with a single user.",
    input_model=ChatUserHistoryInput,
)
async def chat_user_history(input: ChatUserHistoryInput, ctx: ToolContext) -> ToolResult:
    pid = _project_id(ctx, input.project_id)
    if pid is None:
        return ToolResult.error("No project_id.")
    try:
        history = await _backend(ctx).user_chat_history(pid, input.chat_id, auth_claims=ctx.state.get("auth_claims"))
    except BackendError as e:
        return ToolResult.error(f"Backend error: {e}")
    return ToolResult.ok(f"{len(history)} message(s).", data={"history": history})


def build_chat_tools() -> list[Tool]:
    return [chat_list_users, chat_user_history]

"""kanban.* tools — owner-level Board and card management.

Board is a standalone workspace domain.  It is intentionally separate from
bot projects and CMS collections: a board may be linked to zero or many bots.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...integrations.backend import BackendError, format_backend_error
from .._common import backend_from_ctx as _backend
from ..base import Tool, ToolContext, ToolResult, tool


def _claims(ctx: ToolContext) -> dict[str, Any] | None:
    return ctx.state.get("auth_claims")


def _error(action: str, error: BackendError) -> ToolResult:
    return ToolResult.error(format_backend_error(action, error))


class BoardListInput(BaseModel):
    project_id: str | None = Field(
        default=None,
        description="Optional bot/project UUID; only boards linked to that bot are returned.",
    )
    archived: bool = Field(default=False, description="True returns archived boards only.")


@tool(
    name="kanban.list_boards",
    description=(
        "List the current owner's standalone Board/Kanban boards. Use this for Board, "
        "Kanban, lead pipeline, task board, card or column requests — these are NOT CMS "
        "collections and do not require creating a bot project."
    ),
    input_model=BoardListInput,
)
async def list_boards(input: BoardListInput, ctx: ToolContext) -> ToolResult:
    try:
        boards = await _backend(ctx).list_kanban_boards(
            project_id=input.project_id,
            archived=input.archived,
            auth_claims=_claims(ctx),
        )
    except BackendError as error:
        return _error("kanban.list_boards", error)
    compact = [
        {
            "id": board.get("id"),
            "name": board.get("name"),
            "card_type_label": board.get("card_type_label"),
            "card_count": board.get("card_count", 0),
            "project_ids": board.get("project_ids", []),
            "archived_at": board.get("archived_at"),
        }
        for board in boards
    ]
    return ToolResult.ok(
        f"{len(compact)} ta {'arxivlangan' if input.archived else 'faol'} Board topildi:\n"
        + json.dumps(compact, ensure_ascii=False),
        data={"boards": compact},
        data_in_content=True,
    )


class BoardGetInput(BaseModel):
    board_id: str = Field(description="Board UUID from kanban.list_boards.")
    project_id: str | None = Field(
        default=None,
        description="Optional bot/project UUID to filter the returned cards.",
    )


@tool(
    name="kanban.get_board",
    description=(
        "Get one Board with its columns, cards, custom field schema and linked bots. "
        "Call before editing cards so real column UUIDs are used."
    ),
    input_model=BoardGetInput,
    untrusted_output=True,
)
async def get_board(input: BoardGetInput, ctx: ToolContext) -> ToolResult:
    try:
        board = await _backend(ctx).get_kanban_board(
            input.board_id, project_id=input.project_id, auth_claims=_claims(ctx)
        )
    except BackendError as error:
        return _error("kanban.get_board", error)
    return ToolResult.ok(
        json.dumps(board, ensure_ascii=False, default=str),
        data={"board": board},
        data_in_content=True,
    )


class ColumnSpec(BaseModel):
    name: str
    color: str = "#64748b"
    wip_limit: int | None = Field(default=None, ge=1)
    is_terminal: bool = False


class BoardCreateInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    icon: str = "layout-kanban"
    color: str = "#6366f1"
    card_type_label: str = "Karta"
    field_schema: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional Board card custom-field definitions; this is not a CMS schema.",
    )
    project_ids: list[str] = Field(
        default_factory=list,
        description="Existing bot UUIDs to link. Empty makes a global Board.",
    )
    columns: list[ColumnSpec] = Field(
        default_factory=list,
        description="Optional columns. Empty uses backend defaults: Yangi, Jarayonda, Bajarildi.",
    )


@tool(
    name="kanban.create_board",
    description=(
        "Create a standalone Board/Kanban pipeline, optionally linked to existing bots. "
        "Never create a CMS collection or a new bot as a substitute for this operation."
    ),
    input_model=BoardCreateInput,
)
async def create_board(input: BoardCreateInput, ctx: ToolContext) -> ToolResult:
    body = input.model_dump()
    try:
        board = await _backend(ctx).create_kanban_board(body, auth_claims=_claims(ctx))
    except BackendError as error:
        return _error("kanban.create_board", error)
    return ToolResult.ok(
        f"Board `{board.get('name')}` yaratildi (id={board.get('id')}).",
        data={"board": board},
    )


class BoardUpdateInput(BaseModel):
    board_id: str
    name: str | None = Field(default=None, max_length=120)
    description: str | None = None
    icon: str | None = None
    color: str | None = None
    card_type_label: str | None = None
    field_schema: list[dict[str, Any]] | None = None


@tool(
    name="kanban.update_board",
    description="Update Board metadata or card custom fields; omitted values are preserved.",
    input_model=BoardUpdateInput,
)
async def update_board(input: BoardUpdateInput, ctx: ToolContext) -> ToolResult:
    backend = _backend(ctx)
    try:
        current = await backend.get_kanban_board(input.board_id, auth_claims=_claims(ctx))
        patch = input.model_dump(exclude={"board_id"}, exclude_none=True)
        body = {
            "name": current.get("name", ""),
            "description": current.get("description", ""),
            "icon": current.get("icon", "layout-kanban"),
            "color": current.get("color", "#6366f1"),
            "card_type_label": current.get("card_type_label", "Karta"),
            "field_schema": current.get("field_schema") or [],
            **patch,
        }
        board = await backend.update_kanban_board(input.board_id, body, auth_claims=_claims(ctx))
    except BackendError as error:
        return _error("kanban.update_board", error)
    return ToolResult.ok("Board yangilandi.", data={"board": board})


class BoardArchiveInput(BaseModel):
    board_id: str
    action: Literal["archive", "restore"]


@tool(
    name="kanban.set_board_archive",
    description="Archive a Board or restore an archived Board. Archive is reversible.",
    input_model=BoardArchiveInput,
)
async def set_board_archive(input: BoardArchiveInput, ctx: ToolContext) -> ToolResult:
    try:
        if input.action == "archive":
            await _backend(ctx).archive_kanban_board(input.board_id, auth_claims=_claims(ctx))
            board = None
        else:
            board = await _backend(ctx).restore_kanban_board(input.board_id, auth_claims=_claims(ctx))
    except BackendError as error:
        return _error("kanban.set_board_archive", error)
    return ToolResult.ok(
        "Board arxivlandi." if input.action == "archive" else "Board tiklandi.",
        data={"board": board} if board else None,
    )


class BoardLinkProjectInput(BaseModel):
    board_id: str
    project_id: str = Field(description="Existing bot/project UUID. Never create one implicitly.")
    linked: bool = Field(description="True links the bot; false unlinks it.")


@tool(
    name="kanban.link_project",
    description=(
        "Link or unlink an EXISTING bot to a Board. A Board can be global or linked to many bots; "
        "this never creates a bot project."
    ),
    input_model=BoardLinkProjectInput,
)
async def link_project(input: BoardLinkProjectInput, ctx: ToolContext) -> ToolResult:
    try:
        await _backend(ctx).link_kanban_project(
            input.board_id,
            input.project_id,
            linked=input.linked,
            auth_claims=_claims(ctx),
        )
    except BackendError as error:
        return _error("kanban.link_project", error)
    verb = "bog'landi" if input.linked else "ajratildi"
    return ToolResult.ok(f"Bot Board'dan {verb}.")


class CardCreateInput(BaseModel):
    board_id: str
    column_id: str = Field(description="Real column UUID from kanban.get_board.")
    title: str = Field(min_length=1, max_length=240)
    description: str = ""
    kind: str = "task"
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    due_at: str | None = Field(default=None, description="ISO-8601 timestamp.")
    source_project_id: str | None = Field(
        default=None, description="Optional existing bot UUID that produced/owns this card."
    )
    external_key: str | None = None
    custom_fields: dict[str, Any] = Field(default_factory=dict)


@tool(
    name="kanban.create_card",
    description=(
        "Create a task/lead/card on an existing Board column. First call kanban.get_board and "
        "use a real column_id; do not store the card as a CMS document."
    ),
    input_model=CardCreateInput,
)
async def create_card(input: CardCreateInput, ctx: ToolContext) -> ToolResult:
    body = input.model_dump(exclude={"board_id"}, exclude_none=True)
    try:
        card = await _backend(ctx).create_kanban_card(input.board_id, body, auth_claims=_claims(ctx))
    except BackendError as error:
        return _error("kanban.create_card", error)
    return ToolResult.ok(
        f"Board kartasi `{card.get('title')}` yaratildi (id={card.get('id')}).",
        data={"card": card},
    )


class CardUpdateInput(BaseModel):
    card_id: str
    title: str | None = Field(default=None, max_length=240)
    description: str | None = None
    kind: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] | None = None
    due_at: str | None = None
    custom_fields: dict[str, Any] | None = None


@tool(
    name="kanban.update_card",
    description="Update an existing Board card; omitted values are preserved.",
    input_model=CardUpdateInput,
)
async def update_card(input: CardUpdateInput, ctx: ToolContext) -> ToolResult:
    backend = _backend(ctx)
    try:
        current = await backend.get_kanban_card(input.card_id, auth_claims=_claims(ctx))
        patch = input.model_dump(exclude={"card_id"}, exclude_none=True)
        body = {
            "title": current.get("title", ""),
            "description": current.get("description", ""),
            "kind": current.get("kind", "task"),
            "priority": current.get("priority", "normal"),
            "due_at": current.get("due_at"),
            "custom_fields": current.get("custom_fields") or {},
            **patch,
        }
        card = await backend.update_kanban_card(input.card_id, body, auth_claims=_claims(ctx))
    except BackendError as error:
        return _error("kanban.update_card", error)
    return ToolResult.ok("Board kartasi yangilandi.", data={"card": card})


class CardMoveInput(BaseModel):
    card_id: str
    column_id: str = Field(description="Destination column UUID from kanban.get_board.")
    position: int = Field(default=0, ge=0)


@tool(
    name="kanban.move_card",
    description="Move/reorder a Board card to a real destination column and zero-based position.",
    input_model=CardMoveInput,
)
async def move_card(input: CardMoveInput, ctx: ToolContext) -> ToolResult:
    try:
        card = await _backend(ctx).move_kanban_card(
            input.card_id,
            input.column_id,
            input.position,
            auth_claims=_claims(ctx),
        )
    except BackendError as error:
        return _error("kanban.move_card", error)
    return ToolResult.ok("Board kartasi ko'chirildi.", data={"card": card})


class CardArchiveInput(BaseModel):
    card_id: str
    action: Literal["archive", "restore"]


@tool(
    name="kanban.set_card_archive",
    description="Archive a Board card or restore it to its original column. Archive is reversible.",
    input_model=CardArchiveInput,
)
async def set_card_archive(input: CardArchiveInput, ctx: ToolContext) -> ToolResult:
    try:
        if input.action == "archive":
            card = await _backend(ctx).archive_kanban_card(input.card_id, auth_claims=_claims(ctx))
        else:
            card = await _backend(ctx).restore_kanban_card(input.card_id, auth_claims=_claims(ctx))
    except BackendError as error:
        return _error("kanban.set_card_archive", error)
    return ToolResult.ok(
        "Board kartasi arxivlandi." if input.action == "archive" else "Board kartasi tiklandi.",
        data={"card": card},
    )


class ArchivedCardListInput(BaseModel):
    board_id: str
    project_id: str | None = None


@tool(
    name="kanban.list_archived_cards",
    description="List archived cards of an active Board so a user can choose one to restore.",
    input_model=ArchivedCardListInput,
    untrusted_output=True,
)
async def list_archived_cards(input: ArchivedCardListInput, ctx: ToolContext) -> ToolResult:
    try:
        cards = await _backend(ctx).list_archived_kanban_cards(
            input.board_id,
            project_id=input.project_id,
            auth_claims=_claims(ctx),
        )
    except BackendError as error:
        return _error("kanban.list_archived_cards", error)
    return ToolResult.ok(
        f"{len(cards)} ta arxivlangan karta:\n" + json.dumps(list(cards), ensure_ascii=False, default=str),
        data={"cards": list(cards)},
        data_in_content=True,
    )


def build_kanban_tools() -> list[Tool]:
    return [
        list_boards,
        get_board,
        create_board,
        update_board,
        set_board_archive,
        link_project,
        create_card,
        update_card,
        move_card,
        set_card_archive,
        list_archived_cards,
    ]

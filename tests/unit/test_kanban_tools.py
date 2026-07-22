from __future__ import annotations

from typing import Any

import pytest

from agent.tools import kanban as kanban_module
from agent.tools.base import ToolContext
from agent.tools.kanban import build_kanban_tools


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def list_kanban_boards(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("list", kwargs))
        return [{"id": "board-1", "name": "Sales", "project_ids": []}]

    async def create_kanban_board(self, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create", body))
        return {"id": "board-2", **body}

    async def get_kanban_board(self, board_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get", board_id))
        return {
            "id": board_id,
            "name": "Sales",
            "description": "",
            "icon": "layout-kanban",
            "color": "#6366f1",
            "card_type_label": "Lead",
            "field_schema": [],
            "columns": [{"id": "column-1", "name": "Yangi"}],
            "cards": [],
        }


def _ctx(backend: FakeBackend) -> ToolContext:
    return ToolContext(
        run_id="run-1",
        project_id=None,
        user_id="user-1",
        state={"auth_claims": {"user_id": "user-1"}},
        services={"backend": backend},
    )


def _tools() -> dict[str, Any]:
    return {tool.name: tool for tool in build_kanban_tools()}


@pytest.mark.asyncio
async def test_board_intent_lists_global_boards_without_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(kanban_module, "_backend", lambda _ctx: backend)
    result = await _tools()["kanban.list_boards"].run({}, _ctx(backend))

    assert not result.is_error
    assert "Sales" in result.content
    assert backend.calls == [
        (
            "list",
            {
                "project_id": None,
                "archived": False,
                "auth_claims": {"user_id": "user-1"},
            },
        )
    ]


@pytest.mark.asyncio
async def test_create_board_uses_board_api_not_project_or_cms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(kanban_module, "_backend", lambda _ctx: backend)
    result = await _tools()["kanban.create_board"].run(
        {"name": "Leadlar", "card_type_label": "Lead"}, _ctx(backend)
    )

    assert not result.is_error
    assert backend.calls[0][0] == "create"
    body = backend.calls[0][1]
    assert body["name"] == "Leadlar"
    assert body["field_schema"] == []
    assert body["project_ids"] == []


def test_kanban_tools_cover_board_card_and_archive_operations() -> None:
    names = set(_tools())
    assert {
        "kanban.list_boards",
        "kanban.get_board",
        "kanban.create_board",
        "kanban.update_board",
        "kanban.set_board_archive",
        "kanban.link_project",
        "kanban.create_card",
        "kanban.update_card",
        "kanban.move_card",
        "kanban.set_card_archive",
        "kanban.list_archived_cards",
    } <= names

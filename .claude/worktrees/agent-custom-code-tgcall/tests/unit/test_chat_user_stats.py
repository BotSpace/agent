"""chat.user_stats tool — bitta chaqiruvda foydalanuvchi statistikasi."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agent.integrations.backend import BackendClient, BackendError
from agent.tools.base import ToolContext
from agent.tools.chat import build_chat_tools


def _tools():
    return {t.name: t for t in build_chat_tools()}


class FakeBackend(BackendClient):
    """BackendClient subclass — HTTP yo'q, in-memory javoblar."""

    def __init__(
        self,
        *,
        users: list[dict[str, Any]] | None = None,
        collections: list[dict[str, Any]] | None = None,
        counts: dict[str, int] | None = None,
        users_raise: bool = False,
        list_collections_raise: bool = False,
        count_raise_for: set[str] | None = None,
        aggregate_unsupported: bool = False,
    ) -> None:
        # super().__init__ ni chaqirmaymiz — httpx client kerak emas.
        self._users = users or []
        self._collections = collections or []
        self._counts = counts or {}
        self._users_raise = users_raise
        self._list_collections_raise = list_collections_raise
        self._count_raise_for = count_raise_for or set()
        self._aggregate_unsupported = aggregate_unsupported

    async def list_chat_users(self, project_id, *, auth_claims=None):  # type: ignore[override]
        if self._users_raise:
            raise BackendError("boom", status=500)
        return list(self._users)

    async def cms_list_collections(self, project_id, *, auth_claims=None):  # type: ignore[override]
        if self._list_collections_raise:
            raise BackendError("boom", status=500)
        return list(self._collections)

    async def cms_aggregate(self, project_id, name, pipeline, *, limit=500, auth_claims=None):  # type: ignore[override]
        if self._aggregate_unsupported or name in self._count_raise_for:
            raise BackendError("aggregate not supported", status=400)
        n = self._counts.get(name, 0)
        if n == 0:
            return {"results": []}
        return {"results": [{"n": n}], "count": 1}

    async def cms_list_documents(self, project_id, name, *, filter=None, sort=None, limit=50, skip=0, auth_claims=None):  # type: ignore[override]
        if name in self._count_raise_for:
            raise BackendError("boom", status=500)
        return {"results": [], "total": self._counts.get(name, 0), "limit": limit, "skip": skip}


def _ctx(backend: BackendClient) -> ToolContext:
    return ToolContext(
        run_id="r",
        project_id="42",
        user_id="u",
        state={"project_id": "42"},
        services={"backend": backend},
    )


@pytest.mark.asyncio
async def test_user_stats_aggregates_users_and_collections():
    now = datetime.now(tz=UTC)
    recent = (now - timedelta(days=2)).isoformat()
    old = (now - timedelta(days=200)).isoformat()
    backend = FakeBackend(
        users=[
            {"chat_id": 1, "created_at": recent, "last_message": recent},
            {"chat_id": 2, "created_at": old, "last_message": recent},
            {"chat_id": 3, "created_at": old, "last_message": old},
        ],
        collections=[{"name": "users"}, {"name": "orders"}, {"name": "empty"}],
        counts={"users": 120, "orders": 7, "empty": 0},
    )
    res = await _tools()["chat.user_stats"].run({"days": 30}, _ctx(backend))

    assert not res.is_error
    d = res.data
    assert d["total_chat_users"] == 3
    # 1 ta created recent (chat_id=1)
    assert d["new_users_30d"] == 1
    # last_message recent: chat_id 1 va 2
    assert d["active_users_30d"] == 2
    assert {"name": "users", "count": 120} in d["collections"]
    assert {"name": "orders", "count": 7} in d["collections"]
    assert {"name": "empty", "count": 0} in d["collections"]
    assert d["total_documents"] == 127
    # Markdown jadval bor
    assert "| Collection | Documents |" in res.content
    assert "users" in res.content


@pytest.mark.asyncio
async def test_user_stats_no_date_fields_returns_null_windows():
    backend = FakeBackend(
        users=[{"chat_id": 1, "username": "a"}, {"chat_id": 2, "username": "b"}],
        collections=[{"name": "leads"}],
        counts={"leads": 5},
    )
    res = await _tools()["chat.user_stats"].run({}, _ctx(backend))

    assert not res.is_error
    d = res.data
    assert d["total_chat_users"] == 2
    assert d["new_users_30d"] is None
    assert d["active_users_30d"] is None
    assert d["total_documents"] == 5


@pytest.mark.asyncio
async def test_user_stats_count_via_find_fallback():
    """aggregate $count ishlamasa, find total ga tushadi — crash yo'q."""
    backend = FakeBackend(
        users=[],
        collections=[{"name": "users"}],
        counts={"users": 9},
        aggregate_unsupported=True,
    )
    res = await _tools()["chat.user_stats"].run({}, _ctx(backend))

    assert not res.is_error
    assert {"name": "users", "count": 9} in res.data["collections"]
    assert res.data["total_documents"] == 9


@pytest.mark.asyncio
async def test_user_stats_count_unavailable_is_null_not_crash():
    """Ham aggregate ham find xato bersa — count None, crash yo'q."""
    backend = FakeBackend(
        users=[],
        collections=[{"name": "broken"}],
        count_raise_for={"broken"},
        aggregate_unsupported=True,
    )
    res = await _tools()["chat.user_stats"].run({}, _ctx(backend))

    assert not res.is_error
    assert {"name": "broken", "count": None} in res.data["collections"]
    assert res.data["total_documents"] == 0
    assert "n/a" in res.content


@pytest.mark.asyncio
async def test_user_stats_chat_users_error_does_not_crash():
    backend = FakeBackend(
        users_raise=True,
        collections=[{"name": "orders"}],
        counts={"orders": 3},
    )
    res = await _tools()["chat.user_stats"].run({}, _ctx(backend))

    assert not res.is_error
    assert res.data["total_chat_users"] == 0
    assert "chat_users_error" in res.data
    # Collection statistikasi baribir keladi
    assert {"name": "orders", "count": 3} in res.data["collections"]


@pytest.mark.asyncio
async def test_user_stats_collections_error_does_not_crash():
    backend = FakeBackend(
        users=[{"chat_id": 1}],
        list_collections_raise=True,
    )
    res = await _tools()["chat.user_stats"].run({}, _ctx(backend))

    assert not res.is_error
    assert res.data["total_chat_users"] == 1
    assert res.data["collections"] == []
    assert "collections_error" in res.data


@pytest.mark.asyncio
async def test_user_stats_no_project_id():
    backend = FakeBackend()
    ctx = ToolContext(run_id="r", project_id=None, user_id="u", state={}, services={"backend": backend})
    res = await _tools()["chat.user_stats"].run({}, ctx)
    assert res.is_error
    assert "project_id" in res.content


@pytest.mark.asyncio
async def test_user_stats_unix_timestamp_dates():
    """Unix epoch (sec va ms) sanalarni ham qabul qiladi."""
    now = datetime.now(tz=UTC)
    recent_sec = (now - timedelta(days=1)).timestamp()
    recent_ms = recent_sec * 1000.0
    backend = FakeBackend(
        users=[
            {"chat_id": 1, "last_seen": recent_sec},
            {"chat_id": 2, "last_seen": recent_ms},
        ],
        collections=[],
    )
    res = await _tools()["chat.user_stats"].run({"days": 7}, _ctx(backend))
    assert not res.is_error
    assert res.data["active_users_7d"] == 2

"""cms.update_many / cms.delete_many — filter bo'yicha bulk (client-side loop).

Backend'da atomik bulk endpoint yo'q: tool'lar find(filter) → per-doc
update/delete qiladi. Bu yerda: bo'sh-filter guard, N mos hujjat → N chaqiruv,
va #5 data_in_content YAML-skip tekshiriladi.
"""

import asyncio

from agent.integrations.backend import BackendClient
from agent.mcp.server import _result_to_mcp
from agent.tools.base import ToolContext
from agent.tools.cms import delete_many, update_many


def _run(coro):
    return asyncio.run(coro)


class _FakeBackend(BackendClient):
    """find() ga sobit rows qaytaradi, update/delete chaqiruvlarini yozib boradi."""

    def __init__(self, rows):
        self._rows = rows
        self.updated: list[tuple[str, dict]] = []
        self.deleted: list[str] = []

    async def cms_list_documents(self, pid, name, *, filter=None, sort=None,
                                 limit=50, skip=0, auth_claims=None):
        # limit'ni hurmat qil (guard/cap testlari uchun).
        return {"results": self._rows[:limit], "total": len(self._rows)}

    async def cms_update(self, pid, name, doc_id, patch, *, auth_claims=None):
        self.updated.append((doc_id, patch))
        return {"document": {"_id": doc_id, **patch}}

    async def cms_delete(self, pid, name, doc_id, *, auth_claims=None):
        self.deleted.append(doc_id)


def _ctx(backend):
    return ToolContext(
        run_id="r1", project_id="123", user_id="u1",
        state={"project_id": "123"}, services={"backend": backend},
    )


def test_update_many_patches_every_match():
    be = _FakeBackend([{"_id": "a"}, {"_id": "b"}, {"_id": "c"}])
    res = _run(update_many.run(
        {"collection": "parts", "filter": {"code": "123"}, "patch": {"file_id": "X"}}, _ctx(be)
    ))
    assert not res.is_error
    assert [d for d, _ in be.updated] == ["a", "b", "c"]
    assert all(p == {"file_id": "X"} for _, p in be.updated)
    assert res.data["updated"] == 3


def test_update_many_rejects_empty_filter():
    be = _FakeBackend([{"_id": "a"}])
    res = _run(update_many.run({"collection": "parts", "filter": {}, "patch": {"x": 1}}, _ctx(be)))
    assert res.is_error
    assert be.updated == []  # hech narsa yangilanmadi


def test_update_many_rejects_empty_patch():
    be = _FakeBackend([{"_id": "a"}])
    res = _run(update_many.run({"collection": "parts", "filter": {"a": 1}, "patch": {}}, _ctx(be)))
    assert res.is_error
    assert be.updated == []


def test_delete_many_soft_deletes_every_match():
    be = _FakeBackend([{"_id": "a"}, {"_id": "b"}])
    res = _run(delete_many.run({"collection": "parts", "filter": {"old": True}}, _ctx(be)))
    assert not res.is_error
    assert be.deleted == ["a", "b"]
    assert res.data["deleted"] == 2


def test_delete_many_rejects_empty_filter():
    be = _FakeBackend([{"_id": "a"}])
    res = _run(delete_many.run({"collection": "parts", "filter": {}}, _ctx(be)))
    assert res.is_error
    assert be.deleted == []


def test_data_in_content_skips_yaml_append():
    """#5 — content allaqachon qatorlarni saqlasa, MCP qatlami YAML qo'shmaydi."""
    res = _result_to_mcp("header only", is_error=False,
                         data={"results": [{"a": 1}]}, data_in_content=True)
    assert res.content[0].text == "header only"


def test_default_still_appends_yaml():
    res = _result_to_mcp("hdr", is_error=False, data={"x": 9})
    assert "x: 9" in res.content[0].text

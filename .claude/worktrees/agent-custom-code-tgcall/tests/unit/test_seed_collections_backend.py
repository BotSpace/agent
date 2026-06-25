"""_seed_collections backend'dan kolleksiya nomlarini oladi (stale state emas).

Bug: flow-builder sub-agent state'ida cms-manager keyinroq qo'shgan 'courses'
yo'q edi → simulyator bo'sh ko'rardi → agent 'kurslar mavjud emas' deb aylanib qolardi.
"""

import pytest

from agent.tools.flow.power import _seed_collections


class _Backend:
    def __init__(self):
        self.calls = []

    async def cms_list_collections(self, pid, *, auth_claims=None):
        # backend'da 'courses' bor — state'da yo'q edi
        return [{"name": "courses"}, {"name": "users"}]

    async def cms_list_documents(self, pid, name, *, limit=50, auth_claims=None):
        self.calls.append(name)
        if name == "courses":
            return {"results": [{"name": "Python", "price": 100}]}
        return {"results": []}


class _Ctx:
    def __init__(self, backend, collections=None):
        self.state = {"collections": collections or [], "auth_claims": {}}
        self.project_id = "1217"
        self.services = {"backend": backend}


@pytest.mark.asyncio
async def test_seed_uses_backend_when_state_empty():
    backend = _Backend()
    ctx = _Ctx(backend, collections=[])  # state BO'SH (sub-agent holati)
    out = await _seed_collections(ctx, "1217")
    assert "courses" in out
    assert out["courses"] == [{"name": "Python", "price": 100}]
    assert "courses" in backend.calls

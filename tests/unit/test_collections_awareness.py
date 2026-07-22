"""Tests for CMS collection awareness (PART 3).

- load_project_collections never raises (returns [] on failure).
- build_dynamic_block lists "Available collections: <name>{fields}".
"""
import asyncio

from agent.integrations.backend import load_project_collections
from agent.prompts.system import build_dynamic_block


def _run(coro):
    return asyncio.run(coro)


class _FakeBackend:
    def __init__(self, cols, detail):
        self._cols = cols
        self._detail = detail

    async def cms_list_collections(self, pid, *, auth_claims=None):
        return self._cols

    async def cms_get_collection(self, pid, name, *, auth_claims=None):
        return self._detail.get(name, {})


def test_load_collections_compact_shape():
    backend = _FakeBackend(
        cols=[{"name": "orders"}, {"name": "leads"}],
        detail={
            "orders": {"fields": [{"name": "total"}, {"name": "status"}]},
            "leads": {"fields": [{"name": "phone"}]},
        },
    )
    out = _run(load_project_collections("123", client=backend))
    assert out == [
        {
            "name": "orders",
            "fields": ["total", "status"],
            "field_kinds": {},
            "field_contracts": [
                {"name": "total", "kind": "text", "required": False},
                {"name": "status", "kind": "text", "required": False},
            ],
        },
        {
            "name": "leads",
            "fields": ["phone"],
            "field_kinds": {},
            "field_contracts": [{"name": "phone", "kind": "text", "required": False}],
        },
    ]


def test_load_collections_no_project_returns_empty():
    assert _run(load_project_collections(None)) == []
    assert _run(load_project_collections("global")) == []
    assert _run(load_project_collections("")) == []


def test_load_collections_list_failure_returns_empty():
    class Boom:
        async def cms_list_collections(self, pid, *, auth_claims=None):
            raise RuntimeError("backend down")

    out = _run(load_project_collections("123", client=Boom()))
    assert out == []


def test_load_collections_isolated_field_failure():
    class PartlyBroken:
        async def cms_list_collections(self, pid, *, auth_claims=None):
            return [{"name": "ok_col"}, {"name": "bad_col"}]

        async def cms_get_collection(self, pid, name, *, auth_claims=None):
            if name == "bad_col":
                raise RuntimeError("field fetch failed")
            return {"fields": [{"name": "a"}]}

    out = _run(load_project_collections("123", client=PartlyBroken()))
    # bad_col survives with empty fields; ok_col keeps its fields.
    assert out == [
        {
            "name": "ok_col",
            "fields": ["a"],
            "field_kinds": {},
            "field_contracts": [{"name": "a", "kind": "text", "required": False}],
        },
        {"name": "bad_col", "fields": [], "field_kinds": {}, "field_contracts": []},
    ]


def test_dynamic_block_lists_collections():
    block = build_dynamic_block(
        {
            "project_id": "p1",
            "collections": [
                {"name": "orders", "fields": ["total", "status"]},
                {"name": "leads", "fields": []},
            ],
        }
    )
    assert "Available collections:" in block
    assert "orders(total, status)" in block
    assert "leads" in block


def test_dynamic_block_exposes_field_types_and_relation_targets():
    block = build_dynamic_block(
        {
            "collections": [
                {
                    "name": "registrations",
                    "fields": ["phone", "course_id"],
                    "field_kinds": {"phone": "phone", "course_id": "relation"},
                    "field_contracts": [
                        {
                            "name": "course_id",
                            "kind": "relation",
                            "relation_collection": "courses",
                        }
                    ],
                }
            ]
        }
    )
    assert "registrations(phone:phone, course_id:relation→courses)" in block


def test_dynamic_block_no_collections_no_line():
    block = build_dynamic_block({"project_id": "p1"})
    assert "Available collections" not in block

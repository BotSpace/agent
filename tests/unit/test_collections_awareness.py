"""Tests for CMS collection awareness (PART 3).

- load_project_collections never raises (returns [] on failure).
- build_dynamic_block lists "Available collections: <name>{fields}".
"""

import asyncio

import pytest

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


def test_load_collections_uses_slug_as_runtime_identifier():
    backend = _FakeBackend(
        cols=[{"name": "Chat tarixi", "slug": "chat_history"}],
        detail={"chat_history": {"fields": [{"name": "message", "kind": "text"}]}},
    )
    out = _run(load_project_collections("123", client=backend))
    assert out[0]["name"] == "chat_history"
    assert out[0]["field_kinds"] == {"message": "text"}


def test_load_collections_resolves_go_relation_id_back_to_target_slug():
    backend = _FakeBackend(
        cols=[
            {"id": "courses-id", "name": "Kurslar", "slug": "courses"},
            {"id": "registrations-id", "name": "Yozilishlar", "slug": "registrations"},
        ],
        detail={
            "courses": {"fields": [{"name": "name", "type": "text"}]},
            "registrations": {
                "fields": [
                    {
                        "name": "course_id",
                        "type": "relation",
                        "relation_collection_id": "courses-id",
                    }
                ]
            },
        },
    )

    out = _run(load_project_collections("123", client=backend))
    registration = next(item for item in out if item["name"] == "registrations")
    assert registration["field_contracts"] == [
        {
            "name": "course_id",
            "kind": "relation",
            "required": False,
            "relation_collection": "courses",
        }
    ]


def test_load_collections_list_failure_is_reported():
    class Boom:
        async def cms_list_collections(self, pid, *, auth_claims=None):
            raise RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        _run(load_project_collections("123", client=Boom()))


def test_load_collections_field_failure_is_reported():
    class PartlyBroken:
        async def cms_list_collections(self, pid, *, auth_claims=None):
            return [{"name": "ok_col"}, {"name": "bad_col"}]

        async def cms_get_collection(self, pid, name, *, auth_claims=None):
            if name == "bad_col":
                raise RuntimeError("field fetch failed")
            return {"fields": [{"name": "a"}]}

    with pytest.raises(RuntimeError, match="field fetch failed"):
        _run(load_project_collections("123", client=PartlyBroken()))


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

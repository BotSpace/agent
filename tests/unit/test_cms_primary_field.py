"""Tests for cms.create_collection primary_field passthrough.

The agent CreateCollectionInput must accept `primary_field` and forward it to
the backend create body (backend serializer already accepts it). When it is not
set, the key must be omitted from the body so the backend default is untouched.
"""

import asyncio

from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.cms import CreateCollectionInput, FieldSpec, create_collection


def _run(coro):
    return asyncio.run(coro)


class _CaptureBackend(BackendClient):
    """Records the body passed to cms_create_collection.

    Subclasses BackendClient (without calling its __init__) so the cms tool's
    `_backend(ctx)` isinstance check accepts it instead of building a real
    HTTP client.
    """

    def __init__(self):
        self.bodies: list[dict] = []

    async def cms_create_collection(self, pid, body, *, auth_claims=None):
        self.bodies.append(body)
        return {"id": 7, "name": body.get("name")}

    async def cms_add_field(self, pid, collection, body, *, auth_claims=None):
        return body


def _ctx(backend):
    return ToolContext(
        run_id="r1",
        project_id="123",
        user_id="u1",
        state={"project_id": "123"},
        services={"backend": backend},
    )


def test_primary_field_in_input_model():
    inp = CreateCollectionInput(name="products", primary_field="name")
    assert inp.primary_field == "name"


def test_primary_field_forwarded_to_backend_body():
    backend = _CaptureBackend()
    res = _run(create_collection.run({"name": "products", "primary_field": "name"}, _ctx(backend)))

    assert not res.is_error
    assert len(backend.bodies) == 1
    body = backend.bodies[0]
    assert body["primary_field"] == "name"
    assert body["name"] == "products"
    # project_id must NOT leak into the create body
    assert "project_id" not in body


def test_primary_field_omitted_when_none():
    backend = _CaptureBackend()
    res = _run(create_collection.run({"name": "leads"}, _ctx(backend)))

    assert not res.is_error
    body = backend.bodies[0]
    # absent -> key removed so backend default stays intact
    assert "primary_field" not in body


def test_relation_field_uses_collection_slug():
    field = FieldSpec.model_validate(
        {
            "name": "course",
            "kind": "relation",
            "relation_collection": "courses",
        }
    )
    assert field.relation_collection == "courses"


def test_create_collection_updates_run_local_contract():
    backend = _CaptureBackend()
    ctx = _ctx(backend)
    ctx.state["collections"] = [
        {"name": "existing", "fields": ["name"], "field_kinds": {"name": "text"}}
    ]

    res = _run(
        create_collection.run(
            {
                "name": "registrations",
                "fields": [
                    {"name": "phone", "kind": "phone", "required": True},
                    {
                        "name": "course",
                        "kind": "relation",
                        "relation_collection": "courses",
                    },
                ],
            },
            ctx,
        )
    )

    contracts = res.state_updates["collections"]
    assert [item["name"] for item in contracts] == ["existing", "registrations"]
    registrations = contracts[1]
    assert registrations["field_kinds"] == {"phone": "phone", "course": "relation"}
    assert registrations["field_contracts"][1]["relation_collection"] == "courses"

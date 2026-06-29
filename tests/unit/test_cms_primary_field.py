"""Tests for cms.create_collection primary_field passthrough.

The agent CreateCollectionInput must accept `primary_field` and forward it to
the backend create body (backend serializer already accepts it). When it is not
set, the key must be omitted from the body so the backend default is untouched.
"""

import asyncio

from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.cms import CreateCollectionInput, create_collection


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

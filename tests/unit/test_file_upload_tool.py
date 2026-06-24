"""Unit tests for file.upload tool."""

from __future__ import annotations

import pytest

from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.file import build_file_tools


class _FakeBackend(BackendClient):
    """BackendClient subclass that overrides network methods with in-memory stubs."""

    def __init__(self):
        # Skip real __init__ — no httpx client needed.
        self._base_url = "http://fake"
        self._internal_token = None
        self.upload_calls: list[dict] = []
        self.list_calls: list = []
        self.delete_calls: list = []

    async def upload_file(
        self,
        filename,
        content_bytes,
        content_type,
        *,
        project,
        ttl=None,
        auth_claims=None,
    ):
        self.upload_calls.append(
            {
                "filename": filename,
                "content_bytes": content_bytes,
                "content_type": content_type,
                "project": project,
                "ttl": ttl,
            }
        )
        return {
            "id": 42,
            "name": filename,
            "url": f"https://cdn.example.com/{filename}",
            "size": len(content_bytes),
            "mime": content_type,
            "expires_at": None,
            "uuid": "abc-123",
        }

    async def list_files(self, project_id=None, *, page=None, page_size=None, auth_claims=None):
        self.list_calls.append(project_id)
        return []

    async def delete_file(self, file_id, *, auth_claims=None):
        self.delete_calls.append(file_id)


def _ctx(backend=None, project_id="7"):
    return ToolContext(
        run_id="r",
        project_id=project_id,
        user_id="u",
        services={"backend": backend or _FakeBackend()},
        state={"project_id": project_id},
    )


def _tools():
    return {t.name: t for t in build_file_tools()}


@pytest.mark.asyncio
async def test_file_upload_content():
    backend = _FakeBackend()
    ctx = _ctx(backend)
    tools = _tools()

    res = await tools["file.upload"].run(
        {"filename": "hello.txt", "content": "Hello world"},
        ctx,
    )
    assert not res.is_error, res.content
    assert res.data["id"] == 42
    assert res.data["name"] == "hello.txt"
    assert "uploaded" in res.content.lower() or "id=42" in res.content

    call = backend.upload_calls[0]
    assert call["filename"] == "hello.txt"
    assert call["content_bytes"] == b"Hello world"
    assert call["content_type"] == "text/plain"
    assert call["project"] == "7"
    assert call["ttl"] is None


@pytest.mark.asyncio
async def test_file_upload_with_ttl():
    backend = _FakeBackend()
    ctx = _ctx(backend)
    tools = _tools()

    res = await tools["file.upload"].run(
        {"filename": "data.json", "content": "raw json data here", "ttl": 3600},
        ctx,
    )
    assert not res.is_error
    call = backend.upload_calls[0]
    assert call["ttl"] == 3600
    assert call["content_type"] == "application/json"


@pytest.mark.asyncio
async def test_file_upload_missing_source():
    tools = _tools()
    ctx = _ctx()

    res = await tools["file.upload"].run(
        {"filename": "empty.txt"},
        ctx,
    )
    assert res.is_error


@pytest.mark.asyncio
async def test_file_upload_both_sources_rejected():
    tools = _tools()
    ctx = _ctx()

    res = await tools["file.upload"].run(
        {"filename": "x.txt", "content": "hi", "url": "https://example.com/x.txt"},
        ctx,
    )
    assert res.is_error


@pytest.mark.asyncio
async def test_file_upload_no_project_id():
    tools = _tools()
    ctx = ToolContext(
        run_id="r",
        project_id=None,
        user_id="u",
        services={"backend": _FakeBackend()},
        state={},
    )
    res = await tools["file.upload"].run(
        {"filename": "a.txt", "content": "hi"},
        ctx,
    )
    assert res.is_error


@pytest.mark.asyncio
async def test_build_file_tools_includes_upload():
    names = {t.name for t in build_file_tools()}
    assert "file.upload" in names
    assert "file.list" in names
    assert "file.delete" in names

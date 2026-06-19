import pytest

from agent.integrations.backend import BackendClient
from agent.tools.base import ToolContext
from agent.tools.credentials import build_credentials_tools


class _FakeBackend(BackendClient):
    """Stub recording oauth_start calls and returning a canned URL.

    Subclasses BackendClient so the credentials tool's `isinstance` guard
    accepts it; __init__ is overridden to avoid opening a real httpx client.
    """

    def __init__(self, *, authorize_url: str | None = "https://accounts.google.com/o/oauth2/v2/auth?state=abc"):
        self.oauth_start_calls: list[tuple[str, str]] = []
        self._authorize_url = authorize_url

    async def oauth_start(self, provider, type_key, *, auth_claims=None):
        self.oauth_start_calls.append((provider, type_key))
        out: dict = {}
        if self._authorize_url is not None:
            out["authorize_url"] = self._authorize_url
        return out


def _tools():
    return {t.name: t for t in build_credentials_tools()}


def _ctx(backend):
    return ToolContext(
        run_id="r",
        project_id="42",
        user_id="u",
        services={"backend": backend},
    )


def test_connect_oauth_is_registered():
    assert "credentials.connect_oauth" in _tools()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "type_key",
    ["google_email", "google_sheets", "google_drive"],
)
async def test_connect_oauth_accepts_google_types(type_key):
    backend = _FakeBackend()
    res = await _tools()["credentials.connect_oauth"].run({"type_key": type_key}, _ctx(backend))

    assert not res.is_error
    # all google_* map to the "google" provider
    assert backend.oauth_start_calls == [("google", type_key)]
    # the link is surfaced both in the human text and structured data
    assert "https://accounts.google.com/o/oauth2/v2/auth?state=abc" in res.content
    assert res.data["authorize_url"] == "https://accounts.google.com/o/oauth2/v2/auth?state=abc"
    assert res.data["type_key"] == type_key


@pytest.mark.asyncio
@pytest.mark.parametrize("type_key", ["telegram", "openai", "ollama", "ai_studio"])
async def test_connect_oauth_rejects_non_oauth_types(type_key):
    backend = _FakeBackend()
    res = await _tools()["credentials.connect_oauth"].run({"type_key": type_key}, _ctx(backend))

    assert res.is_error
    # backend must NOT be hit for a non-oauth type
    assert backend.oauth_start_calls == []
    assert "credentials.create" in res.content


@pytest.mark.asyncio
async def test_connect_oauth_errors_when_no_authorize_url():
    backend = _FakeBackend(authorize_url=None)
    res = await _tools()["credentials.connect_oauth"].run(
        {"type_key": "google_sheets"}, _ctx(backend)
    )

    assert res.is_error
    assert "authorize_url" in res.content

"""MCP auth — bsp_ exchange (respx mock), JWT verify, cache/expiry."""

from __future__ import annotations

import time

import httpx
import jwt
import pytest
import respx

from agent.config import Settings
from agent.mcp.auth import (
    MCPAuthError,
    _invalidate_exchange_cache,
    authenticate,
)


def _settings() -> Settings:
    return Settings(
        JWT_ALGORITHM="HS256",
        SESSION_SECRET="s" * 40,
        BACKEND_BASE_URL="http://backend.test",
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    _invalidate_exchange_cache()
    yield
    _invalidate_exchange_cache()


# ── JWT verify path ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jwt_verify_ok():
    s = _settings()
    token = jwt.encode(
        {"user": {"id": 99}, "access": "acc-jwt", "exp": int(time.time()) + 3600},
        s.session_secret,
        algorithm="HS256",
    )
    user = await authenticate(f"Bearer {token}", settings=s)
    assert user.user_id == "99"
    assert user.claims["access_token"] == "acc-jwt"
    assert user.claims["user_id"] == "99"


@pytest.mark.asyncio
async def test_jwt_expired_rejected():
    s = _settings()
    token = jwt.encode(
        {"sub": "1", "exp": int(time.time()) - 10},
        s.session_secret,
        algorithm="HS256",
    )
    with pytest.raises(MCPAuthError):
        await authenticate(f"Bearer {token}", settings=s)


@pytest.mark.asyncio
async def test_missing_or_malformed_header():
    s = _settings()
    with pytest.raises(MCPAuthError):
        await authenticate(None, settings=s)
    with pytest.raises(MCPAuthError):
        await authenticate("Token abc", settings=s)
    with pytest.raises(MCPAuthError):
        await authenticate("Bearer ", settings=s)


# ── bsp_ exchange path ───────────────────────────────────────────────────────


def _make_access(user_id: int, exp_in: int = 3600) -> str:
    return jwt.encode(
        {"user": {"id": user_id}, "exp": int(time.time()) + exp_in},
        "irrelevant-not-verified-but-long-enough-for-hs256",
        algorithm="HS256",
    )


@pytest.mark.asyncio
@respx.mock
async def test_bsp_exchange_unwraps_envelope():
    s = _settings()
    access = _make_access(123)
    route = respx.post("http://backend.test/auth/token/exchange").mock(
        return_value=httpx.Response(
            200, json={"status": True, "data": {"access": access, "refresh": "ref-1"}}
        )
    )
    user = await authenticate("Bearer bsp_abc123", settings=s)
    assert route.called
    assert user.user_id == "123"
    assert user.claims["access_token"] == access
    assert user.claims["refresh_token"] == "ref-1"


@pytest.mark.asyncio
@respx.mock
async def test_bsp_exchange_cached_second_call():
    s = _settings()
    access = _make_access(55)
    route = respx.post("http://backend.test/auth/token/exchange").mock(
        return_value=httpx.Response(200, json={"data": {"access": access, "refresh": "r"}})
    )
    u1 = await authenticate("Bearer bsp_same", settings=s)
    u2 = await authenticate("Bearer bsp_same", settings=s)
    assert u1.user_id == u2.user_id == "55"
    # Ikkinchi call cache'dan keldi — backend faqat bir marta chaqirildi.
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_bsp_exchange_re_exchanges_near_expiry():
    s = _settings()
    # Birinchi access deyarli muddati o'tgan (leeway 60s ichida) → qayta exchange.
    near = _make_access(7, exp_in=10)
    fresh = _make_access(7, exp_in=3600)
    route = respx.post("http://backend.test/auth/token/exchange").mock(
        side_effect=[
            httpx.Response(200, json={"data": {"access": near, "refresh": "r1"}}),
            httpx.Response(200, json={"data": {"access": fresh, "refresh": "r2"}}),
        ]
    )
    u1 = await authenticate("Bearer bsp_exp", settings=s)
    u2 = await authenticate("Bearer bsp_exp", settings=s)
    assert u1.claims["access_token"] == near
    assert u2.claims["access_token"] == fresh
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_bsp_exchange_rejected():
    s = _settings()
    respx.post("http://backend.test/auth/token/exchange").mock(
        return_value=httpx.Response(401, json={"detail": "invalid"})
    )
    with pytest.raises(MCPAuthError):
        await authenticate("Bearer bsp_bad", settings=s)

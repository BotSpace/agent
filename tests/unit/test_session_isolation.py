"""Stateful MCP: har ulanish (session) alohida client_key oladi.

Bug tarixi: header-asoslangan yondashuv ishlamasdi — stateful rejimda tool
handler alohida sessiya task'ida ishlaydi va request-kontekst contextvar'i unga
yetib bormaydi. Yechim: SDK `request_ctx.get().session` obyektidan kalit olish
(_session_client_key) — bir token bilan ikki terminal ham izolyatsiyalanadi.
"""

import json
import re

import httpx
import pytest


async def _open_session(c):
    H = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    r = await c.post("/", headers=H, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "1"}}})
    sid = r.headers.get("mcp-session-id")
    assert sid, "stateful rejimda initialize session id qaytarishi shart"
    H2 = {**H, "Mcp-Session-Id": sid}
    await c.post("/", headers=H2, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return H2


async def _client_key(c, headers):
    r = await c.post("/", headers=headers, json={
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "session_status", "arguments": {}}})
    # Javob endi YAML (structuredContent JSON emas): `client_key: sid-...`.
    m = re.search(r'client_key["\s:]+\'?"?([^\s"\',}]+)', json.dumps(r.json()))
    assert m, f"client_key topilmadi: {r.text[:300]}"
    return m.group(1)


@pytest.mark.asyncio
async def test_two_sessions_same_token_are_isolated(monkeypatch):
    import agent.mcp.server as srv
    from agent.mcp.auth import AuthedUser

    async def fake_auth(authorization):
        # IKKALA "terminal" ham AYNI token (bir xil fallback hash)
        return AuthedUser(user_id="42", claims={}, client_key="sametokenhash000")

    monkeypatch.setattr(srv, "authenticate", fake_auth)
    app, sm = srv.build_mcp_asgi_app()
    async with srv.mcp_lifespan(sm):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            ha = await _open_session(c)
            hb = await _open_session(c)
            ka1 = await _client_key(c, ha)
            kb = await _client_key(c, hb)
            ka2 = await _client_key(c, ha)

    assert ka1.startswith("sid-"), f"session kaliti emas: {ka1}"
    assert kb.startswith("sid-")
    assert ka1 != kb, "ikki session BIR XIL kalit oldi — izolyatsiya buzilgan"
    assert ka1 == ka2, "bir session ichida kalit barqaror emas"

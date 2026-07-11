"""Tests for the per-IP rate limiter + exemptions."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.middleware import RateLimitMiddleware


def _app(limit: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit=limit, window_sec=60.0)

    @app.get("/runs")
    async def runs():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/mcp/x")
    async def mcp():
        return {"ok": True}

    return app


def test_blocks_after_limit_per_ip():
    client = TestClient(_app(limit=3))
    for _ in range(3):
        assert client.get("/runs").status_code == 200
    r = client.get("/runs")
    assert r.status_code == 429
    assert r.json()["detail"]["code"] == "rate_limited"
    assert "Retry-After" in r.headers


def test_health_and_mcp_are_exempt():
    client = TestClient(_app(limit=1))
    # First /runs consumes the single slot.
    assert client.get("/runs").status_code == 200
    assert client.get("/runs").status_code == 429
    # /health and /mcp/* are never limited.
    for _ in range(5):
        assert client.get("/health").status_code == 200
        assert client.get("/mcp/x").status_code == 200


def test_separate_ips_have_separate_budgets():
    client = TestClient(_app(limit=1))
    assert client.get("/runs", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    # Different IP → its own budget, not blocked.
    assert client.get("/runs", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    # Same first IP again → blocked.
    assert client.get("/runs", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429

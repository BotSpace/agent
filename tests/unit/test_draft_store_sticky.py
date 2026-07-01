"""DraftStore sticky restore (#2) — load_latest oxirgi TANLANGAN loyihani qaytaradi.

Bu SQL-darajali test (active_at ORDER BY) — jonli Postgres talab qiladi. DB
bo'lmasa skip bo'ladi (CI'da AGENT_DATABASE_URL bor). Regressiya: boshqa loyihada
mutation `updated_at`ni ko'tarsa ham, restore explicit tanlangan loyihaga yopishadi.
"""

import asyncio

import psycopg
import pytest

from agent.config import get_settings
from agent.config import normalize_pg_url as _normalize
from agent.mcp.draft_store import DraftStore


def _run(coro):
    return asyncio.run(coro)


def _db_available(dsn: str) -> bool:
    try:
        with psycopg.connect(_normalize(dsn), connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture
def store():
    dsn = get_settings().database_url
    if not _db_available(dsn):
        pytest.skip("Postgres yo'q — sticky SQL testini o'tkazib yuboramiz.")
    st = DraftStore(dsn)
    uid = "sticky-test-user"
    # Toza start.
    _run(st.clear(user_id=uid, project_id="1383"))
    _run(st.clear(user_id=uid, project_id="1381"))
    yield st, uid
    _run(st.clear(user_id=uid, project_id="1383"))
    _run(st.clear(user_id=uid, project_id="1381"))


def _upsert(st, uid, pid):
    _run(st.upsert(
        user_id=uid, project_id=pid, project_name=f"P{pid}",
        working_flow={"nodes": []}, baseline_flow=None, flow_dirty=True,
        collections=[], snapshots=[],
    ))


def test_load_latest_prefers_last_selected(store):
    st, uid = store
    # 1383 tanlanadi (explicit), keyin 1381 boshqa mutation bilan yangilanadi.
    _upsert(st, uid, "1383")
    _run(st.touch_active(user_id=uid, project_id="1383"))
    _upsert(st, uid, "1381")  # updated_at endi 1381 uchun yangiroq
    # 1381 tanlanmagan (active_at NULL) → restore hali 1383 bo'lishi kerak.
    latest = _run(st.load_latest(user_id=uid))
    assert latest is not None
    assert latest["project_id"] == "1383"


def test_load_latest_falls_back_to_updated_when_none_selected(store):
    st, uid = store
    # Hech biri explicit tanlanmagan → oxirgi tahrirlangan (1381) qaytadi.
    _upsert(st, uid, "1383")
    _upsert(st, uid, "1381")
    latest = _run(st.load_latest(user_id=uid))
    assert latest is not None
    assert latest["project_id"] == "1381"

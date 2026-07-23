"""Faza 2.4 — cross-replica interrupt resolution."""

import asyncio

import pytest
from redis.exceptions import ConnectionError

from agent.runtime import worker
from agent.runtime.store import ResolutionStore, get_resolution_store


def test_resolution_store_is_configured_when_redis_dependency_is_installed():
    assert isinstance(get_resolution_store(), ResolutionStore)


def test_submit_reports_redis_failure_when_no_local_waiter(monkeypatch):
    class BrokenStore:
        def publish(self, request_id, payload):
            raise RuntimeError("redis down")

    monkeypatch.setattr("agent.runtime.store.get_resolution_store", lambda: BrokenStore())
    with pytest.raises(RuntimeError, match="redis down"):
        worker.submit_resolution("no-such-request", {"approved": True})


@pytest.mark.asyncio
async def test_wait_and_submit_roundtrip_local_fast_path(monkeypatch):
    class WaitingStore:
        async def wait(self, request_id):
            await asyncio.Future()

    monkeypatch.setattr("agent.runtime.store.get_resolution_store", lambda: WaitingStore())
    loop = asyncio.get_running_loop()
    request_id = "req-roundtrip-1"

    async def _resolve_soon():
        # Retry until the waiter has registered its future (mirrors the real
        # /approve POST racing the worker's _wait_resolution).
        for _ in range(100):
            if worker.submit_resolution(request_id, {"approved": True}):
                return
            await asyncio.sleep(0)
        raise AssertionError("waiter never registered")

    resolver = asyncio.create_task(_resolve_soon())
    answer = await worker._wait_resolution(request_id, loop)
    await resolver

    assert answer == {"approved": True}
    # Future map cleaned up on exit.
    assert request_id not in worker._RESOLUTIONS


def test_publish_reports_redis_failure():
    store = ResolutionStore("redis://127.0.0.1:1/0")
    with pytest.raises(ConnectionError):
        store.publish("req", {"approved": True})

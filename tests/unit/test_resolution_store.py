"""Faza 2.4 — cross-replica interrupt resolution.

Redis isn't installed in the unit-test env, so ``get_resolution_store()`` returns
None and the worker falls back to the in-process future. These tests pin that
fallback contract (the graph's resolve/resume API must keep working) and the
Redis-backed store's degrade-to-disabled behaviour.
"""

import asyncio

import pytest

from agent.runtime import worker
from agent.runtime.store import ResolutionStore, get_resolution_store


def test_resolution_store_absent_without_redis():
    # redis not installed → in-memory fallback (None store).
    assert get_resolution_store() is None


def test_submit_returns_false_when_no_local_waiter():
    # No local future + no Redis → nothing to resolve → 409 signalled to route.
    assert worker.submit_resolution("no-such-request", {"approved": True}) is False


@pytest.mark.asyncio
async def test_wait_and_submit_roundtrip_in_memory():
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


def test_publish_disables_on_redis_failure():
    # Unreachable/absent Redis → publish() degrades to disabled + returns False,
    # never raising into the request handler.
    store = ResolutionStore("redis://127.0.0.1:1/0")
    assert store.publish("req", {"approved": True}) is False
    assert store._disabled is True

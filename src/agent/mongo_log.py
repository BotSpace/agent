"""MongoDB LLM conversation logger — writes to the `llm` database.

Every agent run is one document in `llm.conversations`:
  {
    _id:            request_id (str),
    project_id, user_id, thread_id,
    prompt:         initial user message,
    messages:       [{role, content, model?, usage?, ts}],
    tool_calls:     [{id, name, arguments, result, is_error, ts}],
    final_response: str | None,
    status:         "running" | "completed" | "failed",
    error:          str | None,
    created_at, completed_at
  }

Used by: src/agent/runtime/store.py (call log_* after every PG write).
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)

_DB_NAME = "llm"
_COL_NAME = "conversations"

# Module-level lazy singleton
_client: Any = None
_col: Any = None


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


def _get_col() -> Any:
    """Return Motor collection, connecting lazily on first call."""
    global _client, _col
    if _col is not None:
        return _col
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        from .config import get_settings

        s = get_settings()
        _client = AsyncIOMotorClient(s.mongo_base_url, serverSelectionTimeoutMS=3000)
        _col = _client[_DB_NAME][_COL_NAME]
        logger.info("LLM Mongo logger connected to %s/%s", _DB_NAME, _COL_NAME)
    except Exception as exc:
        logger.warning("LLM Mongo logger: failed to connect — %s", exc)
        _col = None
    return _col


def _fire(coro: Any) -> None:
    """Schedule a coroutine as a fire-and-forget task (never blocks caller)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        pass


async def _safe(coro: Any) -> None:
    try:
        await coro
    except Exception as exc:
        logger.debug("LLM Mongo logger write error: %s", exc)


# ── public API ────────────────────────────────────────────────────────────────

def log_run_start(
    request_id: str,
    project_id: str | None,
    user_id: str | None,
    thread_id: str | None,
    prompt: str,
) -> None:
    col = _get_col()
    if col is None:
        return

    async def _write() -> None:
        await _safe(col.insert_one({
            "_id": request_id,
            "project_id": project_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "prompt": prompt,
            # Full conversation in order: user, assistant, tool_call, tool_result, ...
            "messages": [{"role": "user", "content": prompt, "ts": _now()}],
            "tool_calls": [],
            "final_response": None,
            "status": "running",
            "error": None,
            "created_at": _now(),
            "completed_at": None,
        }))

    _fire(_write())


def log_run_complete(request_id: str, final_response: str) -> None:
    col = _get_col()
    if col is None:
        return

    async def _write() -> None:
        await _safe(col.update_one(
            {"_id": request_id},
            {"$set": {"final_response": final_response, "status": "completed", "completed_at": _now()}},
        ))

    _fire(_write())


def log_run_fail(request_id: str, error: str) -> None:
    col = _get_col()
    if col is None:
        return

    async def _write() -> None:
        await _safe(col.update_one(
            {"_id": request_id},
            {"$set": {"status": "failed", "error": error, "completed_at": _now()}},
        ))

    _fire(_write())


def log_assistant_message(
    request_id: str,
    text: str,
    model: str | None = None,
    usage: dict[str, Any] | None = None,
) -> None:
    col = _get_col()
    if col is None:
        return

    entry: dict[str, Any] = {"role": "assistant", "content": text, "ts": _now()}
    if model:
        entry["model"] = model
    if usage:
        entry["usage"] = usage

    async def _write() -> None:
        await _safe(col.update_one(
            {"_id": request_id},
            {"$push": {"messages": entry}},  # into chronological messages list
        ))

    _fire(_write())


def log_tool_call(
    request_id: str,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> None:
    col = _get_col()
    if col is None:
        return

    ts = _now()
    tool_entry: dict[str, Any] = {
        "id": call_id,
        "name": name,
        "arguments": arguments,
        "result": None,
        "is_error": None,
        "ts": ts,
    }
    # Also appears in chronological messages for training data
    msg_entry: dict[str, Any] = {
        "role": "tool_call",
        "id": call_id,
        "name": name,
        "arguments": arguments,
        "ts": ts,
    }

    async def _write() -> None:
        await _safe(col.update_one(
            {"_id": request_id},
            {"$push": {"tool_calls": tool_entry, "messages": msg_entry}},
        ))

    _fire(_write())


def log_tool_result(
    request_id: str,
    call_id: str,
    name: str,
    content: str,
    is_error: bool,
) -> None:
    col = _get_col()
    if col is None:
        return

    ts = _now()
    msg_entry: dict[str, Any] = {
        "role": "tool_result",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
        "is_error": is_error,
        "ts": ts,
    }

    async def _write() -> None:
        await _safe(col.update_one(
            {"_id": request_id, "tool_calls.id": call_id},
            {
                "$set": {
                    "tool_calls.$.result": content,
                    "tool_calls.$.is_error": is_error,
                    "tool_calls.$.result_ts": ts,
                },
                "$push": {"messages": msg_entry},
            },
        ))

    _fire(_write())

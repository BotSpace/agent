"""MCP A-band trace logger — prompt + tool-call training data → MongoDB.

Stores one document per MCP tool call (and one synthetic `__prompt__` record per
build_bot prompt pull) into the SAME Mongo used by the LLM conversation logger
(`mongo_log.py`), but in a SEPARATE `mcp` collection of the `llm` database.

Document shape (A-band / AI training data):
    {
        trace_id,         # unique id for this trace record
        user_id,          # authenticated user
        client_key,       # per-client hash prefix (NOT the raw token)
        project_id,       # active project (or explicit override) when known
        prompt,           # ONLY set on the __prompt__ record; None otherwise
        tool_name,        # MCP tool name, or "__prompt__" for the prompt record
        arguments,        # tool args with token/auth fields REDACTED
        result_summary,   # short text summary of the tool result
        is_error,
        duration_ms,
        created_at,
    }

Design — mirrors mongo_log.py exactly:
  - Lazy Motor singleton against settings.mongo_base_url.
  - FIRE-AND-FORGET: writes never block or fail the MCP tool call.
  - GRACEFUL DEGRADE: if Mongo is down/unreachable, tracing is silently a no-op
    and the MCP keeps working — only the trace is lost.

Kept deliberately separate from mongo_log.py (LLM conversations) so the two
write paths can never interfere; only the connection settings are shared.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_DB_NAME = "llm"
_COL_NAME = "mcp"

# Sentinel tool name for the prompt-only record (the only record carrying `prompt`).
PROMPT_TOOL = "__prompt__"

# Argument keys whose values are redacted before persisting (case-insensitive
# substring match). Covers tokens, secrets, auth blocks, passwords.
_REDACT_KEYS = (
    "token",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth_claims",
    "access",
    "refresh",
    "api_key",
    "apikey",
    "credential",
)

_REDACTED = "***redacted***"
_MAX_SUMMARY = 800

# Module-level lazy singleton (independent of mongo_log's client).
_client: Any = None
_col: Any = None

# Strong references to in-flight fire-and-forget tasks so they aren't garbage
# collected before completing; discarded on done.
_pending: set[Any] = set()


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


def _get_col() -> Any:
    """Return the Motor `llm.mcp` collection, connecting lazily on first call."""
    global _client, _col
    if _col is not None:
        return _col
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        from ..config import get_settings

        s = get_settings()
        _client = AsyncIOMotorClient(s.mongo_base_url, serverSelectionTimeoutMS=3000)
        _col = _client[_DB_NAME][_COL_NAME]
        logger.info("MCP trace logger connected to %s/%s", _DB_NAME, _COL_NAME)
    except Exception as exc:
        logger.warning("MCP trace logger: failed to connect — %s", exc)
        _col = None
    return _col


def _fire(coro: Any) -> None:
    """Schedule a coroutine as a fire-and-forget task (never blocks caller)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (e.g. called from sync context) — drop the trace.
        return
    task = loop.create_task(coro)
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def _safe(coro: Any) -> None:
    try:
        await coro
    except Exception as exc:
        logger.debug("MCP trace write error: %s", exc)


def _redact(value: Any) -> Any:
    """Deep-copy `value`, masking token/auth-like fields. Never raises."""
    try:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                key = str(k).lower()
                if any(marker in key for marker in _REDACT_KEYS):
                    out[k] = _REDACTED
                else:
                    out[k] = _redact(v)
            return out
        if isinstance(value, (list, tuple)):
            return [_redact(v) for v in value]
        return value
    except Exception:
        return _REDACTED


def _summarize(content: str | None) -> str:
    if not content:
        return ""
    text = str(content)
    if len(text) > _MAX_SUMMARY:
        return text[:_MAX_SUMMARY] + "…"
    return text


# ── public API ────────────────────────────────────────────────────────────────


def log_tool_call(
    *,
    user_id: str | None,
    client_key: str | None,
    project_id: str | None,
    tool_name: str,
    arguments: dict[str, Any] | None,
    result_summary: str | None,
    is_error: bool,
    duration_ms: int,
) -> None:
    """Persist one MCP tool-call trace (fire-and-forget, graceful degrade)."""
    col = _get_col()
    if col is None:
        return

    doc = {
        "trace_id": uuid.uuid4().hex,
        "user_id": user_id,
        "client_key": client_key,
        "project_id": project_id,
        "prompt": None,
        "tool_name": tool_name,
        "arguments": _redact(arguments or {}),
        "result_summary": _summarize(result_summary),
        "is_error": bool(is_error),
        "duration_ms": int(duration_ms),
        "created_at": _now(),
    }
    _fire(_safe(col.insert_one(doc)))


def log_prompt(
    *,
    user_id: str | None,
    client_key: str | None,
    project_id: str | None,
    prompt: str,
) -> None:
    """Persist the user prompt as a separate `__prompt__` trace record."""
    col = _get_col()
    if col is None:
        return

    doc = {
        "trace_id": uuid.uuid4().hex,
        "user_id": user_id,
        "client_key": client_key,
        "project_id": project_id,
        "prompt": prompt,
        "tool_name": PROMPT_TOOL,
        "arguments": {},
        "result_summary": "",
        "is_error": False,
        "duration_ms": 0,
        "created_at": _now(),
    }
    _fire(_safe(col.insert_one(doc)))


def _reset() -> None:
    """Test hook — drop the cached client so settings can be re-read."""
    global _client, _col
    _client = None
    _col = None


__all__ = ["PROMPT_TOOL", "log_prompt", "log_tool_call"]

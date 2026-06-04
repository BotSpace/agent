"""Engine-side flow simulator client (#1).

The simulator is a separate engine service that runs a flow against a scripted
sequence of user inputs and returns, per step, the bot messages / edits /
deletes / visited nodes / errors — without touching Telegram or a real bot.

Contract (QAT'IY — engine team owns the other side):
  POST {simulator_url}/simulate
  Request : {"flow": {nodes, edges},
             "steps": [{"type": "command|text|contact|location|click|click_text", ...}],
             "collections": {name: [docs]},
             "chat": {id, first_name, username}}
  Response: {"ok": bool,
             "steps": [{"input", "bot_messages": [{message_id, text, parse_mode,
                        keyboard, reply_keyboard, removed_keyboard}],
                        "edits", "deletes", "nodes": [{id, type}], "errors"}],
             "final_state": {state, context},
             "collections_after", "errors"}

This module is a thin transport wrapper: it never inspects the flow itself, only
relays the request and surfaces a typed connection error so the MCP tool can tell
the user "the simulator service isn't running yet" distinctly from a flow error.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 90.0


class SimulatorUnavailableError(RuntimeError):
    """The simulator service couldn't be reached (connect error / timeout).

    Distinct from a simulation that ran but reported flow errors — those come
    back inside the normal response payload.
    """


async def simulate(
    *,
    simulator_url: str,
    flow: dict[str, Any],
    steps: list[dict[str, Any]],
    collections: dict[str, list[dict[str, Any]]] | None = None,
    chat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST a simulation request and return the parsed JSON response.

    Raises SimulatorUnavailableError if the service is unreachable (connect
    error / timeout). Any non-2xx HTTP status is surfaced as a normal dict with
    an ``errors`` entry so the tool can render it without a crash.
    """
    url = simulator_url.rstrip("/") + "/simulate"
    payload: dict[str, Any] = {
        "flow": flow or {"nodes": [], "edges": []},
        "steps": steps or [],
        "collections": collections or {},
        "chat": chat or {"id": 1, "first_name": "Test", "username": "test_user"},
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
        raise SimulatorUnavailableError(str(exc)) from exc
    except httpx.HTTPError as exc:  # other transport-level failures
        raise SimulatorUnavailableError(str(exc)) from exc

    if resp.status_code >= 400:
        body = resp.text[:500]
        return {
            "ok": False,
            "steps": [],
            "errors": [f"Simulyator HTTP {resp.status_code}: {body}"],
        }
    try:
        data = resp.json()
    except ValueError as exc:
        return {
            "ok": False,
            "steps": [],
            "errors": [f"Simulyator javobini o'qib bo'lmadi: {exc}"],
        }
    return data if isinstance(data, dict) else {"ok": False, "steps": [], "errors": ["Bad response"]}

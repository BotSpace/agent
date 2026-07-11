"""`sleep` — agent kutish tool'i.

Ba'zi backend amallari asinxron: `modules.enable` pod deploy qiladi (bir necha
soniyada bootlanadi), `release.deploy`/`release.debug` konteynerni ko'taradi.
Agent natijani DARROV so'rasa "hali tayyor emas" oladi. `sleep(seconds)` agentga
qisqa kutishni beradi, so'ng qayta so'raydi.

Agent-only: MCP klientlariga (dashboard konstruktori) ko'rsatilmaydi
(`expose_to_mcp=False`).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from .base import Tool, ToolContext, ToolResult, tool

# Agentni cheksiz to'xtatib qo'ymaslik uchun yuqori chegara.
_MAX_SLEEP_SECONDS = 30.0


class SleepInput(BaseModel):
    seconds: float = Field(
        description=(
            "Necha soniya kutish (0 dan katta). Yuqori chegara 30s. Asinxron "
            "backend amalidan keyin qayta so'rashdan oldin qisqa kutish uchun."
        ),
        gt=0,
    )


@tool(
    name="sleep",
    description=(
        "Wait for the given number of seconds before continuing (max 30s). Use it "
        "to let an async backend op settle before re-checking — e.g. after "
        "modules.enable (the module pod needs to boot) or release.deploy/debug "
        "(the container needs to come up) — then re-run the status/manifest call. "
        "Do NOT use it to poll in a tight loop; sleep once, then re-check."
    ),
    input_model=SleepInput,
    expose_to_mcp=False,
)
async def sleep_tool(input: SleepInput, ctx: ToolContext) -> ToolResult:
    secs = min(input.seconds, _MAX_SLEEP_SECONDS)
    await asyncio.sleep(secs)
    return ToolResult.ok(f"Waited {secs:g}s.")


def build_sleep_tools() -> list[Tool]:
    return [sleep_tool]

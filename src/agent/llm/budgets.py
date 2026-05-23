"""Per-run budget guardrails — token, tool-call, wall-clock."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .base import Usage


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_input_tokens: int
    max_output_tokens: int
    max_tool_calls: int
    max_wall_seconds: int

    # ── runtime counters ───────────────────────────────────────────────
    used_input: int = 0
    used_output: int = 0
    used_tool_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def add_usage(self, usage: Usage) -> None:
        self.used_input += usage.input_tokens
        self.used_output += usage.output_tokens
        self._check()

    def add_tool_call(self) -> None:
        self.used_tool_calls += 1
        self._check()

    def _check(self) -> None:
        if self.used_input > self.max_input_tokens:
            raise BudgetExceeded(
                f"Input token budget exceeded ({self.used_input}/{self.max_input_tokens})"
            )
        if self.used_output > self.max_output_tokens:
            raise BudgetExceeded(
                f"Output token budget exceeded ({self.used_output}/{self.max_output_tokens})"
            )
        if self.used_tool_calls > self.max_tool_calls:
            raise BudgetExceeded(
                f"Tool call budget exceeded ({self.used_tool_calls}/{self.max_tool_calls})"
            )
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.max_wall_seconds:
            raise BudgetExceeded(f"Wall-clock budget exceeded ({elapsed:.0f}s/{self.max_wall_seconds}s)")

    def snapshot(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.used_input,
            "output_tokens": self.used_output,
            "tool_calls": self.used_tool_calls,
            "wall_seconds": time.monotonic() - self.started_at,
        }

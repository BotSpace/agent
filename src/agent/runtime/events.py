"""Agent event types — SSE stream va store uchun yagona payload format."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "run_started",
    "thinking_delta",
    "assistant_chunk",
    "assistant_message",
    "tool_call_pending",
    "tool_call",
    "tool_result",
    "plan_set",
    "plan_update",
    "flow_changed",
    "workflow_progress",
    "context_compacted",
    "approval_required",
    "approval_resolved",
    "clarification_required",
    "clarification_resolved",
    "budget_update",
    "usage_updated",
    "token_usage",
    "sub_agent_started",
    "sub_agent_finished",
    "run_completed",
    "run_failed",
    "limit_exceeded",
    "error",
]


class AgentEvent(BaseModel):
    seq: int = Field(default=0, description="Monotonic per-run sequence number")
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default="", description="ISO-8601 timestamp")

    def to_sse(self) -> str:
        import json as _json

        body = _json.dumps(
            {"seq": self.seq, "type": self.type, "payload": self.payload, "at": self.at},
            ensure_ascii=False,
        )
        return f"event: {self.type}\ndata: {body}\n\n"

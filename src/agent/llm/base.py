"""Provider-agnostic LLM interface.

Har provider (Anthropic / OpenAI / Bedrock) shu protokolni amalga oshiradi.
Yuqori qatlamlar provider farqini bilmaydi — faqat `LLMClient` orqali murojaat qiladi.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class TextPart:
    text: str
    kind: Literal["text"] = "text"


@dataclass
class ToolUsePart:
    """LLM tomonidan chaqirilgan tool — assistant message ichida."""

    id: str
    name: str
    arguments: dict[str, Any]
    kind: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultPart:
    """Tool natijasi — user/tool message ichida."""

    tool_use_id: str
    content: str
    is_error: bool = False
    kind: Literal["tool_result"] = "tool_result"


@dataclass
class ThinkingPart:
    """Extended thinking (Claude) yoki reasoning tokens (OpenAI o-series)."""

    text: str
    kind: Literal["thinking"] = "thinking"


ContentPart = TextPart | ToolUsePart | ToolResultPart | ThinkingPart


@dataclass
class Message:
    role: Role
    content: list[ContentPart] = field(default_factory=list)

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls(role="user", content=[TextPart(text=text)])

    @classmethod
    def assistant_text(cls, text: str) -> "Message":
        return cls(role="assistant", content=[TextPart(text=text)])

    @classmethod
    def system_text(cls, text: str) -> "Message":
        return cls(role="system", content=[TextPart(text=text)])

    @classmethod
    def system_parts(cls, static: str, dynamic: str) -> "Message":
        """Two-block system message: cached static prefix + uncached dynamic suffix.

        The static block carries the cache_control breakpoint (set by the LLM
        client).  The dynamic block appears AFTER it so it is NOT included in
        the cached prefix — it changes every request.
        """
        return cls(role="system", content=[TextPart(text=static), TextPart(text=dynamic)])

    @classmethod
    def tool_result(cls, tool_use_id: str, content: str, is_error: bool = False) -> "Message":
        return cls(
            role="tool",
            content=[ToolResultPart(tool_use_id=tool_use_id, content=content, is_error=is_error)],
        )

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolUsePart]:
        return [p for p in self.content if isinstance(p, ToolUsePart)]


@dataclass
class ToolSpec:
    """JSON-schema describing a tool exposed to the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class GenerateOptions:
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    tools: list[ToolSpec] = field(default_factory=list)
    tool_choice: Literal["auto", "any", "none"] | dict[str, Any] = "auto"
    thinking_budget: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class GenerateResult:
    message: Message
    stop_reason: str
    usage: Usage
    raw: Any = None


# ── Streaming events ───────────────────────────────────────────────────────


@dataclass
class StreamStart:
    kind: Literal["start"] = "start"


@dataclass
class TextDelta:
    delta: str
    kind: Literal["text_delta"] = "text_delta"


@dataclass
class ThinkingDelta:
    delta: str
    kind: Literal["thinking_delta"] = "thinking_delta"


@dataclass
class ToolUseStart:
    id: str
    name: str
    kind: Literal["tool_use_start"] = "tool_use_start"


@dataclass
class ToolUseDelta:
    id: str
    arguments_delta: str
    kind: Literal["tool_use_delta"] = "tool_use_delta"


@dataclass
class ToolUseStop:
    id: str
    kind: Literal["tool_use_stop"] = "tool_use_stop"


@dataclass
class StreamStop:
    message: Message
    stop_reason: str
    usage: Usage
    kind: Literal["stop"] = "stop"


StreamEvent = (
    StreamStart
    | TextDelta
    | ThinkingDelta
    | ToolUseStart
    | ToolUseDelta
    | ToolUseStop
    | StreamStop
)


def normalize_tool_turns(messages: list[Message]) -> list[Message]:
    """Normalize an internal Message list so it satisfies strict API requirements.

    Both Anthropic Messages API and Bedrock Converse enforce:
      1. Roles must strictly ALTERNATE (no two consecutive same-role messages).
      2. The first non-system message must be ``user``.
      3. Every ``tool_result``/``toolResult`` for an assistant turn must live in
         the SINGLE user turn immediately following that assistant turn, and only
         results matching the prior turn's tool_use ids are kept (orphans dropped).

    This function operates on the internal ``Message`` list BEFORE any
    provider-specific block formatting, so both providers can call it first and
    then apply their own serialization.

    Steps:
      (a) ``tool``-role messages are re-labeled as ``user``.
      (b) Adjacent same-role messages are merged (content parts concatenated).
      (c) For each user turn: keep only ToolResultPart items whose
          ``tool_use_id`` matches a ToolUsePart in the immediately preceding
          assistant turn; drop the rest (orphans).
      (d) Empty messages (no content parts) are removed.
      (e) Step (b)+(d) are repeated once to handle new adjacency exposed by
          orphan removal.
      (f) Leading non-user (assistant) messages are dropped until the first
          message is ``user``.

    System messages are passed through unchanged.
    """
    # (a) re-label `tool` → `user`
    relabeled: list[Message] = []
    for msg in messages:
        if msg.role == "tool":
            relabeled.append(Message(role="user", content=list(msg.content)))
        else:
            relabeled.append(msg)

    def merge_adjacent(msgs: list[Message]) -> list[Message]:
        out: list[Message] = []
        for msg in msgs:
            if out and out[-1].role == msg.role and msg.role != "system":
                out[-1].content.extend(msg.content)
            else:
                out.append(Message(role=msg.role, content=list(msg.content)))
        return out

    # (b) first merge pass — collapses consecutive same-role messages
    merged = merge_adjacent(relabeled)

    # (c) prune orphan ToolResultParts
    for i, msg in enumerate(merged):
        if msg.role != "user":
            continue
        prev_tool_use_ids: set[str] = set()
        if i > 0 and merged[i - 1].role == "assistant":
            prev_tool_use_ids = {
                p.id
                for p in merged[i - 1].content
                if isinstance(p, ToolUsePart)
            }
        kept: list[ContentPart] = []
        for part in msg.content:
            if isinstance(part, ToolResultPart):
                if part.tool_use_id in prev_tool_use_ids:
                    kept.append(part)
                # else: orphan — drop silently
            else:
                kept.append(part)
        msg.content = kept

    # (d) remove empty messages (only non-system empties matter)
    non_empty = [m for m in merged if m.role == "system" or m.content]

    # (e) re-merge to collapse adjacency exposed by orphan removal
    re_merged = merge_adjacent(non_empty)

    # (f) drop leading non-user (assistant) turns
    while re_merged and re_merged[0].role not in ("user", "system"):
        re_merged.pop(0)

    return re_merged


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic protocol — har provider amalga oshiradi."""

    provider: str

    async def generate(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> GenerateResult: ...

    async def stream(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> AsyncIterator[StreamEvent]: ...


class LLMError(RuntimeError):
    """Provider tomonidan qaytarilgan barcha xatolar uchun umumiy ildiz."""

    def __init__(self, message: str, *, provider: str, status: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status

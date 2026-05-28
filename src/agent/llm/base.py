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
    response_format: dict[str, Any] | None = None
    """OpenAI-style response_format. E.g. {"type": "json_object"} to force JSON."""


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

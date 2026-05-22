"""Anthropic Claude provider — to'g'ridan-to'g'ri SDK orqali (LiteLLM kerak emas)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic, APIStatusError

from .base import (
    ContentPart,
    GenerateOptions,
    GenerateResult,
    LLMClient,
    LLMError,
    Message,
    StreamEvent,
    StreamStart,
    StreamStop,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolResultPart,
    ToolUseDelta,
    ToolUsePart,
    ToolUseStart,
    ToolUseStop,
    Usage,
)


_ALIAS_MAP = {
    # Internal short names → Anthropic API identifiers
    "claude-opus-4-7": "claude-opus-4-5-20251015",   # mavjud bo'lganga moslash
    "claude-sonnet-4-6": "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
}


def resolve_model(name: str) -> str:
    return _ALIAS_MAP.get(name, name)


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set", provider=self.provider)
        self._client = AsyncAnthropic(api_key=api_key)

    # ── conversion: internal → Anthropic ───────────────────────────────
    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system_text: str | None = None
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_text = (system_text + "\n\n" if system_text else "") + msg.text
                continue
            content: list[dict[str, Any]] = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ThinkingPart):
                    content.append({"type": "thinking", "thinking": part.text})
                elif isinstance(part, ToolUsePart):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": part.id,
                            "name": part.name,
                            "input": part.arguments,
                        }
                    )
                elif isinstance(part, ToolResultPart):
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": part.tool_use_id,
                            "content": part.content,
                            "is_error": part.is_error,
                        }
                    )
            # Anthropic — `tool` role yo'q, tool_result `user` ichida boradi
            role = "user" if msg.role == "tool" else msg.role
            out.append({"role": role, "content": content})
        return system_text, out

    @staticmethod
    def _from_anthropic_content(blocks: list[Any]) -> list[ContentPart]:
        parts: list[ContentPart] = []
        for b in blocks:
            t = getattr(b, "type", None)
            if t == "text":
                parts.append(TextPart(text=b.text))
            elif t == "thinking":
                parts.append(ThinkingPart(text=getattr(b, "thinking", "")))
            elif t == "tool_use":
                parts.append(
                    ToolUsePart(
                        id=b.id,
                        name=b.name,
                        arguments=dict(b.input) if b.input else {},
                    )
                )
        return parts

    @staticmethod
    def _tool_spec_for_api(options: GenerateOptions) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in options.tools
        ]

    @staticmethod
    def _build_request(messages: list[Message], options: GenerateOptions) -> dict[str, Any]:
        system_text, msgs = AnthropicClient._to_anthropic_messages(messages)
        req: dict[str, Any] = {
            "model": resolve_model(options.model),
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "messages": msgs,
        }
        if system_text:
            req["system"] = system_text
        if options.tools:
            req["tools"] = AnthropicClient._tool_spec_for_api(options)
            if options.tool_choice == "any":
                req["tool_choice"] = {"type": "any"}
            elif options.tool_choice == "none":
                req["tool_choice"] = {"type": "none"}
            elif isinstance(options.tool_choice, dict):
                req["tool_choice"] = options.tool_choice
        if options.stop_sequences:
            req["stop_sequences"] = options.stop_sequences
        if options.thinking_budget:
            req["thinking"] = {"type": "enabled", "budget_tokens": options.thinking_budget}
        return req

    # ── generate ───────────────────────────────────────────────────────
    async def generate(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> GenerateResult:
        req = self._build_request(messages, options)
        try:
            response = await self._client.messages.create(**req)
        except APIStatusError as e:
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        parts = self._from_anthropic_content(response.content)
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )
        return GenerateResult(
            message=Message(role="assistant", content=parts),
            stop_reason=response.stop_reason or "end_turn",
            usage=usage,
            raw=response,
        )

    # ── stream ─────────────────────────────────────────────────────────
    async def stream(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> AsyncIterator[StreamEvent]:
        req = self._build_request(messages, options)
        accumulated_parts: list[ContentPart] = []
        tool_buffers: dict[int, dict[str, Any]] = {}  # index → {id, name, json_str}
        stop_reason = "end_turn"
        usage = Usage()

        try:
            async with self._client.messages.stream(**req) as stream:
                yield StreamStart()
                async for event in stream:
                    et = event.type
                    if et == "content_block_start":
                        block = event.content_block
                        if block.type == "text":
                            accumulated_parts.append(TextPart(text=""))
                        elif block.type == "thinking":
                            accumulated_parts.append(ThinkingPart(text=""))
                        elif block.type == "tool_use":
                            tool_buffers[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "json": "",
                            }
                            accumulated_parts.append(
                                ToolUsePart(id=block.id, name=block.name, arguments={})
                            )
                            yield ToolUseStart(id=block.id, name=block.name)
                    elif et == "content_block_delta":
                        d = event.delta
                        if d.type == "text_delta":
                            last = accumulated_parts[-1]
                            if isinstance(last, TextPart):
                                last.text += d.text
                            yield TextDelta(delta=d.text)
                        elif d.type == "thinking_delta":
                            last = accumulated_parts[-1]
                            if isinstance(last, ThinkingPart):
                                last.text += d.thinking
                            yield ThinkingDelta(delta=d.thinking)
                        elif d.type == "input_json_delta":
                            buf = tool_buffers.get(event.index)
                            if buf is not None:
                                buf["json"] += d.partial_json
                                yield ToolUseDelta(
                                    id=buf["id"], arguments_delta=d.partial_json
                                )
                    elif et == "content_block_stop":
                        buf = tool_buffers.get(event.index)
                        if buf is not None:
                            try:
                                args = json.loads(buf["json"]) if buf["json"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            # Last accumulated tool_use part matches by id
                            for p in reversed(accumulated_parts):
                                if isinstance(p, ToolUsePart) and p.id == buf["id"]:
                                    p.arguments = args
                                    break
                            yield ToolUseStop(id=buf["id"])
                    elif et == "message_delta":
                        delta = event.delta
                        if hasattr(delta, "stop_reason") and delta.stop_reason:
                            stop_reason = delta.stop_reason
                        if event.usage:
                            usage.output_tokens = getattr(event.usage, "output_tokens", 0)
                    elif et == "message_start":
                        if event.message.usage:
                            usage.input_tokens = getattr(event.message.usage, "input_tokens", 0)
                            usage.cache_creation_tokens = (
                                getattr(event.message.usage, "cache_creation_input_tokens", 0) or 0
                            )
                            usage.cache_read_tokens = (
                                getattr(event.message.usage, "cache_read_input_tokens", 0) or 0
                            )
        except APIStatusError as e:
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        msg = Message(role="assistant", content=accumulated_parts)
        yield StreamStop(message=msg, stop_reason=stop_reason, usage=usage)

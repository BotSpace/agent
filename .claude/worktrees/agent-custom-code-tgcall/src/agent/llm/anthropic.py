"""Anthropic Claude provider — to'g'ridan-to'g'ri SDK orqali (LiteLLM kerak emas)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from anthropic import APIStatusError, AsyncAnthropic

from ..observability.metrics import (
    LLM_REQUEST_DURATION,
    LLM_REQUESTS,
    local_token_breakdown,
    record_llm_usage,
)
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
    normalize_tool_turns,
)


def _mark_last_content_block(msg: dict) -> None:
    """Place cache_control on the last content block of a converted message dict.

    Modifies `msg` in-place.  Safe to call multiple times — the breakpoint
    is idempotent (same value overwrites same value).
    """
    content = msg.get("content")
    if isinstance(content, list) and content:
        last_block = content[-1]
        if isinstance(last_block, dict):
            last_block["cache_control"] = {"type": "ephemeral"}
    elif isinstance(content, str):
        # Rare: plain-string content — convert to block form with cache_control.
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]


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
    def _to_anthropic_messages(
        messages: list[Message],
        *,
        inject_rolling_cache: bool = True,
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic API format.

        Returns (system, conversation_messages).

        system is one of:
          - None              — no system message
          - str               — single-block (plain text); cache_control is added
                                directly on the string-form system when possible,
                                or left bare for callers that already handle it.
          - list[dict]        — multi-block; block 0 carries cache_control (static
                                CORE_IDENTITY + catalog cached prefix).

        Rolling cache breakpoints (inject_rolling_cache=True, default):
          The last message in `messages` is assumed to be the dynamic-context
          tail (uncached, changes every turn).  Cache breakpoints are placed on
          the last AND second-to-last history messages (i.e., out[-2] and out[-3])
          using the Anthropic two-breakpoint rolling pattern.  The tail (out[-1])
          is intentionally NOT marked.

          Breakpoint budget: static-system (1) + rolling history (up to 2) = 3
          total, well within the 4-breakpoint limit.

          When there is no history (first turn with only dynamic tail), no
          rolling breakpoints are placed.

        Normalization: normalize_tool_turns() is called first so that parallel
        tool calls + the rolling-cache trailing dynamic-context user message
        never produce consecutive same-role messages, orphan tool_results are
        dropped, and the sequence always leads with `user`.
        """
        # ── normalize internal messages (tool→user, merge, prune orphans) ─
        messages = normalize_tool_turns(messages)

        system_parts: list[TextPart] = []
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                # Collect TextPart blocks; ignore non-text parts in system.
                for p in msg.content:
                    if isinstance(p, TextPart):
                        system_parts.append(p)
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
            # After normalization all `tool`-role messages are already `user`.
            out.append({"role": msg.role, "content": content})

        # ── rolling cache breakpoints on history ──────────────────────
        # Structure: out[-1] = dynamic-context tail (uncached).
        # out[:-1] = conversation history.  Mark the last and second-to-last
        # history messages.  Minimum 2 history messages required for both
        # breakpoints; with only 1 history message we mark just that one.
        if inject_rolling_cache and len(out) >= 2:
            # Indices of the last and second-to-last HISTORY messages
            # (excluding the dynamic tail at out[-1]).
            candidates = [-2, -3]
            for idx in candidates:
                try:
                    target = out[idx]
                except IndexError:
                    continue
                _mark_last_content_block(target)

        # ── system payload ─────────────────────────────────────────────
        if not system_parts:
            return None, out

        # All system blocks are returned as list[dict].
        # Block 0 (static CORE_IDENTITY + catalog) always carries cache_control.
        # Multi-block form (legacy two-block) is still supported: block 0 cached,
        # block 1+ uncached.
        system_blocks: list[dict[str, Any]] = []
        for i, p in enumerate(system_parts):
            block: dict[str, Any] = {"type": "text", "text": p.text}
            if i == 0:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)
        return system_blocks, out

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
        system_payload, msgs = AnthropicClient._to_anthropic_messages(messages)
        req: dict[str, Any] = {
            "model": resolve_model(options.model),
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "messages": msgs,
        }
        if system_payload:
            # system_payload is always list[dict] now — cache_control is on block 0.
            req["system"] = system_payload
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
        local = local_token_breakdown(messages, options)
        req = self._build_request(messages, options)
        t0 = time.monotonic()
        try:
            response = await self._client.messages.create(**req)
        except APIStatusError as e:
            LLM_REQUESTS.labels(self.provider, options.model, "error").inc()
            LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        LLM_REQUESTS.labels(self.provider, options.model, "success").inc()
        LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)

        parts = self._from_anthropic_content(response.content)
        raw_usage = response.usage
        usage_dict = {
            "input_tokens": getattr(raw_usage, "input_tokens", 0),
            "output_tokens": getattr(raw_usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
        }
        record_llm_usage(
            provider=self.provider,
            model=options.model,
            usage=usage_dict,
            local_breakdown=local,
        )
        usage = Usage(
            input_tokens=usage_dict["input_tokens"],
            output_tokens=usage_dict["output_tokens"],
            cache_creation_tokens=usage_dict["cache_creation_input_tokens"],
            cache_read_tokens=usage_dict["cache_read_input_tokens"],
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
        local = local_token_breakdown(messages, options)
        req = self._build_request(messages, options)
        accumulated_parts: list[ContentPart] = []
        tool_buffers: dict[int, dict[str, Any]] = {}  # index → {id, name, json_str}
        stop_reason = "end_turn"
        usage = Usage()
        t0 = time.monotonic()

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
            LLM_REQUESTS.labels(self.provider, options.model, "error").inc()
            LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        LLM_REQUESTS.labels(self.provider, options.model, "success").inc()
        LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
        record_llm_usage(
            provider=self.provider,
            model=options.model,
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_tokens,
                "cache_creation_input_tokens": usage.cache_creation_tokens,
            },
            local_breakdown=local,
        )

        msg = Message(role="assistant", content=accumulated_parts)
        yield StreamStop(message=msg, stop_reason=stop_reason, usage=usage)

"""OpenAI provider — Chat Completions API + tool calls + streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import APIStatusError, AsyncOpenAI

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
    ToolResultPart,
    ToolUseDelta,
    ToolUsePart,
    ToolUseStart,
    ToolUseStop,
    Usage,
)


# OpenAI function names must match ^[a-zA-Z0-9_-]+$, while internal tool
# names use dotted namespaces like "flow.get".
def _sanitize_tool_name(name: str) -> str:
    return name.replace(".", "--")


def _restore_tool_name(name: str) -> str:
    return name.replace("--", ".")


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set", provider=self.provider)
        self._client = AsyncOpenAI(api_key=api_key)

    # ── conversion: internal → OpenAI ──────────────────────────────────
    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                out.append({"role": "system", "content": msg.text})
                continue
            if msg.role == "tool":
                # Each tool_result becomes a separate {role: "tool", ...} message
                for part in msg.content:
                    if isinstance(part, ToolResultPart):
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_use_id,
                                "content": part.content,
                            }
                        )
                continue
            if msg.role == "user":
                out.append({"role": "user", "content": msg.text})
                continue
            # assistant
            text_segments: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    text_segments.append(part.text)
                elif isinstance(part, ToolUsePart):
                    tool_calls.append(
                        {
                            "id": part.id,
                            "type": "function",
                            "function": {
                                "name": _sanitize_tool_name(part.name),
                                "arguments": json.dumps(part.arguments),
                            },
                        }
                    )
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            joined = "\n".join(t for t in text_segments if t)
            if joined:
                assistant_msg["content"] = joined
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
        return out

    @staticmethod
    def _tool_spec_for_api(options: GenerateOptions) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": _sanitize_tool_name(t.name),
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in options.tools
        ]

    @staticmethod
    def _resolve_tool_choice(options: GenerateOptions) -> Any:
        if options.tool_choice == "any":
            return "required"
        if options.tool_choice == "none":
            return "none"
        if isinstance(options.tool_choice, dict):
            return options.tool_choice
        return "auto"

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        for prefix in ("o1", "o3", "o4", "gpt-4o", "gpt-5"):
            if model.startswith(prefix):
                return True
        return False

    def _build_request(self, messages: list[Message], options: GenerateOptions) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": options.model,
            "messages": self._to_openai_messages(messages),
            "temperature": options.temperature,
        }
        if self._uses_max_completion_tokens(options.model):
            req["max_completion_tokens"] = options.max_tokens
        else:
            req["max_tokens"] = options.max_tokens
        if options.tools:
            req["tools"] = self._tool_spec_for_api(options)
            req["tool_choice"] = self._resolve_tool_choice(options)
        if options.stop_sequences:
            req["stop"] = options.stop_sequences
        return req

    # ── generate ───────────────────────────────────────────────────────
    async def generate(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> GenerateResult:
        try:
            response = await self._client.chat.completions.create(**self._build_request(messages, options))
        except APIStatusError as e:
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        choice = response.choices[0]
        parts: list[ContentPart] = []
        if choice.message.content:
            parts.append(TextPart(text=choice.message.content))
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            parts.append(
                ToolUsePart(
                    id=tc.id,
                    name=_restore_tool_name(tc.function.name),
                    arguments=args,
                )
            )

        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return GenerateResult(
            message=Message(role="assistant", content=parts),
            stop_reason=choice.finish_reason or "stop",
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
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        accumulated_text = ""
        tool_acc: dict[int, dict[str, Any]] = {}  # index → {id, name, arguments_json}
        stop_reason = "stop"
        usage = Usage()

        try:
            stream = await self._client.chat.completions.create(**req)
            yield StreamStart()
            async for chunk in stream:
                if chunk.usage:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    stop_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    accumulated_text += delta.content
                    yield TextDelta(delta=delta.content)
                for tc_delta in delta.tool_calls or []:
                    idx = tc_delta.index
                    buf = tool_acc.setdefault(
                        idx,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        buf["name"] = _restore_tool_name(tc_delta.function.name)
                        yield ToolUseStart(id=buf["id"], name=buf["name"])
                    if tc_delta.function and tc_delta.function.arguments:
                        buf["arguments"] += tc_delta.function.arguments
                        yield ToolUseDelta(
                            id=buf["id"],
                            arguments_delta=tc_delta.function.arguments,
                        )
        except APIStatusError as e:
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        # Finalize
        parts: list[ContentPart] = []
        if accumulated_text:
            parts.append(TextPart(text=accumulated_text))
        for buf in tool_acc.values():
            try:
                args = json.loads(buf["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            parts.append(ToolUsePart(id=buf["id"], name=buf["name"], arguments=args))
            yield ToolUseStop(id=buf["id"])
        yield StreamStop(
            message=Message(role="assistant", content=parts),
            stop_reason=stop_reason,
            usage=usage,
        )

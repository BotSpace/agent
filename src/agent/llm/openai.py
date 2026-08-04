"""OpenAI provider — Chat Completions API + tool calls + streaming."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from ..observability.metrics import local_token_breakdown, record_llm_usage
from ._utils import restore_tool_name as _restore_tool_name
from ._utils import sanitize_tool_name as _sanitize_tool_name
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
    ToolResultPart,
    ToolUseDelta,
    ToolUsePart,
    ToolUseStart,
    ToolUseStop,
    Usage,
)


def _oai_mark_message(msg: dict[str, Any]) -> None:
    """Place cache_control on the last content block of an OpenAI message dict.

    OpenAI/OpenRouter cache_control sits inside content blocks.
    For string content, convert to block form first.
    Modifies `msg` in-place.
    """
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]
    elif isinstance(content, list) and content:
        last_block = content[-1]
        if isinstance(last_block, dict):
            last_block["cache_control"] = {"type": "ephemeral"}


def _wrap_for_langsmith(client: AsyncOpenAI) -> AsyncOpenAI:
    """If LangSmith tracing is enabled, wrap the OpenAI client so every
    chat-completions call is auto-traced. No-op when LangSmith isn't
    configured or the langsmith package isn't installed."""
    enabled = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").lower() == "true"
    if not enabled or not os.getenv("LANGSMITH_API_KEY"):
        return client
    try:
        from langsmith.wrappers import wrap_openai  # type: ignore[import-untyped]
    except ImportError:
        return client
    return wrap_openai(client)


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set", provider=self.provider)
        self._client = _wrap_for_langsmith(AsyncOpenAI(api_key=api_key))

    # ── conversion: internal → OpenAI ──────────────────────────────────
    @staticmethod
    def _sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop stale tool calls/results that cannot form a valid OpenAI turn."""
        sanitized: list[dict[str, Any]] = []
        pending_ids: set[str] = set()
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                following_result_ids: set[str] = set()
                for candidate in messages[index + 1 :]:
                    if candidate.get("role") != "tool":
                        break
                    result_id = candidate.get("tool_call_id")
                    if isinstance(result_id, str):
                        following_result_ids.add(result_id)
                filtered_calls = [
                    call
                    for call in message["tool_calls"]
                    if call.get("id") in following_result_ids
                ]
                cleaned = dict(message)
                if filtered_calls:
                    cleaned["tool_calls"] = filtered_calls
                    pending_ids = {str(call["id"]) for call in filtered_calls}
                else:
                    cleaned.pop("tool_calls", None)
                    pending_ids.clear()
                if cleaned.get("content") or cleaned.get("tool_calls"):
                    sanitized.append(cleaned)
                continue
            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id in pending_ids:
                    sanitized.append(message)
                    pending_ids.discard(tool_call_id)
                continue
            pending_ids.clear()
            sanitized.append(message)
        return sanitized

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                text_parts = [p for p in msg.content if isinstance(p, TextPart)]
                if len(text_parts) > 1:
                    # Multi-block system message (static + dynamic split).
                    # Keep as a list of content blocks so _inject_cache_markers
                    # can place the breakpoint on block 0 (static prefix).
                    out.append({
                        "role": "system",
                        "content": [{"type": "text", "text": p.text} for p in text_parts],
                    })
                else:
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
            # DeepSeek (strict) rejects an assistant message with NEITHER content
            # nor tool_calls — e.g. a max-thinking turn that emitted only
            # reasoning_content (dropped from history) and no answer. Such a
            # message carries nothing, so skip it instead of 400-ing the next turn.
            if "content" not in assistant_msg and "tool_calls" not in assistant_msg:
                continue
            out.append(assistant_msg)
        return OpenAIClient._sanitize_tool_pairs(out)

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

    @staticmethod
    def _needs_explicit_cache_marker(model: str) -> bool:
        """Models routed through OpenRouter that REQUIRE cache_control
        markers — automatic prefix caching is not enabled for them.
        OpenAI gpt-*, DeepSeek, Grok, Gemini 2.5 cache implicitly so
        we leave them alone."""
        m = model.lower()
        return (
            m.startswith("qwen/")
            or m.startswith("alibaba/")
            or m.startswith("anthropic/")
            or "claude" in m
            or m.startswith("google/gemini-1.5")
            or m.startswith("google/gemini-2.0")
        )

    @classmethod
    def _local_breakdown(
        cls, messages: list[Message], options: GenerateOptions
    ) -> dict[str, int]:
        """Delegate to shared helper in metrics module."""
        return local_token_breakdown(messages, options)

    def _build_request(self, messages: list[Message], options: GenerateOptions) -> dict[str, Any]:
        oai_messages = self._to_openai_messages(messages)
        if self._needs_explicit_cache_marker(options.model):
            self._inject_cache_markers(oai_messages)
        req: dict[str, Any] = {
            "model": options.model,
            "messages": oai_messages,
            "temperature": options.temperature,
        }
        if self._uses_max_completion_tokens(options.model):
            req["max_completion_tokens"] = options.max_tokens
        else:
            req["max_tokens"] = options.max_tokens
        if options.tools:
            req["tools"] = self._tool_spec_for_api(options)
            req["tool_choice"] = self._resolve_tool_choice(options)
            req["parallel_tool_calls"] = True
        if options.stop_sequences:
            req["stop"] = options.stop_sequences
        self._apply_reasoning(req, options)
        return req

    def _apply_reasoning(self, req: dict[str, Any], options: GenerateOptions) -> None:
        """Provider hook: attach reasoning/thinking params. No-op for plain OpenAI;
        DeepSeek overrides to toggle thinking mode + reasoning_effort."""
        return None

    @staticmethod
    def _inject_cache_markers(oai_messages: list[dict[str, Any]]) -> None:
        """Place cache_control breakpoints for rolling incremental prompt caching.

        Breakpoint 1 — static system prefix (CORE_IDENTITY + catalog):
          Placed on the first (and only) content block of the system message.
          When the system has a single string, it is converted to block form.
          When the system has two content blocks (legacy static+dynamic split),
          block 0 is marked.

        Breakpoints 2-3 -- rolling conversation history:
          The last message in `oai_messages` is the dynamic-context tail
          (uncached; changes every turn).  The last and second-to-last
          NON-SYSTEM messages that are NOT the tail are the most-recent stable
          history messages.  We place cache_control on them so Anthropic/Qwen
          cache the conversation history prefix up to that point.  The tail
          itself is intentionally NOT marked.

          Total breakpoints: 1 (system) + up to 2 (history) = 3 max.
          This is within the 4-breakpoint budget.

        Providers that honor cache_control (Anthropic / Qwen / Gemini via
        OpenRouter) will cache the prefix up to and including each marked block.
        Tool schemas in the `tools` array are part of the same prefix and get
        cached together by Anthropic (best-effort by other providers).
        """
        # ── Breakpoint 1: system block ────────────────────────────────
        for m in oai_messages:
            if m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, str):
                    # Single-string form — convert to block form with cache_control.
                    m["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(content, list) and content:
                    # Block-list form: mark block 0 (static prefix).
                    first = content[0]
                    if isinstance(first, dict):
                        first["cache_control"] = {"type": "ephemeral"}
                break  # only first system message

        # ── Breakpoints 2-3: last and second-to-last history messages ──
        # non_system = all non-system messages in order.
        # non_system[-1] = dynamic-context tail (do NOT mark).
        # non_system[-2] = last history message (mark).
        # non_system[-3] = second-to-last history message (mark, if exists).
        non_system = [m for m in oai_messages if m.get("role") != "system"]
        # Need at least: tail + 1 history message to place any rolling breakpoint.
        if len(non_system) >= 2:
            candidates = [-2, -3]
            for idx in candidates:
                try:
                    target = non_system[idx]
                except IndexError:
                    continue
                _oai_mark_message(target)

    # ── generate ───────────────────────────────────────────────────────
    async def generate(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> GenerateResult:
        from ..observability.metrics import (
            LLM_REQUEST_DURATION,
            LLM_REQUESTS,
        )

        local = self._local_breakdown(messages, options)
        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(**self._build_request(messages, options))
        except APIStatusError as e:
            LLM_REQUESTS.labels(self.provider, options.model, "error").inc()
            LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        LLM_REQUESTS.labels(self.provider, options.model, "success").inc()
        LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
        try:
            usage_dict = response.usage.model_dump() if response.usage else None
        except AttributeError:
            usage_dict = dict(response.usage) if response.usage else None
        record_llm_usage(
            provider=self.provider,
            model=options.model,
            usage=usage_dict,
            local_breakdown=local,
        )

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

        # OpenAI/OpenRouter report cached prompt tokens under
        # prompt_tokens_details.cached_tokens — surface them as cache_read so
        # the event/display reflects the (already working) prompt caching
        # instead of showing the full input as if nothing were cached.
        cached_read = 0
        if response.usage is not None:
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details is not None:
                cached_read = getattr(details, "cached_tokens", 0) or 0
            elif isinstance(getattr(response.usage, "model_extra", None), dict):
                pd = response.usage.model_extra.get("prompt_tokens_details") or {}
                cached_read = (pd or {}).get("cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            cache_read_tokens=cached_read,
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
        from ..observability.metrics import (
            LLM_REQUEST_DURATION,
            LLM_REQUESTS,
        )

        local = self._local_breakdown(messages, options)
        req = self._build_request(messages, options)
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        accumulated_text = ""
        tool_acc: dict[int, dict[str, Any]] = {}  # index → {id, name, arguments_json}
        stop_reason = "stop"
        usage = Usage()
        raw_usage_dict: dict | None = None
        t0 = time.monotonic()

        try:
            stream = await self._client.chat.completions.create(**req)
            yield StreamStart()
            async for chunk in stream:
                if chunk.usage:
                    details = getattr(chunk.usage, "prompt_tokens_details", None)
                    cached_read = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        cache_read_tokens=cached_read,
                    )
                    try:
                        raw_usage_dict = chunk.usage.model_dump()
                    except AttributeError:
                        raw_usage_dict = dict(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    stop_reason = choice.finish_reason
                delta = choice.delta
                # DeepSeek reasoner models (and other reasoning-capable OpenAI-
                # compatible endpoints) stream their chain-of-thought in a separate
                # `reasoning_content` field BEFORE the final answer. Surface it as a
                # thinking delta so the UI shows live progress instead of a silent
                # wait at the hard step. Not accumulated into the reply text or the
                # returned message — DeepSeek discards reasoning on the next turn.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning is None and getattr(delta, "model_extra", None):
                    reasoning = delta.model_extra.get("reasoning_content")
                if reasoning:
                    yield ThinkingDelta(delta=reasoning)
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
            LLM_REQUESTS.labels(self.provider, options.model, "error").inc()
            LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
            raise LLMError(str(e), provider=self.provider, status=e.status_code) from e

        LLM_REQUESTS.labels(self.provider, options.model, "success").inc()
        LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
        record_llm_usage(
            provider=self.provider,
            model=options.model,
            usage=raw_usage_dict,
            local_breakdown=local,
        )

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

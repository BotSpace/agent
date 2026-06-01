"""AWS Bedrock provider — Anthropic Claude on Bedrock (Converse API).

Boshqa Bedrock modellari (Llama, Mistral) ham Converse API qabul qiladi,
shuning uchun bu provider asosan model-agnostik.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import boto3
from botocore.exceptions import ClientError

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
    ToolResultPart,
    ToolUseDelta,
    ToolUsePart,
    ToolUseStart,
    ToolUseStop,
    Usage,
)

# Bedrock Claude 4-avlod modellari on-demand'da chaqirilmaydi — cross-region
# inference profile (us. prefix) talab qilinadi.
_BEDROCK_ALIAS = {
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "claude-opus-4-5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


def resolve_bedrock_model(name: str) -> str:
    return _BEDROCK_ALIAS.get(name, name)


# Bedrock tool name regex is [a-zA-Z0-9_-]+ — no dots allowed.
# We swap "." → "--" on the way out and reverse it on tool_use callbacks.
def _sanitize_tool_name(name: str) -> str:
    return name.replace(".", "--")


def _restore_tool_name(name: str) -> str:
    return name.replace("--", ".")


class BedrockClient(LLMClient):
    provider = "bedrock"

    def __init__(
        self,
        *,
        region: str,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        session = boto3.Session(
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
            region_name=region,
        )
        self._client = session.client("bedrock-runtime")
        self._region = region

    # ── conversion ─────────────────────────────────────────────────────
    @staticmethod
    def _to_bedrock_messages(
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert internal messages to a VALID Bedrock Converse message list.

        Converse is far stricter than the OpenAI-compatible path:
          1. Roles must strictly ALTERNATE (no two consecutive same-role msgs).
          2. The first non-system message must be ``user``.
          3. Every ``toolResult`` answering an assistant turn must live in the
             SINGLE user message immediately following that assistant turn, and
             the toolResult set must not exceed the prior turn's toolUse set
             (one result per use, matching ``toolUseId``).

        The internal format keeps each tool result in its own ``tool``-role
        message, and parallel tool calls produce several of them in a row.  A
        naive 1:1 conversion then emits consecutive ``user`` messages, each with
        one toolResult, which Converse counts as "more toolResults than the
        previous turn's toolUses" → ``ValidationException``.

        Strategy: (a) convert each message to a block list, (b) merge adjacent
        same-role messages into one, (c) prune orphan toolResults against the
        preceding assistant turn's toolUseIds, (d) ensure the sequence leads
        with ``user`` and strictly alternates.
        """
        system_blocks: list[dict[str, Any]] = []
        # Step 1 — flatten each internal message into (role, content-blocks).
        flat: list[tuple[str, list[dict[str, Any]]]] = []
        for msg in messages:
            if msg.role == "system":
                system_blocks.append({"text": msg.text})
                continue
            content: list[dict[str, Any]] = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    # Drop empty text blocks — Converse rejects empty content.
                    if part.text != "":
                        content.append({"text": part.text})
                elif isinstance(part, ToolUsePart):
                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": part.id,
                                "name": _sanitize_tool_name(part.name),
                                "input": part.arguments,
                            }
                        }
                    )
                elif isinstance(part, ToolResultPart):
                    content.append(
                        {
                            "toolResult": {
                                "toolUseId": part.tool_use_id,
                                "content": [{"text": part.content}],
                                "status": "error" if part.is_error else "success",
                            }
                        }
                    )
            if not content:
                # An empty turn (e.g. assistant with only blank text) cannot be
                # emitted; skipping it also avoids spurious same-role adjacency.
                continue
            role = "user" if msg.role == "tool" else msg.role
            flat.append((role, content))

        # Step 2 — merge adjacent same-role messages into one Bedrock message.
        # Collapses N consecutive toolResult `user` turns into one, and folds a
        # trailing dynamic-context `user` text into the same user turn.
        merged: list[dict[str, Any]] = []
        for role, content in flat:
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"].extend(content)
            else:
                merged.append({"role": role, "content": list(content)})

        # Step 3 — prune orphan toolResults. Within a user turn, keep only
        # toolResults whose toolUseId matches a toolUse from the immediately
        # preceding assistant turn. Orphans (after interrupts/resumes such as
        # ask_user / flow.clear) would make the toolResult count exceed the
        # prior toolUse count and trigger the ValidationException.
        for i, m in enumerate(merged):
            if m["role"] != "user":
                continue
            prev_tool_use_ids: set[str] = set()
            if i > 0 and merged[i - 1]["role"] == "assistant":
                prev_tool_use_ids = {
                    b["toolUse"]["toolUseId"]
                    for b in merged[i - 1]["content"]
                    if "toolUse" in b
                }
            kept: list[dict[str, Any]] = []
            for block in m["content"]:
                if "toolResult" in block:
                    if block["toolResult"]["toolUseId"] in prev_tool_use_ids:
                        kept.append(block)
                    # else: orphan toolResult — drop it.
                else:
                    kept.append(block)
            m["content"] = kept

        # A turn may now be empty (its only block was an orphan toolResult).
        merged = [m for m in merged if m["content"]]

        # Re-merge: dropping empty turns can expose new same-role adjacency
        # (e.g. assistant, [dropped user], assistant).
        normalized: list[dict[str, Any]] = []
        for m in merged:
            if normalized and normalized[-1]["role"] == m["role"]:
                normalized[-1]["content"].extend(m["content"])
            else:
                normalized.append(m)

        # Step 4 — Converse requires the first message to be `user`. If the
        # history starts with an assistant turn (rare; should not happen with a
        # real conversation), drop leading assistant turns until a user leads.
        while normalized and normalized[0]["role"] != "user":
            normalized.pop(0)

        return system_blocks, normalized

    @staticmethod
    def _tool_config(options: GenerateOptions) -> dict[str, Any] | None:
        if not options.tools:
            return None
        tools = [
            {
                "toolSpec": {
                    "name": _sanitize_tool_name(t.name),
                    "description": t.description,
                    "inputSchema": {"json": t.input_schema},
                }
            }
            for t in options.tools
        ]
        cfg: dict[str, Any] = {"tools": tools}
        if options.tool_choice == "any":
            cfg["toolChoice"] = {"any": {}}
        elif options.tool_choice == "none":
            pass  # Bedrock has no explicit "none" — caller must omit tools
        return cfg

    def _build_request(self, messages: list[Message], options: GenerateOptions) -> dict[str, Any]:
        system, msgs = self._to_bedrock_messages(messages)
        req: dict[str, Any] = {
            "modelId": resolve_bedrock_model(options.model),
            "messages": msgs,
            "inferenceConfig": {
                "maxTokens": options.max_tokens,
                "temperature": options.temperature,
            },
        }
        if system:
            req["system"] = system
        if options.stop_sequences:
            req["inferenceConfig"]["stopSequences"] = options.stop_sequences
        tc = self._tool_config(options)
        if tc is not None:
            req["toolConfig"] = tc
        return req

    @staticmethod
    def _from_bedrock_content(content: list[dict[str, Any]]) -> list[ContentPart]:
        parts: list[ContentPart] = []
        for block in content:
            if "text" in block:
                parts.append(TextPart(text=block["text"]))
            elif "toolUse" in block:
                tu = block["toolUse"]
                parts.append(
                    ToolUsePart(
                        id=tu["toolUseId"],
                        name=_restore_tool_name(tu["name"]),
                        arguments=dict(tu.get("input") or {}),
                    )
                )
        return parts

    # ── generate (non-streaming) ───────────────────────────────────────
    async def generate(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> GenerateResult:
        import anyio

        local = local_token_breakdown(messages, options)
        req = self._build_request(messages, options)
        t0 = time.monotonic()
        try:
            response = await anyio.to_thread.run_sync(lambda: self._client.converse(**req))
        except ClientError as e:
            LLM_REQUESTS.labels(self.provider, options.model, "error").inc()
            LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
            raise LLMError(str(e), provider=self.provider) from e

        LLM_REQUESTS.labels(self.provider, options.model, "success").inc()
        LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)

        output_msg = response["output"]["message"]
        parts = self._from_bedrock_content(output_msg.get("content", []))
        usage_data = response.get("usage", {})
        record_llm_usage(
            provider=self.provider,
            model=options.model,
            usage=usage_data,
            local_breakdown=local,
        )
        usage = Usage(
            input_tokens=usage_data.get("inputTokens", 0),
            output_tokens=usage_data.get("outputTokens", 0),
        )
        return GenerateResult(
            message=Message(role="assistant", content=parts),
            stop_reason=response.get("stopReason", "end_turn"),
            usage=usage,
            raw=response,
        )

    # ── stream ─────────────────────────────────────────────────────────
    async def stream(
        self,
        messages: list[Message],
        options: GenerateOptions,
    ) -> AsyncIterator[StreamEvent]:
        import anyio

        local = local_token_breakdown(messages, options)
        req = self._build_request(messages, options)
        t0 = time.monotonic()
        try:
            response = await anyio.to_thread.run_sync(lambda: self._client.converse_stream(**req))
        except ClientError as e:
            LLM_REQUESTS.labels(self.provider, options.model, "error").inc()
            LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
            raise LLMError(str(e), provider=self.provider) from e

        accumulated_parts: list[ContentPart] = []
        # contentBlockStart/Delta/Stop use index
        tool_buffers: dict[int, dict[str, Any]] = {}
        stop_reason = "end_turn"
        usage = Usage()
        raw_usage_data: dict[str, Any] = {}
        yield StreamStart()

        stream_iter = response["stream"]
        # boto3 EventStream is synchronous; iterate in a thread
        events = await anyio.to_thread.run_sync(lambda: list(stream_iter))
        for event in events:
            if "messageStart" in event:
                pass
            elif "contentBlockStart" in event:
                start = event["contentBlockStart"]
                idx = start["contentBlockIndex"]
                start_obj = start.get("start", {})
                if "toolUse" in start_obj:
                    tu = start_obj["toolUse"]
                    real_name = _restore_tool_name(tu["name"])
                    tool_buffers[idx] = {"id": tu["toolUseId"], "name": real_name, "json": ""}
                    accumulated_parts.append(
                        ToolUsePart(id=tu["toolUseId"], name=real_name, arguments={})
                    )
                    yield ToolUseStart(id=tu["toolUseId"], name=real_name)
                else:
                    accumulated_parts.append(TextPart(text=""))
            elif "contentBlockDelta" in event:
                d = event["contentBlockDelta"]
                idx = d["contentBlockIndex"]
                delta = d.get("delta", {})
                if "text" in delta:
                    text = delta["text"]
                    # Append to last TextPart at idx
                    for p in reversed(accumulated_parts):
                        if isinstance(p, TextPart):
                            p.text += text
                            break
                    yield TextDelta(delta=text)
                elif "toolUse" in delta:
                    buf = tool_buffers.get(idx)
                    if buf:
                        partial = delta["toolUse"].get("input", "")
                        buf["json"] += partial
                        yield ToolUseDelta(id=buf["id"], arguments_delta=partial)
            elif "contentBlockStop" in event:
                idx = event["contentBlockStop"]["contentBlockIndex"]
                buf = tool_buffers.get(idx)
                if buf:
                    try:
                        args = json.loads(buf["json"]) if buf["json"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    for p in reversed(accumulated_parts):
                        if isinstance(p, ToolUsePart) and p.id == buf["id"]:
                            p.arguments = args
                            break
                    yield ToolUseStop(id=buf["id"])
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "end_turn")
            elif "metadata" in event:
                u = event["metadata"].get("usage", {})
                raw_usage_data = u
                usage = Usage(
                    input_tokens=u.get("inputTokens", 0),
                    output_tokens=u.get("outputTokens", 0),
                )

        LLM_REQUESTS.labels(self.provider, options.model, "success").inc()
        LLM_REQUEST_DURATION.labels(self.provider, options.model).observe(time.monotonic() - t0)
        record_llm_usage(
            provider=self.provider,
            model=options.model,
            usage=raw_usage_data,
            local_breakdown=local,
        )

        yield StreamStop(
            message=Message(role="assistant", content=accumulated_parts),
            stop_reason=stop_reason,
            usage=usage,
        )

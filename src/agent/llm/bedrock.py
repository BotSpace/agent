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
from botocore.config import Config as BotoConfig
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
    normalize_tool_turns,
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


# Bedrock Converse prompt-caching marker. Unlike the Anthropic native API
# (which uses inline ``cache_control: {type: ephemeral}`` on content blocks),
# Bedrock Converse caches by inserting a standalone ``cachePoint`` block into
# the system / tools / messages arrays — everything BEFORE the marker is cached.
_CACHE_POINT: dict[str, Any] = {"cachePoint": {"type": "default"}}


def _bedrock_supports_caching(model_id: str) -> bool:
    """Prompt caching via cachePoint is only valid for Anthropic Claude on
    Bedrock (3.5v2 / 3.7 / 4.x). Other families (Qwen, Llama, Mistral, ...)
    reject cachePoint blocks with a ValidationException, so gate on Claude.
    """
    return "anthropic.claude" in model_id.lower()


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
        # Bedrock on-demand Converse is rate-limited (RPM/TPM). The agent fans
        # out across many sub-agent calls, which bursts into ThrottlingException
        # ("Too many requests") and crashes the sub-agent. `adaptive` retry mode
        # adds a client-side rate-limiter + exponential backoff with jitter, so
        # bursts are smoothed out instead of failing. Note: this rides out
        # transient/burst throttling, NOT a hard account quota ceiling — for
        # sustained throttling, request a Bedrock quota increase for the model.
        self._client = session.client(
            "bedrock-runtime",
            config=BotoConfig(
                retries={"max_attempts": 10, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=120,
            ),
        )
        self._region = region

    # ── conversion ─────────────────────────────────────────────────────
    @staticmethod
    def _to_bedrock_messages(
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert internal messages to a VALID Bedrock Converse message list.

        Delegates normalization (tool→user relabeling, adjacent-role merge,
        orphan-toolResult pruning, empty-turn removal, leading-user guarantee)
        to the shared ``normalize_tool_turns`` helper from ``base``, then
        serializes each normalized message into Bedrock Converse block dicts.
        """
        # --- system blocks (extracted before normalization) ---------------
        system_blocks: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_blocks.append({"text": msg.text})

        # --- normalize non-system messages --------------------------------
        normalized = normalize_tool_turns(messages)

        # --- serialize to Bedrock block dicts -----------------------------
        out: list[dict[str, Any]] = []
        for msg in normalized:
            if msg.role == "system":
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
            if content:
                out.append({"role": msg.role, "content": content})

        return system_blocks, out

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
        model_id = resolve_bedrock_model(options.model)
        cache = _bedrock_supports_caching(model_id)

        # Rolling conversation cache: a cachePoint at the END of the last
        # message caches the whole prefix; on the next turn that prefix is
        # identical (the breakpoint becomes the readable cache point), so it is
        # read from cache and only the new tail is written. Combined with the
        # static system + tools breakpoints this gives full prompt caching.
        if cache and msgs:
            msgs[-1] = {**msgs[-1], "content": [*msgs[-1]["content"], _CACHE_POINT]}

        req: dict[str, Any] = {
            "modelId": model_id,
            "messages": msgs,
            "inferenceConfig": {
                "maxTokens": options.max_tokens,
                "temperature": options.temperature,
            },
        }
        if system:
            # cachePoint after the static system blocks (CORE_IDENTITY + node
            # catalog) — the largest, most stable chunk of the prompt.
            req["system"] = [*system, _CACHE_POINT] if cache else system
        if options.stop_sequences:
            req["inferenceConfig"]["stopSequences"] = options.stop_sequences
        tc = self._tool_config(options)
        if tc is not None:
            if cache and tc.get("tools"):
                # cachePoint as the last tools entry caches all tool schemas.
                tc = {**tc, "tools": [*tc["tools"], _CACHE_POINT]}
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
            cache_read_tokens=usage_data.get("cacheReadInputTokens", 0) or 0,
            cache_creation_tokens=usage_data.get("cacheWriteInputTokens", 0) or 0,
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
                    cache_read_tokens=u.get("cacheReadInputTokens", 0) or 0,
                    cache_creation_tokens=u.get("cacheWriteInputTokens", 0) or 0,
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

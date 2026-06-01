"""Unit tests: LLM metrics instrumentation for bedrock and anthropic providers.

Verifies that calling generate (mocked) increments LLM_INPUT_TOKENS,
LLM_OUTPUT_TOKENS and LLM_REQUESTS for the correct provider/model labels.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm.base import GenerateOptions, Message
from agent.observability.metrics import (
    LLM_INPUT_TOKENS,
    LLM_OUTPUT_TOKENS,
    LLM_REQUESTS,
    local_token_breakdown,
    record_llm_usage,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _counter_value(counter, *label_values: str) -> float:
    """Read a prometheus Counter value by label tuple."""
    return counter.labels(*label_values)._value.get()


def _sample_messages() -> list[Message]:
    return [
        Message.system_text("You are a helpful assistant."),
        Message.user_text("Hello, world!"),
    ]


def _sample_options(model: str = "test-model") -> GenerateOptions:
    return GenerateOptions(model=model, max_tokens=128, temperature=0.0)


# ── record_llm_usage normalization ────────────────────────────────────────


class TestRecordLlmUsageNormalization:
    """record_llm_usage should accept all three provider usage dict shapes."""

    def test_openai_flavor(self) -> None:
        before = _counter_value(LLM_OUTPUT_TOKENS, "openai", "norm-model")
        record_llm_usage(
            provider="openai",
            model="norm-model",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 20},
            },
        )
        assert _counter_value(LLM_OUTPUT_TOKENS, "openai", "norm-model") == before + 50
        assert _counter_value(LLM_INPUT_TOKENS, "openai", "norm-model", "prompt_total") >= 100
        assert _counter_value(LLM_INPUT_TOKENS, "openai", "norm-model", "cache_read") >= 30
        assert _counter_value(LLM_INPUT_TOKENS, "openai", "norm-model", "cache_write") >= 20

    def test_bedrock_flavor(self) -> None:
        before = _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "bedrock-norm")
        record_llm_usage(
            provider="bedrock",
            model="bedrock-norm",
            usage={
                "inputTokens": 200,
                "outputTokens": 80,
                "cacheReadInputTokens": 60,
                "cacheWriteInputTokens": 40,
            },
        )
        assert _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "bedrock-norm") == before + 80
        assert _counter_value(LLM_INPUT_TOKENS, "bedrock", "bedrock-norm", "prompt_total") >= 200
        assert _counter_value(LLM_INPUT_TOKENS, "bedrock", "bedrock-norm", "cache_read") >= 60
        assert _counter_value(LLM_INPUT_TOKENS, "bedrock", "bedrock-norm", "cache_write") >= 40

    def test_anthropic_sdk_flavor(self) -> None:
        before = _counter_value(LLM_OUTPUT_TOKENS, "anthropic", "anth-norm")
        record_llm_usage(
            provider="anthropic",
            model="anth-norm",
            usage={
                "input_tokens": 150,
                "output_tokens": 70,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 30,
            },
        )
        assert _counter_value(LLM_OUTPUT_TOKENS, "anthropic", "anth-norm") == before + 70
        assert _counter_value(LLM_INPUT_TOKENS, "anthropic", "anth-norm", "prompt_total") >= 150
        assert _counter_value(LLM_INPUT_TOKENS, "anthropic", "anth-norm", "cache_read") >= 50
        assert _counter_value(LLM_INPUT_TOKENS, "anthropic", "anth-norm", "cache_write") >= 30

    def test_missing_cache_fields_tolerated(self) -> None:
        """Cache fields absent — must not raise, cache counters stay at zero increment."""
        before_read = _counter_value(LLM_INPUT_TOKENS, "bedrock", "nocache-model", "cache_read")
        before_write = _counter_value(LLM_INPUT_TOKENS, "bedrock", "nocache-model", "cache_write")
        record_llm_usage(
            provider="bedrock",
            model="nocache-model",
            usage={"inputTokens": 10, "outputTokens": 5},
        )
        assert _counter_value(LLM_INPUT_TOKENS, "bedrock", "nocache-model", "cache_read") == before_read
        assert _counter_value(LLM_INPUT_TOKENS, "bedrock", "nocache-model", "cache_write") == before_write

    def test_none_usage_is_noop(self) -> None:
        before = _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "null-model")
        record_llm_usage(provider="bedrock", model="null-model", usage=None)
        assert _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "null-model") == before


# ── local_token_breakdown ─────────────────────────────────────────────────


class TestLocalTokenBreakdown:
    def test_returns_four_categories(self) -> None:
        msgs = _sample_messages()
        opts = _sample_options()
        result = local_token_breakdown(msgs, opts)
        assert set(result.keys()) == {"system", "tool_schema", "user", "tool_result"}

    def test_system_and_user_nonzero(self) -> None:
        msgs = _sample_messages()
        opts = _sample_options()
        result = local_token_breakdown(msgs, opts)
        assert result["system"] > 0
        assert result["user"] > 0
        assert result["tool_result"] == 0
        assert result["tool_schema"] == 0


# ── Bedrock provider metrics ───────────────────────────────────────────────


class TestBedrockMetrics:
    @pytest.fixture
    def bedrock_client(self):
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.client.return_value = MagicMock()
            from agent.llm.bedrock import BedrockClient
            client = BedrockClient(region="us-east-1")
            return client

    def test_generate_increments_tokens(self, bedrock_client) -> None:
        """generate() with mocked Converse response increments LLM metrics."""
        fake_response = {
            "output": {"message": {"content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 42,
                "outputTokens": 17,
            },
        }

        async def run():
            with patch("anyio.to_thread.run_sync", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = fake_response
                msgs = _sample_messages()
                opts = _sample_options(model="bedrock-test-generate")
                before_output = _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "bedrock-test-generate")
                before_input = _counter_value(LLM_INPUT_TOKENS, "bedrock", "bedrock-test-generate", "prompt_total")
                before_req = _counter_value(LLM_REQUESTS, "bedrock", "bedrock-test-generate", "success")
                await bedrock_client.generate(msgs, opts)
                assert _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "bedrock-test-generate") == before_output + 17
                assert _counter_value(LLM_INPUT_TOKENS, "bedrock", "bedrock-test-generate", "prompt_total") == before_input + 42
                assert _counter_value(LLM_REQUESTS, "bedrock", "bedrock-test-generate", "success") == before_req + 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_generate_error_increments_error_counter(self, bedrock_client) -> None:
        from botocore.exceptions import ClientError

        from agent.llm.base import LLMError

        async def run():
            error_response = {"Error": {"Code": "ServiceUnavailableException", "Message": "down"}}
            exc = ClientError(error_response, "Converse")
            with patch("anyio.to_thread.run_sync", new_callable=AsyncMock, side_effect=exc):
                msgs = _sample_messages()
                opts = _sample_options(model="bedrock-test-err")
                before = _counter_value(LLM_REQUESTS, "bedrock", "bedrock-test-err", "error")
                with pytest.raises(LLMError):
                    await bedrock_client.generate(msgs, opts)
                assert _counter_value(LLM_REQUESTS, "bedrock", "bedrock-test-err", "error") == before + 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_stream_increments_tokens(self, bedrock_client) -> None:
        """stream() with mocked converse_stream response increments LLM metrics."""
        fake_stream_response = {
            "stream": [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": {"inputTokens": 55, "outputTokens": 22}}},
            ]
        }

        async def run():
            with patch("anyio.to_thread.run_sync", new_callable=AsyncMock) as mock_run:
                # First call: converse_stream returns fake_stream_response
                # Second call: list(stream_iter) is also run_sync
                mock_run.side_effect = [
                    fake_stream_response,
                    fake_stream_response["stream"],
                ]
                msgs = _sample_messages()
                opts = _sample_options(model="bedrock-test-stream")
                before_output = _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "bedrock-test-stream")
                before_input = _counter_value(LLM_INPUT_TOKENS, "bedrock", "bedrock-test-stream", "prompt_total")
                before_req = _counter_value(LLM_REQUESTS, "bedrock", "bedrock-test-stream", "success")
                async for _ in bedrock_client.stream(msgs, opts):
                    pass
                assert _counter_value(LLM_OUTPUT_TOKENS, "bedrock", "bedrock-test-stream") == before_output + 22
                assert _counter_value(LLM_INPUT_TOKENS, "bedrock", "bedrock-test-stream", "prompt_total") == before_input + 55
                assert _counter_value(LLM_REQUESTS, "bedrock", "bedrock-test-stream", "success") == before_req + 1

        asyncio.get_event_loop().run_until_complete(run())


# ── Anthropic provider metrics ─────────────────────────────────────────────


class TestAnthropicMetrics:
    @pytest.fixture
    def anthropic_client(self):
        from agent.llm.anthropic import AnthropicClient
        return AnthropicClient(api_key="fake-key")

    def _make_fake_usage(self, input_tokens=100, output_tokens=40, cache_read=0, cache_write=0):
        u = MagicMock()
        u.input_tokens = input_tokens
        u.output_tokens = output_tokens
        u.cache_read_input_tokens = cache_read
        u.cache_creation_input_tokens = cache_write
        return u

    def test_generate_increments_tokens(self, anthropic_client) -> None:
        fake_usage = self._make_fake_usage(input_tokens=100, output_tokens=40)
        fake_response = MagicMock()
        fake_response.content = []
        fake_response.usage = fake_usage
        fake_response.stop_reason = "end_turn"

        async def run():
            with patch.object(
                anthropic_client._client.messages,
                "create",
                new_callable=AsyncMock,
                return_value=fake_response,
            ):
                msgs = _sample_messages()
                opts = _sample_options(model="anthropic-test-generate")
                before_output = _counter_value(LLM_OUTPUT_TOKENS, "anthropic", "anthropic-test-generate")
                before_input = _counter_value(LLM_INPUT_TOKENS, "anthropic", "anthropic-test-generate", "prompt_total")
                before_req = _counter_value(LLM_REQUESTS, "anthropic", "anthropic-test-generate", "success")
                await anthropic_client.generate(msgs, opts)
                assert _counter_value(LLM_OUTPUT_TOKENS, "anthropic", "anthropic-test-generate") == before_output + 40
                assert _counter_value(LLM_INPUT_TOKENS, "anthropic", "anthropic-test-generate", "prompt_total") == before_input + 100
                assert _counter_value(LLM_REQUESTS, "anthropic", "anthropic-test-generate", "success") == before_req + 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_generate_error_increments_error_counter(self, anthropic_client) -> None:
        import httpx
        from anthropic import APIStatusError

        from agent.llm.base import LLMError

        async def run():
            exc = APIStatusError(
                "rate limit",
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com/")),
                body={},
            )
            with patch.object(
                anthropic_client._client.messages,
                "create",
                new_callable=AsyncMock,
                side_effect=exc,
            ):
                msgs = _sample_messages()
                opts = _sample_options(model="anthropic-test-err")
                before = _counter_value(LLM_REQUESTS, "anthropic", "anthropic-test-err", "error")
                with pytest.raises(LLMError):
                    await anthropic_client.generate(msgs, opts)
                assert _counter_value(LLM_REQUESTS, "anthropic", "anthropic-test-err", "error") == before + 1

        asyncio.get_event_loop().run_until_complete(run())

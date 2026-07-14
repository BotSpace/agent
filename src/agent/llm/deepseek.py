"""DeepSeek provider — OpenAI-compatible API at api.deepseek.com.

DeepSeek exposes an OpenAI Chat Completions-compatible API (tool/function
calling + streaming supported), so we inherit OpenAIClient and only swap the
base_url + api_key. Models: `deepseek-v4-flash` / `deepseek-v4-pro` (the legacy
`deepseek-chat` / `deepseek-reasoner` names are deprecated 2026-07-24). DeepSeek
uses the standard `max_tokens` parameter (never `max_completion_tokens`).
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from .base import GenerateOptions, LLMError
from .openai import OpenAIClient, _wrap_for_langsmith

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(OpenAIClient):
    provider = "deepseek"

    def __init__(self, api_key: str | None, *, base_url: str | None = None) -> None:
        if not api_key:
            raise LLMError("DEEPSEEK_API_KEY is not set", provider=self.provider)
        self._client = _wrap_for_langsmith(
            AsyncOpenAI(api_key=api_key, base_url=base_url or DEEPSEEK_BASE_URL)
        )

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        # DeepSeek's OpenAI-compatible endpoint expects `max_tokens`; it does not
        # accept the `max_completion_tokens` alias OpenAI's own o-series/gpt-5 use.
        return False

    def _apply_reasoning(self, req: dict[str, Any], options: GenerateOptions) -> None:
        # DeepSeek V4 thinking mode. Contract (api-docs.deepseek.com/guides/thinking_mode):
        #   ON : reasoning_effort="high"|"max" + extra_body={"thinking":{"type":"enabled"}}
        #   OFF: extra_body={"thinking":{"type":"disabled"}} (reasoning_effort ignored)
        # None → send nothing, leaving DeepSeek's default (enabled, high).
        effort = options.reasoning_effort
        if effort is None:
            return None
        if effort == "off":
            req.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
            return None
        # high | max
        req["reasoning_effort"] = effort
        req.setdefault("extra_body", {})["thinking"] = {"type": "enabled"}
        return None

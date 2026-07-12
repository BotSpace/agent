"""OpenRouter provider — OpenAI-compatible API at openrouter.ai/api/v1.

OpenRouter agentlash Gemma 4, Llama, Mistral, Claude va h.k. modellarni
bitta API'dan ishlatish imkonini beradi. API OpenAI Chat Completions
sxemasi bilan to'liq mos — shuning uchun OpenAIClient'dan meros olamiz va
faqat base_url va api_key'ni o'zgartiramiz.

Tool-calling (function calling) ko'pchilik OpenRouter modellarida
qo'llab-quvvatlanadi (gemma-4, llama-4, qwen-3, claude, gpt-*).
"""

from __future__ import annotations

from openai import AsyncOpenAI

from .base import LLMError
from .openai import OpenAIClient, _wrap_for_langsmith

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAIClient):
    provider = "openrouter"

    def __init__(
        self,
        api_key: str | None,
        *,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set", provider=self.provider)
        default_headers: dict[str, str] = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name
        self._client = _wrap_for_langsmith(
            AsyncOpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers=default_headers or None,
            )
        )

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        """OpenRouter har provider o'z maximum sxemasini ishlatadi.

        OpenAI'ning gpt-4o/gpt-5/o-series'lari `max_completion_tokens` so'raydi,
        boshqalar (gemma, llama, claude-via-OR, ...) `max_tokens`. OpenRouter
        kerakli parametrni avtomatik translate qiladi, shu sababli biz har doim
        `max_tokens` bera olamiz — eski OpenAI logikasini override qilamiz.
        """
        return False

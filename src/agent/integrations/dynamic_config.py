"""Dynamic LLM config resolver.

Backend public-config API'dan provider/model/region oladi va env settings
ustidan ustunlik bilan qo'llaydi. Har run boshida bir marta chaqiriladi;
natija shu run ichida barqaror (bir xil).

Public config kalitlari (backend tomonidan belgilangan):
  ai_agent.provider  — e.g. "bedrock" | "openai" | "ollama" | "gemini" | "anthropic"
  ai_agent.model     — e.g. "qwen.qwen3-coder-480b-a35b-v1:0"
  ai_agent.region    — e.g. "us-west-2"

Internal config kalitlari (X-Internal-Token kerak, secrets bilan):
  ai_agent.api_key   — LLM provider API key (admindan boshqariladi)
  ai_agent.base_url  — LLM provider base URL override

Ustunlik tartibi (har kalit uchun):
  1. Backend internal-config qiymati — mavjud va bo'sh bo'lmasa
  2. Backend public-config qiymati (faqat base_url uchun) — mavjud va bo'sh bo'lmasa
  3. Env-based settings (settings.llm_provider, settings.model_*, settings.aws_region,
     settings.openai_api_key / openrouter_api_key / anthropic_api_key)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config import get_settings
from .backend import fetch_internal_config, fetch_public_config

if TYPE_CHECKING:
    from ..llm.router import ModelRoles

logger = logging.getLogger(__name__)

_PUBLIC_KEY_PROVIDER = "ai_agent.provider"
_PUBLIC_KEY_MODEL = "ai_agent.model"
_PUBLIC_KEY_REGION = "ai_agent.region"
_PUBLIC_KEY_BASE_URL = "ai_agent.base_url"

_INTERNAL_KEY_API_KEY = "ai_agent.api_key"
_INTERNAL_KEY_BASE_URL = "ai_agent.base_url"


def _nonempty(value: object) -> str | None:
    """Return the value as str if it is a non-empty string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


async def effective_provider() -> str:
    """Return the active LLM provider string, backend config over env."""
    cfg = await fetch_public_config()
    from_backend = _nonempty(cfg.get(_PUBLIC_KEY_PROVIDER))
    if from_backend:
        logger.info("LLM provider from backend config: %s", from_backend)
        return from_backend
    s = get_settings()
    return s.llm_provider


async def effective_region() -> str:
    """Return the active AWS region, backend config over env."""
    cfg = await fetch_public_config()
    from_backend = _nonempty(cfg.get(_PUBLIC_KEY_REGION))
    if from_backend:
        logger.info("LLM region from backend config: %s", from_backend)
        return from_backend
    s = get_settings()
    return s.aws_region


async def effective_model_roles() -> ModelRoles:
    """Build ModelRoles with backend config model applied to planner/executor/critic.

    When ``ai_agent.model`` is set in backend config it overrides all three
    inference roles (planner, executor, critic).  ``embedding`` always comes
    from env — it is not controlled remotely.
    """
    from ..llm.router import ModelRoles

    cfg = await fetch_public_config()
    s = get_settings()

    model_override = _nonempty(cfg.get(_PUBLIC_KEY_MODEL))
    if model_override:
        logger.info("LLM model from backend config: %s (all roles)", model_override)
        return ModelRoles(
            planner=model_override,
            executor=model_override,
            critic=model_override,
            embedding=s.model_embedding,
        )

    return ModelRoles.from_settings(s)


async def effective_api_key(provider: str | None = None) -> str | None:
    """Return the active LLM API key, internal-config over env.

    Reads ``ai_agent.api_key`` from the backend internal-config endpoint.
    Falls back to the env-based per-provider key when internal config is
    absent, empty, or unreachable.

    ``provider`` is used only for the env fallback selection.
    """
    internal_cfg = await fetch_internal_config()
    from_internal = _nonempty(internal_cfg.get(_INTERNAL_KEY_API_KEY))
    if from_internal:
        logger.info("LLM api_key from internal config (provider=%s)", provider)
        return from_internal

    s = get_settings()
    p = provider or s.llm_provider
    if p == "openai":
        return s.openai_api_key
    if p == "openrouter":
        return s.openrouter_api_key
    if p == "anthropic":
        return s.anthropic_api_key
    # bedrock does not use an api_key field
    return None


async def effective_base_url(provider: str | None = None) -> str | None:
    """Return the active LLM base URL override, internal-config > public-config > None.

    Reads ``ai_agent.base_url`` from internal config first, then from public
    config. Returns ``None`` when neither source provides a non-empty value so
    the provider client uses its own default base URL.
    """
    internal_cfg = await fetch_internal_config()
    from_internal = _nonempty(internal_cfg.get(_INTERNAL_KEY_BASE_URL))
    if from_internal:
        logger.info("LLM base_url from internal config (provider=%s): %s", provider, from_internal)
        return from_internal

    pub_cfg = await fetch_public_config()
    from_public = _nonempty(pub_cfg.get(_PUBLIC_KEY_BASE_URL))
    if from_public:
        logger.info("LLM base_url from public config (provider=%s): %s", provider, from_public)
        return from_public

    return None


async def build_effective_client():
    """Build an LLMClient with effective provider, region, api_key, and base_url.

    Precedence for each dimension:
      - provider / region: backend public-config > env
      - api_key: backend internal-config > env per-provider key
      - base_url: backend internal-config > backend public-config > provider default

    For bedrock: api_key/base_url overrides are not applicable — AWS credentials
    always come from env (access_key_id / secret_access_key / region).
    For openrouter: effective api_key is the OpenRouter key; base_url defaults to
    ``https://openrouter.ai/api/v1`` unless overridden; site_url/app_name come
    from env settings.
    """
    from ..llm.anthropic import AnthropicClient
    from ..llm.bedrock import BedrockClient
    from ..llm.openai import OpenAIClient
    from ..llm.openrouter import OPENROUTER_BASE_URL, OpenRouterClient
    from ..llm.router import build_client

    provider = await effective_provider()
    s = get_settings()

    if provider == "bedrock":
        region = await effective_region()
        logger.info("Building BedrockClient with effective region=%s", region)
        return BedrockClient(
            region=region,
            access_key_id=s.aws_access_key_id,
            secret_access_key=s.aws_secret_access_key,
        )

    api_key = await effective_api_key(provider)
    base_url = await effective_base_url(provider)

    if provider == "openrouter":
        effective_key = api_key or s.openrouter_api_key
        effective_burl = base_url or OPENROUTER_BASE_URL
        logger.info(
            "Building OpenRouterClient with effective key=%s base_url=%s",
            "***" if effective_key else None,
            effective_burl,
        )
        from openai import AsyncOpenAI

        from ..llm.base import LLMError
        from ..llm.openai import _wrap_for_langsmith

        if not effective_key:
            raise LLMError("OPENROUTER_API_KEY is not set", provider="openrouter")
        default_headers: dict[str, str] = {}
        if s.openrouter_site_url:
            default_headers["HTTP-Referer"] = s.openrouter_site_url
        if s.openrouter_app_name:
            default_headers["X-Title"] = s.openrouter_app_name
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._client = _wrap_for_langsmith(
            AsyncOpenAI(
                api_key=effective_key,
                base_url=effective_burl,
                default_headers=default_headers or None,
            )
        )
        return client

    if provider == "openai":
        effective_key = api_key or s.openai_api_key
        if base_url:
            logger.info("Building OpenAIClient with base_url override: %s", base_url)
            from openai import AsyncOpenAI

            from ..llm.base import LLMError
            from ..llm.openai import _wrap_for_langsmith

            if not effective_key:
                raise LLMError("OPENAI_API_KEY is not set", provider="openai")
            oa_client = OpenAIClient.__new__(OpenAIClient)
            oa_client._client = _wrap_for_langsmith(
                AsyncOpenAI(api_key=effective_key, base_url=base_url)
            )
            return oa_client
        if api_key and api_key != s.openai_api_key:
            logger.info("Building OpenAIClient with effective api_key from internal config")
            return OpenAIClient(api_key=api_key)
        return build_client(provider, s)

    if provider == "anthropic":
        effective_key = api_key or s.anthropic_api_key
        if effective_key and effective_key != s.anthropic_api_key:
            logger.info("Building AnthropicClient with effective api_key from internal config")
            return AnthropicClient(api_key=effective_key)
        return build_client(provider, s)

    return build_client(provider, s)

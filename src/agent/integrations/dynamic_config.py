"""Dynamic LLM config resolver.

Backend public/internal config API'lari yagona LLM konfiguratsiya manbai.
Har run boshida bir marta chaqiriladi;
natija shu run ichida barqaror (bir xil).

Public config kalitlari (backend tomonidan belgilangan):
  ai_agent.provider  — e.g. "bedrock" | "openai" | "ollama" | "gemini" | "anthropic"
  ai_agent.model     — e.g. "qwen.qwen3-coder-480b-a35b-v1:0"
  ai_agent.region    — e.g. "us-west-2"

Internal config kalitlari (X-Internal-Token kerak, secrets bilan):
  ai_agent.api_key               — LLM provider API key (admindan boshqariladi)
  ai_agent.base_url              — LLM provider base URL override
  ai_agent.aws_access_key_id     — Bedrock AWS access key id
  ai_agent.aws_secret_access_key — Bedrock AWS secret access key

Provider/model/region public endpointdan, secretlar internal endpointdan olinadi.
Providerning standart API URL'i protokol standarti sifatida kodda qoladi.
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
_INTERNAL_KEY_AWS_ACCESS_KEY_ID = "ai_agent.aws_access_key_id"
_INTERNAL_KEY_AWS_SECRET_ACCESS_KEY = "ai_agent.aws_secret_access_key"

# Provider-default base URLs — used as the final fallback when neither internal
# nor public config supplies a base_url. bedrock has no default (region-based).
_PROVIDER_DEFAULT_BASE_URL: dict[str, str | None] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "https://ollama.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com",
    "bedrock": None,
}


def _nonempty(value: object) -> str | None:
    """Return the value as str if it is a non-empty string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


async def effective_provider() -> str:
    """Return the backend-owned active LLM provider."""
    cfg = await fetch_public_config()
    provider = _nonempty(cfg.get(_PUBLIC_KEY_PROVIDER))
    if not provider:
        raise RuntimeError(f"backend config {_PUBLIC_KEY_PROVIDER} is required")
    logger.info("LLM provider from backend config: %s", provider)
    return provider


async def effective_region() -> str:
    """Return the backend-owned AWS region."""
    cfg = await fetch_public_config()
    region = _nonempty(cfg.get(_PUBLIC_KEY_REGION))
    if not region:
        raise RuntimeError(f"backend config {_PUBLIC_KEY_REGION} is required for Bedrock")
    logger.info("LLM region from backend config: %s", region)
    return region


async def effective_model_roles() -> ModelRoles:
    """Build ModelRoles with backend config model applied to planner/executor/critic.

    When ``ai_agent.model`` is set in backend config it overrides all three
    inference roles (planner, executor, critic).  ``embedding`` always comes
    from env — it is not controlled remotely.
    """
    from ..llm.router import ModelRoles

    cfg = await fetch_public_config()
    model_override = _nonempty(cfg.get(_PUBLIC_KEY_MODEL))
    if not model_override:
        raise RuntimeError(f"backend config {_PUBLIC_KEY_MODEL} is required")
    s = get_settings()
    logger.info("LLM model from backend config: %s (all roles)", model_override)
    return ModelRoles(
        planner=model_override,
        executor=model_override,
        critic=model_override,
        embedding=s.model_embedding,
    )


async def effective_api_key(provider: str | None = None) -> str | None:
    """Return the backend-owned LLM API key (Bedrock may legitimately omit it)."""
    internal_cfg = await fetch_internal_config()
    api_key = _nonempty(internal_cfg.get(_INTERNAL_KEY_API_KEY))
    if api_key:
        logger.info("LLM api_key from internal config (provider=%s)", provider)
    return api_key


async def effective_aws_access_key_id() -> str | None:
    """Return the active AWS access key id, internal-config over env.

    An empty value is valid when the pod uses the AWS workload identity chain.
    """
    internal_cfg = await fetch_internal_config()
    from_internal = _nonempty(internal_cfg.get(_INTERNAL_KEY_AWS_ACCESS_KEY_ID))
    if from_internal:
        logger.info("AWS access_key_id from internal config")
        return from_internal

    return None


async def effective_aws_secret_access_key() -> str | None:
    """Return the active AWS secret access key, internal-config over env.

    An empty value is valid when the pod uses the AWS workload identity chain.
    """
    internal_cfg = await fetch_internal_config()
    from_internal = _nonempty(internal_cfg.get(_INTERNAL_KEY_AWS_SECRET_ACCESS_KEY))
    if from_internal:
        logger.info("AWS secret_access_key from internal config")
        return from_internal

    return None


async def effective_base_url(provider: str | None = None) -> str | None:
    """Return backend override or the provider protocol's canonical endpoint."""
    internal_cfg = await fetch_internal_config()
    from_internal = _nonempty(internal_cfg.get(_INTERNAL_KEY_BASE_URL))
    if from_internal:
        logger.info("LLM base_url from internal config (provider=%s): %s", provider, from_internal)
        return from_internal

    default = _PROVIDER_DEFAULT_BASE_URL.get(provider) if provider else None
    if default:
        logger.info("LLM base_url from provider default (provider=%s): %s", provider, default)
    return default


async def build_effective_client():
    """Build an LLMClient with effective provider, region, api_key, and base_url.

    Provider/model/region come from backend public config; secrets and optional
    base URL override come from backend internal config.
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
        access_key_id = await effective_aws_access_key_id()
        secret_access_key = await effective_aws_secret_access_key()
        logger.info("Building BedrockClient with effective region=%s", region)
        return BedrockClient(
            region=region,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

    api_key = await effective_api_key(provider)
    base_url = await effective_base_url(provider)

    if provider == "openrouter":
        effective_key = api_key
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

    if provider == "deepseek":
        from ..llm.deepseek import DEEPSEEK_BASE_URL, DeepSeekClient

        effective_key = api_key
        effective_burl = base_url or DEEPSEEK_BASE_URL
        logger.info(
            "Building DeepSeekClient with effective key=%s base_url=%s",
            "***" if effective_key else None,
            effective_burl,
        )
        return DeepSeekClient(api_key=effective_key, base_url=effective_burl)

    if provider == "openai":
        effective_key = api_key
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
        return OpenAIClient(api_key=effective_key)

    if provider == "anthropic":
        return AnthropicClient(api_key=api_key)

    return build_client(provider, s)

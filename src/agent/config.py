from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai", "bedrock", "openrouter"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── server ──────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="AGENT_HOST")
    port: int = Field(default=8100, alias="AGENT_PORT")
    log_level: str = Field(default="INFO", alias="AGENT_LOG_LEVEL")
    cors_origins: str = Field(
        default="http://localhost:3010,http://localhost:3001",
        alias="AGENT_CORS_ORIGINS",
    )

    # ── mcp (Model Context Protocol) ────────────────────────────────────
    mcp_enabled: bool = Field(default=True, alias="AGENT_MCP_ENABLED")
    mcp_port: int = Field(default=8200, alias="AGENT_MCP_PORT")

    # ── llm ─────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(default="openai", alias="AGENT_LLM_PROVIDER")
    model_planner: str = Field(default="gpt-5.2", alias="AGENT_MODEL_PLANNER")
    model_executor: str = Field(default="gpt-5.2", alias="AGENT_MODEL_EXECUTOR")
    model_critic: str = Field(default="gpt-5.2", alias="AGENT_MODEL_CRITIC")
    model_embedding: str = Field(
        default="text-embedding-3-small", alias="AGENT_MODEL_EMBEDDING"
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_site_url: str | None = Field(
        default="https://botspace.uz", alias="OPENROUTER_SITE_URL",
        description="HTTP-Referer header for OpenRouter rankings/attribution.",
    )
    openrouter_app_name: str | None = Field(
        default="Botmother Agent", alias="OPENROUTER_APP_NAME",
        description="X-Title header for OpenRouter dashboard display.",
    )
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # ── budgets ─────────────────────────────────────────────────────────
    max_input_tokens: int = Field(default=200_000, alias="AGENT_MAX_INPUT_TOKENS")
    max_output_tokens: int = Field(default=16_000, alias="AGENT_MAX_OUTPUT_TOKENS")
    max_tool_calls: int = Field(default=80, alias="AGENT_MAX_TOOL_CALLS")
    max_wall_seconds: int = Field(default=900, alias="AGENT_MAX_WALL_SECONDS")

    # ── persistence ─────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+psycopg://agent:agent@localhost:5433/agent",
        alias="AGENT_DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6380/0", alias="AGENT_REDIS_URL")

    # ── external services ───────────────────────────────────────────────
    backend_base_url: str = Field(
        default="http://localhost:8000", alias="BACKEND_BASE_URL"
    )
    backend_internal_token: str | None = Field(
        default=None, alias="BACKEND_INTERNAL_TOKEN"
    )
    mongo_base_url: str = Field(
        default="mongodb://localhost:27017", alias="MONGO_BASE_URL"
    )
    mongo_db_prefix: str = Field(default="project_", alias="MONGO_DB_PREFIX")
    engine_base_url: str = Field(
        default="http://localhost:9100", alias="ENGINE_BASE_URL"
    )
    # Engine-side flow simulator service — `flow.simulate` tool POSTs the
    # working flow + a list of user steps and gets back per-step bot messages.
    simulator_url: str = Field(
        default="http://engine-simulator:8200", alias="AGENT_SIMULATOR_URL"
    )

    # ── auth ────────────────────────────────────────────────────────────
    session_secret: str = Field(
        default="change-me-to-a-32-char-secret-string", alias="SESSION_SECRET"
    )
    jwt_algorithm: str = Field(default="RS256", alias="JWT_ALGORITHM")
    jwt_public_key_path: str = Field(
        default="keys/public.pem", alias="JWT_PUBLIC_KEY_PATH"
    )
    internal_api_token: str | None = Field(
        default=None, alias="AGENT_INTERNAL_API_TOKEN"
    )
    # MCP OAuth (resource server) — discovery metadata uchun tashqi URL'lar.
    public_base_url: str = Field(
        default="https://botspace.uz", alias="AGENT_PUBLIC_BASE_URL",
        description="MCP resource'ning canonical bazasi — resource = {base}/mcp.",
    )
    oauth_issuer: str = Field(
        default="https://api.botspace.uz", alias="AGENT_OAUTH_ISSUER",
        description="OAuth authorization server (backend) issuer URL'i.",
    )

    # ── memory / rag ────────────────────────────────────────────────────
    rag_enabled: bool = Field(default=True, alias="AGENT_RAG_ENABLED")
    rag_top_k: int = Field(default=6, alias="AGENT_RAG_TOP_K")
    # Template RAG re-ingestion interval (hours). 0 disables the
    # background scheduler entirely (manual `python -m
    # agent.scripts.ingest_templates` only).
    template_sync_interval_hours: float = Field(
        default=6.0, alias="AGENT_TEMPLATE_SYNC_INTERVAL_HOURS"
    )
    # Delay before the first sync after pod startup (seconds). Keeps
    # the lifespan handler returning fast — first sync runs in
    # background after the API is already serving requests.
    template_sync_initial_delay_seconds: int = Field(
        default=60, alias="AGENT_TEMPLATE_SYNC_INITIAL_DELAY_SECONDS"
    )
    # Module-catalog RAG re-ingestion interval (hours). Mirrors the
    # template scheduler: 0 disables the background loop entirely
    # (manual `python -m agent.scripts.ingest_modules` only). The module
    # catalog changes when the platform lists/de-lists a module, so a
    # slightly longer default than templates is fine.
    module_sync_interval_hours: float = Field(
        default=12.0, alias="AGENT_MODULE_SYNC_INTERVAL_HOURS"
    )
    # Delay before the first module sync after pod startup (seconds).
    # Offset from the template sync's default so the two ingest loops
    # don't hammer the backend / embedder at the same instant on boot.
    module_sync_initial_delay_seconds: int = Field(
        default=90, alias="AGENT_MODULE_SYNC_INITIAL_DELAY_SECONDS"
    )

    # ── observability ───────────────────────────────────────────────────
    otel_endpoint: str | None = Field(default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="botmother-agent", alias="LANGSMITH_PROJECT"
    )

    # ── derived ─────────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def normalize_pg_url(url: str) -> str:
    """Strip the SQLAlchemy psycopg driver suffix so psycopg/langgraph accept the DSN."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


# Known insecure placeholder secrets that MUST NOT reach production. The default
# value in Settings.session_secret is here so a fresh .env boots loudly instead
# of running with a guessable signing secret.
_PLACEHOLDER_SECRETS = {
    "",
    "change-me-to-a-32-char-secret-string",
    "changeme",
    "secret",
}


def _resolve_public_key_path(settings: Settings) -> Path:
    p = Path(settings.jwt_public_key_path)
    if not p.is_absolute():
        # config.py = src/agent/config.py → parents[2] is the repo root, where
        # keys/ lives (mirrors deps._load_public_key / mcp.auth._public_key).
        p = Path(__file__).resolve().parents[2] / settings.jwt_public_key_path
    return p


def validate_security_config(settings: Settings) -> None:
    """Fail-fast on insecure auth configuration at startup.

    RS256 is the single supported production scheme: the backend signs tokens
    with its private key and the agent verifies them with the public key. Boot
    must abort (rather than silently accept an insecure default) when:

      * ``JWT_ALGORITHM`` is not ``RS256`` (no symmetric-secret scheme in prod);
      * the RS256 public key cannot be located (``JWT_PUBLIC_KEY`` env var or
        the ``JWT_PUBLIC_KEY_PATH`` file);
      * ``SESSION_SECRET`` is still a known placeholder value.

    Called from ``create_app`` (real startup) — not from the pydantic model —
    so unit tests that construct ``Settings`` with explicit overrides are
    unaffected.
    """
    errors: list[str] = []

    if settings.jwt_algorithm != "RS256":
        errors.append(
            f"JWT_ALGORITHM must be 'RS256' (got {settings.jwt_algorithm!r}); "
            "the agent verifies backend-signed tokens with an RS256 public key."
        )

    if not os.getenv("JWT_PUBLIC_KEY"):
        key_path = _resolve_public_key_path(settings)
        if not key_path.exists():
            errors.append(
                f"RS256 public key not found: set JWT_PUBLIC_KEY env or place the "
                f"PEM at {key_path} (JWT_PUBLIC_KEY_PATH={settings.jwt_public_key_path})."
            )

    if settings.session_secret.strip() in _PLACEHOLDER_SECRETS:
        errors.append(
            "SESSION_SECRET is a placeholder — set a real, unique secret "
            "(shared with the dashboard) before starting the service."
        )

    if errors:
        raise RuntimeError(
            "Insecure auth configuration — refusing to start:\n  - "
            + "\n  - ".join(errors)
        )

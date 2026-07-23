"""Faza 2.4 — RS256 fail-fast: refuse to boot with an insecure JWT config."""

import pytest

from agent.config import Settings, validate_security_config


def _settings(**over) -> Settings:
    base = dict(
        JWT_ALGORITHM="RS256",
        SESSION_SECRET="a-real-unique-secret-value-1234567890",
        BACKEND_INTERNAL_TOKEN="test-internal-token",
    )
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def test_rejects_non_rs256(monkeypatch):
    monkeypatch.setenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nx\n-----END PUBLIC KEY-----")
    with pytest.raises(RuntimeError, match="RS256"):
        validate_security_config(_settings(JWT_ALGORITHM="HS256"))


def test_rejects_placeholder_secret(monkeypatch):
    monkeypatch.setenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nx\n-----END PUBLIC KEY-----")
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        validate_security_config(_settings(SESSION_SECRET="change-me-to-a-32-char-secret-string"))


def test_rejects_missing_public_key(monkeypatch):
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)
    with pytest.raises(RuntimeError, match="public key"):
        validate_security_config(_settings(JWT_PUBLIC_KEY_PATH="keys/definitely-missing.pem"))


def test_accepts_valid_config(monkeypatch):
    monkeypatch.setenv("JWT_PUBLIC_KEY", "-----BEGIN PUBLIC KEY-----\nx\n-----END PUBLIC KEY-----")
    # Should not raise.
    validate_security_config(_settings())

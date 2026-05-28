"""FastAPI dependencies — auth, store, services."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import Settings, get_settings
from ..runtime.store import RunStore, get_run_store

logger = logging.getLogger(__name__)

bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _load_public_key(path: str) -> str:
    import os
    env_key = os.getenv("JWT_PUBLIC_KEY")
    if env_key:
        return env_key
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / path
    if p.exists():
        return p.read_text()
    raise FileNotFoundError(f"Public key not found at {p} and JWT_PUBLIC_KEY env var not set")


@dataclass
class CurrentUser:
    user_id: str
    claims: dict


async def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    internal = request.headers.get("X-Internal-Token")
    if internal and settings.internal_api_token and internal == settings.internal_api_token:
        return CurrentUser(user_id="__internal__", claims={"role": "internal"})

    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        if settings.jwt_algorithm == "RS256":
            public_key = _load_public_key(settings.jwt_public_key_path)
            claims = jwt.decode(
                creds.credentials,
                public_key,
                algorithms=["RS256"],
                options={"verify_aud": False},
            )
        else:
            claims = jwt.decode(
                creds.credentials,
                settings.session_secret,
                algorithms=[settings.jwt_algorithm],
                options={"verify_aud": False},
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    user_block = claims.get("user") or {}
    user_id = str(
        user_block.get("id") or claims.get("sub") or claims.get("user_id") or ""
    )
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user id")

    access_token = claims.get("access") or creds.credentials
    # Refresh token JWT ichida bo'lishi shart emas (xavfsizroq) — frontend
    # uni X-Refresh-Token header'i orqali yuboradi. Header bo'lmasa, JWT
    # ichidagi `refresh` field'iga qaytamiz (eski mijozlar uchun).
    refresh_token = (
        request.headers.get("X-Refresh-Token")
        or request.headers.get("x-refresh-token")
        or claims.get("refresh")
    )
    return CurrentUser(
        user_id=user_id,
        claims={**claims, "access_token": access_token, "refresh_token": refresh_token},
    )


def get_store(_settings: Settings = Depends(get_settings)) -> RunStore:
    return get_run_store()

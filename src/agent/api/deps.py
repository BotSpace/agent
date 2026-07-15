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

    def _decode(token: str, *, verify_exp: bool = True) -> dict:
        """Decode a backend JWT (access or refresh) with the configured alg.
        Signature is always verified; only `exp` can be relaxed."""
        opts = {"verify_aud": False, "verify_exp": verify_exp}
        if settings.jwt_algorithm == "RS256":
            return jwt.decode(
                token,
                _load_public_key(settings.jwt_public_key_path),
                algorithms=["RS256"],
                options=opts,
            )
        return jwt.decode(
            token,
            settings.session_secret,
            algorithms=[settings.jwt_algorithm],
            options=opts,
        )

    refresh_header = (
        request.headers.get("X-Refresh-Token")
        or request.headers.get("x-refresh-token")
    )
    try:
        claims = _decode(creds.credentials)
    except jwt.ExpiredSignatureError:
        # Access token expired. A long-open dashboard tab still holds a valid
        # refresh token — accept the run if that refresh is itself signature- and
        # exp-valid (the session is alive). The access JWT is decoded with `exp`
        # relaxed (signature still verified, so it's authentic), and downstream
        # BackendClient refreshes the access on its first 401. Without a valid
        # refresh, the session really is over → 401.
        if not refresh_header:
            raise HTTPException(status_code=401, detail="Token expired") from None
        try:
            _decode(refresh_header)  # validates signature + exp
            claims = _decode(creds.credentials, verify_exp=False)
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Token expired") from None
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
    refresh_token = refresh_header or claims.get("refresh")
    return CurrentUser(
        user_id=user_id,
        claims={
            **claims,
            "user_id": user_id,  # explicit copy for BackendClient (_auth_kw)
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    )


async def require_internal(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Gate for server-to-server admin endpoints: requires the shared internal
    token (``X-Internal-Token``). Unlike ``get_current_user`` it does NOT accept
    a user JWT, so a logged-in end user can never reach admin-only reads — only
    a trusted backend/admin service holding the token can."""
    internal = request.headers.get("X-Internal-Token")
    if not (internal and settings.internal_api_token and internal == settings.internal_api_token):
        raise HTTPException(status_code=403, detail="internal token required")


def get_store(_settings: Settings = Depends(get_settings)) -> RunStore:
    return get_run_store()

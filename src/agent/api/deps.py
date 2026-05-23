"""FastAPI dependencies — auth, store, services."""

from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import Settings, get_settings
from ..runtime.store import RunStore, get_run_store


bearer = HTTPBearer(auto_error=False)


@dataclass
class CurrentUser:
    user_id: str
    claims: dict


async def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    # 1) Internal token for service-to-service (Django/backend → agent)
    internal = request.headers.get("X-Internal-Token")
    if internal and settings.internal_api_token and internal == settings.internal_api_token:
        return CurrentUser(user_id="__internal__", claims={"role": "internal"})

    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        claims = jwt.decode(
            creds.credentials,
            settings.session_secret,
            algorithms=[settings.jwt_algorithm],
            options={"verify_aud": False},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    user_block = claims.get("user") or {}
    user_id = str(user_block.get("id") or claims.get("sub") or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user id")
    # Attach access + refresh so backend tools can forward and auto-refresh on 401.
    access_token = claims.get("access") or creds.credentials
    refresh_token = claims.get("refresh")
    return CurrentUser(
        user_id=user_id,
        claims={**claims, "access_token": access_token, "refresh_token": refresh_token},
    )


def get_store(_settings: Settings = Depends(get_settings)) -> RunStore:
    return get_run_store()

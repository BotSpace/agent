"""MCP auth — dual rejim: `bsp_` personal access token va oddiy JWT.

Ikkala holatda ham natija `AuthedUser`: user_id + claims. claims ichida
`access_token`, `refresh_token`, `user_id` bo'ladi — bu to'g'ridan-to'g'ri
tool'larga `auth_claims` sifatida beriladi (BackendClient `_auth_kw` shu
shaklni kutadi).

`bsp_` token:
    Backend `POST {backend_base_url}/auth/token/exchange/` body `{"token": "bsp_..."}`
    → `{"status": true, "data": {"access": "...", "refresh": "..."}}` (envelope!).
    Token sha256 hash bo'yicha in-memory cache; access `exp`dan ~60s oldin
    qayta exchange.

JWT token:
    `deps.py`dagi kabi verify (RS256 public key, yoki HS settings.jwt_algorithm).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import jwt

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

# Access `exp`dan shuncha soniya OLDIN tokenni qayta exchange qilamiz — soat
# farqi / network latency tufayli muddati o'tgan token ishlatilib qolmasligi uchun.
_EXCHANGE_REFRESH_LEEWAY = 60.0


class MCPAuthError(Exception):
    """Auth muvaffaqiyatsiz — MCP so'rovi 401 bilan rad etiladi."""


@dataclass
class AuthedUser:
    """`deps.CurrentUser`'ga mos struktura — user_id + tool'larga uzatiladigan claims.

    `client_key` — bitta user'ning ALOHIDA MCP klient ulanishini ajratish uchun.
    Bu yerda token-hash (Authorization sha256 prefiksi) — FALLBACK qiymat.
    Asosiy kimlik server.py `_session_client_key()`da: tool handler ichida SDK
    `request_ctx.get().session` obyektidan `sid-...` kalit olinadi (stateful
    transportda har ulanish = bitta sessiya obyekti). Shu sabab bir xil token
    (ikki terminal / sub-agent, bir Claude Code config) bo'lsa ham har ulanish
    ALOHIDA kalit oladi — aktiv loyihasi to'qnashmaydi. Token-hash faqat sessiya
    konteksti bo'lmagan chaqiruvlar uchun ishlatiladi.
    Xavfsizlik: client_key faqat hash/session prefiksi, xom token EMAS —
    log'ga yozish xavfsiz, lekin baribir butun token'ni hech qachon log'lamaymiz.
    """

    user_id: str
    claims: dict
    client_key: str = ""


# ── bsp_ exchange cache ──────────────────────────────────────────────────────


@dataclass
class _ExchangedToken:
    user_id: str
    access: str
    refresh: str | None
    exp: float  # access token `exp` (epoch seconds); 0 = noma'lum


# token_sha256 → exchanged
_exchange_cache: dict[str, _ExchangedToken] = {}


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# Per-client identifikator uzunligi (hex belgilar). 16 hex = 64 bit — bitta
# user'ning bir nechta klientini ajratish uchun to'qnashuvga deyarli chidamli.
_CLIENT_KEY_LEN = 16


def _client_key(authorization: str) -> str:
    """Authorization header'idan barqaror per-client kalit (sha256 prefiksi).

    Xom token'ni hech qachon qaytarmaymiz — faqat hash prefiksi. Log'ga yozilsa
    ham token oshkor bo'lmaydi.
    """
    return hashlib.sha256(authorization.encode("utf-8")).hexdigest()[:_CLIENT_KEY_LEN]


def _invalidate_exchange_cache() -> None:
    """Test'lar uchun — in-process cache'ni tozalaydi."""
    _exchange_cache.clear()


def _public_key(settings: Settings) -> str:
    """RS256 public key — JWT_PUBLIC_KEY env yoki keys/public.pem.

    `deps._load_public_key` bilan bir xil mantiq, lekin u path argumentini
    `lru_cache`laydi — bu yerda Settings orqali toza chaqiramiz.
    """
    import os

    env_key = os.getenv("JWT_PUBLIC_KEY")
    if env_key:
        return env_key
    p = Path(settings.jwt_public_key_path)
    if not p.is_absolute():
        # config.py joylashuvidan repo root: src/agent/config.py → parents[2] = src,
        # parents[3] = repo root. Bu yerda __file__ src/agent/mcp/auth.py.
        p = Path(__file__).resolve().parents[3] / settings.jwt_public_key_path
    if p.exists():
        return p.read_text()
    raise MCPAuthError("JWT public key topilmadi (JWT_PUBLIC_KEY env yoki keys/public.pem)")


def _claims_to_user(claims: dict, *, fallback_access: str) -> AuthedUser:
    """JWT claims'dan AuthedUser yasaydi — `deps.get_current_user` bilan bir xil."""
    user_block = claims.get("user") or {}
    user_id = str(
        user_block.get("id") or claims.get("sub") or claims.get("user_id") or ""
    )
    if not user_id:
        raise MCPAuthError("Token'da user id yo'q")
    access_token = claims.get("access") or fallback_access
    refresh_token = claims.get("refresh")
    return AuthedUser(
        user_id=user_id,
        claims={
            **claims,
            "user_id": user_id,
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
    )


def _verify_jwt(token: str, settings: Settings) -> AuthedUser:
    try:
        if settings.jwt_algorithm == "RS256":
            claims = jwt.decode(
                token,
                _public_key(settings),
                algorithms=["RS256"],
                options={"verify_aud": False},
            )
        else:
            claims = jwt.decode(
                token,
                settings.session_secret,
                algorithms=[settings.jwt_algorithm],
                options={"verify_aud": False},
            )
    except jwt.ExpiredSignatureError as exc:
        raise MCPAuthError("Token muddati o'tgan") from exc
    except jwt.PyJWTError as exc:
        raise MCPAuthError(f"Yaroqsiz token: {exc}") from exc
    return _claims_to_user(claims, fallback_access=token)


def _decode_exp(access: str) -> float:
    """Access JWT'dan `exp` ni imzo tekshirmasdan o'qiydi (faqat cache TTL uchun)."""
    try:
        payload = jwt.decode(access, options={"verify_signature": False})
        exp = payload.get("exp")
        return float(exp) if exp is not None else 0.0
    except Exception:
        return 0.0


async def _exchange_bsp(token: str, settings: Settings) -> _ExchangedToken:
    """`bsp_` personal token → JWT access+refresh, cache bilan."""
    key = _hash_token(token)
    cached = _exchange_cache.get(key)
    now = time.time()
    if cached and (cached.exp == 0.0 or cached.exp - _EXCHANGE_REFRESH_LEEWAY > now):
        return cached

    url = settings.backend_base_url.rstrip("/") + "/auth/token/exchange"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={"token": token},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
    except Exception as exc:
        raise MCPAuthError(f"Token exchange xatosi (backend ulanmadi): {exc}") from exc

    if resp.status_code >= 400:
        # Token logga chiqmasligi uchun faqat status'ni ko'rsatamiz.
        logger.warning("bsp_ token exchange rad etildi: HTTP %s", resp.status_code)
        raise MCPAuthError(f"Personal access token yaroqsiz (HTTP {resp.status_code})")

    body = resp.json() or {}
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    access = (data or {}).get("access")
    refresh = (data or {}).get("refresh")
    if not isinstance(access, str) or not access:
        raise MCPAuthError("Token exchange javobida `access` yo'q")

    # user_id ni access JWT'dan olamiz (claims ichida user.id/sub/user_id).
    try:
        payload = jwt.decode(access, options={"verify_signature": False})
    except Exception as exc:
        raise MCPAuthError(f"Exchange qilingan access tokenni o'qib bo'lmadi: {exc}") from exc
    user_block = payload.get("user") or {}
    user_id = str(
        user_block.get("id") or payload.get("sub") or payload.get("user_id") or ""
    )
    if not user_id:
        raise MCPAuthError("Exchange qilingan token'da user id yo'q")

    exchanged = _ExchangedToken(
        user_id=user_id,
        access=access,
        refresh=refresh if isinstance(refresh, str) and refresh else None,
        exp=_decode_exp(access),
    )
    _exchange_cache[key] = exchanged
    return exchanged


async def authenticate(
    authorization: str | None, *, settings: Settings | None = None
) -> AuthedUser:
    """`Authorization: Bearer <token>` header'dan AuthedUser yasaydi.

    `bsp_` prefiks → backend exchange; aks holda JWT verify.
    Auth muvaffaqiyatsiz bo'lsa `MCPAuthError` ko'taradi (caller 401 qaytaradi).
    """
    settings = settings or get_settings()
    if not authorization:
        raise MCPAuthError("Authorization header yo'q")
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise MCPAuthError("Authorization `Bearer <token>` formatida bo'lishi kerak")
    token = parts[1].strip()

    # Per-client kalit — xom token (parts[1]) bo'yicha, header prefiksisiz, shunda
    # "Bearer "/"bearer " holatlarida ham bir xil bo'ladi. Aynan shu token bir
    # klientni ifodalaydi (DCR'da har klient alohida access token oladi).
    client_key = _client_key(token)

    if token.startswith("bsp_"):
        exchanged = await _exchange_bsp(token, settings)
        return AuthedUser(
            user_id=exchanged.user_id,
            claims={
                "user_id": exchanged.user_id,
                "access_token": exchanged.access,
                "refresh_token": exchanged.refresh,
            },
            client_key=client_key,
        )
    user = _verify_jwt(token, settings)
    user.client_key = client_key
    return user

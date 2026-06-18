"""Django backend REST client — agent shu orqali project/release/env/file boshqaradi.

Backend endpoint pattern (dashboard `services.ts`'dan olingan):
- /api/project/                   GET, POST  /  /:id/  GET, PATCH, DELETE
- /api/relase/                    GET (?project=), POST  (release deploy)
- /api/enviroment/                GET (?project=), POST  /  /:id/  DELETE
- /api/file/                      GET (?project=)  /  /:id/  DELETE
- /api/collection/                GET (?project=)  /  /:id/  GET (metadata; data is in MongoDB)
- /api/subscription/usage/        GET (current user token usage)
- /api/chat/users/:project        GET
- /api/chat/messages/:project/:chat  GET
- /api/message/                   GET, POST, DELETE (broadcast)

Auth: agent o'z internal token bilan boradi (`X-Internal-Token` header), yoki
foydalanuvchi access token'i forward qilinadi (`Authorization: Bearer ...`).

Public config:
- /api/v1/shared/config/public/   GET (AllowAny) — flat JSON with ai_agent.* keys

Internal config:
- /api/v1/shared/config/internal/ GET (X-Internal-Token) — flat JSON with ALL keys
  incl. secrets (e.g. ai_agent.api_key, ai_agent.base_url).
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings

logger = logging.getLogger(__name__)

# ── Public config fetcher (TTL cache, no auth) ───────────────────────────────

_PUBLIC_CONFIG_TTL = 60.0  # seconds
_public_config_cache: dict[str, Any] = {}
_public_config_fetched_at: float = 0.0


async def fetch_public_config() -> dict[str, Any]:
    """GET {BACKEND_BASE_URL}/api/v1/shared/config/public/ and cache for TTL.

    Returns flat dict ``{"ai_agent.provider": "...", ...}``.
    NEVER raises — on any network/parse error returns ``{}`` so caller can
    fall back to env-based settings.
    """
    global _public_config_cache, _public_config_fetched_at

    now = time.monotonic()
    if _public_config_cache and (now - _public_config_fetched_at) < _PUBLIC_CONFIG_TTL:
        return _public_config_cache

    s = get_settings()
    url = s.backend_base_url.rstrip("/") + "/api/v1/shared/config/public/"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            # BaseViewSetMixin wraps payloads as {"status": true, "data": {...}};
            # peel the envelope so we get the flat {key: value} config dict.
            data = _unwrap_envelopes(resp.json())
        if not isinstance(data, dict):
            logger.warning("fetch_public_config: unexpected response type %s", type(data))
            data = {}
        _public_config_cache = data
        _public_config_fetched_at = now
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_public_config failed (falling back to env): %s", exc)
        return {}


def _invalidate_public_config_cache() -> None:
    """Reset the in-process cache — useful in tests."""
    global _public_config_cache, _public_config_fetched_at
    _public_config_cache = {}
    _public_config_fetched_at = 0.0


# ── Internal config fetcher (TTL cache, X-Internal-Token auth) ───────────────

_INTERNAL_CONFIG_TTL = 60.0  # seconds
_internal_config_cache: dict[str, Any] = {}
_internal_config_fetched_at: float = 0.0


async def fetch_internal_config() -> dict[str, Any]:
    """GET {BACKEND_BASE_URL}/api/v1/shared/config/internal/ and cache for TTL.

    Uses ``X-Internal-Token`` header (same token as ``report_token_usage``).
    Returns flat dict with ALL config keys including secrets such as
    ``ai_agent.api_key`` and ``ai_agent.base_url``.

    NEVER raises — returns ``{}`` when:
      - ``BACKEND_INTERNAL_TOKEN`` is not configured in settings
      - Any network / HTTP / parse error occurs
    """
    global _internal_config_cache, _internal_config_fetched_at

    now = time.monotonic()
    if _internal_config_cache and (now - _internal_config_fetched_at) < _INTERNAL_CONFIG_TTL:
        return _internal_config_cache

    s = get_settings()
    if not s.backend_internal_token:
        logger.debug("fetch_internal_config: BACKEND_INTERNAL_TOKEN not set — skipping")
        return {}

    url = s.backend_base_url.rstrip("/") + "/api/v1/shared/config/internal/"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                url,
                headers={
                    "Accept": "application/json",
                    "X-Internal-Token": s.backend_internal_token,
                },
            )
            resp.raise_for_status()
            # Peel the {"status": true, "data": {...}} envelope to a flat dict.
            data = _unwrap_envelopes(resp.json())
        if not isinstance(data, dict):
            logger.warning("fetch_internal_config: unexpected response type %s", type(data))
            data = {}
        _internal_config_cache = data
        _internal_config_fetched_at = now
        return data
    except Exception as exc:
        logger.warning("fetch_internal_config failed (falling back to env): %s", exc)
        return {}


def _invalidate_internal_config_cache() -> None:
    """Reset the internal config in-process cache — useful in tests."""
    global _internal_config_cache, _internal_config_fetched_at
    _internal_config_cache = {}
    _internal_config_fetched_at = 0.0


class BackendError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.details = details
        # Parsed once at construction so tool code can program against it
        # without re-walking the envelope.
        self.payload = _unwrap_envelopes(details)
        self.field_errors = _extract_field_errors(self.payload)
        self.detail_message = _extract_detail(self.payload)
        self.codes = _extract_codes(self.payload)

    def __str__(self) -> str:
        base = super().__str__()
        if self.details is None:
            return base
        try:
            body = _format_details(self.details)
        except Exception:
            body = ""
        return f"{base}\n{body}" if body else base

    def human_message(self) -> str:
        """One-line human-readable summary (without HTTP location prefix)."""
        if self.field_errors:
            first_field, msgs = next(iter(self.field_errors.items()))
            text = msgs[0] if msgs else ""
            if first_field in {"non_field_errors", "detail", "errors"}:
                return text
            return f"{first_field}: {text}"
        if self.detail_message:
            return self.detail_message
        if isinstance(self.payload, str):
            return self.payload
        return f"HTTP {self.status}" if self.status else "Backend error"


def format_backend_error(action: str | None, err: "BackendError") -> str:
    """Tool'lar uchun yagona format — barcha mavjud sabab manbalarini chiqaradi.

    Birinchi qator: action + HTTP status. Keyin har field xatosi alohida
    qator, yoki detail_message, yoki raw payload — qaysi biri mavjud bo'lsa.
    Bu LLM uchun aniq xato sababini ko'rishga imkon beradi (oddiy "Backend
    error 400" o'rniga: "phone: unique constraint violated").
    """
    header = (
        f"{action} muvaffaqiyatsiz (HTTP {err.status or '?'})"
        if action else f"Backend xato (HTTP {err.status or '?'})"
    )
    parts: list[str] = [header]

    if err.field_errors:
        parts.append("Field validatsiya xatolari:")
        for field, msgs in err.field_errors.items():
            joined = "; ".join(str(m) for m in msgs)
            if field in {"non_field_errors", "detail", "errors"}:
                parts.append(f"  • {joined}")
            else:
                parts.append(f"  • {field}: {joined}")
    elif err.detail_message:
        parts.append(f"Sabab: {err.detail_message}")
    elif err.payload is not None:
        raw = str(err.payload)
        if len(raw) > 500:
            raw = raw[:497] + "..."
        parts.append(f"Backend javobi: {raw}")
    else:
        parts.append("(backend body bo'sh — pod log'da batafsilroq bo'lishi mumkin)")

    if err.codes:
        parts.append(f"Backend kod(lar)i: {err.codes}")

    return "\n".join(parts)


def _unwrap_envelopes(data: Any, depth: int = 0) -> Any:
    """Peel repeated ``{"status": <bool>, "data": …}`` envelopes.

    Combined with `BaseViewSetMixin.finalize_response` + project-level
    `custom_exception_handler`, errors arrive double-wrapped:
        {"status": false, "data": {"status": false, "detail": "ValidationError",
                                    "errors": {"plan_id": ["…"]}}}
    This walks down until it sees actual content.
    """
    while (
        isinstance(data, dict)
        and "data" in data
        and ("status" in data or "errors" in data or "detail" in data)
        and depth < 4
    ):
        # Only descend if the inner `data` is itself a dict/list — otherwise
        # we'd lose information (e.g. {"status": true, "data": "ok"}).
        inner = data.get("data")
        if not isinstance(inner, (dict, list)):
            break
        data = inner
        depth += 1
    return data


def _extract_field_errors(payload: Any) -> dict[str, list[str]]:
    """Return DRF-style field → [messages] regardless of envelope shape."""
    if not isinstance(payload, dict):
        return {}
    # Preferred shape from custom_exception_handler: {"errors": {…}}
    errors = payload.get("errors")
    if isinstance(errors, dict):
        return {k: _as_msg_list(v) for k, v in errors.items()}
    if isinstance(errors, list):
        return {"non_field_errors": [str(x) for x in errors]}
    # Bare DRF ValidationError dict: {"plan_id": [...], "project": [...]}.
    # Heuristic: every value is a list/str of messages and no envelope keys.
    envelope_keys = {"status", "detail", "code", "incident_id"}
    if all(
        k not in envelope_keys and isinstance(v, (list, str))
        for k, v in payload.items()
    ) and payload:
        return {k: _as_msg_list(v) for k, v in payload.items()}
    return {}


def _as_msg_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _extract_codes(payload: Any) -> list[str]:
    """Flatten the backend's `codes` map into a flat list for easy matching."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("codes")
    if raw is None and isinstance(payload.get("code"), str):
        return [payload["code"]]
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(raw)
    return [c for c in out if c]


def _extract_detail(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if isinstance(detail, str) and detail not in {"ValidationError", "ParseError"}:
        return detail
    return None


def _format_details(details: Any) -> str:
    """Convert DRF error envelope into a compact human-readable string.

    Output is line-oriented and field-prefixed so the LLM can match on
    substrings (the system prompt encodes a few of these as decision triggers).
    """
    if details is None:
        return ""
    payload = _unwrap_envelopes(details)
    lines: list[str] = []
    detail = _extract_detail(payload)
    if detail:
        lines.append(detail)
    for field, msgs in _extract_field_errors(payload).items():
        joined = "; ".join(msgs)
        if field in {"non_field_errors", "detail"}:
            lines.append(joined)
        else:
            lines.append(f"- {field}: {joined}")
    if isinstance(payload, dict):
        code = payload.get("code")
        if isinstance(code, str):
            lines.append(f"(code: {code})")
    if lines:
        return "\n".join(lines)
    if isinstance(payload, list):
        return "\n".join(str(x) for x in payload)
    return str(payload)


class PagedList(list):
    """A ``list`` that also carries DRF pagination metadata.

    Subclassing ``list`` keeps every existing caller working unchanged (len,
    iteration, indexing, JSON-serialisation) while exposing the true totals so
    list tools can tell the LLM "10 of 137" instead of silently implying 10.
    """

    total_items: int = 0
    total_pages: int = 1
    current_page: int = 1
    page_size: int = 0

    @property
    def has_next(self) -> bool:
        return self.current_page < self.total_pages


def _unwrap_list(payload: Any) -> "PagedList":
    if isinstance(payload, list):
        pl = PagedList(payload)
        pl.total_items = len(payload)
        pl.page_size = len(payload)
        return pl
    if not isinstance(payload, dict):
        return PagedList()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        results = payload["results"] if isinstance(payload.get("results"), list) else []
    pl = PagedList(results)
    meta = data if isinstance(data, dict) else {}
    pl.total_items = int(meta.get("total_items") or meta.get("count") or len(results))
    pl.total_pages = int(meta.get("total_pages") or 1)
    pl.current_page = int(meta.get("current_page") or 1)
    pl.page_size = int(meta.get("page_size") or len(results))
    return pl


def pagination_summary(items: Any) -> str:
    """Short hint for the LLM when a list is paginated, else ''.

    e.g. " (page 1/14, 137 total — call again with page=2 for the rest)".
    """
    pages = getattr(items, "total_pages", 1) or 1
    if pages <= 1:
        return ""
    cur = getattr(items, "current_page", 1) or 1
    total = getattr(items, "total_items", len(items))
    size = getattr(items, "page_size", 0) or 0
    # Include page_size so the LLM repeats the SAME size on the next call —
    # otherwise it may switch (e.g. page_size=100) and request a now-out-of-range
    # page. Keeping page_size constant keeps total_pages valid.
    keep = f" with page_size={size}" if size else ""
    more = f" call again with page={cur + 1}{keep} for the rest." if cur < pages else ""
    return f" (page {cur}/{pages}, {total} total —{more})"


def _unwrap_item(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


def _auth_kw(auth_claims: dict | None) -> dict[str, Any]:
    """Split agent auth claims into BackendClient._request kwargs.

    Yangi rejim: BackendClient agent'ning INTERNAL_API_TOKEN bilan
    chaqiradi va `X-User-Id` header'ida foydalanuvchi id'sini beradi —
    foydalanuvchi JWT'si forward qilinmaydi. Backend
    InternalTokenAuthentication shu pattern'ni qabul qiladi.

    Eski rejim (JWT forward + refresh) hali ham qoldirilgan
    fallback sifatida — internal token sozlanmagan dev/test
    muhitlarida tools ishlasin.
    """
    if not auth_claims:
        return {}
    user_block = auth_claims.get("user") or {}
    user_id = (
        auth_claims.get("user_id")
        or user_block.get("id")
        or auth_claims.get("sub")
    )
    return {
        "user_id": str(user_id) if user_id is not None else None,
        "bearer": auth_claims.get("access_token"),
        "refresh_token": auth_claims.get("refresh_token"),
        "auth_claims": auth_claims,
    }


class BackendClient:
    """Async httpx client with Django envelope unwrapping."""

    def __init__(self, base_url: str, *, internal_token: str | None = None, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._internal_token = internal_token
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(
        self, *, bearer: str | None = None, user_id: str | None = None
    ) -> dict[str, str]:
        """Compose auth headers.

        Priority:
          1. Internal token + X-User-Id (system-to-system) — used when
             both INTERNAL_API_TOKEN is configured AND we know the
             user_id (which we do after agent's JWT verification).
          2. Bearer JWT (legacy / fallback when no internal token).
          3. Internal token only — for endpoints that don't need a
             user context (e.g. /api/template/internal/list/).
        """
        h = {"Accept": "application/json"}
        if self._internal_token and user_id:
            h["X-Internal-Token"] = self._internal_token
            h["X-User-Id"] = user_id
        elif bearer:
            h["Authorization"] = f"Bearer {bearer}"
        elif self._internal_token:
            h["X-Internal-Token"] = self._internal_token
        return h

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _try_refresh(self, refresh_token: str) -> tuple[str, str | None] | None:
        """Exchange a refresh token for a fresh access (and refresh) token.

        Backend SimpleJWT konfiguratsiyasi ROTATE_REFRESH_TOKENS=True va
        BLACKLIST_AFTER_ROTATION=True bilan ishlaydi. Bu degani:
          - Har refresh paytida YANGI refresh ham qaytadi
          - Eski refresh blacklist'ga qo'shiladi (qayta ishlatib bo'lmaydi)
        Shu sababli yangi refresh'ni HAR DOIM saqlash zarur — keyingi
        refresh'larda undan foydalanamiz. Aks holda 2-marotaba refresh
        urinilganda blacklist'dagi eski token ishlatilib 401 qaytadi.

        Returns: (new_access, new_refresh_or_None) yoki None — agar refresh fail.
        """
        if not refresh_token:
            return None
        url = self._base_url + "/auth/token/refresh/"
        try:
            resp = await self._client.post(
                url,
                json={"refresh": refresh_token},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Agent token refresh failed: %s %s",
                    resp.status_code,
                    resp.text[:200] if resp.text else "",
                )
                return None
            data = resp.json() or {}
            root = data.get("data") if isinstance(data.get("data"), dict) else data
            token = (root or {}).get("token") or {}
            access = token.get("access") or root.get("access")
            new_refresh = token.get("refresh") or root.get("refresh")
            if isinstance(access, str) and access:
                logger.info(
                    "Agent access token refreshed mid-run (rotated_refresh=%s)",
                    bool(new_refresh),
                )
                return (
                    access,
                    new_refresh if isinstance(new_refresh, str) and new_refresh else None,
                )
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("Agent token refresh exception: %s", e)
            return None

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
    )
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        user_id: str | None = None,
        bearer: str | None = None,
        refresh_token: str | None = None,
        auth_claims: dict | None = None,
    ) -> Any:
        url = self._base_url + path
        use_internal = bool(self._internal_token and user_id)
        resp = await self._client.request(
            method,
            url,
            params=params,
            json=json,
            headers=self._headers(bearer=bearer, user_id=user_id),
        )
        # Refresh-on-401 only when we're using a forwarded user JWT.
        # When the request was made with the internal token + X-User-Id
        # there is no refresh to perform — the internal token does not
        # expire, and a 401 there means the X-User-Id is invalid.
        if (
            resp.status_code == 401
            and not use_internal
            and refresh_token
        ):
            refreshed = await self._try_refresh(refresh_token)
            if refreshed:
                new_access, new_refresh = refreshed
                # auth_claims dict'i `ctx.state["auth_claims"]` ga reference;
                # in-place yangilash keyingi tool call'larni darhol yangi
                # token'lar bilan ishlatadi. Backend rotation rejimida bo'lsa,
                # YANGI refresh ham saqlanadi — aks holda eski blacklist'dagi
                # refresh ikkinchi marotaba ishlatilib 401 qaytaradi.
                if auth_claims is not None:
                    auth_claims["access_token"] = new_access
                    if new_refresh:
                        auth_claims["refresh_token"] = new_refresh
                resp = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=self._headers(bearer=new_access),
                )
        # Out-of-range pagination is NOT an error. DRF returns 404
        # {"detail": "Invalid page."} when the requested page is past the end
        # (e.g. page=2 with a page_size that fits everything on page 1). For a
        # paginated LIST request that just means "no more rows" — return an
        # empty paginated envelope so callers get an empty PagedList instead of
        # a hard failure. Genuine 404s (missing object → detail "Not found.")
        # still raise below.
        if resp.status_code == 404 and params and params.get("page"):
            try:
                body = resp.json()
                detail = str(body.get("detail", "")).lower() if isinstance(body, dict) else ""
            except Exception:
                detail = ""
            if "invalid page" in detail:
                try:
                    requested_page = int(params.get("page"))
                except (TypeError, ValueError):
                    requested_page = 1
                return {
                    "results": [],
                    "total_items": 0,
                    "total_pages": requested_page,
                    "current_page": requested_page,
                    "page_size": 0,
                }
        if resp.status_code >= 400:
            try:
                details = resp.json()
            except Exception:
                details = resp.text
            raise BackendError(
                f"{method} {path} → {resp.status_code}",
                status=resp.status_code,
                details=details,
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ── projects ───────────────────────────────────────────────────────
    async def list_projects(
        self,
        *,
        search: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {}
        if search:
            params["search"] = search
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request("GET", "/api/project/", params=params or None, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def get_project(self, project_id: str | int, *, auth_claims: dict | None = None) -> dict[str, Any]:
        d = await self._request("GET", f"/api/project/{project_id}/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def create_project(self, body: dict[str, Any], *, auth_claims: dict | None = None) -> dict[str, Any]:
        d = await self._request("POST", "/api/project/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def update_project(
        self, project_id: str | int, body: dict[str, Any], *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request("PATCH", f"/api/project/{project_id}/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def delete_project(self, project_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/project/{project_id}/", **_auth_kw(auth_claims))

    # ── env vars ───────────────────────────────────────────────────────
    async def list_env(
        self,
        project_id: str | int,
        *,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {"project": project_id}
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request("GET", "/api/enviroment/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def create_env(self, body: dict[str, Any], *, auth_claims: dict | None = None) -> dict[str, Any]:
        d = await self._request("POST", "/api/enviroment/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def delete_env(self, env_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/enviroment/{env_id}/", **_auth_kw(auth_claims))

    # ── releases (deploy) ──────────────────────────────────────────────
    async def list_releases(
        self,
        project_id: str | int,
        *,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {"project": project_id}
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request("GET", "/api/relase/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def create_release(
        self,
        project_id: str | int,
        *,
        plan_id: int | None = None,
        months: int | None = None,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"project": project_id}
        if plan_id is not None:
            body["plan_id"] = plan_id
        if months is not None:
            body["months"] = months
        d = await self._request("POST", "/api/relase/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def get_release(
        self, release_id: str | int, *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request("GET", f"/api/relase/{release_id}/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def start_debug(
        self, project_id: str | int, *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        """Start (or refresh) the project's DEBUG container — no bot token needed,
        auto-expires (~30 min) via the backend's cleanup task. Same endpoint the
        constructor's 'Debug Run' uses: GET /api/project/{id}/debug/."""
        d = await self._request(
            "GET", f"/api/project/{project_id}/debug/", **_auth_kw(auth_claims)
        )
        return _unwrap_item(d) if isinstance(d, dict) else {}

    async def get_release_logs(
        self,
        release_id: str | int,
        *,
        tail: int = 200,
        since_seconds: int | None = None,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"tail": tail}
        if since_seconds is not None:
            params["since_seconds"] = since_seconds
        d = await self._request(
            "GET",
            f"/api/relase/{release_id}/logs/",
            params=params,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(d)

    async def get_debug_logs(
        self,
        project_id: str | int,
        *,
        tail: int = 200,
        since_seconds: int | None = None,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        """DEBUG konteyner pod loglari — release logs bilan bir xil snapshot format."""
        params: dict[str, Any] = {"tail": tail}
        if since_seconds is not None:
            params["since_seconds"] = since_seconds
        d = await self._request(
            "GET",
            f"/api/project/{project_id}/debug/logs/",
            params=params,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(d)

    # ── credentials (owner-scoped, read-only for the agent) ────────────
    async def list_credentials(
        self,
        *,
        type_key: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        """Owner-scoped credentials (GET /api/v1/credentials/?type_key=).

        Secrets are masked server-side; only id/name/type_key/label/mode are
        useful for picking a `credential_id`. Owner-scoped via request.user —
        NOT filtered by project. Paginated DRF list.
        """
        params: dict[str, Any] = {}
        if type_key:
            params["type_key"] = type_key
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request(
            "GET", "/api/v1/credentials/", params=params or None, **_auth_kw(auth_claims)
        )
        return _unwrap_list(d)

    # ── plans (billing) ───────────────────────────────────────────────
    async def list_plans(self, *, auth_claims: dict | None = None) -> list[dict[str, Any]]:
        d = await self._request("GET", "/api/plans/", **_auth_kw(auth_claims))
        return _unwrap_list(d)

    # ── files ──────────────────────────────────────────────────────────
    async def list_files(
        self,
        project_id: str | int | None = None,
        *,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project"] = project_id
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request("GET", "/api/file/", params=params or None, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def delete_file(self, file_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/file/{file_id}/", **_auth_kw(auth_claims))

    # ── collections (metadata) ─────────────────────────────────────────
    async def list_collections(
        self,
        project_id: str | int,
        *,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {"project": project_id}
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request("GET", "/api/collection/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def get_collection(
        self, collection_id: str | int, *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request("GET", f"/api/collection/{collection_id}/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    # ── chat (CRM users + history) ─────────────────────────────────────
    async def list_chat_users(
        self,
        project_id: str | int,
        *,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {}
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request(
            "GET", f"/api/chat/users/{project_id}/", params=params or None, **_auth_kw(auth_claims)
        )
        return _unwrap_list(d)

    async def user_chat_history(
        self,
        project_id: str | int,
        chat_id: str | int,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        d = await self._request(
            "GET", f"/api/chat/messages/{project_id}/{chat_id}/", **_auth_kw(auth_claims)
        )
        return _unwrap_list(d)

    # ── messages (broadcast) ───────────────────────────────────────────
    async def list_messages(
        self,
        project_id: str | int | None = None,
        *,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project"] = project_id
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request("GET", "/api/message/", params=params or None, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def create_message(
        self, body: dict[str, Any], *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request("POST", "/api/message/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def delete_message(self, message_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/message/{message_id}/", **_auth_kw(auth_claims))

    # ── subscription usage ────────────────────────────────────────────
    async def subscription_current(self, *, auth_claims: dict | None = None) -> dict[str, Any]:
        d = await self._request("GET", "/api/subscription/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def subscription_upgrade(
        self,
        *,
        plan_id: int,
        months: int = 1,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            "/api/subscription/upgrade/",
            json={"plan_id": plan_id, "months": months},
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(d)

    async def subscription_usage(self, *, auth_claims: dict | None = None) -> dict[str, Any]:
        d = await self._request("GET", "/api/subscription/usage/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    # ── payments ───────────────────────────────────────────────────────
    async def create_payment_link(
        self,
        *,
        method: str,
        amount: int,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            "/api/payment/create-transaction/",
            json={"method": method, "amount": amount},
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(d)

    # ── cms ──────────────────────────────────────────────────────────
    def _cms(self, path: str) -> str:
        return f"/api/cms{path}"

    async def cms_list_collections(
        self,
        project_id: str | int,
        *,
        page: int | None = None,
        page_size: int | None = None,
        auth_claims: dict | None = None,
    ) -> "PagedList":
        params: dict[str, Any] = {}
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        d = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/"),
            params=params or None,
            **_auth_kw(auth_claims),
        )
        return _unwrap_list(d)

    async def cms_get_collection(
        self, project_id: str | int, name: str, *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request(
            "GET", self._cms(f"/projects/{project_id}/collections/{name}/"), **_auth_kw(auth_claims)
        )
        return _unwrap_item(d)

    async def cms_create_collection(
        self, project_id: str | int, body: dict[str, Any], *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/"),
            json=body,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(d)

    async def cms_delete_collection(
        self, project_id: str | int, name: str, *, auth_claims: dict | None = None
    ) -> None:
        await self._request(
            "DELETE",
            self._cms(f"/projects/{project_id}/collections/{name}/"),
            **_auth_kw(auth_claims),
        )

    async def cms_add_field(
        self,
        project_id: str | int,
        name: str,
        body: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/fields/"),
            json=body,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(d)

    async def cms_update_field(
        self,
        project_id: str | int,
        name: str,
        field_name: str,
        body: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "PATCH",
            self._cms(f"/projects/{project_id}/collections/{name}/fields/{field_name}/"),
            json=body,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(d)

    async def cms_remove_field(
        self,
        project_id: str | int,
        name: str,
        field_name: str,
        *,
        auth_claims: dict | None = None,
    ) -> None:
        await self._request(
            "DELETE",
            self._cms(f"/projects/{project_id}/collections/{name}/fields/{field_name}/"),
            **_auth_kw(auth_claims),
        )

    async def cms_list_documents(
        self,
        project_id: str | int,
        name: str,
        *,
        filter: dict[str, Any] | None = None,
        sort: str | None = None,
        limit: int = 50,
        skip: int = 0,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        import json as _json

        params: dict[str, Any] = {"limit": limit, "skip": skip}
        if filter:
            params["filter"] = _json.dumps(filter)
        if sort:
            params["sort"] = sort
        d = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/"),
            params=params,
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, dict) else {"results": []}

    async def cms_get_document(
        self, project_id: str | int, name: str, document_id: str, *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/{document_id}/"),
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, dict) else {}

    async def cms_insert(
        self,
        project_id: str | int,
        name: str,
        document: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/"),
            json=document,
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, dict) else {}

    async def cms_bulk_insert(
        self,
        project_id: str | int,
        name: str,
        documents: list[dict[str, Any]],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/bulk/"),
            json={"documents": documents},
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, dict) else {}

    async def cms_update(
        self,
        project_id: str | int,
        name: str,
        document_id: str,
        patch: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "PATCH",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/{document_id}/"),
            json=patch,
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, dict) else {}

    async def cms_delete(
        self,
        project_id: str | int,
        name: str,
        document_id: str,
        *,
        auth_claims: dict | None = None,
    ) -> None:
        await self._request(
            "DELETE",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/{document_id}/"),
            **_auth_kw(auth_claims),
        )

    async def cms_restore(
        self,
        project_id: str | int,
        name: str,
        document_id: str,
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            self._cms(
                f"/projects/{project_id}/collections/{name}/documents/{document_id}/restore/"
            ),
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, dict) else {}

    async def cms_aggregate(
        self,
        project_id: str | int,
        name: str,
        pipeline: list[dict[str, Any]],
        *,
        limit: int = 500,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        d = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/aggregate/"),
            json={"pipeline": pipeline, "limit": limit},
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, dict) else {}

    async def cms_audit(
        self,
        project_id: str | int,
        name: str,
        *,
        action: str | None = None,
        document_id: str | None = None,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        params = {k: v for k, v in {"action": action, "document_id": document_id}.items() if v}
        d = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/{name}/audit/"),
            params=params or None,
            **_auth_kw(auth_claims),
        )
        return d if isinstance(d, list) else _unwrap_list(d)

    async def report_token_usage(
        self,
        user_id: str | int,
        input_tokens: int,
        output_tokens: int,
        project_id: str | int | None = None,
        *,
        auth_claims: dict | None = None,  # noqa: ARG002 — interface uchun saqlangan, bu endpoint ishlatmaydi
    ) -> None:
        """Report AI token usage to backend for subscription tracking.

        BU SYSTEM-TO-SYSTEM endpoint — foydalanuvchi Bearer JWT'ni qabul qilmaydi.
        `X-Internal-Token` header orqali avtorizatsiya qiladi (backend tomonida
        `INTERNAL_API_TOKEN` settings'da bo'lishi kerak).

        Agar agent settings'ida `BACKEND_INTERNAL_TOKEN` yo'q bo'lsa, bu
        endpoint chaqirilmaydi — bu rejim'da subscription token tracking
        ishlamaydi (xato emas).
        """
        if not self._internal_token:
            # Internal token sozlanmagan — silent skip, run muvaffaqiyatini buzmaymiz.
            logger.info("Token usage report skipped (BACKEND_INTERNAL_TOKEN not set)")
            return
        # System-to-system: backend X-Internal-Token bilan BIRGA X-User-Id
        # header'ini ham talab qiladi ("X-Internal-Token requires X-User-Id").
        # Shu sabab user_id ni _request'ga uzatamiz — _headers ikkala header'ni
        # qo'shadi (aks holda 401 authentication_failed bo'ladi).
        # Backend project_id ni IntegerField sifatida qabul qiladi — "global",
        # bo'sh string yoki noma'lum qiymatlar 400 qaytaradi. Faqat haqiqiy
        # numeric id'lar yuboriladi, qolgan holatlarda null.
        pid_payload: int | None = None
        if project_id is not None:
            pid_str = str(project_id).strip()
            if pid_str.isdigit():
                pid_payload = int(pid_str)
        await self._request(
            "POST",
            "/api/subscription/usage/report/",
            json={
                "user_id": int(user_id) if str(user_id).strip().isdigit() else user_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "project_id": pid_payload,
            },
            user_id=str(user_id),
        )

    async def list_internal_templates(self) -> list[dict[str, Any]]:
        """Fetch ALL verified templates WITH their flow JSON for RAG indexing.

        Uses X-Internal-Token; no user JWT involved. Backend returns
        {"results": [...]} with each template's flow embedded.
        """
        if not self._internal_token:
            logger.warning("list_internal_templates skipped — BACKEND_INTERNAL_TOKEN not set")
            return []
        d = await self._request("GET", "/api/template/internal/list/")
        if isinstance(d, dict) and isinstance(d.get("results"), list):
            return d["results"]
        return []

    async def get_internal_template(self, identifier: str | int) -> dict[str, Any] | None:
        """Fetch a single template (by id or slug) WITH flow JSON."""
        if not self._internal_token:
            return None
        try:
            return await self._request("GET", f"/api/template/internal/{identifier}/")
        except BackendError as e:
            if e.status == 404:
                return None
            raise


@lru_cache(maxsize=2)
def get_backend_client(_settings_id: int = 0) -> BackendClient:
    s = get_settings()
    return BackendClient(s.backend_base_url, internal_token=s.backend_internal_token)


def _flow_is_nonempty(flow: Any) -> bool:
    return isinstance(flow, dict) and bool(flow.get("nodes") or flow.get("edges"))


async def load_project_collections(
    project_id: str | int | None,
    *,
    auth_claims: dict[str, Any] | None = None,
    client: BackendClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch the project's CMS collections + their field names, compactly.

    Returns `[{"name": str, "fields": [str, ...]}]` so the flow-builder knows
    which collections (and fields) the cms-manager created and can reference
    them correctly in flow nodes / expressions.

    Robust by contract: NEVER raises. On any failure (no project, backend
    error, malformed payload) it returns `[]` and logs a warning, so a backend
    hiccup can't break runs. Per-collection field fetch failures are isolated —
    one bad collection won't drop the others.
    """
    pid = "" if project_id is None else str(project_id)
    if not pid or pid == "global":
        return []
    try:
        backend = client or get_backend_client()
        cols = await backend.cms_list_collections(pid, auth_claims=auth_claims)
    except Exception:
        logger.warning(
            "Failed to list collections for project=%s; continuing without them",
            pid,
            exc_info=True,
        )
        return []

    out: list[dict[str, Any]] = []
    for c in cols or []:
        name = c.get("name") if isinstance(c, dict) else None
        if not name:
            continue
        fields: list[str] = []
        # field name → kind (e.g. {"is_active": "bool"}). Kept separate from
        # `fields` so existing name-only consumers (system prompt, validator
        # cross-check) stay unchanged; the validator's filter-kind check (#4)
        # reads `field_kinds` when present.
        field_kinds: dict[str, str] = {}
        try:
            detail = await backend.cms_get_collection(pid, name, auth_claims=auth_claims)
            for f in (detail.get("fields") or []) if isinstance(detail, dict) else []:
                fname = f.get("name") if isinstance(f, dict) else None
                if fname:
                    fields.append(str(fname))
                    kind = f.get("kind") if isinstance(f, dict) else None
                    if kind:
                        field_kinds[str(fname)] = str(kind)
        except Exception:
            logger.warning(
                "Failed to load fields for collection=%s (project=%s); listing name only",
                name,
                pid,
                exc_info=True,
            )
        out.append({"name": str(name), "fields": fields, "field_kinds": field_kinds})
    return out


async def load_saved_flow(
    project_id: str | int | None,
    *,
    body_baseline: dict[str, Any] | None = None,
    auth_claims: dict[str, Any] | None = None,
    client: BackendClient | None = None,
) -> dict[str, Any] | None:
    """Resolve the baseline flow for a new run.

    A fresh conversation starts with an empty working_flow unless the client
    passes baseline_flow. Without this the agent would rebuild from scratch and
    effectively wipe the saved flow. So when `body_baseline` is missing/empty we
    load the project's SAVED flow (`project["flow"]`) from the backend.

    Robust by contract: NEVER raises. On any failure (no project, backend error,
    malformed payload) it returns `body_baseline` unchanged and logs a warning,
    so a backend hiccup can't break runs.
    """
    if _flow_is_nonempty(body_baseline):
        return body_baseline
    pid = "" if project_id is None else str(project_id)
    if not pid or pid == "global":
        return body_baseline
    try:
        backend = client or get_backend_client()
        project = await backend.get_project(pid, auth_claims=auth_claims)
        flow = project.get("flow")
        if _flow_is_nonempty(flow):
            logger.info(
                "Seeded baseline_flow from saved project flow (project=%s, nodes=%d)",
                pid,
                len((flow or {}).get("nodes") or []),
            )
            return flow
    except Exception:
        logger.warning(
            "Failed to load saved project flow for baseline; starting empty (project=%s)",
            pid,
            exc_info=True,
        )
    return body_baseline

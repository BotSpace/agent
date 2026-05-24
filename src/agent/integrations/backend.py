"""Django backend REST client — agent shu orqali project/release/env/file boshqaradi.

Backend endpoint pattern (dashboard `services.ts`'dan olingan):
- /api/project/                   GET, POST  /  /:id/  GET, PATCH, DELETE
- /api/relase/                    GET (?project=), POST  (release deploy)
- /api/enviroment/                GET (?project=), POST  /  /:id/  DELETE
- /api/file/                      GET (?project=)  /  /:id/  DELETE
- /api/collection/                GET (?project=)  /  /:id/  GET (metadata; data is in MongoDB)
- /api/project/:id/plan/          GET (current plan), /upgrade/ POST, /usage/ GET
- /api/chat/users/:project        GET
- /api/chat/messages/:project/:chat  GET
- /api/message/                   GET, POST, DELETE (broadcast)

Auth: agent o'z internal token bilan boradi (`X-Internal-Token` header), yoki
foydalanuvchi access token'i forward qilinadi (`Authorization: Bearer ...`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings, get_settings


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


def _unwrap_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, list):
        return results
    if isinstance(payload.get("results"), list):
        return payload["results"]
    return []


def _unwrap_item(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


def _auth_kw(auth_claims: dict | None) -> dict[str, Any]:
    """Split agent auth claims into BackendClient._request kwargs."""
    if not auth_claims:
        return {}
    return {
        "bearer": auth_claims.get("access_token"),
        "refresh_token": auth_claims.get("refresh_token"),
    }


class BackendClient:
    """Async httpx client with Django envelope unwrapping."""

    def __init__(self, base_url: str, *, internal_token: str | None = None, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._internal_token = internal_token
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self, *, bearer: str | None = None) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if bearer:
            h["Authorization"] = f"Bearer {bearer}"
        elif self._internal_token:
            h["X-Internal-Token"] = self._internal_token
        return h

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _try_refresh(self, refresh_token: str) -> str | None:
        """Exchange a refresh token for a fresh access token.

        Mirrors `dashboard/src/shared/api/server.ts`. Returns the new access
        token, or None if refresh failed.
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
                return None
            data = resp.json() or {}
            root = data.get("data") if isinstance(data.get("data"), dict) else data
            token = (root or {}).get("token") or {}
            access = token.get("access") or root.get("access")
            return access if isinstance(access, str) and access else None
        except Exception:
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
        bearer: str | None = None,
        refresh_token: str | None = None,
    ) -> Any:
        url = self._base_url + path
        resp = await self._client.request(
            method, url, params=params, json=json, headers=self._headers(bearer=bearer)
        )
        # Auto-refresh on 401 if a refresh token is available.
        if resp.status_code == 401 and refresh_token:
            new_access = await self._try_refresh(refresh_token)
            if new_access:
                resp = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=self._headers(bearer=new_access),
                )
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
        self, *, search: str | None = None, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        d = await self._request(
            "GET", "/api/project/", params={"search": search} if search else None, **_auth_kw(auth_claims)
        )
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
        self, project_id: str | int, *, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        d = await self._request(
            "GET", "/api/enviroment/", params={"project": project_id}, **_auth_kw(auth_claims)
        )
        return _unwrap_list(d)

    async def create_env(self, body: dict[str, Any], *, auth_claims: dict | None = None) -> dict[str, Any]:
        d = await self._request("POST", "/api/enviroment/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def delete_env(self, env_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/enviroment/{env_id}/", **_auth_kw(auth_claims))

    # ── releases (deploy) ──────────────────────────────────────────────
    async def list_releases(
        self, project_id: str | int, *, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        d = await self._request(
            "GET", "/api/relase/", params={"project": project_id}, **_auth_kw(auth_claims)
        )
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

    # ── plans (billing) ───────────────────────────────────────────────
    async def list_plans(self, *, auth_claims: dict | None = None) -> list[dict[str, Any]]:
        d = await self._request("GET", "/api/plans/", **_auth_kw(auth_claims))
        return _unwrap_list(d)

    # ── files ──────────────────────────────────────────────────────────
    async def list_files(
        self, project_id: str | int | None = None, *, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        params = {"project": project_id} if project_id is not None else None
        d = await self._request("GET", "/api/file/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def delete_file(self, file_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/file/{file_id}/", **_auth_kw(auth_claims))

    # ── collections (metadata) ─────────────────────────────────────────
    async def list_collections(
        self, project_id: str | int, *, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        d = await self._request(
            "GET", "/api/collection/", params={"project": project_id}, **_auth_kw(auth_claims)
        )
        return _unwrap_list(d)

    async def get_collection(
        self, collection_id: str | int, *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request("GET", f"/api/collection/{collection_id}/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    # ── chat (CRM users + history) ─────────────────────────────────────
    async def list_chat_users(
        self, project_id: str | int, *, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        d = await self._request("GET", f"/api/chat/users/{project_id}/", **_auth_kw(auth_claims))
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
        self, project_id: str | int | None = None, *, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        params = {"project": project_id} if project_id is not None else None
        d = await self._request("GET", "/api/message/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(d)

    async def create_message(
        self, body: dict[str, Any], *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request("POST", "/api/message/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def delete_message(self, message_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/message/{message_id}/", **_auth_kw(auth_claims))

    # ── plan ───────────────────────────────────────────────────────────
    async def project_plan(
        self, project_id: str | int, *, auth_claims: dict | None = None
    ) -> dict[str, Any]:
        d = await self._request("GET", f"/api/project/{project_id}/plan/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    # ── cms ──────────────────────────────────────────────────────────
    def _cms(self, path: str) -> str:
        return f"/api/cms{path}"

    async def cms_list_collections(
        self, project_id: str | int, *, auth_claims: dict | None = None
    ) -> list[dict[str, Any]]:
        d = await self._request(
            "GET", self._cms(f"/projects/{project_id}/collections/"), **_auth_kw(auth_claims)
        )
        return d if isinstance(d, list) else _unwrap_list(d)

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

    # ── user / subscription token usage ─────────────────────────────────
    async def get_me(self, *, auth_claims: dict | None = None) -> dict[str, Any]:
        """Return authenticated backend user profile, including ai_token_usage."""
        d = await self._request("GET", "/api/v1/auth/me/", **_auth_kw(auth_claims))
        return _unwrap_item(d)

    async def report_token_usage(
        self,
        user_id: str | int,
        input_tokens: int,
        output_tokens: int,
        project_id: str | int | None = None,
    ) -> None:
        """Report AI token usage to backend for subscription tracking."""
        await self._request(
            "POST",
            "/api/subscription/usage/report/",
            json={
                "user_id": int(user_id),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "project_id": str(project_id) if project_id else None,
            },
        )

@lru_cache(maxsize=2)
def get_backend_client(_settings_id: int = 0) -> BackendClient:
    s = get_settings()
    return BackendClient(s.backend_base_url, internal_token=s.backend_internal_token)

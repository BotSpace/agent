"""Per-(user, project) MCP workspace — flow tahriri state-based bo'lgani uchun.

agentv3 flow tool'lari `ctx.state` ichidagi `working_flow`/`baseline_flow`/
`flow_dirty` bilan ishlaydi (HTTP run'larda bu graph state'da yashaydi). MCP
stateless transport (`stateless_http=True`) bo'lgani uchun har so'rov mustaqil —
shu sabab biz session state'ni server-side, (user_id, project_id) bo'yicha
in-memory saqlaymiz. TTL bilan lazy cleanup.

Token rotation muhim: BackendClient `auth_claims` dict'ini IN-PLACE yangilaydi
(access/refresh rotation). Shu sabab har tool call'da workspace'dagi `auth_claims`
dict'ining ICHINI joriy so'rov claims'i bilan yangilaymiz (reference saqlanadi),
yangi dict bilan almashtirmaymiz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Workspace TTL — oxirgi tegishdan keyin shuncha soniya yashaydi.
_DEFAULT_TTL_SECONDS = 2 * 60 * 60  # 2 soat


@dataclass
class Workspace:
    """Bitta (user, project) sessiyasi uchun flow va auth holati."""

    user_id: str
    project_id: str | None = None
    working_flow: dict[str, Any] | None = None
    baseline_flow: dict[str, Any] | None = None
    flow_dirty: bool = False
    auth_claims: dict[str, Any] = field(default_factory=dict)
    collections: list[dict[str, Any]] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def to_tool_state(self) -> dict[str, Any]:
        """`ToolContext.state` uchun dict — flow tool'lari shu kalitlarni o'qiydi."""
        state: dict[str, Any] = {
            "project_id": self.project_id,
            "auth_claims": self.auth_claims,
            "flow_dirty": self.flow_dirty,
        }
        if self.working_flow is not None:
            state["working_flow"] = self.working_flow
        if self.baseline_flow is not None:
            state["baseline_flow"] = self.baseline_flow
        if self.collections:
            state["collections"] = self.collections
        return state

    def merge_state_updates(self, updates: dict[str, Any]) -> None:
        """Tool natijasidagi `state_updates`ni workspace'ga qaytarib singdiradi."""
        if not updates:
            return
        if "working_flow" in updates:
            self.working_flow = updates["working_flow"]
        if "baseline_flow" in updates:
            self.baseline_flow = updates["baseline_flow"]
        if "flow_dirty" in updates:
            self.flow_dirty = bool(updates["flow_dirty"])
        if updates.get("project_id"):
            self.project_id = str(updates["project_id"])
        if "collections" in updates:
            self.collections = updates["collections"] or []
        # auth_claims tool ichida in-place yangilanadi (reference saqlangan) —
        # alohida merge shart emas, lekin to'liq dict kelsa qabul qilamiz.
        if "auth_claims" in updates and isinstance(updates["auth_claims"], dict):
            self.auth_claims.update(updates["auth_claims"])

    def sync_auth_claims(self, claims: dict[str, Any]) -> None:
        """Joriy so'rov claims'ini workspace dict'iga IN-PLACE singdiradi.

        Reference saqlanadi — BackendClient mid-call refresh qilganda yangilangan
        token shu dict'da qoladi va keyingi call'larda ishlatiladi.
        """
        if self.auth_claims is None:
            self.auth_claims = {}
        self.auth_claims.update(claims)


class WorkspaceStore:
    """In-memory (user_id, project_id?) → Workspace, TTL bilan lazy cleanup.

    Har user uchun bitta aktiv workspace tutamiz (kalit = user_id). Loyiha
    `session_use_project` orqali tanlanadi va shu workspace ichida yashaydi —
    bu MCP klientining odatdagi ish oqimiga mos (bitta foydalanuvchi, ketma-ket
    loyihalar bilan ishlaydi).
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, Workspace] = {}

    def _cleanup(self) -> None:
        now = time.monotonic()
        stale = [k for k, w in self._store.items() if now - w.last_used > self._ttl]
        for k in stale:
            self._store.pop(k, None)

    def get(self, user_id: str) -> Workspace:
        self._cleanup()
        ws = self._store.get(user_id)
        if ws is None:
            ws = Workspace(user_id=user_id)
            self._store[user_id] = ws
        ws.touch()
        return ws

    def reset(self, user_id: str) -> None:
        self._store.pop(user_id, None)

    def clear(self) -> None:
        self._store.clear()


_store: WorkspaceStore | None = None


def get_workspace_store() -> WorkspaceStore:
    global _store
    if _store is None:
        _store = WorkspaceStore()
    return _store

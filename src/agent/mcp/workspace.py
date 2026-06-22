"""Per-(user, project) MCP workspace — flow tahriri state-based bo'lgani uchun.

agent flow tool'lari `ctx.state` ichidagi `working_flow`/`baseline_flow`/
`flow_dirty` bilan ishlaydi (HTTP run'larda bu graph state'da yashaydi). MCP
stateless transport (`stateless_http=True`) bo'lgani uchun har so'rov mustaqil —
shu sabab biz session state'ni server-side, (user_id, project_id) bo'yicha
in-memory saqlaymiz. TTL bilan lazy cleanup. Draft `draft_store` orqali agent'ning
o'z Postgres bazasiga write-through persist qilinadi (#1) — pod restart'da ham
saqlanmagan draft yo'qolmaydi.

Token rotation muhim: BackendClient `auth_claims` dict'ini IN-PLACE yangilaydi
(access/refresh rotation). Shu sabab har tool call'da workspace'dagi `auth_claims`
dict'ining ICHINI joriy so'rov claims'i bilan yangilaymiz (reference saqlanadi),
yangi dict bilan almashtirmaymiz.

Concurrency (#2): har user uchun bitta `asyncio.Lock` — lookup + execute + merge
atomik bo'lib, parallel flow_patch'lar bir-birini "Aktiv loyiha yo'q" deb sindir-
masin.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any

# Workspace TTL — oxirgi tegishdan keyin shuncha soniya yashaydi.
_DEFAULT_TTL_SECONDS = 2 * 60 * 60  # 2 soat

# Aktiv-loyiha pointer TTL — per-(user, client) pointer'lar to'planib qolmasin.
# Workspace TTL'dan uzunroq: foydalanuvchi sessiyalararo qaytib kelganda ham
# "qaysi loyihada edim" eslab qolinadi (workspace qayta yuklanadi).
_DEFAULT_ACTIVE_TTL_SECONDS = 24 * 60 * 60  # 24 soat

# Snapshot ring (undo) — destructive amaldan oldin olinadi.
_MAX_SNAPSHOTS = 5


@dataclass
class Workspace:
    """Bitta (user, project) sessiyasi uchun flow va auth holati."""

    user_id: str
    project_id: str | None = None
    project_name: str | None = None
    working_flow: dict[str, Any] | None = None
    baseline_flow: dict[str, Any] | None = None
    flow_dirty: bool = False
    auth_claims: dict[str, Any] = field(default_factory=dict)
    collections: list[dict[str, Any]] = field(default_factory=list)
    # Per-project external module node types (dynamic catalog extension), loaded
    # at session_use_project. Each entry = normalized manifest (see
    # backend.load_project_module_nodes). Empty when no modules are enabled.
    module_nodes: list[dict[str, Any]] = field(default_factory=list)
    # Undo ring: each entry = {label, working_flow, flow_dirty}. Newest last.
    snapshots: list[dict[str, Any]] = field(default_factory=list)
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
        if self.module_nodes:
            state["module_nodes"] = self.module_nodes
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
        if "module_nodes" in updates:
            self.module_nodes = updates["module_nodes"] or []
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

    # ── snapshot / undo (#11) ──────────────────────────────────────────
    def is_mutated(self, updates: dict[str, Any]) -> bool:
        """True if state_updates change the draft (working_flow/flow_dirty)."""
        return bool(updates) and (
            "working_flow" in updates or "flow_dirty" in updates
        )

    def push_snapshot(self, label: str) -> None:
        """Take an undo snapshot of the current working_flow before a
        destructive action (apply_template / clear / save)."""
        snap = {
            "label": label,
            "working_flow": copy.deepcopy(self.working_flow),
            "flow_dirty": self.flow_dirty,
        }
        self.snapshots.append(snap)
        if len(self.snapshots) > _MAX_SNAPSHOTS:
            self.snapshots = self.snapshots[-_MAX_SNAPSHOTS:]

    def pop_snapshot(self) -> dict[str, Any] | None:
        """Restore the most recent snapshot; returns it (or None if empty)."""
        if not self.snapshots:
            return None
        snap = self.snapshots.pop()
        self.working_flow = snap.get("working_flow")
        self.flow_dirty = bool(snap.get("flow_dirty"))
        return snap


class WorkspaceStore:
    """In-memory (user_id, project_id) → Workspace, TTL bilan lazy cleanup.

    Loyiha `session_use_project` orqali tanlanadi. Bitta user bir nechta loyiha
    bilan ishlashi mumkin — har biri alohida draft (#1). Har user uchun bitta
    `asyncio.Lock` (#2) — atomik lookup+execute+merge.

    Aktiv-loyiha pointer'i per-CLIENT: kalit `(user_id, client_key)`. Stateless
    HTTP'da bitta user IKKI Claude Code klientidan ulansa, har biri o'z aktiv
    loyihasiga ega bo'ladi — A klient `use_project(1)` qilsa, B klient
    `use_project(2)` qilsa, A'ning flow tool'lari 1'da, B'niki 2'da ishlaydi.
    Workspace'lar (draft'lar) esa `(user_id, project_id)` bo'yicha UMUMIY qoladi —
    bitta loyihada ikki klient ishlasa bitta draft ustida ishlaydi (per-user lock
    serializatsiya qiladi). Faqat "qaysi loyiha aktiv" pointer'i per-client.

    Eski API (`client_key` berilmaganda `client_key=""`) ham ishlaydi — testlar va
    klient identifikatori bo'lmagan holatlar uchun yagona "bo'sh klient" pointer'i.
    """

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        active_ttl_seconds: float = _DEFAULT_ACTIVE_TTL_SECONDS,
    ) -> None:
        self._ttl = ttl_seconds
        self._active_ttl = active_ttl_seconds
        self._store: dict[tuple[str, str | None], Workspace] = {}
        # Active pointer: (user_id, client_key) → (project_id, last_used_monotonic).
        # Per-client — bir user'ning har klienti o'z aktiv loyihasiga ega.
        self._active: dict[tuple[str, str], tuple[str | None, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(self, user_id: str) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock

    def _cleanup(self) -> None:
        now = time.monotonic()
        stale = [k for k, w in self._store.items() if now - w.last_used > self._ttl]
        for k in stale:
            self._store.pop(k, None)
        # Eski (bekor qilingan) per-client pointer'larni ham tozalaymiz — aks holda
        # ko'p ulanadigan klientlar pointer'lari cheksiz to'planib qoladi.
        stale_active = [
            k for k, (_pid, ts) in self._active.items() if now - ts > self._active_ttl
        ]
        for k in stale_active:
            self._active.pop(k, None)

    def get(
        self,
        user_id: str,
        project_id: str | None = None,
        *,
        client_key: str = "",
    ) -> Workspace:
        """Return the workspace for (user, project). If project_id is None, use
        this (user, client)'s currently-active project (set by
        session_use_project). Workspace'lar user+project bo'yicha umumiy."""
        self._cleanup()
        if project_id is None:
            project_id = self._active_get(user_id, client_key)
        key = (user_id, project_id)
        ws = self._store.get(key)
        if ws is None:
            ws = Workspace(user_id=user_id, project_id=project_id)
            self._store[key] = ws
        ws.touch()
        return ws

    def set_active(
        self, user_id: str, project_id: str, *, client_key: str = ""
    ) -> Workspace:
        """Mark project_id active for THIS (user, client) and return its shared
        workspace (creating it if needed)."""
        self._cleanup()
        self._active[(user_id, client_key)] = (project_id, time.monotonic())
        key = (user_id, project_id)
        ws = self._store.get(key)
        if ws is None:
            ws = Workspace(user_id=user_id, project_id=project_id)
            self._store[key] = ws
        ws.touch()
        return ws

    def _active_get(self, user_id: str, client_key: str) -> str | None:
        entry = self._active.get((user_id, client_key))
        if entry is None:
            return None
        project_id, _ts = entry
        # Touch the pointer so an actively-used client never expires mid-session.
        self._active[(user_id, client_key)] = (project_id, time.monotonic())
        return project_id

    def active_project_id(self, user_id: str, *, client_key: str = "") -> str | None:
        return self._active_get(user_id, client_key)

    def reset(self, user_id: str) -> None:
        for key in [k for k in self._store if k[0] == user_id]:
            self._store.pop(key, None)
        for ak in [k for k in self._active if k[0] == user_id]:
            self._active.pop(ak, None)
        self._locks.pop(user_id, None)

    def clear(self) -> None:
        self._store.clear()
        self._active.clear()
        self._locks.clear()


_store: WorkspaceStore | None = None


def get_workspace_store() -> WorkspaceStore:
    global _store
    if _store is None:
        _store = WorkspaceStore()
    return _store

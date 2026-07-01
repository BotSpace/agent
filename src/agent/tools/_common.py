"""Tool paketlari uchun umumiy helper'lar — DI xaltasi + project-id resolver.

ponytail: bu helper'lar oldin 9 tool modulida turli nom bilan (`_backend`,
`_project_id`, `_pid`, `_power_project_id`, `_concrete_project_id`) copy-paste edi.
Bitta manba, bitta nom.
"""

from __future__ import annotations

from typing import Any

from ..integrations.backend import BackendClient, get_backend_client
from .base import ToolContext

_PLACEHOLDERS = {"", "global", "none", "null"}


def backend_from_ctx(ctx: ToolContext) -> BackendClient:
    """Backend klientni DI xaltasidan oladi, bo'lmasa global singletonga tushadi."""
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()


def resolve_project_id(
    ctx: ToolContext, override: str | int | None = None
) -> str | int | None:
    """Loose resolver: aniq arg → sessiya state. Placeholder filtrsiz."""
    return override or ctx.state.get("project_id")


def _concrete(value: Any) -> str | None:
    """Placeholder'larni ('', global, none, null) None deb qaytaradi."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _PLACEHOLDERS:
        return None
    return text


def resolve_concrete_project_id(
    ctx: ToolContext, override: Any = None
) -> str | None:
    """Strict resolver: aniq arg → state → ctx.project_id, har biri placeholder-filtrlangan.

    Global kontekstda boshlanib keyin loyiha tanlagan sessiyalar uchun — hech qachon
    `/global/` yubormaydi (flow.save/simulate/snapshot shunga tayanadi)."""
    return (
        _concrete(override)
        or _concrete(ctx.state.get("project_id"))
        or _concrete(ctx.project_id)
    )

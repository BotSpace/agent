"""Umumiy backend-resolver — DI xaltasidan klientni oladi, bo'lmasa globalga tushadi.

ponytail: 6 tool paketida bir xil `_backend` copy-paste edi — bittaga yig'ildi.
Release paketi o'z variantini saqlaydi (isinstance emas, `is not None` + strict scope).
"""

from __future__ import annotations

from ..integrations.backend import BackendClient, get_backend_client
from .base import ToolContext


def backend_from_ctx(ctx: ToolContext) -> BackendClient:
    backend = ctx.services.get("backend")
    if isinstance(backend, BackendClient):
        return backend
    return get_backend_client()

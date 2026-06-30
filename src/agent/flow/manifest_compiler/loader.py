"""Node manifestni YAGONA MANBADAN — backenddan — yuklaydi.

Agent endi manifest nusxasini SAQLAMAYDI (yana bir "manba" chalg'itmasligi uchun):
- Monorepo (lokal/test): backend kanonik faylini to'g'ridan-to'g'ri o'qiydi
  (`../backend/core/apps/api/data/builtin_node_manifests.json`) — network'siz.
- Prod konteyner: backend `GET /api/v1/shared/manifest/builtin-nodes/` dan
  sinxron oladi (bir martalik, lru_cache).

`load_node_manifests()` → `{node_type: manifest_dict}`. Manifest dict backend
`NodeManifest` shaklida (type, category, content[], defaults, aiDoc, ...).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

# Manifest dict — backend NodeManifest (plain dict).
NodeManifest = dict[str, Any]

_WORKSPACE = Path(__file__).resolve().parents[5]
_BACKEND_SIBLING = (
    _WORKSPACE / "backend" / "core" / "apps" / "api" / "data" / "builtin_node_manifests.json"
)
_BACKEND_PATH = "/api/v1/shared/manifest/builtin-nodes/"


def _fetch_from_backend() -> list:
    from ...config import get_settings

    url = get_settings().backend_base_url.rstrip("/") + _BACKEND_PATH
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 (trusted internal URL)
        payload = json.loads(resp.read())
    manifests = payload.get("manifests") if isinstance(payload, dict) else payload
    if not isinstance(manifests, list):
        raise RuntimeError(f"backend manifest javobi noto'g'ri: {url}")
    return manifests


@lru_cache(maxsize=1)
def _raw_entries() -> list:
    """Built-in manifest yozuvlari (monorepo fayl yoki backend fetch)."""
    if _BACKEND_SIBLING.exists():
        return json.loads(_BACKEND_SIBLING.read_bytes())
    return _fetch_from_backend()


@lru_cache(maxsize=1)
def manifest_checksum() -> str:
    """Yuklangan manifestning sha256'i (sinxronlik/diagnostika uchun)."""
    return hashlib.sha256(json.dumps(_raw_entries(), sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_node_manifests() -> dict[str, NodeManifest]:
    """Node type → manifest dict. Type bo'yicha noyob (birinchisi ustun)."""
    out: dict[str, NodeManifest] = {}
    for entry in _raw_entries():
        node_type = entry.get("type")
        if node_type and node_type not in out:
            out[node_type] = entry
    return out

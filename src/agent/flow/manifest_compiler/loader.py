"""Node manifestni YAGONA MANBADAN — backenddan — yuklaydi.

Agent manifest nusxasini SAQLAMAYDI. Barcha muhitlarda manifest backend
`GET /api/v1/shared/manifest/builtin-nodes/` endpointidan sinxron olinadi va
process davomida cache qilinadi. Servislar alohida deploy qilinishi sababli
sibling repository filesystem pathlariga bog'lanilmaydi.

`load_node_manifests()` → `{node_type: manifest_dict}`. Manifest dict backend
`NodeManifest` shaklida (type, category, content[], defaults, aiDoc, ...).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from functools import lru_cache
from typing import Any

# Manifest dict — backend NodeManifest (plain dict).
NodeManifest = dict[str, Any]

_BACKEND_PATH = "/api/v1/shared/manifest/builtin-nodes/"


def _fetch_from_backend() -> list:
    return _fetch_payload()["manifests"]


@lru_cache(maxsize=1)
def _fetch_payload() -> dict:
    """Backend builtin-nodes javobini (manifests + categories) bir marta oladi."""
    from ...config import get_settings

    url = get_settings().backend_base_url.rstrip("/") + _BACKEND_PATH
    with urllib.request.urlopen(url, timeout=15) as resp:
        payload = json.loads(resp.read())
    if not isinstance(payload, dict) or not isinstance(payload.get("manifests"), list):
        raise RuntimeError(f"backend manifest javobi noto'g'ri: {url}")
    return payload


@lru_cache(maxsize=1)
def load_node_category_docs() -> dict[str, dict]:
    """Kategoriya → {label, doc} — backend node_categories.json'dan."""
    cats = _fetch_payload().get("categories")
    if not isinstance(cats, dict):
        raise RuntimeError("backend manifest categories must be an object")
    return cats


@lru_cache(maxsize=1)
def _raw_entries() -> list:
    """Built-in manifest yozuvlarini backend API'dan yuklaydi."""
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

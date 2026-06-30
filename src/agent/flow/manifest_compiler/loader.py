"""Vendored node manifestni yuklash.

`load_node_manifests()` → `{node_type: manifest_dict}`. Manifest dict frontend
`NodeManifest` shaklida (type, category, content[], defaults, handles, ...).
Modul loader patterniga o'xshash, lekin manba statik vendored fayl.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# Manifest dict — frontend NodeManifest (plain dict; tip-modeli shart emas).
NodeManifest = dict[str, Any]

_REGISTRY_PATH = Path(__file__).resolve().parent / "runtime-node-registry.json"


def _read_bytes() -> bytes:
    return _REGISTRY_PATH.read_bytes()


@lru_cache(maxsize=1)
def manifest_checksum() -> str:
    """Vendored manifestning sha256'i (sync drift testi uchun)."""
    return hashlib.sha256(_read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_node_manifests() -> dict[str, NodeManifest]:
    """Node type → manifest dict. `status == "runtime"` yozuvlar indekslanadi.

    Bir xil type ikki marta uchrasa — birinchisi (variant'lar emas, asosiy)
    ustun; lekin registry'da type'lar noyob, shuning uchun amalda to'qnashuv yo'q.
    """
    entries = json.loads(_read_bytes())
    out: dict[str, NodeManifest] = {}
    for entry in entries:
        node_type = entry.get("type")
        if not node_type or node_type in out:
            continue
        out[node_type] = entry
    return out

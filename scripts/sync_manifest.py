#!/usr/bin/env python3
"""Backend kanonik node manifestini agentga vendoring qiladi (offline fallback).

Yagona manba — backend `data/builtin_node_manifests.json` (AI docs bilan).
Jonli rejimda agent backend `/project/{id}/node-manifests/` (built-in + modullar)
dan oladi; bu vendored nusxa esa offline/boot fallback. Skript nusxani yangilaydi
yoki `--check` rejimida drift borligini tekshiradi (CI uchun).

Foydalanish:
    python scripts/sync_manifest.py            # nusxani yangilaydi
    python scripts/sync_manifest.py --check     # drift bo'lsa exit 1

Backend repo yo'li: --backend bayrog'i yoki MANIFEST_BACKEND_DIR env, aks holda
monorepo sibling (`../backend`).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_VENDORED = (
    _AGENT_ROOT / "src" / "agent" / "flow" / "manifest_compiler"
    / "runtime-node-registry.json"
)
_REL_SOURCE = Path("core/apps/api/data/builtin_node_manifests.json")


def _backend_source(arg: str | None) -> Path:
    base = arg or os.environ.get("MANIFEST_BACKEND_DIR") or str(_AGENT_ROOT.parent / "backend")
    return Path(base) / _REL_SOURCE


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="backend repo ildizi")
    ap.add_argument("--check", action="store_true", help="drift bo'lsa exit 1 (yozmaydi)")
    args = ap.parse_args()

    source = _backend_source(args.backend)
    if not source.exists():
        print(f"❌ Backend manifest topilmadi: {source}", file=sys.stderr)
        return 2

    src_bytes = source.read_bytes()
    src_sha = _sha(src_bytes)

    if args.check:
        if not _VENDORED.exists():
            print(f"❌ Vendored nusxa yo'q: {_VENDORED}", file=sys.stderr)
            return 1
        if _sha(_VENDORED.read_bytes()) != src_sha:
            print("❌ Drift: vendored manifest frontend bilan mos emas. "
                  "`python scripts/sync_manifest.py` ishga tushiring.", file=sys.stderr)
            return 1
        print(f"✅ Sync (sha256={src_sha[:12]})")
        return 0

    _VENDORED.parent.mkdir(parents=True, exist_ok=True)
    _VENDORED.write_bytes(src_bytes)
    print(f"✅ Yangilandi: {_VENDORED.relative_to(_AGENT_ROOT)} (sha256={src_sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

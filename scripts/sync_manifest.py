#!/usr/bin/env python3
"""Frontend node manifestini agentga vendoring qiladi (sync).

Yagona manba — frontend `runtime-node-registry.json`. Agent uni o'z image'iga
nusxalab oladi (offline, deterministik). Bu skript nusxani yangilaydi yoki
`--check` rejimida drift borligini tekshiradi (CI uchun).

Foydalanish:
    python scripts/sync_manifest.py            # nusxani yangilaydi
    python scripts/sync_manifest.py --check     # drift bo'lsa exit 1

Frontend repo yo'li: --frontend bayrog'i yoki MANIFEST_FRONTEND_DIR env, aks holda
monorepo sibling (`../bot-constructor-panel`).
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
_REL_SOURCE = Path("src/features/bot-constructor/manifest/runtime-node-registry.json")


def _frontend_source(arg: str | None) -> Path:
    base = arg or os.environ.get("MANIFEST_FRONTEND_DIR") or str(
        _AGENT_ROOT.parent / "bot-constructor-panel"
    )
    return Path(base) / _REL_SOURCE


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default=None, help="frontend repo ildizi")
    ap.add_argument("--check", action="store_true", help="drift bo'lsa exit 1 (yozmaydi)")
    args = ap.parse_args()

    source = _frontend_source(args.frontend)
    if not source.exists():
        print(f"❌ Frontend manifest topilmadi: {source}", file=sys.stderr)
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

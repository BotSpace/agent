"""Bosqich 0 darvozasi — field-type mapping kontrakti to'liq qoplanganini tekshiradi.

Yangi manifest field type qo'shilsa (frontend yoki backend), u FIELD_TYPE_MAP'ga
ham qo'shilmaguncha bu test sinadi → kompilyator hech qachon noma'lum type'ga
duch kelmaydi.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.flow.manifest_compiler.field_type_map import (
    BACKEND_ALLOWED_FIELD_TYPES,
    FIELD_TYPE_MAP,
    FRONTEND_NODE_FIELD_TYPES,
    FieldKind,
    known_field_types,
)

_VALID_KINDS = {
    getattr(FieldKind, name) for name in dir(FieldKind) if not name.startswith("_")
}

# Sibling repolar (monorepo). CI/Docker'da bo'lmasligi mumkin → o'sha holatda skip.
_AGENT_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _AGENT_ROOT.parent
_BACKEND_MODULE_SERVICE = (
    _WORKSPACE / "backend" / "core" / "apps" / "api" / "services" / "module_service.py"
)
_FRONTEND_TYPES = (
    _WORKSPACE / "bot-constructor-panel" / "src" / "features" / "bot-constructor"
    / "manifest" / "types.ts"
)


def test_every_known_field_type_is_mapped():
    missing = sorted(known_field_types() - FIELD_TYPE_MAP.keys())
    assert not missing, f"FIELD_TYPE_MAP qoplamagan field type'lar: {missing}"


def test_backend_allowed_types_mapped():
    missing = sorted(BACKEND_ALLOWED_FIELD_TYPES - FIELD_TYPE_MAP.keys())
    assert not missing, f"Backend ALLOWED_FIELD_TYPES jadvalda yo'q: {missing}"


def test_frontend_field_types_mapped():
    missing = sorted(FRONTEND_NODE_FIELD_TYPES - FIELD_TYPE_MAP.keys())
    assert not missing, f"Frontend NodeFieldManifest type'lar jadvalda yo'q: {missing}"


def test_all_specs_have_valid_kind():
    bad = {t: s.kind for t, s in FIELD_TYPE_MAP.items() if s.kind not in _VALID_KINDS}
    assert not bad, f"Noto'g'ri FieldKind: {bad}"


def test_no_orphan_map_entries():
    """Jadvalda bo'lib, ikki manbada ham yo'q type — yo eskirgan, yo manba ro'yxati eskirgan."""
    orphans = sorted(FIELD_TYPE_MAP.keys() - known_field_types())
    assert not orphans, f"Manbasiz FIELD_TYPE_MAP yozuvlari: {orphans}"


# ── Best-effort drift detektorlari (sibling repo bo'lsa) ─────────────────────────


@pytest.mark.skipif(not _BACKEND_MODULE_SERVICE.exists(), reason="backend repo yo'q")
def test_backend_allowed_copy_matches_source():
    """BACKEND_ALLOWED_FIELD_TYPES nusxasi real backend faylga mos."""
    src = _BACKEND_MODULE_SERVICE.read_text(encoding="utf-8")
    block = re.search(r"ALLOWED_FIELD_TYPES\s*=\s*\{(.*?)\}", src, re.DOTALL)
    assert block, "backend'da ALLOWED_FIELD_TYPES topilmadi"
    real = set(re.findall(r'"([^"]+)"', block.group(1)))
    assert real == set(BACKEND_ALLOWED_FIELD_TYPES), (
        f"Backend bilan drift: faqat-backend={real - set(BACKEND_ALLOWED_FIELD_TYPES)}, "
        f"faqat-nusxa={set(BACKEND_ALLOWED_FIELD_TYPES) - real}"
    )


@pytest.mark.skipif(not _FRONTEND_TYPES.exists(), reason="frontend repo yo'q")
def test_frontend_field_types_copy_matches_source():
    """FRONTEND_NODE_FIELD_TYPES nusxasi types.ts NodeFieldManifest union'iga mos."""
    src = _FRONTEND_TYPES.read_text(encoding="utf-8")
    # `//` qatorli kommentlarni olib tashla — ular ichidagi `;` (masalan
    # "(generic; ...") union'ni regex bilan ajratishni buzadi.
    src = re.sub(r"//[^\n]*", "", src)
    block = re.search(
        r"interface NodeFieldManifest\s*\{\s*type:\s*(.*?);", src, re.DOTALL
    )
    assert block, "types.ts'da NodeFieldManifest.type union topilmadi"
    real = set(re.findall(r'"([^"]+)"', block.group(1)))
    assert real == set(FRONTEND_NODE_FIELD_TYPES), (
        f"Frontend bilan drift: faqat-frontend={real - set(FRONTEND_NODE_FIELD_TYPES)}, "
        f"faqat-nusxa={set(FRONTEND_NODE_FIELD_TYPES) - real}"
    )

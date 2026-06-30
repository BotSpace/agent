"""Bosqich 1 darvozasi — vendored manifest yuklanadi va frontend bilan sinxron."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.flow.manifest_compiler.loader import load_node_manifests, manifest_checksum

_AGENT_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _AGENT_ROOT.parent
_BACKEND_REGISTRY = (
    _WORKSPACE / "backend" / "core" / "apps" / "api" / "data" / "builtin_node_manifests.json"
)


def test_manifests_load():
    manifests = load_node_manifests()
    assert len(manifests) >= 50, f"kutilgan 50+ node, olindi {len(manifests)}"
    # Asosiy built-in node'lar mavjud.
    for node_type in ("CommandTriggerNode", "SendTextMessageNode", "CustomCodeNode",
                      "IfConditionNode", "HTTPRequestNode"):
        assert node_type in manifests, f"{node_type} manifest yo'q"


def test_manifest_entries_have_required_shape():
    for node_type, manifest in load_node_manifests().items():
        assert manifest.get("type") == node_type
        assert manifest.get("category"), f"{node_type}: category yo'q"
        # content ixtiyoriy (ba'zi node'lar maydonsiz), lekin bo'lsa ro'yxat.
        if "content" in manifest:
            assert isinstance(manifest["content"], list)


@pytest.mark.skipif(not _BACKEND_REGISTRY.exists(), reason="backend repo yo'q")
def test_vendored_matches_backend():
    """Vendored nusxa backend kanonik manbasi bilan bayt-aniqlikda mos (drift detektori)."""
    backend_sha = hashlib.sha256(_BACKEND_REGISTRY.read_bytes()).hexdigest()
    assert manifest_checksum() == backend_sha, (
        "Vendored manifest backend bilan mos emas — "
        "`python scripts/sync_manifest.py` ishga tushiring."
    )

"""Bosqich 1 darvozasi — vendored manifest yuklanadi va frontend bilan sinxron."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.flow.manifest_compiler.loader import load_node_manifests, manifest_checksum

_AGENT_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _AGENT_ROOT.parent
_FRONTEND_REGISTRY = (
    _WORKSPACE / "bot-constructor-panel" / "src" / "features" / "bot-constructor"
    / "manifest" / "runtime-node-registry.json"
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


@pytest.mark.skipif(not _FRONTEND_REGISTRY.exists(), reason="frontend repo yo'q")
def test_vendored_matches_frontend():
    """Vendored nusxa frontend manbasi bilan bayt-aniqlikda mos (drift detektori)."""
    front_sha = hashlib.sha256(_FRONTEND_REGISTRY.read_bytes()).hexdigest()
    assert manifest_checksum() == front_sha, (
        "Vendored manifest frontend bilan mos emas — "
        "`python scripts/sync_manifest.py` ishga tushiring."
    )

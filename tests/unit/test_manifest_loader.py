"""Loader darvozasi — manifest backend yagona manbadan yuklanadi (nusxa yo'q)."""

from __future__ import annotations

from agent.flow.manifest_compiler.loader import load_node_manifests


def test_manifests_load():
    manifests = load_node_manifests()
    assert len(manifests) >= 50, f"kutilgan 50+ node, olindi {len(manifests)}"
    for node_type in ("CommandTriggerNode", "SendTextMessageNode", "CustomCodeNode",
                      "IfConditionNode", "HTTPRequestNode"):
        assert node_type in manifests, f"{node_type} manifest yo'q"


def test_manifest_entries_have_required_shape():
    for node_type, manifest in load_node_manifests().items():
        assert manifest.get("type") == node_type
        assert manifest.get("category"), f"{node_type}: category yo'q"
        if "content" in manifest:
            assert isinstance(manifest["content"], list)


def test_ai_docs_present():
    """Manifest AI docs olib yuradi (LLM uchun): node aiDoc + field description."""
    manifests = load_node_manifests()
    cm = manifests["CommandTriggerNode"]
    assert cm.get("aiDoc"), "CommandTrigger aiDoc yo'q"
    cc = manifests["CustomCodeNode"]
    js = [f for f in cc.get("content", []) if f.get("key") == "jsCode"]
    assert js and js[0].get("description"), "CustomCode jsCode description yo'q"

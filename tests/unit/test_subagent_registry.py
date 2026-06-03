"""Tests for sub-agent registry — main vs domain tool boundaries."""
from agent.prompts.subagent_specs import REGISTRY, tool_allowed
from agent.tools import build_default_registry, build_main_registry


def test_four_subagents_registered():
    assert set(REGISTRY.keys()) == {
        "flow-builder",
        "crm-manager",
        "cms-manager",
        "deploy-manager",
    }


def test_tool_allowed_wildcard():
    assert tool_allowed("flow.add_command_trigger", ["flow.*"])
    assert tool_allowed("flow.connect", ["flow.*"])
    assert tool_allowed("plan.set", ["plan.*"])
    assert not tool_allowed("crm.find", ["flow.*"])
    assert tool_allowed("plan.set", ["plan.set", "plan.update"])
    assert not tool_allowed("plan.complete", ["plan.set", "plan.update"])


def test_main_registry_has_only_dispatcher_tools():
    main = build_main_registry()
    names = {t.name for t in main.all()}

    # task.run is the dispatcher's primary lever
    assert "task.run" in names
    # plan + project + search are kept for the main agent itself
    for must in ("plan.set", "project.list", "project.get", "search.semantic"):
        assert must in names, f"{must} missing from main registry"

    # Domain tools must be ABSENT from main registry — they belong to sub-agents.
    forbidden_prefixes = ("flow.", "crm.", "cms.", "broadcast.", "chat.", "env.",
                          "release.", "plans.", "subscription.", "payment.", "file.")
    for name in names:
        for pfx in forbidden_prefixes:
            assert not name.startswith(pfx), f"{name} must not be in main registry"


def test_each_subagent_tools_resolve_to_real_registry():
    """Every sub-agent's allowed_tools pattern must match at least one real tool."""
    full = build_default_registry()
    all_names = [t.name for t in full.all()]
    for spec in REGISTRY.values():
        matched = [n for n in all_names if tool_allowed(n, spec.allowed_tools)]
        assert matched, f"sub-agent {spec.name} matched zero tools"


def test_flow_builder_owns_all_flow_tools():
    full = build_default_registry()
    flow_names = [t.name for t in full.all() if t.name.startswith("flow.")]
    spec = REGISTRY["flow-builder"]
    for n in flow_names:
        assert tool_allowed(n, spec.allowed_tools), f"flow-builder missing {n}"


def test_deploy_manager_owns_release_env_billing():
    full = build_default_registry()
    spec = REGISTRY["deploy-manager"]
    for n in (t.name for t in full.all()):
        if n.startswith(("env.", "release.", "plans.", "subscription.", "payment.", "file.")):
            assert tool_allowed(n, spec.allowed_tools), f"deploy-manager missing {n}"


def test_crm_manager_owns_chat_broadcast_and_cms_reads():
    full = build_default_registry()
    spec = REGISTRY["crm-manager"]
    # chat.* and broadcast.* must all be accessible
    for n in (t.name for t in full.all()):
        if n.startswith(("chat.", "broadcast.")):
            assert tool_allowed(n, spec.allowed_tools), f"crm-manager missing {n}"
    # Explicit cms read tools must be accessible
    for n in ("cms.list_collections", "cms.get_collection", "cms.find",
              "cms.get_document", "cms.aggregate"):
        assert tool_allowed(n, spec.allowed_tools), f"crm-manager missing cms read tool {n}"
    # cms write tools must NOT be accessible
    for n in ("cms.insert", "cms.update", "cms.delete", "cms.bulk_insert",
              "cms.create_collection", "cms.delete_collection",
              "cms.add_field", "cms.update_field", "cms.remove_field"):
        assert not tool_allowed(n, spec.allowed_tools), f"crm-manager must not have write tool {n}"


def test_cms_manager_owns_cms():
    full = build_default_registry()
    spec = REGISTRY["cms-manager"]
    for n in (t.name for t in full.all()):
        if n.startswith("cms."):
            assert tool_allowed(n, spec.allowed_tools), f"cms-manager missing {n}"


def test_subagents_dont_overlap_on_domain():
    """A flow.* tool should only be in flow-builder, etc."""
    full = build_default_registry()
    all_names = [t.name for t in full.all()]
    # crm.* namespace removed — no tools registered under it any more
    namespaces = {
        "flow.": "flow-builder",
        "cms.": "cms-manager",
        "broadcast.": "crm-manager",
        "chat.": "crm-manager",
        "env.": "deploy-manager",
        "release.": "deploy-manager",
        "plans.": "deploy-manager",
        "subscription.": "deploy-manager",
        "payment.": "deploy-manager",
        "file.": "deploy-manager",
    }
    # crm-manager ATAYLAB read-only cms query tool'lariga ega (segmentlash:
    # cms.find/aggregate'dan telegram_id ro'yxati olib broadcast'ga uzatadi).
    # Yozish (insert/update/delete/schema) faqat cms-manager'da qoladi.
    ALLOWED_CROSS = {
        ("crm-manager", "cms.list_collections"),
        ("crm-manager", "cms.get_collection"),
        ("crm-manager", "cms.find"),
        ("crm-manager", "cms.get_document"),
        ("crm-manager", "cms.aggregate"),
    }
    for prefix, owner in namespaces.items():
        for name in all_names:
            if not name.startswith(prefix):
                continue
            for sa_name, spec in REGISTRY.items():
                if sa_name == owner:
                    continue
                if (sa_name, name) in ALLOWED_CROSS:
                    continue
                assert not tool_allowed(name, spec.allowed_tools), (
                    f"{name} bleeds into {sa_name}; should be only in {owner}"
                )

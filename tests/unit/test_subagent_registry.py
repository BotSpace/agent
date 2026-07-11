"""Tests for the single-analyzer sub-agent design + full-toolset main agent."""
from agent.prompts.subagent_specs import REGISTRY, tool_allowed
from agent.tools import build_default_registry, build_main_registry


def test_only_analyzer_registered():
    assert set(REGISTRY.keys()) == {"analyzer"}


def test_tool_allowed_wildcard():
    assert tool_allowed("flow.add_command_trigger", ["flow.*"])
    assert tool_allowed("flow.connect", ["flow.*"])
    assert tool_allowed("plan.set", ["plan.*"])
    assert not tool_allowed("crm.find", ["flow.*"])
    assert tool_allowed("plan.set", ["plan.set", "plan.update"])
    assert not tool_allowed("plan.complete", ["plan.set", "plan.update"])


def test_main_registry_has_full_toolset():
    # The main agent now does the work itself — it holds every domain tool
    # AND task.run (to spawn the analyzer for investigation).
    main = build_main_registry()
    names = {t.name for t in main.all()}
    assert "task.run" in names
    for must in (
        "plan.set", "project.list", "search.semantic",
        "flow.save", "flow.add_node", "cms.find", "broadcast.send",
        "release.deploy", "env.list", "credentials.list",
    ):
        assert must in names, f"{must} missing from main registry"
    # flow.clear was removed entirely.
    assert "flow.clear" not in names


def test_analyzer_has_all_tools_except_task_run():
    # "give the sub-agent all capabilities except spawning another sub-agent."
    full = build_default_registry()
    spec = REGISTRY["analyzer"]
    for t in full.all():
        if t.name == "task.run":
            assert not tool_allowed(t.name, spec.allowed_tools), "analyzer must NOT have task.run"
        else:
            assert tool_allowed(t.name, spec.allowed_tools), f"analyzer missing {t.name}"


def test_each_subagent_tools_resolve_to_real_registry():
    """Every sub-agent's allowed_tools pattern must match at least one real tool."""
    full = build_default_registry()
    all_names = [t.name for t in full.all()]
    for spec in REGISTRY.values():
        matched = [n for n in all_names if tool_allowed(n, spec.allowed_tools)]
        assert matched, f"sub-agent {spec.name} matched zero tools"

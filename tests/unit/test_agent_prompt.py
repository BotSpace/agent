from agent.graph.state import AgentState
from agent.prompts.system import build_system_prompt


def test_main_agent_prompt_is_dispatcher():
    prompt = build_system_prompt(AgentState())

    # Main agent is a dispatcher — no direct domain tool usage.
    assert "dispatcher" in prompt.lower()
    # Sub-agents named
    assert "flow-builder" in prompt
    assert "crm-manager" in prompt
    assert "cms-manager" in prompt
    assert "deploy-manager" in prompt
    # Main agent's own tools
    assert "task.run" in prompt
    assert "plan.set" in prompt
    assert "project.list" in prompt
    # Domain tools must NOT be advertised on the main agent.
    assert "flow.add_node" not in prompt
    assert "flow.upsert_node" not in prompt
    assert "flow.connect_button" not in prompt
    assert "crm.find" not in prompt
    assert "cms.create_collection" not in prompt


def test_prompt_warns_on_flow_dirty():
    state = AgentState()
    state["flow_dirty"] = True  # type: ignore[typeddict-unknown-key]
    state["baseline_flow"] = {"nodes": [{"id": "x"}], "edges": []}  # type: ignore[typeddict-unknown-key]
    prompt = build_system_prompt(state)
    assert "saqlanmagan" in prompt.lower()

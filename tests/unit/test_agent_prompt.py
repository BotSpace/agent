from agent.graph.state import AgentState
from agent.prompts.system import build_system_prompt


def test_main_agent_prompt_does_the_work_itself():
    prompt = build_system_prompt(AgentState())

    # Main agent now BUILDS/manages bots directly — it advertises its own name,
    # its full tool namespaces, and the folded bot-building guide.
    assert "Botmother AI Agent" in prompt
    assert "Bot qurish qo'llanmasi" in prompt  # detailed building guide folded in
    # Its own dispatching/planning levers plus the analysis-only sub-agent.
    assert "task.run" in prompt
    assert "analyzer" in prompt
    assert "plan.set" not in prompt  # planning alohida read-only plan mode orqali
    assert "project.list" in prompt
    # The old 4-subagent dispatcher persona is gone.
    assert "flow-builder" not in prompt
    assert "crm-manager" not in prompt
    assert "deploy-manager" not in prompt


def test_prompt_warns_on_flow_dirty():
    state = AgentState()
    state["flow_dirty"] = True  # type: ignore[typeddict-unknown-key]
    state["baseline_flow"] = {"nodes": [{"id": "x"}], "edges": []}  # type: ignore[typeddict-unknown-key]
    prompt = build_system_prompt(state)
    assert "saqlanmagan" in prompt.lower()

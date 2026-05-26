from agent.graph.state import AgentState
from agent.prompts.system import build_system_prompt


def test_system_prompt_requires_external_flow_tools():
    prompt = build_system_prompt(AgentState())

    assert "flow.ai_build" not in prompt
    assert "flow.add_command_trigger" in prompt
    assert "flow.ask_text" in prompt
    assert "flow.connect_button" in prompt
    assert "flow.connect_branch" in prompt
    assert "flow.upsert_node" in prompt
    assert "flow.connect" in prompt
    assert "Flow JSONni foydalanuvchiga matn sifatida yozma" in prompt

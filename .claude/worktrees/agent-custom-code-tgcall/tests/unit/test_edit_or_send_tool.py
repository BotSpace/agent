"""flow.add_edit_or_send typed tool mavjudligi va MCP'ga chiqishi."""

from agent.tools.flow import build_flow_tools


def test_edit_or_send_typed_tool_registered():
    tools = {t.name: t for t in build_flow_tools()}
    assert "flow.add_edit_or_send" in tools
    desc = tools["flow.add_edit_or_send"].description
    assert "PREFERRED" in desc

    send_desc = tools["flow.add_send_text_message"].description
    assert "ONLY" in send_desc  # reply keyboard cheklovi aytilgan

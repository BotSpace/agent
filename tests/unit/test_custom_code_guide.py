"""CustomCode docs_guide topic + edit rendering."""

from agent.mcp import server as mcp_server


def test_custom_code_topic_registered():
    assert "custom_code" in mcp_server._GUIDE_TOPICS
    text = mcp_server._load_guide("custom_code").lower()
    assert "edit_or_send" in text
    assert "callback_query.message.message_id" in text
    assert "public_" in text  # yangi public_ prefiks (vars) tushuntirilgan
    assert "private" in text  # maxfiy o'zgaruvchilar tushuntirilgan
    assert "state_" in text  # legacy state_ prefiks ham eslatilgan


def test_simulate_step_shows_edit_text():
    from agent.tools.flow.power import _fmt_simulate_step

    step = {
        "input": {"type": "click", "data": "page:100:2"},
        "bot_messages": [],
        "edits": [{"message_id": 7, "text": "Sahifa: 2/3"}],
        "nodes": [{"id": "show-page"}],
    }
    lines = "\n".join(_fmt_simulate_step(3, step))
    assert "edit" in lines.lower()
    assert "Sahifa: 2/3" in lines

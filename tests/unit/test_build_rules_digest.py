"""Birinchi qurish chaqirig'ida qoidalar digest'i avtomatik berilishi."""

from agent.mcp import server as mcp_server
from agent.mcp.server import _is_build_tool, _mark_rules_served


def setup_function():
    mcp_server._rules_served.clear()


def test_build_tool_detection():
    assert _is_build_tool("flow.add_edit_or_send")
    assert _is_build_tool("flow.ask_text")
    assert _is_build_tool("flow.connect_button")
    assert _is_build_tool("flow.apply_template")
    assert not _is_build_tool("flow.get")
    assert not _is_build_tool("flow.save")
    assert not _is_build_tool("cms.find")


def test_served_once_per_client():
    assert _mark_rules_served("u1", "cA") is True
    assert _mark_rules_served("u1", "cA") is False  # ikkinchi marta berilmaydi
    assert _mark_rules_served("u1", "cB") is True   # boshqa klient — yana beriladi


def test_digest_loads_and_mentions_key_rules():
    text = mcp_server._load_build_rules_digest()
    assert "edit_or_send" in text.lower() or "editorsend" in text.lower().replace("_", "")
    assert "inline" in text.lower()

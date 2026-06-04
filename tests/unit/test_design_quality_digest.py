"""Dizayn-sifat ko'rsatmasi digest va instructions'da borligi."""

from agent.mcp import server as mcp_server


def test_digest_mentions_emoji_and_html():
    text = mcp_server._load_build_rules_digest().lower()
    assert "emoji" in text
    assert "html" in text
    assert "first_name" in text


def test_instructions_mention_design_quality():
    text = mcp_server._load_instructions().lower()
    assert "design quality" in text or "emoji" in text
